from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from analyses._paths import DATA_ROOT, RUNS_ROOT, OUTPUT_ROOT, FSL_ROOT, ATLAS_ROOT, add_dataset_path_args, explicit_dataset_paths
from typing import Any, Dict, Iterable, List, Sequence

import nibabel as nib
import nibabel.processing as nib_processing
import numpy as np
import torch
from scipy import ndimage, stats

from foodpref.dataset import FMRIBetaDataset, FMRISample, discover_samples, filter_records_by_subjects
from analyses.layercam.gradcam_report import (
    CORTICAL_DIVISIONS,
    DIVISION_ORDER,
    LOW_RATING,
    HIGH_RATING,
    RUNS_ROOT,
    benjamini_hochberg,
    compute_layercam_batch,
    disable_inplace_relu,
    positive_saliency_volume,
)
from foodpref.model import FMRIConvNet
from foodpref.train import select_device
from foodpref.utils import ensure_dir, save_json


DEFAULT_RUN_DIR_NAME = "20260503_200532_fmri_rating14_mergedk06_seed42_fixed5"
DEFAULT_LOCAL_OUTPUT_CANDIDATES = (
    DATA_ROOT / "derived",
    DATA_ROOT / "derived" / "merged" / "k06_seed42",
)


@dataclass(frozen=True)
class ROI:
    name: str
    division: str
    atlas_group: str
    voxel_indices: np.ndarray
    num_voxels: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate subject-level out-of-fold Layer-CAM-style saliency maps and "
            "run one-sample TFCE-FWE inference for the fixed k=6 fMRI CNN run."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "Completed k=6 run directory containing original fixed folds and final checkpoints."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <run-dir>/gradcam_subject_tfce.",
    )
    parser.add_argument(
        "--local-output-root",
        default=None,
        help=(
            "Optional local replacement for the server output root when run_config "
            "paths are not present on this machine."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4, help="CAM generation batch size.")
    parser.add_argument("--only-fold", type=int, default=None, help="Optional fold for diagnostics.")
    parser.add_argument("--checkpoint-name", default="final.pt")
    parser.add_argument(
        "--min-correct-trials-per-subject",
        type=int,
        default=1,
        help="Minimum correct trials required for a subject to enter a class-wise group map.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=5000,
        help="Number of FSL randomise permutations for TFCE-FWE.",
    )
    parser.add_argument("--randomise-bin", default=None, help="Path to FSL randomise.")
    parser.add_argument(
        "--skip-randomise",
        action="store_true",
        help="Generate subject maps and t maps without running FSL randomise.",
    )
    parser.add_argument("--seed", type=int, default=42, help="FSL randomise seed.")
    parser.add_argument(
        "--fsl-atlas-root",
        default=None,
        help="FSL atlas root containing HarvardOxford atlases. ROI stats are skipped if unavailable.",
    )
    parser.add_argument(
        "--skip-roi",
        action="store_true",
        help="Skip Harvard-Oxford ROI subject-level summaries.",
    )
    add_dataset_path_args(parser)
    return parser


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def save_nifti(path: Path, data: np.ndarray, reference_img: nib.spatialimages.SpatialImage) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine=reference_img.affine, header=reference_img.header)
    nib.save(image, str(target))


def latest_or_default_run_dir() -> Path:
    default_path = RUNS_ROOT / DEFAULT_RUN_DIR_NAME
    if default_path.exists():
        return default_path.resolve()
    candidates = sorted(path for path in RUNS_ROOT.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {RUNS_ROOT}.")
    return candidates[-1].resolve()


def _existing_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.exists() else None


def _replace_prefix(path_text: str | None, source_root: str | None, target_root: Path | None) -> Path | None:
    if not path_text or not source_root or target_root is None:
        return None
    normalized = str(path_text).replace("\\", "/")
    source = str(source_root).replace("\\", "/").rstrip("/")
    target = str(target_root).replace("\\", "/").rstrip("/")
    if normalized == source:
        return Path(target)
    if normalized.startswith(source + "/"):
        return Path(target + normalized[len(source) :])
    return None


def _local_output_candidates(user_value: str | None) -> List[Path]:
    candidates: List[Path] = []
    if user_value:
        candidates.append(Path(user_value).expanduser())
    candidates.extend(DEFAULT_LOCAL_OUTPUT_CANDIDATES)
    unique: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_data_paths(run_config: Dict[str, Any], local_output_root: str | None) -> Dict[str, str]:
    raw_beta_index = run_config.get("beta_index_csv")
    raw_models_root = run_config.get("models_root")
    raw_brain_mask_root = run_config.get("brain_mask_root")
    server_output_root = str(run_config.get("default_output_root") or "")

    beta_index_path = _existing_path(raw_beta_index)
    models_root_path = _existing_path(raw_models_root)
    brain_mask_root_path = _existing_path(raw_brain_mask_root)
    if beta_index_path is not None and models_root_path is not None and brain_mask_root_path is not None:
        return {
            "beta_index_csv": str(beta_index_path),
            "models_root": str(models_root_path),
            "brain_mask_root": str(brain_mask_root_path),
            "path_prefix_from": "",
            "path_prefix_to": "",
            "resolution_mode": "run_config_paths_exist",
        }

    for candidate_root in _local_output_candidates(local_output_root):
        remapped_beta = _replace_prefix(str(raw_beta_index), server_output_root, candidate_root)
        remapped_models = _replace_prefix(str(raw_models_root), server_output_root, candidate_root)
        remapped_masks = _replace_prefix(str(raw_brain_mask_root), server_output_root, candidate_root)
        if (
            remapped_beta is not None
            and remapped_beta.exists()
            and remapped_models is not None
            and remapped_models.exists()
            and remapped_masks is not None
            and remapped_masks.exists()
        ):
            return {
                "beta_index_csv": str(remapped_beta.resolve()),
                "models_root": str(remapped_models.resolve()),
                "brain_mask_root": str(remapped_masks.resolve()),
                "path_prefix_from": server_output_root,
                "path_prefix_to": str(candidate_root.resolve()),
                "resolution_mode": "local_output_root_remap",
            }

    raise FileNotFoundError(
        "Could not resolve k=6 data paths. On the server, run_config paths should exist. "
        "On a local copy, pass --local-output-root pointing to the copied output root. "
        f"beta_index_csv={raw_beta_index!r}, models_root={raw_models_root!r}, "
        f"brain_mask_root={raw_brain_mask_root!r}"
    )


def dropout_from_config(run_config: Dict[str, Any]) -> float:
    for key in ("dropout", "paper_fmri_dropout"):
        value = run_config.get(key)
        if value is not None:
            return float(value)
    return 0.5


def load_common_mask(records: Sequence[FMRISample]) -> tuple[nib.spatialimages.SpatialImage, np.ndarray, Dict[str, Any]]:
    if not records:
        raise ValueError("Cannot build a common mask with no records.")
    masks: List[np.ndarray] = []
    mask_paths = sorted({str(record.mask_path) for record in records})
    reference_img = nib.load(mask_paths[0])
    reference_shape = tuple(int(value) for value in reference_img.shape)
    reference_affine = np.asarray(reference_img.affine)

    for mask_path in mask_paths:
        mask_img = nib.load(mask_path)
        if tuple(int(value) for value in mask_img.shape) != reference_shape:
            raise ValueError(f"Mask shape mismatch for {mask_path}: {mask_img.shape} vs {reference_shape}")
        if not np.allclose(np.asarray(mask_img.affine), reference_affine, atol=1e-4):
            raise ValueError(f"Mask affine mismatch for {mask_path}.")
        data = np.asarray(mask_img.dataobj)
        masks.append(np.asarray(np.isfinite(data) & (data > 0), dtype=bool))

    common_mask = np.logical_and.reduce(masks)
    if int(common_mask.sum()) <= 0:
        raise ValueError("Intersection common brain mask is empty.")
    summary = {
        "mode": "intersection_of_unique_record_masks",
        "num_unique_masks": int(len(mask_paths)),
        "reference_mask": str(mask_paths[0]),
        "reference_shape": list(reference_shape),
        "voxels_common_mask": int(common_mask.sum()),
        "voxels_first_mask": int(masks[0].sum()),
    }
    return reference_img, common_mask, summary


def target_to_rating(target: int) -> int:
    if int(target) == 0:
        return LOW_RATING
    if int(target) == 1:
        return HIGH_RATING
    raise ValueError(f"Expected target 0 or 1, got {target}.")


def load_model_for_fold(
    *,
    checkpoint_path: Path,
    run_config: Dict[str, Any],
    device: torch.device,
) -> tuple[FMRIConvNet, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_kwargs: Dict[str, Any] = {
        "input_shape": tuple(int(value) for value in checkpoint["input_shape"]),
    }
    constructor_params = inspect.signature(FMRIConvNet).parameters
    if "hidden_dim" in constructor_params:
        model_kwargs["hidden_dim"] = int(
            run_config.get("hidden_dim") or checkpoint.get("config", {}).get("hidden_dim") or 128
        )
    if "dropout" in constructor_params:
        model_kwargs["dropout"] = dropout_from_config({**checkpoint.get("config", {}), **run_config})
    model = FMRIConvNet(**model_kwargs)
    disable_inplace_relu(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def empty_subject_entry(shape: Sequence[int]) -> Dict[str, Any]:
    return {"sum": np.zeros(tuple(shape), dtype=np.float32), "count": 0}


def append_subject_cam(
    subject_maps: Dict[int, Dict[str, Dict[str, Any]]],
    *,
    target: int,
    subject_id: str,
    cam: np.ndarray,
) -> None:
    target_maps = subject_maps.setdefault(int(target), {})
    entry = target_maps.setdefault(subject_id, empty_subject_entry(cam.shape))
    entry["sum"] += np.asarray(cam, dtype=np.float32)
    entry["count"] = int(entry["count"]) + 1


def compute_subject_mean_maps(
    subject_maps: Dict[int, Dict[str, Dict[str, Any]]],
    *,
    min_correct_trials: int,
    common_mask: np.ndarray,
) -> Dict[int, Dict[str, np.ndarray]]:
    output: Dict[int, Dict[str, np.ndarray]] = {}
    for target, by_subject in subject_maps.items():
        output[target] = {}
        for subject_id, entry in sorted(by_subject.items()):
            count = int(entry["count"])
            if count < int(min_correct_trials):
                continue
            mean_map = np.asarray(entry["sum"] / float(count), dtype=np.float32)
            mean_map = np.where(common_mask, mean_map, 0.0).astype(np.float32, copy=False)
            output[target][subject_id] = mean_map
    return output


def compute_t_map(subject_stack: np.ndarray) -> np.ndarray:
    stack = np.asarray(subject_stack, dtype=np.float32)
    n_subjects = int(stack.shape[0])
    if n_subjects < 2:
        return np.zeros(stack.shape[1:], dtype=np.float32)
    mean = np.mean(stack, axis=0, dtype=np.float64)
    std = np.std(stack, axis=0, ddof=1, dtype=np.float64)
    stderr = std / math.sqrt(float(n_subjects))
    t_values = np.divide(mean, stderr, out=np.zeros_like(mean), where=stderr > 1e-12)
    t_values[~np.isfinite(t_values)] = 0.0
    return t_values.astype(np.float32)


def save_subject_4d(path: Path, maps_by_subject: Dict[str, np.ndarray], reference_img: nib.spatialimages.SpatialImage) -> List[str]:
    subjects = sorted(maps_by_subject)
    if not subjects:
        raise ValueError(f"No subject maps available for {path}.")
    stack = np.stack([np.asarray(maps_by_subject[subject], dtype=np.float32) for subject in subjects], axis=-1)
    save_nifti(path, stack, reference_img)
    return subjects


def save_subject_maps(
    output_dir: Path,
    *,
    target: int,
    maps_by_subject: Dict[str, np.ndarray],
    reference_img: nib.spatialimages.SpatialImage,
) -> None:
    rating = target_to_rating(target)
    class_dir = ensure_dir(output_dir / "subject_maps" / f"rating_{rating}")
    for subject_id, subject_map in sorted(maps_by_subject.items()):
        save_nifti(class_dir / f"{subject_id}_rating_{rating}_mean_layercam.nii.gz", subject_map, reference_img)


def find_randomise_binary(user_value: str | None) -> str | None:
    if user_value:
        candidate = Path(user_value).expanduser()
        return str(candidate) if candidate.exists() else user_value
    path_hit = shutil.which("randomise")
    if path_hit:
        return path_hit
    candidates: List[Path] = []
    fsldir = os.environ.get("FSLDIR")
    if fsldir:
        candidates.extend(
            [
                Path(fsldir) / "bin" / "randomise",
                Path(fsldir) / "share" / "fsl" / "bin" / "randomise",
            ]
        )
    home = Path.home()
    candidates.extend(
        [
            home / "fsl" / "bin" / "randomise",
            home / "fsl" / "share" / "fsl" / "bin" / "randomise",
            Path("/usr/local/fsl/bin/randomise"),
            Path("/usr/local/fsl/share/fsl/bin/randomise"),
            Path("/opt/fsl/bin/randomise"),
            Path("/opt/fsl/share/fsl/bin/randomise"),
            Path("/usr/lib/fsl/6.0/randomise"),
            Path("/usr/lib/fsl/6.0/bin/randomise"),
            Path("/usr/share/fsl/6.0/bin/randomise"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def run_randomise_tfce(
    *,
    input_4d: Path,
    output_root: Path,
    mask_path: Path,
    permutations: int,
    seed: int,
    randomise_bin: str,
) -> Dict[str, str]:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    command = [
        randomise_bin,
        "-i",
        str(input_4d),
        "-o",
        str(output_root),
        "-1",
        "-m",
        str(mask_path),
        "-n",
        str(int(permutations)),
        "-T",
        "-R",
        f"--seed={int(seed)}",
    ]
    subprocess.run(command, check=True)
    return {
        "command": " ".join(command),
        "tfce_corrp": str(output_root.with_name(output_root.name + "_tfce_corrp_tstat1.nii.gz")),
        "tfce_p": str(output_root.with_name(output_root.name + "_tfce_p_tstat1.nii.gz")),
        "raw_tstat": str(output_root.with_name(output_root.name + "_tstat1.nii.gz")),
    }


def connected_component_rows(
    *,
    significance_mask: np.ndarray,
    t_map: np.ndarray,
    corrp_map: np.ndarray,
    reference_img: nib.spatialimages.SpatialImage,
) -> List[Dict[str, Any]]:
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, num_labels = ndimage.label(np.asarray(significance_mask, dtype=bool), structure=structure)
    rows: List[Dict[str, Any]] = []
    affine = np.asarray(reference_img.affine)
    voxel_volume = abs(float(np.linalg.det(affine[:3, :3])))
    for cluster_id in range(1, int(num_labels) + 1):
        cluster_mask = labels == cluster_id
        voxel_count = int(cluster_mask.sum())
        if voxel_count <= 0:
            continue
        cluster_t = np.where(cluster_mask, t_map, -np.inf)
        peak_flat = int(np.argmax(cluster_t))
        peak_index = np.unravel_index(peak_flat, t_map.shape)
        peak_xyz = nib.affines.apply_affine(affine, np.asarray(peak_index, dtype=np.float64))
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "num_voxels": voxel_count,
                "volume_mm3": float(voxel_count * voxel_volume),
                "peak_t": float(t_map[peak_index]),
                "peak_tfce_corrp": float(corrp_map[peak_index]),
                "peak_i": int(peak_index[0]),
                "peak_j": int(peak_index[1]),
                "peak_k": int(peak_index[2]),
                "peak_x_mm": float(peak_xyz[0]),
                "peak_y_mm": float(peak_xyz[1]),
                "peak_z_mm": float(peak_xyz[2]),
            }
        )
    rows.sort(key=lambda row: (float(row["peak_t"]), int(row["num_voxels"])), reverse=True)
    for new_id, row in enumerate(rows, start=1):
        row["cluster_id"] = int(new_id)
    return rows


def threshold_tfce_outputs(
    *,
    output_dir: Path,
    target: int,
    reference_img: nib.spatialimages.SpatialImage,
    fallback_t_map: np.ndarray,
    randomise_outputs: Dict[str, str],
) -> Dict[str, Any]:
    rating = target_to_rating(target)
    corrp_path = Path(randomise_outputs["tfce_corrp"])
    tstat_path = Path(randomise_outputs["raw_tstat"])
    if not corrp_path.exists():
        raise FileNotFoundError(f"FSL randomise did not create expected TFCE corrp image: {corrp_path}")
    corrp_img = nib.load(str(corrp_path))
    corrp_map = np.asarray(corrp_img.dataobj, dtype=np.float32)
    if tstat_path.exists():
        t_map = np.asarray(nib.load(str(tstat_path)).dataobj, dtype=np.float32)
    else:
        t_map = np.asarray(fallback_t_map, dtype=np.float32)

    significance_mask = np.asarray(corrp_map >= 0.95, dtype=bool)
    masked_t = np.where(significance_mask, t_map, 0.0).astype(np.float32)
    mask_out = output_dir / f"rating_{rating}_tfce_fwe05_mask.nii.gz"
    masked_t_out = output_dir / f"rating_{rating}_tfce_fwe05_masked_t.nii.gz"
    save_nifti(mask_out, significance_mask.astype(np.float32), reference_img)
    save_nifti(masked_t_out, masked_t, reference_img)

    cluster_rows = connected_component_rows(
        significance_mask=significance_mask,
        t_map=t_map,
        corrp_map=corrp_map,
        reference_img=reference_img,
    )
    write_csv(
        output_dir / f"rating_{rating}_tfce_fwe05_clusters.csv",
        cluster_rows,
        [
            "cluster_id",
            "num_voxels",
            "volume_mm3",
            "peak_t",
            "peak_tfce_corrp",
            "peak_i",
            "peak_j",
            "peak_k",
            "peak_x_mm",
            "peak_y_mm",
            "peak_z_mm",
        ],
    )
    return {
        "tfce_corrp_path": str(corrp_path),
        "raw_tstat_path": str(tstat_path),
        "fwe05_mask_path": str(mask_out),
        "fwe05_masked_t_path": str(masked_t_out),
        "num_fwe05_voxels": int(significance_mask.sum()),
        "num_fwe05_clusters": int(len(cluster_rows)),
        "clusters_csv": str((output_dir / f"rating_{rating}_tfce_fwe05_clusters.csv").resolve()),
    }


def parse_label_xml(xml_path: Path) -> List[tuple[int, str]]:
    root = ET.parse(xml_path).getroot()
    labels: List[tuple[int, str]] = []
    for label in root.findall(".//label"):
        value = int(label.attrib["index"]) + 1
        text = str(label.text or "").strip()
        if text:
            labels.append((value, text))
    return labels


def build_world_x_grid(shape: Sequence[int], affine: np.ndarray) -> np.ndarray:
    x_idx = np.arange(int(shape[0]), dtype=np.float32)[:, None, None]
    y_idx = np.arange(int(shape[1]), dtype=np.float32)[None, :, None]
    z_idx = np.arange(int(shape[2]), dtype=np.float32)[None, None, :]
    return affine[0, 0] * x_idx + affine[0, 1] * y_idx + affine[0, 2] * z_idx + affine[0, 3]


def should_skip_subcortical_roi(name: str) -> bool:
    lowered = name.lower()
    return "white matter" in lowered or "ventricle" in lowered or "cerebral cortex" in lowered


def default_atlas_root(user_value: str | None) -> Path | None:
    candidates: List[Path] = []
    if user_value:
        candidates.append(Path(user_value).expanduser())
    fsldir = os.environ.get("FSLDIR")
    if fsldir:
        candidates.append(Path(fsldir) / "data" / "atlases")
    candidates.extend(
        [
            ATLAS_ROOT,
            Path("/usr/local/fsl/data/atlases"),
            Path("/opt/fsl/data/atlases"),
        ]
    )
    for candidate in candidates:
        if (candidate / "HarvardOxford-Cortical.xml").exists():
            return candidate.resolve()
    return None


def build_rois(reference_img: nib.spatialimages.SpatialImage, atlas_root: Path, common_mask: np.ndarray) -> tuple[List[ROI], Dict[str, Any]]:
    cortical_img = nib.load(str(atlas_root / "HarvardOxford" / "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"))
    subcortical_img = nib.load(str(atlas_root / "HarvardOxford" / "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz"))
    cortical_resampled = nib_processing.resample_from_to(cortical_img, reference_img, order=0)
    subcortical_resampled = nib_processing.resample_from_to(subcortical_img, reference_img, order=0)
    cortical_data = np.asarray(cortical_resampled.dataobj, dtype=np.int16)
    subcortical_data = np.asarray(subcortical_resampled.dataobj, dtype=np.int16)
    world_x = build_world_x_grid(reference_img.shape, reference_img.affine)

    rois: List[ROI] = []
    for value, base_name in parse_label_xml(atlas_root / "HarvardOxford-Cortical.xml"):
        division = CORTICAL_DIVISIONS.get(base_name)
        if division is None:
            raise KeyError(f"Missing cortical division mapping for ROI {base_name!r}.")
        base_mask = np.asarray((cortical_data == int(value)) & common_mask, dtype=bool)
        if not bool(base_mask.any()):
            continue
        for side_name, side_mask in (
            ("Left", np.asarray(base_mask & (world_x < 0.0), dtype=bool)),
            ("Right", np.asarray(base_mask & (world_x >= 0.0), dtype=bool)),
        ):
            if not bool(side_mask.any()):
                continue
            rois.append(
                ROI(
                    name=f"{side_name} {base_name}",
                    division=division,
                    atlas_group="cortical",
                    voxel_indices=np.flatnonzero(side_mask.reshape(-1)),
                    num_voxels=int(side_mask.sum()),
                )
            )

    for value, base_name in parse_label_xml(atlas_root / "HarvardOxford-Subcortical.xml"):
        if should_skip_subcortical_roi(base_name):
            continue
        base_mask = np.asarray((subcortical_data == int(value)) & common_mask, dtype=bool)
        if not bool(base_mask.any()):
            continue
        for side_name, side_mask in (
            ("Left", np.asarray(base_mask & (world_x < 0.0), dtype=bool)),
            ("Right", np.asarray(base_mask & (world_x >= 0.0), dtype=bool)),
        ):
            if not bool(side_mask.any()):
                continue
            rois.append(
                ROI(
                    name=f"{side_name} {base_name}",
                    division="Subcortical",
                    atlas_group="subcortical",
                    voxel_indices=np.flatnonzero(side_mask.reshape(-1)),
                    num_voxels=int(side_mask.sum()),
                )
            )

    summary = {
        "atlas_root": str(atlas_root),
        "num_rois": int(len(rois)),
        "num_cortical_rois": int(sum(roi.atlas_group == "cortical" for roi in rois)),
        "num_subcortical_rois": int(sum(roi.atlas_group == "subcortical" for roi in rois)),
        "division_counts": {division: int(sum(roi.division == division for roi in rois)) for division in DIVISION_ORDER},
    }
    return rois, summary


def compute_roi_subject_statistics(
    *,
    maps_by_subject: Dict[str, np.ndarray],
    rois: Sequence[ROI],
) -> List[Dict[str, Any]]:
    subjects = sorted(maps_by_subject)
    if not subjects:
        return []
    flat_maps = {subject: np.asarray(maps_by_subject[subject], dtype=np.float32).reshape(-1) for subject in subjects}
    p_values: List[float] = []
    rows: List[Dict[str, Any]] = []
    for roi in rois:
        values = np.asarray([float(flat_maps[subject][roi.voxel_indices].mean()) for subject in subjects], dtype=np.float64)
        mean_value = float(np.mean(values)) if values.size else 0.0
        median_value = float(np.median(values)) if values.size else 0.0
        std_value = float(np.std(values, ddof=0)) if values.size else 0.0
        if values.size >= 2 and not np.allclose(values, 0.0):
            try:
                statistic, p_value = stats.wilcoxon(values, alternative="greater", zero_method="wilcox")
                statistic = float(statistic)
                p_value = float(p_value)
            except ValueError:
                statistic = 0.0
                p_value = 1.0
        else:
            statistic = 0.0
            p_value = 1.0
        p_values.append(p_value)
        rows.append(
            {
                "roi_name": roi.name,
                "division": roi.division,
                "atlas_group": roi.atlas_group,
                "num_voxels": int(roi.num_voxels),
                "num_subjects": int(values.size),
                "mean_value": mean_value,
                "median_value": median_value,
                "std_value": std_value,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    q_values = benjamini_hochberg(p_values)
    for row, q_value in zip(rows, q_values.tolist()):
        row["q_value"] = float(q_value)
        row["significant_fdr_0_05"] = bool(q_value < 0.05)
    rows.sort(key=lambda row: float(row["mean_value"]), reverse=True)
    return rows


def generate_subject_maps(
    *,
    run_dir: Path,
    output_dir: Path,
    run_config: Dict[str, Any],
    fold_assignments: Dict[str, Any],
    records: Sequence[FMRISample],
    reference_img: nib.spatialimages.SpatialImage,
    common_mask: np.ndarray,
    device: torch.device,
    batch_size: int,
    only_fold: int | None,
    checkpoint_name: str,
) -> tuple[Dict[int, Dict[str, Dict[str, Any]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    subject_maps: Dict[int, Dict[str, Dict[str, Any]]] = {0: {}, 1: {}}
    sample_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []

    for fold_payload in fold_assignments["folds"]:
        fold_index = int(fold_payload["fold_index"])
        if only_fold is not None and fold_index != int(only_fold):
            continue
        val_records = filter_records_by_subjects(records, fold_payload["val_subjects"])
        val_dataset = FMRIBetaDataset(val_records)
        checkpoint_path = run_dir / f"fold_{fold_index:02d}" / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Could not find checkpoint for fold {fold_index:02d}: {checkpoint_path}")
        print(f"[subject-tfce] fold {fold_index:02d}: loading {checkpoint_path}", flush=True)
        model, checkpoint = load_model_for_fold(checkpoint_path=checkpoint_path, run_config=run_config, device=device)

        fold_total = 0
        fold_correct = 0
        fold_correct_by_target = {0: 0, 1: 0}
        for start in range(0, len(val_dataset), int(batch_size)):
            indices = range(start, min(start + int(batch_size), len(val_dataset)))
            items = [val_dataset[index] for index in indices]
            x = torch.stack([item["x"] for item in items], dim=0).to(device)
            y = torch.stack([item["y"] for item in items], dim=0).to(device)
            with torch.no_grad():
                logits = model.forward_raw(x)
                probs_high_tensor = torch.sigmoid(logits)
                preds_tensor = (probs_high_tensor >= 0.5).long()
                correct_mask_tensor = preds_tensor.eq(y.long())

            probs_high = probs_high_tensor.detach().cpu().numpy().astype(np.float32)
            preds = preds_tensor.detach().cpu().numpy().astype(np.int64)
            correct_mask = correct_mask_tensor.detach().cpu().numpy().astype(bool)
            cams = np.zeros((len(items), *reference_img.shape), dtype=np.float32)
            if bool(correct_mask.any()):
                x_correct = x[correct_mask_tensor].detach().clone().requires_grad_(True)
                y_correct = y[correct_mask_tensor].detach().clone()
                with torch.enable_grad():
                    cam_output = compute_layercam_batch(model, x_correct, y_correct)
                cams[np.flatnonzero(correct_mask)] = cam_output["cams"].detach().cpu().numpy().astype(np.float32)

            fold_total += int(len(items))
            fold_correct += int(correct_mask.sum())
            for local_index, dataset_index in enumerate(indices):
                item = items[local_index]
                record = val_records[int(dataset_index)]
                target = int(item["y"].item())
                pred = int(preds[local_index])
                is_correct = bool(correct_mask[local_index])
                sample_rows.append(
                    {
                        "fold_index": fold_index,
                        "subject_id": record.subject_id,
                        "session_id": record.session_id,
                        "run_id": record.run_id,
                        "trial_id": record.trial_id,
                        "sample_path": record.sample_path,
                        "mask_path": record.mask_path,
                        "true_rating": int(record.rating),
                        "true_target": target,
                        "pred_target": pred,
                        "prob_high": float(probs_high[local_index]),
                        "correct": is_correct,
                    }
                )
                if not is_correct:
                    continue
                fold_correct_by_target[target] += 1
                cam = positive_saliency_volume(cams[local_index], brain_mask=common_mask)
                append_subject_cam(subject_maps, target=target, subject_id=record.subject_id, cam=cam)

        fold_rows.append(
            {
                "fold_index": fold_index,
                "num_val_trials": int(fold_total),
                "num_correct_trials": int(fold_correct),
                "correct_fraction": float(fold_correct / fold_total) if fold_total > 0 else 0.0,
                "num_correct_rating_1": int(fold_correct_by_target[0]),
                "num_correct_rating_4": int(fold_correct_by_target[1]),
                "checkpoint_name": checkpoint_name,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "reported_epoch": int(checkpoint.get("reported_epoch", checkpoint.get("epoch", 0))),
                "selection_policy": str(checkpoint.get("selection_policy", "unknown")),
            }
        )
        print(
            f"[subject-tfce] fold {fold_index:02d}: correct {fold_correct}/{fold_total} "
            f"(rating1={fold_correct_by_target[0]}, rating4={fold_correct_by_target[1]})",
            flush=True,
        )

    write_csv(
        output_dir / "heldout_trial_predictions.csv",
        sample_rows,
        [
            "fold_index",
            "subject_id",
            "session_id",
            "run_id",
            "trial_id",
            "sample_path",
            "mask_path",
            "true_rating",
            "true_target",
            "pred_target",
            "prob_high",
            "correct",
        ],
    )
    return subject_maps, sample_rows, fold_rows


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else latest_or_default_run_dir()
    output_dir = ensure_dir(args.output_dir or (run_dir / "gradcam_subject_tfce"))
    run_config = read_json(run_dir / "run_config.json")
    fold_assignments = read_json(run_dir / "fold_assignments.json")
    resolved_paths = explicit_dataset_paths(args) or resolve_data_paths(run_config, args.local_output_root)
    device = select_device(args.device)
    randomise_bin = find_randomise_binary(args.randomise_bin)
    if not args.skip_randomise and randomise_bin is None:
        raise FileNotFoundError(
            "FSL randomise was not found before CAM generation. Either load FSL so "
            "`randomise` is in PATH, pass --randomise-bin /path/to/randomise, or use "
            "--skip-randomise to generate subject maps without TFCE-FWE."
        )

    records, dataset_summary = discover_samples(
        beta_index_csv=resolved_paths["beta_index_csv"],
        models_root=resolved_paths["models_root"],
        brain_mask_root=resolved_paths["brain_mask_root"],
        path_prefix_from=resolved_paths["path_prefix_from"],
        path_prefix_to=resolved_paths["path_prefix_to"],
    )
    reference_img, common_mask, common_mask_summary = load_common_mask(records)
    common_mask_path = output_dir / "common_brain_mask.nii.gz"
    save_nifti(common_mask_path, common_mask.astype(np.float32), reference_img)

    subject_accumulators, sample_rows, fold_rows = generate_subject_maps(
        run_dir=run_dir,
        output_dir=output_dir,
        run_config=run_config,
        fold_assignments=fold_assignments,
        records=records,
        reference_img=reference_img,
        common_mask=common_mask,
        device=device,
        batch_size=int(args.batch_size),
        only_fold=None if args.only_fold is None else int(args.only_fold),
        checkpoint_name=str(args.checkpoint_name),
    )

    subject_mean_maps = compute_subject_mean_maps(
        subject_accumulators,
        min_correct_trials=int(args.min_correct_trials_per_subject),
        common_mask=common_mask,
    )

    atlas_root = None if args.skip_roi else default_atlas_root(args.fsl_atlas_root)
    rois: List[ROI] = []
    roi_inventory: Dict[str, Any] | None = None
    if atlas_root is not None:
        rois, roi_inventory = build_rois(reference_img, atlas_root, common_mask)
    elif not args.skip_roi:
        print("[subject-tfce] Harvard-Oxford atlas not found; ROI summaries will be skipped.", flush=True)

    classes_summary: Dict[str, Any] = {}
    for target in (0, 1):
        rating = target_to_rating(target)
        maps_by_subject = subject_mean_maps.get(target, {})
        save_subject_maps(output_dir, target=target, maps_by_subject=maps_by_subject, reference_img=reference_img)

        if not maps_by_subject:
            classes_summary[str(target)] = {
                "rating": int(rating),
                "num_subjects": 0,
                "num_correct_trials": int(sum(int(entry["count"]) for entry in subject_accumulators.get(target, {}).values())),
            }
            continue

        subjects = sorted(maps_by_subject)
        subject_counts = {
            subject: int(subject_accumulators[target][subject]["count"])
            for subject in subjects
        }
        subject_4d_path = output_dir / f"rating_{rating}_subject_4d.nii.gz"
        save_subject_4d(subject_4d_path, maps_by_subject, reference_img)
        stack = np.stack([maps_by_subject[subject] for subject in subjects], axis=0)
        group_mean = np.asarray(np.mean(stack, axis=0), dtype=np.float32)
        t_map = compute_t_map(stack)
        group_mean_path = output_dir / f"rating_{rating}_group_mean_layercam.nii.gz"
        t_map_path = output_dir / f"rating_{rating}_subject_t_map.nii.gz"
        save_nifti(group_mean_path, group_mean, reference_img)
        save_nifti(t_map_path, t_map, reference_img)

        roi_csv_path = None
        top_rois: List[Dict[str, Any]] = []
        if rois:
            roi_rows = compute_roi_subject_statistics(maps_by_subject=maps_by_subject, rois=rois)
            roi_csv_path = output_dir / f"rating_{rating}_roi_subject_stats.csv"
            write_csv(
                roi_csv_path,
                roi_rows,
                [
                    "roi_name",
                    "division",
                    "atlas_group",
                    "num_voxels",
                    "num_subjects",
                    "mean_value",
                    "median_value",
                    "std_value",
                    "wilcoxon_statistic",
                    "p_value",
                    "q_value",
                    "significant_fdr_0_05",
                ],
            )
            top_rois = [
                {
                    "roi_name": row["roi_name"],
                    "division": row["division"],
                    "mean_value": float(row["mean_value"]),
                    "q_value": float(row["q_value"]),
                    "significant_fdr_0_05": bool(row["significant_fdr_0_05"]),
                }
                for row in roi_rows[:15]
            ]

        randomise_summary: Dict[str, Any] | None = None
        if not args.skip_randomise:
            if randomise_bin is None:
                raise FileNotFoundError("FSL randomise was not found. Use --randomise-bin or --skip-randomise.")
            if len(subjects) < 2:
                print(f"[subject-tfce] rating {rating}: fewer than 2 subjects; skipping randomise.", flush=True)
            else:
                print(f"[subject-tfce] rating {rating}: running randomise TFCE ({args.permutations} permutations)", flush=True)
                randomise_outputs = run_randomise_tfce(
                    input_4d=subject_4d_path,
                    output_root=output_dir / f"rating_{rating}_randomise",
                    mask_path=common_mask_path,
                    permutations=int(args.permutations),
                    seed=int(args.seed),
                    randomise_bin=randomise_bin,
                )
                randomise_summary = {
                    **randomise_outputs,
                    **threshold_tfce_outputs(
                        output_dir=output_dir,
                        target=target,
                        reference_img=reference_img,
                        fallback_t_map=t_map,
                        randomise_outputs=randomise_outputs,
                    ),
                }

        classes_summary[str(target)] = {
            "rating": int(rating),
            "subject_ids": subjects,
            "num_subjects": int(len(subjects)),
            "num_correct_trials": int(sum(subject_counts.values())),
            "min_correct_trials_per_subject": int(min(subject_counts.values())) if subject_counts else 0,
            "max_correct_trials_per_subject": int(max(subject_counts.values())) if subject_counts else 0,
            "correct_trials_by_subject": subject_counts,
            "subject_4d_path": str(subject_4d_path.resolve()),
            "group_mean_map": str(group_mean_path.resolve()),
            "subject_t_map": str(t_map_path.resolve()),
            "roi_subject_stats_csv": None if roi_csv_path is None else str(roi_csv_path.resolve()),
            "top_rois_by_subject_mean": top_rois,
            "randomise": randomise_summary,
        }

    analysis_summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "checkpoint_name": str(args.checkpoint_name),
        "only_fold": None if args.only_fold is None else int(args.only_fold),
        "method": {
            "saliency": (
                "Layer-CAM-style positive class-discriminative CAM from the final convolutional "
                "feature block; no global average pooling; trilinear upsampling to input fMRI space."
            ),
            "cam_sign": "non-negative positive attribution maps after ReLU; not signed evidence.",
            "model_selection": "per-fold fixed final.pt checkpoints; no best-fold selection.",
            "evaluation_split": "correctly classified held-out trials generated out-of-fold.",
            "group_unit": "subject-level mean CAM per rating condition.",
            "inference": (
                "one-sample subject-level test against zero with FSL randomise TFCE-FWE correction "
                "when --skip-randomise is not used."
            ),
            "roi_unit": "subject-level ROI mean CAM; Wilcoxon signed-rank against zero with BH-FDR.",
        },
        "resolved_paths": resolved_paths,
        "dataset_summary": dataset_summary,
        "common_mask": common_mask_summary,
        "roi_inventory": roi_inventory,
        "fold_summary": fold_rows,
        "num_prediction_rows": int(len(sample_rows)),
        "classes": classes_summary,
    }
    save_json(output_dir / "analysis_summary.json", analysis_summary)
    print(f"[subject-tfce] analysis complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
