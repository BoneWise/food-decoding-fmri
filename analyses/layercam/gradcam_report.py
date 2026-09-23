from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from analyses._paths import DATA_ROOT, RUNS_ROOT, OUTPUT_ROOT, FSL_ROOT, ATLAS_ROOT, add_dataset_path_args, explicit_dataset_paths
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import nibabel as nib
import nibabel.processing as nib_processing
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from foodpref.dataset import FMRIBetaDataset, discover_samples, filter_records_by_subjects
from foodpref.model import FMRIConvNet
from foodpref.train import select_device
from foodpref.utils import HIGH_RATING, LOW_RATING, ensure_dir, save_json


LOCAL_OUTPUT_ROOT = DATA_ROOT / "derived"
FSL_ATLAS_ROOT = ATLAS_ROOT
DIVISION_ORDER = ["Occipital", "Parietal", "Temporal", "Frontal", "Subcortical"]


CORTICAL_DIVISIONS = {
    "Frontal Pole": "Frontal",
    "Insular Cortex": "Subcortical",
    "Superior Frontal Gyrus": "Frontal",
    "Middle Frontal Gyrus": "Frontal",
    "Inferior Frontal Gyrus, pars triangularis": "Frontal",
    "Inferior Frontal Gyrus, pars opercularis": "Frontal",
    "Precentral Gyrus": "Frontal",
    "Temporal Pole": "Temporal",
    "Superior Temporal Gyrus, anterior division": "Temporal",
    "Superior Temporal Gyrus, posterior division": "Temporal",
    "Middle Temporal Gyrus, anterior division": "Temporal",
    "Middle Temporal Gyrus, posterior division": "Temporal",
    "Middle Temporal Gyrus, temporooccipital part": "Temporal",
    "Inferior Temporal Gyrus, anterior division": "Temporal",
    "Inferior Temporal Gyrus, posterior division": "Temporal",
    "Inferior Temporal Gyrus, temporooccipital part": "Temporal",
    "Postcentral Gyrus": "Parietal",
    "Superior Parietal Lobule": "Parietal",
    "Supramarginal Gyrus, anterior division": "Parietal",
    "Supramarginal Gyrus, posterior division": "Parietal",
    "Angular Gyrus": "Parietal",
    "Lateral Occipital Cortex, superior division": "Occipital",
    "Lateral Occipital Cortex, inferior division": "Occipital",
    "Intracalcarine Cortex": "Occipital",
    "Frontal Medial Cortex": "Frontal",
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": "Frontal",
    "Subcallosal Cortex": "Subcortical",
    "Paracingulate Gyrus": "Frontal",
    "Cingulate Gyrus, anterior division": "Frontal",
    "Cingulate Gyrus, posterior division": "Frontal",
    "Precuneous Cortex": "Parietal",
    "Cuneal Cortex": "Occipital",
    "Frontal Orbital Cortex": "Frontal",
    "Parahippocampal Gyrus, anterior division": "Temporal",
    "Parahippocampal Gyrus, posterior division": "Temporal",
    "Lingual Gyrus": "Occipital",
    "Temporal Fusiform Cortex, anterior division": "Temporal",
    "Temporal Fusiform Cortex, posterior division": "Temporal",
    "Temporal Occipital Fusiform Cortex": "Occipital",
    "Occipital Fusiform Gyrus": "Occipital",
    "Frontal Operculum Cortex": "Frontal",
    "Central Opercular Cortex": "Parietal",
    "Parietal Operculum Cortex": "Parietal",
    "Planum Polare": "Temporal",
    "Heschl's Gyrus (includes H1 and H2)": "Temporal",
    "Planum Temporale": "Temporal",
    "Supracalcarine Cortex": "Occipital",
    "Occipital Pole": "Occipital",
}


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
            "Generate paper-style Grad-CAM / Layer-CAM brain-region visualizations "
            "from a completed 10-fold fMRI run."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Completed run directory containing the original fixed folds and final checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <run-dir>/paper_gradcam.",
    )
    parser.add_argument(
        "--local-output-root",
        default=str(LOCAL_OUTPUT_ROOT),
        help="Local dataset root used to override server paths when needed.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device used for saliency generation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size used during out-of-fold saliency generation.",
    )
    parser.add_argument(
        "--only-fold",
        type=int,
        default=None,
        help="Optional single fold index to analyze for diagnostics.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="final.pt",
        help=(
            "Checkpoint filename to load from each fold directory. Use final.pt for the "
            "fixed-epoch paper run; pass best.pt only for legacy early-stopping runs."
        ),
    )
    add_dataset_path_args(parser)
    return parser


def latest_run_dir() -> Path:
    candidates = sorted(path for path in RUNS_ROOT.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {RUNS_ROOT}.")
    return candidates[-1]


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


def resolve_existing_path(candidate: str | Path | None) -> Path | None:
    if candidate is None:
        return None
    path = Path(candidate).expanduser()
    return path.resolve() if path.exists() else None


def resolve_data_paths(run_config: Dict[str, Any], local_output_root: Path) -> Dict[str, str]:
    local_output_root = local_output_root.expanduser().resolve()
    local_beta_index_csv = local_output_root / "beta_index.csv"
    local_models_root = local_output_root / "models"
    local_brain_mask_root = local_output_root / "cache" / "masks"

    server_output_root = str(run_config.get("default_output_root") or run_config.get("beta_index_csv", ""))
    server_output_root = server_output_root.replace("\\", "/")
    if server_output_root.endswith("/beta_index.csv"):
        server_output_root = str(Path(server_output_root).parent).replace("\\", "/")

    beta_index_csv = resolve_existing_path(run_config.get("beta_index_csv")) or local_beta_index_csv.resolve()
    models_root = resolve_existing_path(run_config.get("models_root")) or local_models_root.resolve()
    brain_mask_root = resolve_existing_path(run_config.get("brain_mask_root")) or local_brain_mask_root.resolve()

    if not Path(beta_index_csv).exists():
        raise FileNotFoundError(f"Could not resolve beta_index.csv. Tried {run_config.get('beta_index_csv')} and {local_beta_index_csv}.")
    if not Path(models_root).exists():
        raise FileNotFoundError(f"Could not resolve models_root. Tried {run_config.get('models_root')} and {local_models_root}.")
    if not Path(brain_mask_root).exists():
        raise FileNotFoundError(
            f"Could not resolve brain_mask_root. Tried {run_config.get('brain_mask_root')} and {local_brain_mask_root}."
        )

    return {
        "beta_index_csv": str(Path(beta_index_csv).resolve()),
        "models_root": str(Path(models_root).resolve()),
        "brain_mask_root": str(Path(brain_mask_root).resolve()),
        "path_prefix_from": server_output_root,
        "path_prefix_to": str(local_output_root),
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
    return (
        affine[0, 0] * x_idx
        + affine[0, 1] * y_idx
        + affine[0, 2] * z_idx
        + affine[0, 3]
    )


def should_skip_subcortical_roi(name: str) -> bool:
    lowered = name.lower()
    return (
        "white matter" in lowered
        or "ventricle" in lowered
        or "cerebral cortex" in lowered
    )


def build_rois(reference_img: nib.spatialimages.SpatialImage) -> tuple[List[ROI], Dict[str, Any]]:
    cortical_img = nib.load(str(FSL_ATLAS_ROOT / "HarvardOxford" / "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"))
    subcortical_img = nib.load(str(FSL_ATLAS_ROOT / "HarvardOxford" / "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz"))

    cortical_resampled = nib_processing.resample_from_to(cortical_img, reference_img, order=0)
    subcortical_resampled = nib_processing.resample_from_to(subcortical_img, reference_img, order=0)
    cortical_data = np.asarray(cortical_resampled.dataobj, dtype=np.int16)
    subcortical_data = np.asarray(subcortical_resampled.dataobj, dtype=np.int16)
    world_x = build_world_x_grid(reference_img.shape, reference_img.affine)

    rois: List[ROI] = []

    for value, base_name in parse_label_xml(FSL_ATLAS_ROOT / "HarvardOxford-Cortical.xml"):
        division = CORTICAL_DIVISIONS.get(base_name)
        if division is None:
            raise KeyError(f"Missing cortical division mapping for ROI {base_name!r}.")
        base_mask = cortical_data == int(value)
        if not bool(base_mask.any()):
            continue
        for side_name, side_mask in (
            ("Left", np.asarray(base_mask & (world_x < 0.0), dtype=bool)),
            ("Right", np.asarray(base_mask & (world_x > 0.0), dtype=bool)),
        ):
            flat_indices = np.flatnonzero(side_mask.ravel())
            if flat_indices.size == 0:
                continue
            rois.append(
                ROI(
                    name=f"{side_name} {base_name}",
                    division=division,
                    atlas_group="cortical",
                    voxel_indices=flat_indices.astype(np.int64, copy=False),
                    num_voxels=int(flat_indices.size),
                )
            )

    for value, base_name in parse_label_xml(FSL_ATLAS_ROOT / "HarvardOxford-Subcortical.xml"):
        if should_skip_subcortical_roi(base_name):
            continue
        base_mask = subcortical_data == int(value)
        flat_indices = np.flatnonzero(base_mask.ravel())
        if flat_indices.size == 0:
            continue
        rois.append(
            ROI(
                name=base_name,
                division="Subcortical",
                atlas_group="subcortical",
                voxel_indices=flat_indices.astype(np.int64, copy=False),
                num_voxels=int(flat_indices.size),
            )
        )

    summary = {
        "num_rois": int(len(rois)),
        "num_cortical_rois": int(sum(roi.atlas_group == "cortical" for roi in rois)),
        "num_subcortical_rois": int(sum(roi.atlas_group == "subcortical" for roi in rois)),
        "division_counts": {
            division: int(sum(roi.division == division for roi in rois)) for division in DIVISION_ORDER
        },
    }
    return rois, summary


def positive_saliency_volume(volume: np.ndarray, *, brain_mask: np.ndarray | None = None) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if brain_mask is not None:
        array = np.where(np.asarray(brain_mask, dtype=bool), array, 0.0)
    array = np.clip(array, 0.0, None)
    return array.astype(np.float32, copy=False)


def normalize_projection(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(np.percentile(np.abs(array), 99.0)) if array.size else 0.0
    if vmax <= 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip(np.abs(array) / vmax, 0.0, 1.0).astype(np.float32, copy=False)


def mip_triptych(background_volume: np.ndarray, overlay_volume: np.ndarray | None) -> np.ndarray:
    projections = []
    cmap = plt.get_cmap("turbo")
    for axis in range(3):
        background = normalize_projection(np.max(np.abs(background_volume), axis=axis))
        base_rgb = np.stack([background, background, background], axis=-1)
        if overlay_volume is None:
            projections.append(base_rgb)
            continue
        heat = np.max(overlay_volume, axis=axis)
        if float(heat.max()) > float(heat.min()):
            heat = (heat - float(heat.min())) / float(heat.max() - heat.min())
        else:
            heat = np.zeros_like(heat, dtype=np.float32)
        heat_rgb = cmap(heat)[..., :3]
        alpha = np.clip(heat, 0.0, 1.0)[..., None] * 0.9
        projections.append((1.0 - alpha) * base_rgb + alpha * heat_rgb)

    target_height = max(panel.shape[0] for panel in projections)
    padded = []
    for panel in projections:
        pad_total = target_height - panel.shape[0]
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        padded.append(
            np.pad(
                panel,
                ((pad_top, pad_bottom), (0, 0), (0, 0)),
                mode="constant",
                constant_values=1.0,
            )
        )
    separator = np.ones((target_height, 6, 3), dtype=np.float32)
    return np.concatenate([padded[0], separator, padded[1], separator, padded[2]], axis=1)


def save_nifti(path: Path, data: np.ndarray, reference_img: nib.spatialimages.SpatialImage) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine=reference_img.affine, header=reference_img.header)
    nib.save(image, str(target))


def find_last_relu(model: FMRIConvNet) -> torch.nn.Module:
    for module in reversed(list(model.feature_extractor.children())):
        if isinstance(module, torch.nn.ReLU):
            return module
    raise ValueError("Could not find a ReLU layer in model.feature_extractor.")


def disable_inplace_relu(module: torch.nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, torch.nn.ReLU):
            child.inplace = False


def compute_layercam_batch(
    model: FMRIConvNet,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    activation_store: Dict[str, torch.Tensor] = {}
    handle = find_last_relu(model).register_forward_hook(
        lambda _module, _inputs, output: activation_store.setdefault("value", output)
    )
    try:
        model.zero_grad(set_to_none=True)
        logits = model.forward_raw(x)
        probs_high = torch.sigmoid(logits)
        preds = (probs_high >= 0.5).long()

        activation = activation_store.get("value")
        if activation is None:
            raise RuntimeError("Forward hook did not capture the last ReLU activation.")
        activation.retain_grad()

        target_scores = torch.where(y.long() == 1, logits, -logits)
        target_scores.sum().backward()
        gradients = activation.grad
        if gradients is None:
            raise RuntimeError("Activation gradient is missing after backward().")
        raw_cam = torch.relu((torch.relu(gradients) * activation).sum(dim=1, keepdim=True))
        upsampled = F.interpolate(raw_cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        cams = upsampled.squeeze(1)

        return {
            "logits": logits.detach(),
            "probs_high": probs_high.detach(),
            "preds": preds.detach(),
            "cams": cams.detach(),
        }
    finally:
        handle.remove()


def batched_ranges(length: int, batch_size: int) -> Iterable[range]:
    for start in range(0, int(length), int(batch_size)):
        yield range(start, min(start + int(batch_size), int(length)))


def empty_accumulator(shape: Sequence[int], num_rois: int) -> Dict[str, Any]:
    return {
        "num_total_trials": 0,
        "num_correct_trials": 0,
        "sum_map": np.zeros(shape, dtype=np.float64),
        "sumsq_map": np.zeros(shape, dtype=np.float64),
        "sum_background": np.zeros(shape, dtype=np.float64),
        "roi_rows": [],
        "sample_rows": [],
    }


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p_array = np.asarray([float(value) for value in p_values], dtype=np.float64)
    if p_array.size == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(p_array)
    ranked = p_array[order]
    q_values = np.empty_like(ranked)
    n = float(ranked.size)
    running = 1.0
    for reverse_index in range(ranked.size - 1, -1, -1):
        rank = float(reverse_index + 1)
        candidate = ranked[reverse_index] * n / rank
        running = min(running, candidate)
        q_values[reverse_index] = running
    restored = np.empty_like(q_values)
    restored[order] = np.clip(q_values, 0.0, 1.0)
    return restored


def compute_roi_statistics(
    roi_rows: np.ndarray,
    rois: Sequence[ROI],
) -> List[Dict[str, Any]]:
    if roi_rows.ndim != 2 or roi_rows.shape[1] != len(rois):
        raise ValueError(f"roi_rows must have shape [trials, num_rois], got {roi_rows.shape} and {len(rois)} ROIs.")

    p_values: List[float] = []
    rows: List[Dict[str, Any]] = []
    for roi_index, roi in enumerate(rois):
        values = np.asarray(roi_rows[:, roi_index], dtype=np.float64)
        mean_value = float(np.mean(values)) if values.size else 0.0
        median_value = float(np.median(values)) if values.size else 0.0
        try:
            if values.size == 0 or np.allclose(values, 0.0):
                p_value = 1.0
                statistic = 0.0
            else:
                statistic, p_value = stats.wilcoxon(values, alternative="greater", zero_method="wilcox")
                statistic = float(statistic)
                p_value = float(p_value)
        except ValueError:
            statistic = 0.0
            p_value = 1.0
        p_values.append(p_value)
        rows.append(
            {
                "roi_name": roi.name,
                "division": roi.division,
                "atlas_group": roi.atlas_group,
                "num_voxels": roi.num_voxels,
                "num_trials": int(values.size),
                "mean_value": mean_value,
                "median_value": median_value,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )

    q_values = benjamini_hochberg(p_values)
    for row, q_value in zip(rows, q_values.tolist()):
        row["q_value"] = float(q_value)
        row["significant_fdr_0_05"] = bool(q_value < 0.05)
    return rows


def compute_cliffs_delta(
    values_high: np.ndarray,
    values_low: np.ndarray,
) -> float:
    n_high = int(values_high.size)
    n_low = int(values_low.size)
    if n_high == 0 or n_low == 0:
        return 0.0
    try:
        statistic, _ = stats.mannwhitneyu(values_high, values_low, alternative="two-sided")
        return float((2.0 * float(statistic) / float(n_high * n_low)) - 1.0)
    except ValueError:
        return 0.0


def compute_between_class_roi_statistics(
    roi_rows_low: np.ndarray,
    roi_rows_high: np.ndarray,
    rois: Sequence[ROI],
) -> List[Dict[str, Any]]:
    if roi_rows_low.ndim != 2 or roi_rows_low.shape[1] != len(rois):
        raise ValueError(
            f"roi_rows_low must have shape [trials, num_rois], got {roi_rows_low.shape} and {len(rois)} ROIs."
        )
    if roi_rows_high.ndim != 2 or roi_rows_high.shape[1] != len(rois):
        raise ValueError(
            f"roi_rows_high must have shape [trials, num_rois], got {roi_rows_high.shape} and {len(rois)} ROIs."
        )

    p_values: List[float] = []
    rows: List[Dict[str, Any]] = []
    for roi_index, roi in enumerate(rois):
        values_low = np.asarray(roi_rows_low[:, roi_index], dtype=np.float64)
        values_high = np.asarray(roi_rows_high[:, roi_index], dtype=np.float64)
        mean_low = float(np.mean(values_low)) if values_low.size else 0.0
        mean_high = float(np.mean(values_high)) if values_high.size else 0.0
        median_low = float(np.median(values_low)) if values_low.size else 0.0
        median_high = float(np.median(values_high)) if values_high.size else 0.0
        try:
            statistic, p_value = stats.mannwhitneyu(values_high, values_low, alternative="two-sided")
            statistic = float(statistic)
            p_value = float(p_value)
        except ValueError:
            statistic = 0.0
            p_value = 1.0
        p_values.append(p_value)
        diff_mean = mean_high - mean_low
        diff_median = median_high - median_low
        cliffs_delta = compute_cliffs_delta(values_high, values_low)
        rows.append(
            {
                "roi_name": roi.name,
                "division": roi.division,
                "atlas_group": roi.atlas_group,
                "num_voxels": roi.num_voxels,
                "num_trials_rating_1": int(values_low.size),
                "num_trials_rating_4": int(values_high.size),
                "mean_rating_1": mean_low,
                "mean_rating_4": mean_high,
                "median_rating_1": median_low,
                "median_rating_4": median_high,
                "diff_mean_rating4_minus_rating1": diff_mean,
                "diff_median_rating4_minus_rating1": diff_median,
                "mannwhitney_u": statistic,
                "p_value": p_value,
                "cliffs_delta_rating4_vs_rating1": cliffs_delta,
            }
        )

    q_values = benjamini_hochberg(p_values)
    for row, q_value in zip(rows, q_values.tolist()):
        row["q_value"] = float(q_value)
        row["significant_fdr_0_05"] = bool(q_value < 0.05)
        if row["diff_mean_rating4_minus_rating1"] > 0:
            row["preferred_rating"] = 4
        elif row["diff_mean_rating4_minus_rating1"] < 0:
            row["preferred_rating"] = 1
        else:
            row["preferred_rating"] = 0
    return rows


def division_summary_rows(roi_stats: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for division in DIVISION_ORDER:
        division_rows = [row for row in roi_stats if row["division"] == division]
        values = [float(row["median_value"]) for row in division_rows]
        rows.append(
            {
                "division": division,
                "num_rois": int(len(division_rows)),
                "median_of_roi_medians": float(np.median(values)) if values else 0.0,
                "mean_of_roi_means": float(np.mean([float(row["mean_value"]) for row in division_rows])) if values else 0.0,
                "num_significant_rois": int(sum(bool(row["significant_fdr_0_05"]) for row in division_rows)),
            }
        )
    return rows


def figure_group_triptychs(
    *,
    output_path: Path,
    accumulators: Dict[int, Dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12.5, 8.0))
    class_specs = [
        (0, f"Rating {LOW_RATING} | correct held-out trials"),
        (1, f"Rating {HIGH_RATING} | correct held-out trials"),
    ]
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for axis, (target, label) in zip(axes, class_specs):
        accumulator = accumulators[target]
        count = int(accumulator["num_correct_trials"])
        if count <= 0:
            axis.text(0.5, 0.5, f"{label}\n(no correct trials)", ha="center", va="center", fontsize=14)
            axis.axis("off")
            continue
        background = np.asarray(accumulator["sum_background"] / count, dtype=np.float32)
        mean_map = np.asarray(accumulator["sum_map"] / count, dtype=np.float32)
        triptych = mip_triptych(background, mean_map)
        axis.imshow(triptych)
        axis.set_title(f"{label} (n={count})", fontsize=14)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Held-Out Group Saliency Maps (paper-path Layer-CAM style)", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_division_distribution(
    *,
    output_path: Path,
    rating1_roi_stats: Sequence[Dict[str, Any]],
    rating4_roi_stats: Sequence[Dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13.5, 5.3), sharey=True)
    panel_specs = [
        ("A", rating1_roi_stats, f"Rating {LOW_RATING}"),
        ("B", rating4_roi_stats, f"Rating {HIGH_RATING}"),
    ]
    for axis, (panel_label, roi_stats, title_text) in zip(axes, panel_specs):
        grouped = [[float(row["median_value"]) for row in roi_stats if row["division"] == division] for division in DIVISION_ORDER]
        box = axis.boxplot(grouped, patch_artist=True, widths=0.55, showfliers=False)
        for patch in box["boxes"]:
            patch.set(facecolor="#d9d9d9", edgecolor="#4d4d4d", linewidth=1.0)
        for median in box["medians"]:
            median.set(color="#111111", linewidth=1.3)
        rng = np.random.default_rng(42)
        for position, values in enumerate(grouped, start=1):
            if not values:
                continue
            x = rng.normal(loc=position, scale=0.055, size=len(values))
            axis.scatter(x, values, s=16, color="#222222", alpha=0.65, linewidths=0)
        axis.set_xticks(np.arange(1, len(DIVISION_ORDER) + 1))
        axis.set_xticklabels(DIVISION_ORDER, rotation=20, ha="right")
        axis.set_ylabel("Median saliency per ROI")
        axis.set_ylim(0.0, 0.12)
        axis.set_yticks(np.arange(0.0, 0.121, 0.02))
        axis.set_title(f"{panel_label}. {title_text}", loc="left", fontsize=13)
        axis.grid(axis="y", alpha=0.18, linewidth=0.6)
    fig.suptitle("Regional Distribution of Saliency Values", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_top_rois(
    *,
    output_path: Path,
    roi_stats: Sequence[Dict[str, Any]],
    title_text: str,
) -> None:
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10.5, 13.5))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    x_ticks = [1e-2, 2e-2, 5e-2, 1e-1]
    x_tick_labels = ["0.01", "0.02", "0.05", "0.10"]
    for axis, division in zip(axes, DIVISION_ORDER):
        division_rows = [row for row in roi_stats if row["division"] == division]
        top_rows = sorted(division_rows, key=lambda row: float(row["mean_value"]), reverse=True)[:5]
        top_rows = list(reversed(top_rows))
        names = [row["roi_name"] + (" *" if bool(row["significant_fdr_0_05"]) else "") for row in top_rows]
        values = [max(float(row["mean_value"]), 1e-6) for row in top_rows]
        axis.barh(names, values, color="#bdbdbd", edgecolor="#4d4d4d", linewidth=0.9)
        axis.set_xscale("log")
        axis.set_xlim(1e-2, 1.2e-1)
        axis.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
        axis.xaxis.set_major_formatter(ticker.FixedFormatter(x_tick_labels))
        axis.xaxis.set_minor_locator(ticker.NullLocator())
        axis.tick_params(axis="x", labelsize=9, pad=2)
        axis.tick_params(axis="y", labelsize=8)
        axis.set_title(division, loc="left", fontsize=12)
        axis.grid(axis="x", alpha=0.18, linewidth=0.6)
    fig.suptitle(title_text, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_between_class_difference(
    *,
    output_path: Path,
    comparison_rows: Sequence[Dict[str, Any]],
) -> None:
    positive_rows = [
        row for row in comparison_rows
        if float(row["diff_mean_rating4_minus_rating1"]) > 0 and bool(row["significant_fdr_0_05"])
    ]
    negative_rows = [
        row for row in comparison_rows
        if float(row["diff_mean_rating4_minus_rating1"]) < 0 and bool(row["significant_fdr_0_05"])
    ]
    positive_rows = sorted(positive_rows, key=lambda row: float(row["diff_mean_rating4_minus_rating1"]), reverse=True)[:15]
    negative_rows = sorted(negative_rows, key=lambda row: float(row["diff_mean_rating4_minus_rating1"]))[:15]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15.0, 8.5))
    panel_specs = [
        (
            axes[0],
            positive_rows,
            "Rating 4 > Rating 1",
            "#bdbdbd",
            lambda row: float(row["diff_mean_rating4_minus_rating1"]),
        ),
        (
            axes[1],
            negative_rows,
            "Rating 1 > Rating 4",
            "#bdbdbd",
            lambda row: abs(float(row["diff_mean_rating4_minus_rating1"])),
        ),
    ]
    for axis, rows, title_text, color, value_fn in panel_specs:
        if not rows:
            axis.text(0.5, 0.5, "No significant ROI", ha="center", va="center", fontsize=12)
            axis.axis("off")
            continue
        display_rows = list(reversed(rows))
        names = [f'{row["roi_name"]} *' for row in display_rows]
        values = [float(value_fn(row)) for row in display_rows]
        axis.barh(names, values, color=color, edgecolor="#4d4d4d", linewidth=0.9)
        axis.set_title(title_text, fontsize=13)
        axis.grid(axis="x", alpha=0.18, linewidth=0.6)
        axis.set_xlabel("Absolute mean saliency difference")
    fig.suptitle("Top ROI Separation Between Rating 1 and Rating 4", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_run(
    *,
    run_dir: Path,
    output_dir: Path,
    local_output_root: Path,
    device_name: str,
    batch_size: int,
    only_fold: int | None = None,
    checkpoint_name: str = "final.pt",
    dataset_paths: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    run_config = read_json(run_dir / "run_config.json")
    fold_assignments = read_json(run_dir / "fold_assignments.json")
    resolved_paths = dataset_paths or resolve_data_paths(run_config, local_output_root)
    device = select_device(device_name)

    records, dataset_summary = discover_samples(
        beta_index_csv=resolved_paths["beta_index_csv"],
        models_root=resolved_paths["models_root"],
        brain_mask_root=resolved_paths["brain_mask_root"],
        path_prefix_from=resolved_paths["path_prefix_from"],
        path_prefix_to=resolved_paths["path_prefix_to"],
    )
    if not records:
        raise ValueError("No usable records were discovered for Grad-CAM analysis.")

    reference_img = nib.load(str(records[0].mask_path))
    rois, roi_inventory = build_rois(reference_img)

    accumulators = {
        0: empty_accumulator(reference_img.shape, len(rois)),
        1: empty_accumulator(reference_img.shape, len(rois)),
    }

    sample_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []

    for fold_payload in fold_assignments["folds"]:
        fold_index = int(fold_payload["fold_index"])
        if only_fold is not None and fold_index != int(only_fold):
            continue
        print(f"[gradcam] fold {fold_index:02d}: loading held-out validation records", flush=True)
        val_records = filter_records_by_subjects(records, fold_payload["val_subjects"])
        val_dataset = FMRIBetaDataset(val_records)
        checkpoint_path = run_dir / f"fold_{fold_index:02d}" / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Could not find checkpoint for fold {fold_index:02d}: {checkpoint_path}. "
                "For legacy early-stopping runs, pass --checkpoint-name best.pt."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = FMRIConvNet(
            input_shape=tuple(int(value) for value in checkpoint["input_shape"]),
            hidden_dim=int(run_config["hidden_dim"]),
            dropout=float(run_config.get("dropout", 0.5)),
        )
        disable_inplace_relu(model)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        fold_total = 0
        fold_correct = 0

        for batch_range in batched_ranges(len(val_dataset), batch_size):
            items = [val_dataset[index] for index in batch_range]
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
            correct_indices = np.flatnonzero(correct_mask)
            if correct_indices.size > 0:
                x_correct = x[correct_mask_tensor].detach().clone().requires_grad_(True)
                y_correct = y[correct_mask_tensor].detach().clone()
                with torch.enable_grad():
                    cam_output = compute_layercam_batch(model, x_correct, y_correct)
                cams[correct_indices] = cam_output["cams"].detach().cpu().numpy().astype(np.float32)

            fold_total += int(len(items))
            fold_correct += int(correct_mask.sum())

            for local_index, dataset_index in enumerate(batch_range):
                item = items[local_index]
                record = val_records[int(dataset_index)]
                target = int(item["y"].item())
                rating = int(record.rating)
                pred = int(preds[local_index])
                prob_high = float(probs_high[local_index])
                is_correct = bool(correct_mask[local_index])

                accumulators[target]["num_total_trials"] += 1
                sample_row = {
                    "fold_index": fold_index,
                    "subject_id": record.subject_id,
                    "session_id": record.session_id,
                    "run_id": record.run_id,
                    "trial_id": record.trial_id,
                    "sample_path": record.sample_path,
                    "mask_path": record.mask_path,
                    "true_rating": rating,
                    "true_target": target,
                    "pred_target": pred,
                    "prob_high": prob_high,
                    "correct": is_correct,
                }
                sample_rows.append(sample_row)

                if not is_correct:
                    continue

                brain_mask = val_dataset._load_mask(Path(record.mask_path))
                cam = positive_saliency_volume(cams[local_index], brain_mask=brain_mask)
                background = normalize_projection(np.abs(item["x"][0].numpy()))
                flat_cam = cam.reshape(-1)
                roi_vector = np.asarray(
                    [float(flat_cam[roi.voxel_indices].mean()) for roi in rois],
                    dtype=np.float32,
                )

                accumulators[target]["num_correct_trials"] += 1
                accumulators[target]["sum_map"] += cam
                accumulators[target]["sumsq_map"] += np.square(cam, dtype=np.float64)
                accumulators[target]["sum_background"] += background
                accumulators[target]["roi_rows"].append(roi_vector)
                accumulators[target]["sample_rows"].append(sample_row)

        fold_rows.append(
            {
                "fold_index": fold_index,
                "num_val_trials": int(fold_total),
                "num_correct_trials": int(fold_correct),
                "correct_fraction": float(fold_correct / fold_total) if fold_total > 0 else 0.0,
                "checkpoint_name": checkpoint_name,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "reported_epoch": int(checkpoint.get("reported_epoch", checkpoint.get("epoch", checkpoint.get("best_epoch", 0)))),
                "selection_policy": str(checkpoint.get("selection_policy", "legacy_checkpoint")),
                "reported_val_loss": float(
                    checkpoint.get("val_metrics", checkpoint.get("best_val_metrics", {})).get("loss", math.nan)
                ),
            }
        )
        print(
            f"[gradcam] fold {fold_index:02d}: correct held-out trials {fold_correct}/{fold_total}",
            flush=True,
        )

    roi_stats_by_target: Dict[int, List[Dict[str, Any]]] = {}
    roi_rows_by_target: Dict[int, np.ndarray] = {}

    analysis_summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "batch_size": int(batch_size),
        "method": {
            "saliency": "paper-path Layer-CAM style Grad-CAM (no GAP, positive gradient weighting)",
            "selection": (
                f"{checkpoint_name} checkpoint from fold {int(only_fold):02d}"
                if only_fold is not None
                else f"per-fold {checkpoint_name} checkpoint"
            ),
            "evaluation_split": (
                f"correct held-out validation trials from fold {int(only_fold):02d}"
                if only_fold is not None
                else "correct held-out validation trials aggregated across 10 folds"
            ),
            "per_trial_saliency_normalization": "none",
            "roi_test": "one-sample Wilcoxon signed-rank against zero with BH-FDR",
        },
        "resolved_paths": resolved_paths,
        "dataset_summary": dataset_summary,
        "roi_inventory": roi_inventory,
        "fold_summary": fold_rows,
        "classes": {},
    }

    for target, rating in ((0, LOW_RATING), (1, HIGH_RATING)):
        accumulator = accumulators[target]
        num_correct = int(accumulator["num_correct_trials"])
        if num_correct <= 0:
            analysis_summary["classes"][str(target)] = {
                "rating": int(rating),
                "num_total_trials": int(accumulator["num_total_trials"]),
                "num_correct_trials": 0,
            }
            continue

        roi_rows = np.stack(accumulator["roi_rows"], axis=0)
        roi_rows_by_target[target] = roi_rows
        roi_stats = compute_roi_statistics(roi_rows, rois)
        roi_stats_by_target[target] = roi_stats
        division_rows = division_summary_rows(roi_stats)
        save_csv_rows = []
        for row in roi_stats:
            save_row = {"rating": int(rating), **row}
            save_csv_rows.append(save_row)
        write_csv(
            output_dir / f"rating_{rating}_roi_summary.csv",
            save_csv_rows,
            [
                "rating",
                "roi_name",
                "division",
                "atlas_group",
                "num_voxels",
                "num_trials",
                "mean_value",
                "median_value",
                "wilcoxon_statistic",
                "p_value",
                "q_value",
                "significant_fdr_0_05",
            ],
        )

        mean_map = np.asarray(accumulator["sum_map"] / num_correct, dtype=np.float32)
        save_nifti(output_dir / f"rating_{rating}_mean_saliency.nii.gz", mean_map, reference_img)

        if num_correct > 1:
            numerator = accumulator["sum_map"]
            denominator = float(num_correct)
            mean64 = numerator / denominator
            variance = np.maximum(
                (accumulator["sumsq_map"] - (np.square(numerator) / denominator)) / float(num_correct - 1),
                0.0,
            )
            stderr = np.sqrt(variance / denominator)
            t_map = np.divide(mean64, stderr, out=np.zeros_like(mean64), where=stderr > 1e-12).astype(np.float32)
        else:
            t_map = np.zeros(reference_img.shape, dtype=np.float32)
        save_nifti(output_dir / f"rating_{rating}_t_stat.nii.gz", t_map, reference_img)

        analysis_summary["classes"][str(target)] = {
            "rating": int(rating),
            "num_total_trials": int(accumulator["num_total_trials"]),
            "num_correct_trials": num_correct,
            "correct_fraction_within_rating": (
                float(num_correct / accumulator["num_total_trials"])
                if int(accumulator["num_total_trials"]) > 0
                else 0.0
            ),
            "num_significant_rois_fdr_0_05": int(sum(bool(row["significant_fdr_0_05"]) for row in roi_stats)),
            "top_rois_by_mean": sorted(
                (
                    {
                        "roi_name": row["roi_name"],
                        "division": row["division"],
                        "mean_value": float(row["mean_value"]),
                        "q_value": float(row["q_value"]),
                        "significant_fdr_0_05": bool(row["significant_fdr_0_05"]),
                    }
                    for row in roi_stats
                ),
                key=lambda row: row["mean_value"],
                reverse=True,
            )[:15],
            "division_summary": division_rows,
        }

        figure_top_rois(
            output_path=output_dir / f"rating_{rating}_top_rois.png",
            roi_stats=roi_stats,
            title_text=f"Top-Ranked ROIs Contributing to Rating {rating} Decoding",
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

    figure_group_triptychs(
        output_path=output_dir / "group_saliency_triptychs.png",
        accumulators=accumulators,
    )

    figure_division_distribution(
        output_path=output_dir / "regional_distribution.png",
        rating1_roi_stats=roi_stats_by_target.get(0, []),
        rating4_roi_stats=roi_stats_by_target.get(1, []),
    )

    if 0 in roi_rows_by_target and 1 in roi_rows_by_target:
        comparison_rows = compute_between_class_roi_statistics(
            roi_rows_low=roi_rows_by_target[0],
            roi_rows_high=roi_rows_by_target[1],
            rois=rois,
        )
        write_csv(
            output_dir / "rating_1_vs_rating_4_roi_comparison.csv",
            comparison_rows,
            [
                "roi_name",
                "division",
                "atlas_group",
                "num_voxels",
                "num_trials_rating_1",
                "num_trials_rating_4",
                "mean_rating_1",
                "mean_rating_4",
                "median_rating_1",
                "median_rating_4",
                "diff_mean_rating4_minus_rating1",
                "diff_median_rating4_minus_rating1",
                "mannwhitney_u",
                "p_value",
                "q_value",
                "significant_fdr_0_05",
                "cliffs_delta_rating4_vs_rating1",
                "preferred_rating",
            ],
        )
        figure_between_class_difference(
            output_path=output_dir / "rating_1_vs_rating_4_top_differences.png",
            comparison_rows=comparison_rows,
        )
        significant_rows = [row for row in comparison_rows if bool(row["significant_fdr_0_05"])]
        significant_rating4 = sorted(
            (
                row for row in significant_rows
                if int(row["preferred_rating"]) == 4
            ),
            key=lambda row: float(row["diff_mean_rating4_minus_rating1"]),
            reverse=True,
        )
        significant_rating1 = sorted(
            (
                row for row in significant_rows
                if int(row["preferred_rating"]) == 1
            ),
            key=lambda row: float(row["diff_mean_rating4_minus_rating1"]),
        )
        analysis_summary["between_class_comparison"] = {
            "contrast": f"Rating {LOW_RATING} vs Rating {HIGH_RATING}",
            "statistical_test": "two-sided Mann-Whitney U across correct held-out trials with BH-FDR",
            "effect_size": "Cliff's delta (positive means Rating 4 > Rating 1)",
            "num_significant_rois_fdr_0_05": int(len(significant_rows)),
            "num_significant_rois_preferring_rating_4": int(len(significant_rating4)),
            "num_significant_rois_preferring_rating_1": int(len(significant_rating1)),
            "top_rois_rating_4_greater_than_rating_1": [
                {
                    "roi_name": row["roi_name"],
                    "division": row["division"],
                    "diff_mean_rating4_minus_rating1": float(row["diff_mean_rating4_minus_rating1"]),
                    "q_value": float(row["q_value"]),
                    "cliffs_delta_rating4_vs_rating1": float(row["cliffs_delta_rating4_vs_rating1"]),
                }
                for row in significant_rating4[:15]
            ],
            "top_rois_rating_1_greater_than_rating_4": [
                {
                    "roi_name": row["roi_name"],
                    "division": row["division"],
                    "diff_mean_rating4_minus_rating1": float(row["diff_mean_rating4_minus_rating1"]),
                    "q_value": float(row["q_value"]),
                    "cliffs_delta_rating4_vs_rating1": float(row["cliffs_delta_rating4_vs_rating1"]),
                }
                for row in significant_rating1[:15]
            ],
        }

    save_json(output_dir / "analysis_summary.json", analysis_summary)
    print(f"[gradcam] analysis complete: {output_dir}", flush=True)
    return analysis_summary


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = latest_run_dir() if args.run_dir is None else Path(args.run_dir).expanduser().resolve()
    output_dir = ensure_dir(args.output_dir or (run_dir / "paper_gradcam"))
    local_output_root = Path(args.local_output_root).expanduser().resolve()
    analysis_summary = analyze_run(
        run_dir=run_dir,
        output_dir=output_dir,
        local_output_root=local_output_root,
        device_name=args.device,
        batch_size=int(args.batch_size),
        only_fold=None if args.only_fold is None else int(args.only_fold),
        checkpoint_name=str(args.checkpoint_name),
        dataset_paths=explicit_dataset_paths(args),
    )
    print(json.dumps(analysis_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
