#!/usr/bin/env python3
"""Build the N=30 searchlight MVPA cluster-FWE result.

This script reuses the completed subject-level rating4-vs-rating1 searchlight
maps, excludes sub-016, restacks the remaining balanced-accuracy-minus-chance
maps, and runs FSL randomise with a cluster-forming threshold of one-sided
p < 0.001 and cluster-level FWE p < 0.05.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage, stats

from analyses._paths import OUTPUT_ROOT, RUNS_ROOT, FSL_ROOT


SOURCE_BATCH_ROOT = (OUTPUT_ROOT / "searchlight" /  # pass --source-batch-root for a new subject-level run
    "20260425_075751_rating4_vs_rating1_libsvmLinearSVC_ses3fold_radius6mm_fastindex"
)
MVPA1_ROOT = RUNS_ROOT / "searchlight_mvpa1"
ANALYSIS_DIRNAME = "searchlight_rating4_vs_rating1_ses3fold_n30_cluster_p001_cluster_fwe"
DEFAULT_OUTPUT_ROOT = MVPA1_ROOT / "RESULTS" / ANALYSIS_DIRNAME
DEFAULT_WORKBENCH_ROOT = MVPA1_ROOT / "WORKBENCH" / ANALYSIS_DIRNAME
DEFAULT_RANDOMISE_BIN = FSL_ROOT / "bin" / "randomise"
DEFAULT_PERMUTATIONS = 5000
DEFAULT_SEED = 42
DEFAULT_EXCLUDE_SUBJECTS = ("sub-016",)
CLUSTER_FORMING_P = 0.001
CLUSTER_FWE_P = 0.05
ANALYSIS_NAME = "rating4_vs_rating1_searchlight_balanced_accuracy_minus_chance_n30_cluster_p001_cluster_fwe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch-root", type=Path, default=SOURCE_BATCH_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workbench-root", type=Path, default=DEFAULT_WORKBENCH_ROOT)
    parser.add_argument("--randomise-bin", type=Path, default=DEFAULT_RANDOMISE_BIN)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--exclude-subject", action="append", default=list(DEFAULT_EXCLUDE_SUBJECTS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-randomise", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def save_like(path: Path, data: np.ndarray, reference_img: nib.Nifti1Image) -> None:
    image = nib.Nifti1Image(np.asarray(data), affine=reference_img.affine, header=reference_img.header.copy())
    image.set_data_dtype(np.asarray(data).dtype)
    nib.save(image, str(path))


def clean_float_image(source_path: Path, target_path: Path) -> nib.Nifti1Image:
    image = nib.load(str(source_path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    save_like(target_path, data, image)
    return nib.load(str(target_path))


def stack_niftis(paths: Sequence[Path]) -> nib.Nifti1Image:
    if not paths:
        raise ValueError("No NIfTI files were provided for stacking.")
    first = nib.load(str(paths[0]))
    arrays: list[np.ndarray] = []
    for path in paths:
        image = nib.load(str(path))
        if image.shape != first.shape:
            raise ValueError(f"Shape mismatch: {path} has {image.shape}, expected {first.shape}")
        if not np.allclose(image.affine, first.affine):
            raise ValueError(f"Affine mismatch: {path}")
        arrays.append(np.asarray(image.dataobj, dtype=np.float32))
    stacked = np.stack(arrays, axis=-1).astype(np.float32)
    return nib.Nifti1Image(stacked, affine=first.affine, header=first.header.copy())


def resolve_randomise_bin(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    resolved = shutil.which(str(path))
    if resolved is None:
        raise FileNotFoundError(f"Could not resolve randomise binary from {path}")
    return Path(resolved).resolve()


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        ensure_dir(target.parent)
        shutil.copy2(source, target)


def copy_map(source: Path, target: Path) -> str:
    ensure_dir(target.parent)
    shutil.copy2(source, target)
    return str(target)


def load_source_summary(source_batch_root: Path) -> dict[str, Any]:
    summary_path = source_batch_root / "group_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing source group summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_subject_manifest(
    source_summary: dict[str, Any],
    excluded_subjects: set[str],
    subject_map_dir: Path,
) -> tuple[list[dict[str, Any]], list[Path], list[str]]:
    manifest_rows: list[dict[str, Any]] = []
    target_maps: list[Path] = []
    subject_ids: list[str] = []

    for subject in sorted(source_summary["subjects"], key=lambda row: str(row["subject_id"])):
        subject_id = str(subject["subject_id"])
        source_map = Path(subject["balanced_accuracy_minus_chance_map"])
        included = subject_id not in excluded_subjects
        exclusion_reason = ""
        if subject_id in excluded_subjects:
            exclusion_reason = "manually_excluded_incomplete_session_run_coverage"
        elif int(subject.get("num_cv_groups", 0)) != 3:
            included = False
            exclusion_reason = "missing_required_three_session_cv_groups"
        elif not source_map.exists():
            included = False
            exclusion_reason = "missing_subject_balanced_accuracy_minus_chance_map"

        target_map = subject_map_dir / f"{subject_id}_searchlight_balancedAccMinusChance.nii.gz"
        row = {
            "subject_id": subject_id,
            "included_in_n30": int(included),
            "exclusion_reason": exclusion_reason,
            "num_runs": int(subject.get("num_runs", 0)),
            "num_examples": int(subject.get("num_examples", 0)),
            "num_cv_groups": int(subject.get("num_cv_groups", 0)),
            "fold_keys": ";".join(subject.get("fold_keys", [])),
            "folds_json": json.dumps(subject.get("folds", []), ensure_ascii=False),
            "mean_searchlight_balanced_accuracy": float(subject.get("mean_searchlight_balanced_accuracy", np.nan)),
            "std_searchlight_balanced_accuracy": float(subject.get("std_searchlight_balanced_accuracy", np.nan)),
            "source_balanced_accuracy_minus_chance_map": str(source_map),
            "copied_balanced_accuracy_minus_chance_map": str(target_map) if included else "",
        }
        manifest_rows.append(row)

        if included:
            clean_float_image(source_map, target_map)
            target_maps.append(target_map)
            subject_ids.append(subject_id)

    if len(subject_ids) != 30:
        raise RuntimeError(f"Expected N=30 after exclusion, found N={len(subject_ids)}")
    if "sub-016" in subject_ids:
        raise RuntimeError("sub-016 is still included; expected it to be excluded.")
    return manifest_rows, target_maps, subject_ids


def compute_group_maps(input_4d_path: Path, mask_path: Path, output_dir: Path) -> dict[str, str]:
    image = nib.load(str(input_4d_path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    mask_img = nib.load(str(mask_path))
    mask = np.asarray(mask_img.dataobj) > 0
    if image.shape[:3] != mask.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match input shape {image.shape[:3]}")

    n_subjects = data.shape[-1]
    mean_data = np.mean(data, axis=-1).astype(np.float32)
    sd_data = np.std(data, axis=-1, ddof=1).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_data = mean_data / (sd_data / np.sqrt(n_subjects))
    t_data = np.nan_to_num(t_data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    p_data = stats.t.sf(t_data, df=n_subjects - 1).astype(np.float32)
    p_data[~mask] = 1.0
    mean_data[~mask] = 0.0
    sd_data[~mask] = 0.0
    t_data[~mask] = 0.0

    paths = {
        "mean_map": output_dir / "searchlight_balanced_accuracy_minus_chance_n30_mean.nii.gz",
        "sd_map": output_dir / "searchlight_balanced_accuracy_minus_chance_n30_sd.nii.gz",
        "tstat_map": output_dir / "searchlight_balanced_accuracy_minus_chance_n30_tstat.nii.gz",
        "p_uncorrected_positive_map": output_dir
        / "searchlight_balanced_accuracy_minus_chance_n30_p_uncorrected_positive.nii.gz",
    }
    save_like(paths["mean_map"], mean_data, image)
    save_like(paths["sd_map"], sd_data, image)
    save_like(paths["tstat_map"], t_data, image)
    save_like(paths["p_uncorrected_positive_map"], p_data, image)
    return {key: str(path) for key, path in paths.items()}


def run_randomise_cluster(
    randomise_bin: Path,
    input_4d: Path,
    mask_path: Path,
    out_root: Path,
    permutations: int,
    seed: int,
    cluster_forming_t: float,
) -> list[str]:
    env = dict(os.environ)
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    if len(randomise_bin.parents) >= 4:
        env.setdefault("FSLDIR", str(randomise_bin.parents[3]))
    command = [
        str(randomise_bin),
        "-i",
        str(input_4d),
        "-o",
        str(out_root),
        "-m",
        str(mask_path),
        "-1",
        "-n",
        str(int(permutations)),
        "-c",
        f"{cluster_forming_t:.12g}",
        "-x",
        f"--seed={int(seed)}",
    ]
    ensure_dir(out_root.parent)
    subprocess.run(command, check=True, env=env)
    return command


def mni_from_ijk(affine: np.ndarray, ijk: np.ndarray) -> tuple[float, float, float]:
    xyz = nib.affines.apply_affine(affine, ijk)
    return tuple(float(v) for v in xyz)


def build_cluster_table(
    sig_mask: np.ndarray,
    tstat: np.ndarray,
    corrp: np.ndarray,
    mean_data: np.ndarray,
    reference_img: nib.Nifti1Image,
    connectivity: int,
) -> list[dict[str, Any]]:
    labels, n_clusters = ndimage.label(sig_mask > 0, structure=ndimage.generate_binary_structure(3, connectivity))
    rows: list[dict[str, Any]] = []
    for cluster_id in range(1, int(n_clusters) + 1):
        voxels = np.argwhere(labels == cluster_id)
        if voxels.size == 0:
            continue
        values = tstat[labels == cluster_id]
        mean_values = mean_data[labels == cluster_id]
        corrp_values = corrp[labels == cluster_id]
        peak_idx = int(np.argmax(values))
        peak_ijk = voxels[peak_idx]
        peak_x, peak_y, peak_z = mni_from_ijk(reference_img.affine, peak_ijk)
        peak_mean = float(mean_values[peak_idx])
        rows.append(
            {
                "component_id": cluster_id,
                "connectivity": f"{connectivity}_scipy_structure",
                "size_voxels": int(voxels.shape[0]),
                "peak_t": float(values[peak_idx]),
                "peak_cluster_corrp_1minus_p": float(corrp_values[peak_idx]),
                "peak_cluster_fwe_p": float(1.0 - corrp_values[peak_idx]),
                "peak_mean_balanced_accuracy_minus_chance": peak_mean,
                "peak_mean_balanced_accuracy": float(0.5 + peak_mean),
                "peak_i": int(peak_ijk[0]),
                "peak_j": int(peak_ijk[1]),
                "peak_k": int(peak_ijk[2]),
                "peak_x_mni": peak_x,
                "peak_y_mni": peak_y,
                "peak_z_mni": peak_z,
            }
        )
    rows.sort(key=lambda row: (-int(row["size_voxels"]), -float(row["peak_t"])))
    return rows


def summarize_cluster_randomise(
    out_root: Path,
    mean_map_path: Path,
    reference_img: nib.Nifti1Image,
    cluster_forming_t: float,
    output_root: Path,
) -> dict[str, Any]:
    tstat_path = out_root.with_name(out_root.name + "_tstat1.nii.gz")
    cluster_extent_path = out_root.with_name(out_root.name + "_clustere_tstat1.nii.gz")
    cluster_corrp_path = out_root.with_name(out_root.name + "_clustere_corrp_tstat1.nii.gz")
    cluster_uncorrp_path = out_root.with_name(out_root.name + "_clustere_p_tstat1.nii.gz")
    voxel_corrp_path = out_root.with_name(out_root.name + "_vox_corrp_tstat1.nii.gz")
    voxel_uncorrp_path = out_root.with_name(out_root.name + "_vox_p_tstat1.nii.gz")
    if not cluster_corrp_path.exists():
        available = sorted(path.name for path in out_root.parent.glob(out_root.name + "*"))
        raise FileNotFoundError(f"Missing cluster corrected image for {out_root}. Available: {available}")

    tstat_img = nib.load(str(tstat_path))
    tstat = np.asarray(tstat_img.dataobj, dtype=np.float32)
    if not cluster_extent_path.exists():
        mask_data = np.asarray(reference_img.dataobj) > 0
        supra = (tstat >= cluster_forming_t) & mask_data
        labels_full, n_full = ndimage.label(supra, structure=ndimage.generate_binary_structure(3, 3))
        extent_full = np.zeros(tstat.shape, dtype=np.float32)
        if n_full:
            sizes_full = np.bincount(labels_full.ravel())
            for cluster_index in range(1, int(n_full) + 1):
                extent_full[labels_full == cluster_index] = sizes_full[cluster_index]
        save_like(cluster_extent_path, extent_full, tstat_img)
    corrp_img = nib.load(str(cluster_corrp_path))
    corrp = np.asarray(corrp_img.dataobj, dtype=np.float32)
    mean_img = nib.load(str(mean_map_path))
    mean_data = np.asarray(mean_img.dataobj, dtype=np.float32)

    sig_mask = ((corrp >= (1.0 - CLUSTER_FWE_P)) & (tstat >= cluster_forming_t)).astype(np.uint8)
    sig_mask_path = out_root.with_name(out_root.name + "_clusterFWE05_sigmask.nii.gz")
    masked_tstat_path = out_root.with_name(out_root.name + "_clusterFWE05_tstat_masked.nii.gz")
    masked_mean_path = out_root.with_name(out_root.name + "_clusterFWE05_mean_minus_chance_masked.nii.gz")
    masked_corrp_path = out_root.with_name(out_root.name + "_clusterFWE05_corrp_masked.nii.gz")
    save_like(sig_mask_path, sig_mask, reference_img)
    save_like(masked_tstat_path, np.where(sig_mask > 0, tstat, 0.0).astype(np.float32), tstat_img)
    save_like(masked_mean_path, np.where(sig_mask > 0, mean_data, 0.0).astype(np.float32), mean_img)
    save_like(masked_corrp_path, np.where(sig_mask > 0, corrp, 0.0).astype(np.float32), corrp_img)

    rows_6 = build_cluster_table(sig_mask, tstat, corrp, mean_data, reference_img, connectivity=1)
    rows_26 = build_cluster_table(sig_mask, tstat, corrp, mean_data, reference_img, connectivity=3)
    write_csv(output_root / "cluster_table_balanced_accuracy_minus_chance_cluster_p001_cluster_fwe_p05_6conn.csv", rows_6)
    write_csv(output_root / "cluster_table_balanced_accuracy_minus_chance_cluster_p001_cluster_fwe_p05_26conn.csv", rows_26)

    labels_6, _ = ndimage.label(sig_mask > 0, structure=ndimage.generate_binary_structure(3, 1))
    labels_26, _ = ndimage.label(sig_mask > 0, structure=ndimage.generate_binary_structure(3, 3))
    labels_6_path = out_root.with_name(out_root.name + "_clusterFWE05_clusterLabels_6conn.nii.gz")
    labels_26_path = out_root.with_name(out_root.name + "_clusterFWE05_clusterLabels_26conn.nii.gz")
    save_like(labels_6_path, labels_6.astype(np.int16), reference_img)
    save_like(labels_26_path, labels_26.astype(np.int16), reference_img)

    sig_t = tstat[sig_mask > 0]
    sig_mean = mean_data[sig_mask > 0]
    sig_corrp = corrp[sig_mask > 0]
    cluster_extent_critical = None
    if cluster_extent_path.exists() and sig_mask.sum() > 0:
        extent = np.asarray(nib.load(str(cluster_extent_path)).dataobj, dtype=np.float32)
        positive_extents = extent[(extent > 0) & np.isfinite(extent)]
        if positive_extents.size:
            cluster_extent_critical = int(np.min(positive_extents[positive_extents >= np.min(extent[sig_mask > 0])]))

    return {
        "tstat_path": str(tstat_path),
        "cluster_extent_path": str(cluster_extent_path) if cluster_extent_path.exists() else None,
        "cluster_corrp_path": str(cluster_corrp_path),
        "cluster_uncorrp_path": str(cluster_uncorrp_path) if cluster_uncorrp_path.exists() else None,
        "voxel_corrp_path": str(voxel_corrp_path) if voxel_corrp_path.exists() else None,
        "voxel_uncorrp_path": str(voxel_uncorrp_path) if voxel_uncorrp_path.exists() else None,
        "significant_mask_path": str(sig_mask_path),
        "cluster_labels_6conn_path": str(labels_6_path),
        "cluster_labels_26conn_path": str(labels_26_path),
        "cluster_fwe_p05_masked_tstat_path": str(masked_tstat_path),
        "cluster_fwe_p05_masked_mean_minus_chance_path": str(masked_mean_path),
        "cluster_fwe_p05_masked_corrp_path": str(masked_corrp_path),
        "cluster_table_6conn_csv": str(output_root / "cluster_table_balanced_accuracy_minus_chance_cluster_p001_cluster_fwe_p05_6conn.csv"),
        "cluster_table_26conn_csv": str(output_root / "cluster_table_balanced_accuracy_minus_chance_cluster_p001_cluster_fwe_p05_26conn.csv"),
        "n_significant_voxels_cluster_fwe_p05": int(sig_mask.sum()),
        "n_significant_components_6conn": len(rows_6),
        "n_significant_components_26conn": len(rows_26),
        "largest_component_size_voxels_6conn": int(rows_6[0]["size_voxels"]) if rows_6 else 0,
        "largest_component_size_voxels_26conn": int(rows_26[0]["size_voxels"]) if rows_26 else 0,
        "max_cluster_corrp_1minus_p_in_sigmask": float(np.max(sig_corrp)) if sig_corrp.size else None,
        "min_cluster_fwe_p_in_sigmask": float(1.0 - np.max(sig_corrp)) if sig_corrp.size else None,
        "min_tstat_in_sigmask": float(np.min(sig_t)) if sig_t.size else None,
        "max_tstat_in_sigmask": float(np.max(sig_t)) if sig_t.size else None,
        "max_mean_minus_chance_in_sigmask": float(np.max(sig_mean)) if sig_mean.size else None,
        "max_balanced_accuracy_in_sigmask": float(0.5 + np.max(sig_mean)) if sig_mean.size else None,
        "cluster_extent_critical_value_voxels_from_randomise_stdout": None,
        "cluster_extent_critical_value_voxels_inferred_note": cluster_extent_critical,
        "top_components_26conn_by_size": rows_26[:20],
        "top_components_6conn_by_size": rows_6[:20],
    }


def materialize_workbench_volume_maps(
    workbench_root: Path,
    group_map_paths: dict[str, str],
    cluster_summary: dict[str, Any],
) -> dict[str, Any]:
    volume_dir = ensure_dir(workbench_root / "volume_maps")
    maps = {
        "tstat_uncorrected": copy_map(
            Path(group_map_paths["tstat_map"]),
            volume_dir / "Map_searchlight_rating4-vs-rating1_tstat_uncorrected.nii.gz",
        ),
        "mean_balanced_accuracy_minus_chance_uncorrected": copy_map(
            Path(group_map_paths["mean_map"]),
            volume_dir / "Map_searchlight_rating4-vs-rating1_meanBalancedAccMinusChance_uncorrected.nii.gz",
        ),
        "cluster_sigmask": copy_map(
            Path(cluster_summary["significant_mask_path"]),
            volume_dir / "Map_searchlight_rating4-vs-rating1_clusterP001_clusterFWE05_sigmask.nii.gz",
        ),
        "tstat_cluster_p001_cluster_fwe05": copy_map(
            Path(cluster_summary["cluster_fwe_p05_masked_tstat_path"]),
            volume_dir / "Map_searchlight_rating4-vs-rating1_tstat_clusterP001_clusterFWE05.nii.gz",
        ),
        "mean_balanced_accuracy_minus_chance_cluster_p001_cluster_fwe05": copy_map(
            Path(cluster_summary["cluster_fwe_p05_masked_mean_minus_chance_path"]),
            volume_dir / "Map_searchlight_rating4-vs-rating1_meanBalancedAccMinusChance_clusterP001_clusterFWE05.nii.gz",
        ),
        "cluster_corrp_cluster_p001_cluster_fwe05": copy_map(
            Path(cluster_summary["cluster_fwe_p05_masked_corrp_path"]),
            volume_dir / "Map_searchlight_rating4-vs-rating1_clusterCorrp_clusterP001_clusterFWE05.nii.gz",
        ),
        "randomise_cluster_extent_full": copy_map(
            Path(cluster_summary["cluster_extent_path"]),
            volume_dir / "Map_searchlight_rating4-vs-rating1_randomise_clusterExtent_full.nii.gz",
        )
        if cluster_summary.get("cluster_extent_path")
        else None,
        "randomise_cluster_extent_corrp_full": copy_map(
            Path(cluster_summary["cluster_corrp_path"]),
            volume_dir / "Map_searchlight_rating4-vs-rating1_randomise_clusterExtentCorrp_full.nii.gz",
        ),
        "cluster_labels_6conn": copy_map(
            Path(cluster_summary["cluster_labels_6conn_path"]),
            volume_dir / "Map_searchlight_rating4-vs-rating1_clusterP001_clusterFWE05_clusterLabels_6conn.nii.gz",
        ),
        "cluster_labels_26conn": copy_map(
            Path(cluster_summary["cluster_labels_26conn_path"]),
            volume_dir / "Map_searchlight_rating4-vs-rating1_clusterP001_clusterFWE05_clusterLabels_26conn.nii.gz",
        ),
    }
    manifest = {
        "directory": str(volume_dir),
        "note": "Volume maps are named for later Connectome Workbench display or volume-to-surface mapping.",
        "maps": maps,
    }
    with (volume_dir / "workbench_volume_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def main() -> None:
    args = parse_args()
    source_batch_root = args.source_batch_root.resolve()
    output_root = args.output_root.resolve()
    workbench_root = args.workbench_root.resolve()
    randomise_bin = resolve_randomise_bin(args.randomise_bin)
    excluded_subjects = set(args.exclude_subject)

    if args.overwrite:
        if output_root.exists():
            shutil.rmtree(output_root)
        if workbench_root.exists():
            shutil.rmtree(workbench_root)
    elif output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty. Use --overwrite: {output_root}")

    ensure_dir(output_root)
    ensure_dir(workbench_root)
    subject_map_dir = ensure_dir(output_root / "subject_maps")
    group_input_dir = ensure_dir(output_root / "group_inputs")
    randomise_dir = ensure_dir(output_root / "randomise_cluster_p001")
    source_snapshot_dir = ensure_dir(output_root / "source_snapshot")

    source_summary = load_source_summary(source_batch_root)
    copy_if_exists(source_batch_root / "group_summary.json", source_snapshot_dir / "source_group_summary_n31.json")
    copy_if_exists(source_batch_root / "batch_config.json", source_snapshot_dir / "source_batch_config.json")
    copy_if_exists(source_batch_root / "subject_selection.json", source_snapshot_dir / "source_subject_selection_n31.json")
    copy_if_exists(source_batch_root / "queue_status.json", source_snapshot_dir / "source_queue_status_n31.json")

    center_mask_source = source_batch_root / "center_mask.nii.gz"
    feature_mask_source = source_batch_root / "feature_mask.nii.gz"
    if not center_mask_source.exists():
        center_mask_source = Path(source_summary["center_mask_source"])
    center_mask_path = group_input_dir / "center_mask.nii.gz"
    feature_mask_path = group_input_dir / "feature_mask.nii.gz"
    shutil.copy2(center_mask_source, center_mask_path)
    copy_if_exists(feature_mask_source, feature_mask_path)

    manifest_rows, subject_maps, subject_ids = build_subject_manifest(
        source_summary=source_summary,
        excluded_subjects=excluded_subjects,
        subject_map_dir=subject_map_dir,
    )
    write_csv(output_root / "subject_map_manifest.csv", manifest_rows)

    stack_img = stack_niftis(subject_maps)
    group_4d_path = group_input_dir / "searchlight_balanced_accuracy_minus_chance_n30_4d.nii.gz"
    nib.save(stack_img, str(group_4d_path))

    group_map_paths = compute_group_maps(
        input_4d_path=group_4d_path,
        mask_path=center_mask_path,
        output_dir=group_input_dir,
    )

    df = len(subject_ids) - 1
    cluster_forming_t = float(stats.t.isf(CLUSTER_FORMING_P, df=df))
    randomise_out_root = randomise_dir / "randomise_balanced_accuracy_minus_chance_cluster_p001"
    command: list[str] | None = None
    if not args.skip_randomise:
        command = run_randomise_cluster(
            randomise_bin=randomise_bin,
            input_4d=group_4d_path,
            mask_path=center_mask_path,
            out_root=randomise_out_root,
            permutations=args.permutations,
            seed=args.seed,
            cluster_forming_t=cluster_forming_t,
        )

    cluster_summary: dict[str, Any] | None = None
    if randomise_out_root.with_name(randomise_out_root.name + "_clustere_corrp_tstat1.nii.gz").exists():
        cluster_summary = summarize_cluster_randomise(
            out_root=randomise_out_root,
            mean_map_path=Path(group_map_paths["mean_map"]),
            reference_img=nib.load(str(center_mask_path)),
            cluster_forming_t=cluster_forming_t,
            output_root=output_root,
        )
        workbench_volume_manifest = materialize_workbench_volume_maps(
            workbench_root=workbench_root,
            group_map_paths=group_map_paths,
            cluster_summary=cluster_summary,
        )
    else:
        workbench_volume_manifest = None

    included_rows = [row for row in manifest_rows if int(row["included_in_n30"]) == 1]
    mean_bacc_values = [float(row["mean_searchlight_balanced_accuracy"]) for row in included_rows]
    analysis_summary = {
        "analysis": ANALYSIS_NAME,
        "pipeline": [
            "Reuse completed subject-level rating4 vs rating1 searchlight MVPA maps",
            "Subject-level model: LIBSVM linear SVM, C=1, class_weight=balanced",
            "Cross-validation: leave-one-session-out 3-fold using session labels",
            "Searchlight radius: 6 mm; output metric: balanced accuracy minus chance",
            "Exclude sub-016 before group analysis because session/run coverage is incomplete",
            "Stack the remaining 30 subject-level maps into a 4D group input",
            "Group-level one-sample sign-flipping permutation test against 0",
            "Cluster inference: cluster-forming one-sided p < 0.001, cluster-level FWE p < 0.05",
        ],
        "source_batch_root": str(source_batch_root),
        "output_root": str(output_root),
        "workbench_root": str(workbench_root),
        "n_subjects": len(subject_ids),
        "subject_ids": subject_ids,
        "excluded_subjects": sorted(excluded_subjects),
        "exclusion_note": "sub-016 had incomplete session/run coverage in the source searchlight batch and is excluded from the N=30 group result.",
        "source_batch": {
            "analysis_tag": source_summary.get("analysis_tag"),
            "cv_group": source_summary.get("cv_group"),
            "required_num_cv_groups": source_summary.get("required_num_cv_groups"),
            "radius_mm": source_summary.get("radius_mm"),
            "num_centers": source_summary.get("num_centers"),
            "center_mask_source": source_summary.get("center_mask_source"),
            "feature_mask_source": source_summary.get("feature_mask_source"),
        },
        "group_inputs": {
            "group_4d_path": str(group_4d_path),
            "center_mask_path": str(center_mask_path),
            "feature_mask_path": str(feature_mask_path) if feature_mask_path.exists() else None,
            **group_map_paths,
        },
        "subject_level_summary": {
            "mean_subject_searchlight_balanced_accuracy_mean": float(np.mean(mean_bacc_values)),
            "mean_subject_searchlight_balanced_accuracy_std": float(np.std(mean_bacc_values, ddof=1)),
            "mean_subject_searchlight_balanced_accuracy_minus_chance_mean": float(np.mean(mean_bacc_values) - 0.5),
        },
        "cluster_inference": {
            "cluster_forming_p": CLUSTER_FORMING_P,
            "cluster_forming_test": "one-sided positive t-test against chance",
            "df": df,
            "cluster_forming_t_threshold": cluster_forming_t,
            "cluster_level_fwe_p": CLUSTER_FWE_P,
            "fsl_corrp_significance_rule": "corrp >= 0.95 means cluster-level FWE p <= 0.05",
        },
        "randomise": {
            "permutations": args.permutations,
            "seed": args.seed,
            "randomise_bin": str(randomise_bin),
            "command": command,
        },
        "cluster_fwe": cluster_summary,
        "workbench_volume_manifest": workbench_volume_manifest,
    }
    with (output_root / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False)

    if cluster_summary is not None:
        write_csv(
            output_root / "randomise_cluster_summary.csv",
            [
                {
                    "analysis": ANALYSIS_NAME,
                    "n_subjects": len(subject_ids),
                    "permutations": args.permutations,
                    "seed": args.seed,
                    "cluster_forming_p": CLUSTER_FORMING_P,
                    "cluster_forming_t_threshold": cluster_forming_t,
                    "cluster_level_fwe_p": CLUSTER_FWE_P,
                    **cluster_summary,
                }
            ],
        )


if __name__ == "__main__":
    main()
