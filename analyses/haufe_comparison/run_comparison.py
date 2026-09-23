from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing as nib_processing
import numpy as np
from scipy import stats

from .common import PROJECT_ROOT, config, read_json, save_map, sha256_file, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare frozen k=1 Haufe and k=6 Layer-CAM maps descriptively."
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_atlas_labels(path: Path) -> list[tuple[int, str]]:
    root = ET.parse(path).getroot()
    result: list[tuple[int, str]] = []
    for label in root.findall(".//label"):
        name = str(label.text or "").strip()
        if name:
            result.append((int(label.attrib["index"]) + 1, name))
    return result


def world_x_grid(shape: tuple[int, ...], affine: np.ndarray) -> np.ndarray:
    x = np.arange(shape[0], dtype=np.float32)[:, None, None]
    return affine[0, 0] * x + affine[0, 3]


def top_k_mask(signal: np.ndarray, shared_mask: np.ndarray, count: int) -> tuple[np.ndarray, float]:
    flat_shared = np.flatnonzero(shared_mask.ravel())
    values = np.asarray(signal, dtype=np.float64).ravel()[flat_shared]
    if not np.all(np.isfinite(values)):
        raise ValueError("Top-k signal contains non-finite shared-mask values.")
    # Stable sort makes the finite top-k selection deterministic even at ties.
    ranked = np.argsort(-values, kind="mergesort")
    selected = flat_shared[ranked[:count]]
    mask = np.zeros(shared_mask.size, dtype=bool)
    mask[selected] = True
    return mask.reshape(shared_mask.shape), float(values[ranked[count - 1]])


def map_overlap(left: np.ndarray, right: np.ndarray, shared_count: int) -> dict[str, float | int]:
    left_count = int(left.sum())
    right_count = int(right.sum())
    intersection = int((left & right).sum())
    union = int((left | right).sum())
    expected = float(left_count * right_count) / float(shared_count)
    return {
        "left_count": left_count,
        "right_count": right_count,
        "intersection_voxels": intersection,
        "intersection_fraction_shared_mask": float(intersection / shared_count),
        "left_mask_overlap_fraction": float(intersection / left_count),
        "right_mask_overlap_fraction": float(intersection / right_count),
        "dice": float(2 * intersection / (left_count + right_count)),
        "jaccard": float(intersection / union),
        "independent_random_expected_intersection_voxels": expected,
        "observed_to_independent_expected_intersection": float(intersection / expected),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    pearson = stats.pearsonr(left, right).statistic
    spearman = stats.spearmanr(left, right).statistic
    return float(pearson), float(spearman)


def build_roi_masks(
    *, reference: nib.spatialimages.SpatialImage, shared_mask: np.ndarray, atlas_root: Path
) -> list[dict[str, Any]]:
    cortical_image = nib.load(str(atlas_root / "HarvardOxford" / "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"))
    subcortical_image = nib.load(str(atlas_root / "HarvardOxford" / "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz"))
    cortical = np.asarray(
        nib_processing.resample_from_to(cortical_image, reference, order=0).dataobj, dtype=np.int16
    )
    subcortical = np.asarray(
        nib_processing.resample_from_to(subcortical_image, reference, order=0).dataobj, dtype=np.int16
    )
    x_world = world_x_grid(tuple(reference.shape), reference.affine)
    masks: list[dict[str, Any]] = []
    for value, name in parse_atlas_labels(atlas_root / "HarvardOxford-Cortical.xml"):
        base = (cortical == value) & shared_mask
        for side, side_mask in (("Left", base & (x_world < 0)), ("Right", base & (x_world >= 0))):
            if side_mask.any():
                masks.append(
                    {
                        "roi_name": f"{side} {name}",
                        "atlas_group": "cortical",
                        "mask": np.asarray(side_mask, dtype=bool),
                    }
                )
    for value, name in parse_atlas_labels(atlas_root / "HarvardOxford-Subcortical.xml"):
        lowered = name.lower()
        if any(term in lowered for term in ("white matter", "ventricle", "cerebral cortex")):
            continue
        region = (subcortical == value) & shared_mask
        if region.any():
            masks.append({"roi_name": name, "atlas_group": "subcortical", "mask": np.asarray(region, dtype=bool)})
    return masks


def roi_coverage_rows(
    *, masks: list[dict[str, Any]], selected_masks: dict[str, np.ndarray], fraction: float, minimum_size: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    ranked_rows: list[dict[str, Any]] = []
    for condition, selected in selected_masks.items():
        for roi in masks:
            roi_mask = roi["mask"]
            size = int(roi_mask.sum())
            selected_count = int((roi_mask & selected).sum())
            coverage = float(selected_count / size)
            row = {
                "map_condition": condition,
                "roi_name": roi["roi_name"],
                "atlas_group": roi["atlas_group"],
                "roi_voxels_within_shared_mask": size,
                "selected_top20_voxels": selected_count,
                "roi_top20_coverage": coverage,
                "enrichment_vs_global_top20": float(coverage / fraction),
                "eligible_for_ranking": bool(size >= minimum_size),
            }
            all_rows.append(row)
            if size >= minimum_size:
                ranked_rows.append(row)
    return all_rows, ranked_rows


def plot_dice_matrix(rows: list[dict[str, Any]], path: Path) -> None:
    ordered_haufe = ["haufe_rating1", "haufe_rating4"]
    ordered_cam = ["layercam_tfce_t_rating1", "layercam_tfce_t_rating4"]
    matrix = np.zeros((2, 2), dtype=np.float64)
    for row in rows:
        if row["left_map"] in ordered_haufe and row["right_map"] in ordered_cam:
            matrix[ordered_haufe.index(row["left_map"]), ordered_cam.index(row["right_map"])] = row["dice"]
    fig, axis = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(2), ["Layer-CAM t r1", "Layer-CAM t r4"])
    axis.set_yticks(range(2), ["Haufe r1", "Haufe r4"])
    axis.set_title("Top-20% mask Dice overlap\n(descriptive)")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", color="white")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label("Dice coefficient")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    settings = config()
    manifest_path = PROJECT_ROOT / "inputs" / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run python code/prepare_inputs.py before this comparison.")
    manifest = read_json(manifest_path)
    frozen = {key: Path(value["frozen_path"]) for key, value in manifest["sources"].items()}
    reference = nib.load(str(frozen["haufe_mean_map"]))
    haufe = np.asarray(reference.dataobj, dtype=np.float64)
    cam_rating1 = np.asarray(nib.load(str(frozen["layercam_rating1_map"])).dataobj, dtype=np.float64)
    cam_rating4 = np.asarray(nib.load(str(frozen["layercam_rating4_map"])).dataobj, dtype=np.float64)
    cam_rating1_t = np.asarray(
        nib.load(str(frozen["layercam_rating1_tfce_masked_t"])).dataobj, dtype=np.float64
    )
    cam_rating4_t = np.asarray(
        nib.load(str(frozen["layercam_rating4_tfce_masked_t"])).dataobj, dtype=np.float64
    )
    haufe_mask = np.asanyarray(nib.load(str(frozen["haufe_feature_mask"])).dataobj) > 0
    cam_mask = np.asanyarray(nib.load(str(frozen["layercam_common_mask"])).dataobj) > 0
    shared = (
        haufe_mask
        & cam_mask
        & np.isfinite(haufe)
        & np.isfinite(cam_rating1)
        & np.isfinite(cam_rating4)
        & np.isfinite(cam_rating1_t)
        & np.isfinite(cam_rating4_t)
    )
    shared_count = int(shared.sum())
    fraction = float(settings["selection"]["fraction"])
    selected_count = int(math.ceil(fraction * shared_count))
    if shared_count <= 0 or selected_count <= 0:
        raise ValueError("Shared mask is empty or top-20% voxel count is zero.")

    continuous_oriented = {
        "haufe_rating1": -haufe,
        "haufe_rating4": haufe,
        "layercam_groupmean_rating1": cam_rating1,
        "layercam_groupmean_rating4": cam_rating4,
    }
    selection_oriented = {
        "haufe_rating1": -haufe,
        "haufe_rating4": haufe,
        "layercam_tfce_t_rating1": cam_rating1_t,
        "layercam_tfce_t_rating4": cam_rating4_t,
    }
    masks: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    for name, signal in selection_oriented.items():
        masks[name], thresholds[name] = top_k_mask(signal, shared, selected_count)
    outputs = PROJECT_ROOT / "outputs"
    save_map(outputs / "masks" / "shared_mask.nii.gz", shared.astype(np.float32), reference)
    for name, selection in masks.items():
        save_map(outputs / "masks" / f"{name}_top20_mask.nii.gz", selection.astype(np.float32), reference)
    save_map(outputs / "maps" / "haufe_rating4_oriented_pattern.nii.gz", np.where(shared, continuous_oriented["haufe_rating4"], 0.0), reference)
    save_map(outputs / "maps" / "haufe_rating1_oriented_pattern.nii.gz", np.where(shared, continuous_oriented["haufe_rating1"], 0.0), reference)

    continuous_values = {
        name: np.asarray(signal[shared], dtype=np.float64) for name, signal in continuous_oriented.items()
    }
    continuous_rows: list[dict[str, Any]] = []
    top20_rows: list[dict[str, Any]] = []
    for left in ("haufe_rating1", "haufe_rating4"):
        for right in ("layercam_groupmean_rating1", "layercam_groupmean_rating4"):
            pearson, spearman = correlation(continuous_values[left], continuous_values[right])
            continuous_rows.append(
                {
                    "left_map": left,
                    "right_map": right,
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "num_shared_voxels": shared_count,
                    "descriptive_only": True,
                }
            )
    for left in ("haufe_rating1", "haufe_rating4"):
        for right in ("layercam_tfce_t_rating1", "layercam_tfce_t_rating4"):
            top20_rows.append({"left_map": left, "right_map": right, **map_overlap(masks[left], masks[right], shared_count)})
    for left, right in (("haufe_rating1", "haufe_rating4"), ("layercam_groupmean_rating1", "layercam_groupmean_rating4")):
        pearson, spearman = correlation(continuous_values[left], continuous_values[right])
        continuous_rows.append(
            {
                "left_map": left,
                "right_map": right,
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "num_shared_voxels": shared_count,
                "descriptive_only": True,
            }
        )
    for left, right in (("haufe_rating1", "haufe_rating4"), ("layercam_tfce_t_rating1", "layercam_tfce_t_rating4")):
        top20_rows.append({"left_map": left, "right_map": right, **map_overlap(masks[left], masks[right], shared_count)})

    atlas_root = Path(settings["sources"]["harvard_oxford_atlas_root"])
    roi_masks = build_roi_masks(reference=reference, shared_mask=shared, atlas_root=atlas_root)
    all_roi_rows, ranking_rows = roi_coverage_rows(
        masks=roi_masks,
        selected_masks=masks,
        fraction=fraction,
        minimum_size=int(settings["roi_description"]["minimum_roi_voxels_for_ranking"]),
    )
    top_roi_rows: list[dict[str, Any]] = []
    for name in masks:
        rows = [row for row in ranking_rows if row["map_condition"] == name]
        rows.sort(
            key=lambda row: (
                float(row["enrichment_vs_global_top20"]),
                int(row["selected_top20_voxels"]),
            ),
            reverse=True,
        )
        for rank, row in enumerate(rows[:20], start=1):
            top_roi_rows.append({"rank": rank, **row})

    tables = outputs / "tables"
    write_csv(tables / "continuous_map_comparison.csv", continuous_rows)
    write_csv(tables / "top20_overlap_comparison.csv", top20_rows)
    write_csv(tables / "roi_top20_coverage_all.csv", all_roi_rows)
    write_csv(tables / "roi_top20_enrichment_top20.csv", top_roi_rows)
    threshold_rows = [
        {
            "map_condition": name,
            "selected_voxels": int(masks[name].sum()),
            "shared_mask_voxels": shared_count,
            "fraction": float(masks[name].sum() / shared_count),
            "selection_threshold_in_oriented_map": thresholds[name],
        }
        for name in masks
    ]
    write_csv(tables / "top20_selection_thresholds.csv", threshold_rows)
    plot_dice_matrix(top20_rows, outputs / "figures" / "haufe_layercam_top20_dice_matrix.png")

    layercam_summary = read_json(frozen["layercam_analysis_summary"])
    result = {
        "experiment_id": settings["experiment_id"],
        "shared_mask_voxels": shared_count,
        "top20_selected_voxels_per_map": selected_count,
        "selection_fraction": fraction,
        "thresholds": thresholds,
        "layercam_correct_decision_inputs": {
            "rating1": {
                "num_subjects": layercam_summary["classes"]["0"]["num_subjects"],
                "num_correct_trials": layercam_summary["classes"]["0"]["num_correct_trials"],
            },
            "rating4": {
                "num_subjects": layercam_summary["classes"]["1"]["num_subjects"],
                "num_correct_trials": layercam_summary["classes"]["1"]["num_correct_trials"],
            },
        },
        "interpretation_boundary": settings["comparison"]["interpretation_boundary"],
        "files": {
            "continuous_comparison": str(tables / "continuous_map_comparison.csv"),
            "top20_overlap": str(tables / "top20_overlap_comparison.csv"),
            "roi_top20": str(tables / "roi_top20_enrichment_top20.csv"),
            "dice_figure": str(outputs / "figures" / "haufe_layercam_top20_dice_matrix.png"),
        },
        "output_hashes": {
            "continuous_comparison": sha256_file(tables / "continuous_map_comparison.csv"),
            "top20_overlap": sha256_file(tables / "top20_overlap_comparison.csv"),
            "roi_top20": sha256_file(tables / "roi_top20_enrichment_top20.csv"),
        },
    }
    write_json(outputs / "results_summary.json", result)
    print(
        f"Compared four maps on {shared_count} shared voxels; selected {selected_count} top-20% voxels per map.",
        flush=True,
    )


if __name__ == "__main__":
    main()
