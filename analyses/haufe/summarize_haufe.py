from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    PROJECT_ROOT,
    fold_dir,
    load_binary_mask,
    load_config,
    map_vector_to_image,
    read_json,
    save_nifti,
    sha256_file,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create descriptive group summaries for completed foldwise Haufe patterns."
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if float(first.std()) == 0.0 or float(second.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def array_quantiles(array: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(array)),
        "p001": float(np.quantile(array, 0.001)),
        "p01": float(np.quantile(array, 0.01)),
        "median": float(np.median(array)),
        "p99": float(np.quantile(array, 0.99)),
        "p999": float(np.quantile(array, 0.999)),
        "maximum": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    config = load_config()
    expected_folds = list(range(1, int(config["input_definition"]["expected_num_folds"]) + 1))
    completed = [index for index in expected_folds if (fold_dir(index) / "fold_complete.json").is_file()]
    if completed != expected_folds and not args.allow_incomplete:
        raise RuntimeError(
            f"Expected completed folds {expected_folds}, found {completed}. "
            "Use --allow-incomplete only for interim inspection."
        )
    if not completed:
        raise RuntimeError("No completed Haufe folds are available to summarize.")

    patterns: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index in completed:
        output = fold_dir(fold_index)
        summary = read_json(output / "fold_summary.json")
        pattern = np.load(output / "haufe_pattern_vector_rating4_minus_rating1.npy")
        if pattern.ndim != 1 or pattern.size != int(config["input_definition"]["expected_num_features"]):
            raise ValueError(f"Fold {fold_index:02d} Haufe vector has unexpected shape {pattern.shape}.")
        if not np.all(np.isfinite(pattern)):
            raise ValueError(f"Fold {fold_index:02d} Haufe vector has non-finite values.")
        patterns.append(np.asarray(pattern, dtype=np.float64))
        fold_rows.append(
            {
                "fold_index": fold_index,
                "num_train": int(summary["num_train"]),
                "num_held_out": int(summary["num_held_out"]),
                "fit_seconds": float(summary["fit_seconds"]),
                "balanced_accuracy": float(summary["reproduction"]["metrics_from_refit_predictions"]["balanced_accuracy"]),
                "auc": float(summary["reproduction"]["metrics_from_refit_predictions"]["auc"]),
                "archived_prediction_label_mismatches": int(
                    summary["reproduction"]["archived_prediction_label_mismatches"]
                ),
                "max_abs_archived_decision_score_difference": float(
                    summary["reproduction"]["max_abs_archived_decision_score_difference"]
                ),
                "decision_variance_sample": float(summary["haufe"]["decision_variance_sample"]),
                "haufe_formula_check_max_abs_error": float(
                    summary["haufe"]["formula_check_max_abs_error"]
                ),
            }
        )
    stacked = np.stack(patterns, axis=0)
    mean_pattern = stacked.mean(axis=0)
    standard_deviation = stacked.std(axis=0, ddof=1) if len(completed) > 1 else np.zeros_like(mean_pattern)
    signed_consistency = np.sign(stacked).mean(axis=0)
    # +1 means all completed folds have positive direction, -1 all negative.

    mask, reference = load_binary_mask(config["source"]["feature_mask"])
    if int(mask.sum()) != mean_pattern.size:
        raise ValueError("Source feature mask does not match Haufe vectors.")
    maps_dir = PROJECT_ROOT / "outputs" / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    vector_paths = {
        "mean": maps_dir / "mean_haufe_pattern_vector_rating4_minus_rating1.npy",
        "standard_deviation": maps_dir / "fold_sd_haufe_pattern_vector.npy",
        "signed_consistency": maps_dir / "fold_signed_consistency_vector.npy",
    }
    np.save(vector_paths["mean"], mean_pattern)
    np.save(vector_paths["standard_deviation"], standard_deviation)
    np.save(vector_paths["signed_consistency"], signed_consistency)
    map_paths = {
        "mean": maps_dir / "mean_haufe_pattern_rating4_minus_rating1.nii.gz",
        "standard_deviation": maps_dir / "fold_sd_haufe_pattern.nii.gz",
        "signed_consistency": maps_dir / "fold_signed_consistency.nii.gz",
    }
    save_nifti(map_paths["mean"], map_vector_to_image(mean_pattern, mask, reference))
    save_nifti(map_paths["standard_deviation"], map_vector_to_image(standard_deviation, mask, reference))
    save_nifti(map_paths["signed_consistency"], map_vector_to_image(signed_consistency, mask, reference))

    similarity_rows: list[dict[str, Any]] = []
    correlations: list[float] = []
    for left in range(len(completed)):
        for right in range(left + 1, len(completed)):
            correlation = safe_correlation(stacked[left], stacked[right])
            correlations.append(correlation)
            similarity_rows.append(
                {
                    "fold_left": completed[left],
                    "fold_right": completed[right],
                    "spatial_pearson_r": correlation,
                }
            )
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    write_csv(tables_dir / "fold_haufe_summary.csv", fold_rows)
    write_csv(tables_dir / "fold_map_similarity.csv", similarity_rows)
    finite_correlations = np.asarray([value for value in correlations if np.isfinite(value)], dtype=np.float64)
    group_summary = {
        "experiment_id": config["experiment_id"],
        "completed_folds": completed,
        "num_features": int(mean_pattern.size),
        "positive_direction": config["haufe_transform"]["positive_direction"],
        "aggregation": config["haufe_transform"]["aggregation"],
        "inference_boundary": config["haufe_transform"]["inference_boundary"],
        "mean_pattern_quantiles": array_quantiles(mean_pattern),
        "fold_standard_deviation_quantiles": array_quantiles(standard_deviation),
        "signed_consistency_quantiles": array_quantiles(signed_consistency),
        "pairwise_fold_spatial_correlation": {
            "num_pairs": int(finite_correlations.size),
            "mean": float(finite_correlations.mean()) if finite_correlations.size else float("nan"),
            "minimum": float(finite_correlations.min()) if finite_correlations.size else float("nan"),
            "maximum": float(finite_correlations.max()) if finite_correlations.size else float("nan"),
        },
        "maps": {
            key: {"path": str(map_paths[key]), "sha256": sha256_file(map_paths[key])}
            for key in map_paths
        },
        "vectors": {
            key: {"path": str(vector_paths[key]), "sha256": sha256_file(vector_paths[key])}
            for key in vector_paths
        },
    }
    write_json(PROJECT_ROOT / "outputs" / "results_summary.json", group_summary)
    print(
        f"Summarized {len(completed)} descriptive Haufe maps; mean fold-map r="
        f"{group_summary['pairwise_fold_spatial_correlation']['mean']:.4f}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
