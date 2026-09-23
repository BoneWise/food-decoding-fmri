from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .common import (
    PROJECT_ROOT,
    fold_dir,
    load_binary_mask,
    load_config,
    read_csv,
    read_json,
    sha256_file,
    source_snapshot_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Haufe project provenance and outputs.")
    parser.add_argument(
        "--raw-only", action="store_true", help="Validate frozen source and completed fold files only."
    )
    parser.add_argument(
        "--full-source-hash",
        action="store_true",
        help="Also re-hash the 5.8-GiB referenced source feature matrix.",
    )
    return parser.parse_args()


def check(condition: bool, label: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"check": label, "passed": bool(condition)})
    if not condition:
        raise RuntimeError(f"Validation failed: {label}")


def validate_source(config: dict[str, Any], *, full_source_hash: bool, checks: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_path = source_snapshot_path("source_manifest.json")
    check(manifest_path.is_file(), "source manifest exists", checks)
    manifest = read_json(manifest_path)
    for key in (
        "frozen_run_config.json",
        "frozen_feature_manifest.json",
        "frozen_cv_summary.json",
        "frozen_sample_index.csv",
        "frozen_oof_predictions.csv",
        "frozen_fold_assignments.json",
    ):
        snapshot = source_snapshot_path(key)
        check(snapshot.is_file(), f"frozen snapshot exists: {key}", checks)
    for key, expected in manifest["frozen_snapshots"].items():
        actual = sha256_file(expected["path"])
        check(actual == expected["sha256"], f"frozen snapshot hash: {key}", checks)

    source_matrix = Path(manifest["source_matrix"]["path"])
    check(source_matrix.is_file(), "referenced source matrix exists", checks)
    matrix = np.load(source_matrix, mmap_mode="r")
    expected_shape = (
        int(config["input_definition"]["expected_num_samples"]),
        int(config["input_definition"]["expected_num_features"]),
    )
    check(tuple(matrix.shape) == expected_shape, "source matrix dimensions", checks)
    check(matrix.dtype == np.dtype("float32"), "source matrix dtype", checks)
    if full_source_hash:
        print("Re-hashing referenced source feature matrix for final provenance validation...", flush=True)
        expected_hash = manifest["source_artifacts"]["feature_matrix"]["sha256"]
        check(sha256_file(source_matrix) == expected_hash, "source matrix SHA-256", checks)
    return manifest


def validate_fold(
    *,
    fold_index: int,
    feature_count: int,
    mask: np.ndarray,
    checks: list[dict[str, Any]],
) -> np.ndarray:
    output = fold_dir(fold_index)
    complete_path = output / "fold_complete.json"
    check(complete_path.is_file(), f"fold {fold_index:02d} completion marker", checks)
    complete = read_json(complete_path)
    check(complete.get("status") == "complete", f"fold {fold_index:02d} completed status", checks)
    files = complete["files"]
    for name, descriptor in files.items():
        path = Path(descriptor["path"])
        check(path.is_file(), f"fold {fold_index:02d} file exists: {name}", checks)
        check(sha256_file(path) == descriptor["sha256"], f"fold {fold_index:02d} file hash: {name}", checks)
    summary = read_json(output / "fold_summary.json")
    check(summary["reproduction"]["passed"] is True, f"fold {fold_index:02d} archived prediction reproduction", checks)
    check(
        int(summary["reproduction"]["archived_prediction_label_mismatches"]) == 0,
        f"fold {fold_index:02d} zero prediction mismatches",
        checks,
    )
    check(
        float(summary["reproduction"]["max_abs_archived_decision_score_difference"])
        <= float(summary["reproduction"]["score_tolerance"]),
        f"fold {fold_index:02d} score reproduction tolerance",
        checks,
    )
    check(
        float(summary["haufe"]["formula_check_max_abs_error"]) <= 1.0e-10,
        f"fold {fold_index:02d} Haufe direct-formula check",
        checks,
    )
    check(
        float(summary["haufe"]["decision_function_max_abs_error"]) <= 1.0e-10,
        f"fold {fold_index:02d} manual SVM-score check",
        checks,
    )
    vector = np.load(output / "haufe_pattern_vector_rating4_minus_rating1.npy")
    weights = np.load(output / "svm_weight_vector.npy")
    check(vector.shape == (feature_count,), f"fold {fold_index:02d} Haufe vector shape", checks)
    check(weights.shape == (feature_count,), f"fold {fold_index:02d} SVM weight shape", checks)
    check(np.all(np.isfinite(vector)), f"fold {fold_index:02d} finite Haufe vector", checks)
    check(np.all(np.isfinite(weights)), f"fold {fold_index:02d} finite SVM weights", checks)
    image = nib.load(str(output / "haufe_pattern_rating4_minus_rating1.nii.gz"))
    image_data = np.asanyarray(image.dataobj)
    check(image_data.shape == mask.shape, f"fold {fold_index:02d} NIfTI shape", checks)
    check(np.array_equal(image_data[mask], vector.astype(np.float32)), f"fold {fold_index:02d} NIfTI vector mapping", checks)
    check(np.all(image_data[~mask] == 0.0), f"fold {fold_index:02d} NIfTI outside-mask zeros", checks)
    prediction_rows = read_csv(output / "refit_predictions.csv")
    check(
        all(row["prediction_match"] == "True" for row in prediction_rows),
        f"fold {fold_index:02d} rowwise prediction matches",
        checks,
    )
    return np.asarray(vector, dtype=np.float64)


def validate_group(
    *,
    config: dict[str, Any],
    fold_vectors: list[np.ndarray],
    mask: np.ndarray,
    checks: list[dict[str, Any]],
) -> None:
    maps = PROJECT_ROOT / "outputs" / "maps"
    summary_path = PROJECT_ROOT / "outputs" / "results_summary.json"
    check(summary_path.is_file(), "group results summary exists", checks)
    group = read_json(summary_path)
    expected = np.stack(fold_vectors).mean(axis=0)
    observed = np.load(maps / "mean_haufe_pattern_vector_rating4_minus_rating1.npy")
    check(np.array_equal(observed, expected), "group mean vector equals foldwise mean", checks)
    for name, filename, vector in (
        ("mean", "mean_haufe_pattern_rating4_minus_rating1.nii.gz", observed),
        ("standard deviation", "fold_sd_haufe_pattern.nii.gz", np.load(maps / "fold_sd_haufe_pattern_vector.npy")),
        ("signed consistency", "fold_signed_consistency.nii.gz", np.load(maps / "fold_signed_consistency_vector.npy")),
    ):
        path = maps / filename
        check(path.is_file(), f"group {name} NIfTI exists", checks)
        image_data = np.asanyarray(nib.load(str(path)).dataobj)
        check(image_data.shape == mask.shape, f"group {name} NIfTI shape", checks)
        check(np.array_equal(image_data[mask], vector.astype(np.float32)), f"group {name} NIfTI vector mapping", checks)
        check(np.all(image_data[~mask] == 0.0), f"group {name} NIfTI outside-mask zeros", checks)
    check(
        group["inference_boundary"] == config["haufe_transform"]["inference_boundary"],
        "group summary preserves inferential boundary",
        checks,
    )


def main() -> None:
    args = parse_args()
    config = load_config()
    checks: list[dict[str, Any]] = []
    validate_source(config, full_source_hash=args.full_source_hash, checks=checks)
    mask, _ = load_binary_mask(config["source"]["feature_mask"])
    feature_count = int(config["input_definition"]["expected_num_features"])
    check(int(mask.sum()) == feature_count, "feature mask voxel count", checks)
    expected_folds = list(range(1, int(config["input_definition"]["expected_num_folds"]) + 1))
    fold_vectors = [
        validate_fold(fold_index=index, feature_count=feature_count, mask=mask, checks=checks)
        for index in expected_folds
    ]
    if not args.raw_only:
        validate_group(config=config, fold_vectors=fold_vectors, mask=mask, checks=checks)
    report = {
        "status": "passed",
        "raw_only": bool(args.raw_only),
        "full_source_hash": bool(args.full_source_hash),
        "num_checks": len(checks),
        "checks": checks,
    }
    target = PROJECT_ROOT / "outputs" / "validation" / (
        "raw_validation_report.json" if args.raw_only else "validation_report.json"
    )
    write_json(target, report)
    print(f"Validation passed: {len(checks)}/{len(checks)} checks.", flush=True)


if __name__ == "__main__":
    main()
