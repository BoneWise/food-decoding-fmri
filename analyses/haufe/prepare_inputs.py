from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import (
    PROJECT_ROOT,
    load_binary_mask,
    load_config,
    read_csv,
    read_json,
    sha256_file,
    source_snapshot_path,
    validate_record_rows,
    write_json,
)


SNAPSHOTS = {
    "run_config": "frozen_run_config.json",
    "feature_manifest": "frozen_feature_manifest.json",
    "cv_summary": "frozen_cv_summary.json",
    "sample_index": "frozen_sample_index.csv",
    "oof_predictions": "frozen_oof_predictions.csv",
    "fold_assignments": "frozen_fold_assignments.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze and validate source artifacts for the k=1 SVM Haufe project."
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace existing frozen lightweight source snapshots."
    )
    return parser.parse_args()


def copy_snapshot(source: Path, target: Path, force: bool) -> None:
    if target.exists() and not force:
        if sha256_file(source) != sha256_file(target):
            raise RuntimeError(
                f"Existing frozen snapshot differs from source: {target}. Use --force only after review."
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def validate_folds(folds: dict, record_rows: list[dict[str, str]], expected_folds: int) -> None:
    if int(folds.get("num_folds", -1)) != expected_folds:
        raise ValueError("Unexpected number of folds in frozen fold assignments.")
    subjects = {row["subject_id"] for row in record_rows}
    held_out_counts = {subject: 0 for subject in subjects}
    seen_indices: set[int] = set()
    for fold in folds["folds"]:
        train = set(map(str, fold["train_subjects"]))
        held_out = set(map(str, fold["val_subjects"]))
        if train & held_out or train | held_out != subjects:
            raise ValueError(f"Invalid participant split in fold {fold['fold_index']}.")
        for subject in held_out:
            held_out_counts[subject] += 1
        for row in record_rows:
            if row["subject_id"] in held_out:
                index = int(row["row_index"])
                if index in seen_indices:
                    raise ValueError(f"Sample {index} is held out in more than one fold.")
                seen_indices.add(index)
    if any(count != 1 for count in held_out_counts.values()):
        raise ValueError("Each subject must occur in exactly one held-out fold.")
    if len(seen_indices) != len(record_rows):
        raise ValueError("Held-out folds do not cover every sample exactly once.")


def validate_oof_predictions(rows: list[dict[str, str]], record_rows: list[dict[str, str]]) -> None:
    if len(rows) != len(record_rows):
        raise ValueError("Frozen OOF predictions do not have one row per source sample.")
    by_index = {int(row["row_index"]): row for row in rows}
    if len(by_index) != len(record_rows):
        raise ValueError("OOF predictions contain duplicate row indices.")
    for source in record_rows:
        prediction = by_index.get(int(source["row_index"]))
        if prediction is None:
            raise ValueError(f"Missing OOF prediction for row {source['row_index']}.")
        for field in ("subject_id", "session_id", "run_id", "rating", "target"):
            if str(prediction[field]) != str(source[field]):
                raise ValueError(
                    f"OOF/source mismatch for row {source['row_index']} field {field}: "
                    f"{prediction[field]!r} != {source[field]!r}."
                )


def main() -> None:
    args = parse_args()
    config = load_config()
    source = {key: Path(value).expanduser().resolve() for key, value in config["source"].items()}
    required = [
        "run_config",
        "feature_manifest",
        "cv_summary",
        "sample_index",
        "oof_predictions",
        "feature_matrix",
        "feature_mask",
        "fold_assignments",
        "original_svm_script",
    ]
    missing = [key for key in required if not source[key].is_file()]
    if missing:
        raise FileNotFoundError(f"Required source artifacts are missing: {missing}")

    for key, snapshot_name in SNAPSHOTS.items():
        copy_snapshot(source[key], source_snapshot_path(snapshot_name), args.force)

    run_config = read_json(source_snapshot_path(SNAPSHOTS["run_config"]))
    feature_manifest = read_json(source_snapshot_path(SNAPSHOTS["feature_manifest"]))
    cv_summary = read_json(source_snapshot_path(SNAPSHOTS["cv_summary"]))
    record_rows = validate_record_rows(
        read_csv(source_snapshot_path(SNAPSHOTS["sample_index"])),
        int(config["input_definition"]["expected_num_samples"]),
    )
    folds = read_json(source_snapshot_path(SNAPSHOTS["fold_assignments"]))
    validate_folds(folds, record_rows, int(config["input_definition"]["expected_num_folds"]))
    validate_oof_predictions(read_csv(source_snapshot_path(SNAPSHOTS["oof_predictions"])), record_rows)

    matrix = np.load(source["feature_matrix"], mmap_mode="r")
    expected_shape = (
        int(config["input_definition"]["expected_num_samples"]),
        int(config["input_definition"]["expected_num_features"]),
    )
    if tuple(matrix.shape) != expected_shape or matrix.dtype != np.dtype("float32"):
        raise ValueError(
            f"Unexpected source feature matrix: shape={matrix.shape}, dtype={matrix.dtype}; "
            f"expected={expected_shape}, float32."
        )
    mask, _ = load_binary_mask(source["feature_mask"])
    if int(mask.sum()) != expected_shape[1]:
        raise ValueError("Feature-mask voxel count does not match source matrix width.")
    if int(feature_manifest["num_features"]) != expected_shape[1]:
        raise ValueError("Feature manifest does not match source matrix width.")
    if int(cv_summary["num_samples"]) != expected_shape[0]:
        raise ValueError("CV summary does not match source matrix length.")

    print("Hashing the referenced source feature matrix once for provenance...", flush=True)
    artifacts = {}
    for key, path in source.items():
        if path.is_file():
            artifacts[key] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
    manifest = {
        "experiment_id": config["experiment_id"],
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": artifacts,
        "source_matrix": {
            "path": str(source["feature_matrix"]),
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "feature_mask_voxels": int(mask.sum()),
        },
        "frozen_snapshots": {
            key: {
                "path": str(source_snapshot_path(snapshot_name)),
                "sha256": sha256_file(source_snapshot_path(snapshot_name)),
            }
            for key, snapshot_name in SNAPSHOTS.items()
        },
    }
    write_json(source_snapshot_path("source_manifest.json"), manifest)
    write_json(
        source_snapshot_path("preparation_report.json"),
        {
            "status": "passed",
            "num_samples": len(record_rows),
            "num_subjects": len({row["subject_id"] for row in record_rows}),
            "num_folds": len(folds["folds"]),
            "num_features": int(mask.sum()),
            "matrix_shape": list(matrix.shape),
            "matrix_dtype": str(matrix.dtype),
            "working_directory": str(PROJECT_ROOT),
        },
    )
    print(
        f"Prepared {len(record_rows)} samples, {len(folds['folds'])} folds, "
        f"and {int(mask.sum())} frozen features.",
        flush=True,
    )


if __name__ == "__main__":
    main()
