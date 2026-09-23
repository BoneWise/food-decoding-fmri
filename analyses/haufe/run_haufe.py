from __future__ import annotations

import argparse
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.svm import LinearSVC

from .common import (
    PROJECT_ROOT,
    fold_dir,
    load_binary_mask,
    load_config,
    map_vector_to_image,
    read_csv,
    read_json,
    save_nifti,
    sha256_array,
    sha256_file,
    source_snapshot_path,
    validate_record_rows,
    write_csv,
    write_json,
)


SCORE_TOLERANCE = 1.0e-10
FORMULA_TOLERANCE = 1.0e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refit frozen matched k=1 SVM folds and calculate training-only Haufe patterns."
    )
    parser.add_argument("--only-fold", type=int, default=None, help="Run exactly one fold (1--10).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a completed named fold. This never changes the source SVM run.",
    )
    return parser.parse_args()


def metric_summary(y_true: np.ndarray, pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, scores)),
    }


def validate_frozen_source(config: dict[str, Any]) -> tuple[
    list[dict[str, str]], dict[str, Any], dict[int, dict[str, str]], np.memmap
]:
    source_manifest = read_json(source_snapshot_path("source_manifest.json"))
    expected = config["input_definition"]
    records = validate_record_rows(
        read_csv(source_snapshot_path("frozen_sample_index.csv")), int(expected["expected_num_samples"])
    )
    folds = read_json(source_snapshot_path("frozen_fold_assignments.json"))
    if len(folds["folds"]) != int(expected["expected_num_folds"]):
        raise ValueError("Frozen fold assignment count does not match configuration.")
    source_predictions = read_csv(source_snapshot_path("frozen_oof_predictions.csv"))
    by_index = {int(row["row_index"]): row for row in source_predictions}
    if len(by_index) != len(records):
        raise ValueError("Frozen OOF predictions do not contain one unique row per sample.")
    matrix_path = Path(source_manifest["source_matrix"]["path"])
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Referenced source feature matrix is missing: {matrix_path}")
    matrix = np.load(matrix_path, mmap_mode="r")
    expected_shape = (int(expected["expected_num_samples"]), int(expected["expected_num_features"]))
    if tuple(matrix.shape) != expected_shape or matrix.dtype != np.dtype("float32"):
        raise ValueError("Referenced source matrix no longer matches frozen dimensions/dtype.")
    return records, folds, by_index, matrix


def fold_indices(records: list[dict[str, str]], fold: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    train_subjects = set(map(str, fold["train_subjects"]))
    held_out_subjects = set(map(str, fold["val_subjects"]))
    all_subjects = {row["subject_id"] for row in records}
    if train_subjects & held_out_subjects or train_subjects | held_out_subjects != all_subjects:
        raise ValueError(f"Fold {fold['fold_index']} does not form a participant-disjoint partition.")
    train = np.asarray(
        [int(row["row_index"]) for row in records if row["subject_id"] in train_subjects], dtype=np.int64
    )
    held_out = np.asarray(
        [int(row["row_index"]) for row in records if row["subject_id"] in held_out_subjects], dtype=np.int64
    )
    if len(np.intersect1d(train, held_out)):
        raise ValueError(f"Fold {fold['fold_index']} has overlapping rows.")
    if len(train) + len(held_out) != len(records):
        raise ValueError(f"Fold {fold['fold_index']} does not cover all frozen records.")
    return train, held_out


def make_classifier(config: dict[str, Any], fold_index: int) -> LinearSVC:
    model = config["frozen_model"]
    return LinearSVC(
        C=float(model["C"]),
        class_weight=str(model["class_weight"]),
        dual=bool(model["dual"]),
        max_iter=int(model["max_iter"]),
        tol=float(model["tol"]),
        random_state=int(model["seed"]) + int(fold_index),
    )


def haufe_pattern(x_train: np.ndarray, classifier: LinearSVC) -> tuple[np.ndarray, dict[str, Any]]:
    """Return Cov(X, s) / Var(s) without materializing a feature covariance matrix."""
    weights = np.asarray(classifier.coef_[0], dtype=np.float64)
    intercept = float(classifier.intercept_[0])
    if not np.array_equal(classifier.classes_, np.asarray([0, 1])):
        raise RuntimeError(f"Unexpected SVM class order: {classifier.classes_!r}.")
    decision_from_classifier = np.asarray(classifier.decision_function(x_train), dtype=np.float64).reshape(-1)
    decision = np.asarray(x_train @ weights + intercept, dtype=np.float64).reshape(-1)
    decision_implementation_error = float(np.max(np.abs(decision - decision_from_classifier)))
    if decision_implementation_error > FORMULA_TOLERANCE:
        raise RuntimeError(
            f"Manual SVM decision calculation differs from decision_function by "
            f"{decision_implementation_error:.3e}."
        )
    centered_decision = decision - float(decision.mean())
    denominator = float(np.dot(centered_decision, centered_decision))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Training-fold SVM decision variance is non-positive.")

    # Because centered_decision sums to zero, X.T @ centered_decision equals
    # (X - mean(X)).T @ centered_decision. The common covariance denominator
    # cancels against Var(decision), so this is exactly the stated Haufe ratio.
    numerator = np.zeros(x_train.shape[1], dtype=np.float64)
    block_rows = 256
    for start in range(0, x_train.shape[0], block_rows):
        stop = min(start + block_rows, x_train.shape[0])
        numerator += np.asarray(
            x_train[start:stop].T @ centered_decision[start:stop], dtype=np.float64
        ).reshape(-1)
    pattern = numerator / denominator
    if not np.all(np.isfinite(pattern)):
        raise ValueError("Haufe pattern contains non-finite values.")

    # A small deterministic direct calculation catches centering/order mistakes
    # without allocating a p-by-p covariance matrix.
    check_indices = np.unique(
        np.linspace(0, pattern.size - 1, num=min(31, pattern.size), dtype=np.int64)
    )
    selected = np.asarray(x_train[:, check_indices], dtype=np.float64)
    selected_centered = selected - selected.mean(axis=0, keepdims=True)
    direct = (selected_centered.T @ centered_decision) / denominator
    formula_max_abs_error = float(np.max(np.abs(direct - pattern[check_indices])))
    if formula_max_abs_error > FORMULA_TOLERANCE:
        raise RuntimeError(
            f"Haufe direct-formula check failed: max error {formula_max_abs_error:.3e}."
        )
    metadata = {
        "formula": "Cov(X_train, decision_score) / Var(decision_score)",
        "num_training_samples": int(x_train.shape[0]),
        "num_features": int(x_train.shape[1]),
        "decision_mean": float(decision.mean()),
        "decision_variance_sample": float(np.var(decision, ddof=1)),
        "decision_ss_centered": denominator,
        "decision_function_max_abs_error": decision_implementation_error,
        "accumulation_block_rows": block_rows,
        "formula_check_num_features": int(check_indices.size),
        "formula_check_max_abs_error": formula_max_abs_error,
        "explicit_full_covariance_constructed": False,
    }
    return pattern, metadata


def reproduce_against_archived(
    *,
    fold_index: int,
    held_out_indices: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    source_predictions: dict[int, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_pred = []
    source_scores = []
    expected_targets = []
    for local_index, row_index in enumerate(held_out_indices):
        archived = source_predictions[int(row_index)]
        if int(archived["fold_index"]) != fold_index:
            raise RuntimeError(
                f"Archived row {row_index} is assigned to fold {archived['fold_index']}, not {fold_index}."
            )
        observed_pred = int(predictions[local_index])
        archived_pred = int(archived["pred"])
        observed_score = float(scores[local_index])
        archived_score = float(archived["decision_score"])
        source_pred.append(archived_pred)
        source_scores.append(archived_score)
        expected_targets.append(int(archived["target"]))
        rows.append(
            {
                "fold_index": fold_index,
                "row_index": int(row_index),
                "subject_id": archived["subject_id"],
                "session_id": archived["session_id"],
                "run_id": archived["run_id"],
                "rating": int(archived["rating"]),
                "target": int(archived["target"]),
                "archived_pred": archived_pred,
                "refit_pred": observed_pred,
                "archived_decision_score": archived_score,
                "refit_decision_score": observed_score,
                "decision_score_delta": observed_score - archived_score,
                "prediction_match": bool(observed_pred == archived_pred),
            }
        )
    source_pred_array = np.asarray(source_pred, dtype=np.int64)
    source_scores_array = np.asarray(source_scores, dtype=np.float64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    label_mismatches = int(np.sum(predictions_array != source_pred_array))
    max_abs_score_difference = float(np.max(np.abs(scores_array - source_scores_array)))
    if label_mismatches != 0 or max_abs_score_difference > SCORE_TOLERANCE:
        raise RuntimeError(
            f"Fold {fold_index:02d} does not reproduce archived SVM predictions: "
            f"label_mismatches={label_mismatches}, "
            f"max_abs_score_difference={max_abs_score_difference:.3e}."
        )
    metrics = metric_summary(np.asarray(expected_targets), predictions_array, scores_array)
    reproduction = {
        "archived_prediction_label_mismatches": label_mismatches,
        "max_abs_archived_decision_score_difference": max_abs_score_difference,
        "score_tolerance": SCORE_TOLERANCE,
        "passed": True,
        "metrics_from_refit_predictions": metrics,
    }
    return rows, reproduction


def run_fold(
    *,
    config: dict[str, Any],
    records: list[dict[str, str]],
    fold: dict[str, Any],
    source_predictions: dict[int, dict[str, str]],
    matrix: np.memmap,
    mask: np.ndarray,
    mask_image: Any,
    force: bool,
) -> None:
    fold_index = int(fold["fold_index"])
    output = fold_dir(fold_index)
    complete_path = output / "fold_complete.json"
    if complete_path.exists() and not force:
        print(f"fold {fold_index:02d} already complete; resuming", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    train_indices, held_out_indices = fold_indices(records, fold)
    targets = np.asarray([int(row["target"]) for row in records], dtype=np.int64)
    y_train = targets[train_indices]
    print(
        f"fold {fold_index:02d} | train n={len(train_indices)} subjects={len(fold['train_subjects'])} "
        f"| held-out n={len(held_out_indices)} subjects={','.join(fold['val_subjects'])}",
        flush=True,
    )

    # This mirrors the archived refit call: the selected rows are a temporary
    # float32 training matrix, while the frozen source matrix remains read-only.
    x_train = np.asarray(matrix[train_indices], dtype=np.float32)
    classifier = make_classifier(config, fold_index)
    started = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(x_train, y_train)
    fit_seconds = time.time() - started

    pattern, pattern_metadata = haufe_pattern(x_train, classifier)
    weights = np.asarray(classifier.coef_[0], dtype=np.float64)
    del x_train

    x_held_out = np.asarray(matrix[held_out_indices], dtype=np.float32)
    scores = np.asarray(classifier.decision_function(x_held_out), dtype=np.float64)
    predictions = np.asarray(classifier.predict(x_held_out), dtype=np.int64)
    del x_held_out
    prediction_rows, reproduction = reproduce_against_archived(
        fold_index=fold_index,
        held_out_indices=held_out_indices,
        predictions=predictions,
        scores=scores,
        source_predictions=source_predictions,
    )

    vector_path = output / "haufe_pattern_vector_rating4_minus_rating1.npy"
    weight_path = output / "svm_weight_vector.npy"
    map_path = output / "haufe_pattern_rating4_minus_rating1.nii.gz"
    prediction_path = output / "refit_predictions.csv"
    np.save(vector_path, pattern.astype(np.float64, copy=False))
    np.save(weight_path, weights)
    save_nifti(map_path, map_vector_to_image(pattern, mask, mask_image))
    write_csv(prediction_path, prediction_rows)
    summary = {
        "experiment_id": config["experiment_id"],
        "fold_index": fold_index,
        "training_subjects": list(fold["train_subjects"]),
        "held_out_subjects": list(fold["val_subjects"]),
        "num_train": int(len(train_indices)),
        "num_held_out": int(len(held_out_indices)),
        "train_class_counts": np.bincount(y_train, minlength=2).astype(int).tolist(),
        "classifier": config["frozen_model"],
        "fit_seconds": float(fit_seconds),
        "n_iter": int(np.max(np.atleast_1d(classifier.n_iter_))),
        "convergence_warning": any(issubclass(item.category, ConvergenceWarning) for item in caught),
        "reproduction": reproduction,
        "haufe": pattern_metadata,
        "direction": "positive values map to the rating-4 LinearSVC decision direction",
    }
    write_json(output / "fold_summary.json", summary)
    complete = {
        "status": "complete",
        "fold_index": fold_index,
        "files": {
            "fold_summary": {"path": str(output / "fold_summary.json"), "sha256": sha256_file(output / "fold_summary.json")},
            "predictions": {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
            "weights": {"path": str(weight_path), "sha256": sha256_file(weight_path)},
            "haufe_vector": {"path": str(vector_path), "sha256": sha256_file(vector_path)},
            "haufe_nifti": {"path": str(map_path), "sha256": sha256_file(map_path)},
        },
        "array_hashes": {
            "weights": sha256_array(weights),
            "haufe_pattern": sha256_array(pattern),
        },
        "project_run_script_sha256": sha256_file(Path(__file__)),
    }
    write_json(complete_path, complete)
    print(
        f"fold {fold_index:02d} complete | fit={fit_seconds / 60.0:.1f} min | "
        f"BA={reproduction['metrics_from_refit_predictions']['balanced_accuracy']:.4f} | "
        f"Haufe formula error={pattern_metadata['formula_check_max_abs_error']:.2e}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    config = load_config()
    if not source_snapshot_path("source_manifest.json").is_file():
        raise FileNotFoundError("Run python code/prepare_inputs.py before running Haufe transformation.")
    records, folds, source_predictions, matrix = validate_frozen_source(config)
    mask, mask_image = load_binary_mask(config["source"]["feature_mask"])
    if int(mask.sum()) != matrix.shape[1]:
        raise ValueError("Feature mask and frozen matrix do not have the same feature count.")
    selected = list(folds["folds"])
    if args.only_fold is not None:
        selected = [fold for fold in selected if int(fold["fold_index"]) == int(args.only_fold)]
        if not selected:
            raise ValueError(f"Unknown fold {args.only_fold}.")
    for fold in selected:
        run_fold(
            config=config,
            records=records,
            fold=fold,
            source_predictions=source_predictions,
            matrix=matrix,
            mask=mask,
            mask_image=mask_image,
            force=args.force,
        )
    completed = sorted(
        int(path.parent.name.split("_")[-1])
        for path in (PROJECT_ROOT / "outputs" / "folds").glob("fold_*/fold_complete.json")
    )
    write_json(
        PROJECT_ROOT / "outputs" / "pipeline_status.json",
        {
            "completed_folds": completed,
            "expected_folds": [int(fold["fold_index"]) for fold in folds["folds"]],
            "all_expected_folds_present": completed
            == [int(fold["fold_index"]) for fold in folds["folds"]],
            "run_script_sha256": sha256_file(Path(__file__)),
        },
    )


if __name__ == "__main__":
    main()
