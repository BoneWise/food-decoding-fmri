from __future__ import annotations

"""Run a linear-SVM baseline with the exact subject folds used by a CNN run.

The feature preprocessing intentionally mirrors ``FMRIBetaDataset``: each beta
map is z-scored within its subject-specific brain mask, values outside that mask
are set to zero, and a fixed common-mask subset is flattened for the SVM.
"""

import argparse
import csv
import hashlib
import json
import math
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import nibabel as nib
import numpy as np
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC

from .dataset import FMRISample, discover_samples
from .cli import parse_with_config
from .cv import load_subject_folds
from dataclasses import asdict
from .utils import rating_to_target


PREPROCESSING_NAMES = {
    "cnn_zscore": "cnn_samplewise_brainmask_zscore_then_commonmask_flatten",
    "svm_minmax": "svm_samplewise_commonmask_minmax",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Matched cross-subject linear SVM using archived CNN folds."
    )
    parser.add_argument("--beta-index-csv", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--brain-mask-root", required=True)
    parser.add_argument("--path-prefix-from", default="")
    parser.add_argument("--path-prefix-to", default="")
    parser.add_argument("--feature-mask", required=True)
    parser.add_argument("--fold-assignments", required=True)
    parser.add_argument("--cnn-predictions", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument(
        "--preprocessing",
        choices=tuple(PREPROCESSING_NAMES),
        default="cnn_zscore",
        help="cnn_zscore mirrors FMRIBetaDataset; svm_minmax mirrors whole_brain_mvpa.py.",
    )
    parser.add_argument("--class-weight", choices=("balanced", "none"), default="balanced")
    parser.add_argument("--max-iter", type=int, default=50_000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-fold", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-invalid-brain-fraction", type=float, default=0.05)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    return parser


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty table: {target}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_folds(path: str | Path, records: Sequence[FMRISample]) -> list[dict[str, Any]]:
    return [asdict(fold) for fold in load_subject_folds(path, records)]


def load_binary_mask(path: str | Path) -> tuple[np.ndarray, nib.spatialimages.SpatialImage]:
    image = nib.load(str(Path(path).expanduser().resolve()))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D mask at {path}, got {data.shape}.")
    mask = np.asarray(np.isfinite(data) & (data > 0), dtype=bool)
    if not mask.any():
        raise ValueError(f"Mask is empty: {path}")
    return mask, image


def matrix_manifest(
    *,
    args: argparse.Namespace,
    records: Sequence[FMRISample],
    feature_mask: np.ndarray,
) -> dict[str, Any]:
    return {
        "beta_index_csv": str(Path(args.beta_index_csv).expanduser().resolve()),
        "beta_index_sha256": sha256_file(args.beta_index_csv),
        "feature_mask": str(Path(args.feature_mask).expanduser().resolve()),
        "feature_mask_sha256": sha256_file(args.feature_mask),
        "preprocessing": PREPROCESSING_NAMES[args.preprocessing],
        "num_samples": len(records),
        "num_subjects": len({record.subject_id for record in records}),
        "num_features": int(feature_mask.sum()),
        "ratings": {str(rating): sum(record.rating == rating for record in records) for rating in (1, 4)},
        "dtype": "float32",
    }


def build_or_load_feature_matrix(
    *,
    args: argparse.Namespace,
    records: Sequence[FMRISample],
    feature_mask: np.ndarray,
    output_dir: Path,
) -> np.memmap:
    matrix_path = output_dir / f"features_{args.preprocessing}_commonmask.npy"
    manifest_path = output_dir / "feature_manifest.json"
    expected = matrix_manifest(args=args, records=records, feature_mask=feature_mask)

    if matrix_path.exists() and manifest_path.exists() and not args.rebuild_features:
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "beta_index_sha256",
            "feature_mask_sha256",
            "preprocessing",
            "num_samples",
            "num_features",
            "dtype",
        )
        if all(observed.get(key) == expected.get(key) for key in comparable_keys):
            matrix = np.load(matrix_path, mmap_mode="r")
            if tuple(matrix.shape) != (len(records), int(feature_mask.sum())):
                raise ValueError("Cached feature matrix shape does not match its manifest.")
            print(f"Reusing feature matrix: {matrix_path} shape={matrix.shape}", flush=True)
            return matrix
        raise ValueError(
            "Cached feature matrix manifest differs from current inputs. "
            "Use --rebuild-features after confirming the change is intended."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        matrix_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(records), int(feature_mask.sum())),
    )
    mask_cache: dict[str, np.ndarray] = {}
    sample_rows: list[dict[str, Any]] = []
    started = time.time()
    for index, record in enumerate(records):
        mask_key = str(Path(record.mask_path).expanduser().resolve())
        brain_mask = mask_cache.get(mask_key)
        if brain_mask is None:
            brain_mask, _ = load_binary_mask(mask_key)
            if brain_mask.shape != feature_mask.shape:
                raise ValueError(f"Brain-mask shape mismatch for {record.subject_id}: {brain_mask.shape}")
            if np.any(feature_mask & ~brain_mask):
                raise ValueError(
                    f"The common feature mask is not a subset of {record.subject_id}'s brain mask."
                )
            mask_cache[mask_key] = brain_mask

        image = nib.load(str(Path(record.sample_path).expanduser().resolve()))
        volume = np.asarray(image.dataobj, dtype=np.float32)
        if volume.shape != feature_mask.shape:
            raise ValueError(f"Beta-map shape mismatch: {record.sample_path} has {volume.shape}")
        finite_brain = brain_mask & np.isfinite(volume)
        invalid_fraction = 1.0 - float(finite_brain.sum()) / float(brain_mask.sum())
        if invalid_fraction > float(args.max_invalid_brain_fraction):
            raise ValueError(
                f"{record.sample_path} has invalid brain fraction {invalid_fraction:.6f}."
            )
        values = volume[finite_brain]
        mean = float(values.mean())
        std = float(values.std())
        row = np.zeros(int(feature_mask.sum()), dtype=np.float32)
        raw_features = np.asarray(volume[feature_mask], dtype=np.float32)
        finite_features = np.isfinite(raw_features)
        feature_min = math.nan
        feature_max = math.nan
        if args.preprocessing == "cnn_zscore":
            if std > 0.0:
                row[finite_features] = (raw_features[finite_features] - mean) / std
        elif args.preprocessing == "svm_minmax":
            finite_values = raw_features[finite_features]
            feature_min = float(finite_values.min())
            feature_max = float(finite_values.max())
            if feature_max > feature_min:
                row[finite_features] = (finite_values - feature_min) / (feature_max - feature_min)
        else:
            raise ValueError(f"Unsupported preprocessing: {args.preprocessing}")
        matrix[index] = row
        sample_rows.append(
            {
                "row_index": index,
                "subject_id": record.subject_id,
                "session_id": record.session_id,
                "run_id": record.run_id,
                "trial_id": "" if record.trial_id is None else record.trial_id,
                "rating": record.rating,
                "target": rating_to_target(record.rating),
                "sample_path": record.sample_path,
                "mask_path": record.mask_path,
                "brain_mean": mean,
                "brain_std": std,
                "feature_min": feature_min,
                "feature_max": feature_max,
                "invalid_brain_fraction": invalid_fraction,
            }
        )
        if (index + 1) % int(args.log_every) == 0 or index + 1 == len(records):
            elapsed = time.time() - started
            rate = (index + 1) / max(elapsed, 1e-9)
            remaining = (len(records) - index - 1) / max(rate, 1e-9)
            print(
                f"features {index + 1}/{len(records)} | {rate:.2f} maps/s | ETA {remaining / 60:.1f} min",
                flush=True,
            )
    matrix.flush()
    expected["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    expected["matrix_path"] = str(matrix_path.resolve())
    write_json(manifest_path, expected)
    write_csv(output_dir / "sample_index.csv", sample_rows)
    return np.load(matrix_path, mmap_mode="r")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    result: dict[str, Any] = {
        "num_examples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_high": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_high": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_low": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_high": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": cm.astype(int).tolist(),
    }
    result["auc"] = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else math.nan
    return result


def summarize_values(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std_sample": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "sem": float(stats.sem(array)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def one_sample_inference(values: Sequence[float], *, chance: float, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    test = stats.ttest_1samp(array, popmean=chance, alternative="greater")
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, len(array), size=(100_000, len(array)))
    boot_means = array[boot_indices].mean(axis=1)
    centered = array - chance
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(100_000, len(array)))
    null_means = chance + (signs * centered).mean(axis=1)
    observed = float(array.mean())
    return {
        "chance": float(chance),
        "alternative": "greater",
        "t": float(test.statistic),
        "df": int(len(array) - 1),
        "p_t_one_sided": float(test.pvalue),
        "mean_bootstrap_95_ci": [float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))],
        "p_signflip_one_sided": float((1 + np.sum(null_means >= observed)) / (len(null_means) + 1)),
        "num_bootstrap": 100_000,
        "num_signflips": 100_000,
    }


def paired_inference(cnn_values: Sequence[float], svm_values: Sequence[float], *, seed: int) -> dict[str, Any]:
    cnn = np.asarray(cnn_values, dtype=np.float64)
    svm = np.asarray(svm_values, dtype=np.float64)
    differences = cnn - svm
    test_cnn_greater = stats.ttest_rel(cnn, svm, alternative="greater")
    test_svm_greater = stats.ttest_rel(svm, cnn, alternative="greater")
    test_two_sided = stats.ttest_rel(cnn, svm, alternative="two-sided")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(100_000, len(differences)))
    boot = differences[indices].mean(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(100_000, len(differences)))
    null = (signs * differences).mean(axis=1)
    observed = float(differences.mean())
    absolute_observed = abs(observed)
    return {
        "contrast": "cnn_minus_svm",
        "mean_difference": observed,
        "std_difference_sample": float(differences.std(ddof=1)),
        "t_cnn_minus_svm": float(test_two_sided.statistic),
        "df": int(len(differences) - 1),
        "p_t_cnn_greater_one_sided": float(test_cnn_greater.pvalue),
        "p_t_svm_greater_one_sided": float(test_svm_greater.pvalue),
        "p_t_two_sided": float(test_two_sided.pvalue),
        "mean_difference_bootstrap_95_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "p_signflip_cnn_greater_one_sided": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
        "p_signflip_svm_greater_one_sided": float((1 + np.sum(null <= observed)) / (len(null) + 1)),
        "p_signflip_two_sided": float((1 + np.sum(np.abs(null) >= absolute_observed)) / (len(null) + 1)),
        "num_bootstrap": 100_000,
        "num_signflips": 100_000,
    }


def run_fold(
    *,
    args: argparse.Namespace,
    fold: dict[str, Any],
    records: Sequence[FMRISample],
    matrix: np.ndarray,
    output_dir: Path,
) -> None:
    fold_index = int(fold["fold_index"])
    fold_dir = output_dir / f"fold_{fold_index:02d}"
    prediction_path = fold_dir / "predictions.csv"
    summary_path = fold_dir / "summary.json"
    if args.resume and prediction_path.exists() and summary_path.exists():
        print(f"fold {fold_index:02d} already complete; resuming", flush=True)
        return
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_subjects = set(map(str, fold["train_subjects"]))
    val_subjects = set(map(str, fold["val_subjects"]))
    train_indices = np.asarray([i for i, record in enumerate(records) if record.subject_id in train_subjects])
    val_indices = np.asarray([i for i, record in enumerate(records) if record.subject_id in val_subjects])
    y_all = np.asarray([rating_to_target(record.rating) for record in records], dtype=np.int64)
    y_train = y_all[train_indices]
    y_val = y_all[val_indices]
    print(
        f"fold {fold_index:02d} | train n={len(train_indices)} subjects={len(train_subjects)} "
        f"| test n={len(val_indices)} subjects={sorted(val_subjects)}",
        flush=True,
    )
    classifier = LinearSVC(
        C=float(args.svm_c),
        class_weight=None if args.class_weight == "none" else "balanced",
        dual=True,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        random_state=int(args.seed) + fold_index,
    )
    started = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(matrix[train_indices], y_train)
    scores = np.asarray(classifier.decision_function(matrix[val_indices]), dtype=np.float64)
    preds = np.asarray(classifier.predict(matrix[val_indices]), dtype=np.int64)
    metrics = classification_metrics(y_val, preds, scores)
    metrics.update(
        {
            "fold_index": fold_index,
            "train_subjects": sorted(train_subjects),
            "val_subjects": sorted(val_subjects),
            "num_train": int(len(train_indices)),
            "num_val": int(len(val_indices)),
            "train_class_counts": np.bincount(y_train, minlength=2).astype(int).tolist(),
            "val_class_counts": np.bincount(y_val, minlength=2).astype(int).tolist(),
            "fit_seconds": float(time.time() - started),
            "n_iter": int(np.max(np.atleast_1d(classifier.n_iter_))),
            "convergence_warning": any(issubclass(item.category, ConvergenceWarning) for item in caught),
        }
    )
    rows: list[dict[str, Any]] = []
    for local_index, global_index in enumerate(val_indices):
        record = records[int(global_index)]
        rows.append(
            {
                "fold_index": fold_index,
                "row_index": int(global_index),
                "subject_id": record.subject_id,
                "session_id": record.session_id,
                "run_id": record.run_id,
                "trial_id": "" if record.trial_id is None else record.trial_id,
                "sample_path": record.sample_path,
                "rating": record.rating,
                "target": int(y_val[local_index]),
                "pred": int(preds[local_index]),
                "decision_score": float(scores[local_index]),
                "correct": bool(preds[local_index] == y_val[local_index]),
            }
        )
    write_csv(prediction_path, rows)
    write_json(summary_path, metrics)
    print(
        f"fold {fold_index:02d} done | BA={metrics['balanced_accuracy']:.4f} "
        f"ACC={metrics['accuracy']:.4f} AUC={metrics['auc']:.4f} "
        f"time={metrics['fit_seconds'] / 60:.1f} min n_iter={metrics['n_iter']}",
        flush=True,
    )


def summarize_completed_run(
    *,
    args: argparse.Namespace,
    records: Sequence[FMRISample],
    folds: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, str]] = []
    for fold in folds:
        fold_index = int(fold["fold_index"])
        summary_path = output_dir / f"fold_{fold_index:02d}" / "summary.json"
        prediction_path = output_dir / f"fold_{fold_index:02d}" / "predictions.csv"
        if not summary_path.exists() or not prediction_path.exists():
            return
        fold_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        fold_rows.append(
            {
                key: value
                for key, value in fold_summary.items()
                if key not in {"train_subjects", "val_subjects", "confusion_matrix"}
            }
        )
        prediction_rows.extend(read_csv(prediction_path))
    if len(prediction_rows) != len(records):
        raise ValueError(f"Expected {len(records)} OOF predictions, found {len(prediction_rows)}.")
    prediction_rows.sort(key=lambda row: int(row["row_index"]))
    if [int(row["row_index"]) for row in prediction_rows] != list(range(len(records))):
        raise ValueError("OOF prediction row indices do not cover every sample exactly once.")
    write_csv(output_dir / "oof_predictions.csv", prediction_rows)
    write_csv(output_dir / "fold_metrics.csv", fold_rows)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[row["subject_id"]].append(row)
    subject_rows: list[dict[str, Any]] = []
    for subject_id in sorted(grouped):
        rows = grouped[subject_id]
        y = np.asarray([int(row["target"]) for row in rows])
        pred = np.asarray([int(row["pred"]) for row in rows])
        scores = np.asarray([float(row["decision_score"]) for row in rows])
        metrics = classification_metrics(y, pred, scores)
        subject_rows.append(
            {
                "subject_id": subject_id,
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                "tn": metrics["confusion_matrix"][0][0],
                "fp": metrics["confusion_matrix"][0][1],
                "fn": metrics["confusion_matrix"][1][0],
                "tp": metrics["confusion_matrix"][1][1],
            }
        )
    write_csv(output_dir / "subject_metrics.csv", subject_rows)

    y_all = np.asarray([int(row["target"]) for row in prediction_rows])
    pred_all = np.asarray([int(row["pred"]) for row in prediction_rows])
    scores_all = np.asarray([float(row["decision_score"]) for row in prediction_rows])
    fold_summary = {
        metric: summarize_values(float(row[metric]) for row in fold_rows)
        for metric in ("accuracy", "balanced_accuracy", "f1_high", "auc")
    }
    subject_summary = {
        metric: summarize_values(float(row[metric]) for row in subject_rows)
        for metric in ("accuracy", "balanced_accuracy", "f1_high", "auc")
    }
    svm_ba = [float(row["balanced_accuracy"]) for row in subject_rows]
    inference: dict[str, Any] = {
        "svm_subject_balanced_accuracy_vs_chance": one_sample_inference(
            svm_ba, chance=0.5, seed=int(args.seed) + 10_000
        )
    }

    comparison_rows: list[dict[str, Any]] = []
    cnn_summary: dict[str, Any] | None = None
    if args.cnn_predictions:
        cnn_raw = read_csv(args.cnn_predictions)
        cnn_grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in cnn_raw:
            cnn_grouped[row["subject_id"]].append(row)
        cnn_by_subject: dict[str, dict[str, Any]] = {}
        for subject_id, rows in cnn_grouped.items():
            true_key = "true_target" if "true_target" in rows[0] else "target"
            pred_key = "pred_target" if "pred_target" in rows[0] else "pred"
            score_key = "prob_high" if "prob_high" in rows[0] else "decision_score"
            y = np.asarray([int(row[true_key]) for row in rows])
            pred = np.asarray([int(row[pred_key]) for row in rows])
            score = np.asarray([float(row[score_key]) for row in rows])
            cnn_by_subject[subject_id] = classification_metrics(y, pred, score)
        svm_by_subject = {row["subject_id"]: row for row in subject_rows}
        if set(cnn_by_subject) != set(svm_by_subject):
            raise ValueError("CNN and SVM predictions do not contain the same subjects.")
        cnn_ba: list[float] = []
        ordered_svm_ba: list[float] = []
        for subject_id in sorted(svm_by_subject):
            cnn_value = float(cnn_by_subject[subject_id]["balanced_accuracy"])
            svm_value = float(svm_by_subject[subject_id]["balanced_accuracy"])
            cnn_ba.append(cnn_value)
            ordered_svm_ba.append(svm_value)
            comparison_rows.append(
                {
                    "subject_id": subject_id,
                    "cnn_balanced_accuracy": cnn_value,
                    "svm_balanced_accuracy": svm_value,
                    "cnn_minus_svm": cnn_value - svm_value,
                }
            )
        write_csv(output_dir / "cnn_svm_subject_comparison.csv", comparison_rows)
        cnn_summary = {
            "subject_balanced_accuracy": summarize_values(cnn_ba),
            "subject_balanced_accuracy_vs_chance": one_sample_inference(
                cnn_ba, chance=0.5, seed=int(args.seed) + 20_000
            ),
        }
        inference["cnn_minus_svm_subject_balanced_accuracy"] = paired_inference(
            cnn_ba, ordered_svm_ba, seed=int(args.seed) + 30_000
        )

    summary = {
        "analysis": "matched_cross_subject_linear_svm_rating1_vs_rating4",
        "preprocessing": PREPROCESSING_NAMES[args.preprocessing],
        "classifier": {
            "name": "LinearSVC",
            "C": float(args.svm_c),
            "class_weight": None if args.class_weight == "none" else "balanced",
            "dual": True,
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
        },
        "num_samples": len(records),
        "num_subjects": len(subject_rows),
        "num_folds": len(folds),
        "pooled_oof": classification_metrics(y_all, pred_all, scores_all),
        "fold_level": fold_summary,
        "subject_level": subject_summary,
        "inference": inference,
        "cnn_reference": cnn_summary,
    }
    write_json(output_dir / "cv_summary.json", summary)
    print(
        "complete | "
        f"fold BA={fold_summary['balanced_accuracy']['mean']:.4f} +/- "
        f"{fold_summary['balanced_accuracy']['std_sample']:.4f} | "
        f"subject BA={subject_summary['balanced_accuracy']['mean']:.4f} +/- "
        f"{subject_summary['balanced_accuracy']['std_sample']:.4f} | "
        f"p(one-sided)={inference['svm_subject_balanced_accuracy_vs_chance']['p_t_one_sided']:.6g}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_with_config(build_parser(), argv)
    if args.svm_c <= 0 or args.max_iter <= 0 or args.tol <= 0:
        raise ValueError("SVM parameters C, max_iter, and tol must be positive.")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, dataset_summary = discover_samples(
        beta_index_csv=args.beta_index_csv,
        models_root=args.models_root,
        brain_mask_root=args.brain_mask_root,
        path_prefix_from=args.path_prefix_from,
        path_prefix_to=args.path_prefix_to,
    )
    folds = load_folds(args.fold_assignments, records)
    feature_mask, feature_mask_image = load_binary_mask(args.feature_mask)
    if tuple(feature_mask_image.shape) != tuple(nib.load(records[0].sample_path).shape):
        raise ValueError("Feature-mask and beta-map shapes differ.")
    config = vars(args).copy()
    config.update(
        {
            "beta_index_csv": str(Path(args.beta_index_csv).expanduser().resolve()),
            "feature_mask": str(Path(args.feature_mask).expanduser().resolve()),
            "fold_assignments": str(Path(args.fold_assignments).expanduser().resolve()),
            "output_dir": str(output_dir),
            "preprocessing": PREPROCESSING_NAMES[args.preprocessing],
            "dataset_summary": dataset_summary,
            "num_features": int(feature_mask.sum()),
            "fold_assignments_sha256": sha256_file(args.fold_assignments),
        }
    )
    write_json(output_dir / "run_config.json", config)
    matrix = build_or_load_feature_matrix(
        args=args,
        records=records,
        feature_mask=feature_mask,
        output_dir=output_dir,
    )
    selected = folds
    if args.only_fold is not None:
        selected = [fold for fold in folds if int(fold["fold_index"]) == int(args.only_fold)]
        if not selected:
            raise ValueError(f"Fold {args.only_fold} is not present in the assignment file.")
    for fold in selected:
        run_fold(args=args, fold=fold, records=records, matrix=matrix, output_dir=output_dir)
    summarize_completed_run(args=args, records=records, folds=folds, output_dir=output_dir)


if __name__ == "__main__":
    main()
