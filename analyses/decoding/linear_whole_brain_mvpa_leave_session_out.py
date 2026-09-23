from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from analyses._paths import DATA_ROOT, RUNS_ROOT, OUTPUT_ROOT, FSL_ROOT, ATLAS_ROOT
from typing import Any, Dict, List, Sequence


import nibabel as nib
import numpy as np
import torch
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC, SVC

from foodpref.dataset import FMRISample, discover_samples
from analyses.searchlight.paper_searchlight_mvpa import (
    DEFAULT_SESSIONWISE_BETA_INDEX_CSV,
    DEFAULT_SESSIONWISE_BRAIN_MASK_ROOT,
    DEFAULT_SESSIONWISE_MODELS_ROOT,
    WholeBrainFeatureExtractor,
    _infer_default_mask,
    build_subject_design_matrix,
    group_records_by_subject,
    load_mask,
    prepare_fold_features,
)
from foodpref.utils import compute_binary_classification_metrics, ensure_dir, mean_and_std, save_json, set_seed


DEFAULT_RUN_NAME = "rating14_wholebrain_linear_svm_leave_session_out"
DEFAULT_SAVE_ROOT = str(OUTPUT_ROOT / "within_subject")
DEFAULT_REQUIRED_NUM_SESSIONS = 3
DEFAULT_SVM_BACKEND = "libsvm_linear"
DEFAULT_CLASS_WEIGHT = "balanced"
DEFAULT_SVM_C = 1.0
DEFAULT_MAX_ITER = 10000
DEFAULT_LIBSVM_CACHE_SIZE_MB = 1024.0
DEFAULT_LOG_EVERY = 250


@dataclass(frozen=True)
class LeaveSessionOutConfig:
    beta_index_csv: str
    models_root: str | None
    brain_mask_root: str | None
    path_prefix_from: str
    path_prefix_to: str
    save_root: str
    run_name: str | None
    feature_mask: str | None
    feature_mask_strategy: str
    only_subjects: List[str]
    exclude_subjects: List[str]
    required_num_sessions: int
    seed: int
    svm_c: float
    svm_backend: str
    class_weight: str
    fold_standardize: bool
    libsvm_cache_size_mb: float
    max_iter: int
    log_every: int


def _normalize_optional_path(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_subject_list(value: str) -> List[str]:
    parts = [item.strip() for item in str(value).split(",")]
    return [item for item in parts if item]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run whole-brain linear MVPA within each subject using the same "
            "3-fold leave-one-session-out design as the searchlight analysis."
        )
    )
    parser.add_argument("--beta-index-csv", default=DEFAULT_SESSIONWISE_BETA_INDEX_CSV)
    parser.add_argument("--models-root", default=DEFAULT_SESSIONWISE_MODELS_ROOT)
    parser.add_argument("--brain-mask-root", default=DEFAULT_SESSIONWISE_BRAIN_MASK_ROOT)
    parser.add_argument("--path-prefix-from", default="")
    parser.add_argument("--path-prefix-to", default="")
    parser.add_argument("--save-root", default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--feature-mask",
        default=None,
        help=(
            "Fixed whole-brain feature mask. If omitted, the script first tries the "
            "same default group masks as searchlight, then falls back to an "
            "intersection of included subject masks."
        ),
    )
    parser.add_argument("--exclude-subjects", default="")
    parser.add_argument("--only-subjects", default="")
    parser.add_argument("--required-num-sessions", type=int, default=DEFAULT_REQUIRED_NUM_SESSIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svm-c", type=float, default=DEFAULT_SVM_C)
    parser.add_argument(
        "--svm-backend",
        choices=("libsvm_linear", "linear_svc"),
        default=DEFAULT_SVM_BACKEND,
    )
    parser.add_argument(
        "--class-weight",
        choices=("balanced", "none"),
        default=DEFAULT_CLASS_WEIGHT,
        help="Matches searchlight default: balanced class weights for imbalanced session folds.",
    )
    parser.add_argument(
        "--no-fold-standardize",
        action="store_true",
        help="Disable train-fold-only z-scoring. By default this matches searchlight.",
    )
    parser.add_argument("--libsvm-cache-size-mb", type=float, default=DEFAULT_LIBSVM_CACHE_SIZE_MB)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    return parser


def parse_args(argv: list[str] | None = None) -> LeaveSessionOutConfig:
    args = build_parser().parse_args(argv)
    if int(args.required_num_sessions) <= 1:
        raise ValueError(f"--required-num-sessions must be > 1, got {args.required_num_sessions}.")
    if float(args.svm_c) <= 0.0:
        raise ValueError(f"--svm-c must be > 0, got {args.svm_c}.")
    if float(args.libsvm_cache_size_mb) <= 0.0:
        raise ValueError(f"--libsvm-cache-size-mb must be > 0, got {args.libsvm_cache_size_mb}.")
    if int(args.max_iter) <= 0:
        raise ValueError(f"--max-iter must be > 0, got {args.max_iter}.")
    if int(args.log_every) <= 0:
        raise ValueError(f"--log-every must be > 0, got {args.log_every}.")

    requested_feature_mask = _normalize_optional_path(args.feature_mask)
    feature_mask = requested_feature_mask
    feature_mask_strategy = "explicit_user_path"
    if feature_mask is None:
        feature_mask = _infer_default_mask(args.beta_index_csv)
        feature_mask_strategy = "auto_searchlight_default_group_mask" if feature_mask is not None else "subject_mask_intersection"

    if feature_mask is not None:
        feature_mask_path = Path(feature_mask).expanduser().resolve()
        if not feature_mask_path.exists():
            raise FileNotFoundError(f"--feature-mask does not exist: {feature_mask_path}")
        feature_mask = str(feature_mask_path)

    return LeaveSessionOutConfig(
        beta_index_csv=str(Path(args.beta_index_csv).expanduser().resolve()),
        models_root=_normalize_optional_path(args.models_root),
        brain_mask_root=_normalize_optional_path(args.brain_mask_root),
        path_prefix_from=str(args.path_prefix_from),
        path_prefix_to=str(args.path_prefix_to),
        save_root=str(Path(args.save_root).expanduser().resolve()),
        run_name=None if args.run_name is None else str(args.run_name).strip() or None,
        feature_mask=feature_mask,
        feature_mask_strategy=feature_mask_strategy,
        only_subjects=sorted(_parse_subject_list(args.only_subjects)),
        exclude_subjects=sorted(_parse_subject_list(args.exclude_subjects)),
        required_num_sessions=int(args.required_num_sessions),
        seed=int(args.seed),
        svm_c=float(args.svm_c),
        svm_backend=str(args.svm_backend),
        class_weight=str(args.class_weight),
        fold_standardize=not bool(args.no_fold_standardize),
        libsvm_cache_size_mb=float(args.libsvm_cache_size_mb),
        max_iter=int(args.max_iter),
        log_every=int(args.log_every),
    )


def config_to_dict(config: LeaveSessionOutConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["analysis_alignment"] = {
        "matches_searchlight": [
            "session-wise beta inputs by default",
            "within-subject leave-one-session-out 3-fold CV",
            "linear SVM with C=1",
            "balanced class weights by default",
            "train-fold-only standardization by default",
        ],
        "matches_cnn_reporting": [
            "primary accuracy metric is balanced_accuracy",
        ],
        "not_used": [
            "CNN",
            "searchlight neighborhoods",
            "cross-subject CV",
            "leave-one-run-out CV",
        ],
    }
    payload["primary_metric"] = "mean_fold_balanced_accuracy"
    return payload


def make_run_dir(config: LeaveSessionOutConfig) -> Path:
    save_root = ensure_dir(config.save_root)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = config.run_name or DEFAULT_RUN_NAME
    return ensure_dir(save_root / f"{timestamp}_{run_name}")


def save_feature_mask(path: str | Path, mask: np.ndarray, *, reference_image: nib.Nifti1Image) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(mask.astype(np.uint8), affine=reference_image.affine, header=reference_image.header.copy())
    image.set_data_dtype(np.uint8)
    nib.save(image, str(target))


def select_subjects(
    subject_records: Dict[str, List[FMRISample]],
    *,
    config: LeaveSessionOutConfig,
) -> tuple[List[str], List[Dict[str, Any]]]:
    included_subject_ids: List[str] = []
    excluded_subjects: List[Dict[str, Any]] = []
    only_subjects_set = set(config.only_subjects)
    exclude_subjects_set = set(config.exclude_subjects)

    for subject_id in sorted(subject_records):
        session_keys = sorted({record.session_id for record in subject_records[subject_id]})
        run_keys = sorted({(record.session_id, record.run_id) for record in subject_records[subject_id]})
        exclude_reason: str | None = None
        if only_subjects_set and subject_id not in only_subjects_set:
            exclude_reason = "not_in_only_subjects_filter"
        elif subject_id in exclude_subjects_set:
            exclude_reason = "manually_excluded"
        elif len(session_keys) != int(config.required_num_sessions):
            exclude_reason = f"num_sessions_{len(session_keys)}_!=_{config.required_num_sessions}"

        if exclude_reason is not None:
            excluded_subjects.append(
                {
                    "subject_id": subject_id,
                    "reason": exclude_reason,
                    "num_sessions": int(len(session_keys)),
                    "session_keys": session_keys,
                    "num_runs": int(len(run_keys)),
                    "run_keys": [f"{session_id}:{run_id}" for session_id, run_id in run_keys],
                }
            )
            continue
        included_subject_ids.append(subject_id)

    return included_subject_ids, excluded_subjects


def make_subject_intersection_mask(
    subject_records: Dict[str, List[FMRISample]],
    *,
    included_subject_ids: Sequence[str],
    out_path: Path,
) -> tuple[np.ndarray, nib.Nifti1Image, str]:
    common_mask: np.ndarray | None = None
    reference_image: nib.Nifti1Image | None = None
    source_mask_paths: List[str] = []

    for subject_id in included_subject_ids:
        mask_paths = sorted({record.mask_path for record in subject_records[subject_id]})
        if not mask_paths:
            raise ValueError(f"{subject_id} has no mask_path entries.")
        mask_path = mask_paths[0]
        mask, image = load_mask(mask_path)
        if reference_image is None:
            reference_image = image
        elif mask.shape != common_mask.shape:
            raise ValueError(f"Mask shape mismatch for {subject_id}: {mask.shape} vs {common_mask.shape}.")
        common_mask = mask if common_mask is None else (common_mask & mask)
        source_mask_paths.append(mask_path)

    if common_mask is None or reference_image is None:
        raise ValueError("Cannot create an intersection mask without included subjects.")
    if int(common_mask.sum()) <= 0:
        raise ValueError("The subject-mask intersection is empty.")

    save_feature_mask(out_path, common_mask, reference_image=reference_image)
    return common_mask, reference_image, str(out_path)


def resolve_feature_mask(
    *,
    config: LeaveSessionOutConfig,
    subject_records: Dict[str, List[FMRISample]],
    included_subject_ids: Sequence[str],
    run_dir: Path,
) -> tuple[np.ndarray, nib.Nifti1Image, str, str]:
    if config.feature_mask is not None:
        feature_mask, reference_image = load_mask(config.feature_mask)
        copied_mask_path = run_dir / "feature_mask.nii.gz"
        save_feature_mask(copied_mask_path, feature_mask, reference_image=reference_image)
        return feature_mask, reference_image, str(Path(config.feature_mask).expanduser().resolve()), config.feature_mask_strategy

    generated_path = run_dir / "feature_mask_subject_intersection.nii.gz"
    feature_mask, reference_image, mask_path = make_subject_intersection_mask(
        subject_records,
        included_subject_ids=included_subject_ids,
        out_path=generated_path,
    )
    return feature_mask, reference_image, mask_path, "subject_mask_intersection"


def make_classifier(config: LeaveSessionOutConfig, *, fold_index: int):
    class_weight = None if config.class_weight == "none" else config.class_weight
    if config.svm_backend == "libsvm_linear":
        return SVC(
            C=config.svm_c,
            kernel="linear",
            class_weight=class_weight,
            cache_size=config.libsvm_cache_size_mb,
            max_iter=-1,
        )
    if config.svm_backend == "linear_svc":
        return LinearSVC(
            C=config.svm_c,
            class_weight=class_weight,
            dual=True,
            max_iter=config.max_iter,
            random_state=config.seed + fold_index,
        )
    raise ValueError(f"Unsupported svm_backend={config.svm_backend!r}")


def save_fold_predictions_csv(
    path: str | Path,
    *,
    records: Sequence[FMRISample],
    targets: np.ndarray,
    preds: np.ndarray,
    decision_scores: np.ndarray,
) -> None:
    target_path = Path(path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject_id",
                "session_id",
                "run_id",
                "trial_id",
                "sample_path",
                "rating",
                "target",
                "pred",
                "decision_score",
            ],
        )
        writer.writeheader()
        for record, target, pred, score in zip(records, targets, preds, decision_scores, strict=True):
            writer.writerow(
                {
                    "subject_id": record.subject_id,
                    "session_id": record.session_id,
                    "run_id": record.run_id,
                    "trial_id": "" if record.trial_id is None else int(record.trial_id),
                    "sample_path": record.sample_path,
                    "rating": int(record.rating),
                    "target": int(target),
                    "pred": int(pred),
                    "decision_score": float(score),
                }
            )


def _class_counts(values: np.ndarray) -> Dict[str, int]:
    return {
        "rating1": int(np.sum(values == 0)),
        "rating4": int(np.sum(values == 1)),
    }


def analyze_subject(
    *,
    subject_id: str,
    records: Sequence[FMRISample],
    feature_mask: np.ndarray,
    extractor: WholeBrainFeatureExtractor,
    config: LeaveSessionOutConfig,
    run_dir: Path,
) -> Dict[str, Any]:
    subject_dir = ensure_dir(run_dir / "subjects" / subject_id)
    x, y, fold_ids, fold_keys = build_subject_design_matrix(
        records,
        feature_mask=feature_mask,
        extractor=extractor,
        log_every=config.log_every,
        subject_id=subject_id,
        cv_group="session",
    )
    ordered_records = sorted(
        records,
        key=lambda record: (
            record.session_id,
            record.run_id,
            -1 if record.trial_id is None else int(record.trial_id),
            record.sample_path,
        ),
    )

    if len(fold_keys) != int(config.required_num_sessions):
        raise ValueError(
            f"{subject_id} has {len(fold_keys)} session CV groups, expected {config.required_num_sessions}."
        )

    fold_summaries: List[Dict[str, Any]] = []
    all_targets: List[int] = []
    all_preds: List[int] = []

    for zero_based_fold_index, fold_key in enumerate(fold_keys):
        fold_index = zero_based_fold_index + 1
        val_indices = np.flatnonzero(fold_ids == zero_based_fold_index)
        train_indices = np.flatnonzero(fold_ids != zero_based_fold_index)
        y_train = y[train_indices]
        y_val = y[val_indices]
        if len(set(int(value) for value in y_train.tolist())) != 2:
            raise ValueError(f"{subject_id} fold {fold_index} training split lacks one class.")
        if len(set(int(value) for value in y_val.tolist())) != 2:
            raise ValueError(f"{subject_id} fold {fold_index} validation split lacks one class.")

        x_train, x_val = prepare_fold_features(
            x[train_indices],
            x[val_indices],
            fold_standardize=config.fold_standardize,
        )
        if x_train.shape[1] == 0:
            raise ValueError(f"{subject_id} fold {fold_index} has zero usable features after fold filtering.")

        classifier = make_classifier(config, fold_index=fold_index)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", category=ConvergenceWarning)
            classifier.fit(x_train, y_train)
        convergence_warnings = [
            str(item.message)
            for item in caught_warnings
            if issubclass(item.category, ConvergenceWarning)
        ]

        val_preds = np.asarray(classifier.predict(x_val), dtype=np.int64)
        decision_scores = np.asarray(classifier.decision_function(x_val), dtype=np.float32)
        metrics = compute_binary_classification_metrics(
            torch.from_numpy(y_val),
            torch.from_numpy(val_preds),
        )
        val_records = [ordered_records[index] for index in val_indices.tolist()]
        save_fold_predictions_csv(
            subject_dir / f"fold_{fold_index:02d}_predictions.csv",
            records=val_records,
            targets=y_val,
            preds=val_preds,
            decision_scores=decision_scores,
        )

        fold_summary = {
            "fold_index": int(fold_index),
            "cv_group": "session",
            "fold_key": ":".join(fold_key),
            "session_id": fold_key[0],
            "num_train_samples": int(len(train_indices)),
            "num_val_samples": int(len(val_indices)),
            "train_class_counts": _class_counts(y_train),
            "val_class_counts": _class_counts(y_val),
            "num_features_after_fold_filter": int(x_train.shape[1]),
            "accuracy": float(metrics["accuracy"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "f1_high": float(metrics["f1_high"]),
            "precision_high": float(metrics["precision_high"]),
            "recall_high": float(metrics["recall_high"]),
            "recall_low": float(metrics["recall_low"]),
            "confusion_matrix": metrics["confusion_matrix"],
            "convergence_warnings": convergence_warnings,
        }
        fold_summaries.append(fold_summary)
        all_targets.extend(int(value) for value in y_val.tolist())
        all_preds.extend(int(value) for value in val_preds.tolist())
        print(
            f"{subject_id} | fold {fold_index:02d}/{len(fold_keys):02d} | "
            f"session {fold_key[0]} | "
            f"bal_acc {fold_summary['balanced_accuracy']:.4f} | "
            f"acc {fold_summary['accuracy']:.4f}",
            flush=True,
        )

    pooled_metrics = compute_binary_classification_metrics(
        torch.tensor(all_targets, dtype=torch.long),
        torch.tensor(all_preds, dtype=torch.long),
    )
    subject_summary = {
        "subject_id": subject_id,
        "num_cv_groups": int(len(fold_keys)),
        "fold_keys": [":".join(fold_key) for fold_key in fold_keys],
        "num_runs": int(len({(record.session_id, record.run_id) for record in records})),
        "num_examples": int(len(ordered_records)),
        "num_features_in_mask": int(feature_mask.sum()),
        "primary_metric": "mean_fold_balanced_accuracy",
        "mean_fold_accuracy": mean_and_std(summary["accuracy"] for summary in fold_summaries),
        "mean_fold_balanced_accuracy": mean_and_std(summary["balanced_accuracy"] for summary in fold_summaries),
        "mean_fold_f1_high": mean_and_std(summary["f1_high"] for summary in fold_summaries),
        "pooled_accuracy": float(pooled_metrics["accuracy"]),
        "pooled_balanced_accuracy": float(pooled_metrics["balanced_accuracy"]),
        "pooled_f1_high": float(pooled_metrics["f1_high"]),
        "pooled_confusion_matrix": pooled_metrics["confusion_matrix"],
        "svm_backend": config.svm_backend,
        "svm_c": float(config.svm_c),
        "class_weight": config.class_weight,
        "fold_standardize": bool(config.fold_standardize),
        "folds": fold_summaries,
    }
    save_json(subject_dir / "summary.json", subject_summary)
    return subject_summary


def _ttest(values: Sequence[float], *, popmean: float) -> Dict[str, float | None]:
    values_array = np.asarray(list(values), dtype=np.float64)
    if values_array.size < 2:
        return {"t_stat": None, "p_value": None}
    result = stats.ttest_1samp(values_array, popmean=popmean)
    return {
        "t_stat": None if np.isnan(result.statistic) else float(result.statistic),
        "p_value": None if np.isnan(result.pvalue) else float(result.pvalue),
    }


def summarize_group(
    subject_summaries: Sequence[Dict[str, Any]],
    *,
    excluded_subjects: Sequence[Dict[str, Any]],
    feature_mask_source: str,
    feature_mask_strategy: str,
) -> Dict[str, Any]:
    mean_fold_accuracies = [float(summary["mean_fold_accuracy"]["mean"]) for summary in subject_summaries]
    mean_fold_bal_accs = [float(summary["mean_fold_balanced_accuracy"]["mean"]) for summary in subject_summaries]
    pooled_accuracies = [float(summary["pooled_accuracy"]) for summary in subject_summaries]
    pooled_bal_accs = [float(summary["pooled_balanced_accuracy"]) for summary in subject_summaries]

    return {
        "primary_metric": "mean_fold_balanced_accuracy",
        "num_subjects_included": int(len(subject_summaries)),
        "feature_mask_source": feature_mask_source,
        "feature_mask_strategy": feature_mask_strategy,
        "subjects": list(subject_summaries),
        "excluded_subjects": list(excluded_subjects),
        "mean_fold_accuracy": mean_and_std(mean_fold_accuracies),
        "mean_fold_balanced_accuracy": mean_and_std(mean_fold_bal_accs),
        "pooled_accuracy": mean_and_std(pooled_accuracies),
        "pooled_balanced_accuracy": mean_and_std(pooled_bal_accs),
        "ttest_mean_fold_accuracy_vs_chance_0p5": _ttest(mean_fold_accuracies, popmean=0.5),
        "ttest_mean_fold_balanced_accuracy_vs_chance_0p5": _ttest(mean_fold_bal_accs, popmean=0.5),
    }


def save_subject_metrics_csv(path: str | Path, subject_summaries: Sequence[Dict[str, Any]]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject_id",
                "num_examples",
                "num_runs",
                "num_cv_groups",
                "mean_fold_accuracy",
                "mean_fold_balanced_accuracy",
                "pooled_accuracy",
                "pooled_balanced_accuracy",
            ],
        )
        writer.writeheader()
        for summary in subject_summaries:
            writer.writerow(
                {
                    "subject_id": summary["subject_id"],
                    "num_examples": summary["num_examples"],
                    "num_runs": summary["num_runs"],
                    "num_cv_groups": summary["num_cv_groups"],
                    "mean_fold_accuracy": summary["mean_fold_accuracy"]["mean"],
                    "mean_fold_balanced_accuracy": summary["mean_fold_balanced_accuracy"]["mean"],
                    "pooled_accuracy": summary["pooled_accuracy"],
                    "pooled_balanced_accuracy": summary["pooled_balanced_accuracy"],
                }
            )


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    set_seed(config.seed)
    run_dir = make_run_dir(config)

    records, dataset_summary = discover_samples(
        beta_index_csv=config.beta_index_csv,
        models_root=config.models_root,
        brain_mask_root=config.brain_mask_root,
        path_prefix_from=config.path_prefix_from,
        path_prefix_to=config.path_prefix_to,
    )
    subject_records = group_records_by_subject(records)
    included_subject_ids, excluded_subjects = select_subjects(subject_records, config=config)
    feature_mask, _reference_image, feature_mask_source, feature_mask_strategy = resolve_feature_mask(
        config=config,
        subject_records=subject_records,
        included_subject_ids=included_subject_ids,
        run_dir=run_dir,
    )

    save_json(run_dir / "run_config.json", config_to_dict(config))
    save_json(run_dir / "dataset_summary.json", dataset_summary)
    save_json(
        run_dir / "subject_selection.json",
        {
            "included_subject_ids": included_subject_ids,
            "excluded_subjects": excluded_subjects,
            "required_num_sessions": int(config.required_num_sessions),
            "only_subjects": config.only_subjects,
            "explicitly_excluded_subjects": config.exclude_subjects,
        },
    )

    extractor = WholeBrainFeatureExtractor()
    subject_summaries: List[Dict[str, Any]] = []
    for subject_id in included_subject_ids:
        subject_summaries.append(
            analyze_subject(
                subject_id=subject_id,
                records=subject_records[subject_id],
                feature_mask=feature_mask,
                extractor=extractor,
                config=config,
                run_dir=run_dir,
            )
        )

    group_summary = summarize_group(
        subject_summaries,
        excluded_subjects=excluded_subjects,
        feature_mask_source=feature_mask_source,
        feature_mask_strategy=feature_mask_strategy,
    )
    save_json(run_dir / "group_summary.json", group_summary)
    save_subject_metrics_csv(run_dir / "subject_metrics.csv", subject_summaries)
    print(f"run_dir: {run_dir}")
    print(
        "group summary | "
        f"subjects {group_summary['num_subjects_included']} | "
        f"primary mean_fold_bal_acc {group_summary['mean_fold_balanced_accuracy']['mean']:.4f} +/- "
        f"{group_summary['mean_fold_balanced_accuracy']['std']:.4f} | "
        f"mean_fold_acc {group_summary['mean_fold_accuracy']['mean']:.4f} +/- "
        f"{group_summary['mean_fold_accuracy']['std']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
