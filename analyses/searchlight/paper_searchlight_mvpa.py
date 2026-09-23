from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from analyses._paths import DATA_ROOT, RUNS_ROOT, OUTPUT_ROOT, FSL_ROOT, ATLAS_ROOT
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from scipy.ndimage import generate_binary_structure, label
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC, SVC

from foodpref.config import (
    DEFAULT_BETA_INDEX_CSV,
    DEFAULT_BRAIN_MASK_ROOT,
    DEFAULT_MODELS_ROOT,
    DEFAULT_SAVE_ROOT,
)
from foodpref.dataset import FMRISample, discover_samples
from foodpref.utils import ensure_dir, rating_to_target, save_json, set_seed


DEFAULT_RUN_NAME = "rating4_vs_rating1_libsvmLinearSVC_ses3fold_radius6mm"
DEFAULT_SVM_C = 1.0
DEFAULT_MAX_ITER = 10000
DEFAULT_CV_GROUP = "session"
DEFAULT_REQUIRED_NUM_RUNS = 3
DEFAULT_LOG_EVERY = 250
DEFAULT_RADIUS_MM = 6.0
DEFAULT_SVM_BACKEND = "libsvm_linear"
DEFAULT_CLASS_WEIGHT = "balanced"
DEFAULT_LIBSVM_CACHE_SIZE_MB = 1024.0
DEFAULT_N_JOBS = 1
DEFAULT_CHUNK_SIZE = 250
DEFAULT_FSL_RANDOMISE = str(FSL_ROOT / "bin" / "randomise")
DEFAULT_SESSIONWISE_OUTPUT_ROOT = str(DATA_ROOT / "derived" / "sessionwise")
DEFAULT_SESSIONWISE_BETA_INDEX_CSV = f"{DEFAULT_SESSIONWISE_OUTPUT_ROOT}/beta_index.csv"
DEFAULT_SESSIONWISE_MODELS_ROOT = f"{DEFAULT_SESSIONWISE_OUTPUT_ROOT}/models"
DEFAULT_SESSIONWISE_BRAIN_MASK_ROOT = f"{DEFAULT_SESSIONWISE_OUTPUT_ROOT}/cache/masks"
DEFAULT_FIXED_MASK_CANDIDATES = (
    "group_randomise_rating14_no_quality_filter/group_inputs/group_mask_intersection.nii.gz",
    "group_randomise_rating14_q3/group_inputs/group_mask_intersection.nii.gz",
    "group_randomise_rating14_q2/group_inputs/group_mask_intersection.nii.gz",
    "group_randomise_rating14_q3_test/group_inputs/group_mask_intersection.nii.gz",
)


@dataclass(frozen=True)
class SearchlightConfig:
    beta_index_csv: str
    models_root: str | None
    brain_mask_root: str | None
    path_prefix_from: str
    path_prefix_to: str
    save_root: str
    run_name: str | None
    feature_mask: str
    feature_mask_strategy: str
    center_mask: str | None
    center_mask_strategy: str
    exclude_subjects: List[str]
    only_subjects: List[str]
    cv_group: str
    required_num_runs: int
    radius_mm: float
    center_step: int
    max_centers: int | None
    seed: int
    svm_c: float
    svm_backend: str
    class_weight: str
    fold_standardize: bool
    libsvm_cache_size_mb: float
    max_iter: int
    n_jobs: int
    chunk_size: int
    log_every: int
    run_randomise: bool
    permutations: int
    randomise_bin: str


def _normalize_optional_path(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_subject_list(value: str) -> List[str]:
    parts = [item.strip() for item in str(value).split(",")]
    return [item for item in parts if item]


def _infer_default_mask(beta_index_csv: str) -> str | None:
    output_root = Path(beta_index_csv).expanduser().resolve().parent
    project_root = output_root.parent
    desktop_output_root = DATA_ROOT / "derived"
    for relative_path in DEFAULT_FIXED_MASK_CANDIDATES:
        raw_path = Path(relative_path)
        candidates = [raw_path] if raw_path.is_absolute() else [
            output_root / raw_path,
            project_root / "output" / raw_path,
            desktop_output_root / raw_path,
        ]
        for candidate in candidates:
            candidate = candidate.expanduser().resolve()
            if candidate.exists():
                return str(candidate)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run paper-style searchlight MVPA within each subject using leave-one-session-out "
            "or leave-one-run-out cross-validation."
        )
    )
    parser.add_argument("--beta-index-csv", default=DEFAULT_SESSIONWISE_BETA_INDEX_CSV)
    parser.add_argument("--models-root", default=DEFAULT_SESSIONWISE_MODELS_ROOT)
    parser.add_argument("--brain-mask-root", default=DEFAULT_SESSIONWISE_BRAIN_MASK_ROOT)
    parser.add_argument("--path-prefix-from", default="")
    parser.add_argument("--path-prefix-to", default="")
    parser.add_argument("--save-root", default=str(OUTPUT_ROOT / "searchlight"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--feature-mask",
        default=None,
        help=(
            "Optional fixed standard-space whole-brain feature mask. If omitted, the script first "
            "tries to auto-discover a group-level MNI mask under the output tree."
        ),
    )
    parser.add_argument(
        "--center-mask",
        default=None,
        help=(
            "Optional searchlight-center mask. If omitted, the feature mask is reused. "
            "This is useful when we want to pilot on an ROI or a thinned center grid."
        ),
    )
    parser.add_argument("--exclude-subjects", default="")
    parser.add_argument("--only-subjects", default="")
    parser.add_argument(
        "--cv-group",
        choices=("session", "run"),
        default=DEFAULT_CV_GROUP,
        help="Cross-validation unit. `session` gives the primary 3-fold leave-one-session-out analysis.",
    )
    parser.add_argument(
        "--required-num-runs",
        type=int,
        default=DEFAULT_REQUIRED_NUM_RUNS,
        help=(
            "Required number of CV groups per subject. Kept as a legacy name; with "
            "--cv-group session the default 3 means three BIDS sessions."
        ),
    )
    parser.add_argument(
        "--required-num-groups",
        type=int,
        default=None,
        help="Clearer alias for --required-num-runs. If set, it overrides --required-num-runs.",
    )
    parser.add_argument("--radius-mm", type=float, default=DEFAULT_RADIUS_MM)
    parser.add_argument(
        "--center-step",
        type=int,
        default=1,
        help="Keep every Nth center voxel after mask extraction. Useful for pilot runs.",
    )
    parser.add_argument(
        "--max-centers",
        type=int,
        default=None,
        help="Optional cap on the number of searchlight centers after thinning.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svm-c", type=float, default=DEFAULT_SVM_C)
    parser.add_argument(
        "--svm-backend",
        choices=("linear_svc", "libsvm_linear"),
        default=DEFAULT_SVM_BACKEND,
        help=(
            "Linear SVM backend. `linear_svc` is the practical default for searchlight; "
            "`libsvm_linear` is closer to the paper but much slower."
        ),
    )
    parser.add_argument(
        "--class-weight",
        choices=("balanced", "none"),
        default=DEFAULT_CLASS_WEIGHT,
        help="Use balanced class weights by default because session-wise run-beta counts can be imbalanced.",
    )
    parser.add_argument(
        "--no-fold-standardize",
        action="store_true",
        help="Disable train-fold-only z-scoring inside each searchlight sphere.",
    )
    parser.add_argument("--libsvm-cache-size-mb", type=float, default=DEFAULT_LIBSVM_CACHE_SIZE_MB)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument(
        "--run-randomise",
        action="store_true",
        help=(
            "Optionally run group-level FSL randomise on subject searchlight maps after subtracting chance "
            "level 0.5. This is a practical group-inference add-on, not an exact reimplementation of the "
            "paper's cluster-forming-threshold procedure."
        ),
    )
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--randomise-bin", default=DEFAULT_FSL_RANDOMISE)
    return parser


def parse_args(argv: list[str] | None = None) -> SearchlightConfig:
    args = build_parser().parse_args(argv)
    required_num_runs = int(args.required_num_groups) if args.required_num_groups is not None else int(args.required_num_runs)
    if required_num_runs <= 1:
        raise ValueError(f"Required number of CV groups must be > 1, got {required_num_runs}.")
    if float(args.radius_mm) <= 0.0:
        raise ValueError(f"--radius-mm must be > 0, got {args.radius_mm}.")
    if int(args.center_step) <= 0:
        raise ValueError(f"--center-step must be > 0, got {args.center_step}.")
    if args.max_centers is not None and int(args.max_centers) <= 0:
        raise ValueError(f"--max-centers must be > 0 when provided, got {args.max_centers}.")
    if float(args.svm_c) <= 0.0:
        raise ValueError(f"--svm-c must be > 0, got {args.svm_c}.")
    if float(args.libsvm_cache_size_mb) <= 0.0:
        raise ValueError(f"--libsvm-cache-size-mb must be > 0, got {args.libsvm_cache_size_mb}.")
    if int(args.max_iter) <= 0:
        raise ValueError(f"--max-iter must be > 0, got {args.max_iter}.")
    if int(args.n_jobs) == 0:
        raise ValueError("--n-jobs cannot be 0.")
    if int(args.chunk_size) <= 0:
        raise ValueError(f"--chunk-size must be > 0, got {args.chunk_size}.")
    if int(args.log_every) <= 0:
        raise ValueError(f"--log-every must be > 0, got {args.log_every}.")
    if int(args.permutations) <= 0:
        raise ValueError(f"--permutations must be > 0, got {args.permutations}.")

    requested_feature_mask = _normalize_optional_path(args.feature_mask)
    feature_mask = requested_feature_mask or _infer_default_mask(args.beta_index_csv)
    if feature_mask is None:
        raise FileNotFoundError(
            "Could not infer a fixed standard-space whole-brain feature mask. Pass --feature-mask explicitly."
        )
    feature_mask_path = Path(feature_mask).expanduser().resolve()
    if not feature_mask_path.exists():
        raise FileNotFoundError(f"--feature-mask does not exist: {feature_mask_path}")

    requested_center_mask = _normalize_optional_path(args.center_mask)
    center_mask_path: Path | None = None
    center_mask_strategy = "reuse_feature_mask"
    if requested_center_mask is not None:
        center_mask_path = Path(requested_center_mask).expanduser().resolve()
        if not center_mask_path.exists():
            raise FileNotFoundError(f"--center-mask does not exist: {center_mask_path}")
        center_mask_strategy = "explicit_user_path"

    return SearchlightConfig(
        beta_index_csv=str(Path(args.beta_index_csv).expanduser().resolve()),
        models_root=_normalize_optional_path(args.models_root),
        brain_mask_root=_normalize_optional_path(args.brain_mask_root),
        path_prefix_from=str(args.path_prefix_from),
        path_prefix_to=str(args.path_prefix_to),
        save_root=str(Path(args.save_root).expanduser().resolve()),
        run_name=None if args.run_name is None else str(args.run_name).strip() or None,
        feature_mask=str(feature_mask_path),
        feature_mask_strategy="explicit_user_path" if requested_feature_mask else "auto_fixed_group_mask",
        center_mask=None if center_mask_path is None else str(center_mask_path),
        center_mask_strategy=center_mask_strategy,
        exclude_subjects=sorted(_parse_subject_list(args.exclude_subjects)),
        only_subjects=sorted(_parse_subject_list(args.only_subjects)),
        cv_group=str(args.cv_group),
        required_num_runs=required_num_runs,
        radius_mm=float(args.radius_mm),
        center_step=int(args.center_step),
        max_centers=None if args.max_centers is None else int(args.max_centers),
        seed=int(args.seed),
        svm_c=float(args.svm_c),
        svm_backend=str(args.svm_backend),
        class_weight=str(args.class_weight),
        fold_standardize=not bool(args.no_fold_standardize),
        libsvm_cache_size_mb=float(args.libsvm_cache_size_mb),
        max_iter=int(args.max_iter),
        n_jobs=int(args.n_jobs),
        chunk_size=int(args.chunk_size),
        log_every=int(args.log_every),
        run_randomise=bool(args.run_randomise),
        permutations=int(args.permutations),
        randomise_bin=str(args.randomise_bin),
    )


def config_to_dict(config: SearchlightConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["paper_alignment_notes"] = {
        "matched": [
            "unsmoothed_beta_inputs",
            "fixed_standard_space_whole_brain_mask",
            "searchlight_radius_6mm",
            "linear_svm_local_decoding",
            "within_subject_cross_validation",
        ],
        "adapted_for_this_project": [
            "run_wise_rating_condition_betas",
            "rating1_vs_rating4_labels",
            "primary_leave_one_session_out_3fold_cross_validation",
            "balanced_class_weights_for_imbalanced_session_samples",
            "train_fold_only_searchlight_standardization",
            "default_linear_svc_backend_for_runtime",
        ],
    }
    return payload


def make_run_dir(config: SearchlightConfig) -> Path:
    save_root = ensure_dir(config.save_root)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = config.run_name or DEFAULT_RUN_NAME
    return ensure_dir(save_root / f"{timestamp}_{run_name}")


def _format_number_token(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "p")


def analysis_tag(config: SearchlightConfig) -> str:
    backend = "linearSVC" if config.svm_backend == "linear_svc" else "libsvmLinearSVC"
    cv_token = "ses" if config.cv_group == "session" else "run"
    return (
        f"rating4_vs_rating1_{backend}_"
        f"{cv_token}{int(config.required_num_runs)}fold_"
        f"radius{_format_number_token(config.radius_mm)}mm"
    )


def load_mask(mask_path: str | Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(Path(mask_path).expanduser().resolve()))
    data = np.asarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape={tuple(data.shape)} for {mask_path}.")
    mask = np.asarray(np.isfinite(data) & (data > 0), dtype=bool)
    if int(mask.sum()) <= 0:
        raise ValueError(f"Mask contains no positive voxels: {mask_path}")
    return mask, image


def save_nifti(path: str | Path, data: np.ndarray, *, reference_image: nib.Nifti1Image) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(data), affine=reference_image.affine, header=reference_image.header.copy())
    image.set_data_dtype(np.asarray(data).dtype)
    nib.save(image, str(target))


def build_sphere_offsets(radius_mm: float, voxel_sizes: Sequence[float]) -> np.ndarray:
    voxel_sizes = tuple(float(value) for value in voxel_sizes[:3])
    max_steps = [int(math.ceil(radius_mm / max(size, 1e-6))) for size in voxel_sizes]
    offsets: List[Tuple[int, int, int]] = []
    for dx in range(-max_steps[0], max_steps[0] + 1):
        for dy in range(-max_steps[1], max_steps[1] + 1):
            for dz in range(-max_steps[2], max_steps[2] + 1):
                squared_distance = (
                    (dx * voxel_sizes[0]) ** 2
                    + (dy * voxel_sizes[1]) ** 2
                    + (dz * voxel_sizes[2]) ** 2
                )
                if squared_distance <= float(radius_mm) ** 2 + 1e-6:
                    offsets.append((dx, dy, dz))
    if not offsets:
        raise ValueError(f"No sphere offsets were generated for radius_mm={radius_mm}.")
    return np.asarray(offsets, dtype=np.int16)


def build_center_coordinates(mask: np.ndarray, *, center_step: int, max_centers: int | None) -> np.ndarray:
    coords = np.argwhere(mask)
    if int(center_step) > 1:
        coords = coords[:: int(center_step)]
    if max_centers is not None and len(coords) > int(max_centers):
        selected = np.linspace(0, len(coords) - 1, num=int(max_centers), dtype=np.int64)
        coords = coords[selected]
    if coords.size == 0:
        raise ValueError("The center mask produced zero searchlight centers.")
    return np.asarray(coords, dtype=np.int16)


def build_feature_index_map(feature_mask: np.ndarray) -> np.ndarray:
    feature_index_map = np.full(feature_mask.shape, fill_value=-1, dtype=np.int32)
    feature_index_map[feature_mask] = np.arange(int(feature_mask.sum()), dtype=np.int32)
    return feature_index_map


def chunk_coordinates(center_coords: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(center_coords), int(chunk_size)):
        yield center_coords[start : start + int(chunk_size)]


class WholeBrainFeatureExtractor:
    def extract_features(self, record: FMRISample, feature_mask: np.ndarray) -> np.ndarray:
        image = nib.load(str(Path(record.sample_path).expanduser().resolve()))
        volume = np.asarray(image.dataobj, dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D beta map, got shape={tuple(volume.shape)} for {record.sample_path}.")
        if volume.shape != feature_mask.shape:
            raise ValueError(
                f"Feature mask shape {tuple(feature_mask.shape)} does not match "
                f"beta-map shape {tuple(volume.shape)} for {record.sample_path}."
            )

        values = np.asarray(volume[feature_mask], dtype=np.float32)
        finite_mask = np.isfinite(values)
        features = np.zeros_like(values, dtype=np.float32)
        if bool(finite_mask.any()):
            finite_values = values[finite_mask]
            value_min = float(finite_values.min())
            value_max = float(finite_values.max())
            if value_max > value_min:
                features[finite_mask] = (finite_values - value_min) / (value_max - value_min)
        features[~np.isfinite(features)] = 0.0
        return features


def group_records_by_subject(records: Sequence[FMRISample]) -> Dict[str, List[FMRISample]]:
    grouped: Dict[str, List[FMRISample]] = defaultdict(list)
    for record in records:
        grouped[record.subject_id].append(record)
    return grouped


def build_subject_design_matrix(
    records: Sequence[FMRISample],
    *,
    feature_mask: np.ndarray,
    extractor: WholeBrainFeatureExtractor,
    log_every: int,
    subject_id: str,
    cv_group: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, ...]]]:
    ordered_records = sorted(
        records,
        key=lambda record: (
            record.session_id,
            record.run_id,
            -1 if record.trial_id is None else int(record.trial_id),
            record.sample_path,
        ),
    )
    if cv_group == "session":
        fold_keys = sorted({(record.session_id,) for record in ordered_records})
    elif cv_group == "run":
        fold_keys = sorted({(record.session_id, record.run_id) for record in ordered_records})
    else:
        raise ValueError(f"Unsupported cv_group={cv_group!r}.")
    fold_key_to_index = {key: index for index, key in enumerate(fold_keys)}

    x = np.zeros((len(ordered_records), int(feature_mask.sum())), dtype=np.float32)
    y = np.zeros(len(ordered_records), dtype=np.int64)
    fold_ids = np.zeros(len(ordered_records), dtype=np.int16)
    for index, record in enumerate(ordered_records):
        x[index] = extractor.extract_features(record, feature_mask)
        y[index] = int(rating_to_target(record.rating))
        fold_key = (record.session_id,) if cv_group == "session" else (record.session_id, record.run_id)
        fold_ids[index] = int(fold_key_to_index[fold_key])
        if (index + 1) % int(log_every) == 0 or (index + 1) == len(ordered_records):
            print(f"{subject_id} | loaded {index + 1}/{len(ordered_records)} samples", flush=True)
    return x, y, fold_ids, fold_keys


def make_classifier(config: SearchlightConfig, *, seed_offset: int):
    class_weight = None if config.class_weight == "none" else config.class_weight
    if config.svm_backend == "linear_svc":
        return LinearSVC(
            C=config.svm_c,
            class_weight=class_weight,
            dual=True,
            max_iter=config.max_iter,
            random_state=config.seed + seed_offset,
        )
    if config.svm_backend == "libsvm_linear":
        return SVC(
            C=config.svm_c,
            kernel="linear",
            class_weight=class_weight,
            cache_size=config.libsvm_cache_size_mb,
            max_iter=-1,
        )
    raise ValueError(f"Unsupported svm_backend={config.svm_backend!r}")


def compute_searchlight_neighborhood(
    center_coord: np.ndarray,
    *,
    sphere_offsets: np.ndarray,
    feature_index_map: np.ndarray,
) -> np.ndarray:
    coords = sphere_offsets.astype(np.int32) + center_coord.astype(np.int32)
    valid = np.all((coords >= 0) & (coords < np.asarray(feature_index_map.shape, dtype=np.int32)), axis=1)
    coords = coords[valid]
    indices = feature_index_map[coords[:, 0], coords[:, 1], coords[:, 2]]
    indices = indices[indices >= 0]
    return np.asarray(indices, dtype=np.int32)


def prepare_fold_features(
    x_train: np.ndarray,
    x_val: np.ndarray,
    *,
    fold_standardize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    train_finite = np.all(np.isfinite(x_train), axis=0)
    val_finite = np.all(np.isfinite(x_val), axis=0)
    train_std = np.std(x_train, axis=0, ddof=0)
    good = train_finite & val_finite & np.isfinite(train_std) & (train_std > 0.0)
    if not bool(np.any(good)):
        return np.empty((x_train.shape[0], 0), dtype=np.float32), np.empty((x_val.shape[0], 0), dtype=np.float32)

    x_train = np.asarray(x_train[:, good], dtype=np.float32)
    x_val = np.asarray(x_val[:, good], dtype=np.float32)
    if fold_standardize:
        mu = np.mean(x_train, axis=0, dtype=np.float64).astype(np.float32)
        sd = np.std(x_train, axis=0, ddof=0, dtype=np.float64).astype(np.float32)
        sd[sd <= 0.0] = 1.0
        x_train = (x_train - mu) / sd
        x_val = (x_val - mu) / sd
    return x_train, x_val


def evaluate_center(
    center_coord: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    train_indices_by_fold: Sequence[np.ndarray],
    val_indices_by_fold: Sequence[np.ndarray],
    sphere_offsets: np.ndarray,
    feature_index_map: np.ndarray,
    config: SearchlightConfig,
) -> tuple[float, float, int]:
    neighborhood = compute_searchlight_neighborhood(
        center_coord,
        sphere_offsets=sphere_offsets,
        feature_index_map=feature_index_map,
    )
    if neighborhood.size == 0:
        return 0.5, 0.5, 0

    fold_accuracies: List[float] = []
    fold_bal_accuracies: List[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for fold_index, (train_indices, val_indices) in enumerate(
            zip(train_indices_by_fold, val_indices_by_fold, strict=True),
            start=1,
        ):
            classifier = make_classifier(config, seed_offset=fold_index)
            x_train = x[np.ix_(train_indices, neighborhood)]
            x_val = x[np.ix_(val_indices, neighborhood)]
            y_train = y[train_indices]
            y_val = y[val_indices]
            x_train, x_val = prepare_fold_features(
                x_train,
                x_val,
                fold_standardize=config.fold_standardize,
            )
            if x_train.shape[1] == 0:
                fold_accuracies.append(0.5)
                fold_bal_accuracies.append(0.5)
                continue
            classifier.fit(x_train, y_train)
            preds = np.asarray(classifier.predict(x_val), dtype=np.int64)

            tn = int(np.sum((y_val == 0) & (preds == 0)))
            fp = int(np.sum((y_val == 0) & (preds == 1)))
            fn = int(np.sum((y_val == 1) & (preds == 0)))
            tp = int(np.sum((y_val == 1) & (preds == 1)))
            total = tn + fp + fn + tp
            accuracy = float((tn + tp) / total) if total > 0 else 0.0
            recall_high = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            recall_low = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            balanced_accuracy = 0.5 * (recall_high + recall_low)
            fold_accuracies.append(accuracy)
            fold_bal_accuracies.append(balanced_accuracy)

    return float(np.mean(fold_accuracies)), float(np.mean(fold_bal_accuracies)), int(neighborhood.size)


def process_center_chunk(
    center_coords_chunk: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    train_indices_by_fold: Sequence[np.ndarray],
    val_indices_by_fold: Sequence[np.ndarray],
    sphere_offsets: np.ndarray,
    feature_index_map: np.ndarray,
    config: SearchlightConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    accuracy_values = np.zeros(len(center_coords_chunk), dtype=np.float32)
    bal_values = np.zeros(len(center_coords_chunk), dtype=np.float32)
    sphere_sizes = np.zeros(len(center_coords_chunk), dtype=np.int32)
    for index, center_coord in enumerate(center_coords_chunk):
        accuracy, bal_accuracy, sphere_size = evaluate_center(
            center_coord,
            x=x,
            y=y,
            train_indices_by_fold=train_indices_by_fold,
            val_indices_by_fold=val_indices_by_fold,
            sphere_offsets=sphere_offsets,
            feature_index_map=feature_index_map,
            config=config,
        )
        accuracy_values[index] = accuracy
        bal_values[index] = bal_accuracy
        sphere_sizes[index] = sphere_size
    return accuracy_values, bal_values, sphere_sizes


def subject_searchlight(
    *,
    subject_id: str,
    records: Sequence[FMRISample],
    feature_mask: np.ndarray,
    center_coords: np.ndarray,
    sphere_offsets: np.ndarray,
    feature_index_map: np.ndarray,
    reference_image: nib.Nifti1Image,
    config: SearchlightConfig,
    run_dir: Path,
) -> Dict[str, Any]:
    extractor = WholeBrainFeatureExtractor()
    x, y, fold_ids, fold_keys = build_subject_design_matrix(
        records,
        feature_mask=feature_mask,
        extractor=extractor,
        log_every=config.log_every,
        subject_id=subject_id,
        cv_group=config.cv_group,
    )

    if len(fold_keys) != int(config.required_num_runs):
        raise ValueError(
            f"{subject_id} has {len(fold_keys)} {config.cv_group} CV groups, expected {config.required_num_runs}. "
            "Subjects with incomplete CV coverage should be excluded."
        )

    train_indices_by_fold: List[np.ndarray] = []
    val_indices_by_fold: List[np.ndarray] = []
    fold_summaries: List[Dict[str, Any]] = []
    for fold_index, fold_key in enumerate(fold_keys):
        val_indices = np.flatnonzero(fold_ids == fold_index)
        train_indices = np.flatnonzero(fold_ids != fold_index)
        y_train = y[train_indices]
        y_val = y[val_indices]
        if len(set(int(value) for value in y_train.tolist())) != 2:
            raise ValueError(f"{subject_id} fold {fold_index + 1} training split lacks one class.")
        if len(set(int(value) for value in y_val.tolist())) != 2:
            raise ValueError(f"{subject_id} fold {fold_index + 1} validation split lacks one class.")
        train_indices_by_fold.append(train_indices)
        val_indices_by_fold.append(val_indices)
        fold_summaries.append(
            {
                "fold_index": int(fold_index + 1),
                "cv_group": config.cv_group,
                "fold_key": ":".join(fold_key),
                "train_n": int(len(train_indices)),
                "test_n": int(len(val_indices)),
                "train_class_counts": {
                    "rating1": int(np.sum(y_train == 0)),
                    "rating4": int(np.sum(y_train == 1)),
                },
                "test_class_counts": {
                    "rating1": int(np.sum(y_val == 0)),
                    "rating4": int(np.sum(y_val == 1)),
                },
            }
        )

    center_chunks = list(chunk_coordinates(center_coords, config.chunk_size))
    parallel_results = Parallel(n_jobs=config.n_jobs, prefer="threads")(
        delayed(process_center_chunk)(
            chunk,
            x=x,
            y=y,
            train_indices_by_fold=train_indices_by_fold,
            val_indices_by_fold=val_indices_by_fold,
            sphere_offsets=sphere_offsets,
            feature_index_map=feature_index_map,
            config=config,
        )
        for chunk in center_chunks
    )

    accuracy_values = np.concatenate([result[0] for result in parallel_results], axis=0)
    bal_values = np.concatenate([result[1] for result in parallel_results], axis=0)
    sphere_sizes = np.concatenate([result[2] for result in parallel_results], axis=0)

    accuracy_map = np.zeros(feature_mask.shape, dtype=np.float32)
    bal_map = np.zeros(feature_mask.shape, dtype=np.float32)
    accuracy_centered_map = np.zeros(feature_mask.shape, dtype=np.float32)
    bal_centered_map = np.zeros(feature_mask.shape, dtype=np.float32)
    sphere_size_map = np.zeros(feature_mask.shape, dtype=np.int16)
    accuracy_map[center_coords[:, 0], center_coords[:, 1], center_coords[:, 2]] = accuracy_values
    bal_map[center_coords[:, 0], center_coords[:, 1], center_coords[:, 2]] = bal_values
    accuracy_centered_map[center_coords[:, 0], center_coords[:, 1], center_coords[:, 2]] = accuracy_values - 0.5
    bal_centered_map[center_coords[:, 0], center_coords[:, 1], center_coords[:, 2]] = bal_values - 0.5
    sphere_size_map[center_coords[:, 0], center_coords[:, 1], center_coords[:, 2]] = np.asarray(
        np.minimum(sphere_sizes, np.iinfo(np.int16).max),
        dtype=np.int16,
    )

    subject_dir = ensure_dir(run_dir / "subject_maps")
    tag = analysis_tag(config)
    accuracy_path = subject_dir / f"{subject_id}_{tag}_accuracy.nii.gz"
    bal_path = subject_dir / f"{subject_id}_{tag}_balancedAcc.nii.gz"
    accuracy_centered_path = subject_dir / f"{subject_id}_{tag}_accuracyMinusChance.nii.gz"
    bal_centered_path = subject_dir / f"{subject_id}_{tag}_balancedAccMinusChance.nii.gz"
    sphere_size_path = subject_dir / f"{subject_id}_{tag}_sphereSize.nii.gz"
    save_nifti(accuracy_path, accuracy_map, reference_image=reference_image)
    save_nifti(bal_path, bal_map, reference_image=reference_image)
    save_nifti(accuracy_centered_path, accuracy_centered_map, reference_image=reference_image)
    save_nifti(bal_centered_path, bal_centered_map, reference_image=reference_image)
    save_nifti(sphere_size_path, sphere_size_map, reference_image=reference_image)

    summary = {
        "subject_id": subject_id,
        "cv_group": config.cv_group,
        "num_cv_groups": int(len(fold_keys)),
        "fold_keys": [":".join(fold_key) for fold_key in fold_keys],
        "folds": fold_summaries,
        "num_runs": int(len({(record.session_id, record.run_id) for record in records})),
        "num_examples": int(len(records)),
        "num_centers": int(len(center_coords)),
        "radius_mm": float(config.radius_mm),
        "analysis_tag": tag,
        "mean_searchlight_accuracy": float(np.mean(accuracy_values)),
        "std_searchlight_accuracy": float(np.std(accuracy_values, ddof=0)),
        "mean_searchlight_balanced_accuracy": float(np.mean(bal_values)),
        "std_searchlight_balanced_accuracy": float(np.std(bal_values, ddof=0)),
        "mean_sphere_size_voxels": float(np.mean(sphere_sizes)),
        "min_sphere_size_voxels": int(np.min(sphere_sizes)),
        "max_sphere_size_voxels": int(np.max(sphere_sizes)),
        "accuracy_map": str(accuracy_path),
        "balanced_accuracy_map": str(bal_path),
        "accuracy_minus_chance_map": str(accuracy_centered_path),
        "balanced_accuracy_minus_chance_map": str(bal_centered_path),
        "sphere_size_map": str(sphere_size_path),
    }
    save_json(subject_dir / f"{subject_id}_summary.json", summary)
    print(
        f"{subject_id} | centers {len(center_coords)} | "
        f"mean_acc {summary['mean_searchlight_accuracy']:.4f} | "
        f"mean_bal_acc {summary['mean_searchlight_balanced_accuracy']:.4f}",
        flush=True,
    )
    return summary


def stack_niftis(nifti_paths: Sequence[Path]) -> nib.Nifti1Image:
    if not nifti_paths:
        raise ValueError("stack_niftis requires at least one path.")
    first = nib.load(str(nifti_paths[0]))
    arrays = []
    for path in nifti_paths:
        image = nib.load(str(path))
        if image.shape != first.shape:
            raise ValueError(f"Mismatched shapes while stacking: {path} has {image.shape}, expected {first.shape}")
        arrays.append(np.asarray(image.dataobj, dtype=np.float32))
    stacked = np.stack(arrays, axis=-1)
    return nib.Nifti1Image(stacked, affine=first.affine, header=first.header.copy())


def voxelwise_group_summary(
    maps_4d_path: Path,
    *,
    mask: np.ndarray,
    reference_image: nib.Nifti1Image,
    out_prefix: Path,
) -> Dict[str, str]:
    image = nib.load(str(maps_4d_path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    masked_data = data[mask]

    mean_map = np.zeros(mask.shape, dtype=np.float32)
    t_map = np.zeros(mask.shape, dtype=np.float32)
    p_map = np.ones(mask.shape, dtype=np.float32)
    mean_map[mask] = np.asarray(masked_data.mean(axis=1), dtype=np.float32)
    if masked_data.shape[1] >= 2:
        ttest = stats.ttest_1samp(masked_data, popmean=0.0, axis=1)
        t_map[mask] = np.asarray(np.nan_to_num(ttest.statistic, nan=0.0), dtype=np.float32)
        p_map[mask] = np.asarray(np.nan_to_num(ttest.pvalue, nan=1.0), dtype=np.float32)

    mean_path = out_prefix.with_name(out_prefix.name + "_mean.nii.gz")
    t_path = out_prefix.with_name(out_prefix.name + "_tstat.nii.gz")
    p_path = out_prefix.with_name(out_prefix.name + "_p_uncorrected.nii.gz")
    save_nifti(mean_path, mean_map, reference_image=reference_image)
    save_nifti(t_path, t_map, reference_image=reference_image)
    save_nifti(p_path, p_map, reference_image=reference_image)
    return {
        "mean_map": str(mean_path),
        "tstat_map": str(t_path),
        "p_uncorrected_map": str(p_path),
    }


def resolve_randomise_bin(path_text: str) -> Path:
    candidate = Path(path_text).expanduser()
    if candidate.exists():
        return candidate.resolve()
    resolved = shutil.which(path_text)
    if resolved is None:
        raise FileNotFoundError(f"Could not resolve randomise binary from {path_text!r}.")
    return Path(resolved).resolve()


def run_randomise(randomise_bin: Path, input_4d: Path, mask_path: Path, out_root: Path, permutations: int, seed: int) -> List[str]:
    env = dict(os.environ)
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    fsl_dir = randomise_bin.parents[3] if len(randomise_bin.parents) >= 4 else None
    if fsl_dir is not None:
        env.setdefault("FSLDIR", str(fsl_dir))
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
        "-T",
        "-x",
        f"--seed={int(seed)}",
    ]
    subprocess.run(command, check=True, env=env)
    return command


def summarise_randomise_outputs(out_root: Path, reference_image: nib.Nifti1Image) -> Dict[str, object]:
    tstat_path = out_root.with_name(out_root.name + "_tstat1.nii.gz")
    corrp_path = out_root.with_name(out_root.name + "_tfce_corrp_tstat1.nii.gz")
    uncorrp_path = out_root.with_name(out_root.name + "_tfce_p_tstat1.nii.gz")
    if not corrp_path.exists():
        available = sorted(str(path.name) for path in out_root.parent.glob(out_root.name + "*"))
        raise FileNotFoundError(f"Expected corrected p image was not produced for {out_root}. Available: {available}")

    corrp_data = np.asarray(nib.load(str(corrp_path)).dataobj, dtype=np.float32)
    sig_mask = np.asarray(corrp_data >= 0.95, dtype=np.uint8)
    sig_mask_path = out_root.with_name(out_root.name + "_tfce_corrp_tstat1_sigmask_p05.nii.gz")
    save_nifti(sig_mask_path, sig_mask, reference_image=reference_image)

    structure = generate_binary_structure(rank=3, connectivity=1)
    labeled, n_clusters = label(sig_mask > 0, structure=structure)
    cluster_sizes = [int(np.sum(labeled == cluster_index)) for cluster_index in range(1, int(n_clusters) + 1)]
    cluster_sizes.sort(reverse=True)

    return {
        "tstat_path": str(tstat_path) if tstat_path.exists() else None,
        "tfce_corrp_path": str(corrp_path),
        "tfce_uncorrp_path": str(uncorrp_path) if uncorrp_path.exists() else None,
        "significant_mask_path": str(sig_mask_path),
        "n_significant_voxels": int(sig_mask.sum()),
        "n_significant_clusters_6conn": int(n_clusters),
        "largest_cluster_size_voxels": int(cluster_sizes[0]) if cluster_sizes else 0,
        "cluster_sizes_voxels_desc": cluster_sizes[:20],
        "max_tfce_corrp_1minus_p": float(np.max(corrp_data)),
    }


def copy_workbench_map(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return str(dst)


def materialize_workbench_maps(
    *,
    run_dir: Path,
    tag: str,
    group_inputs: Dict[str, str],
    voxelwise_summary: Dict[str, Dict[str, str]],
    randomise_outputs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    wb_dir = ensure_dir(run_dir / "workbench_maps")
    manifest: Dict[str, Any] = {
        "directory": str(wb_dir),
        "note": "Volume NIfTI maps are named for later Connectome Workbench display or volume-to-surface mapping.",
        "maps": {},
    }

    map_specs = {
        "accuracy_4d": (group_inputs["searchlight_accuracy_4d"], f"{tag}_accuracy_4d.nii.gz"),
        "balanced_accuracy_4d": (group_inputs["searchlight_balanced_accuracy_4d"], f"{tag}_balancedAcc_4d.nii.gz"),
        "accuracy_minus_chance_4d": (
            group_inputs["searchlight_accuracy_minus_chance_4d"],
            f"{tag}_accuracyMinusChance_4d.nii.gz",
        ),
        "balanced_accuracy_minus_chance_4d": (
            group_inputs["searchlight_balanced_accuracy_minus_chance_4d"],
            f"{tag}_balancedAccMinusChance_4d.nii.gz",
        ),
    }
    for key, (src_text, dst_name) in map_specs.items():
        manifest["maps"][key] = copy_workbench_map(Path(src_text), wb_dir / dst_name)

    for metric_key, suffix in {
        "accuracy_minus_chance": "accuracyMinusChance",
        "balanced_accuracy_minus_chance": "balancedAccMinusChance",
    }.items():
        summary = voxelwise_summary[metric_key]
        for stat_key, stat_suffix in {
            "mean_map": "mean",
            "tstat_map": "tstat",
            "p_uncorrected_map": "pUncorrected",
        }.items():
            manifest["maps"][f"{metric_key}_{stat_key}"] = copy_workbench_map(
                Path(summary[stat_key]),
                wb_dir / f"{tag}_{suffix}_{stat_suffix}.nii.gz",
            )

    if randomise_outputs:
        manifest["randomise_maps"] = {}
        for metric_key, outputs in randomise_outputs.items():
            for output_key in ("tstat_path", "tfce_corrp_path", "tfce_uncorrp_path", "significant_mask_path"):
                src_text = outputs.get(output_key)
                if not src_text:
                    continue
                src_path = Path(src_text)
                if not src_path.exists():
                    continue
                manifest["randomise_maps"][f"{metric_key}_{output_key}"] = copy_workbench_map(
                    src_path,
                    wb_dir / f"{tag}_{metric_key}_{output_key}.nii.gz",
                )

    save_json(wb_dir / "workbench_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    set_seed(config.seed)
    run_dir = make_run_dir(config)
    tag = analysis_tag(config)

    records, dataset_summary = discover_samples(
        beta_index_csv=config.beta_index_csv,
        models_root=config.models_root,
        brain_mask_root=config.brain_mask_root,
        path_prefix_from=config.path_prefix_from,
        path_prefix_to=config.path_prefix_to,
    )
    feature_mask, reference_image = load_mask(config.feature_mask)
    if config.center_mask is None:
        center_mask = feature_mask.copy()
        center_mask_source = config.feature_mask
    else:
        center_mask, center_mask_image = load_mask(config.center_mask)
        if center_mask.shape != feature_mask.shape:
            raise ValueError(
                f"Center mask shape {center_mask.shape} does not match feature mask shape {feature_mask.shape}."
            )
        center_mask = np.asarray(center_mask & feature_mask, dtype=bool)
        reference_image = center_mask_image
        center_mask_source = config.center_mask

    center_coords = build_center_coordinates(
        center_mask,
        center_step=config.center_step,
        max_centers=config.max_centers,
    )
    feature_index_map = build_feature_index_map(feature_mask)
    sphere_offsets = build_sphere_offsets(config.radius_mm, reference_image.header.get_zooms()[:3])

    save_json(run_dir / "run_config.json", config_to_dict(config))
    save_json(run_dir / "dataset_summary.json", dataset_summary)
    save_nifti(run_dir / "feature_mask.nii.gz", feature_mask.astype(np.uint8), reference_image=reference_image)
    save_nifti(run_dir / "center_mask.nii.gz", center_mask.astype(np.uint8), reference_image=reference_image)

    subject_records = group_records_by_subject(records)
    excluded_subjects: List[Dict[str, Any]] = []
    included_subject_ids: List[str] = []
    only_subjects_set = set(config.only_subjects)
    exclude_subjects_set = set(config.exclude_subjects)
    for subject_id in sorted(subject_records):
        run_keys = sorted({(record.session_id, record.run_id) for record in subject_records[subject_id]})
        if config.cv_group == "session":
            cv_keys = sorted({(record.session_id,) for record in subject_records[subject_id]})
        elif config.cv_group == "run":
            cv_keys = run_keys
        else:
            raise ValueError(f"Unsupported cv_group={config.cv_group!r}.")
        exclude_reason: str | None = None
        if only_subjects_set and subject_id not in only_subjects_set:
            exclude_reason = "not_in_only_subjects_filter"
        elif subject_id in exclude_subjects_set:
            exclude_reason = "manually_excluded"
        elif len(cv_keys) != int(config.required_num_runs):
            exclude_reason = f"num_{config.cv_group}_groups_{len(cv_keys)}_!=_{config.required_num_runs}"
        if exclude_reason is not None:
            excluded_subjects.append(
                {
                    "subject_id": subject_id,
                    "reason": exclude_reason,
                    "num_runs": int(len(run_keys)),
                    "num_cv_groups": int(len(cv_keys)),
                    "cv_group_keys": [":".join(key) for key in cv_keys],
                    "run_keys": [f"{session_id}:{run_id}" for session_id, run_id in run_keys],
                }
            )
            continue
        included_subject_ids.append(subject_id)

    save_json(
        run_dir / "subject_selection.json",
        {
            "included_subject_ids": included_subject_ids,
            "excluded_subjects": excluded_subjects,
            "cv_group": config.cv_group,
            "required_num_cv_groups": int(config.required_num_runs),
            "required_num_runs": int(config.required_num_runs),
            "only_subjects": config.only_subjects,
            "explicitly_excluded_subjects": config.exclude_subjects,
        },
    )

    subject_summaries: List[Dict[str, Any]] = []
    for subject_id in included_subject_ids:
        subject_summaries.append(
            subject_searchlight(
                subject_id=subject_id,
                records=subject_records[subject_id],
                feature_mask=feature_mask,
                center_coords=center_coords,
                sphere_offsets=sphere_offsets,
                feature_index_map=feature_index_map,
                reference_image=reference_image,
                config=config,
                run_dir=run_dir,
            )
        )

    if not subject_summaries:
        raise RuntimeError("No subjects were included in the searchlight analysis.")

    group_dir = ensure_dir(run_dir / "group_inputs")
    accuracy_paths = [Path(summary["accuracy_map"]) for summary in subject_summaries]
    bal_paths = [Path(summary["balanced_accuracy_map"]) for summary in subject_summaries]
    accuracy_centered_paths = [Path(summary["accuracy_minus_chance_map"]) for summary in subject_summaries]
    bal_centered_paths = [Path(summary["balanced_accuracy_minus_chance_map"]) for summary in subject_summaries]

    accuracy_4d_path = group_dir / "searchlight_accuracy_4d.nii.gz"
    bal_4d_path = group_dir / "searchlight_balanced_accuracy_4d.nii.gz"
    accuracy_centered_4d_path = group_dir / "searchlight_accuracy_minus_chance_4d.nii.gz"
    bal_centered_4d_path = group_dir / "searchlight_balanced_accuracy_minus_chance_4d.nii.gz"
    nib.save(stack_niftis(accuracy_paths), str(accuracy_4d_path))
    nib.save(stack_niftis(bal_paths), str(bal_4d_path))
    nib.save(stack_niftis(accuracy_centered_paths), str(accuracy_centered_4d_path))
    nib.save(stack_niftis(bal_centered_paths), str(bal_centered_4d_path))

    voxelwise_summary = {
        "accuracy_minus_chance": voxelwise_group_summary(
            accuracy_centered_4d_path,
            mask=center_mask,
            reference_image=reference_image,
            out_prefix=group_dir / "searchlight_accuracy_minus_chance",
        ),
        "balanced_accuracy_minus_chance": voxelwise_group_summary(
            bal_centered_4d_path,
            mask=center_mask,
            reference_image=reference_image,
            out_prefix=group_dir / "searchlight_balanced_accuracy_minus_chance",
        ),
    }

    group_summary: Dict[str, Any] = {
        "analysis_tag": tag,
        "cv_group": config.cv_group,
        "required_num_cv_groups": int(config.required_num_runs),
        "num_subjects_included": int(len(subject_summaries)),
        "subjects": subject_summaries,
        "excluded_subjects": excluded_subjects,
        "num_centers": int(len(center_coords)),
        "center_mask_source": str(center_mask_source),
        "feature_mask_source": str(config.feature_mask),
        "radius_mm": float(config.radius_mm),
        "center_step": int(config.center_step),
        "max_centers": config.max_centers,
        "mean_subject_searchlight_accuracy": {
            "mean": float(np.mean([summary["mean_searchlight_accuracy"] for summary in subject_summaries])),
            "std": float(np.std([summary["mean_searchlight_accuracy"] for summary in subject_summaries], ddof=0)),
        },
        "mean_subject_searchlight_balanced_accuracy": {
            "mean": float(np.mean([summary["mean_searchlight_balanced_accuracy"] for summary in subject_summaries])),
            "std": float(
                np.std([summary["mean_searchlight_balanced_accuracy"] for summary in subject_summaries], ddof=0)
            ),
        },
        "voxelwise_summary": voxelwise_summary,
        "group_inputs": {
            "searchlight_accuracy_4d": str(accuracy_4d_path),
            "searchlight_balanced_accuracy_4d": str(bal_4d_path),
            "searchlight_accuracy_minus_chance_4d": str(accuracy_centered_4d_path),
            "searchlight_balanced_accuracy_minus_chance_4d": str(bal_centered_4d_path),
        },
    }

    randomise_outputs: Dict[str, Any] | None = None
    if config.run_randomise:
        randomise_dir = ensure_dir(run_dir / "randomise")
        randomise_bin = resolve_randomise_bin(config.randomise_bin)
        randomise_outputs = {}
        for name, input_4d in {
            "accuracy_minus_chance": accuracy_centered_4d_path,
            "balanced_accuracy_minus_chance": bal_centered_4d_path,
        }.items():
            out_root = randomise_dir / f"randomise_{name}"
            command = run_randomise(
                randomise_bin=randomise_bin,
                input_4d=input_4d,
                mask_path=run_dir / "center_mask.nii.gz",
                out_root=out_root,
                permutations=config.permutations,
                seed=config.seed,
            )
            result_summary = summarise_randomise_outputs(out_root, reference_image=reference_image)
            result_summary["command"] = command
            randomise_outputs[name] = result_summary
        group_summary["randomise"] = randomise_outputs

    group_summary["workbench_maps"] = materialize_workbench_maps(
        run_dir=run_dir,
        tag=tag,
        group_inputs=group_summary["group_inputs"],
        voxelwise_summary=voxelwise_summary,
        randomise_outputs=randomise_outputs,
    )

    save_json(run_dir / "group_summary.json", group_summary)
    print(f"run_dir: {run_dir}")
    print(
        "group summary | "
        f"subjects {group_summary['num_subjects_included']} | "
        f"centers {group_summary['num_centers']} | "
        f"mean_subject_acc {group_summary['mean_subject_searchlight_accuracy']['mean']:.4f} | "
        f"mean_subject_bal_acc {group_summary['mean_subject_searchlight_balanced_accuracy']['mean']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
