from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from .cli import parse_with_config


DEFAULT_OUTPUT_ROOT = "data/derived"
DEFAULT_BETA_INDEX_CSV = f"{DEFAULT_OUTPUT_ROOT}/beta_index.csv"
DEFAULT_MODELS_ROOT = f"{DEFAULT_OUTPUT_ROOT}/models"
DEFAULT_BRAIN_MASK_ROOT = f"{DEFAULT_OUTPUT_ROOT}/cache/masks"
DEFAULT_SAVE_ROOT = "outputs/cnn"
DEFAULT_MODEL_ID = "fmri_cnn1_3dcnn"
DEFAULT_SPLIT_MODE = "subject"
DEFAULT_PAIR_MODE = "within_run_sparse_canonical"
FIXED_NUM_FOLDS = 10
FIXED_HIDDEN_DIM = 128
FIXED_NUM_EPOCHS = 5
PAPER_FMRI_DROPOUT = 0.5
DEFAULT_EARLY_STOPPING_WARMUP = 2
DEFAULT_EARLY_STOPPING_PATIENCE = 3
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-3
DEFAULT_RESTORE_BEST_WEIGHTS = False


@dataclass(frozen=True)
class TrainConfig:
    beta_index_csv: str
    models_root: str | None
    brain_mask_root: str | None
    path_prefix_from: str
    path_prefix_to: str
    save_root: str
    run_name: str | None
    only_fold: int | None
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    seed: int
    device: str
    fold_assignments: str | None = None
    num_folds: int = FIXED_NUM_FOLDS
    hidden_dim: int = FIXED_HIDDEN_DIM
    dropout: float = PAPER_FMRI_DROPOUT
    early_stopping_warmup: int = DEFAULT_EARLY_STOPPING_WARMUP
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA
    restore_best_weights: bool = DEFAULT_RESTORE_BEST_WEIGHTS


def _normalize_optional_path(value: str) -> str | None:
    text = str(value).strip()
    return text or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a minimal 3D CNN on fMRI beta maps using subject-level 10-fold CV "
            "and only rating 1 vs rating 4 samples."
        )
    )
    parser.add_argument(
        "--beta-index-csv",
        default=DEFAULT_BETA_INDEX_CSV,
        help="Path to beta_index.csv or beta_index.tsv.",
    )
    parser.add_argument(
        "--models-root",
        default=DEFAULT_MODELS_ROOT,
        help="Optional root used to resolve relative beta_file paths.",
    )
    parser.add_argument(
        "--brain-mask-root",
        default=DEFAULT_BRAIN_MASK_ROOT,
        help="Optional root used when beta_index does not include mask_file.",
    )
    parser.add_argument("--path-prefix-from", default="", help="Optional source path prefix to remap.")
    parser.add_argument("--path-prefix-to", default="", help="Optional target path prefix to remap.")
    parser.add_argument("--save-root", default=DEFAULT_SAVE_ROOT, help="Directory used to save runs.")
    parser.add_argument("--run-name", default=None, help="Optional run name suffix.")
    parser.add_argument(
        "--only-fold",
        type=int,
        default=None,
        help="Run only one fold index in [1, 10]. By default all 10 folds run.",
    )
    parser.add_argument("--fold-assignments", default=None, help="Archived subject folds; otherwise derive folds from this dataset.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=FIXED_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=PAPER_FMRI_DROPOUT)
    parser.add_argument(
        "--epochs",
        type=int,
        default=FIXED_NUM_EPOCHS,
        help=(
            "Fixed number of training epochs. The held-out fold is evaluated only after "
            "this final epoch; no outer-fold early stopping is performed."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--early-stopping-warmup",
        type=int,
        default=DEFAULT_EARLY_STOPPING_WARMUP,
        help="Deprecated; retained for old command compatibility and ignored by the fixed-epoch protocol.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="Deprecated; retained for old command compatibility and ignored by the fixed-epoch protocol.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=DEFAULT_EARLY_STOPPING_MIN_DELTA,
        help="Deprecated; retained for old command compatibility and ignored by the fixed-epoch protocol.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    args = parse_with_config(build_parser(), argv)
    if args.only_fold is not None and not 1 <= int(args.only_fold) <= FIXED_NUM_FOLDS:
        raise ValueError(f"--only-fold must be within [1, {FIXED_NUM_FOLDS}], got {args.only_fold}.")
    if int(args.batch_size) <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}.")
    if int(args.hidden_dim) <= 0:
        raise ValueError(f"--hidden-dim must be > 0, got {args.hidden_dim}.")
    if not 0.0 <= float(args.dropout) < 1.0:
        raise ValueError(f"--dropout must be within [0, 1), got {args.dropout}.")
    if int(args.epochs) <= 0:
        raise ValueError(f"--epochs must be > 0, got {args.epochs}.")
    if int(args.num_workers) < 0:
        raise ValueError(f"--num-workers must be >= 0, got {args.num_workers}.")
    if int(args.early_stopping_warmup) < 0:
        raise ValueError(
            f"--early-stopping-warmup must be >= 0, got {args.early_stopping_warmup}."
        )
    if int(args.early_stopping_patience) <= 0:
        raise ValueError(
            f"--early-stopping-patience must be > 0, got {args.early_stopping_patience}."
        )
    if float(args.early_stopping_min_delta) < 0.0:
        raise ValueError(
            f"--early-stopping-min-delta must be >= 0, got {args.early_stopping_min_delta}."
        )

    return TrainConfig(
        beta_index_csv=str(Path(args.beta_index_csv).expanduser().resolve()),
        models_root=_normalize_optional_path(args.models_root),
        brain_mask_root=_normalize_optional_path(args.brain_mask_root),
        path_prefix_from=str(args.path_prefix_from),
        path_prefix_to=str(args.path_prefix_to),
        save_root=str(Path(args.save_root).expanduser().resolve()),
        run_name=None if args.run_name is None else str(args.run_name).strip() or None,
        only_fold=None if args.only_fold is None else int(args.only_fold),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        device=str(args.device),
        fold_assignments=_normalize_optional_path(args.fold_assignments or ""),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        early_stopping_warmup=int(args.early_stopping_warmup),
        early_stopping_patience=int(args.early_stopping_patience),
        early_stopping_min_delta=float(args.early_stopping_min_delta),
    )


def train_config_to_dict(config: TrainConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["default_output_root"] = DEFAULT_OUTPUT_ROOT
    payload["default_beta_index_csv"] = DEFAULT_BETA_INDEX_CSV
    payload["default_models_root"] = DEFAULT_MODELS_ROOT
    payload["default_brain_mask_root"] = DEFAULT_BRAIN_MASK_ROOT
    payload["default_model_id"] = DEFAULT_MODEL_ID
    payload["default_split_mode"] = DEFAULT_SPLIT_MODE
    payload["default_pair_mode"] = DEFAULT_PAIR_MODE
    payload["fixed_label_policy"] = {"low_rating": 1, "high_rating": 4}
    payload["fixed_num_folds"] = FIXED_NUM_FOLDS
    payload["fixed_hidden_dim"] = FIXED_HIDDEN_DIM
    payload["fixed_num_epochs"] = FIXED_NUM_EPOCHS
    payload["paper_fmri_dropout"] = PAPER_FMRI_DROPOUT
    payload["training_policy"] = {
        "epoch_count": "fixed",
        "checkpoint_selection": "final_epoch_only",
        "outer_fold_usage": "held-out subjects evaluated once after fixed training; no early stopping or checkpoint selection",
    }
    payload["default_early_stopping_warmup"] = DEFAULT_EARLY_STOPPING_WARMUP
    payload["default_early_stopping_patience"] = DEFAULT_EARLY_STOPPING_PATIENCE
    payload["default_early_stopping_min_delta"] = DEFAULT_EARLY_STOPPING_MIN_DELTA
    payload["default_restore_best_weights"] = DEFAULT_RESTORE_BEST_WEIGHTS
    return payload
