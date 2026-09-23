from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Sampler

from .config import FIXED_NUM_FOLDS, TrainConfig, parse_args, train_config_to_dict
from .cv import FoldAssignment, build_subject_folds, load_subject_folds
from .dataset import FMRIBetaDataset, discover_samples, filter_records_by_subjects
from .engine import create_grad_scaler, evaluate_one_epoch, train_one_epoch
from .model import FMRIConvNet
from .utils import append_jsonl, ensure_dir, mean_and_std, save_json, set_seed


def select_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BalancedBinaryBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        targets: Sequence[int],
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        self.targets = [int(target) for target in targets]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._iteration_index = 0

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")
        if self.batch_size % 2 != 0:
            raise ValueError(
                f"BalancedBinaryBatchSampler requires an even batch_size for strict 1:1 batches, got {batch_size}."
            )

        self.neg_indices = [index for index, target in enumerate(self.targets) if int(target) == 0]
        self.pos_indices = [index for index, target in enumerate(self.targets) if int(target) == 1]
        if not self.neg_indices or not self.pos_indices:
            raise ValueError(
                "BalancedBinaryBatchSampler requires both classes to be present in the training set. "
                f"Observed negatives={len(self.neg_indices)}, positives={len(self.pos_indices)}."
            )

        self.half_batch = self.batch_size // 2
        self.num_batches = max(
            int(math.ceil(len(self.neg_indices) / float(self.half_batch))),
            int(math.ceil(len(self.pos_indices) / float(self.half_batch))),
        )

    def __len__(self) -> int:
        return self.num_batches

    @staticmethod
    def _build_balanced_stream(
        indices: Sequence[int],
        *,
        required_count: int,
        rng: random.Random,
    ) -> List[int]:
        if required_count <= 0:
            return []
        pool = list(int(index) for index in indices)
        built: List[int] = []
        while len(built) < required_count:
            shuffled = list(pool)
            rng.shuffle(shuffled)
            built.extend(shuffled)
        return built[:required_count]

    def __iter__(self):
        rng = random.Random(self.seed + self._iteration_index)
        self._iteration_index += 1

        samples_per_class = self.num_batches * self.half_batch
        neg_stream = self._build_balanced_stream(
            self.neg_indices,
            required_count=samples_per_class,
            rng=rng,
        )
        pos_stream = self._build_balanced_stream(
            self.pos_indices,
            required_count=samples_per_class,
            rng=rng,
        )

        for batch_index in range(self.num_batches):
            start = batch_index * self.half_batch
            batch = [
                *neg_stream[start : start + self.half_batch],
                *pos_stream[start : start + self.half_batch],
            ]
            rng.shuffle(batch)
            yield batch


def build_dataloader(
    dataset: FMRIBetaDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    batch_sampler: Sampler[List[int]] | None = None,
) -> DataLoader:
    loader_kwargs = {
        "dataset": dataset,
        "num_workers": int(num_workers),
        "pin_memory": (device.type == "cuda"),
        "persistent_workers": (int(num_workers) > 0),
    }
    if batch_sampler is not None:
        loader_kwargs["batch_sampler"] = batch_sampler
    else:
        loader_kwargs["batch_size"] = int(batch_size)
        loader_kwargs["shuffle"] = bool(shuffle)
    return DataLoader(**loader_kwargs)

def make_run_dir(config: TrainConfig) -> Path:
    save_root = ensure_dir(config.save_root)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = config.run_name or "fmri_rating14_binary"
    return ensure_dir(save_root / f"{timestamp}_{run_name}")


def fold_assignment_payload(assignments: List[FoldAssignment]) -> Dict[str, object]:
    return {
        "num_folds": FIXED_NUM_FOLDS,
        "folds": [
            {
                "fold_index": assignment.fold_index,
                "train_subjects": assignment.train_subjects,
                "val_subjects": assignment.val_subjects,
            }
            for assignment in assignments
        ],
    }


def summarize_cv(fold_summaries: List[Dict[str, object]]) -> Dict[str, object]:
    reported_val_loss = mean_and_std(summary["reported_val_loss"] for summary in fold_summaries)
    reported_val_accuracy = mean_and_std(summary["reported_val_accuracy"] for summary in fold_summaries)
    reported_val_balanced_accuracy = mean_and_std(
        summary["reported_val_balanced_accuracy"] for summary in fold_summaries
    )
    reported_val_f1_high = mean_and_std(summary["reported_val_f1_high"] for summary in fold_summaries)
    peak_val_balanced_accuracy = mean_and_std(
        summary["peak_val_balanced_accuracy"] for summary in fold_summaries
    )
    return {
        "num_folds_ran": int(len(fold_summaries)),
        "selection_policy": "fixed_final_epoch_no_outer_fold_checkpoint_selection",
        "reported_epoch": sorted({int(summary["reported_epoch"]) for summary in fold_summaries}),
        "reported_val_loss": reported_val_loss,
        "reported_val_accuracy": reported_val_accuracy,
        "reported_val_balanced_accuracy": reported_val_balanced_accuracy,
        "reported_val_f1_high": reported_val_f1_high,
        "peak_val_balanced_accuracy": peak_val_balanced_accuracy,
        "final_val_loss": reported_val_loss,
        "final_val_accuracy": reported_val_accuracy,
        "final_val_balanced_accuracy": reported_val_balanced_accuracy,
        "final_val_f1_high": reported_val_f1_high,
        "best_val_accuracy": reported_val_accuracy,
        "best_val_balanced_accuracy": reported_val_balanced_accuracy,
        "best_val_f1_high": reported_val_f1_high,
        "folds": fold_summaries,
    }


def run_fold(
    *,
    config: TrainConfig,
    assignment: FoldAssignment,
    records,
    device: torch.device,
    run_dir: Path,
) -> Dict[str, object]:
    fold_dir = ensure_dir(run_dir / f"fold_{assignment.fold_index:02d}")
    train_records = filter_records_by_subjects(records, assignment.train_subjects)
    val_records = filter_records_by_subjects(records, assignment.val_subjects)
    train_dataset = FMRIBetaDataset(train_records)
    val_dataset = FMRIBetaDataset(val_records)
    train_batch_sampler = BalancedBinaryBatchSampler(
        train_dataset.targets,
        batch_size=config.batch_size,
        seed=config.seed + assignment.fold_index * 1000,
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        device=device,
        batch_sampler=train_batch_sampler,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        device=device,
    )

    model = FMRIConvNet(
        input_shape=train_dataset.input_shape,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scaler = create_grad_scaler(device)

    metrics_path = fold_dir / "metrics.jsonl"
    final_model_path = fold_dir / "final.pt"
    final_train_metrics: Dict[str, object] | None = None
    final_val_metrics: Dict[str, object] | None = None
    peak_val_balanced_accuracy = float("-inf")
    peak_val_balanced_accuracy_epoch = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        metrics_row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": None,
        }

        if epoch == config.epochs:
            val_metrics = evaluate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
            final_train_metrics = dict(train_metrics)
            final_val_metrics = dict(val_metrics)
            metrics_row["val"] = val_metrics
            peak_val_balanced_accuracy = float(val_metrics["balanced_accuracy"])
            peak_val_balanced_accuracy_epoch = int(epoch)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_shape": train_dataset.input_shape,
                    "fold_index": assignment.fold_index,
                    "epoch": int(epoch),
                    "reported_epoch": int(epoch),
                    "train_metrics": final_train_metrics,
                    "val_metrics": final_val_metrics,
                    "selection_policy": "fixed_final_epoch_no_outer_fold_checkpoint_selection",
                    "config": train_config_to_dict(config),
                },
                final_model_path,
            )

        append_jsonl(metrics_path, metrics_row)

        if final_val_metrics is None:
            print(
                f"fold {assignment.fold_index:02d} | epoch {epoch:03d}/{config.epochs:03d} | "
                f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} "
                f"bal_acc {train_metrics['balanced_accuracy']:.4f}"
            )
        else:
            print(
                f"fold {assignment.fold_index:02d} | epoch {epoch:03d}/{config.epochs:03d} | "
                f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} "
                f"bal_acc {train_metrics['balanced_accuracy']:.4f} | "
                f"heldout loss {final_val_metrics['loss']:.4f} acc {final_val_metrics['accuracy']:.4f} "
                f"bal_acc {final_val_metrics['balanced_accuracy']:.4f}"
            )

    if final_train_metrics is None or final_val_metrics is None:
        raise RuntimeError(f"Fold {assignment.fold_index} did not produce a final checkpoint.")

    fold_summary = {
        "fold_index": assignment.fold_index,
        "reported_epoch": int(config.epochs),
        "reported_val_loss": float(final_val_metrics["loss"]),
        "reported_val_accuracy": float(final_val_metrics["accuracy"]),
        "reported_val_balanced_accuracy": float(final_val_metrics["balanced_accuracy"]),
        "reported_val_f1_high": float(final_val_metrics["f1_high"]),
        "reported_train_metrics": final_train_metrics,
        "reported_val_metrics": final_val_metrics,
        "final_checkpoint": str(final_model_path.resolve()),
        "selection_policy": "fixed_final_epoch_no_outer_fold_checkpoint_selection",
        "best_epoch": int(config.epochs),
        "best_val_loss": float(final_val_metrics["loss"]),
        "best_val_metrics": final_val_metrics,
        "peak_val_balanced_accuracy": float(peak_val_balanced_accuracy),
        "peak_val_balanced_accuracy_epoch": int(peak_val_balanced_accuracy_epoch),
        "early_stopped": False,
        "epochs_completed": int(epoch),
        "early_stopping_warmup": int(config.early_stopping_warmup),
        "early_stopping_patience": int(config.early_stopping_patience),
        "early_stopping_min_delta": float(config.early_stopping_min_delta),
        "restore_best_weights": False,
        "train_subjects": assignment.train_subjects,
        "val_subjects": assignment.val_subjects,
        "num_train_samples": int(len(train_dataset)),
        "num_val_samples": int(len(val_dataset)),
        "input_shape": list(train_dataset.input_shape),
    }
    fold_summary["best_val_accuracy"] = fold_summary["reported_val_accuracy"]
    fold_summary["best_val_balanced_accuracy"] = fold_summary["reported_val_balanced_accuracy"]
    fold_summary["best_val_f1_high"] = fold_summary["reported_val_f1_high"]
    save_json(fold_dir / "summary.json", fold_summary)
    return fold_summary


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    set_seed(config.seed)
    device = select_device(config.device)
    run_dir = make_run_dir(config)

    records, dataset_summary = discover_samples(
        beta_index_csv=config.beta_index_csv,
        models_root=config.models_root,
        brain_mask_root=config.brain_mask_root,
        path_prefix_from=config.path_prefix_from,
        path_prefix_to=config.path_prefix_to,
    )
    assignments = (
        load_subject_folds(config.fold_assignments, records)
        if config.fold_assignments
        else build_subject_folds(records, num_folds=config.num_folds, seed=config.seed)
    )
    if len(assignments) != config.num_folds:
        raise ValueError(f"Expected {config.num_folds} subject folds, found {len(assignments)}.")

    save_json(run_dir / "run_config.json", train_config_to_dict(config))
    save_json(run_dir / "dataset_summary.json", dataset_summary)
    save_json(run_dir / "fold_assignments.json", fold_assignment_payload(assignments))

    fold_summaries: List[Dict[str, object]] = []
    for assignment in assignments:
        if config.only_fold is not None and assignment.fold_index != config.only_fold:
            continue
        fold_summaries.append(
            run_fold(
                config=config,
                assignment=assignment,
                records=records,
                device=device,
                run_dir=run_dir,
            )
        )

    if not fold_summaries:
        raise RuntimeError("No fold was run. Check --only-fold.")

    cv_summary = summarize_cv(fold_summaries)
    save_json(run_dir / "cv_summary.json", cv_summary)
    print(f"run_dir: {run_dir}")
    print(
        "cv summary | "
        f"acc {cv_summary['reported_val_accuracy']['mean']:.4f} +/- {cv_summary['reported_val_accuracy']['std']:.4f} | "
        f"bal_acc {cv_summary['reported_val_balanced_accuracy']['mean']:.4f} +/- {cv_summary['reported_val_balanced_accuracy']['std']:.4f}"
    )


if __name__ == "__main__":
    main()
