from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


LOW_RATING = 1
HIGH_RATING = 4
CLASS_NAMES = ("low", "high")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rating_to_target(rating: int) -> int:
    value = int(rating)
    if value == LOW_RATING:
        return 0
    if value == HIGH_RATING:
        return 1
    raise ValueError(f"Only ratings {LOW_RATING} and {HIGH_RATING} are supported, got {rating}.")


def target_to_class_name(target: int) -> str:
    value = int(target)
    if not 0 <= value < len(CLASS_NAMES):
        raise ValueError(f"target must be 0 or 1, got {target}.")
    return CLASS_NAMES[value]


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def mean_and_std(values: Iterable[float]) -> Dict[str, float]:
    values_list = [float(value) for value in values]
    if not values_list:
        return {"mean": 0.0, "std": 0.0}
    if len(values_list) == 1:
        return {"mean": values_list[0], "std": 0.0}
    return {
        "mean": float(np.mean(values_list)),
        "std": float(np.std(values_list, ddof=0)),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def compute_binary_classification_metrics(
    targets: torch.Tensor,
    preds: torch.Tensor,
) -> Dict[str, Any]:
    if targets.shape != preds.shape:
        raise ValueError(f"targets and preds must share shape, got {tuple(targets.shape)} vs {tuple(preds.shape)}.")
    if targets.ndim != 1:
        raise ValueError(f"targets and preds must be rank-1, got {targets.ndim}.")
    if targets.numel() == 0:
        raise ValueError("metrics require at least one example.")

    targets = targets.to(dtype=torch.long, device="cpu")
    preds = preds.to(dtype=torch.long, device="cpu")

    tn = int(((targets == 0) & (preds == 0)).sum().item())
    fp = int(((targets == 0) & (preds == 1)).sum().item())
    fn = int(((targets == 1) & (preds == 0)).sum().item())
    tp = int(((targets == 1) & (preds == 1)).sum().item())

    total = tn + fp + fn + tp
    accuracy = _safe_divide(tp + tn, total)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    balanced_accuracy = 0.5 * (recall + specificity)

    return {
        "num_examples": int(total),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision_high": float(precision),
        "recall_high": float(recall),
        "recall_low": float(specificity),
        "f1_high": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
