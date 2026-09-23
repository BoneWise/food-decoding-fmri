from __future__ import annotations

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .dataset import FMRISample
from .utils import rating_to_target


@dataclass(frozen=True)
class FoldAssignment:
    fold_index: int
    train_subjects: List[str]
    val_subjects: List[str]


def _subject_label_counts(records: Sequence[FMRISample]) -> Dict[str, np.ndarray]:
    counts: Dict[str, np.ndarray] = {}
    for record in records:
        target = rating_to_target(record.rating)
        subject_counts = counts.setdefault(record.subject_id, np.zeros(2, dtype=np.int64))
        subject_counts[target] += 1
    return counts


def build_subject_folds(
    records: Sequence[FMRISample],
    *,
    num_folds: int,
    seed: int,
) -> List[FoldAssignment]:
    if int(num_folds) <= 1:
        raise ValueError(f"num_folds must be > 1, got {num_folds}.")

    counts_by_subject = _subject_label_counts(records)
    subject_ids = sorted(counts_by_subject)
    if len(subject_ids) < int(num_folds):
        raise ValueError(
            f"Need at least {num_folds} subjects for {num_folds}-fold CV, got {len(subject_ids)}."
        )

    shuffled_subjects = list(subject_ids)
    random.Random(seed).shuffle(shuffled_subjects)

    global_counts = np.sum(np.stack([counts_by_subject[subject_id] for subject_id in subject_ids], axis=0), axis=0)
    target_counts = global_counts.astype(np.float64) / float(num_folds)
    subject_capacities = [len(subject_ids) // int(num_folds)] * int(num_folds)
    for index in range(len(subject_ids) % int(num_folds)):
        subject_capacities[index] += 1

    global_high_fraction = float(global_counts[1] / max(int(global_counts.sum()), 1))
    ordered_subjects = sorted(
        shuffled_subjects,
        key=lambda subject_id: (
            -int(counts_by_subject[subject_id].sum()),
            -abs(float(counts_by_subject[subject_id][1] / max(int(counts_by_subject[subject_id].sum()), 1)) - global_high_fraction),
            subject_id,
        ),
    )

    folds: List[Dict[str, object]] = [
        {
            "subjects": [],
            "counts": np.zeros(2, dtype=np.int64),
            "capacity": int(capacity),
        }
        for capacity in subject_capacities
    ]

    for subject_id in ordered_subjects:
        subject_counts = counts_by_subject[subject_id]
        candidate_indices = [
            fold_index
            for fold_index, fold in enumerate(folds)
            if len(fold["subjects"]) < int(fold["capacity"])
        ]
        best_fold_index = min(
            candidate_indices,
            key=lambda fold_index: (
                float(
                    np.sum(
                        ((folds[fold_index]["counts"] + subject_counts - target_counts) ** 2)
                        / np.maximum(target_counts, 1.0)
                    )
                ),
                int((folds[fold_index]["counts"] + subject_counts).sum()),
                len(folds[fold_index]["subjects"]),
                fold_index,
            ),
        )
        folds[best_fold_index]["subjects"].append(subject_id)
        folds[best_fold_index]["counts"] = folds[best_fold_index]["counts"] + subject_counts

    assignments: List[FoldAssignment] = []
    all_subjects = set(subject_ids)
    for fold_index, fold in enumerate(folds, start=1):
        val_subjects = sorted(str(subject_id) for subject_id in fold["subjects"])
        train_subjects = sorted(all_subjects - set(val_subjects))
        assignments.append(
            FoldAssignment(
                fold_index=fold_index,
                train_subjects=train_subjects,
                val_subjects=val_subjects,
            )
        )
    return assignments


def load_subject_folds(path: str | Path, records: Sequence[FMRISample]) -> List[FoldAssignment]:
    """Load frozen folds and reject subject leakage or incomplete held-out coverage."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = sorted(payload["folds"], key=lambda item: int(item["fold_index"]))
    if len(raw) != int(payload.get("num_folds", len(raw))):
        raise ValueError("Fold count does not match num_folds.")
    if len({int(fold["fold_index"]) for fold in raw}) != len(raw):
        raise ValueError("Fold indices must be unique.")
    subjects = {record.subject_id for record in records}
    seen: list[str] = []
    result: List[FoldAssignment] = []
    for fold in raw:
        train = list(map(str, fold["train_subjects"]))
        val = list(map(str, fold["val_subjects"]))
        if len(train) != len(set(train)) or len(val) != len(set(val)):
            raise ValueError("Fold subject lists contain duplicate subjects.")
        if set(train) & set(val):
            raise ValueError(f"Fold {fold['fold_index']} has train/validation subject overlap.")
        if not train or not val or set(train) | set(val) != subjects:
            raise ValueError(f"Fold {fold['fold_index']} does not cover the dataset subjects exactly.")
        seen.extend(val)
        result.append(FoldAssignment(int(fold["fold_index"]), sorted(train), sorted(val)))
    if sorted(seen) != sorted(subjects):
        raise ValueError("Every subject must occur in exactly one held-out fold.")
    return result
