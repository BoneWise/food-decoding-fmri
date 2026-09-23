from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import nibabel as nib
import numpy as np

from .config import DEFAULT_BETA_INDEX_CSV, DEFAULT_BRAIN_MASK_ROOT, DEFAULT_MODELS_ROOT
from .dataset import FMRISample, discover_samples
from .utils import ensure_dir
from .cli import parse_with_config


DEFAULT_KS = "1,2,3,4,5,6"
DEFAULT_SEED = 42
DEFAULT_EXCLUDE_SUBJECTS = "sub-016"


@dataclass(frozen=True)
class MergeGroup:
    subject_id: str
    rating: int
    group_index: int
    records: List[FMRISample]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create higher-SNR rating 1/4 datasets by averaging non-overlapping random "
            "groups of k beta maps within each subject and rating."
        )
    )
    parser.add_argument("--beta-index-csv", default=DEFAULT_BETA_INDEX_CSV)
    parser.add_argument("--models-root", default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--brain-mask-root", default=DEFAULT_BRAIN_MASK_ROOT)
    parser.add_argument("--path-prefix-from", default="")
    parser.add_argument("--path-prefix-to", default="")
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Root directory for generated k datasets. Defaults to "
            "<beta-index parent>/merged_rating14_within_subject."
        ),
    )
    parser.add_argument("--ks", default=DEFAULT_KS, help="Comma-separated k values. Default: 1,2,3,4,5,6.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--exclude-subjects",
        default=DEFAULT_EXCLUDE_SUBJECTS,
        help="Comma-separated subjects to exclude before merging. Default: sub-016.",
    )
    parser.add_argument(
        "--copy-k1",
        action="store_true",
        help="Materialize k=1 beta maps instead of referencing the original beta files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing merged NIfTI files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute and print merge counts; do not write output files.",
    )
    return parser


def parse_int_list(value: str) -> List[int]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("At least one k value is required.")
    values = sorted({int(item) for item in items})
    if any(value <= 0 for value in values):
        raise ValueError(f"All k values must be positive, got {values}.")
    return values


def parse_subject_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def default_output_root(beta_index_csv: str) -> Path:
    return Path(beta_index_csv).expanduser().resolve().parent / "merged_rating14_within_subject"


def group_records(records: Sequence[FMRISample]) -> Dict[tuple[str, int], List[FMRISample]]:
    grouped: Dict[tuple[str, int], List[FMRISample]] = defaultdict(list)
    for record in records:
        grouped[(record.subject_id, int(record.rating))].append(record)
    for key in list(grouped):
        grouped[key] = sorted(
            grouped[key],
            key=lambda record: (
                record.session_id,
                record.run_id,
                -1 if record.trial_id is None else int(record.trial_id),
                record.sample_path,
            ),
        )
    return grouped


def build_merge_groups(
    records: Sequence[FMRISample],
    *,
    k: int,
    seed: int,
) -> List[MergeGroup]:
    rng = np.random.default_rng(int(seed) + int(k) * 100_003)
    groups: List[MergeGroup] = []
    grouped = group_records(records)
    for subject_id, rating in sorted(grouped):
        subject_rating_records = list(grouped[(subject_id, rating)])
        order = rng.permutation(len(subject_rating_records))
        shuffled = [subject_rating_records[int(index)] for index in order]
        num_groups = len(shuffled) // int(k)
        for group_index in range(num_groups):
            start = group_index * int(k)
            groups.append(
                MergeGroup(
                    subject_id=subject_id,
                    rating=int(rating),
                    group_index=int(group_index + 1),
                    records=shuffled[start : start + int(k)],
                )
            )
    return groups


def load_mean_beta(records: Sequence[FMRISample]) -> tuple[np.ndarray, nib.Nifti1Image]:
    if not records:
        raise ValueError("Cannot average an empty merge group.")
    first_img = nib.load(str(Path(records[0].sample_path).expanduser().resolve()))
    arrays = []
    for record in records:
        img = nib.load(str(Path(record.sample_path).expanduser().resolve()))
        if tuple(img.shape) != tuple(first_img.shape):
            raise ValueError(
                f"Shape mismatch while merging {record.sample_path}: {img.shape} vs {first_img.shape}."
            )
        arrays.append(np.asarray(img.dataobj, dtype=np.float32))
    stack = np.stack(arrays, axis=0)
    with np.errstate(invalid="ignore"):
        mean_beta = np.nanmean(stack, axis=0, dtype=np.float32)
    mean_beta = np.nan_to_num(mean_beta, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return mean_beta, first_img


def write_nifti(path: Path, data: np.ndarray, reference_img: nib.spatialimages.SpatialImage) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(
        np.asarray(data, dtype=np.float32),
        affine=reference_img.affine,
        header=reference_img.header.copy(),
    )
    image.set_data_dtype(np.float32)
    nib.save(image, str(path))


def source_values(records: Sequence[FMRISample], attr: str) -> str:
    values = [getattr(record, attr) for record in records]
    return "|".join("" if value is None else str(value) for value in values)


def build_output_row(
    *,
    group: MergeGroup,
    k: int,
    seed: int,
    beta_path: Path,
) -> Dict[str, Any]:
    first = group.records[0]
    return {
        "sub": group.subject_id,
        "session": "ses-merged",
        "run": "run-merged",
        "trial_in_run": int(group.group_index),
        "rating_value": int(group.rating),
        "beta_file": str(beta_path.resolve()),
        "mask_file": first.mask_path,
        "merge_k": int(k),
        "merge_seed": int(seed),
        "merge_group_index": int(group.group_index),
        "source_trial_ids": source_values(group.records, "trial_id"),
        "source_session_ids": source_values(group.records, "session_id"),
        "source_run_ids": source_values(group.records, "run_id"),
        "source_beta_files": source_values(group.records, "sample_path"),
    }


def write_csv_rows(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def summarize_input_counts(records: Sequence[FMRISample]) -> Dict[str, Any]:
    counts: Dict[str, Dict[str, int]] = {}
    for record in records:
        subject_counts = counts.setdefault(record.subject_id, {"rating_1": 0, "rating_4": 0})
        subject_counts[f"rating_{int(record.rating)}"] += 1
    return {
        "num_subjects": int(len(counts)),
        "num_records": int(len(records)),
        "subjects": counts,
    }


def summarize_groups(groups: Sequence[MergeGroup], *, k: int) -> Dict[str, Any]:
    by_subject: Dict[str, Dict[str, int]] = {}
    for group in groups:
        subject_counts = by_subject.setdefault(group.subject_id, {"rating_1": 0, "rating_4": 0})
        subject_counts[f"rating_{int(group.rating)}"] += 1
    return {
        "k": int(k),
        "num_subjects": int(len(by_subject)),
        "num_merged_maps": int(len(groups)),
        "merged_rating_1": int(sum(1 for group in groups if int(group.rating) == 1)),
        "merged_rating_4": int(sum(1 for group in groups if int(group.rating) == 4)),
        "merged_maps_by_subject": by_subject,
    }


def subjects_missing_complete_classes(
    *,
    input_subjects: Iterable[str],
    group_summary: Dict[str, Any],
) -> List[str]:
    by_subject = group_summary["merged_maps_by_subject"]
    missing: List[str] = []
    for subject_id in sorted(str(subject_id) for subject_id in input_subjects):
        subject_counts = by_subject.get(subject_id)
        if subject_counts is None:
            missing.append(subject_id)
            continue
        if int(subject_counts.get("rating_1", 0)) <= 0 or int(subject_counts.get("rating_4", 0)) <= 0:
            missing.append(subject_id)
    return missing


def materialize_k_dataset(
    *,
    groups: Sequence[MergeGroup],
    k: int,
    seed: int,
    output_dir: Path,
    copy_k1: bool,
    overwrite: bool,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    fieldnames = [
        "sub",
        "session",
        "run",
        "trial_in_run",
        "rating_value",
        "beta_file",
        "mask_file",
        "merge_k",
        "merge_seed",
        "merge_group_index",
        "source_trial_ids",
        "source_session_ids",
        "source_run_ids",
        "source_beta_files",
    ]

    for group in groups:
        if int(k) == 1 and not copy_k1:
            beta_path = Path(group.records[0].sample_path).expanduser().resolve()
        else:
            beta_path = (
                output_dir
                / "models"
                / group.subject_id
                / f"rating-{int(group.rating)}"
                / f"merged_k{int(k):02d}_seed{int(seed)}_group{int(group.group_index):04d}.nii.gz"
            )
            if overwrite or not beta_path.exists():
                mean_beta, reference_img = load_mean_beta(group.records)
                write_nifti(beta_path, mean_beta, reference_img)
        rows.append(build_output_row(group=group, k=k, seed=seed, beta_path=beta_path))

    write_csv_rows(output_dir / "beta_index.csv", rows, fieldnames)
    write_csv_rows(output_dir / "merge_manifest.csv", rows, fieldnames)
    summary = summarize_groups(groups, k=k)
    summary.update(
        {
            "seed": int(seed),
            "output_dir": str(output_dir.resolve()),
            "beta_index_csv": str((output_dir / "beta_index.csv").resolve()),
            "merge_manifest_csv": str((output_dir / "merge_manifest.csv").resolve()),
            "k1_materialized": bool(copy_k1) if int(k) == 1 else None,
            "averaging_rule": "non-overlapping random groups within subject and rating",
            "leftover_rule": "records not filling a complete group of k are dropped",
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_with_config(build_parser(), argv)
    ks = parse_int_list(args.ks)
    exclude_subjects = set(parse_subject_list(args.exclude_subjects))
    output_root = ensure_dir(args.output_root or default_output_root(args.beta_index_csv))

    records, dataset_summary = discover_samples(
        beta_index_csv=args.beta_index_csv,
        models_root=args.models_root,
        brain_mask_root=args.brain_mask_root,
        path_prefix_from=args.path_prefix_from,
        path_prefix_to=args.path_prefix_to,
    )
    records = [record for record in records if record.subject_id not in exclude_subjects]
    input_summary = summarize_input_counts(records)
    if input_summary["num_subjects"] <= 0:
        raise ValueError("No subjects remain after exclusions.")

    overall = {
        "source": {
            "beta_index_csv": str(Path(args.beta_index_csv).expanduser().resolve()),
            "models_root": args.models_root,
            "brain_mask_root": args.brain_mask_root,
            "dataset_summary": dataset_summary,
        },
        "excluded_subjects": sorted(exclude_subjects),
        "input_after_exclusion": input_summary,
        "seed": int(args.seed),
        "ks": ks,
        "datasets": {},
    }

    for k in ks:
        groups = build_merge_groups(records, k=k, seed=int(args.seed))
        dry_summary = summarize_groups(groups, k=k)
        missing_subjects = subjects_missing_complete_classes(
            input_subjects=input_summary["subjects"],
            group_summary=dry_summary,
        )
        if missing_subjects:
            raise ValueError(
                f"k={k} would leave subjects without both rating classes after merging: {missing_subjects}. "
                "Use a smaller k or exclude those subjects explicitly."
            )
        dry_summary["seed"] = int(args.seed)
        dry_summary["output_dir"] = str((output_root / f"k{int(k):02d}_seed{int(args.seed)}").resolve())
        if args.dry_run:
            summary = dry_summary
        else:
            k_dir = ensure_dir(output_root / f"k{int(k):02d}_seed{int(args.seed)}")
            summary = materialize_k_dataset(
                groups=groups,
                k=k,
                seed=int(args.seed),
                output_dir=k_dir,
                copy_k1=bool(args.copy_k1),
                overwrite=bool(args.overwrite),
            )
        overall["datasets"][str(k)] = summary
        print(
            f"k={int(k)} | merged_maps={summary['num_merged_maps']} | "
            f"rating1={summary['merged_rating_1']} rating4={summary['merged_rating_4']} | "
            f"subjects={summary['num_subjects']} | output={summary['output_dir']}",
            flush=True,
        )

    if not args.dry_run:
        (output_root / "overall_summary.json").write_text(
            json.dumps(overall, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"overall_summary: {output_root / 'overall_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
