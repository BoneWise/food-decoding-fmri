from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from analyses._paths import DATA_ROOT, RUNS_ROOT, OUTPUT_ROOT, FSL_ROOT, ATLAS_ROOT
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import generate_binary_structure, label


DEFAULT_BETA_INDEX_CSV = str(DATA_ROOT / "derived" / "beta_index.csv")
DEFAULT_MASK_ROOT = str(DATA_ROOT / "derived" / "cache" / "masks")
DEFAULT_FSL_RANDOMISE = str(FSL_ROOT / "bin" / "randomise")


@dataclass(frozen=True)
class TrialRecord:
    subject_id: str
    session_id: str
    run_id: str
    rating: int
    beta_path: Path
    quality_score: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build subject-level rating 1>4 / 4>1 contrast maps from single-trial beta maps "
            "and run group-level FSL randomise with TFCE FWE correction."
        )
    )
    parser.add_argument("--beta-index-csv", default=DEFAULT_BETA_INDEX_CSV)
    parser.add_argument("--mask-root", default=DEFAULT_MASK_ROOT)
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT / "glm"))
    parser.add_argument("--min-quality-score", type=float, default=3.0)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subject-min-valid-units", type=int, default=1)
    parser.add_argument("--randomise-bin", default=DEFAULT_FSL_RANDOMISE)
    parser.add_argument("--skip-randomise", action="store_true")
    return parser


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser().resolve()
    beta_root = Path(args.beta_index_csv).expanduser().resolve().parent
    score_text = str(args.min_quality_score).replace(".", "p")
    return (beta_root / f"group_randomise_rating14_q{score_text}").resolve()


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_trial_records(beta_index_csv: Path, min_quality_score: float) -> tuple[List[TrialRecord], Dict[str, object]]:
    kept: List[TrialRecord] = []
    drop_counts = Counter()
    all_subjects: set[str] = set()

    with beta_index_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = str(row.get("sub", "")).strip()
            if subject_id:
                all_subjects.add(subject_id)

            rating_text = str(row.get("rating_value", row.get("label_value", ""))).strip()
            if rating_text not in {"1", "4"}:
                drop_counts["non_target_rating"] += 1
                continue

            if str(row.get("estimated_successfully", "")).strip() != "1":
                drop_counts["failed_estimation"] += 1
                continue

            quality_text = str(row.get("quality_score", "")).strip()
            try:
                quality_score = float(quality_text)
            except ValueError:
                drop_counts["missing_or_invalid_quality_score"] += 1
                continue
            if quality_score < float(min_quality_score):
                drop_counts["below_min_quality_score"] += 1
                continue

            beta_path = Path(str(row.get("beta_file", "")).strip()).expanduser().resolve()
            if not beta_path.exists():
                drop_counts["missing_beta_file"] += 1
                continue

            kept.append(
                TrialRecord(
                    subject_id=subject_id,
                    session_id=str(row.get("session", "")).strip(),
                    run_id=str(row.get("run", "")).strip(),
                    rating=int(rating_text),
                    beta_path=beta_path,
                    quality_score=quality_score,
                )
            )

    summary = {
        "beta_index_csv": str(beta_index_csv),
        "min_quality_score": float(min_quality_score),
        "rows_kept": int(len(kept)),
        "subjects_seen": int(len(all_subjects)),
        "subjects_kept_any_rating14": int(len({record.subject_id for record in kept})),
        "drop_counts": dict(drop_counts),
    }
    return kept, summary


def subject_mask_path(mask_root: Path, subject_id: str) -> Path:
    candidates = sorted(mask_root.glob(f"{subject_id}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii*"))
    usable = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if not usable:
        raise FileNotFoundError(f"Could not find mask for {subject_id} under {mask_root}.")
    if len(usable) > 1:
        raise ValueError(f"Multiple masks found for {subject_id}: {usable[:5]}")
    return usable[0]


def load_mask(mask_path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(mask_path))
    data = np.asarray(image.dataobj)
    mask = np.asarray(np.isfinite(data) & (data > 0), dtype=bool)
    if int(mask.sum()) <= 0:
        raise ValueError(f"Mask contains no positive voxels: {mask_path}")
    return mask, image


def load_beta_map(beta_path: Path, expected_shape: Tuple[int, int, int]) -> np.ndarray:
    image = nib.load(str(beta_path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if tuple(int(dim) for dim in data.shape) != tuple(int(dim) for dim in expected_shape):
        raise ValueError(f"Unexpected beta shape for {beta_path}: {data.shape} vs {expected_shape}")
    return data


def average_beta_paths(paths: Sequence[Path], brain_mask: np.ndarray) -> np.ndarray:
    if not paths:
        raise ValueError("average_beta_paths requires at least one beta map.")
    accumulator = np.zeros(brain_mask.shape, dtype=np.float64)
    for beta_path in paths:
        volume = load_beta_map(beta_path, expected_shape=brain_mask.shape)
        cleaned = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        cleaned = np.asarray(cleaned, dtype=np.float32)
        cleaned[~brain_mask] = 0.0
        accumulator += cleaned
    mean_volume = accumulator / float(len(paths))
    mean_volume[~brain_mask] = 0.0
    return np.asarray(mean_volume, dtype=np.float32)


def write_nifti(path: Path, data: np.ndarray, reference_image: nib.Nifti1Image) -> None:
    array = np.asarray(data)
    target = nib.Nifti1Image(array, affine=reference_image.affine, header=reference_image.header.copy())
    target.set_data_dtype(array.dtype)
    nib.save(target, str(path))


def build_subject_contrasts(
    records: Sequence[TrialRecord],
    *,
    mask_root: Path,
    subject_min_valid_units: int,
    subject_dir: Path,
) -> tuple[List[Dict[str, object]], Path, Path, nib.Nifti1Image]:
    grouped: Dict[Tuple[str, str, str], Dict[int, List[Path]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[(record.subject_id, record.session_id, record.run_id)][int(record.rating)].append(record.beta_path)

    grouped_subjects: Dict[str, List[Tuple[str, str, Dict[int, List[Path]]]]] = defaultdict(list)
    for (subject_id, session_id, run_id), by_rating in grouped.items():
        grouped_subjects[subject_id].append((session_id, run_id, by_rating))

    subject_summaries: List[Dict[str, object]] = []
    intersection_mask: np.ndarray | None = None
    group_reference_image: nib.Nifti1Image | None = None
    contrast_1gt4_paths: List[Path] = []
    contrast_4gt1_paths: List[Path] = []

    for subject_id in sorted(grouped_subjects):
        mask_path = subject_mask_path(mask_root, subject_id)
        brain_mask, mask_image = load_mask(mask_path)

        valid_unit_contrasts: List[np.ndarray] = []
        valid_units: List[Dict[str, object]] = []
        dropped_missing_condition = 0
        trials_rating1 = 0
        trials_rating4 = 0

        for session_id, run_id, by_rating in sorted(grouped_subjects[subject_id], key=lambda item: (item[0], item[1])):
            rating1_paths = sorted(by_rating.get(1, []))
            rating4_paths = sorted(by_rating.get(4, []))
            if not rating1_paths or not rating4_paths:
                dropped_missing_condition += 1
                continue

            mean_rating1 = average_beta_paths(rating1_paths, brain_mask=brain_mask)
            mean_rating4 = average_beta_paths(rating4_paths, brain_mask=brain_mask)
            valid_unit_contrasts.append(np.asarray(mean_rating1 - mean_rating4, dtype=np.float32))
            trials_rating1 += len(rating1_paths)
            trials_rating4 += len(rating4_paths)
            valid_units.append(
                {
                    "session_id": session_id,
                    "run_id": run_id,
                    "n_rating1_trials": int(len(rating1_paths)),
                    "n_rating4_trials": int(len(rating4_paths)),
                }
            )

        if len(valid_unit_contrasts) < int(subject_min_valid_units):
            subject_summaries.append(
                {
                    "subject_id": subject_id,
                    "included_in_group_analysis": False,
                    "mask_path": str(mask_path),
                    "n_valid_units": int(len(valid_unit_contrasts)),
                    "n_dropped_units_missing_condition": int(dropped_missing_condition),
                    "n_rating1_trials": int(trials_rating1),
                    "n_rating4_trials": int(trials_rating4),
                    "valid_units": valid_units,
                }
            )
            continue

        subject_contrast_1gt4 = np.mean(np.stack(valid_unit_contrasts, axis=0), axis=0, dtype=np.float32)
        subject_contrast_1gt4[~brain_mask] = 0.0
        subject_contrast_4gt1 = np.asarray(-subject_contrast_1gt4, dtype=np.float32)

        subject_path_1gt4 = subject_dir / f"{subject_id}_contrast_1gt4.nii.gz"
        subject_path_4gt1 = subject_dir / f"{subject_id}_contrast_4gt1.nii.gz"
        write_nifti(subject_path_1gt4, subject_contrast_1gt4, mask_image)
        write_nifti(subject_path_4gt1, subject_contrast_4gt1, mask_image)

        contrast_1gt4_paths.append(subject_path_1gt4)
        contrast_4gt1_paths.append(subject_path_4gt1)
        intersection_mask = brain_mask.copy() if intersection_mask is None else np.asarray(intersection_mask & brain_mask, dtype=bool)
        group_reference_image = mask_image

        subject_summaries.append(
            {
                "subject_id": subject_id,
                "included_in_group_analysis": True,
                "mask_path": str(mask_path),
                "n_valid_units": int(len(valid_unit_contrasts)),
                "n_dropped_units_missing_condition": int(dropped_missing_condition),
                "n_rating1_trials": int(trials_rating1),
                "n_rating4_trials": int(trials_rating4),
                "subject_contrast_1gt4": str(subject_path_1gt4),
                "subject_contrast_4gt1": str(subject_path_4gt1),
                # Average within each session+run first so the subject map is closer to
                # a run-level fixed-effect combination rather than a raw trial-count weighted mean.
                "valid_units": valid_units,
            }
        )

    if not contrast_1gt4_paths or intersection_mask is None or group_reference_image is None:
        raise ValueError("No subject-level contrast maps were created.")

    if int(intersection_mask.sum()) <= 0:
        raise ValueError("Intersection mask is empty after combining subject masks.")

    return subject_summaries, contrast_1gt4_paths, contrast_4gt1_paths, nib.Nifti1Image(
        np.asarray(intersection_mask, dtype=np.uint8),
        affine=group_reference_image.affine,
        header=group_reference_image.header.copy(),
    )


def stack_niftis(nifti_paths: Sequence[Path]) -> nib.Nifti1Image:
    if not nifti_paths:
        raise ValueError("stack_niftis requires at least one path.")
    first = nib.load(str(nifti_paths[0]))
    arrays = []
    for nifti_path in nifti_paths:
        image = nib.load(str(nifti_path))
        if image.shape != first.shape:
            raise ValueError(f"Mismatched shapes while stacking: {nifti_path} has {image.shape}, expected {first.shape}")
        arrays.append(np.asarray(image.dataobj, dtype=np.float32))
    stacked = np.stack(arrays, axis=-1)
    return nib.Nifti1Image(stacked, affine=first.affine, header=first.header.copy())


def write_subject_summary_csv(path: Path, subject_summaries: Sequence[Dict[str, object]]) -> None:
    rows: List[Dict[str, object]] = []
    for summary in subject_summaries:
        rows.append(
            {
                "subject_id": summary["subject_id"],
                "included_in_group_analysis": summary["included_in_group_analysis"],
                "n_valid_units": summary["n_valid_units"],
                "n_dropped_units_missing_condition": summary["n_dropped_units_missing_condition"],
                "n_rating1_trials": summary["n_rating1_trials"],
                "n_rating4_trials": summary["n_rating4_trials"],
                "subject_contrast_1gt4": summary.get("subject_contrast_1gt4", ""),
                "subject_contrast_4gt1": summary.get("subject_contrast_4gt1", ""),
                "mask_path": summary["mask_path"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    fsl_dir = FSL_ROOT
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


def load_optional_nifti(path: Path) -> nib.Nifti1Image | None:
    return nib.load(str(path)) if path.exists() else None


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
    write_nifti(sig_mask_path, sig_mask, reference_image)

    structure = generate_binary_structure(rank=3, connectivity=1)
    labeled, n_clusters = label(sig_mask > 0, structure=structure)
    cluster_sizes = []
    for cluster_index in range(1, int(n_clusters) + 1):
        cluster_sizes.append(int(np.sum(labeled == cluster_index)))
    cluster_sizes.sort(reverse=True)

    summary: Dict[str, object] = {
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

    tstat_image = load_optional_nifti(tstat_path)
    if tstat_image is not None:
        tstat_data = np.asarray(tstat_image.dataobj, dtype=np.float32)
        summary["max_tstat"] = float(np.max(tstat_data))
        summary["min_tstat"] = float(np.min(tstat_data))

    return summary


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()

    beta_index_csv = Path(args.beta_index_csv).expanduser().resolve()
    mask_root = Path(args.mask_root).expanduser().resolve()
    output_dir = ensure_dir(resolve_output_dir(args))
    subject_dir = ensure_dir(output_dir / "subject_maps")
    group_dir = ensure_dir(output_dir / "group_inputs")
    randomise_dir = ensure_dir(output_dir / "randomise")

    records, record_summary = load_trial_records(beta_index_csv, min_quality_score=float(args.min_quality_score))
    subject_summaries, contrast_1gt4_paths, contrast_4gt1_paths, mask_image = build_subject_contrasts(
        records,
        mask_root=mask_root,
        subject_min_valid_units=int(args.subject_min_valid_units),
        subject_dir=subject_dir,
    )

    included_subjects = [summary for summary in subject_summaries if bool(summary["included_in_group_analysis"])]
    if not included_subjects:
        raise ValueError("No subjects survived subject-level contrast creation.")

    group_mask_path = group_dir / "group_mask_intersection.nii.gz"
    nib.save(mask_image, str(group_mask_path))

    group_1gt4_4d = stack_niftis(contrast_1gt4_paths)
    group_4gt1_4d = stack_niftis(contrast_4gt1_paths)
    group_1gt4_4d_path = group_dir / "group_1gt4_4d.nii.gz"
    group_4gt1_4d_path = group_dir / "group_4gt1_4d.nii.gz"
    nib.save(group_1gt4_4d, str(group_1gt4_4d_path))
    nib.save(group_4gt1_4d, str(group_4gt1_4d_path))

    write_subject_summary_csv(output_dir / "subject_summary.csv", subject_summaries)

    summary: Dict[str, object] = {
        "analysis": "group_randomise_tfce_rating14_from_single_trial_betas",
        "beta_index_summary": record_summary,
        "assumptions": {
            "contrasts": ["1>4", "4>1"],
            "subject_level_contrast_rule": "mean over valid session+run unit contrasts; each unit contrast = mean(beta rating1) - mean(beta rating4)",
            "group_level_inference": "FSL randomise one-sample test with TFCE and FWE correction",
            "correction_threshold": "FWE p < 0.05",
            "corrp_interpretation": "FSL corrp images are 1-p, so significance mask uses >= 0.95",
            "smoothing": "none added in this script",
        },
        "paths": {
            "output_dir": str(output_dir),
            "subject_summary_csv": str(output_dir / "subject_summary.csv"),
            "group_mask_intersection": str(group_mask_path),
            "group_1gt4_4d": str(group_1gt4_4d_path),
            "group_4gt1_4d": str(group_4gt1_4d_path),
        },
        "group_summary": {
            "n_subjects_included": int(len(included_subjects)),
            "included_subject_ids": [summary["subject_id"] for summary in included_subjects],
            "subject_min_valid_units": int(args.subject_min_valid_units),
            "valid_units_per_subject": {
                summary["subject_id"]: int(summary["n_valid_units"]) for summary in included_subjects
            },
            "group_mask_voxels": int(np.asarray(mask_image.dataobj, dtype=np.uint8).sum()),
        },
        "subject_summaries": subject_summaries,
    }

    if not args.skip_randomise:
        randomise_bin = resolve_randomise_bin(args.randomise_bin)
        contrast_to_input = {
            "1gt4": group_1gt4_4d_path,
            "4gt1": group_4gt1_4d_path,
        }
        randomise_outputs: Dict[str, object] = {}
        for contrast_name, input_4d in contrast_to_input.items():
            out_root = randomise_dir / f"randomise_{contrast_name}"
            command = run_randomise(
                randomise_bin=randomise_bin,
                input_4d=input_4d,
                mask_path=group_mask_path,
                out_root=out_root,
                permutations=int(args.permutations),
                seed=int(args.seed),
            )
            result_summary = summarise_randomise_outputs(out_root, reference_image=mask_image)
            result_summary["command"] = command
            randomise_outputs[contrast_name] = result_summary
        summary["randomise"] = randomise_outputs

    save_json(output_dir / "analysis_summary.json", summary)
    print(json.dumps(summary["group_summary"], ensure_ascii=False, indent=2))
    if "randomise" in summary:
        print(json.dumps(summary["randomise"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
