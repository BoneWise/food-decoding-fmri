#!/usr/bin/env python3
"""Rebuild smoothed condition-GLM UA group TFCE-FWE results.

This script uses the existing smoothed first-level condition GLM and
subject-level fixed-effects outputs from:

    $FOODPREF_GLM_S6_ROOT (default: $FOODPREF_DATA_ROOT/derived/glm_s6_wmcsf),
    written by first_level_condition_glm_s6.m and subject_fixed_effects.m

It then creates a clean UA1 result directory, stacks subject-level contrast
maps, runs FSL randomise with TFCE-FWE correction, and writes summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
from scipy.ndimage import generate_binary_structure, label

from analyses._paths import DATA_ROOT, RUNS_ROOT, FSL_ROOT


SOURCE_ROOT = Path(os.environ.get("FOODPREF_GLM_S6_ROOT", DATA_ROOT / "derived" / "glm_s6_wmcsf"))
UA1_ROOT = RUNS_ROOT / "glm_ua1"
DEFAULT_OUTPUT_ROOT = UA1_ROOT / "RESULTS" / "smoothed_s6_wmcsf_condition_glm_tfce_fwe"
DEFAULT_RANDOMISE_BIN = FSL_ROOT / "bin" / "randomise"
DEFAULT_PERMUTATIONS = 5000
DEFAULT_SEED = 42
CONTRASTS = ("1gt4", "4gt1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--randomise-bin", type=Path, default=DEFAULT_RANDOMISE_BIN)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-randomise", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def resolve_randomise_bin(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    resolved = shutil.which(str(path))
    if resolved is None:
        raise FileNotFoundError(f"Could not resolve randomise binary from {path}")
    return Path(resolved).resolve()


def save_like(path: Path, data: np.ndarray, reference_img: nib.Nifti1Image) -> None:
    img = nib.Nifti1Image(np.asarray(data), affine=reference_img.affine, header=reference_img.header.copy())
    img.set_data_dtype(np.asarray(data).dtype)
    nib.save(img, str(path))


def copy_subject_map(source_path: Path, target_path: Path) -> nib.Nifti1Image:
    image = nib.load(str(source_path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    save_like(target_path, data, image)
    return nib.load(str(target_path))


def stack_niftis(paths: Sequence[Path]) -> nib.Nifti1Image:
    if not paths:
        raise ValueError("No NIfTI files were provided for stacking.")
    first = nib.load(str(paths[0]))
    arrays: list[np.ndarray] = []
    for path in paths:
        image = nib.load(str(path))
        if image.shape != first.shape:
            raise ValueError(f"Shape mismatch: {path} has {image.shape}, expected {first.shape}")
        arrays.append(np.asarray(image.dataobj, dtype=np.float32))
    stacked = np.stack(arrays, axis=-1)
    return nib.Nifti1Image(stacked, affine=first.affine, header=first.header.copy())


def compute_intersection_mask(mask_paths: Sequence[Path], reference_img: nib.Nifti1Image) -> np.ndarray:
    intersection: np.ndarray | None = None
    for path in mask_paths:
        if not path.exists():
            continue
        image = nib.load(str(path))
        if image.shape != reference_img.shape:
            raise ValueError(f"Mask shape mismatch: {path} has {image.shape}, expected {reference_img.shape}")
        data = np.asarray(image.dataobj)
        mask = np.asarray(np.isfinite(data) & (data > 0), dtype=bool)
        intersection = mask if intersection is None else np.asarray(intersection & mask, dtype=bool)
    if intersection is None:
        data = np.asarray(reference_img.dataobj)
        intersection = np.asarray(np.isfinite(data), dtype=bool)
    return intersection


def run_randomise(
    randomise_bin: Path,
    input_4d: Path,
    mask_path: Path,
    out_root: Path,
    permutations: int,
    seed: int,
) -> list[str]:
    env = dict(os.environ)
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    if len(randomise_bin.parents) >= 4:
        env.setdefault("FSLDIR", str(randomise_bin.parents[3]))

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
    ensure_dir(out_root.parent)
    subprocess.run(command, check=True, env=env)
    return command


def summarize_randomise(out_root: Path, reference_img: nib.Nifti1Image) -> dict[str, Any]:
    tstat_path = out_root.with_name(out_root.name + "_tstat1.nii.gz")
    tfce_corrp_path = out_root.with_name(out_root.name + "_tfce_corrp_tstat1.nii.gz")
    tfce_uncorrp_path = out_root.with_name(out_root.name + "_tfce_p_tstat1.nii.gz")
    voxel_corrp_path = out_root.with_name(out_root.name + "_vox_corrp_tstat1.nii.gz")

    if not tfce_corrp_path.exists():
        available = sorted(path.name for path in out_root.parent.glob(out_root.name + "*"))
        raise FileNotFoundError(f"Missing TFCE corrected image for {out_root}. Available: {available}")

    corrp_img = nib.load(str(tfce_corrp_path))
    corrp = np.asarray(corrp_img.dataobj, dtype=np.float32)
    sig_mask = np.asarray(corrp >= 0.95, dtype=np.uint8)

    sig_mask_path = out_root.with_name(out_root.name + "_tfce_corrp_tstat1_sigmask_p05.nii.gz")
    save_like(sig_mask_path, sig_mask, reference_img)

    tstat_img = nib.load(str(tstat_path)) if tstat_path.exists() else None
    masked_tstat_path = out_root.with_name(out_root.name + "_tfce_fwe_p05_tstat_masked.nii.gz")
    min_sig_t = None
    max_sig_t = None
    max_t = None
    min_t = None
    if tstat_img is not None:
        tstat = np.asarray(tstat_img.dataobj, dtype=np.float32)
        masked = np.where(sig_mask > 0, tstat, 0).astype(np.float32)
        save_like(masked_tstat_path, masked, tstat_img)
        finite = tstat[np.isfinite(tstat)]
        if finite.size:
            max_t = float(np.max(finite))
            min_t = float(np.min(finite))
        sig_values = tstat[sig_mask > 0]
        if sig_values.size:
            min_sig_t = float(np.min(sig_values))
            max_sig_t = float(np.max(sig_values))

    structure = generate_binary_structure(rank=3, connectivity=1)
    labeled, n_clusters = label(sig_mask > 0, structure=structure)
    cluster_sizes = [int(np.sum(labeled == idx)) for idx in range(1, int(n_clusters) + 1)]
    cluster_sizes.sort(reverse=True)

    voxel_sig_count = None
    max_voxel_corrp = None
    if voxel_corrp_path.exists():
        voxel_corrp = np.asarray(nib.load(str(voxel_corrp_path)).dataobj, dtype=np.float32)
        voxel_sig_count = int(np.count_nonzero(voxel_corrp >= 0.95))
        max_voxel_corrp = float(np.max(voxel_corrp))

    return {
        "tstat_path": str(tstat_path) if tstat_path.exists() else None,
        "tfce_corrp_path": str(tfce_corrp_path),
        "tfce_uncorrp_path": str(tfce_uncorrp_path) if tfce_uncorrp_path.exists() else None,
        "voxel_corrp_path": str(voxel_corrp_path) if voxel_corrp_path.exists() else None,
        "significant_mask_path": str(sig_mask_path),
        "tfce_fwe_p05_masked_tstat_path": str(masked_tstat_path) if tstat_img is not None else None,
        "n_significant_voxels_tfce_fwe_p05": int(sig_mask.sum()),
        "n_significant_clusters_6conn": int(n_clusters),
        "largest_cluster_size_voxels": int(cluster_sizes[0]) if cluster_sizes else 0,
        "cluster_sizes_voxels_desc_top20": cluster_sizes[:20],
        "max_tfce_corrp_1minus_p": float(np.max(corrp)),
        "threshold_rule": "TFCE-FWE p <= 0.05; FSL corrp is 1-p, so corrp >= 0.95",
        "min_tstat": min_t,
        "max_tstat": max_t,
        "min_tstat_in_sigmask": min_sig_t,
        "max_tstat_in_sigmask": max_sig_t,
        "n_significant_voxels_voxelwise_fwe_p05": voxel_sig_count,
        "max_voxel_corrp_1minus_p": max_voxel_corrp,
    }


def build_group_inputs(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    fixed_summary_csv = source_root / "second_level_fixed_effects_summary.csv"
    contrast_index_csv = source_root / "contrast_index.csv"

    if not fixed_summary_csv.exists():
        raise FileNotFoundError(f"Missing fixed-effects summary: {fixed_summary_csv}")
    if not contrast_index_csv.exists():
        raise FileNotFoundError(f"Missing first-level contrast index: {contrast_index_csv}")

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    elif output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty. Use --overwrite: {output_root}")

    ensure_dir(output_root)
    code_snapshot_dir = ensure_dir(output_root / "code_snapshot")
    input_dir = ensure_dir(output_root / "input_tables")
    subject_map_dir = ensure_dir(output_root / "subject_maps")
    group_dir = ensure_dir(output_root / "group_inputs")

    copy_if_exists(source_root / "README.md", code_snapshot_dir / "SOURCE_README.md")
    copy_if_exists(source_root / "run_minimal_lsa_s6_wmcsf.m", code_snapshot_dir / "run_minimal_lsa_s6_wmcsf.m")
    copy_if_exists(source_root / "run_subject_fixed_effects.m", code_snapshot_dir / "run_subject_fixed_effects.m")
    copy_if_exists(source_root / "run_group_randomise_from_fixed_effects.py", code_snapshot_dir / "source_run_group_randomise_from_fixed_effects.py")
    copy_if_exists(contrast_index_csv, input_dir / "contrast_index.csv")
    copy_if_exists(fixed_summary_csv, input_dir / "second_level_fixed_effects_summary.csv")

    rows = read_csv_rows(fixed_summary_csv)
    by_contrast: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            by_contrast[row["contrast_name"]].append(row)

    manifest_rows: list[dict[str, Any]] = []
    group_summary: dict[str, Any] = {}

    for contrast in CONTRASTS:
        contrast_rows = sorted(by_contrast.get(contrast, []), key=lambda row: row["sub"])
        if not contrast_rows:
            raise ValueError(f"No usable subject fixed-effects rows for {contrast}")

        copied_subject_paths: list[Path] = []
        mask_paths: list[Path] = []
        subject_ids: list[str] = []

        for row in contrast_rows:
            sub = row["sub"]
            source_contrast = Path(row["subject_contrast_path"]).resolve()
            source_mask = Path(row["mask_path"]).resolve() if row.get("mask_path") else None
            if not source_contrast.exists():
                raise FileNotFoundError(f"Missing subject contrast for {sub} {contrast}: {source_contrast}")
            target_contrast = subject_map_dir / f"{sub}_{contrast}_subject_fixed_effect_con_0001.nii.gz"
            copy_subject_map(source_contrast, target_contrast)
            copied_subject_paths.append(target_contrast)
            subject_ids.append(sub)
            if source_mask and source_mask.exists():
                mask_paths.append(source_mask)
            manifest_rows.append(
                {
                    "contrast": contrast,
                    "sub": sub,
                    "n_run_level_images": int(row.get("n_run_level_images") or 0),
                    "source_subject_contrast_path": str(source_contrast),
                    "ua1_subject_contrast_path": str(target_contrast),
                    "source_mask_path": str(source_mask) if source_mask else "",
                }
            )

        reference = nib.load(str(copied_subject_paths[0]))
        intersection = compute_intersection_mask(mask_paths, reference)
        mask_path = group_dir / f"group_mask_intersection_{contrast}.nii.gz"
        save_like(mask_path, np.asarray(intersection, dtype=np.uint8), reference)

        stacked = stack_niftis(copied_subject_paths)
        input_4d_path = group_dir / f"group_{contrast}_4d.nii.gz"
        nib.save(stacked, str(input_4d_path))

        group_summary[contrast] = {
            "n_subjects": len(subject_ids),
            "subject_ids": subject_ids,
            "group_4d_path": str(input_4d_path),
            "group_mask_intersection_path": str(mask_path),
            "group_mask_voxels": int(np.count_nonzero(intersection)),
            "subject_map_paths": [str(path) for path in copied_subject_paths],
        }

    write_csv(output_root / "subject_map_manifest.csv", manifest_rows)
    return {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "fixed_summary_csv": str(fixed_summary_csv),
        "contrast_index_csv": str(contrast_index_csv),
        "group_summary": group_summary,
    }


def main() -> None:
    args = parse_args()
    randomise_bin = resolve_randomise_bin(args.randomise_bin)
    output_root = args.output_root.resolve()

    summary: dict[str, Any] = {
        "analysis": "smoothed_s6_wmcsf_condition_glm_subject_fixed_effects_group_randomise_tfce_fwe",
        "pipeline": [
            "6 mm FWHM smoothing of MNI-space preprocessed BOLD before first-level GLM",
            "condition-level first-level GLM with rating_1, rating_2, rating_3, rating_4 plus task/nuisance regressors",
            "run-level 1>4 and 4>1 contrast images",
            "subject-level fixed-effects one-sample model across available run-level contrasts",
            "group-level one-sample sign-flipping permutation test with TFCE-FWE correction",
        ],
        "randomise": {
            "permutations": int(args.permutations),
            "seed": int(args.seed),
            "randomise_bin": str(randomise_bin),
        },
    }

    build_summary = build_group_inputs(args)
    randomise_dir = ensure_dir(output_root / "randomise")
    summary.update(build_summary)
    summary["contrasts"] = {}

    for contrast in CONTRASTS:
        contrast_group = build_summary["group_summary"][contrast]
        input_4d = Path(contrast_group["group_4d_path"])
        mask_path = Path(contrast_group["group_mask_intersection_path"])
        out_root = randomise_dir / f"randomise_{contrast}"

        command = None
        if not args.skip_randomise:
            command = run_randomise(
                randomise_bin=randomise_bin,
                input_4d=input_4d,
                mask_path=mask_path,
                out_root=out_root,
                permutations=args.permutations,
                seed=args.seed,
            )

        if not args.skip_randomise:
            result = summarize_randomise(out_root, reference_img=nib.load(str(mask_path)))
            result["command"] = command
            result["n_subjects"] = contrast_group["n_subjects"]
            result["subject_ids"] = contrast_group["subject_ids"]
            result["input_4d"] = str(input_4d)
            result["group_mask_intersection"] = str(mask_path)
            summary["contrasts"][contrast] = result

    if summary["contrasts"]:
        rows = []
        for contrast, result in summary["contrasts"].items():
            rows.append(
                {
                    "contrast": contrast,
                    "n_subjects": result["n_subjects"],
                    "n_significant_voxels_tfce_fwe_p05": result["n_significant_voxels_tfce_fwe_p05"],
                    "n_significant_clusters_6conn": result["n_significant_clusters_6conn"],
                    "largest_cluster_size_voxels": result["largest_cluster_size_voxels"],
                    "max_tfce_corrp_1minus_p": result["max_tfce_corrp_1minus_p"],
                    "min_tstat_in_sigmask": result["min_tstat_in_sigmask"],
                    "max_tstat_in_sigmask": result["max_tstat_in_sigmask"],
                    "n_significant_voxels_voxelwise_fwe_p05": result["n_significant_voxels_voxelwise_fwe_p05"],
                    "max_voxel_corrp_1minus_p": result["max_voxel_corrp_1minus_p"],
                    "tfce_corrp_path": result["tfce_corrp_path"],
                    "significant_mask_path": result["significant_mask_path"],
                    "tfce_fwe_p05_masked_tstat_path": result["tfce_fwe_p05_masked_tstat_path"],
                }
            )
        write_csv(output_root / "randomise_summary.csv", rows)

    summary_path = output_root / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
