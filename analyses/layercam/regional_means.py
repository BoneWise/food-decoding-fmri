"""Summarize mean Layer-CAM values relative to each class group-map brain mean.

The participant/class means and group mean are produced by gradcam_subject_tfce.
This final normalization is applied after group averaging, matching the archived
paper tables; it is not an average of participant-normalized ROI values.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from analyses._paths import OUTPUT_ROOT


def normalized_roi_rows(rows, whole_brain_mean):
    if not np.isfinite(whole_brain_mean) or whole_brain_mean <= 0:
        raise ValueError("The class group-map whole-brain mean must be positive and finite.")
    result = []
    for row in rows:
        result.append({
            "roi_name": row["roi_name"],
            "division": row["division"],
            "num_voxels": int(row["num_voxels"]),
            "num_subjects": int(row["num_subjects"]),
            "mean_layercam": float(row["mean_value"]),
            "whole_brain_mean": whole_brain_mean,
            "mean_layercam_whole_brain_1": float(row["mean_value"]) / whole_brain_mean,
        })
    return result


def division_rows(rows, whole_brain_mean):
    result = []
    for division in sorted({row["division"] for row in rows}):
        selected = [row for row in rows if row["division"] == division]
        count = sum(row["num_voxels"] for row in selected)
        mean = sum(row["mean_layercam"] * row["num_voxels"] for row in selected) / count
        result.append({"division": division, "num_voxels": count,
                       "mean_layercam": mean, "whole_brain_mean": whole_brain_mean,
                       "mean_layercam_whole_brain_1": mean / whole_brain_mean})
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layercam-dir", type=Path, required=True,
                        help="k=6 gradcam_subject_tfce output directory.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "regional_means")
    args = parser.parse_args()
    mask_img = nib.load(str(args.layercam_dir / "common_brain_mask.nii.gz"))
    mask = np.asanyarray(mask_img.dataobj) > 0
    if not mask.any():
        raise ValueError("Common brain mask is empty.")
    for rating in (1, 4):
        cam_img = nib.load(str(args.layercam_dir / f"rating_{rating}_group_mean_layercam.nii.gz"))
        if cam_img.shape != mask_img.shape or not np.allclose(cam_img.affine, mask_img.affine):
            raise ValueError("Group map and mask are not on the same grid.")
        cam = np.asanyarray(cam_img.dataobj).astype(np.float64)
        mean = float(np.nanmean(cam[mask]))
        with (args.layercam_dir / f"rating_{rating}_roi_subject_stats.csv").open(newline="", encoding="utf-8") as handle:
            rows = normalized_roi_rows(list(csv.DictReader(handle)), mean)
        write_csv(args.output_dir / f"rating_{rating}_roi_means.csv", rows)
        write_csv(args.output_dir / f"rating_{rating}_division_means.csv", division_rows(rows, mean))


if __name__ == "__main__":
    main()
