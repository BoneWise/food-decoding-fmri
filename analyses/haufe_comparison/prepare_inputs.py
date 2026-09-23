from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

from .common import PROJECT_ROOT, config, sha256_file, write_json


SNAPSHOTS = {
    "haufe_mean_map": "haufe_mean_rating4_minus_rating1.nii.gz",
    "haufe_feature_mask": "haufe_feature_mask.nii.gz",
    "layercam_rating1_map": "layercam_rating1_group_mean.nii.gz",
    "layercam_rating4_map": "layercam_rating4_group_mean.nii.gz",
    "layercam_rating1_tfce_masked_t": "layercam_rating1_tfce_fwe05_masked_t.nii.gz",
    "layercam_rating4_tfce_masked_t": "layercam_rating4_tfce_fwe05_masked_t.nii.gz",
    "layercam_common_mask": "layercam_common_mask.nii.gz",
    "layercam_analysis_summary": "layercam_analysis_summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze existing Haufe and Layer-CAM inputs.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = config()
    manifest: dict[str, object] = {
        "experiment_id": settings["experiment_id"],
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {},
    }
    for key, filename in SNAPSHOTS.items():
        source = Path(settings["sources"][key]).expanduser().resolve()
        target = PROJECT_ROOT / "inputs" / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing source {key}: {source}")
        if target.exists() and not args.force:
            if sha256_file(target) != sha256_file(source):
                raise RuntimeError(f"Existing frozen input differs: {target}; review before --force.")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        manifest["sources"][key] = {
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "frozen_path": str(target),
            "frozen_sha256": sha256_file(target),
            "size_bytes": int(source.stat().st_size),
        }
    reference = nib.load(str(PROJECT_ROOT / "inputs" / SNAPSHOTS["haufe_mean_map"]))
    expected_shape = tuple(reference.shape)
    expected_affine = reference.affine
    for key in (
        "haufe_feature_mask",
        "layercam_rating1_map",
        "layercam_rating4_map",
        "layercam_rating1_tfce_masked_t",
        "layercam_rating4_tfce_masked_t",
        "layercam_common_mask",
    ):
        image = nib.load(str(PROJECT_ROOT / "inputs" / SNAPSHOTS[key]))
        if tuple(image.shape) != expected_shape or not np.allclose(image.affine, expected_affine):
            raise ValueError(f"Input {key} does not share the Haufe image grid.")
    haufe_mask = np.asanyarray(nib.load(str(PROJECT_ROOT / "inputs" / SNAPSHOTS["haufe_feature_mask"])).dataobj) > 0
    cam_mask = np.asanyarray(nib.load(str(PROJECT_ROOT / "inputs" / SNAPSHOTS["layercam_common_mask"])).dataobj) > 0
    shared = haufe_mask & cam_mask
    manifest["image_grid"] = {
        "shape": list(expected_shape),
        "affine": expected_affine.tolist(),
        "haufe_mask_voxels": int(haufe_mask.sum()),
        "layercam_mask_voxels": int(cam_mask.sum()),
        "shared_mask_voxels": int(shared.sum()),
    }
    atlas_root = Path(settings["sources"]["harvard_oxford_atlas_root"]).resolve()
    for relative in (
        "HarvardOxford-Cortical.xml",
        "HarvardOxford-Subcortical.xml",
        "HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz",
        "HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz",
    ):
        if not (atlas_root / relative).is_file():
            raise FileNotFoundError(f"Missing Harvard-Oxford atlas artifact: {atlas_root / relative}")
    manifest["atlas_root"] = str(atlas_root)
    write_json(PROJECT_ROOT / "inputs" / "source_manifest.json", manifest)
    print(f"Prepared frozen maps on {int(shared.sum())} shared voxels.", flush=True)


if __name__ == "__main__":
    main()
