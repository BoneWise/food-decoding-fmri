from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import nibabel.processing as nib_processing
import numpy as np
import torch

from .common import (
    CODE_ROOT, CONFIG_ROOT,
    DEFAULT_ROI_CONFIG_PATH,
    PROJECT_ROOT,
    add_workspace_to_path,
    environment_payload,
    load_config,
    read_json,
    save_nifti,
    sha256_array,
    sha256_file,
    sha256_payload,
    stable_sample_id,
    write_csv,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the inputs for the approved RQ3 experiment."
    )
    parser.add_argument("--config", default=None)
    return parser


def parse_atlas_labels(xml_path: Path) -> dict[str, int]:
    root = ET.parse(xml_path).getroot()
    output: dict[str, int] = {}
    for label in root.findall(".//label"):
        name = str(label.text or "").strip()
        if name:
            output[name] = int(label.attrib["index"]) + 1
    return output


def atlas_mask(
    *,
    reference_img: nib.spatialimages.SpatialImage,
    atlas_img: nib.spatialimages.SpatialImage,
    label_values: Iterable[int],
    common_mask: np.ndarray,
) -> np.ndarray:
    resampled = nib_processing.resample_from_to(atlas_img, reference_img, order=0)
    data = np.asarray(resampled.dataobj, dtype=np.int16)
    mask = np.isin(data, np.asarray(list(label_values), dtype=np.int16))
    return np.asarray(mask & common_mask, dtype=bool)


def build_fixed_roi_masks(
    *,
    reference_img: nib.spatialimages.SpatialImage,
    common_mask: np.ndarray,
    atlas_root: Path,
    roi_config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cortical_xml = atlas_root / "HarvardOxford-Cortical.xml"
    subcortical_xml = atlas_root / "HarvardOxford-Subcortical.xml"
    cortical_img_path = (
        atlas_root
        / "HarvardOxford"
        / "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"
    )
    subcortical_img_path = (
        atlas_root
        / "HarvardOxford"
        / "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz"
    )
    required = [
        cortical_xml,
        subcortical_xml,
        cortical_img_path,
        subcortical_img_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Harvard-Oxford assets: {missing}")

    cortical_values = parse_atlas_labels(cortical_xml)
    subcortical_values = parse_atlas_labels(subcortical_xml)
    cortical_img = nib.load(str(cortical_img_path))
    subcortical_img = nib.load(str(subcortical_img_path))

    masks: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {}
    for condition in ("value", "visual"):
        definition = roi_config[condition]
        missing_cortical = sorted(
            set(definition["cortical_labels"]) - set(cortical_values)
        )
        missing_subcortical = sorted(
            set(definition["subcortical_labels"]) - set(subcortical_values)
        )
        if missing_cortical or missing_subcortical:
            raise KeyError(
                f"Unknown labels for {condition}: "
                f"cortical={missing_cortical}, subcortical={missing_subcortical}"
            )

        cortical_part = atlas_mask(
            reference_img=reference_img,
            atlas_img=cortical_img,
            label_values=[
                cortical_values[name] for name in definition["cortical_labels"]
            ],
            common_mask=common_mask,
        )
        subcortical_part = atlas_mask(
            reference_img=reference_img,
            atlas_img=subcortical_img,
            label_values=[
                subcortical_values[name]
                for name in definition["subcortical_labels"]
            ],
            common_mask=common_mask,
        )
        combined = np.asarray(cortical_part | subcortical_part, dtype=bool)
        expected_voxels = int(definition["expected_voxels"])
        if int(combined.sum()) != expected_voxels:
            raise ValueError(
                f"{condition} mask has {int(combined.sum())} voxels; "
                f"expected {expected_voxels}."
            )
        masks[condition] = combined
        details[condition] = {
            "cortical_labels": list(definition["cortical_labels"]),
            "subcortical_labels": list(definition["subcortical_labels"]),
            "cortical_label_values": {
                name: int(cortical_values[name])
                for name in definition["cortical_labels"]
            },
            "subcortical_label_values": {
                name: int(subcortical_values[name])
                for name in definition["subcortical_labels"]
            },
            "num_voxels": int(combined.sum()),
            "mask_sha256": sha256_array(combined.astype(np.uint8)),
        }

    overlap = np.asarray(masks["value"] & masks["visual"], dtype=bool)
    union = np.asarray(masks["value"] | masks["visual"], dtype=bool)
    expected_union = int(roi_config["union"]["expected_voxels"])
    if int(union.sum()) != expected_union:
        raise ValueError(
            f"union mask has {int(union.sum())} voxels; expected {expected_union}."
        )
    masks["union"] = union
    details["overlap_voxels"] = int(overlap.sum())
    details["union"] = {
        "operation": "logical_or(value, visual)",
        "num_voxels": int(union.sum()),
        "mask_sha256": sha256_array(union.astype(np.uint8)),
    }
    return masks, details


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    workspace = add_workspace_to_path(config)

    from foodpref.dataset import discover_samples
    from analyses.layercam.gradcam_subject_tfce import load_common_mask

    run_dir = Path(config["cnn_run_dir"]).expanduser().resolve()
    beta_index_path = Path(config["beta_index_csv"]).expanduser().resolve()
    models_root = Path(config["models_root"]).expanduser().resolve()
    brain_mask_root = Path(config["brain_mask_root"]).expanduser().resolve()
    atlas_root = Path(config["atlas_root"]).expanduser().resolve()
    expected = config["expected"]
    numerical_workspace_paths = {
        str(relative_path): workspace / str(relative_path)
        for relative_path in config["numerical_workspace_code_files"]
    }

    required_paths = [
        run_dir / "run_config.json",
        run_dir / "fold_assignments.json",
        beta_index_path,
        models_root,
        brain_mask_root,
        atlas_root,
        Path(config["existing_oof_predictions"]).expanduser().resolve(),
        *numerical_workspace_paths.values(),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required input assets are missing: {missing}")

    observed_beta_hash = sha256_file(beta_index_path)
    if observed_beta_hash != str(expected["beta_index_sha256"]):
        raise ValueError(
            "The k=6 beta index does not match the frozen training input: "
            f"{observed_beta_hash} != {expected['beta_index_sha256']}"
        )

    records, dataset_summary = discover_samples(
        beta_index_csv=str(beta_index_path),
        models_root=str(models_root),
        brain_mask_root=str(brain_mask_root),
        path_prefix_from=str(config["path_prefix_from"]),
        path_prefix_to=str(config["path_prefix_to"]),
    )
    rating_counts = Counter(int(record.rating) for record in records)
    observed_summary = {
        "num_samples": int(len(records)),
        "num_subjects": int(len({record.subject_id for record in records})),
        "num_rating_1": int(rating_counts[1]),
        "num_rating_4": int(rating_counts[4]),
    }
    for key, observed in observed_summary.items():
        if observed != int(expected[key]):
            raise ValueError(f"{key}: observed {observed}, expected {expected[key]}")

    fold_assignments = read_json(run_dir / "fold_assignments.json")
    folds = fold_assignments["folds"]
    if len(folds) != int(expected["num_folds"]):
        raise ValueError(
            f"Observed {len(folds)} folds; expected {expected['num_folds']}."
        )
    all_subjects = sorted({record.subject_id for record in records})
    heldout_counts: Counter[str] = Counter()
    subject_to_fold: dict[str, int] = {}
    for fold in folds:
        fold_index = int(fold["fold_index"])
        train_subjects = set(str(value) for value in fold["train_subjects"])
        test_subjects = set(str(value) for value in fold["val_subjects"])
        if train_subjects & test_subjects:
            raise ValueError(f"Fold {fold_index} has train/test subject overlap.")
        if train_subjects | test_subjects != set(all_subjects):
            raise ValueError(f"Fold {fold_index} does not partition all subjects.")
        for subject in test_subjects:
            heldout_counts[subject] += 1
            subject_to_fold[subject] = fold_index
    if any(heldout_counts[subject] != 1 for subject in all_subjects):
        raise ValueError(f"Each subject must be held out once: {dict(heldout_counts)}")

    reference_img, common_mask, common_mask_summary = load_common_mask(records)
    if list(reference_img.shape) != list(expected["input_shape"]):
        raise ValueError(
            f"Input shape {reference_img.shape} != {expected['input_shape']}."
        )
    if int(common_mask.sum()) != int(expected["common_mask_voxels"]):
        raise ValueError(
            f"Common-mask voxels {int(common_mask.sum())} != "
            f"{expected['common_mask_voxels']}."
        )
    existing_common_mask_path = (
        run_dir / "gradcam_subject_tfce" / "common_brain_mask.nii.gz"
    )
    existing_common_mask = (
        np.asarray(nib.load(str(existing_common_mask_path)).dataobj) > 0
    )
    if not np.array_equal(common_mask, existing_common_mask):
        raise ValueError("Recomputed common mask differs from the existing CAM mask.")

    roi_config = read_json(DEFAULT_ROI_CONFIG_PATH)
    fixed_masks, roi_details = build_fixed_roi_masks(
        reference_img=reference_img,
        common_mask=common_mask,
        atlas_root=atlas_root,
        roi_config=roi_config,
    )

    masks_dir = PROJECT_ROOT / "outputs" / "masks"
    save_nifti(
        masks_dir / "common_brain_mask.nii.gz",
        common_mask.astype(np.uint8),
        reference_img,
        dtype=np.uint8,
    )
    for name, mask in fixed_masks.items():
        save_nifti(
            masks_dir / f"{name}_mask.nii.gz",
            mask.astype(np.uint8),
            reference_img,
            dtype=np.uint8,
        )

    sample_rows = []
    file_hash_cache: dict[str, str] = {}

    def cached_file_hash(path_text: str) -> str:
        resolved = str(Path(path_text).expanduser().resolve())
        if resolved not in file_hash_cache:
            file_hash_cache[resolved] = sha256_file(resolved)
        return file_hash_cache[resolved]

    for index, record in enumerate(records):
        sample_path = str(Path(record.sample_path).expanduser().resolve())
        mask_path = str(Path(record.mask_path).expanduser().resolve())
        sample_id = stable_sample_id(
            record.subject_id,
            int(record.rating),
            record.trial_id,
        )
        sample_rows.append(
            {
                "sample_index": int(index),
                "sample_id": sample_id,
                "test_fold": int(subject_to_fold[record.subject_id]),
                "subject_id": record.subject_id,
                "session_id": record.session_id,
                "run_id": record.run_id,
                "trial_id": record.trial_id,
                "rating": int(record.rating),
                "sample_path": sample_path,
                "sample_size_bytes": int(Path(sample_path).stat().st_size),
                "sample_sha256": cached_file_hash(sample_path),
                "mask_path": mask_path,
                "mask_size_bytes": int(Path(mask_path).stat().st_size),
                "mask_sha256": cached_file_hash(mask_path),
            }
        )
    sample_ids = [str(row["sample_id"]) for row in sample_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Stable sample IDs are not unique.")
    sample_manifest_path = write_csv(
        PROJECT_ROOT / "inputs" / "sample_manifest.csv",
        sample_rows,
        [
            "sample_index",
            "sample_id",
            "test_fold",
            "subject_id",
            "session_id",
            "run_id",
            "trial_id",
            "rating",
            "sample_path",
            "sample_size_bytes",
            "sample_sha256",
            "mask_path",
            "mask_size_bytes",
            "mask_sha256",
        ],
    )
    frozen_fold_assignments_path = write_json(
        PROJECT_ROOT / "inputs" / "fold_assignments.json",
        fold_assignments,
    )
    frozen_config = {
        key: value
        for key, value in config.items()
        if not str(key).startswith("_")
    }
    frozen_experiment_config_path = write_json(
        PROJECT_ROOT / "inputs" / "frozen_experiment_config.json",
        frozen_config,
    )
    frozen_roi_definitions_path = write_json(
        PROJECT_ROOT / "inputs" / "frozen_roi_definitions.json",
        roi_config,
    )

    checkpoint_hashes = {}
    checkpoint_metadata = {}
    for fold_index in range(1, int(expected["num_folds"]) + 1):
        checkpoint = (
            run_dir
            / f"fold_{fold_index:02d}"
            / str(config["checkpoint_name"])
        )
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        checkpoint_hashes[f"fold_{fold_index:02d}"] = sha256_file(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu")
        checkpoint_metadata[f"fold_{fold_index:02d}"] = {
            "input_shape": [int(value) for value in payload["input_shape"]],
            "reported_epoch": int(
                payload.get("reported_epoch", payload.get("epoch", 0))
            ),
            "selection_policy": str(payload.get("selection_policy", "")),
        }

    provenance = {
        "experiment_id": config["experiment_id"],
        "config_path": config["_config_path"],
        "workspace_root": str(workspace),
        "inputs": {
            "beta_index_csv": str(beta_index_path),
            "beta_index_sha256": observed_beta_hash,
            "merge_manifest_csv": str(
                Path(config["k06_input_dir"]).expanduser().resolve()
                / "merge_manifest.csv"
            ),
            "merge_manifest_sha256": sha256_file(
                Path(config["k06_input_dir"]).expanduser().resolve()
                / "merge_manifest.csv"
            ),
            "cnn_run_dir": str(run_dir),
            "run_config_sha256": sha256_file(run_dir / "run_config.json"),
            "fold_assignments_sha256": sha256_file(
                run_dir / "fold_assignments.json"
            ),
            "existing_oof_predictions": str(
                Path(config["existing_oof_predictions"]).expanduser().resolve()
            ),
            "existing_oof_predictions_sha256": sha256_file(
                config["existing_oof_predictions"]
            ),
            "checkpoint_sha256": checkpoint_hashes,
            "checkpoint_metadata": checkpoint_metadata,
            "sample_manifest_sha256": sha256_file(sample_manifest_path),
            "sample_content_fingerprint_sha256": sha256_payload(
                [
                    {
                        "sample_id": row["sample_id"],
                        "sample_path": row["sample_path"],
                        "sample_size_bytes": row["sample_size_bytes"],
                        "sample_sha256": row["sample_sha256"],
                        "mask_path": row["mask_path"],
                        "mask_size_bytes": row["mask_size_bytes"],
                        "mask_sha256": row["mask_sha256"],
                    }
                    for row in sample_rows
                ]
            ),
        },
        "dataset_summary": dataset_summary,
        "observed_counts": observed_summary,
        "fold_validation": {
            "num_folds": int(len(folds)),
            "all_subjects": all_subjects,
            "heldout_once": True,
            "train_test_subject_overlap": 0,
        },
        "common_mask": {
            **common_mask_summary,
            "array_sha256": sha256_array(common_mask.astype(np.uint8)),
            "existing_mask_path": str(existing_common_mask_path),
            "matches_existing_mask": True,
        },
        "fixed_roi_masks": roi_details,
        "frozen_definitions": {
            "experiment_config_sha256": sha256_file(config["_config_path"]),
            "roi_definitions_sha256": sha256_file(DEFAULT_ROI_CONFIG_PATH),
            "frozen_experiment_config_sha256": sha256_file(
                frozen_experiment_config_path
            ),
            "frozen_roi_definitions_sha256": sha256_file(
                frozen_roi_definitions_path
            ),
            "frozen_fold_assignments_sha256": sha256_file(
                frozen_fold_assignments_path
            ),
            "source_code_sha256": {
                path.name: sha256_file(path)
                for path in sorted((CODE_ROOT).glob("*.py"))
                if not path.name.startswith("._")
            },
            "numerical_workspace_code_sha256": {
                relative_path: sha256_file(path)
                for relative_path, path in numerical_workspace_paths.items()
            },
            "atlas_sha256": {
                str(path.relative_to(atlas_root)): sha256_file(path)
                for path in [
                    atlas_root / "HarvardOxford-Cortical.xml",
                    atlas_root / "HarvardOxford-Subcortical.xml",
                    atlas_root
                    / "HarvardOxford"
                    / "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz",
                    atlas_root
                    / "HarvardOxford"
                    / "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz",
                ]
            },
            "fixed_mask_file_sha256": {
                name: sha256_file(masks_dir / f"{name}_mask.nii.gz")
                for name in ("value", "visual", "union")
            },
            "common_mask_file_sha256": sha256_file(
                masks_dir / "common_brain_mask.nii.gz"
            ),
        },
        "environment": environment_payload(),
    }
    write_json(PROJECT_ROOT / "inputs" / "provenance.json", provenance)

    print(
        "Prepared inputs: "
        f"{observed_summary['num_samples']} samples, "
        f"{observed_summary['num_subjects']} subjects, "
        f"{len(folds)} folds, common mask={int(common_mask.sum())} voxels.",
        flush=True,
    )
    print(
        "Fixed masks: "
        f"value={int(fixed_masks['value'].sum())}, "
        f"visual={int(fixed_masks['visual'].sum())}, "
        f"union={int(fixed_masks['union'].sum())}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
