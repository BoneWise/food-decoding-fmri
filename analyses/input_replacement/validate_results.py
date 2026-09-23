from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from .common import (
    CODE_ROOT, CONFIG_ROOT,
    PROJECT_ROOT,
    build_numerical_dependency_anchor,
    load_config,
    read_json,
    sha256_array,
    sha256_file,
    sha256_payload,
    stable_sample_id,
    verify_frozen_static_dependencies,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate completeness, leakage boundaries, and reproducibility."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Validate fold-level raw outputs before statistical summarization.",
    )
    return parser


def load_mask(path: Path) -> np.ndarray:
    data = np.asarray(nib.load(str(path)).dataobj)
    return np.asarray(np.isfinite(data) & (data > 0), dtype=bool)


def require(condition: bool, message: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise ValueError(message)


def write_checksum_manifest(root: Path, output_path: Path) -> int:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith("._")
        and "diagnostics" not in path.relative_to(root).parts
        and "outputs/validation" not in path.as_posix()
        and path.name != "checksums.sha256"
        and path.resolve() != output_path.resolve()
    )
    lines = []
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT)
        lines.append(f"{sha256_file(path)}  {relative.as_posix()}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return int(len(paths))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    expected = config["expected"]
    expected_folds = int(expected["num_folds"])
    expected_random = int(config["random_control"]["num_replicates"])
    expected_top = int(
        np.ceil(
            float(config["cam"]["top_fraction"])
            * int(expected["common_mask_voxels"])
        )
    )
    checks: list[dict[str, Any]] = []
    raw_artifact_sha256: dict[str, str] = {}

    provenance_path = PROJECT_ROOT / "inputs" / "provenance.json"
    require(provenance_path.exists(), "input provenance exists", checks)
    provenance = read_json(provenance_path)
    verify_frozen_static_dependencies(config, provenance)
    numerical_dependency_anchor = build_numerical_dependency_anchor(provenance)
    validation_dir = PROJECT_ROOT / "outputs" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        validation_dir / "raw_validation_report.json",
        {
            "experiment_id": config["experiment_id"],
            "stage": "validation_in_progress",
            "passed": False,
        },
    )
    require(
        provenance["inputs"]["beta_index_sha256"]
        == expected["beta_index_sha256"]
        and sha256_file(config["beta_index_csv"])
        == provenance["inputs"]["beta_index_sha256"]
        and sha256_file(config["existing_oof_predictions"])
        == provenance["inputs"]["existing_oof_predictions_sha256"],
        "k=6 beta-index hash matches the frozen source",
        checks,
    )
    live_run_dir = Path(config["cnn_run_dir"]).expanduser().resolve()
    live_run_config_path = live_run_dir / "run_config.json"
    live_fold_assignments_path = live_run_dir / "fold_assignments.json"
    require(
        sha256_file(live_run_config_path)
        == provenance["inputs"]["run_config_sha256"]
        and sha256_file(live_fold_assignments_path)
        == provenance["inputs"]["fold_assignments_sha256"]
        and read_json(live_fold_assignments_path)
        == read_json(PROJECT_ROOT / "inputs" / "fold_assignments.json"),
        "live CNN run configuration and folds match frozen provenance",
        checks,
    )

    common = load_mask(PROJECT_ROOT / "outputs" / "masks" / "common_brain_mask.nii.gz")
    value = load_mask(PROJECT_ROOT / "outputs" / "masks" / "value_mask.nii.gz")
    visual = load_mask(PROJECT_ROOT / "outputs" / "masks" / "visual_mask.nii.gz")
    union = load_mask(PROJECT_ROOT / "outputs" / "masks" / "union_mask.nii.gz")
    for path in [
        Path(config["_config_path"]),
        CONFIG_ROOT / "roi_definitions.json",
        PROJECT_ROOT / "inputs" / "provenance.json",
        PROJECT_ROOT / "inputs" / "fold_assignments.json",
        PROJECT_ROOT / "inputs" / "sample_manifest.csv",
        PROJECT_ROOT / "inputs" / "frozen_experiment_config.json",
        PROJECT_ROOT / "inputs" / "frozen_roi_definitions.json",
        PROJECT_ROOT / "outputs" / "pipeline_status.json",
        PROJECT_ROOT / "outputs" / "masks" / "common_brain_mask.nii.gz",
        PROJECT_ROOT / "outputs" / "masks" / "value_mask.nii.gz",
        PROJECT_ROOT / "outputs" / "masks" / "visual_mask.nii.gz",
        PROJECT_ROOT / "outputs" / "masks" / "union_mask.nii.gz",
    ]:
        raw_artifact_sha256[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)
    require(
        int(common.sum()) == int(expected["common_mask_voxels"]),
        "common mask has the frozen voxel count",
        checks,
    )
    require(
        int(value.sum()) == 4092,
        "value mask contains 4092 voxels",
        checks,
    )
    require(
        int(visual.sum()) == 27651,
        "visual mask contains 27651 voxels",
        checks,
    )
    require(
        int(union.sum()) == 31743,
        "union mask contains 31743 voxels",
        checks,
    )
    require(
        np.array_equal(union, value | visual),
        "union equals value OR visual voxelwise",
        checks,
    )
    require(
        not bool(np.any(value & visual)),
        "value and visual masks do not overlap",
        checks,
    )
    for name, mask in (("value", value), ("visual", visual), ("union", union)):
        require(
            not bool(np.any(mask & ~common)),
            f"{name} mask is contained in the common mask",
            checks,
        )
        require(
            sha256_file(
                PROJECT_ROOT / "outputs" / "masks" / f"{name}_mask.nii.gz"
            )
            == provenance["frozen_definitions"][
                "fixed_mask_file_sha256"
            ][name],
            f"{name} mask file hash matches frozen provenance",
            checks,
        )
    require(
        sha256_file(
            PROJECT_ROOT / "outputs" / "masks" / "common_brain_mask.nii.gz"
        )
        == provenance["frozen_definitions"]["common_mask_file_sha256"],
        "common-mask file hash matches frozen provenance",
        checks,
    )

    fold_assignments = read_json(PROJECT_ROOT / "inputs" / "fold_assignments.json")
    frozen_manifest = pd.read_csv(PROJECT_ROOT / "inputs" / "sample_manifest.csv")
    derived_manifest_sample_ids = [
        stable_sample_id(subject, rating, trial)
        for subject, rating, trial in zip(
            frozen_manifest["subject_id"],
            frozen_manifest["rating"],
            frozen_manifest["trial_id"],
        )
    ]
    if "sample_id" not in frozen_manifest.columns:
        frozen_manifest["sample_id"] = derived_manifest_sample_ids
    require(
        frozen_manifest["sample_id"].astype(str).tolist()
        == derived_manifest_sample_ids
        and not bool(frozen_manifest["sample_id"].duplicated().any()),
        "frozen manifest has unique and internally consistent sample IDs",
        checks,
    )
    require(
        sha256_file(PROJECT_ROOT / "inputs" / "sample_manifest.csv")
        == provenance["inputs"]["sample_manifest_sha256"],
        "sample manifest hash matches frozen provenance",
        checks,
    )
    source_hash_cache: dict[str, str] = {}
    content_fingerprint_rows: list[dict[str, Any]] = []
    all_source_files_match = True
    for row in frozen_manifest.itertuples(index=False):
        sample_path = str(Path(row.sample_path).expanduser().resolve())
        mask_path = str(Path(row.mask_path).expanduser().resolve())
        for path_text in (sample_path, mask_path):
            if path_text not in source_hash_cache:
                source_hash_cache[path_text] = sha256_file(path_text)
        if (
            int(Path(sample_path).stat().st_size)
            != int(row.sample_size_bytes)
            or int(Path(mask_path).stat().st_size)
            != int(row.mask_size_bytes)
            or source_hash_cache[sample_path] != str(row.sample_sha256)
            or source_hash_cache[mask_path] != str(row.mask_sha256)
        ):
            all_source_files_match = False
        content_fingerprint_rows.append(
            {
                "sample_id": str(row.sample_id),
                "sample_path": sample_path,
                "sample_size_bytes": int(row.sample_size_bytes),
                "sample_sha256": source_hash_cache[sample_path],
                "mask_path": mask_path,
                "mask_size_bytes": int(row.mask_size_bytes),
                "mask_sha256": source_hash_cache[mask_path],
            }
        )
    require(
        all_source_files_match
        and sha256_payload(content_fingerprint_rows)
        == provenance["inputs"]["sample_content_fingerprint_sha256"],
        "all beta maps and participant masks exactly match the frozen content manifest",
        checks,
    )
    existing_oof = pd.read_csv(config["existing_oof_predictions"])
    existing_oof["lookup_key"] = [
        (
            int(fold_index),
            str(subject_id),
            int(true_rating),
            int(float(trial_id)),
        )
        for fold_index, subject_id, true_rating, trial_id in zip(
            existing_oof["fold_index"],
            existing_oof["subject_id"],
            existing_oof["true_rating"],
            existing_oof["trial_id"],
        )
    ]
    require(
        not bool(existing_oof["lookup_key"].duplicated().any()),
        "frozen OOF prediction keys are unique",
        checks,
    )
    existing_oof_by_key = {
        tuple(row.lookup_key): row
        for row in existing_oof.itertuples(index=False)
    }
    heldout_subjects: list[str] = []
    total_test_inputs = 0
    full_prediction_frames = []
    baseline_max_differences = []
    for fold in fold_assignments["folds"]:
        fold_index = int(fold["fold_index"])
        fold_dir = PROJECT_ROOT / "outputs" / "folds" / f"fold_{fold_index:02d}"
        complete_path = fold_dir / "fold_complete.json"
        require(complete_path.exists(), f"fold {fold_index:02d} is complete", checks)
        complete = read_json(complete_path)
        training_metadata_path = (
            fold_dir / "metadata" / "training_artifacts.json"
        )
        random_metadata_path = fold_dir / "metadata" / "random_masks.json"
        inference_metadata_path = fold_dir / "metadata" / "inference.json"
        training_metadata = read_json(training_metadata_path)
        random_metadata = read_json(random_metadata_path)
        inference = read_json(inference_metadata_path)
        for path in [
            complete_path,
            training_metadata_path,
            fold_dir / "metadata" / "training_cam_counts.csv",
            random_metadata_path,
            inference_metadata_path,
            fold_dir / "metadata" / "training_global_mean.json",
            fold_dir / "maps" / "training_cam_rating1_participant_mean.nii.gz",
            fold_dir / "maps" / "training_cam_rating4_participant_mean.nii.gz",
            fold_dir / "maps" / "training_cam_rating1_brainmean_normalized.nii.gz",
            fold_dir / "maps" / "training_cam_rating4_brainmean_normalized.nii.gz",
            fold_dir / "maps" / "training_cam_joint_class_balanced.nii.gz",
            fold_dir / "masks" / "cam_top20_mask.nii.gz",
            fold_dir / "masks" / "random_top20_flat_indices.npz",
            fold_dir / "predictions" / "fixed_condition_predictions.csv",
            fold_dir / "predictions" / "random_top20_predictions.npz",
        ]:
            raw_artifact_sha256[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)
        train_subjects = set(str(value) for value in complete["train_subjects"])
        test_subjects = set(str(value) for value in complete["test_subjects"])
        expected_train_subjects = [
            str(value) for value in fold["train_subjects"]
        ]
        expected_test_subjects = [str(value) for value in fold["val_subjects"]]
        require(
            complete["train_subjects"] == expected_train_subjects
            and complete["test_subjects"] == expected_test_subjects,
            f"fold {fold_index:02d} complete marker exactly matches frozen subjects",
            checks,
        )
        require(
            not bool(train_subjects & test_subjects),
            f"fold {fold_index:02d} train/test subjects are disjoint",
            checks,
        )
        heldout_subjects.extend(sorted(test_subjects))
        total_test_inputs += int(complete["num_test_inputs"])
        expected_train_sample_ids = frozen_manifest.loc[
            frozen_manifest["subject_id"].astype(str).isin(
                expected_train_subjects
            ),
            "sample_id",
        ].astype(str).tolist()
        expected_test_sample_ids_ordered = frozen_manifest.loc[
            frozen_manifest["subject_id"].astype(str).isin(
                expected_test_subjects
            ),
            "sample_id",
        ].astype(str).tolist()
        expected_checkpoint_hash = provenance["inputs"]["checkpoint_sha256"][
            f"fold_{fold_index:02d}"
        ]
        checkpoint_path = (
            Path(config["cnn_run_dir"]).expanduser().resolve()
            / f"fold_{fold_index:02d}"
            / str(config["checkpoint_name"])
        )
        require(
            sha256_file(checkpoint_path) == expected_checkpoint_hash
            and complete["checkpoint_sha256"] == expected_checkpoint_hash
            and str(Path(complete["checkpoint"]).resolve())
            == str(checkpoint_path),
            f"fold {fold_index:02d} checkpoint exactly matches provenance",
            checks,
        )

        expected_training_payload = {
            "schema": "rq3_training_artifacts_global_scalar/v3",
            "fold_index": fold_index,
            "checkpoint_sha256": expected_checkpoint_hash,
            "execution_device": str(config["inference"]["device"]),
            "numerical_dependency_anchor": numerical_dependency_anchor,
            "train_subjects": expected_train_subjects,
            "train_sample_ids": expected_train_sample_ids,
            "cam_config": config["cam"],
            "replacement_config": config["replacement"],
        }
        expected_training_signature = sha256_payload(
            expected_training_payload
        )
        require(
            training_metadata.get("dependency_payload")
            == expected_training_payload
            and training_metadata.get("dependency_signature")
            == expected_training_signature
            and sha256_payload(training_metadata["dependency_payload"])
            == training_metadata["dependency_signature"],
            f"fold {fold_index:02d} training dependency signature is exact",
            checks,
        )
        require(
            complete["training_artifacts"] == training_metadata,
            f"fold {fold_index:02d} embeds the exact training metadata",
            checks,
        )
        require(
            int(complete["num_train_inputs"]) == len(expected_train_sample_ids)
            and int(complete["num_test_inputs"])
            == len(expected_test_sample_ids_ordered)
            and int(training_metadata["num_training_inputs"])
            == len(expected_train_sample_ids),
            f"fold {fold_index:02d} training/test counts match the frozen manifest",
            checks,
        )

        cam_counts = pd.read_csv(
            fold_dir / "metadata" / "training_cam_counts.csv"
        )
        require(
            set(cam_counts["subject_id"].astype(str)).issubset(train_subjects),
            f"fold {fold_index:02d} training CAM provenance contains only training subjects",
            checks,
        )
        require(
            not bool(set(cam_counts["subject_id"].astype(str)) & test_subjects),
            f"fold {fold_index:02d} training CAM provenance excludes test subjects",
            checks,
        )

        cam_mask = load_mask(fold_dir / "masks" / "cam_top20_mask.nii.gz")
        require(
            int(cam_mask.sum()) == expected_top,
            f"fold {fold_index:02d} CAM mask has exactly {expected_top} voxels",
            checks,
        )
        require(
            not bool(np.any(cam_mask & ~common)),
            f"fold {fold_index:02d} CAM mask is contained in the common mask",
            checks,
        )
        training_global_mean_payload = read_json(
            fold_dir / "metadata" / "training_global_mean.json"
        )
        training_global_mean = float(training_global_mean_payload["value"])
        joint_cam = np.asarray(
            nib.load(
                str(
                    fold_dir
                    / "maps"
                    / "training_cam_joint_class_balanced.nii.gz"
                )
            ).dataobj,
            dtype=np.float32,
        )
        require(
            bool(training_metadata["training_global_mean"]["is_scalar"])
            and np.isfinite(training_global_mean)
            and training_metadata["training_global_mean"]["aggregation"]
            == "pooled_voxel_mean_without_coordinatewise_map"
            and sha256_array(
                np.asarray([training_global_mean], dtype=np.float64)
            )
            == training_metadata["training_global_mean"]["scalar_sha256"]
            and sha256_array(
                np.asarray([training_global_mean], dtype=np.float64)
            )
            == training_global_mean_payload["scalar_sha256"],
            f"fold {fold_index:02d} training global mean is one verified scalar",
            checks,
        )
        require(
            sha256_array(joint_cam)
            == training_metadata["cam"]["joint_cam_array_sha256"],
            f"fold {fold_index:02d} joint-CAM array hash matches metadata",
            checks,
        )
        require(
            sha256_array(cam_mask.astype(np.uint8))
            == training_metadata["top20_selection"]["mask_array_sha256"]
            and int(
                training_metadata["top20_selection"]["num_selected_voxels"]
            )
            == expected_top
            and float(
                training_metadata["top20_selection"]["fraction_requested"]
            )
            == float(config["cam"]["top_fraction"]),
            f"fold {fold_index:02d} CAM-selection metadata matches its mask",
            checks,
        )

        with np.load(
            fold_dir / "masks" / "random_top20_flat_indices.npz",
            allow_pickle=False,
        ) as payload:
            random_indices = np.asarray(payload["flat_indices"], dtype=np.int64)
        require(
            random_indices.shape == (expected_random, expected_top),
            f"fold {fold_index:02d} has {expected_random} random masks with matched size",
            checks,
        )
        expected_random_payload = {
            "schema": "rq3_random_masks/v2",
            "fold_index": fold_index,
            "common_mask_array_sha256": provenance["common_mask"][
                "array_sha256"
            ],
            "num_selected_voxels": expected_top,
            "random_control_config": config["random_control"],
        }
        expected_random_signature = sha256_payload(expected_random_payload)
        random_indices_path = (
            fold_dir / "masks" / "random_top20_flat_indices.npz"
        )
        require(
            random_metadata.get("dependency_payload")
            == expected_random_payload
            and random_metadata.get("dependency_signature")
            == expected_random_signature
            and sha256_payload(random_metadata["dependency_payload"])
            == random_metadata["dependency_signature"]
            and sha256_array(random_indices.astype(np.uint32))
            == random_metadata["indices_array_sha256"]
            and sha256_file(random_indices_path)
            == random_metadata["file_sha256"],
            f"fold {fold_index:02d} random-mask dependency and file hashes are exact",
            checks,
        )
        require(
            complete["random_masks"] == random_metadata,
            f"fold {fold_index:02d} embeds the exact random-mask metadata",
            checks,
        )
        require(
            int(random_metadata["base_seed"])
            == int(config["random_control"]["seed"])
            and int(random_metadata["fold_seed"])
            == int(config["random_control"]["seed"])
            + fold_index * 1_000_003,
            f"fold {fold_index:02d} random-mask seeds match the frozen config",
            checks,
        )
        flat_common = common.reshape(-1)
        for replicate, row in enumerate(random_indices):
            if np.unique(row).size != expected_top:
                raise ValueError(
                    f"Fold {fold_index:02d} random mask {replicate + 1} "
                    "contains duplicate voxels."
                )
            if not bool(np.all(flat_common[row])):
                raise ValueError(
                    f"Fold {fold_index:02d} random mask {replicate + 1} "
                    "contains voxels outside the common mask."
                )
        checks.append(
            {
                "check": (
                    f"fold {fold_index:02d} all random masks are unique-within-mask "
                    "and contained in the common mask"
                ),
                "passed": True,
            }
        )

        require(
            int(inference["num_random_replicates"]) == expected_random,
            f"fold {fold_index:02d} inference used all random masks",
            checks,
        )
        require(
            float(inference["probability_threshold"])
            == float(config["inference"]["probability_threshold"])
            and set(inference["conditions"])
            == {"full", "cam_top20", "value", "visual", "union"},
            f"fold {fold_index:02d} inference threshold and conditions match config",
            checks,
        )
        expected_inference_payload = {
            "schema": "rq3_inference_global_scalar/v3",
            "fold_index": fold_index,
            "checkpoint_sha256": expected_checkpoint_hash,
            "execution_device": str(config["inference"]["device"]),
            "numerical_dependency_anchor": numerical_dependency_anchor,
            "training_dependency_signature": expected_training_signature,
            "training_global_mean_sha256": training_metadata["training_global_mean"][
                "scalar_sha256"
            ],
            "cam_top20_mask_sha256": training_metadata[
                "top20_selection"
            ]["mask_array_sha256"],
            "random_dependency_signature": expected_random_signature,
            "random_indices_sha256": random_metadata[
                "indices_array_sha256"
            ],
            "fixed_mask_sha256": {
                name: provenance["fixed_roi_masks"][name]["mask_sha256"]
                for name in ("value", "visual", "union")
            },
            "test_subjects": expected_test_subjects,
            "test_sample_ids": expected_test_sample_ids_ordered,
            "inference_config": config["inference"],
            "num_random_replicates": expected_random,
        }
        expected_inference_signature = sha256_payload(
            expected_inference_payload
        )
        require(
            inference.get("dependency_payload") == expected_inference_payload
            and inference.get("dependency_signature")
            == expected_inference_signature
            and sha256_payload(inference["dependency_payload"])
            == inference["dependency_signature"],
            f"fold {fold_index:02d} inference dependency signature is exact",
            checks,
        )
        require(
            complete["inference"] == inference
            and complete.get("execution_device")
            == str(config["inference"]["device"])
            and complete.get("dependency_signatures")
            == {
                "training_artifacts": expected_training_signature,
                "random_masks": expected_random_signature,
                "inference": expected_inference_signature,
            },
            f"fold {fold_index:02d} embeds exact inference metadata and signatures",
            checks,
        )
        replacement_audits = inference["replacement"]["elementwise_audits"]
        require(
            np.isclose(
                float(inference["replacement"]["training_global_mean"]),
                training_global_mean,
                rtol=0.0,
                atol=0.0,
            )
            and inference["replacement"]["assignment"]
            == "every_selected_voxel_receives_the_same_scalar"
            and not bool(inference["replacement"]["renormalized_after_replacement"])
            and set(replacement_audits)
            == {
                "cam_top20",
                "value",
                "visual",
                "union",
                "random_top20_replicate_001",
            }
            and all(
                bool(audit["selected_values_equal_one_global_scalar"])
                and bool(audit["unselected_values_unchanged"])
                and bool(audit["full_tensor_matches_defined_intervention"])
                for audit in replacement_audits.values()
            ),
            f"fold {fold_index:02d} verifies scalar-only replacement elementwise",
            checks,
        )

        fixed_path = (
            fold_dir / "predictions" / "fixed_condition_predictions.csv"
        )
        random_predictions_path = (
            fold_dir / "predictions" / "random_top20_predictions.npz"
        )
        require(
            str(Path(inference["outputs"]["fixed_predictions_csv"]).resolve())
            == str(fixed_path.resolve())
            and sha256_file(fixed_path)
            == inference["outputs"]["fixed_predictions_sha256"]
            and str(
                Path(inference["outputs"]["random_predictions_npz"]).resolve()
            )
            == str(random_predictions_path.resolve())
            and sha256_file(random_predictions_path)
            == inference["outputs"]["random_predictions_sha256"],
            f"fold {fold_index:02d} prediction files and hashes match inference metadata",
            checks,
        )
        fixed = pd.read_csv(fixed_path)
        require(
            "sample_id" in fixed.columns,
            f"fold {fold_index:02d} fixed predictions include stable sample IDs",
            checks,
        )
        derived_fixed_sample_ids = [
            stable_sample_id(subject, rating, trial)
            for subject, rating, trial in zip(
                fixed["subject_id"],
                fixed["rating"],
                fixed["trial_id"],
            )
        ]
        require(
            fixed["sample_id"].astype(str).tolist()
            == derived_fixed_sample_ids,
            f"fold {fold_index:02d} fixed sample IDs match their row metadata",
            checks,
        )
        expected_rows = int(complete["num_test_inputs"]) * 5
        require(
            len(fixed) == expected_rows,
            f"fold {fold_index:02d} has five fixed-condition rows per test input",
            checks,
        )
        require(
            bool(np.isfinite(fixed["prob_high"]).all())
            and bool(
                (
                    (fixed["prob_high"] >= 0.0)
                    & (fixed["prob_high"] <= 1.0)
                ).all()
            )
            and set(fixed["target"].astype(int)).issubset({0, 1})
            and set(fixed["pred_target"].astype(int)).issubset({0, 1}),
            f"fold {fold_index:02d} fixed predictions and labels are valid",
            checks,
        )
        require(
            np.array_equal(
                fixed["pred_target"].to_numpy(dtype=np.int64),
                (
                    fixed["prob_high"].to_numpy(dtype=np.float64)
                    >= float(config["inference"]["probability_threshold"])
                ).astype(np.int64),
            ),
            f"fold {fold_index:02d} saved labels follow the frozen threshold",
            checks,
        )
        unique_counts = fixed.groupby(
            ["sample_id"]
        )["condition"].nunique()
        require(
            bool((unique_counts == 5).all()),
            f"fold {fold_index:02d} every test input has all fixed conditions",
            checks,
        )
        require(
            not bool(fixed.duplicated(["sample_id", "condition"]).any()),
            f"fold {fold_index:02d} has unique sample-by-condition rows",
            checks,
        )
        expected_sample_ids = set(
            frozen_manifest.loc[
                frozen_manifest["test_fold"] == fold_index,
                "sample_id",
            ].astype(str)
        )
        observed_sample_ids = set(fixed["sample_id"].astype(str))
        require(
            observed_sample_ids == expected_sample_ids,
            f"fold {fold_index:02d} test samples exactly match the frozen manifest",
            checks,
        )
        expected_by_id = frozen_manifest.set_index("sample_id").loc[
            fixed["sample_id"].astype(str)
        ]
        row_metadata_matches = (
            fixed["subject_id"].astype(str).to_numpy()
            == expected_by_id["subject_id"].astype(str).to_numpy()
        )
        row_metadata_matches &= (
            fixed["session_id"].astype(str).to_numpy()
            == expected_by_id["session_id"].astype(str).to_numpy()
        )
        row_metadata_matches &= (
            fixed["run_id"].astype(str).to_numpy()
            == expected_by_id["run_id"].astype(str).to_numpy()
        )
        row_metadata_matches &= (
            fixed["trial_id"].to_numpy(dtype=np.int64)
            == expected_by_id["trial_id"].to_numpy(dtype=np.int64)
        )
        row_metadata_matches &= (
            fixed["rating"].to_numpy(dtype=np.int64)
            == expected_by_id["rating"].to_numpy(dtype=np.int64)
        )
        row_metadata_matches &= (
            fixed["sample_path"].astype(str).to_numpy()
            == expected_by_id["sample_path"].astype(str).to_numpy()
        )
        row_metadata_matches &= (
            fixed["fold_index"].to_numpy(dtype=np.int64) == fold_index
        )
        row_metadata_matches &= (
            fixed["target"].to_numpy(dtype=np.int64)
            == (
                expected_by_id["rating"].to_numpy(dtype=np.int64) == 4
            ).astype(np.int64)
        )
        require(
            bool(np.all(row_metadata_matches)),
            f"fold {fold_index:02d} fixed rows exactly match frozen sample metadata",
            checks,
        )
        require(
            set(fixed["condition"].astype(str))
            == {"full", "cam_top20", "value", "visual", "union"},
            f"fold {fold_index:02d} fixed-condition names are exact",
            checks,
        )
        full = fixed.loc[fixed["condition"] == "full"].copy()
        direct_probability_differences: list[float] = []
        direct_label_mismatches = 0
        for row in full.itertuples(index=False):
            key = (
                fold_index,
                str(row.subject_id),
                int(row.rating),
                int(row.trial_id),
            )
            old = existing_oof_by_key[key]
            direct_probability_differences.append(
                abs(float(row.prob_high) - float(old.prob_high))
            )
            direct_label_mismatches += int(
                int(row.pred_target) != int(old.pred_target)
            )
        direct_max_difference = float(max(direct_probability_differences))
        require(
            direct_label_mismatches == 0,
            f"fold {fold_index:02d} full-input labels directly reproduce frozen OOF",
            checks,
        )
        require(
            int(
                inference["baseline_reproduction"][
                    "prediction_label_mismatches"
                ]
            )
            == direct_label_mismatches
            and np.isclose(
                float(
                    inference["baseline_reproduction"][
                        "max_absolute_probability_difference"
                    ]
                ),
                direct_max_difference,
                atol=1e-12,
            ),
            f"fold {fold_index:02d} baseline metadata matches direct OOF comparison",
            checks,
        )
        baseline_max_differences.append(direct_max_difference)
        full_prediction_frames.append(full)

        with np.load(
            random_predictions_path,
            allow_pickle=False,
        ) as payload:
            probabilities = np.asarray(payload["probabilities"])
            targets = np.asarray(payload["targets"])
            random_subjects = np.asarray(payload["subject_ids"]).astype(str)
            random_ratings = np.asarray(payload["ratings"], dtype=np.int64)
            random_trial_ids = np.asarray(payload["trial_ids"], dtype=np.int64)
            random_sample_ids = np.asarray(payload["sample_ids"]).astype(str)
            random_fold_index = int(
                np.asarray(payload["fold_index"]).reshape(-1)[0]
            )
        require(
            probabilities.shape
            == (expected_random, int(complete["num_test_inputs"])),
            f"fold {fold_index:02d} random-prediction array has the expected shape",
            checks,
        )
        require(
            random_sample_ids.tolist() == expected_test_sample_ids_ordered
            and random_fold_index == fold_index,
            f"fold {fold_index:02d} random predictions follow the frozen sample order",
            checks,
        )
        derived_random_sample_ids = [
            stable_sample_id(subject, rating, trial)
            for subject, rating, trial in zip(
                random_subjects,
                random_ratings,
                random_trial_ids,
            )
        ]
        random_expected = frozen_manifest.set_index("sample_id").loc[
            random_sample_ids
        ]
        require(
            random_sample_ids.tolist() == derived_random_sample_ids
            and np.array_equal(
                random_subjects,
                random_expected["subject_id"].astype(str).to_numpy(),
            )
            and np.array_equal(
                random_ratings,
                random_expected["rating"].to_numpy(dtype=np.int64),
            )
            and np.array_equal(
                random_trial_ids,
                random_expected["trial_id"].to_numpy(dtype=np.int64),
            ),
            f"fold {fold_index:02d} random metadata exactly matches frozen samples",
            checks,
        )
        full_by_id = (
            fixed.loc[fixed["condition"] == "full"]
            .set_index("sample_id")
            .loc[random_sample_ids]
        )
        require(
            np.array_equal(
                targets,
                full_by_id["target"].to_numpy(dtype=np.int64),
            ),
            f"fold {fold_index:02d} random targets align with fixed predictions",
            checks,
        )
        require(
            bool(np.isfinite(probabilities).all())
            and bool((probabilities >= 0.0).all())
            and bool((probabilities <= 1.0).all())
            and set(np.asarray(targets, dtype=np.int64).tolist()).issubset(
                {0, 1}
            ),
            f"fold {fold_index:02d} random predictions and targets are valid",
            checks,
        )
        require(
            targets.shape == (int(complete["num_test_inputs"]),),
            f"fold {fold_index:02d} target array has the expected shape",
            checks,
        )

    pipeline_status = read_json(PROJECT_ROOT / "outputs" / "pipeline_status.json")
    require(
        bool(pipeline_status["all_expected_folds_present"])
        and len(pipeline_status["fold_status"]) == expected_folds
        and all(
            bool(row["complete"])
            and bool(row.get("dependency_signatures_match", False))
            and int(row["num_random_replicates"]) == expected_random
            for row in pipeline_status["fold_status"]
        ),
        "pipeline status records all ten signature-valid 100-random-mask folds",
        checks,
    )
    require(
        len(heldout_subjects) == len(set(heldout_subjects)) == int(expected["num_subjects"]),
        "all 30 subjects are held out exactly once",
        checks,
    )
    require(
        total_test_inputs == int(expected["num_samples"]),
        "all 1147 held-out inputs are present exactly once",
        checks,
    )
    all_full = pd.concat(full_prediction_frames, ignore_index=True)
    require(
        len(all_full) == int(expected["num_samples"]),
        "full-input OOF prediction count is 1147",
        checks,
    )
    require(
        max(baseline_max_differences) < 0.001,
        "full-input probability differences from the frozen run are below 0.001",
        checks,
    )

    validation_dir = PROJECT_ROOT / "outputs" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    raw_report = {
        "experiment_id": config["experiment_id"],
        "stage": "raw_fold_outputs",
        "passed": True,
        "experiment_config_sha256": sha256_file(config["_config_path"]),
        "roi_definitions_sha256": sha256_file(
            CONFIG_ROOT / "roi_definitions.json"
        ),
        "num_checks": int(len(checks)),
        "num_passed": int(sum(bool(row["passed"]) for row in checks)),
        "max_absolute_full_probability_difference": float(
            max(baseline_max_differences)
        ),
        "raw_artifact_sha256": raw_artifact_sha256,
        "checks": checks,
    }
    write_json(validation_dir / "raw_validation_report.json", raw_report)
    if args.raw_only:
        print(
            f"Raw validation passed: {raw_report['num_passed']}/"
            f"{raw_report['num_checks']} checks.",
            flush=True,
        )
        return

    results_summary_path = PROJECT_ROOT / "outputs" / "results_summary.json"
    require(results_summary_path.exists(), "group results summary exists", checks)
    primary_path = PROJECT_ROOT / "outputs" / "tables" / "primary_inference.csv"
    require(primary_path.exists(), "primary inference table exists", checks)
    primary = pd.read_csv(primary_path)
    require(
        set(primary["comparison_id"])
        == set(config["statistics"]["primary_contrasts"]),
        "only the four prespecified primary contrasts enter multiplicity correction",
        checks,
    )
    require(
        len(primary) == 4 and bool((primary["n"] == 30).all()),
        "the four primary contrasts each contain 30 participants",
        checks,
    )
    require(
        bool(np.isfinite(primary["mean_effect"]).all()),
        "all primary effects are finite",
        checks,
    )
    require(
        bool(
            (
                (primary["p_raw"] >= 0.0)
                & (primary["p_raw"] <= 1.0)
                & (primary["p_holm"] >= 0.0)
                & (primary["p_holm"] <= 1.0)
                & (primary["p_holm"] + 1e-15 >= primary["p_raw"])
            ).all()
        ),
        "raw and Holm p values are valid and Holm is not smaller than raw",
        checks,
    )
    participant_conditions = pd.read_csv(
        PROJECT_ROOT
        / "outputs"
        / "tables"
        / "participant_condition_metrics.csv"
    )
    condition_counts = participant_conditions.groupby("condition")[
        "subject_id"
    ].nunique()
    require(
        set(condition_counts.index)
        == {
            "full",
            "cam_top20",
            "random_top20_mean",
            "value",
            "visual",
            "union",
        }
        and bool((condition_counts == 30).all()),
        "all six reported conditions contain all 30 participants",
        checks,
    )
    effects = pd.read_csv(
        PROJECT_ROOT / "outputs" / "tables" / "participant_effects.csv"
    )
    for row in primary.itertuples(index=False):
        values = effects.loc[
            effects["comparison_id"] == row.comparison_id,
            "ba_effect",
        ].to_numpy(dtype=np.float64)
        require(
            values.size == 30
            and np.isclose(float(np.mean(values)), float(row.mean_effect), atol=1e-12),
            f"{row.comparison_id} mean effect reproduces from participant rows",
            checks,
        )
    from .summarize_ablation import (
        bootstrap_mean_ci,
        holm_adjust,
        sign_flip_one_sided,
    )

    recomputed_raw_p: list[float] = []
    for contrast_index, contrast in enumerate(
        config["statistics"]["primary_contrasts"]
    ):
        saved = primary.loc[
            primary["comparison_id"] == contrast
        ].iloc[0]
        values = effects.loc[
            effects["comparison_id"] == contrast,
            "ba_effect",
        ].to_numpy(dtype=np.float64)
        seed = int(config["statistics"]["seed"]) + contrast_index * 1009
        recomputed_test = sign_flip_one_sided(
            values,
            num_sign_flips=int(config["statistics"]["num_sign_flips"]),
            seed=seed,
        )
        ci_low, ci_high = bootstrap_mean_ci(
            values,
            num_bootstrap=int(config["statistics"]["num_bootstrap"]),
            seed=seed + 1,
        )
        recomputed_raw_p.append(float(recomputed_test["p_raw"]))
        require(
            np.isclose(
                float(saved["mean_effect"]),
                float(recomputed_test["mean_effect"]),
                atol=1e-12,
            )
            and np.isclose(
                float(saved["p_raw"]),
                float(recomputed_test["p_raw"]),
                atol=1e-12,
            )
            and np.isclose(float(saved["ci_low"]), ci_low, atol=1e-12)
            and np.isclose(float(saved["ci_high"]), ci_high, atol=1e-12)
            and np.isfinite([ci_low, ci_high]).all()
            and ci_low <= ci_high,
            f"{contrast} sign-flip result and bootstrap CI reproduce exactly",
            checks,
        )
    recomputed_holm = holm_adjust(recomputed_raw_p)
    saved_holm = (
        primary.set_index("comparison_id")
        .loc[config["statistics"]["primary_contrasts"], "p_holm"]
        .to_numpy(dtype=np.float64)
    )
    require(
        np.allclose(saved_holm, recomputed_holm, atol=1e-12, rtol=0.0),
        "Holm-adjusted p values reproduce exactly",
        checks,
    )

    results_summary = read_json(results_summary_path)
    summary_hashes_match = True
    for file_entry in results_summary.get("files", {}).values():
        file_path = Path(file_entry["path"])
        if (
            not file_path.exists()
            or sha256_file(file_path) != file_entry["sha256"]
        ):
            summary_hashes_match = False
            break
    require(
        summary_hashes_match,
        "results summary hashes match all reported tables",
        checks,
    )
    require(
        all(
            path.exists()
            for path in [
                PROJECT_ROOT
                / "outputs"
                / "figures"
                / "rq3_global_mean_replacement_performance.png",
                PROJECT_ROOT
                / "outputs"
                / "figures"
                / "rq3_global_mean_replacement_performance.pdf",
                PROJECT_ROOT
                / "outputs"
                / "paper"
                / "methods_results_cn.md",
                PROJECT_ROOT
                / "outputs"
                / "paper"
                / "methods_results_en.md",
            ]
        ),
        "paper-ready figure and bilingual methods/results files exist",
        checks,
    )

    report_path = validation_dir / "validation_report.json"
    report = {
        "experiment_id": config["experiment_id"],
        "passed": True,
        "num_checks": int(len(checks)),
        "num_passed": int(sum(bool(row["passed"]) for row in checks)),
        "max_absolute_full_probability_difference": float(
            max(baseline_max_differences)
        ),
        "checks": checks,
    }
    write_json(report_path, report)
    manifest_path = PROJECT_ROOT / "outputs" / "checksums.sha256"
    num_manifest_files = write_checksum_manifest(PROJECT_ROOT, manifest_path)
    report["checksum_manifest"] = {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "num_files": int(num_manifest_files),
    }
    write_json(report_path, report)
    print(
        f"Validation passed: {report['num_passed']}/{report['num_checks']} checks; "
        f"manifest contains {num_manifest_files} files.",
        flush=True,
    )


if __name__ == "__main__":
    main()
