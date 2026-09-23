from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch

from .common import (
    CODE_ROOT, CONFIG_ROOT,
    PROJECT_ROOT,
    add_workspace_to_path,
    build_numerical_dependency_anchor,
    load_config,
    read_json,
    save_nifti,
    sha256_array,
    sha256_file,
    sha256_payload,
    stable_sample_id,
    verify_frozen_static_dependencies,
    write_csv,
    write_json,
)


FIXED_CONDITIONS = ("full", "cam_top20", "value", "visual", "union")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fold-nested training-CAM voxel-top-20% and predefined-ROI "
            "training-fold global-scalar mean replacement on the fixed k=6 CNN."
        )
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--only-fold",
        type=int,
        default=None,
        help="Run one fold for diagnostics; default runs all ten folds.",
    )
    parser.add_argument("--force-cam", action="store_true")
    parser.add_argument("--force-inference", action="store_true")
    return parser


def batched_indices(length: int, batch_size: int) -> Iterable[range]:
    for start in range(0, int(length), int(batch_size)):
        yield range(start, min(start + int(batch_size), int(length)))


def stack_items(dataset: Any, indices: Sequence[int]) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    items = [dataset[int(index)] for index in indices]
    x = torch.stack([item["x"] for item in items], dim=0)
    y = torch.stack([item["y"] for item in items], dim=0).long()
    return items, x, y


def load_binary_mask(path: Path, expected_shape: Sequence[int]) -> np.ndarray:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj)
    if tuple(int(value) for value in data.shape) != tuple(int(value) for value in expected_shape):
        raise ValueError(f"Mask shape mismatch for {path}: {data.shape} vs {tuple(expected_shape)}")
    return np.asarray(np.isfinite(data) & (data > 0), dtype=bool)


def exact_top_fraction_mask(
    cam_map: np.ndarray,
    common_mask: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}.")
    flat_cam = np.asarray(cam_map, dtype=np.float32).reshape(-1)
    flat_common = np.asarray(common_mask, dtype=bool).reshape(-1)
    common_indices = np.flatnonzero(flat_common).astype(np.int64, copy=False)
    num_selected = int(math.ceil(float(fraction) * int(common_indices.size)))
    values = flat_cam[common_indices]
    # Primary key: descending CAM; deterministic tie-break: ascending flat index.
    order = np.lexsort((common_indices, -values))
    selected_indices = common_indices[order[:num_selected]]
    selected = np.zeros(flat_cam.shape, dtype=bool)
    selected[selected_indices] = True
    selected = selected.reshape(common_mask.shape)
    selected_values = flat_cam[selected_indices]
    metadata = {
        "selection_rule": "exact_count_highest_values_with_flat_index_tie_break",
        "fraction_requested": float(fraction),
        "num_common_mask_voxels": int(common_indices.size),
        "num_selected_voxels": int(num_selected),
        "fraction_observed": float(num_selected / common_indices.size),
        "minimum_selected_cam": float(selected_values.min()),
        "maximum_selected_cam": float(selected_values.max()),
        "num_positive_selected": int(np.sum(selected_values > 0.0)),
        "mask_array_sha256": sha256_array(selected.astype(np.uint8)),
    }
    return selected, metadata


def compute_training_cam_and_global_mean(
    *,
    model: torch.nn.Module,
    train_dataset: Any,
    train_records: Sequence[Any],
    common_mask: np.ndarray,
    reference_img: nib.spatialimages.SpatialImage,
    device: torch.device,
    batch_size: int,
    top_fraction: float,
    fold_dir: Path,
    fold_index: int,
    dependency_signature: str,
    dependency_payload: Mapping[str, Any],
) -> tuple[float, np.ndarray, dict[str, Any]]:
    from analyses.layercam.gradcam_report import compute_layercam_batch, positive_saliency_volume

    shape = tuple(int(value) for value in reference_img.shape)
    num_subjects = int(len({str(record.subject_id) for record in train_records}))
    global_sum = 0.0
    global_voxel_count = 0
    cam_sums: dict[tuple[str, int], np.ndarray] = {}
    cam_counts: Counter[tuple[str, int]] = Counter()
    total_correct = 0
    correct_by_target: Counter[int] = Counter()
    start_time = time.time()

    model.eval()
    for batch_number, indices in enumerate(
        batched_indices(len(train_dataset), int(batch_size)),
        start=1,
    ):
        index_list = list(indices)
        items, x_cpu, y_cpu = stack_items(train_dataset, index_list)

        for local_index, dataset_index in enumerate(index_list):
            record = train_records[int(dataset_index)]
            brain_mask = train_dataset._load_mask(Path(record.mask_path))
            values = np.asarray(
                x_cpu[local_index, 0].numpy()[brain_mask], dtype=np.float64
            )
            if values.size <= 0 or not bool(np.all(np.isfinite(values))):
                raise ValueError(
                    f"Training input has no finite masked voxels: {record.sample_path}"
                )
            global_sum += float(values.sum(dtype=np.float64))
            global_voxel_count += int(values.size)

        x_device = x_cpu.to(device)
        y_device = y_cpu.to(device)
        with torch.enable_grad():
            cam_output = compute_layercam_batch(model, x_device, y_device)
        preds = cam_output["preds"].detach().cpu().numpy().astype(np.int64)
        cams = cam_output["cams"].detach().cpu().numpy().astype(np.float32)
        targets = y_cpu.numpy().astype(np.int64)

        for local_index, dataset_index in enumerate(index_list):
            if int(preds[local_index]) != int(targets[local_index]):
                continue
            record = train_records[int(dataset_index)]
            key = (str(record.subject_id), int(targets[local_index]))
            cam = positive_saliency_volume(cams[local_index], brain_mask=common_mask)
            if key not in cam_sums:
                cam_sums[key] = np.zeros(shape, dtype=np.float32)
            cam_sums[key] += cam
            cam_counts[key] += 1
            total_correct += 1
            correct_by_target[int(targets[local_index])] += 1

        del cam_output, x_device, y_device, x_cpu, y_cpu, cams
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        if batch_number == 1 or batch_number % 25 == 0:
            elapsed = time.time() - start_time
            processed = min(batch_number * int(batch_size), len(train_dataset))
            print(
                f"[fold {fold_index:02d}] training CAM {processed}/{len(train_dataset)} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    if global_voxel_count <= 0:
        raise ValueError("Cannot compute a global replacement mean from zero voxels.")
    training_global_mean = float(global_sum / float(global_voxel_count))
    if not math.isfinite(training_global_mean):
        raise ValueError("Training-fold global replacement mean is not finite.")

    class_maps: dict[int, np.ndarray] = {}
    class_subjects: dict[int, list[str]] = {}
    participant_class_rows: list[dict[str, Any]] = []
    for target in (0, 1):
        subjects = sorted(
            subject
            for subject, candidate_target in cam_sums
            if int(candidate_target) == int(target) and cam_counts[(subject, target)] > 0
        )
        if not subjects:
            raise ValueError(f"No correctly classified training CAMs for target {target}.")
        subject_maps = []
        for subject in subjects:
            count = int(cam_counts[(subject, target)])
            subject_map = np.asarray(cam_sums[(subject, target)] / float(count), dtype=np.float32)
            subject_maps.append(subject_map)
            participant_class_rows.append(
                {
                    "fold_index": int(fold_index),
                    "subject_id": subject,
                    "target": int(target),
                    "rating": 1 if target == 0 else 4,
                    "num_correct_training_inputs": count,
                }
            )
        class_maps[target] = np.asarray(
            np.mean(np.stack(subject_maps, axis=0), axis=0),
            dtype=np.float32,
        )
        class_maps[target][~common_mask] = 0.0
        class_subjects[target] = subjects

    class_brain_means = {
        target: float(np.mean(class_maps[target][common_mask]))
        for target in (0, 1)
    }
    if any(value <= 0.0 for value in class_brain_means.values()):
        raise ValueError(
            f"Class CAM brain means must be positive, got {class_brain_means}."
        )
    normalized_class_maps = {
        target: np.asarray(
            class_maps[target] / class_brain_means[target],
            dtype=np.float32,
        )
        for target in (0, 1)
    }
    joint_cam = np.asarray(
        0.5 * (normalized_class_maps[0] + normalized_class_maps[1]),
        dtype=np.float32,
    )
    joint_cam[~common_mask] = 0.0
    cam_top20_mask, selection_metadata = exact_top_fraction_mask(
        joint_cam,
        common_mask,
        float(top_fraction),
    )

    masks_dir = fold_dir / "masks"
    maps_dir = fold_dir / "maps"
    global_mean_path = write_json(
        fold_dir / "metadata" / "training_global_mean.json",
        {
            "fold_index": int(fold_index),
            "value": float(training_global_mean),
            "definition": (
                "one pooled whole-brain scalar over FMRIBetaDataset-processed "
                "training inputs, restricted to each input's participant brain mask"
            ),
            "aggregation": "pooled_voxel_mean_without_coordinatewise_map",
            "num_training_inputs": int(len(train_dataset)),
            "num_training_subjects": int(num_subjects),
            "num_training_brain_voxels": int(global_voxel_count),
            "sum_training_brain_voxels": float(global_sum),
            "scalar_sha256": sha256_array(
                np.asarray([training_global_mean], dtype=np.float64)
            ),
        },
    )
    save_nifti(
        maps_dir / "training_cam_rating1_participant_mean.nii.gz",
        class_maps[0],
        reference_img,
    )
    save_nifti(
        maps_dir / "training_cam_rating4_participant_mean.nii.gz",
        class_maps[1],
        reference_img,
    )
    save_nifti(
        maps_dir / "training_cam_rating1_brainmean_normalized.nii.gz",
        normalized_class_maps[0],
        reference_img,
    )
    save_nifti(
        maps_dir / "training_cam_rating4_brainmean_normalized.nii.gz",
        normalized_class_maps[1],
        reference_img,
    )
    save_nifti(
        maps_dir / "training_cam_joint_class_balanced.nii.gz",
        joint_cam,
        reference_img,
    )
    save_nifti(
        masks_dir / "cam_top20_mask.nii.gz",
        cam_top20_mask.astype(np.uint8),
        reference_img,
        dtype=np.uint8,
    )
    write_csv(
        fold_dir / "metadata" / "training_cam_counts.csv",
        participant_class_rows,
        [
            "fold_index",
            "subject_id",
            "target",
            "rating",
            "num_correct_training_inputs",
        ],
    )

    metadata = {
        "fold_index": int(fold_index),
        "dependency_signature": str(dependency_signature),
        "dependency_payload": dict(dependency_payload),
        "num_training_inputs": int(len(train_dataset)),
        "num_training_subjects": int(num_subjects),
        "training_global_mean": {
            "value": float(training_global_mean),
            "is_scalar": True,
            "definition": (
                "one pooled whole-brain scalar over FMRIBetaDataset-processed "
                "training inputs, restricted to each input's participant brain mask"
            ),
            "aggregation": "pooled_voxel_mean_without_coordinatewise_map",
            "num_training_brain_voxels": int(global_voxel_count),
            "scalar_sha256": sha256_array(
                np.asarray([training_global_mean], dtype=np.float64)
            ),
        },
        "cam": {
            "correct_training_inputs_only": True,
            "num_correct_training_inputs": int(total_correct),
            "num_correct_rating1": int(correct_by_target[0]),
            "num_correct_rating4": int(correct_by_target[1]),
            "num_subjects_with_correct_rating1": int(len(class_subjects[0])),
            "num_subjects_with_correct_rating4": int(len(class_subjects[1])),
            "participant_equal_weight": True,
            "class_equal_weight": True,
            "class_normalization_before_joint_map": "divide_each_class_map_by_its_common_brain_mean",
            "rating1_common_brain_mean_before_normalization": float(
                class_brain_means[0]
            ),
            "rating4_common_brain_mean_before_normalization": float(
                class_brain_means[1]
            ),
            "joint_cam_array_sha256": sha256_array(joint_cam),
        },
        "top20_selection": selection_metadata,
        "elapsed_seconds": float(time.time() - start_time),
        "files": {
            "training_global_mean": str(global_mean_path.resolve()),
            "rating1_cam": str(
                (maps_dir / "training_cam_rating1_participant_mean.nii.gz").resolve()
            ),
            "rating4_cam": str(
                (maps_dir / "training_cam_rating4_participant_mean.nii.gz").resolve()
            ),
            "rating1_cam_normalized": str(
                (maps_dir / "training_cam_rating1_brainmean_normalized.nii.gz").resolve()
            ),
            "rating4_cam_normalized": str(
                (maps_dir / "training_cam_rating4_brainmean_normalized.nii.gz").resolve()
            ),
            "joint_cam": str(
                (maps_dir / "training_cam_joint_class_balanced.nii.gz").resolve()
            ),
            "cam_top20_mask": str(
                (masks_dir / "cam_top20_mask.nii.gz").resolve()
            ),
        },
    }
    write_json(fold_dir / "metadata" / "training_artifacts.json", metadata)
    return training_global_mean, cam_top20_mask, metadata


def load_training_artifacts(
    fold_dir: Path,
    expected_shape: Sequence[int],
    expected_dependency_signature: str,
    expected_dependency_payload: Mapping[str, Any],
) -> tuple[float, np.ndarray, dict[str, Any]]:
    metadata = read_json(fold_dir / "metadata" / "training_artifacts.json")
    if (
        str(metadata.get("dependency_signature", ""))
        != str(expected_dependency_signature)
    ):
        raise ValueError(
            "Saved training artifacts are not bound to the current frozen dependencies."
        )
    if metadata.get("dependency_payload") != dict(expected_dependency_payload):
        raise ValueError(
            "Saved training-artifact dependency payload differs from the "
            "current frozen dependencies."
        )
    global_mean_payload = read_json(fold_dir / "metadata" / "training_global_mean.json")
    training_global_mean = float(global_mean_payload["value"])
    global_mean_metadata = metadata["training_global_mean"]
    cam_mask = load_binary_mask(
        fold_dir / "masks" / "cam_top20_mask.nii.gz",
        expected_shape,
    )
    scalar_hash = sha256_array(
        np.asarray([training_global_mean], dtype=np.float64)
    )
    if (
        not bool(global_mean_metadata.get("is_scalar", False))
        or not math.isfinite(training_global_mean)
        or global_mean_metadata.get("aggregation")
        != "pooled_voxel_mean_without_coordinatewise_map"
        or scalar_hash != global_mean_metadata.get("scalar_sha256")
        or scalar_hash != global_mean_payload.get("scalar_sha256")
    ):
        raise ValueError("Saved training global mean does not match scalar metadata.")
    if (
        sha256_array(cam_mask.astype(np.uint8))
        != metadata["top20_selection"]["mask_array_sha256"]
    ):
        raise ValueError("Saved CAM top-20% mask hash does not match metadata.")
    if (
        int(cam_mask.sum())
        != int(metadata["top20_selection"]["num_selected_voxels"])
        or float(metadata["top20_selection"]["fraction_requested"])
        != float(expected_dependency_payload["cam_config"]["top_fraction"])
    ):
        raise ValueError(
            "Saved CAM top-20% mask count/fraction does not match metadata."
        )
    return training_global_mean, cam_mask, metadata


def generate_random_indices(
    *,
    common_mask: np.ndarray,
    num_selected: int,
    num_replicates: int,
    base_seed: int,
    fold_index: int,
    output_path: Path,
    dependency_signature: str,
    dependency_payload: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    common_indices = np.flatnonzero(common_mask.reshape(-1)).astype(np.int64, copy=False)
    if int(num_selected) > int(common_indices.size):
        raise ValueError("Random-mask selection exceeds common-mask size.")
    fold_seed = int(base_seed) + int(fold_index) * 1_000_003
    rng = np.random.Generator(np.random.PCG64(fold_seed))
    selected = np.empty((int(num_replicates), int(num_selected)), dtype=np.uint32)
    for replicate in range(int(num_replicates)):
        row = rng.choice(common_indices, size=int(num_selected), replace=False)
        selected[replicate] = np.sort(row).astype(np.uint32, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        flat_indices=selected,
        base_seed=np.asarray([int(base_seed)], dtype=np.int64),
        fold_seed=np.asarray([int(fold_seed)], dtype=np.int64),
    )
    os.replace(temporary, output_path)
    metadata = {
        "sampling": "uniform_without_replacement_inside_common_brain_mask",
        "generator": "numpy.random.Generator(PCG64)",
        "base_seed": int(base_seed),
        "fold_seed": int(fold_seed),
        "num_replicates": int(num_replicates),
        "num_voxels_per_mask": int(num_selected),
        "indices_array_sha256": sha256_array(selected),
        "dependency_signature": str(dependency_signature),
        "dependency_payload": dict(dependency_payload),
        "file": str(output_path.resolve()),
        "file_sha256": sha256_file(output_path),
    }
    return selected, metadata


def apply_global_mean_replacement(
    x: torch.Tensor,
    flat_indices: np.ndarray,
    training_global_mean: float,
) -> torch.Tensor:
    modified = x.clone()
    flat = modified.reshape(modified.shape[0], -1)
    index_tensor = torch.as_tensor(
        np.asarray(flat_indices, dtype=np.int64),
        dtype=torch.long,
        device=flat.device,
    )
    flat[:, index_tensor] = float(training_global_mean)
    return modified


def audit_global_mean_replacement(
    original: torch.Tensor,
    modified: torch.Tensor,
    flat_indices: np.ndarray,
    training_global_mean: float,
) -> dict[str, Any]:
    expected = original.clone()
    expected_flat = expected.reshape(expected.shape[0], -1)
    index_tensor = torch.as_tensor(
        np.asarray(flat_indices, dtype=np.int64),
        dtype=torch.long,
        device=expected_flat.device,
    )
    expected_flat[:, index_tensor] = float(training_global_mean)
    modified_flat = modified.reshape(modified.shape[0], -1)
    selected_equal_scalar = bool(
        torch.equal(
            modified_flat[:, index_tensor],
            expected_flat[:, index_tensor],
        )
    )
    full_tensor_matches_expected = bool(torch.equal(modified, expected))
    if not selected_equal_scalar or not full_tensor_matches_expected:
        raise ValueError(
            "Global-mean replacement audit failed: selected values or untouched "
            "values differ from the defined scalar intervention."
        )
    return {
        "num_selected_voxels": int(index_tensor.numel()),
        "selected_values_equal_one_global_scalar": True,
        "unselected_values_unchanged": True,
        "full_tensor_matches_defined_intervention": True,
    }


def predict_probabilities(
    model: torch.nn.Module,
    x_cpu: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    with torch.inference_mode():
        probabilities = model(x_cpu.to(device))
    return probabilities.detach().cpu().numpy().astype(np.float32)


def evaluate_test_fold(
    *,
    model: torch.nn.Module,
    test_dataset: Any,
    test_records: Sequence[Any],
    training_global_mean: float,
    masks: Mapping[str, np.ndarray],
    random_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    random_chunk_size: int,
    probability_threshold: float,
    fold_dir: Path,
    fold_index: int,
    existing_oof_predictions_path: Path,
    dependency_signature: str,
    dependency_payload: Mapping[str, Any],
) -> dict[str, Any]:
    num_test = int(len(test_dataset))
    num_random = int(random_indices.shape[0])
    condition_probabilities = {
        condition: np.zeros(num_test, dtype=np.float32)
        for condition in FIXED_CONDITIONS
    }
    random_probabilities = np.zeros((num_random, num_test), dtype=np.float32)
    targets = np.zeros(num_test, dtype=np.int64)
    flat_mask_indices = {
        condition: np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1)).astype(
            np.int64,
            copy=False,
        )
        for condition, mask in masks.items()
    }

    sample_rows: list[dict[str, Any]] = []
    replacement_audits: dict[str, dict[str, Any]] = {}
    start_time = time.time()
    for batch_number, indices in enumerate(
        batched_indices(num_test, int(batch_size)),
        start=1,
    ):
        index_list = list(indices)
        items, x_cpu, y_cpu = stack_items(test_dataset, index_list)
        targets[index_list] = y_cpu.numpy().astype(np.int64)

        condition_probabilities["full"][index_list] = predict_probabilities(
            model,
            x_cpu,
            device,
        )
        for condition in ("cam_top20", "value", "visual", "union"):
            modified = apply_global_mean_replacement(
                x_cpu,
                flat_mask_indices[condition],
                training_global_mean,
            )
            if batch_number == 1:
                replacement_audits[condition] = audit_global_mean_replacement(
                    x_cpu,
                    modified,
                    flat_mask_indices[condition],
                    training_global_mean,
                )
            condition_probabilities[condition][index_list] = predict_probabilities(
                model,
                modified,
                device,
            )
            del modified

        for random_start in range(0, num_random, int(random_chunk_size)):
            random_stop = min(random_start + int(random_chunk_size), num_random)
            modified_batches = [
                apply_global_mean_replacement(
                    x_cpu,
                    np.asarray(random_indices[replicate], dtype=np.int64),
                    training_global_mean,
                )
                for replicate in range(random_start, random_stop)
            ]
            if batch_number == 1 and random_start == 0:
                replacement_audits["random_top20_replicate_001"] = (
                    audit_global_mean_replacement(
                        x_cpu,
                        modified_batches[0],
                        np.asarray(random_indices[0], dtype=np.int64),
                        training_global_mean,
                    )
                )
            stacked = torch.cat(modified_batches, dim=0)
            probabilities = predict_probabilities(model, stacked, device).reshape(
                random_stop - random_start,
                len(index_list),
            )
            random_probabilities[random_start:random_stop, index_list] = probabilities
            del modified_batches, stacked, probabilities

        for local_index, dataset_index in enumerate(index_list):
            record = test_records[int(dataset_index)]
            for condition in FIXED_CONDITIONS:
                probability = float(condition_probabilities[condition][int(dataset_index)])
                sample_rows.append(
                    {
                        "sample_id": stable_sample_id(
                            record.subject_id,
                            int(record.rating),
                            record.trial_id,
                        ),
                        "fold_index": int(fold_index),
                        "subject_id": record.subject_id,
                        "session_id": record.session_id,
                        "run_id": record.run_id,
                        "trial_id": record.trial_id,
                        "rating": int(record.rating),
                        "target": int(targets[int(dataset_index)]),
                        "condition": condition,
                        "prob_high": probability,
                        "pred_target": int(probability >= float(probability_threshold)),
                        "sample_path": record.sample_path,
                    }
                )

        del items, x_cpu, y_cpu
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elapsed = time.time() - start_time
        print(
            f"[fold {fold_index:02d}] inference "
            f"{min(batch_number * int(batch_size), num_test)}/{num_test} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    predictions_dir = fold_dir / "predictions"
    fixed_csv = write_csv(
        predictions_dir / "fixed_condition_predictions.csv",
        sample_rows,
        [
            "sample_id",
            "fold_index",
            "subject_id",
            "session_id",
            "run_id",
            "trial_id",
            "rating",
            "target",
            "condition",
            "prob_high",
            "pred_target",
            "sample_path",
        ],
    )
    random_npz_path = predictions_dir / "random_top20_predictions.npz"
    temporary = random_npz_path.with_suffix(random_npz_path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        probabilities=random_probabilities,
        targets=targets,
        subject_ids=np.asarray(
            [str(record.subject_id) for record in test_records],
            dtype="<U16",
        ),
        sample_ids=np.asarray(
            [
                stable_sample_id(
                    record.subject_id,
                    int(record.rating),
                    record.trial_id,
                )
                for record in test_records
            ],
            dtype="<U48",
        ),
        ratings=np.asarray([int(record.rating) for record in test_records], dtype=np.int8),
        trial_ids=np.asarray(
            [-1 if record.trial_id is None else int(record.trial_id) for record in test_records],
            dtype=np.int32,
        ),
        fold_index=np.asarray([int(fold_index)], dtype=np.int16),
    )
    os.replace(temporary, random_npz_path)

    existing_rows: dict[tuple[int, str, int, int], dict[str, str]] = {}
    with existing_oof_predictions_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["fold_index"]),
                str(row["subject_id"]),
                int(row["true_rating"]),
                int(float(row["trial_id"])),
            )
            existing_rows[key] = row
    label_mismatches = 0
    probability_differences: list[float] = []
    missing_existing: list[tuple[int, str, int, int]] = []
    for dataset_index, record in enumerate(test_records):
        key = (
            int(fold_index),
            str(record.subject_id),
            int(record.rating),
            -1 if record.trial_id is None else int(record.trial_id),
        )
        old = existing_rows.get(key)
        if old is None:
            missing_existing.append(key)
            continue
        observed_probability = float(condition_probabilities["full"][dataset_index])
        observed_prediction = int(observed_probability >= float(probability_threshold))
        old_prediction = int(old["pred_target"])
        if observed_prediction != old_prediction:
            label_mismatches += 1
        probability_differences.append(abs(observed_probability - float(old["prob_high"])))
    if missing_existing:
        raise ValueError(
            f"Could not match {len(missing_existing)} samples to existing OOF predictions; "
            f"first={missing_existing[:3]}"
        )
    if label_mismatches:
        raise ValueError(
            f"Full-input predictions differ from the frozen OOF predictions for "
            f"{label_mismatches} samples in fold {fold_index}."
        )
    expected_audit_conditions = {
        "cam_top20",
        "value",
        "visual",
        "union",
        "random_top20_replicate_001",
    }
    if set(replacement_audits) != expected_audit_conditions:
        raise ValueError(
            "Global-mean replacement audit is incomplete: "
            f"observed={sorted(replacement_audits)}, "
            f"expected={sorted(expected_audit_conditions)}"
        )

    metadata = {
        "fold_index": int(fold_index),
        "dependency_signature": str(dependency_signature),
        "dependency_payload": dict(dependency_payload),
        "num_test_inputs": int(num_test),
        "num_random_replicates": int(num_random),
        "conditions": list(FIXED_CONDITIONS),
        "probability_threshold": float(probability_threshold),
        "replacement": {
            "training_global_mean": float(training_global_mean),
            "assignment": "every_selected_voxel_receives_the_same_scalar",
            "renormalized_after_replacement": False,
            "elementwise_audits": replacement_audits,
        },
        "baseline_reproduction": {
            "matched_samples": int(len(probability_differences)),
            "prediction_label_mismatches": int(label_mismatches),
            "max_absolute_probability_difference": float(
                max(probability_differences) if probability_differences else 0.0
            ),
            "mean_absolute_probability_difference": float(
                np.mean(probability_differences) if probability_differences else 0.0
            ),
        },
        "outputs": {
            "fixed_predictions_csv": str(fixed_csv),
            "fixed_predictions_sha256": sha256_file(fixed_csv),
            "random_predictions_npz": str(random_npz_path.resolve()),
            "random_predictions_sha256": sha256_file(random_npz_path),
        },
        "elapsed_seconds": float(time.time() - start_time),
    }
    write_json(fold_dir / "metadata" / "inference.json", metadata)
    return metadata


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    add_workspace_to_path(config)

    from foodpref.dataset import FMRIBetaDataset, discover_samples, filter_records_by_subjects
    from analyses.layercam.gradcam_subject_tfce import (
        load_common_mask,
        load_model_for_fold,
    )
    from foodpref.train import select_device

    provenance_path = PROJECT_ROOT / "inputs" / "provenance.json"
    if not provenance_path.exists():
        raise FileNotFoundError(
            f"{provenance_path} is missing. Run code/prepare_inputs.py first."
        )
    provenance = read_json(provenance_path)
    verify_frozen_static_dependencies(config, provenance)
    frozen_definitions = provenance["frozen_definitions"]
    run_dir = Path(config["cnn_run_dir"]).expanduser().resolve()
    live_run_config_path = run_dir / "run_config.json"
    live_fold_assignments_path = run_dir / "fold_assignments.json"
    frozen_fold_assignments_path = PROJECT_ROOT / "inputs" / "fold_assignments.json"
    if (
        sha256_file(config["beta_index_csv"])
        != provenance["inputs"]["beta_index_sha256"]
    ):
        raise ValueError("The live beta index differs from frozen provenance.")
    if (
        sha256_file(live_run_config_path)
        != provenance["inputs"]["run_config_sha256"]
    ):
        raise ValueError("The live CNN run config differs from frozen provenance.")
    if (
        sha256_file(live_fold_assignments_path)
        != provenance["inputs"]["fold_assignments_sha256"]
    ):
        raise ValueError("The live fold assignments differ from frozen provenance.")
    if (
        sha256_file(frozen_fold_assignments_path)
        != frozen_definitions["frozen_fold_assignments_sha256"]
    ):
        raise ValueError(
            "The fold assignments actually used by this experiment differ "
            "from frozen provenance."
        )
    if read_json(live_fold_assignments_path) != read_json(
        frozen_fold_assignments_path
    ):
        raise ValueError("Live and frozen fold-assignment payloads differ.")
    if (
        sha256_file(config["existing_oof_predictions"])
        != provenance["inputs"]["existing_oof_predictions_sha256"]
    ):
        raise ValueError("The frozen OOF predictions file has changed.")
    run_config = read_json(run_dir / "run_config.json")
    fold_assignments = read_json(frozen_fold_assignments_path)
    sample_manifest_path = PROJECT_ROOT / "inputs" / "sample_manifest.csv"
    if (
        sha256_file(sample_manifest_path)
        != provenance["inputs"]["sample_manifest_sha256"]
    ):
        raise ValueError("The sample manifest differs from frozen provenance.")
    records, dataset_summary = discover_samples(
        beta_index_csv=str(config["beta_index_csv"]),
        models_root=str(config["models_root"]),
        brain_mask_root=str(config["brain_mask_root"]),
        path_prefix_from=str(config["path_prefix_from"]),
        path_prefix_to=str(config["path_prefix_to"]),
    )
    if any(int(value) != 0 for value in dataset_summary["drop_counts"].values()):
        raise ValueError(
            f"Sample discovery dropped frozen inputs: {dataset_summary['drop_counts']}"
        )
    if (
        int(dataset_summary["samples_kept"])
        != int(config["expected"]["num_samples"])
        or int(dataset_summary["subjects_kept"])
        != int(config["expected"]["num_subjects"])
    ):
        raise ValueError("Discovered data do not match the frozen sample counts.")
    frozen_rows: list[dict[str, str]] = []
    with sample_manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        frozen_rows = list(csv.DictReader(handle))
    if len(frozen_rows) != int(config["expected"]["num_samples"]):
        raise ValueError("Frozen sample manifest has the wrong number of rows.")
    file_hash_cache: dict[str, str] = {}

    def cached_file_hash(path_text: str) -> str:
        resolved = str(Path(path_text).expanduser().resolve())
        if resolved not in file_hash_cache:
            file_hash_cache[resolved] = sha256_file(resolved)
        return file_hash_cache[resolved]

    content_fingerprint_rows: list[dict[str, Any]] = []
    frozen_sample_ids: list[str] = []
    discovered_sample_ids: list[str] = []
    for index, (row, record) in enumerate(zip(frozen_rows, records)):
        observed_sample_id = stable_sample_id(
            record.subject_id,
            int(record.rating),
            record.trial_id,
        )
        frozen_sample_id = str(row["sample_id"])
        frozen_sample_ids.append(frozen_sample_id)
        discovered_sample_ids.append(observed_sample_id)
        sample_path = str(Path(record.sample_path).expanduser().resolve())
        mask_path = str(Path(record.mask_path).expanduser().resolve())
        frozen_trial_id = (
            -1
            if str(row["trial_id"]).strip() == ""
            else int(float(row["trial_id"]))
        )
        if (
            int(row["sample_index"]) != index
            or observed_sample_id != frozen_sample_id
            or str(row["subject_id"]) != str(record.subject_id)
            or str(row["session_id"]) != str(record.session_id)
            or str(row["run_id"]) != str(record.run_id)
            or frozen_trial_id
            != (-1 if record.trial_id is None else int(record.trial_id))
            or int(row["rating"]) != int(record.rating)
            or str(Path(row["sample_path"]).expanduser().resolve()) != sample_path
            or str(Path(row["mask_path"]).expanduser().resolve()) != mask_path
        ):
            raise ValueError(
                f"Discovered sample row {index} differs from the frozen manifest."
            )
        sample_size = int(Path(sample_path).stat().st_size)
        mask_size = int(Path(mask_path).stat().st_size)
        sample_hash = cached_file_hash(sample_path)
        mask_hash = cached_file_hash(mask_path)
        if (
            sample_size != int(row["sample_size_bytes"])
            or mask_size != int(row["mask_size_bytes"])
            or sample_hash != str(row["sample_sha256"])
            or mask_hash != str(row["mask_sha256"])
        ):
            raise ValueError(
                f"Frozen beta-map or participant-mask content changed for "
                f"{frozen_sample_id}."
            )
        content_fingerprint_rows.append(
            {
                "sample_id": frozen_sample_id,
                "sample_path": sample_path,
                "sample_size_bytes": sample_size,
                "sample_sha256": sample_hash,
                "mask_path": mask_path,
                "mask_size_bytes": mask_size,
                "mask_sha256": mask_hash,
            }
        )
    if (
        len(frozen_sample_ids) != len(set(frozen_sample_ids))
        or discovered_sample_ids != frozen_sample_ids
        or sha256_payload(content_fingerprint_rows)
        != provenance["inputs"]["sample_content_fingerprint_sha256"]
    ):
        raise ValueError("Discovered samples differ from the frozen sample manifest.")
    reference_img, common_mask, _ = load_common_mask(records)
    if sha256_array(common_mask.astype(np.uint8)) != provenance["common_mask"]["array_sha256"]:
        raise ValueError("Current common mask differs from frozen provenance.")

    fixed_masks = {
        name: load_binary_mask(
            PROJECT_ROOT / "outputs" / "masks" / f"{name}_mask.nii.gz",
            reference_img.shape,
        )
        for name in ("value", "visual", "union")
    }
    for name, mask in fixed_masks.items():
        mask_path = PROJECT_ROOT / "outputs" / "masks" / f"{name}_mask.nii.gz"
        expected_hash = (
            provenance["fixed_roi_masks"][name]["mask_sha256"]
        )
        if (
            sha256_array(mask.astype(np.uint8)) != expected_hash
            or sha256_file(mask_path)
            != frozen_definitions["fixed_mask_file_sha256"][name]
        ):
            raise ValueError(f"The frozen {name} mask has changed.")
    common_mask_file = PROJECT_ROOT / "outputs" / "masks" / "common_brain_mask.nii.gz"
    if (
        sha256_file(common_mask_file)
        != frozen_definitions["common_mask_file_sha256"]
    ):
        raise ValueError("The frozen common-mask file has changed.")
    atlas_root = Path(config["atlas_root"]).expanduser().resolve()
    for relative_path, expected_hash in frozen_definitions[
        "atlas_sha256"
    ].items():
        if sha256_file(atlas_root / relative_path) != expected_hash:
            raise ValueError(f"Frozen atlas asset changed: {relative_path}")
    device_name = str(config["inference"]["device"])
    device = select_device(device_name)
    num_random = int(config["random_control"]["num_replicates"])
    print(
        f"Device={device}; folds={len(fold_assignments['folds'])}; "
        f"random masks per fold={num_random}.",
        flush=True,
    )
    numerical_dependency_anchor = build_numerical_dependency_anchor(provenance)

    processed_folds: list[dict[str, Any]] = []
    for fold in fold_assignments["folds"]:
        fold_index = int(fold["fold_index"])
        if args.only_fold is not None and fold_index != int(args.only_fold):
            continue
        fold_dir = PROJECT_ROOT / "outputs" / "folds" / f"fold_{fold_index:02d}"
        for child in ("maps", "masks", "metadata", "predictions"):
            (fold_dir / child).mkdir(parents=True, exist_ok=True)

        train_subjects = [str(value) for value in fold["train_subjects"]]
        test_subjects = [str(value) for value in fold["val_subjects"]]
        if set(train_subjects) & set(test_subjects):
            raise ValueError(f"Fold {fold_index} has train/test subject overlap.")
        train_records = filter_records_by_subjects(records, train_subjects)
        test_records = filter_records_by_subjects(records, test_subjects)
        train_dataset = FMRIBetaDataset(train_records)
        test_dataset = FMRIBetaDataset(test_records)
        checkpoint_path = (
            run_dir
            / f"fold_{fold_index:02d}"
            / str(config["checkpoint_name"])
        )
        expected_checkpoint_hash = provenance["inputs"]["checkpoint_sha256"][
            f"fold_{fold_index:02d}"
        ]
        if sha256_file(checkpoint_path) != expected_checkpoint_hash:
            raise ValueError(
                f"Fold {fold_index:02d} checkpoint differs from frozen provenance."
            )
        train_sample_ids = [
            stable_sample_id(
                record.subject_id,
                int(record.rating),
                record.trial_id,
            )
            for record in train_records
        ]
        test_sample_ids = [
            stable_sample_id(
                record.subject_id,
                int(record.rating),
                record.trial_id,
            )
            for record in test_records
        ]
        training_dependency_payload = {
            "schema": "rq3_training_artifacts_global_scalar/v3",
            "fold_index": int(fold_index),
            "checkpoint_sha256": expected_checkpoint_hash,
            "execution_device": str(device),
            "numerical_dependency_anchor": numerical_dependency_anchor,
            "train_subjects": train_subjects,
            "train_sample_ids": train_sample_ids,
            "cam_config": config["cam"],
            "replacement_config": config["replacement"],
        }
        training_dependency_signature = sha256_payload(
            training_dependency_payload
        )
        model, checkpoint = load_model_for_fold(
            checkpoint_path=checkpoint_path,
            run_config=run_config,
            device=device,
        )

        training_artifacts_path = fold_dir / "metadata" / "training_artifacts.json"
        if training_artifacts_path.exists() and not args.force_cam:
            training_global_mean, cam_top20_mask, training_metadata = load_training_artifacts(
                fold_dir,
                reference_img.shape,
                training_dependency_signature,
                training_dependency_payload,
            )
            print(f"[fold {fold_index:02d}] reusing training CAM artifacts.", flush=True)
        else:
            training_global_mean, cam_top20_mask, training_metadata = (
                compute_training_cam_and_global_mean(
                    model=model,
                    train_dataset=train_dataset,
                    train_records=train_records,
                    common_mask=common_mask,
                    reference_img=reference_img,
                    device=device,
                    batch_size=int(config["cam"]["batch_size"]),
                    top_fraction=float(config["cam"]["top_fraction"]),
                    fold_dir=fold_dir,
                    fold_index=fold_index,
                    dependency_signature=training_dependency_signature,
                    dependency_payload=training_dependency_payload,
                )
            )

        num_selected = int(cam_top20_mask.sum())
        random_dependency_payload = {
            "schema": "rq3_random_masks/v2",
            "fold_index": int(fold_index),
            "common_mask_array_sha256": provenance["common_mask"][
                "array_sha256"
            ],
            "num_selected_voxels": int(num_selected),
            "random_control_config": config["random_control"],
        }
        random_dependency_signature = sha256_payload(
            random_dependency_payload
        )
        random_indices_path = fold_dir / "masks" / "random_top20_flat_indices.npz"
        random_metadata_path = fold_dir / "metadata" / "random_masks.json"
        reuse_random = (
            random_indices_path.exists()
            and random_metadata_path.exists()
            and not args.force_inference
        )
        if reuse_random:
            with np.load(random_indices_path, allow_pickle=False) as payload:
                random_indices = np.asarray(payload["flat_indices"], dtype=np.uint32)
            random_metadata = read_json(random_metadata_path)
            if (
                random_indices.shape
                != (int(num_random), int(num_selected))
                or sha256_array(random_indices)
                != random_metadata["indices_array_sha256"]
                or int(random_metadata.get("base_seed", -1))
                != int(config["random_control"]["seed"])
                or int(random_metadata.get("fold_seed", -1))
                != int(config["random_control"]["seed"])
                + int(fold_index) * 1_000_003
                or str(random_metadata.get("dependency_signature", ""))
                != random_dependency_signature
                or random_metadata.get("dependency_payload")
                != random_dependency_payload
                or sha256_file(random_indices_path)
                != str(random_metadata.get("file_sha256", ""))
            ):
                reuse_random = False
            else:
                flat_common = common_mask.reshape(-1)
                if not bool(np.all(flat_common[random_indices.astype(np.int64)])):
                    reuse_random = False
                elif any(
                    np.unique(row).size != int(num_selected)
                    for row in random_indices
                ):
                    reuse_random = False
        if not reuse_random:
            random_indices, random_metadata = generate_random_indices(
                common_mask=common_mask,
                num_selected=num_selected,
                num_replicates=num_random,
                base_seed=int(config["random_control"]["seed"]),
                fold_index=fold_index,
                output_path=random_indices_path,
                dependency_signature=random_dependency_signature,
                dependency_payload=random_dependency_payload,
            )
            write_json(random_metadata_path, random_metadata)

        force_inference = bool(args.force_inference or args.force_cam)
        inference_dependency_payload = {
            "schema": "rq3_inference_global_scalar/v3",
            "fold_index": int(fold_index),
            "checkpoint_sha256": expected_checkpoint_hash,
            "execution_device": str(device),
            "numerical_dependency_anchor": numerical_dependency_anchor,
            "training_dependency_signature": training_dependency_signature,
            "training_global_mean_sha256": training_metadata["training_global_mean"][
                "scalar_sha256"
            ],
            "cam_top20_mask_sha256": training_metadata["top20_selection"][
                "mask_array_sha256"
            ],
            "random_dependency_signature": random_dependency_signature,
            "random_indices_sha256": random_metadata[
                "indices_array_sha256"
            ],
            "fixed_mask_sha256": {
                name: provenance["fixed_roi_masks"][name]["mask_sha256"]
                for name in ("value", "visual", "union")
            },
            "test_subjects": test_subjects,
            "test_sample_ids": test_sample_ids,
            "inference_config": config["inference"],
            "num_random_replicates": int(num_random),
        }
        inference_dependency_signature = sha256_payload(
            inference_dependency_payload
        )
        inference_path = fold_dir / "metadata" / "inference.json"
        if inference_path.exists() and not force_inference:
            inference_metadata = read_json(inference_path)
            if (
                str(inference_metadata.get("dependency_signature", ""))
                != inference_dependency_signature
                or inference_metadata.get("dependency_payload")
                != inference_dependency_payload
            ):
                raise ValueError(
                    f"Fold {fold_index} inference cache does not match the "
                    "current frozen dependencies; pass --force-inference."
                )
            if int(inference_metadata["num_random_replicates"]) != int(num_random):
                raise ValueError(
                    f"Fold {fold_index} was completed with "
                    f"{inference_metadata['num_random_replicates']} random masks, "
                    f"not {num_random}. Pass --force-inference to replace it."
                )
            for path_key, hash_key in (
                ("fixed_predictions_csv", "fixed_predictions_sha256"),
                ("random_predictions_npz", "random_predictions_sha256"),
            ):
                if (
                    sha256_file(inference_metadata["outputs"][path_key])
                    != inference_metadata["outputs"][hash_key]
                ):
                    raise ValueError(
                        f"Fold {fold_index} cached {path_key} failed its "
                        "frozen output hash check."
                    )
            print(f"[fold {fold_index:02d}] reusing completed inference.", flush=True)
        else:
            condition_masks = {
                "cam_top20": cam_top20_mask,
                **fixed_masks,
            }
            inference_metadata = evaluate_test_fold(
                model=model,
                test_dataset=test_dataset,
                test_records=test_records,
                training_global_mean=training_global_mean,
                masks=condition_masks,
                random_indices=random_indices,
                device=device,
                batch_size=int(config["inference"]["batch_size"]),
                random_chunk_size=int(
                    config["random_control"]["inference_chunk_size"]
                ),
                probability_threshold=float(
                    config["inference"]["probability_threshold"]
                ),
                fold_dir=fold_dir,
                fold_index=fold_index,
                existing_oof_predictions_path=Path(
                    config["existing_oof_predictions"]
                ).expanduser().resolve(),
                dependency_signature=inference_dependency_signature,
                dependency_payload=inference_dependency_payload,
            )

        fold_complete = {
            "fold_index": int(fold_index),
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "num_train_inputs": int(len(train_records)),
            "num_test_inputs": int(len(test_records)),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_reported_epoch": int(
                checkpoint.get("reported_epoch", checkpoint.get("epoch", 0))
            ),
            "execution_device": str(device),
            "training_artifacts": training_metadata,
            "random_masks": random_metadata,
            "inference": inference_metadata,
            "leakage_checks": {
                "train_test_subject_overlap": sorted(
                    set(train_subjects) & set(test_subjects)
                ),
                "training_cam_uses_test_subjects": False,
                "training_global_mean_uses_test_subjects": False,
                "random_masks_use_test_data_or_labels": False,
            },
            "dependency_signatures": {
                "training_artifacts": training_dependency_signature,
                "random_masks": random_dependency_signature,
                "inference": inference_dependency_signature,
            },
        }
        write_json(fold_dir / "fold_complete.json", fold_complete)
        processed_folds.append(fold_complete)

        del model, train_dataset, test_dataset, training_global_mean, cam_top20_mask
        gc.collect()
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        print(f"[fold {fold_index:02d}] complete.", flush=True)

    fold_status = []
    for expected_fold in range(1, int(config["expected"]["num_folds"]) + 1):
        complete_path = (
            PROJECT_ROOT
            / "outputs"
            / "folds"
            / f"fold_{expected_fold:02d}"
            / "fold_complete.json"
        )
        if not complete_path.exists():
            fold_status.append(
                {
                    "fold_index": expected_fold,
                    "complete": False,
                    "marker_present": False,
                    "dependency_signatures_match": False,
                }
            )
            continue
        payload = read_json(complete_path)
        signatures = payload.get("dependency_signatures", {})
        signatures_match = bool(
            signatures.get("training_artifacts")
            == payload.get("training_artifacts", {}).get(
                "dependency_signature"
            )
            and signatures.get("random_masks")
            == payload.get("random_masks", {}).get("dependency_signature")
            and signatures.get("inference")
            == payload.get("inference", {}).get("dependency_signature")
        )
        random_count = int(
            payload.get("inference", {}).get("num_random_replicates", -1)
        )
        is_complete = bool(
            signatures_match
            and random_count
            == int(config["random_control"]["num_replicates"])
        )
        fold_status.append(
            {
                "fold_index": expected_fold,
                "complete": is_complete,
                "marker_present": True,
                "dependency_signatures_match": signatures_match,
                "num_test_inputs": int(payload["num_test_inputs"]),
                "num_random_replicates": random_count,
            }
        )
    run_summary = {
        "experiment_id": config["experiment_id"],
        "device": str(device),
        "num_random_replicates": int(num_random),
        "folds_processed_this_call": [
            int(payload["fold_index"]) for payload in processed_folds
        ],
        "fold_status": fold_status,
        "all_expected_folds_present": bool(
            len(fold_status) == int(config["expected"]["num_folds"])
            and all(row["complete"] for row in fold_status)
        ),
        "dataset_summary": dataset_summary,
        "config_sha256": sha256_file(config["_config_path"]),
    }
    write_json(PROJECT_ROOT / "outputs" / "pipeline_status.json", run_summary)
    print(
        f"Completed folds: {run_summary['folds_processed_this_call']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
