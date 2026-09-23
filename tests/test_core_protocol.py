"""Protocol-level tests use synthetic volumes and never train a model."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest
import nibabel as nib

torch = pytest.importorskip("torch")

from foodpref.cli import parse_with_config
from foodpref.config import parse_args
from foodpref.cv import build_subject_folds, load_subject_folds
from foodpref.dataset import FMRIBetaDataset, FMRISample
from foodpref.matched_svm import build_or_load_feature_matrix
from foodpref.merge_betas import build_merge_groups
from foodpref.train import BalancedBinaryBatchSampler

ROOT = Path(__file__).resolve().parents[1]


def records_for_subjects(subject_ids, per_class=5):
    return [
        FMRISample(f"{subject}/rating-{rating}/beta-{trial}.nii", "mask.nii", subject,
                   "ses-01", "run-01", trial, rating)
        for subject in subject_ids for rating in (1, 4) for trial in range(per_class)
    ]


def test_subject_folds_never_split_a_participant():
    records = records_for_subjects([f"sub-{i:03d}" for i in range(1, 31)])
    folds = build_subject_folds(records, num_folds=10, seed=42)
    subjects = {record.subject_id for record in records}
    seen = []
    for fold in folds:
        assert set(fold.train_subjects).isdisjoint(fold.val_subjects)
        assert set(fold.train_subjects) | set(fold.val_subjects) == subjects
        assert len(fold.val_subjects) == 3
        seen.extend(fold.val_subjects)
    assert Counter(seen) == Counter({subject: 1 for subject in subjects})
    assert folds == build_subject_folds(list(reversed(records)), num_folds=10, seed=42)


@pytest.mark.parametrize("k", [1, 3, 6])
def test_frozen_folds_cover_each_subject_once(k):
    path = ROOT / f"configs/decoding/fold_assignments_k{k:02d}.json"
    raw = json.loads(path.read_text())
    subject_ids = sorted(set(raw["folds"][0]["train_subjects"] + raw["folds"][0]["val_subjects"]))
    assert len(subject_ids) == 30
    assert len(load_subject_folds(path, records_for_subjects(subject_ids))) == 10


def test_frozen_fold_leakage_is_rejected(tmp_path):
    path = tmp_path / "folds.json"
    path.write_text(json.dumps({"folds": [
        {"fold_index": 1, "train_subjects": ["sub-1", "sub-2"], "val_subjects": ["sub-2"]}
    ]}))
    with pytest.raises(ValueError, match="overlap"):
        load_subject_folds(path, records_for_subjects(["sub-1", "sub-2"]))


def test_merge_groups_preserve_subject_class_and_nonoverlap():
    records = records_for_subjects(["sub-1", "sub-2"], per_class=8)
    groups = build_merge_groups(records, k=3, seed=42)
    assert len(groups) == 8  # two complete groups per subject and rating
    used = []
    for group in groups:
        assert len(group.records) == 3
        assert {r.subject_id for r in group.records} == {group.subject_id}
        assert {r.rating for r in group.records} == {group.rating}
        used.extend(r.sample_path for r in group.records)
    assert len(used) == len(set(used))
    assert groups == build_merge_groups(list(reversed(records)), k=3, seed=42)


def synthetic_record(tmp_path, volume, brain_mask):
    beta_path, mask_path = tmp_path / "beta.nii.gz", tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(volume.astype(np.float32), np.eye(4)), beta_path)
    nib.save(nib.Nifti1Image(brain_mask.astype(np.uint8), np.eye(4)), mask_path)
    return FMRISample(str(beta_path), str(mask_path), "sub-1", "ses-01", "run-01", 1, 1)


def test_samplewise_normalization_uses_only_finite_brain_voxels(tmp_path):
    volume = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    mask = np.ones_like(volume, dtype=bool)
    mask[0, 0, 0] = False
    volume[0, 0, 0] = 1e8  # must not influence in-brain normalization
    volume[0, 0, 1] = np.nan  # one invalid / 26 brain voxels is below 5%
    record = synthetic_record(tmp_path, volume, mask)
    result = FMRIBetaDataset([record])[0]["x"].numpy()[0]
    finite = mask & np.isfinite(volume)
    assert result.shape == volume.shape
    np.testing.assert_allclose(result[finite].mean(), 0, atol=1e-7)
    np.testing.assert_allclose(result[finite].std(), 1, atol=1e-7)
    assert np.all(result[~finite] == 0)
    assert np.all(np.isfinite(result))


def test_invalid_brain_fraction_is_rejected(tmp_path):
    volume = np.ones((3, 3, 3), dtype=np.float32)
    volume.flat[:2] = np.nan
    record = synthetic_record(tmp_path, volume, np.ones_like(volume, dtype=bool))
    with pytest.raises(ValueError, match="max_invalid_brain_fraction"):
        FMRIBetaDataset([record])[0]


def test_constant_volume_maps_to_zeros(tmp_path):
    record = synthetic_record(tmp_path, np.full((3, 3, 3), 4), np.ones((3, 3, 3), dtype=bool))
    assert torch.count_nonzero(FMRIBetaDataset([record])[0]["x"]) == 0


def test_matched_svm_features_equal_cnn_voxels(tmp_path):
    volume = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    brain = np.ones_like(volume, dtype=bool)
    brain[0, 0, 0] = False
    volume[0, 0, 0] = 1e8
    record = synthetic_record(tmp_path, volume, brain)
    feature_mask = brain.copy()
    feature_mask[2, :, :] = False  # normalization remains over the wider brain mask
    common_path = tmp_path / "common.nii.gz"
    nib.save(nib.Nifti1Image(feature_mask.astype(np.uint8), np.eye(4)), common_path)
    index = tmp_path / "index.csv"
    index.write_text("sub,rating_value\nsub-1,1\n")
    args = argparse.Namespace(beta_index_csv=str(index), feature_mask=str(common_path),
                              preprocessing="cnn_zscore", rebuild_features=False,
                              max_invalid_brain_fraction=0.05, log_every=100)
    matrix = build_or_load_feature_matrix(args=args, records=[record], feature_mask=feature_mask,
                                         output_dir=tmp_path / "features")
    cnn = FMRIBetaDataset([record])[0]["x"].numpy()[0]
    np.testing.assert_array_equal(matrix[0], cnn[feature_mask])


def test_balanced_batches_have_equal_class_counts():
    targets = [0] * 17 + [1] * 3
    sampler = BalancedBinaryBatchSampler(targets, batch_size=8, seed=42)
    batches = list(sampler)
    assert len(batches) == 5
    assert all(Counter(targets[index] for index in batch) == {0: 4, 1: 4} for batch in batches)
    assert set(index for batch in batches for index in batch) >= set(range(len(targets)))


def test_frozen_config_and_cli_override():
    config = parse_args(["--config", str(ROOT / "configs/decoding/cnn_k03.json"), "--num-workers", "0"])
    assert config.batch_size == 16
    assert config.epochs == 5
    assert config.dropout == 0.5
    assert config.num_workers == 0
    assert config.fold_assignments.endswith("fold_assignments_k03.json")


def test_config_supplies_required_arguments_and_rejects_typos(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"arguments": {"input": "table.csv"}}))
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    assert parse_with_config(parser, ["--config", str(config)]).input == "table.csv"
    config.write_text(json.dumps({"arguments": {"imput": "table.csv"}}))
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    with pytest.raises(SystemExit):
        parse_with_config(parser, ["--config", str(config)])


def test_svm_configs_use_the_archived_feature_mask():
    import hashlib
    paths = []
    for k in (1, 3, 6):
        config = json.loads((ROOT / f"configs/decoding/svm_k{k:02d}.json").read_text())
        path = ROOT / config["arguments"]["feature_mask"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config["protocol"]["feature_mask_sha256"]
        paths.append(path)
    assert len(set(paths)) == 1
    image = nib.load(paths[0])
    assert int(np.count_nonzero(np.asarray(image.dataobj) > 0)) == 222441
