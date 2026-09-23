"""Protocol checks using synthetic data; no fitting or source-data access."""
from pathlib import Path
import numpy as np
import pytest
from analyses._paths import DATA_ROOT, OUTPUT_ROOT, expand_paths, analysis_root
from analyses.layercam.regional_means import normalized_roi_rows, division_rows
from analyses.searchlight.paper_searchlight_mvpa import build_subject_design_matrix, prepare_fold_features
from foodpref.dataset import FMRISample


def test_session_grouping_keeps_all_runs_of_held_out_session_together():
    records = [FMRISample(f"{ses}/{run}/{rating}.nii", "mask.nii", "participant", ses, run, None, rating)
               for ses in ("ses-03", "ses-01", "ses-02")
               for run in ("run-01", "run-02") for rating in (1, 4)]
    class Extractor:
        def extract_features(self, record, mask):
            return np.array([float(record.rating)])
    x, y, folds, keys = build_subject_design_matrix(records, feature_mask=np.ones((1, 1, 1), dtype=bool),
        extractor=Extractor(), log_every=100, subject_id="participant", cv_group="session")
    assert keys == [("ses-01",), ("ses-02",), ("ses-03",)]
    ordered = sorted(records, key=lambda r: (r.session_id, r.run_id, -1, r.sample_path))
    for i, (session,) in enumerate(keys):
        assert {r.session_id for j, r in enumerate(ordered) if folds[j] == i} == {session}
        assert session not in {r.session_id for j, r in enumerate(ordered) if folds[j] != i}
        assert set(y[folds == i]) == {0, 1}
        assert (folds == i).sum() == 4


def test_fold_standardization_uses_training_values():
    train, heldout = prepare_fold_features(np.array([[0.], [2.]]), np.array([[100.], [102.]]), fold_standardize=True)
    np.testing.assert_allclose(train[:, 0], [-1., 1.])
    np.testing.assert_allclose(heldout[:, 0], [99., 101.])


def test_regional_normalization_is_after_group_average():
    # Two participant maps [[1,3], [9,3]] -> group [5,3], group mean 4.
    group = np.array([[1., 3.], [9., 3.]]).mean(axis=0)
    rows = normalized_roi_rows([{"roi_name":"ROI", "division":"Division", "num_voxels":1,
                                "num_subjects":2, "mean_value":group[0]}], float(group.mean()))
    assert rows[0]["mean_layercam_whole_brain_1"] == 1.25
    assert rows[0]["mean_layercam_whole_brain_1"] != np.mean([1/2, 9/6])
    with pytest.raises(ValueError):
        normalized_roi_rows([], 0)


def test_macro_region_weights_roi_means_by_voxel_count():
    rows = [{"division":"D", "num_voxels":1, "mean_layercam":2.},
            {"division":"D", "num_voxels":3, "mean_layercam":6.}]
    assert division_rows(rows, 2.)[0]["mean_layercam_whole_brain_1"] == 2.5


def test_paths_expand_recursively_and_outputs_are_separate():
    data = expand_paths({"path":"{data}/k06/beta_index.csv", "other":["{outputs}/x", 42]})
    assert data["path"] == str(DATA_ROOT / "k06" / "beta_index.csv")
    assert data["other"] == [str(OUTPUT_ROOT / "x"), 42]
    assert analysis_root("haufe") == OUTPUT_ROOT / "haufe"
