"""Regression checks for the scientific aggregation and portable reference data."""
from pathlib import Path
import sys
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0,str(SCRIPTS))
from reproduce_results import (REFERENCE, binary_metrics, summarize_prediction_groups,
                               ensure_matched_inputs, region_tables, reproduce)
from validate_reference import validate


def test_balanced_accuracy_gives_equal_weight_to_classes():
    # Accuracy is 90%; BA is 50% for a constant low prediction on a 9:1 dataset.
    rows=[{'target':'0','pred':'0'}]*9+[{'target':'1','pred':'0'}]
    assert binary_metrics(rows,'target','pred')['balanced_accuracy']==.5
    with pytest.raises(ValueError,match='both target classes'):
        binary_metrics(rows[:9],'target','pred')


def test_participants_are_not_weighted_by_number_of_inputs():
    rows=[dict(subject_id='A',fold_index='1',target='0',pred='0'),
          dict(subject_id='A',fold_index='1',target='1',pred='1')]
    rows += [dict(subject_id='B',fold_index='2',target=str(i%2),pred=str(1-i%2)) for i in range(20)]
    participants=summarize_prediction_groups(rows,'target','pred')
    assert np.mean([r['balanced_accuracy'] for r in participants])==.5
    assert binary_metrics(rows,'target','pred')['balanced_accuracy']==pytest.approx(2/22)


def test_subject_cannot_cross_heldout_folds():
    rows=[dict(subject_id='A',fold_index=str(i+1),target=str(i),pred=str(i)) for i in range(2)]
    with pytest.raises(ValueError,match='more than one'):
        summarize_prediction_groups(rows,'target','pred')


def test_model_pairing_checks_samples_labels_and_fold_identity():
    cnn=[dict(subject_id='A',sample_path='sources/one/models/A/beta.nii',true_target='1',fold_index='1')]
    svm=[dict(subject_id='A',sample_path='sources/two/models/A/beta.nii',target='1',fold_index='1')]
    ensure_matched_inputs(cnn,svm)
    for field,value in [('target','0'),('fold_index','2'),('sample_path','sources/two/models/A/other.nii')]:
        bad=[dict(svm[0],**{field:value})]
        with pytest.raises(ValueError,match='differ'):ensure_matched_inputs(cnn,bad)
    with pytest.raises(ValueError,match='Duplicate'):ensure_matched_inputs(cnn+cnn,svm)


def test_group_first_normalization_keeps_correct_laterality():
    tables=region_tables(REFERENCE)
    roi={(r['rating'],r['roi_name']):r for r in tables['layercam_all_rois.csv']}
    assert roi[(4,'Right Precuneous Cortex')]['mean_layercam_whole_brain_1']==pytest.approx(1.124871312,abs=1e-8)
    assert roi[(1,'Left Precuneous Cortex')]['mean_layercam_whole_brain_1']==pytest.approx(1.413701617,abs=1e-8)
    for macro in tables['layercam_macroregions.csv']:
        rows=[r for r in roi.values() if r['rating']==macro['rating'] and r['division']==macro['division']]
        weighted=np.average([r['roi_mean_layercam'] for r in rows],weights=[r['num_voxels'] for r in rows])
        assert macro['mean_layercam_whole_brain_1']==pytest.approx(weighted/macro['whole_brain_mean_layercam'])


def test_reference_hashes_and_all_archived_arithmetic():
    result=validate(REFERENCE)
    assert result['passed'],result['failed']


def test_reproduction_never_overwrites_reference():
    with pytest.raises(ValueError,match='overwrite'):
        reproduce(REFERENCE,REFERENCE/'source')
