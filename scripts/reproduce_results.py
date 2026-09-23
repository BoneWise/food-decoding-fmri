#!/usr/bin/env python3
"""Rebuild paper tables from distributed predictions and group-map summaries.

No training, inference, bootstrap resampling, or image-level significance testing
is performed. Columns marked frozen come from the archived inference records.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'results' / 'reference'
DIVISIONS = ['Occipital', 'Temporal', 'Frontal', 'Parietal', 'Subcortical']


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def binary_metrics(rows, target_key, pred_key):
    target = np.asarray([int(row[target_key]) for row in rows])
    pred = np.asarray([int(row[pred_key]) for row in rows])
    if not len(target) or not set(target).issubset({0, 1}) or not set(pred).issubset({0, 1}):
        raise ValueError('Predictions must contain binary targets and labels')
    if set(target) != {0, 1}:
        raise ValueError('Balanced accuracy requires both target classes')
    tn = int(np.sum((target == 0) & (pred == 0)))
    fp = int(np.sum((target == 0) & (pred == 1)))
    fn = int(np.sum((target == 1) & (pred == 0)))
    tp = int(np.sum((target == 1) & (pred == 1)))
    return dict(num_examples=len(rows), tn=tn, fp=fp, fn=fn, tp=tp,
                balanced_accuracy=0.5 * (tn / (tn + fp) + tp / (tp + fn)))


def summarize_prediction_groups(rows, target_key, pred_key):
    groups = defaultdict(list)
    for row in rows:
        groups[row['subject_id']].append(row)
    result = []
    for subject, subset in sorted(groups.items()):
        folds = {row['fold_index'] for row in subset}
        if len(folds) != 1:
            raise ValueError(f'{subject} occurs in more than one held-out fold')
        result.append(dict(subject_id=subject, fold_index=int(next(iter(folds))),
                           **binary_metrics(subset, target_key, pred_key)))
    return result


def sample_identity(row):
    # The historical beta root changed; paths inside models/ identify inputs.
    model_path = row['sample_path'].split('/models/', 1)[-1]
    return row['subject_id'], model_path


def ensure_matched_inputs(cnn, svm):
    def index(rows, target_key):
        result = {}
        for row in rows:
            key = sample_identity(row)
            if key in result:
                raise ValueError(f'Duplicate sample identity: {key}')
            result[key] = int(row[target_key]), int(row['fold_index'])
        return result
    if index(cnn, 'true_target') != index(svm, 'target'):
        raise ValueError('CNN and SVM samples, target labels, or archived folds differ')


def performance_tables(reference):
    cnn_all = read_csv(reference / 'decoding/cnn/oof_predictions_k136.csv')
    archived = {int(item['k']): item for item in
                read_json(reference / 'decoding/cnn/analysis_summary.json')['summaries']}
    participant, performance, paired, folds_out = [], [], [], []
    for k in (1, 3, 6):
        cnn = [row for row in cnn_all if int(row['k']) == k]
        svm = read_csv(reference / f'decoding/svm/k{k:02d}/oof_predictions.csv')
        ensure_matched_inputs(cnn, svm)
        metrics = {}
        for model, rows, target, pred in [('CNN', cnn, 'true_target', 'pred_target'),
                                           ('SVM', svm, 'target', 'pred')]:
            values = summarize_prediction_groups(rows, target, pred)
            if len(values) != 30:
                raise ValueError(f'Expected 30 participants for {model} k={k}')
            metrics[model] = {row['subject_id']: row['balanced_accuracy'] for row in values}
            participant.extend(dict(k=k, model=model, **row) for row in values)
            by_fold = defaultdict(list)
            for row in rows:
                by_fold[int(row['fold_index'])].append(row)
            if set(by_fold) != set(range(1, 11)):
                raise ValueError('Expected ten archived held-out folds')
            folds_out.extend(dict(k=k, model=model, fold_index=f,
                                   **binary_metrics(group, target, pred))
                             for f, group in sorted(by_fold.items()))
            v = np.array(list(metrics[model].values()))
            if model == 'CNN':
                ci = archived[k]['participant_summary']['balanced_accuracy_vs_chance']['bootstrap_ci_95']
            else:
                summary = read_json(reference / f'decoding/svm/k{k:02d}/cv_summary.json')
                ci = summary['inference']['svm_subject_balanced_accuracy_vs_chance']['mean_bootstrap_95_ci']
            performance.append(dict(k=k, model=model, n_participants=len(v),
                num_inputs=len(rows), participant_mean_ba=float(v.mean()),
                participant_sd_sample=float(v.std(ddof=1)), ci_low_frozen=ci[0],
                ci_high_frozen=ci[1], ci_method='archived participant bootstrap 95%; not resampled'))
        subjects = sorted(metrics['CNN'])
        if subjects != sorted(metrics['SVM']):
            raise ValueError('CNN and SVM participants do not match')
        differences = np.array([metrics['SVM'][s] - metrics['CNN'][s] for s in subjects])
        test = stats.ttest_1samp(differences, 0)
        prior = archived[k]['cnn_svm_participant_comparison']['balanced_accuracy']
        ci = prior['bootstrap_ci_95_cnn_minus_svm']
        paired.append(dict(k=k, n_participants=len(subjects), direction='SVM minus CNN',
            mean_ba_difference=float(differences.mean()), sd_sample=float(differences.std(ddof=1)),
            paired_t=float(test.statistic), p_two_sided=float(test.pvalue),
            ci_low_frozen=-ci[1], ci_high_frozen=-ci[0],
            ci_method='archived paired participant bootstrap 95%; direction reversed'))
    return {'decoding_participants.csv': participant, 'decoding_performance.csv': performance,
            'decoding_paired.csv': paired, 'decoding_folds.csv': folds_out}


def region_tables(reference):
    mask_img = nib.load(reference / 'layercam/maps/common_brain_mask.nii.gz')
    mask = mask_img.get_fdata() > 0
    if not mask.any():
        raise ValueError('Empty common brain mask')
    rois, macros, top5, denominators = [], [], [], []
    for rating in (1, 4):
        image = nib.load(reference / f'layercam/maps/rating_{rating}_group_mean_layercam.nii.gz')
        if image.shape != mask.shape or not np.allclose(image.affine, mask_img.affine):
            raise ValueError('Layer-CAM map and mask spaces differ')
        values = image.get_fdata(dtype=np.float64)[mask]
        if not np.all(np.isfinite(values)):
            raise ValueError('Non-finite map values inside the mask')
        denominator = float(values.mean())
        if denominator <= 0:
            raise ValueError('Non-positive whole-brain map mean')
        denominators.append(dict(rating=rating, common_mask_voxels=int(mask.sum()),
                                  whole_brain_mean_layercam=denominator))
        source = read_csv(reference / f'layercam/source/rating_{rating}_roi_subject_stats.csv')
        names = [row['roi_name'] for row in source]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate atlas ROI label')
        normalized = []
        for row in source:
            normalized.append(dict(rating=rating, roi_name=row['roi_name'],
                division=row['division'], atlas_group=row['atlas_group'],
                num_voxels=int(row['num_voxels']), num_subjects=int(row['num_subjects']),
                roi_mean_layercam=float(row['mean_value']), whole_brain_mean_layercam=denominator,
                mean_layercam_whole_brain_1=float(row['mean_value']) / denominator))
        rois.extend(sorted(normalized, key=lambda row: (DIVISIONS.index(row['division']), row['roi_name'])))
        for division in DIVISIONS:
            group = [row for row in normalized if row['division'] == division]
            count = sum(row['num_voxels'] for row in group)
            mean = sum(row['num_voxels'] * row['roi_mean_layercam'] for row in group) / count
            macros.append(dict(rating=rating, division=division, num_voxels=count,
                division_mean_layercam=mean, whole_brain_mean_layercam=denominator,
                mean_layercam_whole_brain_1=mean / denominator))
            ranked = sorted(group, key=lambda row: (-row['mean_layercam_whole_brain_1'], row['roi_name']))[:5]
            top5.extend(dict(rank_within_division=i, **row) for i, row in enumerate(ranked, 1))
    return {'layercam_all_rois.csv': rois, 'layercam_macroregions.csv': macros,
            'layercam_top5_rois.csv': top5, 'layercam_whole_brain_means.csv': denominators}


def session_tables(reference):
    by_subject = defaultdict(list)
    fold_results = []
    for file in sorted((reference / 'session_holdout/subjects').glob('*/fold_*_predictions.csv')):
        rows = read_csv(file)
        subjects = {row['subject_id'] for row in rows}
        sessions = {row['session_id'] for row in rows}
        if len(subjects) != 1 or len(sessions) != 1:
            raise ValueError('Session prediction file mixes participants or sessions')
        subject, session = next(iter(subjects)), next(iter(sessions))
        metrics = binary_metrics(rows, 'target', 'pred')
        by_subject[subject].append((session, metrics['balanced_accuracy']))
        fold_results.append(dict(subject_id=subject, session_id=session, **metrics))
    rows = []
    for subject, values in sorted(by_subject.items()):
        if {session for session, _ in values} != {'ses-01', 'ses-02', 'ses-03'} or len(values) != 3:
            raise ValueError('Expected exactly three held-out sessions per participant')
        rows.append(dict(subject_id=subject, n_sessions=3,
                         mean_session_balanced_accuracy=float(np.mean([v for _, v in values]))))
    v = np.array([row['mean_session_balanced_accuracy'] for row in rows])
    test = stats.ttest_1samp(v, .5)
    summary = [dict(n_participants=len(v), participant_mean_ba=float(v.mean()),
        participant_sd_population=float(v.std(ddof=0)), participant_sd_sample=float(v.std(ddof=1)),
        t=float(test.statistic), p_two_sided=float(test.pvalue),
        sd_convention='archived paper summary uses population SD (ddof=0)')]
    return {'session_holdout_participants.csv': rows, 'session_holdout_folds.csv': fold_results,
            'session_holdout_summary.csv': summary}


def replacement_tables(reference):
    path = reference / 'input_replacement/tables'
    by_condition_subject = defaultdict(list)
    for row in read_csv(path / 'sample_predictions_fixed_conditions.csv'):
        by_condition_subject[(row['condition'], row['subject_id'])].append(row)
    participants = []
    for (condition, subject), rows in sorted(by_condition_subject.items()):
        participants.append(dict(condition=condition, subject_id=subject,
                                 **binary_metrics(rows, 'target', 'pred_target')))
    random_by_subject = defaultdict(list)
    for row in read_csv(path / 'participant_random_replicate_metrics.csv'):
        random_by_subject[row['subject_id']].append(row)
    for subject, rows in sorted(random_by_subject.items()):
        if {int(row['random_replicate']) for row in rows} != set(range(1, 101)) or len(rows) != 100:
            raise ValueError('Expected 100 random mask replicates per participant')
        participants.append(dict(condition='random_top20_mean', subject_id=subject,
            num_examples=int(rows[0]['n_samples']),
            tn=float(np.mean([float(r['tn']) for r in rows])),
            fp=float(np.mean([float(r['fp']) for r in rows])),
            fn=float(np.mean([float(r['fn']) for r in rows])),
            tp=float(np.mean([float(r['tp']) for r in rows])),
            balanced_accuracy=float(np.mean([float(r['balanced_accuracy']) for r in rows]))))
    archived = {row['condition']: row for row in read_csv(path / 'group_condition_summary.csv')}
    group = []
    for condition in archived:
        v = np.array([row['balanced_accuracy'] for row in participants if row['condition'] == condition])
        if len(v) != 30:
            raise ValueError(f'Expected 30 participants for replacement {condition}')
        group.append(dict(condition=condition, n_participants=len(v), mean_ba=float(v.mean()),
            sd_sample=float(v.std(ddof=1)),
            ci_low_frozen=float(archived[condition]['balanced_accuracy_ci_low']),
            ci_high_frozen=float(archived[condition]['balanced_accuracy_ci_high'])))
    return {'input_replacement_participants.csv': participants, 'input_replacement_summary.csv': group}


def reproduce(reference=REFERENCE, output=None):
    reference = Path(reference)
    tables = {}
    for function in (performance_tables, region_tables, session_tables, replacement_tables):
        tables.update(function(reference))
    # These inference/spatial tables are published unchanged; no re-estimation claim.
    for name, source in {
        'input_replacement_inference_frozen.csv': 'input_replacement/tables/primary_inference.csv',
        'haufe_layercam_overlap_frozen.csv': 'map_comparison/tables/top20_overlap_comparison.csv',
        'haufe_layercam_correlation_frozen.csv': 'map_comparison/tables/continuous_map_comparison.csv',
    }.items():
        tables[name] = read_csv(reference / source)
    if output is not None:
        output = Path(output)
        if output.resolve().is_relative_to(reference.resolve()):
            raise ValueError('Generated outputs must not overwrite the reference archive')
        output.mkdir(parents=True, exist_ok=True)
        for name, rows in tables.items():
            write_csv(output / name, rows)
        (output / 'reproduction_scope.json').write_text(json.dumps({
            'computed': 'participant/fold BA, participant paired differences and t tests; group-first regional normalization and voxel-weighted macroregions; session-holdout means; input-replacement participant and group BA',
            'frozen': 'bootstrap confidence intervals, replacement sign-flip/Holm inference, Haufe/Layer-CAM overlap and correlation tables',
            'not_run': 'training, model inference, TFCE/randomise, bootstrap/sign-flip resampling, surface plotting',
            'tables': sorted(tables)}, indent=2) + '\n')
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/reproduced/tables')
    args = parser.parse_args()
    tables = reproduce(args.reference, args.output_dir)
    print(f'Reproduced {len(tables)} tables in {args.output_dir}')


if __name__ == '__main__':
    main()
