#!/usr/bin/env python3
"""Validate distributed source hashes and independently rebuilt paper quantities."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np

from reproduce_results import ROOT, REFERENCE, read_csv, read_json, reproduce


def validate(reference=REFERENCE):
    reference = Path(reference)
    report = []
    def check(name, passed, detail=''):
        report.append(dict(check=name, passed=bool(passed), detail=detail))
    def close(name, actual, expected, atol=1e-10):
        check(name, np.isclose(float(actual), float(expected), rtol=0, atol=atol),
              f'actual={float(actual):.15g}; archived={float(expected):.15g}')
    manifest = read_json(reference / 'provenance.json')
    for entry in manifest['files']:
        rel = Path(entry['destination']).relative_to('results/reference')
        path = reference / rel
        if not path.is_file():
            check(f'file:{rel}', False, 'missing'); continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        check(f'hash:{rel}', digest == entry['distributed_sha256'])
        if path.suffix in {'.csv', '.json', '.md'}:
            check(f'portable_paths:{rel}', not re.search(r'/(?:Users|Volumes|mnt|home)/', path.read_text()))
    for entry in manifest.get('derived_files', []):
        rel = Path(entry['destination']).relative_to('results/reference')
        path = reference / rel
        check(f'derived_snapshot_hash:{rel}', path.is_file() and
              hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'])
    tables = reproduce(reference)
    frozen_cnn = {(int(r['k']), r['subject_id']): r for r in
                  read_csv(reference / 'decoding/cnn/participant_metrics_k136.csv')}
    frozen_svm = {k: {r['subject_id']:r for r in read_csv(reference / f'decoding/svm/k{k:02d}/subject_metrics.csv')}
                  for k in (1, 3, 6)}
    for row in tables['decoding_participants.csv']:
        frozen = (frozen_cnn[(row['k'], row['subject_id'])] if row['model'] == 'CNN'
                  else frozen_svm[row['k']][row['subject_id']])
        prefix = f"participant_BA:{row['model']}:k{row['k']}:{row['subject_id']}"
        close(prefix, row['balanced_accuracy'], frozen['balanced_accuracy'])
        for field in ['tn', 'fp', 'fn', 'tp', 'num_examples']:
            close(f'{prefix}:{field}', row[field], frozen[field], 0)
    summaries = {int(s['k']): s for s in read_json(reference / 'decoding/cnn/analysis_summary.json')['summaries']}
    for row in tables['decoding_performance.csv']:
        s = summaries[row['k']]['cnn_svm_participant_comparison']['balanced_accuracy']
        model = row['model'].lower()
        close(f"group_mean:{model}:k{row['k']}", row['participant_mean_ba'], s[f'{model}_mean'])
        close(f"group_SD:{model}:k{row['k']}", row['participant_sd_sample'], s[f'{model}_sd_sample'])
    for row in tables['decoding_paired.csv']:
        s = summaries[row['k']]['cnn_svm_participant_comparison']['balanced_accuracy']
        for field, expected in [('mean_ba_difference', -s['cnn_minus_svm_mean']),
                                ('paired_t', -s['paired_t']), ('p_two_sided', s['paired_t_p_two_sided'])]:
            close(f"paired:{row['k']}:{field}", row[field], expected)
    frozen_folds = {(int(r['k']), 'CNN', int(r['fold_index'])): r for r in
                    read_csv(reference / 'decoding/cnn/fold_metrics_k136.csv')}
    for k in (1, 3, 6):
        frozen_folds.update({(k, 'SVM', int(r['fold_index'])):r for r in
                            read_csv(reference / f'decoding/svm/k{k:02d}/fold_metrics.csv')})
    for row in tables['decoding_folds.csv']:
        key = row['k'], row['model'], row['fold_index']
        close(f'fold_BA:{key}', row['balanced_accuracy'], frozen_folds[key]['balanced_accuracy'])
    macro = {(int(r['rating']), r['division']):r for r in
             read_csv(reference / 'layercam/source/archived_macroregion_values.csv')}
    for row in tables['layercam_macroregions.csv']:
        key = row['rating'], row['division']
        close(f'macro_mean:{key}', row['mean_layercam_whole_brain_1'], macro[key]['E'])
        close(f'macro_voxels:{key}', row['num_voxels'], macro[key]['num_voxels'], 0)
    top5 = {(int(r['rating']), r['division'], r['roi_name']):r for r in
            read_csv(reference / 'layercam/source/archived_top5_roi_values.csv')}
    generated_keys = {(r['rating'], r['division'], r['roi_name']) for r in tables['layercam_top5_rois.csv']}
    check('top5_ROI_identity', generated_keys == set(top5))
    for row in tables['layercam_top5_rois.csv']:
        key = row['rating'], row['division'], row['roi_name']
        if key in top5:
            close(f'top5_value:{key}', row['mean_layercam_whole_brain_1'], top5[key]['E'])
    s = read_json(reference / 'session_holdout/group_summary.json')
    row = tables['session_holdout_summary.csv'][0]
    close('session_subject_count', row['n_participants'], s['num_subjects_included'], 0)
    close('session_mean', row['participant_mean_ba'], s['mean_fold_balanced_accuracy']['mean'])
    close('session_archived_population_SD', row['participant_sd_population'], s['mean_fold_balanced_accuracy']['std'])
    close('session_p_two_sided', row['p_two_sided'], s['ttest_mean_fold_balanced_accuracy_vs_chance_0p5']['p_value'])
    saved_subjects = {r['subject_id']:r for r in read_csv(reference / 'session_holdout/subject_metrics.csv')}
    for row in tables['session_holdout_participants.csv']:
        close(f"session_subject:{row['subject_id']}",row['mean_session_balanced_accuracy'],
              saved_subjects[row['subject_id']]['mean_fold_balanced_accuracy'])
    replacement = {(r['condition'],r['subject_id']):r for r in
                   read_csv(reference / 'input_replacement/tables/participant_condition_metrics.csv')}
    for row in tables['input_replacement_participants.csv']:
        key = row['condition'], row['subject_id']
        close(f'replacement_participant:{key}',row['balanced_accuracy'],replacement[key]['balanced_accuracy'])
    replacement_group = {r['condition']:r for r in
                         read_csv(reference / 'input_replacement/tables/group_condition_summary.csv')}
    for row in tables['input_replacement_summary.csv']:
        frozen = replacement_group[row['condition']]
        close(f"replacement_mean:{row['condition']}",row['mean_ba'],frozen['balanced_accuracy_mean'])
        close(f"replacement_SD:{row['condition']}",row['sd_sample'],frozen['balanced_accuracy_sd_sample'])
    return dict(passed=all(r['passed'] for r in report), checks=len(report),
                failed=[r for r in report if not r['passed']], results=report,
                scope='hash integrity and arithmetic reproduction from archived predictions/maps/tables; no experimental reruns')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--report', type=Path, default=ROOT/'outputs/validation/reference_validation.json')
    args = parser.parse_args()
    result = validate(args.reference)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(f"{result['checks']-len(result['failed'])}/{result['checks']} reference checks passed")
    if result['failed']:
        print(json.dumps(result['failed'],indent=2))
        raise SystemExit(1)

if __name__ == '__main__':
    main()
