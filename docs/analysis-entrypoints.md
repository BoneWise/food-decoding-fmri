# Analysis entrypoints

Run these commands from the repository root after installing the project. The
analysis modules keep the original calculations and use the `foodpref` package
for sample loading, models and metrics. The dated source locations and separate
original/migrated file hashes are recorded in the repository source manifest.

## Local assets and outputs

Set `FOODPREF_DATA_ROOT` to your prepared data directory. Its default is `data/`.
The examples below use this structure:

```text
data/
  derived/beta_index.csv, models/, cache/masks/
  derived/sessionwise/beta_index.csv, models/, cache/masks/
  derived/merged/k06_seed42/beta_index.csv, models/, merge_manifest.json
outputs/cnn/
  <timestamp>_<new CNN run name>/
outputs/runs/
  <restored original CNN run>/run_config.json, fold_assignments.json, fold_01/final.pt, ...
  <matched SVM run>/features_cnn_zscore_commonmask.npy, sample_index.csv, ...
```

`FOODPREF_RUNS_ROOT` overrides `outputs/runs/`.
`FOODPREF_ANALYSIS_OUTPUT_ROOT` overrides `outputs/analyses/`.
`FSLDIR` selects your FSL installation; `FOODPREF_ATLAS_ROOT` can independently
select its atlas directory. FSL `randomise` and Harvard–Oxford max-probability
25%/2 mm atlases are needed for the corresponding analyses. Source beta maps,
individual masks, CNN checkpoints and the SVM feature matrix are external assets.
Their absence does not prevent any `--help` command.

For an index copied from another machine, either produce a local index with the
same rows or pass supported `--path-prefix-from` / `--path-prefix-to` options.
These options change path resolution only. GLM and input QC require the `beta_file`
column to resolve on the local machine (absolute paths or paths relative to the
working directory). The merged input index must preserve the original sample
composition and order. Do not regenerate participant folds with a new random
split when reproducing the reference results.

## GLM and session-based decoding

The located historical `no_quality_filter` GLM used **minimum quality score
0.0**, 5,000 permutations, seed 42 and 31 included participants. Its common mask
contained 222,441 voxels. This is a usable archived configuration, but its
significant-voxel count does not establish the manuscript claim of 473 voxels.
Likewise, the located searchlight summaries have not established the manuscript
claim of 31,431 voxels in seven components. The commands below reproduce the
specified analysis protocols; they are not presented as verification of those
two manuscript counts. The archived CNN, matched cross-participant SVM and
within-participant decoding use their separately documented analysis sets.

```bash
python -m analyses.glm.fwe_beta_randomise \
  --min-quality-score 0 --permutations 5000 --seed 42

python -m analyses.searchlight.validate_sessionwise_searchlight_inputs

python -m analyses.searchlight.paper_searchlight_mvpa \
  --feature-mask outputs/analyses/glm/group_inputs/group_mask_intersection.nii.gz \
  --cv-group session --required-num-groups 3 --radius-mm 6 \
  --svm-backend libsvm_linear --svm-c 1 --class-weight balanced \
  --run-randomise --permutations 5000 --seed 42

python -m analyses.decoding.linear_whole_brain_mvpa_leave_session_out \
  --feature-mask outputs/analyses/glm/group_inputs/group_mask_intersection.nii.gz \
  --required-num-sessions 3 --svm-backend libsvm_linear \
  --svm-c 1 --class-weight balanced --seed 42
```

Use `--beta-index-csv`, `--models-root` and `--brain-mask-root` when your directory
layout differs. Searchlight and within-participant decoding leave out an entire
BIDS session, including all of its runs. Their train-fold standardization remains
enabled. The earlier run-holdout whole-brain script is not the primary entrypoint.

## Layer-CAM and regional summaries

Use a completed **k=6** CNN run with its fixed folds and `final.pt` files.
New training writes `outputs/cnn/<timestamp>_<run_name>/`; pass that actual
directory to `--run-dir`. Restored archived runs may instead be placed under
`outputs/runs/`. The example below uses a new-run placeholder that must be
replaced by the completed run directory.
`gradcam_subject_tfce` is the manuscript spatial-analysis entrypoint. The preserved
`gradcam_report` module also provides earlier descriptive reports and shared CAM
utilities.

```bash
python -m analyses.layercam.gradcam_subject_tfce \
  --run-dir outputs/cnn/REPLACE_WITH_COMPLETED_K06_RUN \
  --beta-index-csv data/derived/merged/k06_seed42/beta_index.csv --models-root data/derived/merged/k06_seed42/models \
  --brain-mask-root data/derived/cache/masks \
  --output-dir outputs/analyses/layercam_k06 \
  --checkpoint-name final.pt --permutations 5000 --seed 42

python -m analyses.layercam.regional_means \
  --layercam-dir outputs/analyses/layercam_k06
```

Both CAM entrypoints accept the three explicit dataset paths above, together
with optional path-prefix remapping. This allows the original run configuration
to remain unchanged when data move to another machine.

The regional report is **mean Layer-CAM value (whole-brain mean = 1)**. Correct
held-out inputs are averaged within participant and rating, then across
participants. Each region's group mean is divided by that rating's group-map
whole-brain mean. Macroregions use voxel-count-weighted ROI means, following the
existing atlas grouping. This order is preserved: averaging separately normalized
participant maps would produce different values. Historical ROI spellings,
including duplicated left/right tokens inherited from the atlas export, are
retained as stable row identifiers. Figure display labels can be cleaner.

## Input replacement and Haufe analyses

Portable copies of the original experiment settings live under
`configs/analyses/{input_replacement,haufe,haufe_comparison}/`. JSON path tokens
`{repo}`, `{data}`, `{runs}`, `{outputs}` and `{atlas}` expand to the roots above.
For a local experiment copy, set `FOODPREF_INPUT_REPLACEMENT_CONFIG`,
`FOODPREF_HAUFE_CONFIG`, or `FOODPREF_HAUFE_COMPARISON_CONFIG` to its JSON path.
Input-replacement scripts also accept `--config` directly. These paths affect
execution inputs, and do not rewrite fixed reference results.

The input-replacement default device is portable CPU; a local config can select
CUDA or MPS. The archived experiment used MPS. Exact floating-point predictions
can vary across backends, so compare a rerun with the fixed reference outputs.

```bash
python -m analyses.input_replacement.prepare_inputs
python -m analyses.input_replacement.run_ablation
python -m analyses.input_replacement.validate_results --raw-only
python -m analyses.input_replacement.summarize_ablation
python -m analyses.input_replacement.validate_results

python -m analyses.haufe.prepare_inputs
python -m analyses.haufe.run_haufe
python -m analyses.haufe.validate_results --raw-only
python -m analyses.haufe.summarize_haufe
python -m analyses.haufe.validate_results

python -m analyses.haufe_comparison.prepare_inputs
python -m analyses.haufe_comparison.run_comparison
python -m analyses.haufe_comparison.validate_results
```

New RQ3 inputs and outputs are written under
`outputs/analyses/<analysis>/inputs/` and `outputs/analyses/<analysis>/outputs/`.
For Haufe–CAM comparison, adjust the comparison JSON to point to your generated
Layer-CAM directory or restored group maps. The default RQ3 configs retain original run names under the portable run
root for **restoring archived experiments**. They do not automatically select a
new run from `outputs/cnn/`; local configs must explicitly point to its completed
run and matching assets. A newly trained model is a new experiment, even when
its hyperparameters and fixed participant folds match the archived run.

The input-replacement configuration retains the original expected k=6 beta-index
SHA-256 and all count checks (1,147 inputs, 30 participants, ten folds). A locally
rewritten index has a different hash. To reproduce from a relocated index, create
a separate local experiment config, verify the row identities/order and sample
content against the reference metadata, then record that local index hash under
`expected.beta_index_sha256`. Do not label a changed index as byte-identical to
the original. Each new `prepare_inputs` run freezes its current inputs and current
code; all downstream checksum checks remain active.

Original validation reports certify their original code and inputs. They do not
certify these migrated source files. The migration manifest keeps original file
hashes distinct from current file hashes. RQ3 validators are intended for freshly
produced outputs, not for copying an old 'passed' report into a new run directory.

## Checks performed during packaging

- All migrated analysis entrypoints returned `--help` without accessing datasets.
- Synthetic checks cover session isolation, train-only standardization,
  group-first ROI normalization, voxel-weighted macroregions and portable paths.
- Regional summary arithmetic was checked against the existing group maps and
  ROI tables; it reproduced the corrected manuscript regional values.
- No CNN fitting, full searchlight, permutation inference, input-replacement
  inference or SVM/Haufe refitting was rerun while packaging this repository.
