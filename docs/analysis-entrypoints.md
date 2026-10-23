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

## GLM, searchlight and session-based decoding

The manuscript's univariate GLM (473 high>low TFCE-FWE voxels, none for low>high,
31 participants) and searchlight result (31,431 voxels; cluster-forming
p < .001, cluster-level FWE p < .05, 30 participants) are summarized in
`results/reference/glm/analysis_summary.json` and
`results/reference/searchlight/analysis_summary.json`. The 222,441-voxel common
feature mask used by every multivariate analysis is distributed as
`results/reference/decoding/masks/svm_common_mask.nii.gz`.

Univariate GLM: condition-wise first-level models on 6-mm smoothed BOLD (SPM25),
run-to-participant fixed effects, then a 31-participant TFCE-FWE group test.

```matlab
setenv('FOODPREF_FMRIPREP_ROOT', '/path/to/fmriprep');   % fMRIPrep derivatives + events/
setenv('FOODPREF_GLM_S6_ROOT', '/path/to/glm_s6_wmcsf'); % first-level outputs
setenv('SPM_DIR', '/path/to/spm25');
run('analyses/glm/first_level_condition_glm_s6.m');
run('analyses/glm/subject_fixed_effects.m');
```

```bash
FOODPREF_GLM_S6_ROOT=/path/to/glm_s6_wmcsf \
python -m analyses.glm.group_tfce_fwe --permutations 5000 --seed 42
```

Group outputs are written to
`$FOODPREF_RUNS_ROOT/glm_ua1/RESULTS/smoothed_s6_wmcsf_condition_glm_tfce_fwe/`.

Searchlight and within-participant decoding use the session-wise condition betas
(`preprocessing/run_rating14_sessionwise_glm.m`):

```bash
python -m analyses.searchlight.validate_sessionwise_searchlight_inputs

python -m analyses.searchlight.paper_searchlight_mvpa \
  --feature-mask results/reference/decoding/masks/svm_common_mask.nii.gz \
  --cv-group session --required-num-groups 3 --radius-mm 6 \
  --svm-backend libsvm_linear --svm-c 1 --class-weight balanced

python -m analyses.searchlight.group_cluster_fwe \
  --source-batch-root outputs/analyses/searchlight/<subject-level run directory> \
  --permutations 5000 --seed 42

python -m analyses.decoding.linear_whole_brain_mvpa_leave_session_out \
  --feature-mask results/reference/decoding/masks/svm_common_mask.nii.gz \
  --required-num-sessions 3 --svm-backend libsvm_linear \
  --svm-c 1 --class-weight balanced --seed 42
```

`group_cluster_fwe` excludes `sub-016`, tests balanced accuracy minus chance with
one-sided sign flips and writes to
`$FOODPREF_RUNS_ROOT/searchlight_mvpa1/RESULTS/searchlight_rating4_vs_rating1_ses3fold_n30_cluster_p001_cluster_fwe/`.

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

## Averaging granularity (SVM, k = 1-6)

`analyses/granularity/` refits the matched linear SVM for every k = 1-6 and
grouping seeds 42-46 on the fixed k = 1 participant partition, then tests each
fold's model at every k. It reproduces the frozen k = 1, 3, 6 predictions
exactly (`a90_validate_svm.py`, `a91_validate_k1.py`). The scripts are run from
their directory; `FOODPREF_GRANULARITY_ROOT` (default
`outputs/analyses/granularity/`) holds `cache/`, `results/` and
`results_summary/`. `WORK` is a scratch directory for features and kernels.

```bash
cd analyses/granularity
python a01_extract_trials.py --lsa-root <root containing output/beta_index.csv> \
  --out "$FOODPREF_GRANULARITY_ROOT/cache" \
  --feature-mask ../../results/reference/decoding/masks/svm_common_mask.nii.gz \
  [--path-prefix-from <beta_file prefix recorded in beta_index.csv>]
python a10_run.py --k 1 --seed 42 --work "$WORK"   # repeat until it prints COMPLETE
python a11_grid.py "$WORK"                          # repeat until GRID COMPLETE
WORK=$WORK bash a12_weights_grid.sh                 # explicit weights (repeat until done)
bash a13_cross_grid.sh                              # cross-granularity scoring
python a20_stats.py
python a22_cross_stats.py
python a21_haufe_by_k.py "$FSLDIR/data/atlases"     # needs paper_stats s5 output
python a23_regions_match.py "$FSLDIR/data/atlases"
python a24_train1_adjacent.py
```

The drivers stop after a time budget and resume on the next call. Summaries are
written to `results_summary/`; the distributed copies are in
`results/reference/granularity/`.

## Manuscript statistics

`analyses/paper_stats/` computes the remaining manuscript statistics from saved
results. Outputs go to `outputs/analyses/paper_stats/<name>/`; distributed copies
are in `results/reference/paper_stats/`.

| Script | Inputs | Purpose |
|---|---|---|
| `s2_k_paired_tests.py` | `results/reference/tables/decoding_participants.csv` | CNN/SVM comparisons across k (Holm, sign-flip) |
| `s3_atlas_labels.py --atlas-root <atlases>` | GLM and searchlight group results under `$FOODPREF_RUNS_ROOT` | Atlas labels of GLM clusters and searchlight coverage |
| `s4_participant_tests.py --atlas-root <atlases>` | k = 6 Layer-CAM participant maps (`gradcam_subject_tfce`) | Participant-level regional tests |
| `s5_haufe_layercam_regions.py --atlas-root <atlases>` | Layer-CAM group maps, k = 1 Haufe map (`outputs/analyses/haufe/`) | Forward pattern vs. Layer-CAM by region |

Run them in this order; `s5` reads the `s4` output and `a21_haufe_by_k.py` reads
the `s5` output.

## Figures

Figure 2 and Figure 4C-D are drawn from the reference bundle. Figure 1 (design
schematic) and the surface renderings in Figure 3 and Figure 4A-B appear only in
`paper/main.pdf`.

```bash
make fonts     # TeX Gyre Heros OTF -> TTF for matplotlib (TeX Live + fonttools)
make figures   # outputs/figures/fig_granularity.pdf, fig_agreement.pdf
```

`analyses/figures/roi_categories.py <atlases> . <out_dir>` rebuilds
`results/reference/figures/roi_top20_categories.*` (the ROI labels of Figure 4C)
from `regions_match_k6.csv` and the Harvard-Oxford atlas.

## Checks performed during packaging

- All migrated analysis entrypoints returned `--help` without accessing datasets.
- Synthetic checks cover session isolation, train-only standardization,
  group-first ROI normalization, voxel-weighted macroregions and portable paths.
- Regional summary arithmetic was checked against the existing group maps and
  ROI tables; it reproduced the corrected manuscript regional values.
- No CNN fitting, full searchlight, permutation inference, input-replacement
  inference or SVM/Haufe refitting was rerun while packaging this repository.
