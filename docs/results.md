# Results and reproducibility

The repository contains compact reference results from the archived analysis runs.
They can be inspected or plotted without the original beta images, feature matrices,
CNN weights, FSL, or Connectome Workbench.

## Commands

Run from the repository root after installing the project dependencies:

```bash
python scripts/reproduce_results.py
python scripts/validate_reference.py
python scripts/reproduce_figures.py
python -m pytest tests/test_reference_results.py -q
```

Tables are written to `outputs/reproduced/tables/`, validation to
`outputs/validation/reference_validation.json`, and seven figure pairs to
`outputs/figures/`. Each figure is saved as PNG and vector PDF. `captions.md` and
`figure_sources.json` accompany the figures. Generated output is ignored by Git;
the commands do not overwrite the reference archive.

`--reference` accepts another copy of `results/reference/`. The reproduction and
plotting commands also accept `--output-dir`; validation accepts `--report`.

## Distributed files

| Location under `results/reference/` | Contents |
|---|---|
| `decoding/cnn/` | Archived OOF predictions for k = 1, 3, 6; participant and fold metrics; paired statistics; checkpoint re-inference audit |
| `decoding/svm/k01`, `k03`, `k06` | Matched SVM predictions, participant/fold summaries, run metadata |
| `layercam/maps/` | Two k = 6 category group mean maps and their common brain mask |
| `layercam/source/` | Complete raw ROI means, original macroregion/top-five values, selected ROI values |
| `session_holdout/` | The 30-participant session-holdout run, including 90 session prediction files |
| `input_replacement/` | Fixed-condition predictions, random-replicate participant metrics, group results, configurations and archived inference |
| `svm_haufe/` | Fold reproduction and spatial similarity tables, saved configuration |
| `map_comparison/` | Archived Haufe/Layer-CAM overlap, correlation, selected-voxel and ROI tables |
| `tables/` | Readable snapshots of the 16 tables produced by `reproduce_results.py` |
| `background/` | GLM/searchlight provenance investigation; candidate summaries are explicitly excluded from reproduced paper results |
| `provenance.json` | Source aliases, original hashes, distributed hashes and transformations |

## Computations preserved

**Cross-participant decoding.** The script checks that CNN and SVM contain the
same sample identities, targets and held-out participant folds. It computes BA
as the mean of the two class recalls within each participant, then averages
participants equally. Reported SD uses `ddof=1`. Fold BA is saved separately and
is not substituted for participant BA. Paired differences use the same
participants, with direction **SVM minus CNN**. The paired t statistic and
its two-sided p value are recomputed.

The canonical CNN reference is the saved OOF checkpoint re-inference used in the
current paper tables. `fold_reinference_audit_k136.csv` records small historical
classification differences from the original training logs for some k = 1 folds.
Rebuilding these tables reads the saved labels and does not perform model
inference or choose a checkpoint.

**Regional Layer-CAM means.** Regional normalization follows the original
category-wise group-map calculation:

1. Correct held-out inputs were averaged within each participant and category.
2. Participant maps were averaged equally within each category.
3. Each ROI mean is divided by the whole-brain mean of that category's group map.
4. A macroregion mean weights its ROI means by their voxel counts, then uses the
   same whole-brain denominator.

The distributed maps and ROI summaries begin at step 3. The script recomputes
those denominators and all normalized ROI, macroregion and top-five tables; it
does not reconstruct the participant maps or the original TFCE analysis. The
whole-brain means are 0.057552967400379774 (rating 1) and
0.058472977909833225 (rating 4). The output column is
`mean_layercam_whole_brain_1`, described as **mean Layer-CAM value (whole-brain
mean = 1)**. Original source tables retain their historical `E`/`density_ratio`
column names for provenance.

Original ROI-to-macroregion assignments are preserved. In particular, anterior
and posterior cingulate belong to the archived `Frontal` group, precuneus to
`Parietal`, and insula/subcallosal cortex to `Subcortical`. Figures display the
last group as **Insular/subcortical**. Original atlas keys containing duplicated
side names or `Precuneous` are retained in data; figure labels only remove the
duplicate side and correct the display spelling. The full normalized table
includes the original mapping, voxel count and contributing-participant count.

**Session-holdout SVM.** BA is first computed within each of the three held-out
sessions and then averaged within each participant. The archived 30-participant
run excludes `sub-016`. Its paper summary uses population SD (`ddof=0`), whereas
the cross-participant table uses sample SD. Both session SD conventions are
output and explicitly named. The session mean is 0.624074074074074; it is not
replaced by pooled-within-participant BA.

**Input replacement.** Fixed-condition BA is rebuilt from archived labels.
For random replacement, the script first averages 100 replicate BA values
within each participant and then averages participants. It does not threshold
the mean random probability as a substitute for mean random-replicate BA.
No weights, masks, input values or experimental definitions are changed.

## Frozen inference and spatial comparisons

Bootstrap confidence intervals are read from archived result files and have
`_frozen` in generated decoding/replacement column names. No bootstrap or
sign-flip resampling is rerun by these commands. The existing input-replacement
sign-flip p values and Holm correction are copied as
`input_replacement_inference_frozen.csv`.

Haufe/Layer-CAM overlap and continuous correlations are published as frozen
source tables. Dice uses the original top-20% selections, with Layer-CAM ranked
from TFCE-masked t maps. Continuous correlations instead use group mean
Layer-CAM attribution maps. These two input-map definitions remain separate.
`reproduce_figures.py` redraws these tables; it does not recompute the maps.

Original cortical surface figures are distributed separately as historical
figure assets. None of the commands above regenerate those surfaces.

## GLM/searchlight source alignment

The exact saved statistical sources underlying the manuscript's GLM/searchlight
counts have not yet been aligned. The candidate GLM summary located during
migration gives a different TFCE count; the located searchlight summary uses a
cluster-p001 output that also differs from the manuscript description.
`background/source_verification.json` records this discrepancy. Candidate files
retain source hashes for inspection but are not called verified paper evidence,
not used in rebuilt tables, and not used to replace manuscript values.

## Provenance and validation scope

Every copied source file records both its original SHA-256 and the SHA-256 of
the distributed copy. Absolute machine paths were converted to source-relative
aliases; numeric values were preserved. The aliases identify source locations,
not download URLs or executable local defaults. Weights, feature matrices and
participant beta images are not included.

Derived table snapshots have a separate `derived_files` entry identifying their
generator and checksum. Original source records are retained unchanged when a
new derived table is added. Original configuration/validation records are saved
as historical evidence and their old source hashes should not be interpreted
as hashes of the migrated code.

Validation checks distributed hashes, path portability and arithmetic agreement
with archived participant, fold, macroregion, top-five, paired, session and
replacement results. Passing validation establishes this limited agreement;
it does not mean model training, image-level inference, old validation suites,
or GLM/searchlight provenance alignment was rerun. The seven unit tests also
check participant/class weighting, mismatched folds/sample identities, regional
laterality, and protection of the reference archive.
