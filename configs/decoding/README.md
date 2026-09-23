# Fixed decoding configurations

Run commands from the repository root. All local data paths can be changed in a
private copy of the JSON or overridden on the command line. JSON settings use
argparse names with underscores. `--config` loads `arguments`; metadata is not
passed to the runner.

```sh
python -m foodpref.merge_betas --config configs/decoding/merge_betas.json --dry-run
python -m foodpref.train --config configs/decoding/cnn_k01.json
python -m foodpref.matched_svm --config configs/decoding/svm_k01.json
```

The shared layout is `data/derived/beta_index.csv`, `data/derived/models/`,
`data/derived/cache/masks/`, and `data/derived/merged/kXX_seed42/`. It matches the
single-trial MATLAB output and the merge configuration. The matched SVM uses the archived feature mask distributed at
`results/reference/decoding/masks/svm_common_mask.nii.gz` (222,441 voxels).
Its file checksum exactly matches all three original SVM feature manifests;
its affine and full binary array were checked. This mask differs from the
222,479-voxel Layer-CAM mask. To run an explicitly different mask configuration,
pass `--feature-mask`; this changes the experiment definition.

Use k03 or k06 for the other averaging conditions. Training settings are copied
from the fixed-five-epoch experiment records: batch size 16, hidden width 128,
dropout 0.5, Adam learning rate 0.0001 and weight decay 0.0001, seed 42, ten subject
folds. The original records store dropout as `paper_fmri_dropout`; this is exposed
as `dropout` here. Deprecated early-stopping fields are retained for compatibility
and do not select checkpoints. SVM settings are copied from the matched runs:
linear SVC, C=1, balanced class weights, dual solver, 50,000 iterations, tolerance
0.0001, and the same per-sample brain-mask z-score as the CNN.

The archived fold files are preserved byte for byte. The k1 and k3 files match;
the k6 subject assignments differ. Both CNN and SVM read the corresponding frozen
file and validate full subject coverage and train/test separation.

Expected sample counts are 7,038 / 2,326 / 1,147 for k1 / k3 / k6, with 30
participants. Reproduction requires the original beta index and sample identities;
substituting a new index with matching counts does not establish equivalence.
In the archived matched-SVM run, k1 used a reconstructed index of the exact CNN
samples. Use the matching sample index/merge membership documented with the
reference results when preparing local data. The portable k1 path here points to
the same documented derived-data layout as k3 and k6, not to a private source path.

The optional `--cnn-predictions` input to the SVM runner enables participant-level
model comparisons. It is unset in portable run configurations so a freshly trained
model is not silently compared against a different historical run. The k6 source
record supplied its matching held-out predictions. This argument affects reporting,
not SVM training or predictions.

`merge_betas.json` exposes the established averaging rule and exclusion. It
selects the paper's k1/k3/k6 conditions; the original script also supports k2/k4/k5.
Merge groups depend on stable subject, rating, session, run, trial, and sample
ordering. Keep the archived membership for reproducing the exact paper inputs.

## Connect a new CNN run to Layer-CAM

CNN training creates `outputs/cnn/<timestamp>_<run_name>/` and prints its absolute
`run_dir:` at completion. Assign the actual k6 output directory to `CNN_K06_RUN`
below; the historical source-run name in configuration metadata is not the path
of a new run.

```sh
python -m foodpref.train --config configs/decoding/cnn_k06.json
# Set CNN_K06_RUN to the run_dir printed by that command.
python -m analyses.layercam.gradcam_subject_tfce \
  --run-dir "$CNN_K06_RUN" \
  --beta-index-csv data/derived/merged/k06_seed42/beta_index.csv \
  --models-root data/derived/merged/k06_seed42/models \
  --brain-mask-root data/derived/cache/masks \
  --output-dir outputs/analyses/layercam_k06 \
  --checkpoint-name final.pt --permutations 5000 --seed 42
```

The matched-SVM configuration reads the same frozen participant fold file as the
CNN; it does not accept a `--cnn-run-dir` option. Its outputs are written to
`outputs/svm/k01`, `k03`, or `k06`. If using a separate local analysis configuration
for input replacement or Haufe, set its CNN/SVM run paths to these actual outputs
and its Layer-CAM paths to `outputs/analyses/layercam_k06`. The distributed RQ3
configs retain archived run identifiers for restoring the original artifacts;
the analysis guide explains the required local overrides and input checks.
