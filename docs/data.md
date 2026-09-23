# Data and input preparation

The source dataset is **A densely sampled fMRI dataset for investigating food valuation** by Sugawara and colleagues. The original dataset is available on [OpenNeuro, ds007267](https://openneuro.org/datasets/ds007267); the [data descriptor](https://doi.org/10.1038/s41597-026-07323-y) and [authors' analysis repository](https://github.com/szkshnsk/A-densely-sampled-fMRI-dataset-for-investigating-food-valuation) describe the resource. Their repository identifies dataset DOI `10.18112/openneuro.ds007267.v1.0.0`. These links were checked on 2026-09-23. Original participant data, stimulus images, atlases, and software remain subject to their source terms.

## Reproduce distributed results

No source imaging download is needed for `scripts/reproduce_results.py`, `scripts/validate_reference.py`, or the table-driven plotting script. The reference bundle contains anonymous participant IDs, held-out predictions, regional summaries, and selected group maps. Workstation paths have been replaced by stable source aliases; original and distributed hashes are both recorded.

## Estimate beta maps

The MATLAB entrypoints start from prepared fMRIPrep derivatives, rather than running fMRIPrep themselves. They retain the original SPM25 model definitions and expected event/confound columns.

1. Make the fMRIPrep derivatives and event files available locally. Inspect `resolve_input_files` and the event-column handling in the MATLAB scripts to match the recorded input layout (`sub-*/ses-*/func/`, plus an `events/` root).
2. Put SPM25 on the MATLAB path, or set the `spm_dir` variable.
3. Set `input_root` in MATLAB, or set the `FOODPREF_FMRIPREP_ROOT` environment variable. Set `output_root` to the desired local output directory.
4. Run `preprocessing/run_minimal_lsa.m` for single-trial LSA estimates. Run `preprocessing/run_rating14_sessionwise_glm.m` for the condition betas used by session-held-out analyses. The scripts produce their beta indices and QC records.

For the decoding configuration layout, single-trial outputs can be placed at `data/derived/`, with models at `data/derived/models/` and masks at `data/derived/cache/masks/`. The supplied decoding configurations use this mask location.

```bash
python -m foodpref.merge_betas \
  --beta-index-csv data/derived/beta_index.csv \
  --models-root data/derived/models \
  --brain-mask-root data/derived/cache/masks \
  --output-root data/derived/merged --ks 1,3,6 --seed 42
```

Check the generated manifest and sample counts before model execution. k=1/3/6 reference counts are 7,038/2,326/1,147 across the 30 included participants. The archived analysis excluded `sub-016` for this shared-mask decoding setup. The merge implementation preserves the original within-participant, within-rating grouping and tail handling.

## Run decoding

Use the archived fold files and matching data manifests in `configs/decoding/`. For example:

```bash
python -m foodpref.train --config configs/decoding/cnn_k01.json \
  --brain-mask-root data/derived/cache/masks
```

For matched SVM, use the corresponding frozen `--fold-assignments` and configure its exact beta index as described in [the decoding guide](../configs/decoding/README.md). The optional `--cnn-predictions` argument supplies matching model predictions for comparison reporting. The original k=1 SVM source used an index reconstructed to match the CNN input membership. The repository checks frozen membership in its small reference predictions; arbitrary replacement input indices do not establish exact experiment reproduction.

## Large artifacts

The initial repository distributes no raw BOLD data, individual beta collection, large feature matrix, or trained model weights. Full training can generate new checkpoints. Post-hoc reproduction of the archived spatial analyses requires the original associated run/checkpoint artifacts. `docs/artifacts.json` records this distinction; it does not invent download links for artifacts that have not been deposited.
