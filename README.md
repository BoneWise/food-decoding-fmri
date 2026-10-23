# Whole-brain fMRI decoding of subjective food value

Code, compact reference results and manuscript for

> **Whole-Brain fMRI Decoding of Subjective Food Value: Cross-Participant Generalization and Spatial Attribution.** Preprint, 2026.

The manuscript is [`paper/main.pdf`](paper/main.pdf). The analyses use the public dataset of Sugawara et al. (2026, *Scientific Data*), [OpenNeuro ds007267](https://openneuro.org/datasets/ds007267); imaging data are not redistributed here.

## Reproduce the manuscript numbers and figures

The reference bundle in `results/reference/` contains the saved predictions and summary tables behind the manuscript, so tables, statistics and data figures can be rebuilt without imaging data.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[train,spatial,dev]'
make tables validate stats   # tables, hash/arithmetic checks, k comparisons
make fonts figures           # Figure 2 and Figure 4C-D (needs TeX Live + fonttools)
make test
```

Outputs are written to the ignored `outputs/` directory; the reference bundle is never overwritten. See [docs/results.md](docs/results.md) for what is recomputed and what is distributed as saved.

| Input averaging | Inputs | CNN BA, mean ± SD (%) | Matched SVM BA, mean ± SD (%) |
|---|---:|---:|---:|
| k=1 | 7,038 | 53.71 ± 6.56 | 57.87 ± 5.96 |
| k=3 | 2,326 | 54.58 ± 8.43 | 63.80 ± 10.33 |
| k=6 | 1,147 | 57.24 ± 10.60 | 63.33 ± 15.76 |

Balanced accuracy over 30 held-out participants (ten-fold participant-held-out cross-validation).

## Run the full analyses

Starting from the fMRIPrep derivatives of ds007267:

1. Single-trial and session-wise betas: `preprocessing/` (MATLAB, SPM25); see [docs/data.md](docs/data.md).
2. Same-rating averaging, CNN and matched SVM: `src/foodpref/` with the frozen folds in `configs/decoding/`.
3. Averaging granularity (SVM, k = 1-6, five groupings): `analyses/granularity/`.
4. GLM, searchlight and within-participant decoding: `analyses/glm/`, `analyses/searchlight/`, `analyses/decoding/`.
5. Layer-CAM, input replacement and SVM forward patterns: `analyses/layercam/`, `analyses/input_replacement/`, `analyses/haufe/`, `analyses/haufe_comparison/`.
6. Remaining manuscript statistics and data figures: `analyses/paper_stats/`, `analyses/figures/`.

The ordered commands, environment variables and external software (SPM25, FSL, Harvard-Oxford atlases) are in [docs/analysis-entrypoints.md](docs/analysis-entrypoints.md). Trained weights, feature matrices and beta images are not distributed.

## Layout

| Directory | Contents |
|---|---|
| `paper/` | Manuscript PDF |
| `src/foodpref/` | Data handling, CNN, training, matched SVM, same-rating averaging |
| `preprocessing/` | Single-trial (LSA) and session-wise beta estimation |
| `analyses/` | All post-hoc analyses, manuscript statistics and data figures |
| `configs/` | Frozen participant folds and analysis settings |
| `results/reference/` | Saved predictions, summary tables and provenance hashes |
| `scripts/`, `tests/` | Table reproduction, validation and tests |
| `docs/` | Data preparation, analysis entry points, result definitions |

## Citation and license

Please cite the manuscript and the original dataset (Sugawara et al., 2026, doi:[10.1038/s41597-026-07323-y](https://doi.org/10.1038/s41597-026-07323-y)).

The code and the reference results in `results/reference/` are released under the [MIT License](LICENSE). The manuscript `paper/main.pdf` is not covered by this license; its copyright remains with the authors. The source dataset (OpenNeuro ds007267) is released under CC0; FSL, SPM, the Harvard-Oxford atlases and other third-party resources keep their own licenses.
