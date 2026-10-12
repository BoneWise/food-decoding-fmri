# Deep Learning-Based Cross-Participant Decoding of Subjective Food Value

*Manuscript in preparation*

Subjective food value can be decoded from whole-brain fMRI in unseen participants. Linear SVM and 3D-CNN models distinguish high from low current desire to eat above chance under participant-held-out cross-validation.

This repository provides code and frozen results for cross-participant decoding and spatial analysis. Layer-CAM characterizes CNN spatial attributions, with prominent occipital and insular–subcortical patterns across high- and low-value predictions. Complementary GLM and Searchlight analyses examine mean response differences and local decodable information.

[中文说明](README.zh-CN.md) · [Results and provenance](docs/results.md) · [Analysis entrypoints](docs/analysis-entrypoints.md) · [Data preparation](docs/data.md)

## Reproduce tables and figures

Run from the repository root with Python 3.11 or newer. The preparation checks used Python 3.13 on macOS; the exact tested packages are recorded in `environments/tested-python313-macos.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/reproduce_results.py
python scripts/validate_reference.py
python scripts/reproduce_figures.py
```

These commands use the small reference bundle included in the repository. They write new files under ignored `outputs/`, leaving the reference results unchanged. Recomputed statistics and inherited confidence intervals are explicitly distinguished in the results documentation. Surface-rendered figures and task/model diagrams are supplied as original assets; the quick-start plotting command rebuilds the table-based charts.

## Results represented here

Ten-fold cross-validation holds out three participants per fold, covering 30 participants in total. Labels are the lowest and highest ratings of current desire to eat (1 and 4). Balanced accuracy is calculated for each held-out participant and summarized below. For k=3 and k=6, inputs average non-overlapping groups of trials with the same rating within each participant.

| Input averaging | Inputs | CNN BA, mean ± SD (%) | Matched SVM BA, mean ± SD (%) |
|---|---:|---:|---:|
| k=1 | 7,038 | 53.71 ± 6.56 | 57.87 ± 5.96 |
| k=3 | 2,326 | 54.58 ± 8.43 | 63.80 ± 10.33 |
| k=6 | 1,147 | 57.24 ± 10.60 | 63.33 ± 15.76 |

Both models performed above chance in all three input settings; the matched linear SVM achieved higher mean balanced accuracy than the CNN in each setting.

Layer-CAM uses the k=6 CNN. The regional summaries show shared occipital and insular/subcortical emphasis, with posterior-medial, medial-temporal, and specific subcortical regional patterns. Regional values are reported as **mean Layer-CAM value (whole-brain mean = 1)**. Correct held-out inputs are averaged within participant and class, then across participants; regional normalization uses the resulting class group map. The original five-group atlas mapping is recorded with the reference results.

![Regional Layer-CAM values](figures/reproduced/layercam_macroregions.png)

## Repository layout

| Directory | Contents |
|---|---|
| `src/foodpref/` | Data handling, CNN, training, frozen folds, matched SVM, beta averaging |
| `preprocessing/` | MATLAB/SPM single-trial and sessionwise beta estimation |
| `analyses/` | Spatial analyses, session-held-out decoding, input replacement, Haufe comparisons |
| `configs/` | Recorded experiment settings and portable path configuration |
| `results/reference/` | Frozen predictions, compact tables, selected maps, and source hashes |
| `scripts/` | Table reproduction, validation, and plotting entrypoints |
| `figures/original/` | Preserved manuscript figure assets |
| `docs/` | Data, analysis, figure, and migration documentation |
| `tests/` | Lightweight protocol and numerical consistency checks |

## Run the models and spatial analyses

```bash
python -m pip install -e '.[train,spatial,dev]'
python -m foodpref.train --help
python -m foodpref.matched_svm --help
python -m foodpref.merge_betas --help
python -m pytest -q
```

Prepare the beta maps and masks described in [data preparation](docs/data.md), then use the recorded configurations in [configs/decoding](configs/decoding/README.md). The [analysis guide](docs/analysis-entrypoints.md) gives the ordered spatial-analysis commands and external software requirements. Full analyses require local imaging data and, for post-hoc analyses of existing models, the corresponding checkpoints. Large weights and feature matrices are not distributed in this initial repository version.

## Reproducibility and versioning

The initial organization preserves the current manuscript's audited decoding and regional result definitions. The exact historical GLM/Searchlight sources behind the manuscript counts remain to be aligned; those background candidates are explicitly excluded from the validated reproduction tables. See [migration notes](docs/migration.md) and [validation](docs/validation.md) for the exact scope of checks. Moving code or changing paths does not reuse historical validation as evidence for new execution. Original source hashes and relocated-file hashes are recorded separately.

The writing workflow is maintained separately. A manuscript version can cite a fixed repository commit or tag; rerun outputs belong in `outputs/`. The original dataset and its analysis resource are linked in [docs/data.md](docs/data.md). Software citation metadata is in `CITATION.cff`; author-selected open-source licensing is pending as described in `LICENSE`.
