# Deep Learning-Based Cross-Participant Decoding of Subjective Food Value

*Manuscript in preparation*

Subjective food value can be decoded from whole-brain fMRI in unseen participants. Linear SVM and 3D-CNN models distinguish high from low current desire to eat above chance under participant-held-out cross-validation.

This repository provides code and frozen results for cross-participant decoding and spatial analysis. Layer-CAM characterizes CNN spatial attributions, with prominent occipital and insular–subcortical patterns across high- and low-value predictions. Complementary GLM and Searchlight analyses examine mean response differences and local decodable information.

[中文说明](README.zh-CN.md) · [Reproduce figures](docs/README.md) · [Decoding results](docs/results.md)

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

Both models performed above chance in all three input settings.

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
| `docs/` | Figure reproduction and decoding results |
| `tests/` | Lightweight protocol and numerical consistency checks |

## Run the models and spatial analyses

```bash
python -m pip install -e '.[train,spatial,dev]'
python -m foodpref.train --help
python -m foodpref.matched_svm --help
python -m foodpref.merge_betas --help
python -m pytest -q
```

For model execution, prepare local beta maps and masks from the [source data](data/README.md), then follow the [decoding configurations](configs/decoding/README.md). The [environment guide](environments/README.md) lists dependencies for training and spatial analyses.

## Data and citation

The [data guide](data/README.md) links to the source dataset and its analysis resource. Software citation metadata is in `CITATION.cff`; licensing information is in `LICENSE`.
