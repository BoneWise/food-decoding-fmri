# Initial repository validation

Checked on 2026-09-23 in an isolated Python 3.13 environment. The package versions
are recorded in `environments/tested-python313-macos.txt`.

| Check | Outcome |
|---|---|
| Lightweight unit/protocol tests | 26 passed |
| Reference arithmetic and provenance checks | 1,786 / 1,786 passed |
| Tables rebuilt from the distributed inputs | 16 CSV tables plus scope metadata |
| Table-driven figures | Seven PNG/PDF pairs with captions and source mapping |
| Core command-line help | Three entrypoints passed |
| Spatial/supplementary command-line help | Eighteen entrypoints passed |

Tests cover participant-disjoint frozen folds, input normalization, matching
CNN/SVM inputs, participant/class weighting, session isolation, train-only
standardization, regional laterality and group-first normalization. The original
SVM common mask is supplied separately and verified against all three archived
SVM feature manifests. It contains 222,441 voxels; the regional Layer-CAM mask
contains 222,479 voxels. The two masks differ by 38 voxels and are not substituted
for each other.

The seven table-driven figures were visually reviewed across the preparation
checks. Original task, network, and brain-surface figures are retained as source
assets. No full CNN fitting, new model inference, SVM/Haufe refit, Searchlight,
SPM estimation, FSL permutations, or participant-level CAM recomputation was run.
The MATLAB scripts were reviewed for portable input/output handling; MATLAB/SPM
execution was not tested during this packaging step.

Bootstrap confidence intervals and supplementary randomization inference remain
explicitly marked as frozen values. Reproducing those saved tables does not rerun
their inferential procedures.

## Historical background results still needing alignment

The located GLM and Searchlight summaries do not establish the manuscript's
473 / 31,431 voxel counts. The original figure is preserved, and the candidate
source records are documented in
`results/reference/background/source_verification.json`; they are excluded from
the validated paper-result tables. This issue does not change the checked CNN/SVM,
regional, session-holdout or input-replacement results. No replacement statistics
were silently inserted into the manuscript or the reference tables.

## Distribution checks

Only selected code, metadata, compact result files and figures are tracked.
Large feature matrices, checkpoints, imaging collections, local configuration,
virtual environments and rerun outputs are ignored. Component source records
distinguish original hashes from relocated/distributed hashes. The initial
repository is private; open-source licensing has not been assigned by the authors.

Git staging was checked to preserve all 178 reference files byte for byte, including CRLF CSV records. `.gitattributes` disables newline conversion for frozen evidence and fold files so their distributed hashes survive checkout.
