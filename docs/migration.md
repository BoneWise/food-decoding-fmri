# Migration and provenance

This repository was organized on 2026-09-23 from the existing study code and the
audited CCN manuscript materials. The manuscript baseline is the corrected ISBI
v004 regional result set inherited by the CCN working outline. The conference
writing folders and historical drafts are maintained separately.

Source aliases describe the original roots without embedding workstation paths:

| Alias | Original role |
|---|---|
| `analysis-root` | Main Python analysis directory and archived runs |
| `lsa-root` | MATLAB single-trial/sessionwise beta estimation directory |
| `ccn-evidence` | Audited compact manuscript evidence |
| `wacv-analysis` | Archived OOF re-inference summaries inherited by the current manuscript |
| `regional-figures` | Original macroregion/top-five plotting sources |

Code relocation changes imports, path defaults and configuration handling. The
network, label definitions, fixed final epoch, input normalization, frozen
participant splits and regional aggregation order are retained. New tests and
reference checks validate the relocated implementation's documented scope;
historical validation reports are retained as source records, not represented
as new end-to-end runs.

`docs/migration-root.json` records MATLAB and figure sources. Component migration
records and the result bundle's provenance record original and distributed
hashes separately. The baseline run identifiers remain in experiment configs.
New runs write under `outputs/` and cannot silently replace the distributed
reference tables.

Initial scope includes k=1/3/6 CNN and matched SVM, session-held-out within-subject
decoding, GLM, Searchlight, k=6 Layer-CAM, regional summaries and the existing
input-replacement/Haufe analyses. Exploratory grid searches, separate OSF
projects, unrelated VLM work, manuscript drafts, caches and macOS resource-fork
files are omitted from this paper-oriented distribution.
