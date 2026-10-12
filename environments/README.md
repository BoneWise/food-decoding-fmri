# Environments

The lightweight reproduction route requires the base Python dependencies in `pyproject.toml`. Model execution additionally requires `[train]`; the complete spatial Python stack is listed in `[spatial]`; tests use `[dev]`.

`tested-python313-macos.txt` records the isolated environment used for this repository's migration checks on 2026-09-23. It is a validation-environment snapshot, not a claim that every historical experiment used those versions. `core-source-requirements.txt` preserves the original declared requirements; that short file did not fully lock the historical environment. Saved experiment configurations remain the primary record for their run settings.

From the repository root, an exact validation environment can be installed with:

```bash
python -m pip install -r environments/tested-python313-macos.txt
python -m pip install --no-deps -e .
```

External tools/data for complete analyses:

- MATLAB and SPM25 for `preprocessing/*.m`.
- fMRIPrep derivatives in the structure expected by the original beta-estimation scripts; acquisition/preprocessing details follow the source dataset and saved study configuration.
- FSL `randomise` for the GLM and Layer-CAM group-inference commands; configure `FSLDIR`.
- Harvard–Oxford cortical/subcortical 25% 2-mm atlases for the original regional summaries; set `FSLDIR` or `FOODPREF_ATLAS_ROOT` to the local atlas installation.
- Surface templates and Connectome Workbench when reconstructing the original surface renderings. Those rendering assets are preserved here as figure PDFs.

The quick-start table/plot route does not need MATLAB, FSL, Workbench, raw BOLD images, or a GPU.
