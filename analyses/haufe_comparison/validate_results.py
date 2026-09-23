from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from .common import PROJECT_ROOT, config, read_json, sha256_file, write_json


def require(condition: bool, label: str, checks: list[dict[str, object]]) -> None:
    checks.append({"check": label, "passed": bool(condition)})
    if not condition:
        raise RuntimeError(f"Validation failed: {label}")


def main() -> None:
    import argparse
    argparse.ArgumentParser(description="Validate freshly reproduced Haufe–Layer-CAM comparisons.").parse_args()
    settings = config()
    checks: list[dict[str, object]] = []
    manifest = read_json(PROJECT_ROOT / "inputs" / "source_manifest.json")
    for key, source in manifest["sources"].items():
        frozen = Path(source["frozen_path"])
        original = Path(source["source_path"])
        require(frozen.is_file(), f"frozen source exists: {key}", checks)
        require(sha256_file(frozen) == source["frozen_sha256"], f"frozen source hash: {key}", checks)
        require(original.is_file(), f"original source exists: {key}", checks)
        require(sha256_file(original) == source["source_sha256"], f"original source hash: {key}", checks)

    result = read_json(PROJECT_ROOT / "outputs" / "results_summary.json")
    shared = np.asanyarray(nib.load(str(PROJECT_ROOT / "outputs" / "masks" / "shared_mask.nii.gz")).dataobj) > 0
    shared_count = int(shared.sum())
    expected_count = int(result["top20_selected_voxels_per_map"])
    require(shared_count == int(result["shared_mask_voxels"]), "shared-mask count", checks)
    require(expected_count == int(np.ceil(0.2 * shared_count)), "top20 count rule", checks)
    mask_names = ["haufe_rating1", "haufe_rating4", "layercam_tfce_t_rating1", "layercam_tfce_t_rating4"]
    for name in mask_names:
        image_path = PROJECT_ROOT / "outputs" / "masks" / f"{name}_top20_mask.nii.gz"
        image_data = np.asanyarray(nib.load(str(image_path)).dataobj)
        selected = image_data > 0
        require(int(selected.sum()) == expected_count, f"top20 exact count: {name}", checks)
        require(not bool((selected & ~shared).any()), f"top20 stays inside shared mask: {name}", checks)

    overlap_table = np.genfromtxt(
        PROJECT_ROOT / "outputs" / "tables" / "top20_overlap_comparison.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    continuous_table = np.genfromtxt(
        PROJECT_ROOT / "outputs" / "tables" / "continuous_map_comparison.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    require(len(np.atleast_1d(overlap_table)) == 6, "six overlap comparisons", checks)
    require(len(np.atleast_1d(continuous_table)) == 6, "six continuous comparisons", checks)
    require(
        (PROJECT_ROOT / "outputs" / "figures" / "haufe_layercam_top20_dice_matrix.png").is_file(),
        "descriptive Dice figure exists",
        checks,
    )
    require(
        result["interpretation_boundary"] == settings["comparison"]["interpretation_boundary"],
        "result preserves interpretation boundary",
        checks,
    )
    report = {"status": "passed", "num_checks": len(checks), "checks": checks}
    write_json(PROJECT_ROOT / "outputs" / "validation" / "validation_report.json", report)
    print(f"Validation passed: {len(checks)}/{len(checks)} checks.", flush=True)


if __name__ == "__main__":
    main()
