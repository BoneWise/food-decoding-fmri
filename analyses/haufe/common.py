from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from analyses._paths import analysis_root, config_root, config_path, read_config, REPO_ROOT
from typing import Any, Iterable, Sequence

import nibabel as nib
import numpy as np


CODE_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = config_root("haufe")
PROJECT_ROOT = analysis_root("haufe")
CONFIG_PATH = config_path("haufe")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=target.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, target)


def sha256_file(path: str | Path, *, block_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def load_config() -> dict[str, Any]:
    return read_config(CONFIG_PATH)


def load_binary_mask(path: str | Path) -> tuple[np.ndarray, nib.spatialimages.SpatialImage]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected a three-dimensional mask at {path}, got {data.shape}.")
    mask = np.isfinite(data) & (data > 0)
    if not mask.any():
        raise ValueError(f"Mask is empty: {path}")
    return np.asarray(mask, dtype=bool), image


def validate_record_rows(rows: Iterable[dict[str, str]], expected_count: int) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: int(row["row_index"]))
    if len(ordered) != expected_count:
        raise ValueError(f"Expected {expected_count} sample-index rows, found {len(ordered)}.")
    expected = list(range(expected_count))
    observed = [int(row["row_index"]) for row in ordered]
    if observed != expected:
        raise ValueError("Sample-index row_index values are not contiguous and ordered.")
    for row in ordered:
        rating = int(row["rating"])
        if rating not in (1, 4):
            raise ValueError(f"Unexpected rating {rating} in row {row['row_index']}.")
        target = int(row["target"])
        if target != (1 if rating == 4 else 0):
            raise ValueError(f"Rating/target mismatch in row {row['row_index']}.")
    return ordered


def map_vector_to_image(vector: np.ndarray, mask: np.ndarray, reference: nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    vector = np.asarray(vector)
    if vector.ndim != 1 or vector.shape[0] != int(mask.sum()):
        raise ValueError(f"Vector shape {vector.shape} does not match mask size {int(mask.sum())}.")
    volume = np.zeros(mask.shape, dtype=np.float32)
    volume[mask] = vector.astype(np.float32, copy=False)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(volume, reference.affine, header)


def save_nifti(path: str | Path, image: nib.spatialimages.SpatialImage) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.nii.gz")
    nib.save(image, str(temporary))
    os.replace(temporary, target)


def source_snapshot_path(name: str) -> Path:
    return PROJECT_ROOT / "inputs" / name


def fold_dir(fold_index: int) -> Path:
    return PROJECT_ROOT / "outputs" / "folds" / f"fold_{fold_index:02d}"
