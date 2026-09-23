from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from analyses._paths import analysis_root, config_root, config_path, read_config, REPO_ROOT
from typing import Any, Sequence

import nibabel as nib
import numpy as np


CODE_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = config_root("haufe_comparison")
PROJECT_ROOT = analysis_root("haufe_comparison")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=target.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, target)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config() -> dict[str, Any]:
    return read_config(config_path("haufe_comparison"))


def save_map(path: str | Path, data: np.ndarray, reference: nib.spatialimages.SpatialImage) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    temporary = target.with_suffix(".tmp.nii.gz")
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), reference.affine, header), str(temporary))
    os.replace(temporary, target)
