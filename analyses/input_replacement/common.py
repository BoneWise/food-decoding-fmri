from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from analyses._paths import analysis_root, config_root, config_path, read_config, REPO_ROOT
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np


CODE_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = config_root("input_replacement")
PROJECT_ROOT = analysis_root("input_replacement")
DEFAULT_CONFIG_PATH = config_path("input_replacement")
DEFAULT_ROI_CONFIG_PATH = CONFIG_ROOT / "roi_definitions.json"


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    os.replace(temporary, target)
    return target


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path).expanduser().resolve()
    config = read_config(config_path)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def add_workspace_to_path(config: Mapping[str, Any]) -> Path:
    workspace = Path(str(config["workspace_root"])).expanduser().resolve()
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    return workspace


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_nifti(
    path: str | Path,
    data: np.ndarray,
    reference_img: nib.spatialimages.SpatialImage,
    *,
    dtype: np.dtype | type = np.float32,
) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    header = reference_img.header.copy()
    image = nib.Nifti1Image(
        np.asarray(data, dtype=dtype),
        affine=np.asarray(reference_img.affine),
        header=header,
    )
    image.set_data_dtype(dtype)
    nib.save(image, str(target))
    return target


def environment_payload() -> dict[str, Any]:
    import matplotlib
    import pandas
    import scipy
    import sklearn
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "nibabel": nib.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def verify_frozen_static_dependencies(
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    frozen = provenance["frozen_definitions"]
    for filename, expected_hash in frozen["source_code_sha256"].items():
        live_path = CODE_ROOT / filename
        if not live_path.exists() or sha256_file(live_path) != expected_hash:
            raise ValueError(
                f"Project source differs from frozen provenance: {live_path}"
            )

    workspace = Path(str(config["workspace_root"])).expanduser().resolve()
    for relative_path, expected_hash in frozen[
        "numerical_workspace_code_sha256"
    ].items():
        live_path = workspace / relative_path
        if not live_path.exists() or sha256_file(live_path) != expected_hash:
            raise ValueError(
                f"Numerical workspace source differs from frozen provenance: "
                f"{live_path}"
            )

    roi_config_path = CONFIG_ROOT / "roi_definitions.json"
    frozen_config_path = (
        PROJECT_ROOT / "inputs" / "frozen_experiment_config.json"
    )
    frozen_roi_path = PROJECT_ROOT / "inputs" / "frozen_roi_definitions.json"
    frozen_fold_path = PROJECT_ROOT / "inputs" / "fold_assignments.json"
    required_hashes = (
        (Path(str(config["_config_path"])), frozen["experiment_config_sha256"]),
        (roi_config_path, frozen["roi_definitions_sha256"]),
        (
            frozen_config_path,
            frozen["frozen_experiment_config_sha256"],
        ),
        (frozen_roi_path, frozen["frozen_roi_definitions_sha256"]),
        (frozen_fold_path, frozen["frozen_fold_assignments_sha256"]),
    )
    for path, expected_hash in required_hashes:
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ValueError(
                f"Frozen configuration dependency differs: {path}"
            )

    live_config_payload = {
        key: value
        for key, value in config.items()
        if not str(key).startswith("_")
    }
    if read_json(frozen_config_path) != live_config_payload:
        raise ValueError(
            "Frozen experiment configuration is not semantically identical "
            "to the live configuration."
        )
    if read_json(frozen_roi_path) != read_json(roi_config_path):
        raise ValueError(
            "Frozen ROI definitions are not semantically identical to the "
            "live definitions."
        )
    if environment_payload() != provenance["environment"]:
        raise ValueError(
            "The numerical software environment differs from frozen provenance."
        )


def build_numerical_dependency_anchor(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = provenance["frozen_definitions"]
    inputs = provenance["inputs"]
    return {
        "experiment_config_sha256": frozen["experiment_config_sha256"],
        "roi_definitions_sha256": frozen["roi_definitions_sha256"],
        "frozen_experiment_config_sha256": frozen[
            "frozen_experiment_config_sha256"
        ],
        "frozen_roi_definitions_sha256": frozen[
            "frozen_roi_definitions_sha256"
        ],
        "beta_index_sha256": inputs["beta_index_sha256"],
        "sample_manifest_sha256": inputs["sample_manifest_sha256"],
        "sample_content_fingerprint_sha256": inputs[
            "sample_content_fingerprint_sha256"
        ],
        "run_config_sha256": inputs["run_config_sha256"],
        "live_fold_assignments_sha256": inputs["fold_assignments_sha256"],
        "frozen_fold_assignments_sha256": frozen[
            "frozen_fold_assignments_sha256"
        ],
        "existing_oof_predictions_sha256": inputs[
            "existing_oof_predictions_sha256"
        ],
        "common_mask_array_sha256": provenance["common_mask"]["array_sha256"],
        "project_source_code_sha256": frozen["source_code_sha256"],
        "numerical_workspace_code_sha256": frozen[
            "numerical_workspace_code_sha256"
        ],
        "environment_sha256": sha256_payload(provenance["environment"]),
    }


def condition_order() -> list[str]:
    return [
        "full",
        "cam_top20",
        "random_top20_mean",
        "value",
        "visual",
        "union",
    ]


def stable_sample_id(subject_id: str, rating: int, trial_id: int | None) -> str:
    group = -1 if trial_id is None else int(trial_id)
    return f"{str(subject_id)}_rating-{int(rating)}_group-{group:04d}"
