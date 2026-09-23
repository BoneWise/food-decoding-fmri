from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import HIGH_RATING, LOW_RATING, rating_to_target


SUBJECT_PATTERN = re.compile(r"sub-[A-Za-z0-9]+")
SESSION_PATTERN = re.compile(r"ses-[A-Za-z0-9]+")
RUN_PATTERN = re.compile(r"run-[A-Za-z0-9]+")
MODELS_DIR_MARKER = "/models/"


@dataclass(frozen=True)
class FMRISample:
    sample_path: str
    mask_path: str
    subject_id: str
    session_id: str
    run_id: str
    trial_id: Optional[int]
    rating: int


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_missing_value(value: Any) -> bool:
    return _normalize_text(value) == ""


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_missing_value(value):
        return None
    return int(float(value))


def _is_metadata_sidecar(path: Path) -> bool:
    return any(part.endswith(".nii.json") for part in path.parts) or path.name.endswith(".nii.json")


def _infer_from_path(path: Path, pattern: re.Pattern[str], fallback: str) -> str:
    match = pattern.search(str(path))
    return match.group(0) if match is not None else fallback


def _remap_path_text(value: Any, path_prefix_from: str, path_prefix_to: str) -> str:
    text = _normalize_text(value)
    if not text or not path_prefix_from or not path_prefix_to:
        return text
    source = path_prefix_from.replace("\\", "/").rstrip("/")
    target = path_prefix_to.replace("\\", "/").rstrip("/")
    normalized = text.replace("\\", "/")
    if normalized == source:
        return target
    if normalized.startswith(source + "/"):
        return target + normalized[len(source) :]
    return text


def _relative_path_under_models(path_text: str) -> Optional[Path]:
    normalized = path_text.replace("\\", "/")
    if MODELS_DIR_MARKER not in normalized:
        return None
    relative = normalized.split(MODELS_DIR_MARKER, 1)[1].lstrip("/")
    return Path(relative) if relative else None


def _infer_session_id(row: Dict[str, Any]) -> str:
    for key in ("session", "session_id", "ses"):
        value = _normalize_text(row.get(key))
        if value:
            return value if value.startswith("ses-") else f"ses-{value}"
    for key in ("beta_file", "spm_mat"):
        match = SESSION_PATTERN.search(_normalize_text(row.get(key)))
        if match is not None:
            return match.group(0)
    return "ses-unknown"


def _build_reader(handle) -> csv.DictReader:
    preview = handle.read(4096)
    handle.seek(0)
    try:
        dialect = csv.Sniffer().sniff(preview, delimiters=",\t")
    except csv.Error:
        dialect = csv.excel
    return csv.DictReader(handle, dialect=dialect)


def _candidate_beta_paths(
    row: Dict[str, Any],
    models_root: Optional[Path],
    path_prefix_from: str,
    path_prefix_to: str,
) -> List[Path]:
    candidates: List[Path] = []
    seen: set[str] = set()

    def _append(candidate: Optional[Path]) -> None:
        if candidate is None:
            return
        text = str(candidate)
        if text in seen:
            return
        seen.add(text)
        candidates.append(candidate)

    raw_beta_file = _normalize_text(row.get("beta_file"))
    remapped_beta_file = _remap_path_text(raw_beta_file, path_prefix_from, path_prefix_to)

    for text in (remapped_beta_file, raw_beta_file):
        if not text:
            continue
        candidate = Path(text)
        if candidate.is_absolute():
            _append(candidate)
        elif models_root is not None:
            _append(models_root / text)

        if models_root is not None:
            relative = _relative_path_under_models(text)
            if relative is not None:
                _append(models_root / relative)

    if models_root is not None:
        subject_id = _normalize_text(row.get("sub")) or "sub-unknown"
        session_id = _infer_session_id(row)
        run_id = _normalize_text(row.get("run")) or "run-unknown"
        beta_name = Path(remapped_beta_file or raw_beta_file).name
        trial_id = _parse_optional_int(row.get("trial_in_run"))
        if not beta_name and trial_id is not None:
            beta_name = f"beta_{trial_id:04d}.nii"
        if beta_name:
            _append(models_root / subject_id / session_id / run_id / beta_name)

    return candidates


def _resolve_beta_map_path(
    row: Dict[str, Any],
    models_root: Optional[Path],
    path_prefix_from: str,
    path_prefix_to: str,
) -> Path:
    candidates = _candidate_beta_paths(row, models_root, path_prefix_from, path_prefix_to)
    for candidate in candidates:
        if candidate.exists() and not _is_metadata_sidecar(candidate):
            return candidate.resolve()
    preview = [str(candidate) for candidate in candidates[:5]]
    raise FileNotFoundError(
        "Could not resolve beta map from beta index row. "
        f"beta_file={_normalize_text(row.get('beta_file'))!r}, candidates={preview}"
    )


def _resolve_required_path(
    value: Any,
    *,
    csv_path: Path,
    path_prefix_from: str,
    path_prefix_to: str,
) -> Path:
    text = _normalize_text(value)
    if not text:
        raise ValueError(f"Missing required mask_file in {csv_path}.")
    remapped = _remap_path_text(text, path_prefix_from, path_prefix_to)
    candidates: List[Path] = []
    for item in (remapped, text):
        if not item:
            continue
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = csv_path.parent / candidate
        candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists() and not _is_metadata_sidecar(candidate):
            return candidate.resolve()
    preview = [str(candidate) for candidate in candidates[:5]]
    raise FileNotFoundError(f"Could not resolve mask_file={text!r} from {csv_path}. candidates={preview}")


def _resolve_subject_mask_path(subject_id: str, brain_mask_root: Path) -> Path:
    pattern = f"{subject_id}*_desc-brain_mask.nii*"
    matches = sorted(
        candidate.resolve()
        for candidate in brain_mask_root.glob(pattern)
        if candidate.is_file() and not _is_metadata_sidecar(candidate)
    )
    if not matches:
        raise FileNotFoundError(f"Could not find brain mask for {subject_id} under {brain_mask_root}.")
    if len(matches) > 1:
        preview = [str(match) for match in matches[:5]]
        raise ValueError(f"Multiple brain masks found for {subject_id}: {preview}")
    return matches[0]


def discover_samples(
    *,
    beta_index_csv: str,
    models_root: str | None,
    brain_mask_root: str | None,
    path_prefix_from: str,
    path_prefix_to: str,
) -> tuple[List[FMRISample], Dict[str, Any]]:
    csv_path = Path(beta_index_csv).expanduser().resolve()
    resolved_models_root = None if models_root is None else Path(models_root).expanduser().resolve()
    resolved_brain_mask_root = None if brain_mask_root is None else Path(brain_mask_root).expanduser().resolve()

    samples: List[FMRISample] = []
    drop_counts = {
        "missing_rating": 0,
        "non_target_rating": 0,
        "missing_beta_file": 0,
        "missing_mask_file": 0,
    }
    rows_total = 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = _build_reader(handle)
        fieldnames = list(reader.fieldnames or [])
        has_mask_file = "mask_file" in fieldnames
        if not has_mask_file and resolved_brain_mask_root is None:
            raise ValueError(
                f"{csv_path} must provide mask_file, or --brain-mask-root must be set."
            )

        for row in reader:
            rows_total += 1
            rating_value = row.get("rating_value", row.get("label_value"))
            if _is_missing_value(rating_value):
                drop_counts["missing_rating"] += 1
                continue
            try:
                rating = int(float(rating_value))
            except ValueError:
                drop_counts["non_target_rating"] += 1
                continue
            if rating not in {LOW_RATING, HIGH_RATING}:
                drop_counts["non_target_rating"] += 1
                continue

            try:
                sample_path = _resolve_beta_map_path(
                    row,
                    models_root=resolved_models_root,
                    path_prefix_from=path_prefix_from,
                    path_prefix_to=path_prefix_to,
                )
            except FileNotFoundError:
                drop_counts["missing_beta_file"] += 1
                continue

            try:
                if has_mask_file and not _is_missing_value(row.get("mask_file")):
                    mask_path = _resolve_required_path(
                        row.get("mask_file"),
                        csv_path=csv_path,
                        path_prefix_from=path_prefix_from,
                        path_prefix_to=path_prefix_to,
                    )
                else:
                    if resolved_brain_mask_root is None:
                        raise FileNotFoundError("brain_mask_root is missing")
                    subject_id = _normalize_text(row.get("sub")) or "sub-unknown"
                    mask_path = _resolve_subject_mask_path(subject_id, resolved_brain_mask_root)
            except (FileNotFoundError, ValueError):
                drop_counts["missing_mask_file"] += 1
                continue

            subject_id = _normalize_text(row.get("sub")) or _infer_from_path(sample_path, SUBJECT_PATTERN, "sub-unknown")
            run_id = _normalize_text(row.get("run")) or _infer_from_path(sample_path, RUN_PATTERN, "run-unknown")
            samples.append(
                FMRISample(
                    sample_path=str(sample_path),
                    mask_path=str(mask_path),
                    subject_id=subject_id,
                    session_id=_infer_session_id(row),
                    run_id=run_id,
                    trial_id=_parse_optional_int(row.get("trial_in_run")),
                    rating=rating,
                )
            )

    if not samples:
        raise ValueError(f"No usable rating-1/rating-4 beta maps found in {csv_path}. drop_counts={drop_counts}")

    summary = {
        "source_csv": str(csv_path),
        "models_root": None if resolved_models_root is None else str(resolved_models_root),
        "brain_mask_root": None if resolved_brain_mask_root is None else str(resolved_brain_mask_root),
        "rows_total": int(rows_total),
        "samples_kept": int(len(samples)),
        "subjects_kept": int(len({sample.subject_id for sample in samples})),
        "ratings_kept": [LOW_RATING, HIGH_RATING],
        "drop_counts": drop_counts,
    }
    return samples, summary


class FMRIBetaDataset(Dataset):
    def __init__(
        self,
        records: Sequence[FMRISample],
        *,
        max_invalid_brain_fraction: float = 0.05,
    ) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("FMRIBetaDataset requires at least one record.")
        self.max_invalid_brain_fraction = float(max_invalid_brain_fraction)
        self._mask_cache: Dict[str, np.ndarray] = {}
        self._input_shape: Tuple[int, int, int] | None = None

    def __len__(self) -> int:
        return len(self.records)

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        if self._input_shape is None:
            sample = self._load_sample_tensor(self.records[0])
            self._input_shape = tuple(int(dim) for dim in sample.shape[1:])
        return self._input_shape

    @property
    def targets(self) -> List[int]:
        return [rating_to_target(record.rating) for record in self.records]

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        cache_key = str(mask_path)
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached
        mask_image = nib.load(str(mask_path))
        mask_data = np.asarray(mask_image.dataobj)
        if mask_data.ndim != 3:
            raise ValueError(f"Expected 3D mask, got {tuple(mask_data.shape)} for {mask_path}.")
        mask = np.asarray(np.isfinite(mask_data) & (mask_data > 0), dtype=bool)
        if int(mask.sum()) <= 0:
            raise ValueError(f"Mask contains no positive voxels: {mask_path}")
        self._mask_cache[cache_key] = mask
        return mask

    def _load_sample_tensor(self, record: FMRISample) -> torch.Tensor:
        beta_path = Path(record.sample_path)
        mask_path = Path(record.mask_path)
        brain_mask = self._load_mask(mask_path)

        image = nib.load(str(beta_path))
        volume = np.asarray(image.dataobj, dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D beta map, got {tuple(volume.shape)} for {beta_path}.")
        if tuple(int(dim) for dim in volume.shape) != tuple(int(dim) for dim in brain_mask.shape):
            raise ValueError(
                f"Mask shape {tuple(brain_mask.shape)} does not match beta-map shape {tuple(volume.shape)} "
                f"for {beta_path}."
            )

        finite_mask = np.asarray(brain_mask & np.isfinite(volume), dtype=bool)
        n_brain_voxels = int(brain_mask.sum())
        invalid_fraction = 1.0 - (int(finite_mask.sum()) / float(n_brain_voxels))
        if invalid_fraction > self.max_invalid_brain_fraction:
            raise ValueError(
                f"{beta_path} exceeds max_invalid_brain_fraction={self.max_invalid_brain_fraction:.4f}. "
                f"Observed {invalid_fraction:.4f}."
            )

        processed = np.zeros_like(volume, dtype=np.float32)
        if bool(finite_mask.any()):
            masked_values = volume[finite_mask]
            mean = float(masked_values.mean())
            std = float(masked_values.std())
            if std > 0.0:
                processed[finite_mask] = (volume[finite_mask] - mean) / std
        processed[~brain_mask] = 0.0
        processed[~np.isfinite(processed)] = 0.0
        return torch.from_numpy(processed).unsqueeze(0).contiguous()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        return {
            "x": self._load_sample_tensor(record),
            "y": torch.tensor(rating_to_target(record.rating), dtype=torch.long),
            "rating": torch.tensor(int(record.rating), dtype=torch.long),
            "subject_id": record.subject_id,
            "sample_path": record.sample_path,
        }


def filter_records_by_subjects(records: Sequence[FMRISample], subject_ids: Iterable[str]) -> List[FMRISample]:
    subject_set = {str(subject_id) for subject_id in subject_ids}
    return [record for record in records if record.subject_id in subject_set]
