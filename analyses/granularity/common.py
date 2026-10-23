"""Shared helpers for the averaging-granularity analyses (b plan, 2026-10-20).

Replicates two frozen pipelines exactly:
  merging   src/foodpref/merge_betas.py (original make_within_subject_merged_betas.py)
            groups: rng = default_rng(seed + k*100_003); for (subject, rating) in sorted order, permute the
            trials sorted by (session, run, trial) and cut non-overlapping groups of k; leftovers dropped;
            merged map = nan_to_num(nanmean(stack, axis=0, dtype=float32)).
  features  src/foodpref/matched_svm.py (original run_matched_cross_subject_svm.py) (cnn_zscore):
            z-score each map with mean/SD over its finite brain-mask voxels, then take the 222,441-voxel
            common feature mask (non-finite features left at 0).
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analyses._paths import OUTPUT_ROOT, RUNS_ROOT  # noqa: E402

# Working root with cache/ (a01), results/ (a10/a11) and results_summary/ (a07, a20-a24).
ROOT = Path(os.environ.get("FOODPREF_GRANULARITY_ROOT", OUTPUT_ROOT / "granularity")).expanduser().resolve()
FEATURE_MASK = REPO / "results/reference/decoding/masks/svm_common_mask.nii.gz"  # 222,441-voxel common mask
PAPER_STATS = OUTPUT_ROOT / "paper_stats"
CACHE = ROOT / "cache"
FEAT = np.load(CACHE / "feature_mask.npy")
SUBJECTS = sorted(p.name.split("_")[0] for p in CACHE.glob("sub-*_betas.npy"))
P = FEAT.size

def trials(s):
    return list(csv.DictReader(open(CACHE / f"{s}_trials.csv")))

def raw(s):
    return np.load(CACHE / f"{s}_betas.npy", mmap_mode="r")

def feat_pos(s):
    idx = np.load(CACHE / f"{s}_mask.npy")
    pos = np.searchsorted(idx, FEAT)
    assert np.array_equal(idx[pos], FEAT)
    return pos

def groups(k, seed):
    """Return {(subject, rating): [list of row-index tuples]} exactly as the frozen merging script."""
    rng = np.random.default_rng(int(seed) + int(k) * 100_003)
    out = {}
    for s in SUBJECTS:
        tr = trials(s)
        for r in (1, 4):
            rows = sorted([t for t in tr if int(t["rating"]) == r], key=lambda t: (t["session"], t["run"], int(t["trial"])))
            order = rng.permutation(len(rows))
            sh = [rows[int(i)] for i in order]
            out[(s, r)] = [tuple(int(t["row"]) for t in sh[g * k:(g + 1) * k]) for g in range(len(sh) // k)]
    return out

def standardize(maps, pos, k):
    """maps: float32 [n, n_mask]; k=1 keeps NaN (excluded from mean/SD), k>1 maps are already NaN-free."""
    out = np.zeros((maps.shape[0], P), dtype=np.float32)
    for i in range(maps.shape[0]):
        m = maps[i]
        if k == 1:
            fin = np.isfinite(m)
            vals = m[fin]
        else:
            vals = m
        mean, std = float(vals.mean()), float(vals.std())
        f = m[pos]
        ff = np.isfinite(f)
        if std > 0.0:
            out[i, ff] = (f[ff] - mean) / std
    return out

def merged_maps(R, rowgroups, k):
    if k == 1:
        return np.asarray(R[[g[0] for g in rowgroups]], dtype=np.float32)
    out = np.empty((len(rowgroups), R.shape[1]), dtype=np.float32)
    for j, g in enumerate(rowgroups):
        stack = np.asarray(R[list(g)], dtype=np.float32)
        with np.errstate(invalid="ignore"):
            mb = np.nanmean(stack, axis=0, dtype=np.float32)
        out[j] = np.nan_to_num(mb, nan=0.0, posinf=0.0, neginf=0.0)
    return out

def folds(name):
    d = json.load(open(REPO / f"configs/decoding/fold_assignments_{name}.json"))
    return [(int(f["fold_index"]), sorted(f["train_subjects"]), sorted(f["val_subjects"]))
            for f in sorted(d["folds"], key=lambda f: int(f["fold_index"]))]
