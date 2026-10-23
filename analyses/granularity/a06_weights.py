"""a06: explicit SVM weights per fold, w_f = sum_i b_i x_i and intercept b0_f = sum_i b_i (from a04's kernel
coefficients; the intercept is the regularized bias feature of LinearSVC), for the fixed k=1 partition.
Rebuilds the condition's features in scratch, writes results/<cond>/fit_k01/weights.npy [10, 222441] float32
and intercepts.npy, checks w.x + b0 against the stored test decisions, then deletes the scratch features."""
import argparse, csv, subprocess, sys
from pathlib import Path
import numpy as np
import common

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, required=True); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--work", required=True); a = ap.parse_args()
cond = f"k{a.k:02d}_seed{a.seed}"; res = common.ROOT / "results" / cond / "fit_k01"; w = Path(a.work); d = w / cond
if (res / "weights.npy").exists(): print(cond, "weights exist"); sys.exit()
src = res if (res / "fold_01.npz").exists() else d / "fit_k01"
if not (d / "index.csv").exists() or not list(d.glob("sub-*.npy")):
    (d / "index.csv").unlink(missing_ok=True)
    subprocess.run([sys.executable, "a02_features.py", "--k", str(a.k), "--seed", str(a.seed), "--work", str(w)], check=True, cwd=Path(__file__).parent)
idx = list(csv.DictReader(open(d / "index.csv"))); subj = np.array([r["subject"] for r in idx])
fl = common.folds("k01"); n = len(idx); B = np.zeros((n, len(fl))); te_rows, te_dec = {}, {}
for j, (fi, _, _) in enumerate(fl):
    z = np.load(d / "fit_k01" / f"fold_{fi:02d}.npz"); B[z["train_rows"], j] = z["b"]; te_rows[j] = z["test_rows"]; te_dec[j] = z["test_decision"]
W = np.zeros((len(fl), common.P)); b0 = B.sum(axis=0)
for s in common.SUBJECTS:
    rows = np.flatnonzero(subj == s); X = np.load(d / f"{s}.npy").astype(np.float64)
    W += B[rows].T @ X
err = 0.0
for s in common.SUBJECTS:
    rows = np.flatnonzero(subj == s); X = np.load(d / f"{s}.npy").astype(np.float64)
    for j in range(len(fl)):
        m = np.isin(te_rows[j], rows)
        if m.any():
            pos = np.searchsorted(rows, te_rows[j][m]); err = max(err, float(np.abs(X[pos] @ W[j] + b0[j] - te_dec[j][m]).max()))
assert err < 1e-6, err
np.save(res / "weights.npy", W.astype(np.float32)); np.save(res / "intercepts.npy", b0)
for f in d.glob("sub-*.npy"): f.unlink()
print(cond, f"weights saved; max |w.x+b - stored decision| = {err:.2e}")
