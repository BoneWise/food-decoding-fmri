"""a05: Haufe forward pattern per fold, a_f = X_train^T (s - mean s) / sum (s - mean s)^2 (as the frozen
rq3_svm_haufe_k01 code), with s the training-fold decision score; fold patterns averaged with equal weight.
Writes <work>/<cond>/fit_<partition>/haufe_folds.npy [10, 222441] and haufe_mean.npy."""
import argparse, csv
from pathlib import Path
import numpy as np
import common

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cond", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--partition", default="k01"); a = ap.parse_args()
    d = Path(a.work) / a.cond; fd = d / f"fit_{a.partition}"
    idx = list(csv.DictReader(open(d / "index.csv"))); subj = np.array([r["subject"] for r in idx])
    fl = common.folds(a.partition); nf = len(fl); n = len(idx)
    C = np.zeros((n, nf))
    for j, (fi, _, _) in enumerate(fl):
        z = np.load(fd / f"fold_{fi:02d}.npz"); s = z["train_decision"]; c = s - s.mean()
        C[z["train_rows"], j] = c / (c @ c)
    A = np.zeros((common.P, nf))
    for s in common.SUBJECTS:
        rows = np.flatnonzero(subj == s)
        X = np.load(d / f"{s}.npy").astype(np.float64)
        A += X.T @ C[rows]
    np.save(fd / "haufe_folds.npy", A.T.astype(np.float32)); np.save(fd / "haufe_mean.npy", A.mean(axis=1).astype(np.float32))
    print("haufe done", A.shape)

if __name__ == "__main__":
    main()
