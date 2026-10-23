"""a04: cross-participant linear SVM on a precomputed kernel, same problem as the frozen LinearSVC.

Frozen classifier: sklearn LinearSVC(C=1, loss='squared_hinge', penalty='l2', class_weight='balanced',
dual=True, fit_intercept=True, intercept_scaling=1, tol=1e-4). Its primal is
    min_w 1/2 |w|^2 + sum_i C_i max(0, 1 - y_i w.x~_i)^2,   x~ = [x, 1]  (intercept regularized),
which is strictly convex, so its minimizer is unique. We solve the same problem exactly in the span of the
training inputs (w = sum_j b_j x~_j, kernel K~ = X X^T + 1) with primal Newton steps on the active set
(Chapelle, 2007) plus backtracking, to a tighter tolerance than liblinear's 1e-4.
Outputs in <work>/<cond>/fit_<partition>/: predictions.csv, folds.json, fold_XX.npz (train rows, b, train decision)
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import common

def solve(K, y, Ci, tol=1e-10, max_iter=100):
    n = len(y); b = np.zeros(n); f = np.zeros(n)
    def obj(b, f):
        return 0.5 * b @ f + np.sum(Ci * np.maximum(0.0, 1.0 - y * f) ** 2)
    P0 = obj(b, f); it = 0
    for it in range(1, max_iter + 1):
        sv = (y * f) < 1.0
        A = K[np.ix_(sv, sv)].copy(); A[np.diag_indices_from(A)] += 0.5 / Ci[sv]
        bn = np.zeros(n); bn[sv] = np.linalg.solve(A, y[sv])
        d = bn - b; fd = K @ d; t = 1.0; Pc = obj(b, f)
        while True:
            Pn = obj(b + t * d, f + t * fd)
            if Pn <= Pc + 1e-14 * abs(Pc) or t < 1e-6: break   # accept non-increase (within rounding)
            t *= 0.5
        b = b + t * d; f = f + t * fd
        if np.array_equal((y * f) < 1.0, sv) and (t == 1.0 or abs(Pc - Pn) <= 1e-15 * abs(Pc)): break
    sv = (y * f) < 1.0
    r = b - np.where(sv, 2.0 * Ci * (y - f), 0.0)
    kkt = float(np.sqrt(max(r @ (K @ r), 0.0)) / max(np.sqrt(b @ f), 1e-12))
    return b, f, it, kkt

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cond", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--partition", default="k01"); ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--budget", type=float, default=1e9); a = ap.parse_args(); T0 = time.time()
    d = Path(a.work) / a.cond; idx = list(csv.DictReader(open(d / "index.csv")))
    out = d / f"fit_{a.partition}"; out.mkdir(exist_ok=True)
    K = np.load(d / "kernel.npy", mmap_mode="r")
    subj = np.array([r["subject"] for r in idx]); y = np.where(np.array([int(r["target"]) for r in idx]) == 1, 1.0, -1.0)
    preds, info = [], []
    for fi, tr_s, te_s in common.folds(a.partition):
        t0 = time.time()
        if (out / f"fold_{fi:02d}.npz").exists():
            z = np.load(out / f"fold_{fi:02d}.npz")
            for i, s in zip(z["test_rows"], z["test_decision"]):
                preds.append(dict(fold=fi, row=int(i), subject=subj[i], target=int(y[i] > 0), decision=float(s), pred=int(s > 0)))
            info.append(json.load(open(out / f"fold_{fi:02d}.json"))); continue
        if time.time() - T0 > a.budget:
            print("partial fit"); return
        tr = np.flatnonzero(np.isin(subj, tr_s)); te = np.flatnonzero(np.isin(subj, te_s))
        Kt = np.asarray(K[np.ix_(tr, tr)]) + 1.0
        yt = y[tr]; npos, nneg = (yt > 0).sum(), (yt < 0).sum()
        Ci = a.C * np.where(yt > 0, len(yt) / (2.0 * npos), len(yt) / (2.0 * nneg))
        b, f, it, kkt = solve(Kt, yt, Ci)
        s_te = (np.asarray(K[np.ix_(te, tr)]) + 1.0) @ b
        np.savez(out / f"fold_{fi:02d}.npz", train_rows=tr, b=b, train_decision=f, test_rows=te, test_decision=s_te)
        for i, s in zip(te, s_te):
            preds.append(dict(fold=fi, row=int(i), subject=subj[i], target=int(y[i] > 0), decision=float(s), pred=int(s > 0)))
        info.append(dict(fold=fi, n_train=int(len(tr)), n_test=int(len(te)), newton_iter=it, kkt_rel=kkt,
                         n_sv=int(((yt * f) < 1).sum()), seconds=round(time.time() - t0, 1)))
        json.dump(info[-1], open(out / f"fold_{fi:02d}.json", "w")); print(info[-1], flush=True)
    with open(out / "predictions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(preds[0])); w.writeheader(); w.writerows(sorted(preds, key=lambda r: r["row"]))
    json.dump(info, open(out / "folds.json", "w"), indent=1)

if __name__ == "__main__":
    main()
