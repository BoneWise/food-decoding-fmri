"""a24: with training fixed to single trials (k_train = 1), adjacent test-granularity comparisons
(test k vs k-1, k = 2..6; plus k6 vs k3) on participant BA averaged over seeds 42-46; two-sided paired t,
Holm over the five adjacent comparisons. Also prints k=6 forward-pattern values of selected ROIs."""
import csv
import numpy as np
from scipy import stats
import common
C = common.ROOT / "results_summary/cross"; subs = common.SUBJECTS
def ba(t, d):
    p = (d > 0).astype(int); return 0.5 * ((p[t == 1] == 1).mean() + (p[t == 0] == 0).mean())
B = {}
for kt in range(1, 7):
    vals = []
    for sd in (42, 43, 44, 45, 46):
        rows = list(csv.DictReader(open(C / f"seed{sd}_ktest{kt}.csv"))); sub = np.array([r["subject"] for r in rows]); t = np.array([int(r["target"]) for r in rows])
        d = np.array([float(r["dec_ktrain1"]) for r in rows]); vals.append([ba(t[sub == s], d[sub == s]) for s in subs])
    B[kt] = np.mean(vals, axis=0)
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj
out = []
for a, b in ((2, 1), (3, 2), (4, 3), (5, 4), (6, 5)):
    d = 100 * (B[a] - B[b]); t = stats.ttest_1samp(d, 0)
    out.append(dict(comparison=f"train1 test{a}-test{b}", mean=d.mean(), sd_diff=d.std(ddof=1), p_t=t.pvalue, n_positive=int((d > 0).sum())))
for r, h in zip(out, holm([r["p_t"] for r in out])): r["p_holm"] = h
d = 100 * (B[6] - B[3]); t = stats.ttest_1samp(d, 0)
out.append(dict(comparison="train1 test6-test3", mean=d.mean(), sd_diff=d.std(ddof=1), p_t=t.pvalue, n_positive=int((d > 0).sum()), p_holm=float("nan")))
with open(common.ROOT / "results_summary/cross_granularity_train1_adjacent.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
for r in out: print(r)
h = {r["roi"]: r for r in csv.DictReader(open(common.ROOT / "results_summary/haufe_by_k_rois.csv"))}
for n in ("Left Paracingulate Gyrus", "Right Paracingulate Gyrus", "Left Frontal Medial Cortex", "Right Frontal Medial Cortex", "Left Cingulate Gyrus, anterior division"):
    print(n, "z_k1", round(float(h[n]["z_k1"]), 2), "z_k6", round(float(h[n]["z_k6"]), 2))
top = sorted(h.values(), key=lambda r: -float(r["z_k6"]))[:12]
print([(r["roi"], round(float(r["z_k6"]), 2)) for r in top])
