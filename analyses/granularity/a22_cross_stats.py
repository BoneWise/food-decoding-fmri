"""a22: statistics for the cross-granularity transfer (a07). Participant BA for every (k_train, k_test),
averaged over grouping seeds 42-46. Families (Holm within family; paired two-sided t, 100,000 sign flips,
100,000 participant bootstraps; seed 20261022):
  test_avg   BA(train 1, test k) - BA(train 1, test 1): averaging the held-out inputs only
  train_avg  BA(train k, test k) - BA(train 1, test k): averaging the training inputs, same test inputs
  reverse    BA(train k, test 1) - BA(train 1, test 1): averaged-trained readout applied to single trials
Weight similarity: cosine between fold-matched SVM weight vectors of different k_train (mean over folds, seeds).
"""
import csv, json
from itertools import combinations
import numpy as np
from scipy import stats
import common

OUT = common.ROOT / "results_summary"; C = OUT / "cross"; SEEDS = (42, 43, 44, 45, 46); KS = (1, 2, 3, 4, 5, 6)
RNG = np.random.default_rng(20261022); NB = 100_000; subs = common.SUBJECTS
def ba(t, d):
    p = (d > 0).astype(int); return 0.5 * ((p[t == 1] == 1).mean() + (p[t == 0] == 0).mean())
M = np.zeros((len(SEEDS), 6, 6, len(subs)))
for si, sd in enumerate(SEEDS):
    for kt in KS:
        rows = list(csv.DictReader(open(C / f"seed{sd}_ktest{kt}.csv")))
        sub = np.array([r["subject"] for r in rows]); t = np.array([int(r["target"]) for r in rows])
        for ka in KS:
            d = np.array([float(r[f"dec_ktrain{ka}"]) for r in rows])
            for j, s in enumerate(subs): M[si, ka - 1, kt - 1, j] = ba(t[sub == s], d[sub == s])
BA = M.mean(axis=0)   # [k_train, k_test, participant]; train1/test1 identical across seeds
assert np.allclose(M[:, 0, 0, :], M[0, 0, 0, :])
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj
def paired(d):
    tt = stats.ttest_1samp(d, 0.0); bs = d[RNG.integers(0, d.size, size=(NB, d.size))].mean(axis=1)
    fl = np.abs((RNG.choice([-1.0, 1.0], size=(NB, d.size)) * d).mean(axis=1))
    return dict(mean=100 * d.mean(), ci_low=100 * np.percentile(bs, 2.5), ci_high=100 * np.percentile(bs, 97.5), t=tt.statistic,
                p_t=tt.pvalue, p_signflip=(np.sum(fl >= abs(d.mean()) - 1e-12) + 1) / (NB + 1), n_positive=int((d > 0).sum()))
fam = {"test_avg": [(f"train1 test{k} - train1 test1", BA[0, k - 1] - BA[0, 0]) for k in KS[1:]],
       "train_avg": [(f"train{k} test{k} - train1 test{k}", BA[k - 1, k - 1] - BA[0, k - 1]) for k in KS[1:]],
       "reverse": [(f"train{k} test1 - train1 test1", BA[k - 1, 0] - BA[0, 0]) for k in KS[1:]]}
res = []
for f, items in fam.items():
    block = [dict(family=f, comparison=n, **paired(d)) for n, d in items]
    for r, a, b in zip(block, holm([r["p_t"] for r in block]), holm([r["p_signflip"] for r in block])): r["p_holm_t"], r["p_holm_signflip"] = a, b
    res += block
with open(OUT / "cross_granularity_tests.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
with open(OUT / "cross_granularity_matrix.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["k_train"] + [f"test_k{k}" for k in KS])
    for ka in KS: w.writerow([ka] + [f"{100*BA[ka-1, kt-1].mean():.3f}" for kt in KS])
# weight similarity
R = common.ROOT / "results"
def wpath(k, s): return R / ("k01_seed42" if k == 1 else f"k{k:02d}_seed{s}") / "fit_k01" / "weights.npy"
cos = {}
for a, b in combinations(KS, 2):
    vals = []
    for s in SEEDS:
        Wa, Wb = np.load(wpath(a, s)).astype(np.float64), np.load(wpath(b, s)).astype(np.float64)
        vals += [float(Wa[j] @ Wb[j] / np.linalg.norm(Wa[j]) / np.linalg.norm(Wb[j])) for j in range(Wa.shape[0])]
    cos[f"k{a}-k{b}"] = float(np.mean(vals))
seedcos = {}
for k in KS[1:]:
    Ws = [np.load(wpath(k, s)).astype(np.float64) for s in SEEDS]
    seedcos[f"k{k}"] = float(np.mean([Ws[x][j] @ Ws[y][j] / np.linalg.norm(Ws[x][j]) / np.linalg.norm(Ws[y][j]) for x, y in combinations(range(5), 2) for j in range(10)]))
json.dump(dict(weight_cosine_between_k=cos, weight_cosine_between_seeds=seedcos), open(OUT / "cross_granularity_weights.json", "w"), indent=2)
print("BA matrix (rows k_train, cols k_test):")
for ka in KS: print(ka, " ".join(f"{100*BA[ka-1, kt-1].mean():6.2f}" for kt in KS))
for r in res: print(r["family"], r["comparison"], f"{r['mean']:+.2f} [{r['ci_low']:+.2f},{r['ci_high']:+.2f}] p_holm_t={r['p_holm_t']:.4f} sf={r['p_holm_signflip']:.4f} n+={r['n_positive']}")
print(json.dumps(dict(weight_cosine_between_k=cos, weight_cosine_between_seeds=seedcos), indent=1))
# curves for plotting: mean and participant-bootstrap CI per condition and k
curves = []
for name, get in (("matched", lambda k: BA[k - 1, k - 1]), ("train1_testk", lambda k: BA[0, k - 1]), ("traink_test1", lambda k: BA[k - 1, 0])):
    for k in KS:
        v = get(k); bs = v[RNG.integers(0, v.size, size=(NB, v.size))].mean(axis=1)
        curves.append(dict(condition=name, k=k, mean=100 * v.mean(), ci_low=100 * np.percentile(bs, 2.5), ci_high=100 * np.percentile(bs, 97.5)))
with open(OUT / "cross_granularity_curves.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
