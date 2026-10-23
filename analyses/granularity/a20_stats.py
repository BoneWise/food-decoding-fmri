"""a20: participant-level statistics for the SVM k curve (fixed k=1 partition) and grouping-seed robustness.

Participant BA = mean of the two class recalls over that participant's held-out inputs. For k>1 the
participant BA is averaged over grouping seeds 42-46 (each seed = an independent random grouping).
Tests (seed 20261021): two-sided paired t, 100,000 participant bootstraps (percentile CI) and 100,000 sign
flips; Holm within each family (k vs k=1; k vs k-1). Trend: per-participant OLS slope of BA on log2(k).
Sampling SD: expected SD of a participant's BA from binomial sampling of its own test inputs,
var = 0.25 * (r1(1-r1)/n1 + r4(1-r4)/n4) with the participant's own recalls (seed 42 only).
"""
import csv, json
from pathlib import Path
import numpy as np
from scipy import stats
import common

R = common.ROOT / "results"; OUT = common.ROOT / "results_summary"; OUT.mkdir(exist_ok=True)
SEEDS = (42, 43, 44, 45, 46); KS = (1, 2, 3, 4, 5, 6); RNG = np.random.default_rng(20261021); NB = 100_000

def participant_metrics(path):
    rows = list(csv.DictReader(open(path))); out = {}
    for s in common.SUBJECTS:
        t = np.array([int(r["target"]) for r in rows if r["subject"] == s]); p = np.array([int(r["pred"]) for r in rows if r["subject"] == s])
        r4, r1 = (p[t == 1] == 1).mean(), (p[t == 0] == 0).mean()
        out[s] = dict(ba=0.5 * (r1 + r4), r1=r1, r4=r4, n1=int((t == 0).sum()), n4=int((t == 1).sum()))
    return out

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj

def paired(d):
    d = np.asarray(d, float); t = stats.ttest_1samp(d, 0.0)
    bs = d[RNG.integers(0, d.size, size=(NB, d.size))].mean(axis=1)
    fl = np.abs((RNG.choice([-1.0, 1.0], size=(NB, d.size)) * d).mean(axis=1))
    return dict(mean=d.mean(), ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5), t=t.statistic, p_t=t.pvalue,
                p_signflip=(np.sum(fl >= abs(d.mean()) - 1e-12) + 1) / (NB + 1), n_positive=int((d > 0).sum()))

def path(k, seed, part="k01"):
    return R / ("k01_seed42" if k == 1 else f"k{k:02d}_seed{seed}") / f"fit_{part}" / "predictions.csv"

per = {(k, s): participant_metrics(path(k, s)) for k in KS for s in (SEEDS if k > 1 else (42,))}
subs = common.SUBJECTS
BA = {k: np.array([np.mean([per[(k, sd)][s]["ba"] for sd in (SEEDS if k > 1 else (42,))]) for s in subs]) for k in KS}

rows = []
for k in KS:
    seeds = SEEDS if k > 1 else (42,)
    gm = [np.mean([per[(k, sd)][s]["ba"] for s in subs]) for sd in seeds]
    b = BA[k]; bs = b[RNG.integers(0, b.size, size=(NB, b.size))].mean(axis=1)
    s42 = np.array([per[(k, 42)][s]["ba"] for s in subs])
    samp = np.sqrt([0.25 * (m["r1"] * (1 - m["r1"]) / m["n1"] + m["r4"] * (1 - m["r4"]) / m["n4"]) for m in per[(k, 42)].values()])
    obs_var = s42.var(ddof=1); samp_var = np.mean(samp ** 2)
    rows.append(dict(k=k, mean_ba=100 * b.mean(), sd_ba=100 * b.std(ddof=1), ci_low=100 * np.percentile(bs, 2.5), ci_high=100 * np.percentile(bs, 97.5),
                     n_above_chance=int((b > 0.5).sum()), seed_means=";".join(f"{100*g:.2f}" for g in gm),
                     seed_range=100 * (max(gm) - min(gm)), seed_sd=100 * np.std(gm, ddof=1) if len(gm) > 1 else 0.0,
                     inputs_per_participant=np.mean([m["n1"] + m["n4"] for m in per[(k, 42)].values()]),
                     sd_seed42=100 * np.sqrt(obs_var), expected_sampling_sd_seed42=100 * np.sqrt(samp_var),
                     implied_true_sd_seed42=100 * np.sqrt(max(obs_var - samp_var, 0.0)),
                     p_vs_chance_t_one_sided=stats.ttest_1samp(b, 0.5, alternative="greater").pvalue))
with open(OUT / "svm_k_curve.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

comp = []
for fam, pairs in (("vs_k1", [(k, 1) for k in KS[1:]]), ("vs_previous", [(k, k - 1) for k in KS[1:]])):
    block = [dict(family=fam, comparison=f"k{a}-k{b}", **paired(BA[a] - BA[b])) for a, b in pairs]
    for r, ph, pf in zip(block, holm([r["p_t"] for r in block]), holm([r["p_signflip"] for r in block])):
        r["p_holm_t"], r["p_holm_signflip"] = ph, pf
    comp += block
for r in comp:
    for key in ("mean", "ci_low", "ci_high"): r[key] = 100 * r[key]
with open(OUT / "svm_k_paired.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(comp[0])); w.writeheader(); w.writerows(comp)

x = np.log2(np.array(KS, float)); Y = np.stack([BA[k] for k in KS], axis=1)
slopes = np.array([np.polyfit(x, y, 1)[0] for y in Y]); trend = paired(slopes)
# per-seed sensitivity: each seed alone, k vs k=1
seedrows = []
for sd in SEEDS:
    for k in KS[1:]:
        d = np.array([per[(k, sd)][s]["ba"] for s in subs]) - BA[1]; t = stats.ttest_1samp(d, 0.0)
        seedrows.append(dict(seed=sd, k=k, mean_ba=100 * np.mean([per[(k, sd)][s]["ba"] for s in subs]), diff_vs_k1=100 * d.mean(), p_t=t.pvalue))
with open(OUT / "svm_k_by_seed.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(seedrows[0])); w.writeheader(); w.writerows(seedrows)
part = {f"k{k}": dict(fixed_partition=100 * np.mean([per[(k, 42)][s]["ba"] for s in subs]),
                      cnn_partition=100 * np.mean(list(m["ba"] for m in participant_metrics(path(k, 42, f"k{k:02d}")).values())))
        for k in (4, 5, 6)}
with open(OUT / "svm_participant_ba.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["subject"] + [f"k{k}" for k in KS] + ["slope_log2k"])
    for i, s in enumerate(subs): w.writerow([s] + [f"{BA[k][i]:.6f}" for k in KS] + [f"{slopes[i]:.6f}"])
summ = dict(trend_slope_per_doubling_pp=100 * trend["mean"], trend_ci=[100 * trend["ci_low"], 100 * trend["ci_high"]], trend_p_t=trend["p_t"],
            trend_p_signflip=trend["p_signflip"], trend_n_positive=trend["n_positive"], partition_check=part, seeds=SEEDS, n_participants=len(subs))
json.dump(summ, open(OUT / "svm_k_summary.json", "w"), indent=2, default=float)
for r in rows: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
for r in comp: print(r["family"], r["comparison"], f"{r['mean']:+.2f} [{r['ci_low']:+.2f},{r['ci_high']:+.2f}] p_holm_t={r['p_holm_t']:.4f} p_holm_sf={r['p_holm_signflip']:.4f} n+={r['n_positive']}")
print(json.dumps(summ, indent=1, default=float))
