"""S2: within-model paired comparisons of participant balanced accuracy across k.

Input : results/reference/tables/decoding_participants.csv (frozen, read only)
Output: s2_k_paired_tests.csv, s2_k_summary.csv, s2_run_info.json (this folder)

For each decoder (SVM, CNN) and each pair (k3-k1, k6-k1, k6-k3):
  - participant-level BA differences (30 held-out participants)
  - mean, sample SD, percentile bootstrap 95% CI (100,000 participant resamples)
  - two-sided paired t test (primary), Holm-corrected over the 3 comparisons within the decoder
  - two-sided sign-flip test (100,000 flips), Holm-corrected the same way (robustness)
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses._paths import OUTPUT_ROOT  # noqa: E402
SRC = ROOT / "results/reference/tables/decoding_participants.csv"
OUT = OUTPUT_ROOT / "paper_stats" / "s2_k_paired"
OUT.mkdir(parents=True, exist_ok=True)
N_RESAMPLE = 100_000
SEED = 20261019
PAIRS = [(3, 1), (6, 1), (6, 3)]
MODELS = ["SVM", "CNN"]


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


def main():
    rows = list(csv.DictReader(open(SRC)))
    ba = {(r["model"], int(r["k"]), r["subject_id"]): float(r["balanced_accuracy"]) for r in rows}
    subjects = sorted({r["subject_id"] for r in rows})
    rng = np.random.default_rng(SEED)

    summary = []
    for m in MODELS:
        for k in (1, 3, 6):
            v = np.array([ba[(m, k, s)] for s in subjects])
            summary.append(dict(model=m, k=k, n_participants=len(v), mean_ba=v.mean(), sd_sample=v.std(ddof=1)))

    results = []
    for m in MODELS:
        block = []
        for a, b in PAIRS:
            d = np.array([ba[(m, a, s)] - ba[(m, b, s)] for s in subjects])
            n = len(d)
            t = stats.ttest_1samp(d, 0.0)
            boot = d[rng.integers(0, n, size=(N_RESAMPLE, n))].mean(axis=1)
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            flips = rng.choice([-1.0, 1.0], size=(N_RESAMPLE, n))
            null = np.abs((flips * d).mean(axis=1))
            p_flip = (np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (N_RESAMPLE + 1)
            block.append(dict(model=m, comparison=f"k{a} minus k{b}", n_participants=n,
                              mean_ba_difference=d.mean(), sd_sample=d.std(ddof=1),
                              ci_low=ci_lo, ci_high=ci_hi, df=n - 1, paired_t=t.statistic,
                              p_two_sided=t.pvalue, p_signflip=p_flip,
                              n_improved=int((d > 0).sum()), n_worse=int((d < 0).sum())))
        for key, out in (("p_two_sided", "p_holm"), ("p_signflip", "p_signflip_holm")):
            adj = holm([r[key] for r in block])
            for r, p in zip(block, adj):
                r[out] = p
        results.extend(block)

    cols = ["model", "comparison", "n_participants", "mean_ba_difference", "sd_sample", "ci_low", "ci_high",
            "df", "paired_t", "p_two_sided", "p_holm", "p_signflip", "p_signflip_holm", "n_improved", "n_worse"]
    with open(OUT / "s2_k_paired_tests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(results)
    with open(OUT / "s2_k_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    json.dump(dict(input=str(SRC.relative_to(ROOT)), seed=SEED, n_resample=N_RESAMPLE,
                   holm_family="3 comparisons within each decoder", primary="two-sided paired t",
                   python=sys.version.split()[0], numpy=np.__version__, scipy=__import__("scipy").__version__),
              open(OUT / "s2_run_info.json", "w"), indent=2)

    for r in summary:
        print(f"{r['model']} k={r['k']}: mean BA {r['mean_ba']*100:.2f} ± {r['sd_sample']*100:.2f} (n={r['n_participants']})")
    for r in results:
        print(f"{r['model']} {r['comparison']}: {r['mean_ba_difference']*100:+.2f} pp [{r['ci_low']*100:.2f}, {r['ci_high']*100:.2f}], "
              f"t({r['df']})={r['paired_t']:.2f}, p={r['p_two_sided']:.4f}, Holm p={r['p_holm']:.4f}; "
              f"sign-flip p={r['p_signflip']:.4f}, Holm={r['p_signflip_holm']:.4f}; improved {r['n_improved']}/{r['n_participants']}")


if __name__ == "__main__":
    main()
