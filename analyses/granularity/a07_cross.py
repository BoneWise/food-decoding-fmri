"""a07: cross-granularity transfer. For grouping seed s and test granularity k_test, rebuild every
participant's held-out inputs and score them with the fold-matched SVMs trained at each k_train = 1..6
(k_train = 1 from k01_seed42; k_train > 1 from the same seed s). Fixed k=1 partition. Writes
results_summary/cross/seed{s}_ktest{k}.csv. Diagonal cells reproduce the original held-out decisions."""
import argparse, csv
from pathlib import Path
import numpy as np
import common

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--ktest", type=int, required=True); a = ap.parse_args()
out = common.ROOT / "results_summary" / "cross"; out.mkdir(parents=True, exist_ok=True)
of = out / f"seed{a.seed}_ktest{a.ktest}.csv"
if of.exists(): print(of.name, "exists"); raise SystemExit
R = common.ROOT / "results"; KS = (1, 2, 3, 4, 5, 6)
cd = {ka: R / ("k01_seed42" if ka == 1 else f"k{ka:02d}_seed{a.seed}") / "fit_k01" for ka in KS}
W = {ka: np.load(cd[ka] / "weights.npy") for ka in KS}; B0 = {ka: np.load(cd[ka] / "intercepts.npy") for ka in KS}
fl = common.folds("k01"); fold_of = {s: j for j, (_, _, te) in enumerate(fl) for s in te}
G = common.groups(a.ktest, a.seed) if a.ktest > 1 else None
rows = []
for s in common.SUBJECTS:
    R_ = common.raw(s); tr = common.trials(s); pos = common.feat_pos(s); rg, lab = [], []
    for r in (1, 4):
        gg = G[(s, r)] if a.ktest > 1 else [(int(t["row"]),) for t in sorted([t for t in tr if int(t["rating"]) == r], key=lambda t: (t["session"], t["run"], int(t["trial"])))]
        rg += gg; lab += [r] * len(gg)
    X = common.standardize(common.merged_maps(R_, rg, a.ktest), pos, a.ktest).astype(np.float64)
    j = fold_of[s]
    S = {ka: X @ W[ka][j].astype(np.float64) + B0[ka][j] for ka in KS}
    for i, r in enumerate(lab):
        rows.append(dict(subject=s, target=int(r == 4), **{f"dec_ktrain{ka}": float(S[ka][i]) for ka in KS}))
with open(of, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# diagonal check against the stored predictions of the matching condition
if True:
    ka = a.ktest; ref = list(csv.DictReader(open(cd[ka] / "predictions.csv")))
    if ka == 1 or True:
        rs = sorted(ref, key=lambda r: int(r["row"])); d1 = np.array([float(r["decision"]) for r in rs]); d2 = np.array([r[f"dec_ktrain{ka}"] for r in rows])
        print(of.name, len(rows), "inputs; diagonal max |diff| =", f"{np.abs(d1 - d2).max():.2e}" if len(d1) == len(d2) else "length mismatch")
