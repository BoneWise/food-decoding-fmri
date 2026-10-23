"""a02: build standardized common-mask features for one condition (k, grouping seed) into a work dir.
Work dir (VM scratch, deleted after the condition is finished): <work>/k{k:02d}_seed{seed}/
  sub-XXX.npy  float32 [n_inputs, 222441];  index.csv  one row per input (order of the kernel)
"""
import argparse, csv, time
from pathlib import Path
import numpy as np
import common

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, required=True); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work", required=True); a = ap.parse_args()
    d = Path(a.work) / f"k{a.k:02d}_seed{a.seed}"; d.mkdir(parents=True, exist_ok=True)
    if (d / "index.csv").exists(): print("done already"); return
    G = common.groups(a.k, a.seed) if a.k > 1 else None
    rows = []; t0 = time.time()
    for s in common.SUBJECTS:
        R = common.raw(s); tr = common.trials(s); pos = common.feat_pos(s)
        rg, lab = [], []
        for r in (1, 4):
            gg = G[(s, r)] if a.k > 1 else [(int(t["row"]),) for t in sorted([t for t in tr if int(t["rating"]) == r], key=lambda t: (t["session"], t["run"], int(t["trial"])))]
            rg += gg; lab += [r] * len(gg)
        X = common.standardize(common.merged_maps(R, rg, a.k), pos, a.k)
        np.save(d / f"{s}.npy", X)
        for j, (g, r) in enumerate(zip(rg, lab)):
            rows.append(dict(subject=s, local=j, rating=r, target=int(r == 4), source_rows="|".join(map(str, g))))
    with open(d / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"k={a.k} seed={a.seed}: {len(rows)} inputs, {time.time()-t0:.0f} s")

if __name__ == "__main__":
    main()
