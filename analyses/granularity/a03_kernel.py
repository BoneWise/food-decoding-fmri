"""a03: linear kernel K = X X^T (float64) over all inputs of a condition, blockwise and resumable.
Writes <work>/<cond>/kernel.npy (memmap) and kernel_done.json (finished participant pairs)."""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import common

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cond", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--budget", type=float, default=150); a = ap.parse_args()
    d = Path(a.work) / a.cond; idx = list(csv.DictReader(open(d / "index.csv")))
    subs = common.SUBJECTS; n = len(idx)
    off = {}; c = 0
    for s in subs:
        m = sum(1 for r in idx if r["subject"] == s); off[s] = (c, c + m); c += m
    assert c == n
    kp = d / "kernel.npy"
    K = np.lib.format.open_memmap(kp, mode="r+" if kp.exists() else "w+", dtype=np.float64, shape=(n, n))
    donef = d / "kernel_done.json"; done = set(map(tuple, json.load(open(donef)))) if donef.exists() else set()
    t0 = time.time()
    for i, s in enumerate(subs):
        todo = [t for t in subs[i:] if (s, t) not in done]
        if not todo: continue
        Xs = np.load(d / f"{s}.npy").astype(np.float64)
        for t in todo:
            Xt = Xs if t == s else np.load(d / f"{t}.npy").astype(np.float64)
            B = Xs @ Xt.T
            (a0, a1), (b0, b1) = off[s], off[t]
            K[a0:a1, b0:b1] = B; K[b0:b1, a0:a1] = B.T
            done.add((s, t)); del Xt
            if time.time() - t0 > a.budget:
                K.flush(); json.dump(sorted(done), open(donef, "w")); print(f"partial: {len(done)}/465 pairs"); return
        del Xs
    K.flush(); json.dump(sorted(done), open(donef, "w"))
    print(f"kernel complete: {len(done)} pairs, n={n}")

if __name__ == "__main__":
    main()
