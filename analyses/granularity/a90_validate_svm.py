"""a90: compare a04 predictions with a frozen LinearSVC run (row order identical; checked by target and subject)."""
import argparse, csv
from pathlib import Path
import numpy as np
import common

def ba(t, p):
    t, p = np.asarray(t), np.asarray(p)
    return 0.5 * ((p[t == 1] == 1).mean() + (p[t == 0] == 0).mean())

ap = argparse.ArgumentParser(); ap.add_argument("--mine", required=True); ap.add_argument("--frozen", required=True); a = ap.parse_args()
m = {int(r["row"]): r for r in csv.DictReader(open(a.mine))}
f = {int(r["row_index"]): r for r in csv.DictReader(open(a.frozen))}
assert set(m) == set(f), (len(m), len(f))
rows = sorted(m)
for i in rows:
    assert m[i]["subject"] == f[i]["subject_id"] and int(m[i]["target"]) == int(f[i]["target"])
dm = np.array([float(m[i]["decision"]) for i in rows]); df = np.array([float(f[i]["decision_score"]) for i in rows])
pm = np.array([int(m[i]["pred"]) for i in rows]); pf = np.array([int(f[i]["pred"]) for i in rows])
subs = sorted({m[i]["subject"] for i in rows})
bam, baf = [], []
for s in subs:
    r = [j for j, i in enumerate(rows) if m[i]["subject"] == s]
    t = np.array([int(m[rows[j]]["target"]) for j in r])
    bam.append(ba(t, pm[r])); baf.append(ba(t, pf[r]))
print(f"decision r = {np.corrcoef(dm, df)[0,1]:.8f}; max |diff| = {np.abs(dm-df).max():.2e} (scale {np.abs(df).mean():.3f})")
print(f"prediction agreement {np.mean(pm==pf):.5f} ({int((pm!=pf).sum())} of {len(rows)} differ)")
print(f"participant-mean BA: mine {100*np.mean(bam):.3f}% frozen {100*np.mean(baf):.3f}%; max per-participant |diff| {100*np.max(np.abs(np.array(bam)-np.array(baf))):.3f} pp")
