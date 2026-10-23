"""a91: validate k=1 predictions (matched by subject/session/run/trial) and the k=1 Haufe map against frozen runs."""
import csv, sys
from pathlib import Path
import numpy as np, nibabel as nib
import common

work = Path(sys.argv[1]) / "k01_seed42"; fit = work / "fit_k01"
idx = list(csv.DictReader(open(work / "index.csv")))
tri = {s: {int(t["row"]): t for t in common.trials(s)} for s in common.SUBJECTS}
key = {}
for i, r in enumerate(idx):
    t = tri[r["subject"]][int(r["source_rows"])]; key[i] = (r["subject"], t["session"], t["run"], int(t["trial"]))
mine = {key[int(r["row"])]: r for r in csv.DictReader(open(fit / "predictions.csv"))}
fro = {(r["subject_id"], r["session_id"], r["run_id"], int(r["trial_id"])): r for r in
       csv.DictReader(open(common.RUNS_ROOT / "20260719_matched_cross_subject_svm_k01_cnnfolds/oof_predictions.csv"))}
assert set(mine) == set(fro), (len(mine), len(fro), len(set(mine) ^ set(fro)))
ks = sorted(mine)
dm = np.array([float(mine[k]["decision"]) for k in ks]); df = np.array([float(fro[k]["decision_score"]) for k in ks])
pm = np.array([int(mine[k]["pred"]) for k in ks]); pf = np.array([int(fro[k]["pred"]) for k in ks]); t = np.array([int(fro[k]["target"]) for k in ks])
def ba(tt, pp): return 0.5 * ((pp[tt == 1] == 1).mean() + (pp[tt == 0] == 0).mean())
sub = np.array([k[0] for k in ks])
bm = [ba(t[sub == s], pm[sub == s]) for s in common.SUBJECTS]; bf = [ba(t[sub == s], pf[sub == s]) for s in common.SUBJECTS]
print(f"k1 decision r={np.corrcoef(dm, df)[0,1]:.8f} max|diff|={np.abs(dm-df).max():.2e}; preds differ {int((pm!=pf).sum())}/{len(ks)}")
print(f"k1 participant-mean BA mine {100*np.mean(bm):.3f} frozen {100*np.mean(bf):.3f}")
H = np.load(fit / "haufe_mean.npy").astype(np.float64)
hf = np.load(common.OUTPUT_ROOT / "haufe/outputs/maps/mean_haufe_pattern_vector_rating4_minus_rating1.npy").astype(np.float64)
print("frozen haufe vector", hf.shape)
if hf.shape == H.shape:
    print(f"haufe r={np.corrcoef(H, hf)[0,1]:.8f}; max|diff|/max|frozen|={np.abs(H-hf).max()/np.abs(hf).max():.2e}")
