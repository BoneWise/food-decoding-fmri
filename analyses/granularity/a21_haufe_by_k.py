"""a21: how the SVM forward (Haufe) pattern changes with k (fixed k=1 partition; seeds 42-46 for k>1).

Maps: fold-mean Haufe pattern per condition; for k>1 averaged over the five grouping seeds; each map z-scored
within the 222,441 feature voxels. Stability: mean pairwise Pearson r between the 10 fold patterns (averaged
over seeds) and, for k>1, between the 5 seed maps. Regional summary: Harvard-Oxford maxprob thr25 2 mm,
hemispheres split by world x, ROIs with >= 100 voxels (as S5); group assignment from layercam_all_rois.csv.
Matched-k comparison with CNN Layer-CAM (k=6): Spearman across ROIs between mean |z| and the mean of the two
Layer-CAM class maps (S5 table). Descriptive; no spatial inference.
"""
import csv, json, sys, xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path
import numpy as np, nibabel as nib
from nibabel import processing
from scipy import stats
import common

ATLAS = Path(sys.argv[1]); R = common.ROOT / "results"; OUT = common.ROOT / "results_summary"
PROJ = common.REPO
SEEDS = (42, 43, 44, 45, 46); KS = (1, 2, 3, 4, 5, 6)

def labels(xml):
    return [(int(e.get("index")) + 1, e.text.strip()) for e in ET.parse(xml).getroot().iter("label")]

def rois():
    ref = nib.load(str(common.FEATURE_MASK)); fm = np.asarray(ref.dataobj)
    fm = np.isfinite(fm) & (fm > 0); assert np.array_equal(np.flatnonzero(fm), common.FEAT)
    cort = np.asarray(processing.resample_from_to(nib.load(str(ATLAS/"HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz")), ref, order=0).dataobj).astype(int)
    sub = np.asarray(processing.resample_from_to(nib.load(str(ATLAS/"HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz")), ref, order=0).dataobj).astype(int)
    ijk = np.indices(ref.shape).reshape(3, -1); x = (ref.affine[:3, :3] @ ijk + ref.affine[:3, 3:4])[0].reshape(ref.shape)
    out = {}
    for img, xml, skip in ((cort, "HarvardOxford-Cortical.xml", False), (sub, "HarvardOxford-Subcortical.xml", True)):
        for v, n in labels(ATLAS/xml):
            if skip and any(t in n.lower() for t in ("white matter", "ventricle", "cerebral cortex")): continue
            for side, sm in (("Left", x < 0), ("Right", x >= 0)):
                m = ((img == v) & sm).reshape(-1)[common.FEAT]
                if m.sum() >= 100: out[f"{side} {n}"] = m
    return out

def z(v): v = np.asarray(v, np.float64); return (v - v.mean()) / v.std()
def mean_r(maps): return float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in combinations(maps, 2)]))

def cond_dir(k, s): return R / ("k01_seed42" if k == 1 else f"k{k:02d}_seed{s}") / "fit_k01"
maps, fold_stab, seed_stab = {}, {}, {}
for k in KS:
    seeds = SEEDS if k > 1 else (42,)
    per_seed = [np.load(cond_dir(k, s) / "haufe_mean.npy").astype(np.float64) for s in seeds]
    maps[k] = z(np.mean([z(m) for m in per_seed], axis=0))
    fold_stab[k] = float(np.mean([mean_r(list(np.load(cond_dir(k, s) / "haufe_folds.npy").astype(np.float64))) for s in seeds]))
    seed_stab[k] = mean_r([z(m) for m in per_seed]) if k > 1 else None
sim = {f"k{a}-k{b}": float(np.corrcoef(maps[a], maps[b])[0, 1]) for a, b in combinations(KS, 2)}
RO = rois(); division = {t["roi_name"]: t["division"] for t in csv.DictReader(open(PROJ / "results/reference/tables/layercam_all_rois.csv"))}
s5 = {r["roi"]: r for r in csv.DictReader(open(common.PAPER_STATS / "s5_haufe_layercam_regions/s5_roi_values.csv"))}
names = sorted(RO); rows = []
for n in names:
    row = dict(roi=n, group=division.get(n, ""), voxels=int(RO[n].sum()))
    for k in KS:
        row[f"z_k{k}"] = maps[k][RO[n]].mean(); row[f"absz_k{k}"] = np.abs(maps[k][RO[n]]).mean()
    rows.append(row)
with open(OUT / "haufe_by_k_rois.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
groups = ["Occipital", "Parietal", "Temporal", "Frontal", "Subcortical"]
grows = []
for g in groups:
    gm = np.any([RO[n] for n in names if division.get(n) == g], axis=0)
    grows.append(dict(group=g, **{f"absz_k{k}": float(np.abs(maps[k][gm]).mean()) for k in KS}, **{f"z_k{k}": float(maps[k][gm].mean()) for k in KS}))
with open(OUT / "haufe_by_k_groups.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(grows[0])); w.writeheader(); w.writerows(grows)
lc = np.array([float(s5[n]["lc_mean"]) for n in names]); lcd = np.array([float(s5[n]["lc_high_minus_low"]) for n in names])
match = {f"k{k}": dict(magnitude_rho=float(stats.spearmanr([r[f"absz_k{k}"] for r in rows], lc).correlation),
                        direction_rho=float(stats.spearmanr([r[f"z_k{k}"] for r in rows], lcd).correlation)) for k in KS}
summ = dict(fold_stability_mean_r=fold_stab, seed_stability_mean_r=seed_stab, cross_k_r=sim, layercam_k6_roi_match=match, n_rois=len(names))
json.dump(summ, open(OUT / "haufe_by_k_summary.json", "w"), indent=2)
print(json.dumps(summ, indent=1))
for g in grows: print(g["group"], " |z|:", " ".join(f"{g[f'absz_k{k}']:.3f}" for k in KS), " z:", " ".join(f"{g[f'z_k{k}']:+.3f}" for k in KS))
for k in (1, 3, 6):
    top = sorted(rows, key=lambda r: -r[f"z_k{k}"])[:8]; bot = sorted(rows, key=lambda r: r[f"z_k{k}"])[:6]
    print(f"k{k} +:", [(r["roi"].replace("Right ", "R ").replace("Left ", "L "), round(r[f"z_k{k}"], 2)) for r in top])
    print(f"k{k} -:", [(r["roi"].replace("Right ", "R ").replace("Left ", "L "), round(r[f"z_k{k}"], 2)) for r in bot])
