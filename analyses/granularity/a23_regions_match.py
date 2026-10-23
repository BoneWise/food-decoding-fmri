"""a23: matched-granularity (k=6) regional comparison of the SVM forward pattern with CNN Layer-CAM.
Forward pattern: k=6 Haufe maps of seeds 42-46 (fixed partition), each z-scored, averaged, z-scored.
Layer-CAM: group-first ROI values (whole-brain mean = 1) from results/reference/tables/layercam_all_rois.csv,
mean of the two classes and high-minus-low. ROIs: Harvard-Oxford maxprob thr25 2 mm, hemisphere split by
world x, >= 30 voxels in the feature mask (112 ROIs; includes accumbens). Descriptive only."""
import csv, json, sys
import numpy as np
from scipy import stats
import common

import xml.etree.ElementTree as ET
from pathlib import Path
import nibabel as nib
from nibabel import processing
ATLAS = Path(sys.argv[1]); PROJ = common.REPO; OUT = common.ROOT / "results_summary"
def labels(xml): return [(int(e.get("index")) + 1, e.text.strip()) for e in ET.parse(xml).getroot().iter("label")]
ref = nib.load(str(common.FEATURE_MASK))
cort = np.asarray(processing.resample_from_to(nib.load(str(ATLAS/"HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz")), ref, order=0).dataobj).astype(int)
sub = np.asarray(processing.resample_from_to(nib.load(str(ATLAS/"HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz")), ref, order=0).dataobj).astype(int)
ijk = np.indices(ref.shape).reshape(3, -1); x = (ref.affine[:3, :3] @ ijk + ref.affine[:3, 3:4])[0].reshape(ref.shape)
RO = {}
for img, xml, skip in ((cort, "HarvardOxford-Cortical.xml", False), (sub, "HarvardOxford-Subcortical.xml", True)):
    for v, n in labels(ATLAS/xml):
        if skip and any(t in n.lower() for t in ("white matter", "ventricle", "cerebral cortex")): continue
        for side, sm in (("Left", x < 0), ("Right", x >= 0)):
            m = ((img == v) & sm).reshape(-1)[common.FEAT]
            if m.sum() >= 30: RO[f"{side} {n}"] = m
z = lambda v: (v - v.mean()) / v.std()
H = z(np.mean([z(np.load(common.ROOT / f"results/k06_seed{s}/fit_k01/haufe_mean.npy").astype(np.float64)) for s in (42, 43, 44, 45, 46)], axis=0))
lc = {}
for t in csv.DictReader(open(PROJ / "results/reference/tables/layercam_all_rois.csv")):
    lc.setdefault(t["roi_name"], {})[int(t["rating"])] = float(t["mean_layercam_whole_brain_1"]); lc[t["roi_name"]]["group"] = t["division"]
names = sorted(n for n in RO if n in lc)
rows = [dict(roi=n, group=lc[n]["group"], voxels=int(RO[n].sum()), absz=float(np.abs(H[RO[n]]).mean()), z=float(H[RO[n]].mean()),
             lc_low=lc[n][1], lc_high=lc[n][4], lc_mean=(lc[n][1] + lc[n][4]) / 2, lc_high_minus_low=lc[n][4] - lc[n][1]) for n in names]
a = np.array([r["absz"] for r in rows]); l = np.array([r["lc_mean"] for r in rows])
ra, rl = stats.rankdata(-a), stats.rankdata(-l)
for r, p, q in zip(rows, ra, rl): r["rank_svm"], r["rank_cam"] = int(p), int(q)
with open(OUT / "regions_match_k6.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
both = sorted([r for r in rows if r["rank_svm"] <= 20 and r["rank_cam"] <= 20], key=lambda r: r["rank_svm"] + r["rank_cam"])
summ = dict(n_rois=len(rows), spearman_magnitude=float(stats.spearmanr(a, l).correlation), top20_overlap=len(both),
            top20_chance=400 / len(rows), median_absz=float(np.median(a)), median_cam=float(np.median(l)),
            top20_both=[(r["roi"], r["rank_svm"], r["rank_cam"]) for r in both])
json.dump(summ, open(OUT / "regions_match_k6.json", "w"), indent=1); print(json.dumps(summ, indent=1))
for n in ["Right Parahippocampal Gyrus, posterior division", "Left Parahippocampal Gyrus, posterior division", "Right Right Hippocampus", "Right Precuneous Cortex", "Left Precuneous Cortex",
          "Right Cingulate Gyrus, posterior division", "Left Cingulate Gyrus, posterior division", "Right Insular Cortex", "Left Left Accumbens", "Right Right Accumbens",
          "Right Right Thalamus", "Left Left Thalamus", "Right Right Pallidum", "Right Right Putamen", "Right Right Amygdala", "Left Left Caudate", "Right Right Caudate"]:
    r = next(r for r in rows if r["roi"] == n)
    print(f"{n[:44]:44s} SVM {r['rank_svm']:3d} CAM {r['rank_cam']:3d} absz {r['absz']:.2f} z {r['z']:+.2f} camHL {r['lc_high_minus_low']:+.3f}")
