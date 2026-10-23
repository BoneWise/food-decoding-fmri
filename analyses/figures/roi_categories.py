"""Build the 112 Harvard-Oxford ROIs exactly as in averaging-granularity a23_regions_match.py and write a
label volume coding each ROI by whether it is among the top 20 ROIs of the SVM forward pattern only (1), of
CNN Layer-CAM only (2), or of both (3). Validates ROI names and voxel counts against regions_match_k6.csv.
Usage: python roi_categories.py <atlas_root> <project_root> <out_dir> [<granularity summary dir>]
(summary dir defaults to <project_root>/results/reference/granularity)"""
import csv, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, nibabel as nib
from nibabel import processing
ATLAS, PROJ, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = Path(sys.argv[4]) if len(sys.argv) > 4 else PROJ / "results/reference/granularity"
ref = nib.load(str(PROJ / "results/reference/decoding/masks/svm_common_mask.nii.gz"))
_fm = np.asarray(ref.dataobj); FEAT = np.flatnonzero(np.isfinite(_fm) & (_fm > 0))  # as granularity a01 feature_mask.npy
def labels(xml): return [(int(e.get("index")) + 1, e.text.strip()) for e in ET.parse(xml).getroot().iter("label")]
cort = np.asarray(processing.resample_from_to(nib.load(str(ATLAS/"HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz")), ref, order=0).dataobj).astype(int)
sub = np.asarray(processing.resample_from_to(nib.load(str(ATLAS/"HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz")), ref, order=0).dataobj).astype(int)
ijk = np.indices(ref.shape).reshape(3, -1); x = (ref.affine[:3, :3] @ ijk + ref.affine[:3, 3:4])[0].reshape(ref.shape)
RO, FULL = {}, {}
for img, xml, skip in ((cort, "HarvardOxford-Cortical.xml", False), (sub, "HarvardOxford-Subcortical.xml", True)):
    for v, n in labels(ATLAS/xml):
        if skip and any(t in n.lower() for t in ("white matter", "ventricle", "cerebral cortex")): continue
        for side, sm in (("Left", x < 0), ("Right", x >= 0)):
            full = (img == v) & sm
            m = full.reshape(-1)[FEAT]
            if m.sum() >= 30: RO[f"{side} {n}"] = m; FULL[f"{side} {n}"] = full
rm = {r["roi"]: r for r in csv.DictReader(open(SUMMARY / "regions_match_k6.csv"))}
assert set(RO) == set(rm), (len(RO), len(rm), sorted(set(RO) ^ set(rm))[:5])
bad = [n for n in rm if int(rm[n]["voxels"]) != int(RO[n].sum())]
assert not bad, bad[:5]
cat = np.zeros(ref.shape, dtype=np.int16); rows = []
for n, r in rm.items():
    s, c = int(r["rank_svm"]) <= 20, int(r["rank_cam"]) <= 20
    k = 3 if (s and c) else 1 if s else 2 if c else 0
    if k: cat[FULL[n]] = k
    rows.append(dict(roi=n, group=r["group"], rank_svm=r["rank_svm"], rank_cam=r["rank_cam"], category=["none", "svm_only", "cnn_only", "both"][k]))
nib.save(nib.Nifti1Image(cat, ref.affine), str(OUT / "roi_top20_categories.nii.gz"))
with open(OUT / "roi_top20_categories.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(sorted(rows, key=lambda r: r["category"]))
print("validated", len(RO), "ROIs; voxel counts match")
for k in ("both", "svm_only", "cnn_only"):
    print(k, [ (r["roi"], r["rank_svm"], r["rank_cam"]) for r in rows if r["category"] == k])
