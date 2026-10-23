"""S5: region-level comparison of the k=1 SVM Haufe pattern with the k=6 CNN Layer-CAM maps.

Descriptive only (no spatial inference). Reads frozen maps; writes CSVs next to this script.
Haufe: signed fold-mean pattern, rating4 minus rating1 (positive = activity higher with high-value decision scores).
Layer-CAM: class-specific group-mean maps, each divided by its whole-brain mean (group-first).
Comparisons made like-for-like:
  direction: Haufe (signed) vs Layer-CAM high - low;
  magnitude: |Haufe| vs mean of the two Layer-CAM class maps.
Atlas, resampling and hemisphere split as in S3/S4 (Harvard-Oxford maxprob thr25 2 mm, world x).
"""
import argparse, csv, json, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, nibabel as nib
from nibabel import processing
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from analyses._paths import OUTPUT_ROOT, RUNS_ROOT  # noqa: E402
HERE = OUTPUT_ROOT / "paper_stats" / "s5_haufe_layercam_regions"  # output directory
HERE.mkdir(parents=True, exist_ok=True)
RUN = RUNS_ROOT / "20260503_200532_fmri_rating14_mergedk06_seed42_fixed5/gradcam_subject_tfce"
HAUFE = OUTPUT_ROOT / "haufe/outputs/maps/mean_haufe_pattern_rating4_minus_rating1.nii.gz"
GROUPS = ["Occipital", "Parietal", "Temporal", "Frontal", "Subcortical"]
S4 = HERE.parent / "s4_layercam_participant_tests/s4_high_minus_low.csv"

def labels(xml):
    return [(int(e.get("index")) + 1, e.text.strip()) for e in ET.parse(xml).getroot().iter("label")]

def rois(ref, root, mask):
    cort = np.asarray(processing.resample_from_to(nib.load(str(root/"HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz")), ref, order=0).dataobj).astype(int)
    sub = np.asarray(processing.resample_from_to(nib.load(str(root/"HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz")), ref, order=0).dataobj).astype(int)
    ijk = np.indices(ref.shape).reshape(3, -1)
    x = (ref.affine[:3, :3] @ ijk + ref.affine[:3, 3:4])[0].reshape(ref.shape)
    out = {}
    for img, xml, skip in ((cort, "HarvardOxford-Cortical.xml", False), (sub, "HarvardOxford-Subcortical.xml", True)):
        for v, n in labels(root/xml):
            if skip and any(t in n.lower() for t in ("white matter", "ventricle", "cerebral cortex")): continue
            for side, sm in (("Left", x < 0), ("Right", x >= 0)):
                m = (img == v) & sm & mask
                if m.any(): out[f"{side} {n}"] = m
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--atlas-root", required=True); a = ap.parse_args()
    mimg = nib.load(str(RUN/"common_brain_mask.nii.gz")); cmask = np.asarray(mimg.dataobj) > 0
    himg = nib.load(str(HAUFE)); assert np.allclose(himg.affine, mimg.affine) and himg.shape == mimg.shape
    H = np.asarray(himg.dataobj, float)
    l1 = np.asarray(nib.load(str(RUN/"rating_1_group_mean_layercam.nii.gz")).dataobj, float)
    l4 = np.asarray(nib.load(str(RUN/"rating_4_group_mean_layercam.nii.gz")).dataobj, float)
    l1n, l4n = l1 / l1[cmask].mean(), l4 / l4[cmask].mean()   # group-first, as in the manuscript
    mask = cmask & (H != 0) & np.isfinite(H)
    nshared = int(mask.sum())
    Hz = (H - H[mask].mean()) / H[mask].std()
    D, A = l4n - l1n, (l4n + l1n) / 2
    R = rois(mimg, Path(a.atlas_root), mask)
    table = list(csv.DictReader(open(ROOT/"results/reference/tables/layercam_all_rois.csv")))
    division = {t["roi_name"]: t["division"] for t in table}
    rows = []
    for n, m in sorted(R.items()):
        rows.append(dict(roi=n, group=division.get(n, ""), voxels=int(m.sum()),
                         haufe_z=Hz[m].mean(), abs_haufe_z=np.abs(Hz[m]).mean(),
                         lc_low=l1n[m].mean(), lc_high=l4n[m].mean(), lc_high_minus_low=D[m].mean(), lc_mean=A[m].mean()))
    elig = [r for r in rows if r["voxels"] >= 100]
    summ = {"n_shared_voxels": nshared, "n_roi_ge100": len(elig)}
    def rho(x, y):
        return float(stats.spearmanr(x, y).correlation)
    summ["voxel_spearman_direction_haufe_vs_lc_high_minus_low"] = rho(H[mask], D[mask])
    summ["voxel_spearman_magnitude_abs_haufe_vs_lc_mean"] = rho(np.abs(H[mask]), A[mask])
    summ["roi_spearman_direction"] = rho([r["haufe_z"] for r in elig], [r["lc_high_minus_low"] for r in elig])
    summ["roi_spearman_magnitude"] = rho([r["abs_haufe_z"] for r in elig], [r["lc_mean"] for r in elig])
    # sign agreement over ROIs
    agree = [np.sign(r["haufe_z"]) == np.sign(r["lc_high_minus_low"]) for r in elig]
    summ["roi_sign_agreement_fraction"] = float(np.mean(agree))
    # groups
    grows = []
    for g in GROUPS:
        gm = np.any([R[n] for n in R if division.get(n) == g], axis=0)
        grows.append(dict(group=g, voxels=int(gm.sum()), haufe_z=Hz[gm].mean(), abs_haufe_z=np.abs(Hz[gm]).mean(),
                          lc_high_minus_low=D[gm].mean(), lc_mean=A[gm].mean()))
    # S4 regions: Haufe sign vs participant-level class difference
    s4 = list(csv.DictReader(open(S4)))
    srows = []
    for r in s4:
        reg = r["region"]
        m = np.any([R[n] for n in R if division.get(n) == reg], axis=0) if reg in GROUPS else R[reg]
        srows.append(dict(family=r["family"], region=reg, s4_high_minus_low=float(r["high_minus_low_mean"]),
                          s4_p_holm=float(r["p_holm"]), haufe_z=Hz[m].mean()))
    for path, data in ((HERE/"s5_roi_values.csv", rows), (HERE/"s5_group_values.csv", grows), (HERE/"s5_s4_regions.csv", srows)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    json.dump(summ, open(HERE/"s5_summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))
    for g in grows: print(f"{g['group']:12s} haufe_z {g['haufe_z']:+.3f} |z| {g['abs_haufe_z']:.3f} LC hi-lo {g['lc_high_minus_low']:+.3f} LC mean {g['lc_mean']:.3f}")
    for s in srows: print(f"{s['region'][:40]:40s} S4 {s['s4_high_minus_low']:+.3f} (p {s['s4_p_holm']:.3f})  haufe_z {s['haufe_z']:+.3f}")
    top = sorted(elig, key=lambda r: -r["haufe_z"])[:10]; bot = sorted(elig, key=lambda r: r["haufe_z"])[:10]
    mag = sorted(elig, key=lambda r: -r["abs_haufe_z"])[:10]
    print("Haufe most positive (high):", [(r["roi"], round(r["haufe_z"], 2)) for r in top])
    print("Haufe most negative (low):", [(r["roi"], round(r["haufe_z"], 2)) for r in bot])
    print("Haufe largest |z|:", [(r["roi"], round(r["abs_haufe_z"], 2)) for r in mag])

if __name__ == "__main__":
    main()
