"""S4: participant-level tests for the regional Layer-CAM summaries (Figures 4 and 5).

Inputs (read only):
  experiments/20260503_200532_.../gradcam_subject_tfce/subject_maps/rating_{1,4}/sub-XXX_rating_R_mean_layercam.nii.gz
  experiments/20260503_200532_.../gradcam_subject_tfce/common_brain_mask.nii.gz
  results/reference/tables/layercam_all_rois.csv                          (ROI names, groups, validation)
  FSL Harvard-Oxford maxprob thr25 2 mm (--atlas-root)

ROIs are rebuilt exactly as in the frozen summaries (nearest-neighbour resampling,
cortical and subcortical labels split by world x); voxel counts and ROI means recomputed
from the participant maps are checked against layercam_all_rois.csv before testing.

Participant-first value = mean CAM in ROI (or regional group) / participant's whole-brain mean.
Tests (two-sided sign flips, 100,000 flips, seed 20261020):
  - against 1, per class (27 low-value, 29 high-value participants);
  - high minus low, paired, for the 26 participants contributing to both classes.
Holm correction within each figure family: Figure 4 (5 groups), Figure 5 (10 ROIs),
separately for the against-1 tests (both classes together) and the paired tests.
"""
import argparse, csv, json, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, nibabel as nib
from nibabel import processing

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from analyses._paths import OUTPUT_ROOT, RUNS_ROOT  # noqa: E402
HERE = OUTPUT_ROOT / "paper_stats" / "s4_layercam_participant_tests"  # output directory
HERE.mkdir(parents=True, exist_ok=True)
RUN = RUNS_ROOT / "20260503_200532_fmri_rating14_mergedk06_seed42_fixed5/gradcam_subject_tfce"
SEED, NFLIP, NBOOT = 20261020, 100_000, 100_000
GROUPS = ["Occipital", "Parietal", "Temporal", "Frontal", "Subcortical"]
FIG5 = ["Right Parahippocampal Gyrus, posterior division", "Right Right Hippocampus", "Right Precuneous Cortex",
        "Left Precuneous Cortex", "Right Cingulate Gyrus, posterior division", "Left Cingulate Gyrus, posterior division",
        "Right Insular Cortex", "Left Left Accumbens", "Right Right Accumbens", "Right Right Thalamus"]

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

def signflip(d, rng):
    d = np.asarray(d, float); obs = abs(d.mean())
    null = np.abs((rng.choice([-1.0, 1.0], size=(NFLIP, d.size)) * d).mean(axis=1))
    return (np.sum(null >= obs - 1e-12) + 1) / (NFLIP + 1)

def boot(d, rng):
    d = np.asarray(d, float)
    m = d[rng.integers(0, d.size, size=(NBOOT, d.size))].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--atlas-root", required=True); a = ap.parse_args()
    mimg = nib.load(str(RUN/"common_brain_mask.nii.gz")); mask = np.asarray(mimg.dataobj) > 0
    R = rois(mimg, Path(a.atlas_root), mask)
    table = list(csv.DictReader(open(ROOT/"results/reference/tables/layercam_all_rois.csv")))
    names = sorted({t["roi_name"] for t in table}); division = {t["roi_name"]: t["division"] for t in table}
    assert set(names) == set(R), (set(names) ^ set(R))
    # validation: voxel counts and ROI means (mean of participant ROI means, as stored in the
    # frozen table's roi_mean_layercam) must reproduce layercam_all_rois.csv
    maxdiff = 0.0
    for r in (1, 4):
        maps = [np.asarray(nib.load(str(f)).dataobj, float) for f in sorted((RUN/f"subject_maps/rating_{r}").glob("*.nii.gz"))]
        for t in table:
            if int(t["rating"]) != r: continue
            assert int(t["num_voxels"]) == int(R[t["roi_name"]].sum())
            est = np.mean([d[R[t["roi_name"]]].mean() for d in maps])
            maxdiff = max(maxdiff, abs(est - float(t["roi_mean_layercam"])) / float(t["roi_mean_layercam"]))
    assert maxdiff < 5e-4, maxdiff  # float32 storage of participant maps
    gmask = {gname: np.any([R[n] for n in names if division[n] == gname], axis=0) for gname in GROUPS}
    # participant-first values
    vals = {}
    for r in (1, 4):
        for f in sorted((RUN/f"subject_maps/rating_{r}").glob(f"sub-*_rating_{r}_mean_layercam.nii.gz")):
            s = f.name.split("_")[0]
            d = np.asarray(nib.load(str(f)).dataobj, float); wb = d[mask].mean()
            row = {n: d[R[n]].mean() / wb for n in FIG5}
            row.update({g: d[gmask[g]].mean() / wb for g in GROUPS})
            vals[(r, s)] = row
    subs = {r: sorted(s for rr, s in vals if rr == r) for r in (1, 4)}
    both = sorted(set(subs[1]) & set(subs[4]))
    assert (len(subs[1]), len(subs[4]), len(both)) == (27, 29, 26)
    rng = np.random.default_rng(SEED)
    out1, out2 = [], []
    for fam, items in (("Figure 4", GROUPS), ("Figure 5", FIG5)):
        block = []
        for item in items:
            for r in (1, 4):
                v = np.array([vals[(r, s)][item] for s in subs[r]])
                lo, hi = boot(v, rng)
                block.append(dict(family=fam, region=item, value_class="low" if r == 1 else "high", n=v.size,
                                  participant_first_mean=v.mean(), sd_sample=v.std(ddof=1), ci_low=lo, ci_high=hi,
                                  n_above_1=int((v > 1).sum()), p_signflip_vs1=signflip(v - 1, rng)))
        for row, p in zip(block, holm([b["p_signflip_vs1"] for b in block])): row["p_holm"] = p
        out1 += block
        block = []
        for item in items:
            d = np.array([vals[(4, s)][item] - vals[(1, s)][item] for s in both])
            lo, hi = boot(d, rng)
            block.append(dict(family=fam, region=item, n=d.size, high_minus_low_mean=d.mean(), sd_sample=d.std(ddof=1),
                              ci_low=lo, ci_high=hi, n_high_greater=int((d > 0).sum()), p_signflip=signflip(d, rng)))
        for row, p in zip(block, holm([b["p_signflip"] for b in block])): row["p_holm"] = p
        out2 += block
    for path, rows in ((HERE/"s4_vs_whole_brain_mean.csv", out1), (HERE/"s4_high_minus_low.csv", out2)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(seed=SEED, n_flip=NFLIP, n_boot=NBOOT, n_low=27, n_high=29, n_both=26,
                   roi_validation_max_abs_diff=maxdiff, atlas_root=a.atlas_root), open(HERE/"s4_run_info.json", "w"), indent=2)
    for r in out1: print(f"{r['family']} | {r['region'][:38]:38s} {r['value_class']:4s} {r['participant_first_mean']:.3f} [{r['ci_low']:.3f},{r['ci_high']:.3f}] >1:{r['n_above_1']}/{r['n']} p={r['p_signflip_vs1']:.4f} holm={r['p_holm']:.4f}")
    for r in out2: print(f"{r['family']} | {r['region'][:38]:38s} H-L {r['high_minus_low_mean']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] H>L:{r['n_high_greater']}/26 p={r['p_signflip']:.4f} holm={r['p_holm']:.4f}")

if __name__ == "__main__":
    main()
