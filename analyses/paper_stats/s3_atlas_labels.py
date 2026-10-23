"""S3: Harvard-Oxford labels for GLM (4>1) and searchlight results.

Same atlas and ROI construction as the Layer-CAM regional summaries
(analyses/haufe_comparison/run_comparison.py): FSL Harvard-Oxford
max-prob thr25 2 mm, nearest-neighbour resampling to the 97x115x97 data
grid, cortical labels split by world x (<0 left), subcortical labels
without white matter / ventricles / cerebral cortex. Macro-region groups
follow results/reference/tables/layercam_all_rois.csv.

Usage: python s3_atlas_labels.py --atlas-root <fsl/data/atlases>
Reads frozen results only; writes CSV files next to this script.
"""
import argparse, csv, json, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, nibabel as nib
from nibabel import processing
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from analyses._paths import OUTPUT_ROOT, RUNS_ROOT  # noqa: E402
HERE = OUTPUT_ROOT / "paper_stats" / "s3_atlas_labels"  # output directory
HERE.mkdir(parents=True, exist_ok=True)
GLM = RUNS_ROOT / "glm_ua1/RESULTS/smoothed_s6_wmcsf_condition_glm_tfce_fwe"
SL = RUNS_ROOT / "searchlight_mvpa1/RESULTS/searchlight_rating4_vs_rating1_ses3fold_n30_cluster_p001_cluster_fwe"

def labels(xml):
    return [(int(e.get("index")) + 1, e.text.strip()) for e in ET.parse(xml).getroot().iter("label")]

def build(ref, root):
    cort = np.asarray(processing.resample_from_to(nib.load(str(root/"HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz")), ref, order=0).dataobj).astype(int)
    sub = np.asarray(processing.resample_from_to(nib.load(str(root/"HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz")), ref, order=0).dataobj).astype(int)
    ijk = np.indices(ref.shape).reshape(3, -1)
    x = (ref.affine[:3, :3] @ ijk + ref.affine[:3, 3:4])[0].reshape(ref.shape)
    rois = []
    for v, n in labels(root/"HarvardOxford-Cortical.xml"):
        for side, m in (("Left", (cort == v) & (x < 0)), ("Right", (cort == v) & (x >= 0))):
            if m.any(): rois.append((f"{side} {n}", "cortical", m))
    for v, n in labels(root/"HarvardOxford-Subcortical.xml"):
        if any(t in n.lower() for t in ("white matter", "ventricle", "cerebral cortex")): continue
        m = sub == v
        if m.any(): rois.append((n, "subcortical", m))
    lab = np.full(ref.shape, "", dtype=object)
    for n, g, m in rois: lab[m] = n
    return rois, lab

def peak_label(lab, ijk, radius=3):
    i, j, k = ijk
    if lab[i, j, k]: return lab[i, j, k], 0.0
    best = ("unlabeled", None)
    for di in range(-radius, radius+1):
        for dj in range(-radius, radius+1):
            for dk in range(-radius, radius+1):
                a, b, c = i+di, j+dj, k+dk
                if 0 <= a < lab.shape[0] and 0 <= b < lab.shape[1] and 0 <= c < lab.shape[2] and lab[a, b, c]:
                    d = 2.0 * (di*di + dj*dj + dk*dk) ** 0.5
                    if best[1] is None or d < best[1]: best = (lab[a, b, c], d)
    return best

def composition(lab, mask, top=3):
    names, counts = np.unique(lab[mask], return_counts=True)
    order = np.argsort(-counts)
    out = [(n if n else "unlabeled", int(c)) for n, c in zip(names[order], counts[order])]
    return "; ".join(f"{n} ({c})" for n, c in out[:top])

def group_of(groups, name, atlas_group=""):
    # Layer-CAM tables list subcortical ROIs as "<Side> <Side> <Structure>"; brain stem is split by side there.
    g = groups.get(name) or groups.get(f"{name.split()[0]} {name}")
    if not g and (atlas_group == "subcortical" or name == "Brain-Stem"): g = "Subcortical"
    return g or ""

def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--atlas-root", required=True); a = ap.parse_args()
    root = Path(a.atlas_root)
    groups = {}
    for r in csv.DictReader(open(ROOT/"results/reference/tables/layercam_all_rois.csv")): groups[r["roi_name"]] = r["division"]
    # GLM 4>1
    gsig_img = nib.load(str(GLM/"randomise/randomise_4gt1_tfce_corrp_tstat1_sigmask_p05.nii.gz"))
    rois, lab = build(gsig_img, root)
    gsig = np.asarray(gsig_img.dataobj) > 0
    comp, n = ndimage.label(gsig, structure=np.ones((3, 3, 3)))
    rows = []
    for r in csv.DictReader(open(GLM/"cluster_table_4gt1_tfce_fwe_p05.csv")):
        ijk = (int(r["peak_i"]), int(r["peak_j"]), int(r["peak_k"]))
        name, dist = peak_label(lab, ijk)
        cid = comp[ijk]
        rows.append(dict(cluster_id=r["cluster_id"], size_voxels=r["size_voxels"], peak_t=round(float(r["peak_t"]), 2),
                         peak_mni=f"({r['peak_x_mni']}, {r['peak_y_mni']}, {r['peak_z_mni']})", peak_label=name,
                         peak_label_distance_mm=dist, macro_group=group_of(groups, name),
                         component_voxels=int((comp == cid).sum()) if cid else 0,
                         composition_top3=composition(lab, comp == cid) if cid else ""))
    rows.sort(key=lambda z: -int(z["size_voxels"]))
    write(HERE/"glm_4gt1_cluster_labels.csv", rows)
    # searchlight
    ssig_img = nib.load(str(SL/"randomise_cluster_p001/randomise_balanced_accuracy_minus_chance_cluster_p001_clusterFWE05_sigmask.nii.gz"))
    ssig = np.asarray(ssig_img.dataobj) > 0
    center = np.asarray(nib.load(str(SL/"group_inputs/center_mask.nii.gz")).dataobj) > 0
    mean = np.asarray(nib.load(str(SL/"group_inputs/searchlight_balanced_accuracy_minus_chance_n30_mean.nii.gz")).dataobj, dtype=float)
    clab = np.asarray(nib.load(str(SL/"randomise_cluster_p001/randomise_balanced_accuracy_minus_chance_cluster_p001_clusterFWE05_clusterLabels_26conn.nii.gz")).dataobj).astype(int)
    prow = []
    for r in csv.DictReader(open(SL/"cluster_table_balanced_accuracy_minus_chance_cluster_p001_cluster_fwe_p05_26conn.csv")):
        ijk = (int(r["peak_i"]), int(r["peak_j"]), int(r["peak_k"]))
        name, dist = peak_label(lab, ijk)
        cid = clab[ijk]
        prow.append(dict(component_id=r["component_id"], size_voxels=r["size_voxels"], peak_t=round(float(r["peak_t"]), 2),
                         peak_mni=f"({r['peak_x_mni']}, {r['peak_y_mni']}, {r['peak_z_mni']})", peak_label=name,
                         peak_label_distance_mm=dist, macro_group=group_of(groups, name),
                         composition_top3=composition(lab, clab == cid) if cid else ""))
    write(HERE/"searchlight_component_labels.csv", prow)
    rrows = []
    for name, g, m in rois:
        m = m & center
        if not m.any(): continue
        s = m & ssig
        rrows.append(dict(roi_name=name, atlas_group=g, macro_group=group_of(groups, name, g), roi_center_voxels=int(m.sum()),
                          significant_voxels=int(s.sum()), coverage=round(float(s.sum()/m.sum()), 4),
                          mean_local_ba_significant=round(0.5+float(mean[s].mean()), 4) if s.any() else "",
                          mean_local_ba_roi=round(0.5+float(mean[m].mean()), 4)))
    rrows.sort(key=lambda z: -z["coverage"])
    write(HERE/"searchlight_roi_summary.csv", rrows)
    grows = []
    for gname in sorted({r["macro_group"] for r in rrows if r["macro_group"]}):
        sub = [r for r in rrows if r["macro_group"] == gname]
        tot = sum(r["roi_center_voxels"] for r in sub); sig = sum(r["significant_voxels"] for r in sub)
        grows.append(dict(macro_group=gname, roi_center_voxels=tot, significant_voxels=sig, coverage=round(sig/tot, 4)))
    labeled = np.zeros(center.shape, bool)
    for _, _, m in rois: labeled |= m
    grows.append(dict(macro_group="unlabeled (outside atlas ROIs)", roi_center_voxels=int((center & ~labeled).sum()),
                      significant_voxels=int((ssig & ~labeled).sum()), coverage=round(float((ssig & ~labeled).sum()/max(1,(center & ~labeled).sum())), 4)))
    write(HERE/"searchlight_group_summary.csv", grows)
    json.dump(dict(atlas="FSL Harvard-Oxford maxprob thr25 2mm", atlas_root=str(root), glm_sig_voxels=int(gsig.sum()),
                   searchlight_sig_voxels=int(ssig.sum()), searchlight_center_voxels=int(center.sum())),
              open(HERE/"s3_run_info.json", "w"), indent=2)

if __name__ == "__main__":
    main()
