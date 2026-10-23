"""a01: extract rating-1/4 single-trial betas into a compact per-participant cache (read-only on sources).

For each included participant (sub-016 excluded, as in the paper), reads the participant brain mask and every
rating-1/4 single-trial beta listed in minimal_lsa/output/beta_index.csv, and stores the raw float32 values of
the brain-mask voxels in C order (NaNs kept), so that the original merging (nanmean over k maps) and the
original standardization (mean/SD over finite brain-mask voxels) can be reproduced exactly elsewhere.

Outputs (cache/, git-ignored):
  sub-XXX_betas.npy   float32 [n_trials, n_mask_voxels]
  sub-XXX_mask.npy    int64   flat C-order indices of the brain mask in the 97x115x97 grid
  trials.csv          one row per trial: subject, session, run, trial, rating, row in the subject array, beta path
  feature_mask.npy    int64   flat indices of the 222,441-voxel common feature mask
Usage: python a01_extract_trials.py --lsa-root <LSA root containing output/> --out <granularity root>/cache
         --feature-mask results/reference/decoding/masks/svm_common_mask.nii.gz
         [--path-prefix-from <beta_file prefix recorded in beta_index.csv>] [--subjects sub-001,...]
"""
import argparse, csv, time
from pathlib import Path
import numpy as np, nibabel as nib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lsa-root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", default=""); ap.add_argument("--feature-mask", required=True)
    ap.add_argument("--path-prefix-from", default="", help="prefix of beta_file in beta_index.csv to replace by --lsa-root")
    a = ap.parse_args()
    root, out = Path(a.lsa_root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fm = nib.load(a.feature_mask); fmask = np.asarray(fm.dataobj); fmask = np.isfinite(fmask) & (fmask > 0)
    np.save(out / "feature_mask.npy", np.flatnonzero(fmask))
    rows = [r for r in csv.DictReader(open(root / "output/beta_index.csv")) if r["rating_value"] in ("1", "4")]
    rows = [r for r in rows if r["sub"] != "sub-016"]
    want = set(a.subjects.split(",")) if a.subjects else None
    subs = sorted({r["sub"] for r in rows if want is None or r["sub"] in want})
    for s in subs:
        if (out / f"{s}_betas.npy").exists():
            print(s, "exists; skip", flush=True); continue
        t0 = time.time()
        mimg = nib.load(str(root / f"output/cache/masks/{s}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii"))
        md = np.asarray(mimg.dataobj); bmask = np.isfinite(md) & (md > 0)
        assert bmask.shape == fmask.shape and not np.any(fmask & ~bmask)
        idx = np.flatnonzero(bmask)
        srows = sorted([r for r in rows if r["sub"] == s], key=lambda r: (r["session"], r["run"], int(r["trial_in_run"])))
        arr = np.empty((len(srows), idx.size), dtype=np.float32)
        meta = []
        for i, r in enumerate(srows):
            path = Path(r["beta_file"].replace(a.path_prefix_from, str(root)) if a.path_prefix_from else r["beta_file"])
            img = nib.load(str(path))
            assert img.shape == bmask.shape and np.allclose(img.affine, mimg.affine)
            arr[i] = np.asarray(img.dataobj, dtype=np.float32).reshape(-1)[idx]
            meta.append(dict(subject=s, session=r["session"], run=r["run"], trial=int(r["trial_in_run"]),
                             rating=int(r["rating_value"]), row=i, beta_file=r["beta_file"]))
        np.save(out / f"{s}_mask.npy", idx)
        tmp = out / f"{s}_betas.tmp.npy"; np.save(tmp, arr); tmp.rename(out / f"{s}_betas.npy")
        with open(out / f"{s}_trials.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(meta[0])); w.writeheader(); w.writerows(meta)
        print(f"{s}: {len(srows)} trials x {idx.size} voxels, {arr.nbytes/1e6:.0f} MB, {time.time()-t0:.0f} s", flush=True)

if __name__ == "__main__":
    main()
