"""a10: resumable driver for one condition (k, grouping seed): features -> kernel -> fits -> Haufe -> copy
results to results/<cond>/ -> delete the scratch features. Re-run until it prints COMPLETE.
Each call stays within --budget seconds (device shell limit)."""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
import common

def run(args):
    r = subprocess.run([sys.executable] + args, cwd=Path(__file__).parent, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-2000:]); sys.stderr.write(r.stderr[-2000:] if r.returncode else "")
    if r.returncode: raise SystemExit(f"failed: {args}")
    return r.stdout

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, required=True); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--partitions", default="k01"); ap.add_argument("--work", required=True); ap.add_argument("--budget", type=float, default=150)
    a = ap.parse_args(); t0 = time.time()
    cond = f"k{a.k:02d}_seed{a.seed}"; w = Path(a.work); d = w / cond; res = common.ROOT / "results" / cond
    if (res / "COMPLETE").exists(): print(cond, "COMPLETE (already)"); return
    left = lambda: a.budget - (time.time() - t0)
    if not (d / "index.csv").exists():
        run(["a02_features.py", "--k", str(a.k), "--seed", str(a.seed), "--work", str(w)])
    kd = d / "kernel_done.json"
    if not kd.exists() or len(json.load(open(kd))) < 465:
        if left() < 20: print("stop: time"); return
        out = run(["a03_kernel.py", "--cond", cond, "--work", str(w), "--budget", str(max(10, left() - 15))])
        if "partial" in out: return
    for p in a.partitions.split(","):
        fd = d / f"fit_{p}"
        if not (fd / "predictions.csv").exists():
            if left() < 30: print("stop: time"); return
            out = run(["a04_fit.py", "--cond", cond, "--work", str(w), "--partition", p, "--budget", str(max(5, left() - 40))])
            if "partial fit" in out: return
        if not (fd / "haufe_mean.npy").exists():
            if left() < 40: print("stop: time"); return
            run(["a05_haufe.py", "--cond", cond, "--work", str(w), "--partition", p])
    res.mkdir(parents=True, exist_ok=True)
    shutil.copy(d / "index.csv", res / "index.csv")
    for p in a.partitions.split(","):
        (res / f"fit_{p}").mkdir(exist_ok=True)
        for f in ("predictions.csv", "folds.json", "haufe_mean.npy", "haufe_folds.npy"):
            shutil.copy(d / f"fit_{p}" / f, res / f"fit_{p}" / f)
    for f in d.glob("sub-*.npy"): f.unlink()
    (d / "kernel.npy").unlink(missing_ok=True)
    (res / "COMPLETE").write_text(json.dumps({"partitions": a.partitions.split(","), "seconds_last_call": round(time.time() - t0)}))
    print(cond, "COMPLETE")

if __name__ == "__main__":
    main()
