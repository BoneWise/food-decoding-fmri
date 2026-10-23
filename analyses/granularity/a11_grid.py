"""a11: run a10 over the condition grid until the time budget is used; re-run until it prints GRID COMPLETE.
Grid: k = 2..6 x grouping seeds 42-46 on the fixed k=1 participant partition (k01); for seed 42 also the
CNN partitions of k=4, 5, 6 (k=2 and k=3 CNN partitions equal k01)."""
import subprocess, sys, time
from pathlib import Path
import common

BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 150
work = sys.argv[1]; t0 = time.time()
grid = []
for seed in (42, 43, 44, 45, 46):
    for k in (2, 3, 4, 5, 6):
        parts = "k01" + (f",k{k:02d}" if seed == 42 and k in (4, 5, 6) else "")
        grid.append((k, seed, parts))
for k, seed, parts in grid:
    if (common.ROOT / "results" / f"k{k:02d}_seed{seed}" / "COMPLETE").exists(): continue
    left = BUDGET - (time.time() - t0)
    if left < 45: print("budget used"); break
    r = subprocess.run([sys.executable, "a10_run.py", "--k", str(k), "--seed", str(seed), "--partitions", parts,
                        "--work", work, "--budget", str(left)], cwd=Path(__file__).parent, capture_output=True, text=True)
    print(f"k={k} seed={seed}:", (r.stdout.strip().splitlines() or [""])[-1], r.stderr[-300:] if r.returncode else "", flush=True)
    if "COMPLETE" not in r.stdout: break
else:
    print("GRID COMPLETE")
