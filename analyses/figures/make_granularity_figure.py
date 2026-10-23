#!/usr/bin/env python3
"""Figure for the averaging-granularity analyses (b plan, 2026-10-20), in the ccn-v004 figure style.

(A) Linear-SVM cross-participant BA as a function of k (fixed k=1 partition; k>1 averaged over grouping
    seeds 42-46): grey lines = participants, black = participant mean with bootstrap 95% CI.
(B) Cross-granularity transfer: SVM trained and tested at k (matched), trained on single trials and tested
    at k, trained at k and tested on single trials.
(C) Between-participant SD of BA (seed 42) against the SD expected from binomial sampling of each
    participant's own held-out inputs.
Usage: python make_granularity_figure.py --summary <results_summary dir> --out <dir> --font-dir <ttf dir>
"""
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

INK, INK2, GRID, REF = "#222222", "#555555", "#BDBDBD", "#777777"
KS = [1, 2, 3, 4, 5, 6]


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def setup(font_dir):
    fm = font_manager.fontManager
    fm.ttflist = [e for e in fm.ttflist if e.name != "TeX Gyre Heros" or Path(e.fname).parent == Path(font_dir)]
    for f in Path(font_dir).glob("texgyreheros-*.ttf"):
        fm.addfont(str(f))
    plt.rcParams.update({
        "font.family": "TeX Gyre Heros", "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8, "legend.fontsize": 6.6, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": "#444444",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "pdf.fonttype": 42, "savefig.dpi": 300, "mathtext.fontset": "custom", "mathtext.rm": "TeX Gyre Heros",
        "mathtext.it": "TeX Gyre Heros:italic"})


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def letter(fig, ax, s, title, dx=-0.055):
    b = ax.get_position()
    fig.text(b.x0 + dx, b.y1 + 0.05, s, fontsize=9, fontweight="bold", ha="left", va="bottom")
    fig.text(b.x0 + dx + 0.025, b.y1 + 0.05, title, fontsize=8, fontweight="bold", ha="left", va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True); ap.add_argument("--out", required=True); ap.add_argument("--font-dir", required=True)
    a = ap.parse_args(); S = Path(a.summary); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    setup(a.font_dir)
    curve = {int(r["k"]): r for r in read(S / "svm_k_curve.csv")}
    part = read(S / "svm_participant_ba.csv")
    cross = read(S / "cross_granularity_curves.csv")
    fig = plt.figure(figsize=(7.0, 2.45))
    axA = fig.add_axes([0.065, 0.17, 0.26, 0.66])
    axB = fig.add_axes([0.405, 0.17, 0.26, 0.66])
    axC = fig.add_axes([0.745, 0.17, 0.235, 0.66])
    x = np.arange(len(KS))
    # A
    for r in part:
        axA.plot(x, [100 * float(r[f"k{k}"]) for k in KS], color=GRID, lw=0.55, zorder=1)
    m = [float(curve[k]["mean_ba"]) for k in KS]; lo = [float(curve[k]["ci_low"]) for k in KS]; hi = [float(curve[k]["ci_high"]) for k in KS]
    axA.errorbar(x, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], fmt="o-", color=INK, ms=4, lw=1.1, capsize=2.2, elinewidth=0.9, zorder=3)
    axA.axhline(50, color=REF, lw=0.8, ls=(0, (3, 2)), zorder=0)
    axA.set_xticks(x, [str(k) for k in KS]); axA.set_xlim(-0.3, 5.3); axA.set_ylim(30, 100); axA.set_yticks([30, 50, 70, 90])
    axA.set_xlabel("Trials averaged per input ($k$)"); axA.set_ylabel("Balanced accuracy (%)"); clean(axA)
    # B
    styles = {"matched": dict(color=INK, ls="-", marker="o", mfc=INK, label="Train $k$, test $k$"),
              "train1_testk": dict(color=INK, ls=(0, (3, 1.6)), marker="o", mfc="white", label="Train 1, test $k$"),
              "traink_test1": dict(color="#8C8C8C", ls="-", marker="s", mfc="#8C8C8C", label="Train $k$, test 1")}
    off = {"matched": -0.12, "train1_testk": 0.0, "traink_test1": 0.12}
    for name, st in styles.items():
        rr = sorted([r for r in cross if r["condition"] == name], key=lambda r: int(r["k"]))
        mm = np.array([float(r["mean"]) for r in rr]); l = np.array([float(r["ci_low"]) for r in rr]); h = np.array([float(r["ci_high"]) for r in rr])
        axB.errorbar(x + off[name], mm, yerr=[mm - l, h - mm], color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.6,
                     mfc=st["mfc"], mec=st["color"], lw=1.0, capsize=1.8, elinewidth=0.8, label=st["label"], zorder=3)
    axB.axhline(50, color=REF, lw=0.8, ls=(0, (3, 2)), zorder=0)
    axB.set_xticks(x, [str(k) for k in KS]); axB.set_xlim(-0.4, 5.4); axB.set_ylim(50, 75); axB.set_yticks([50, 55, 60, 65, 70])
    axB.set_xlabel("Trials averaged per input ($k$)"); axB.set_ylabel("Balanced accuracy (%)"); clean(axB)
    axB.legend(loc="upper left", frameon=False, handlelength=2.2, borderaxespad=0.1)
    # C
    obs = [float(curve[k]["sd_seed42"]) for k in KS]; smp = [float(curve[k]["expected_sampling_sd_seed42"]) for k in KS]
    axC.plot(x, obs, "o-", color=INK, ms=3.8, lw=1.1, label="Observed")
    axC.plot(x, smp, "o", ls=(0, (3, 1.6)), color="#8C8C8C", mfc="white", ms=3.8, lw=1.0, label="Expected from sampling")
    axC.set_xticks(x, [str(k) for k in KS]); axC.set_xlim(-0.3, 5.3); axC.set_ylim(0, 18); axC.set_yticks([0, 5, 10, 15])
    axC.set_xlabel("Trials averaged per input ($k$)"); axC.set_ylabel("Between-participant SD (pp)"); clean(axC)
    axC.legend(loc="upper left", frameon=False, handlelength=2.2, borderaxespad=0.1)
    letter(fig, axA, "A", "Linear SVM across $k$")
    letter(fig, axB, "B", "Cross-granularity transfer")
    letter(fig, axC, "C", "Dispersion across participants")
    fig.savefig(out / "fig_granularity.pdf", facecolor="white"); fig.savefig(out / "fig_granularity.png", facecolor="white", dpi=220)
    print("saved", out / "fig_granularity.pdf")


if __name__ == "__main__":
    main()
