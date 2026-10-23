#!/usr/bin/env python3
"""Figure 4C-D (2026-10-23; former Figure 5 v5, merged into Figure 4): ROI-level agreement between the SVM forward
pattern and CNN Layer-CAM. Panel letters and titles are added in LaTeX (TikZ), so none are drawn here.

Both panels use the same 112 Harvard-Oxford ROIs and the same rankings (regions_match_k6.csv; built by
averaging-granularity a23_regions_match.py and re-validated by roi_categories.py).
A  Axial slices: ROIs among the top 20 of both methods, of the SVM forward pattern only, or of Layer-CAM only.
B  Forward-pattern magnitude (k = 6, mean |z|) against Layer-CAM (mean of the two class maps, whole-brain
   mean = 1); dashed lines mark the 20th-ranked value on each axis, so the quadrants match the colours in A.

Usage: python make_fig4cd.py <roi_dir> <data_dir> <font_dir> <out_dir>
roi_dir: roi_top20_categories.nii.gz (from roi_categories.py); data_dir: regions_match_k6.csv/.json
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from nilearn import plotting, datasets

INK, INK2, REF = "#222222", "#555555", "#777777"
C_SVM, C_CNN, C_BOTH, C_NONE = "#1BAF7A", "#6250D6", "#EDA100", "#C4C4C4"   # dataviz validator (--pairs all): pass
CALLOUTS = {  # roi -> (label, text x, text y, ha)
    "Right Supracalcarine Cortex": ("R supracalcarine", 0.96, 1.91, "left"),
    "Right Occipital Pole": ("R occipital pole", 2.33, 1.91, "right"),
    "Right Parahippocampal Gyrus, posterior division": ("R posterior\nparahippocampal", 1.60, 1.13, "left"),
    "Left Occipital Pole": ("L occipital pole", 2.33, 0.86, "right"),
    "Right Right Accumbens": ("L/R accumbens", 0.20, 1.66, "left"),
    "Left Frontal Medial Cortex": ("L/R frontal\nmedial", 1.08, 0.70, "left"),
}
CUTS = [-14, -2, 10, 30]


def setup(font_dir):
    fm = font_manager.fontManager
    fm.ttflist = [e for e in fm.ttflist if e.name != "TeX Gyre Heros" or Path(e.fname).parent == Path(font_dir).resolve()]
    for f in Path(font_dir).glob("texgyreheros-*.ttf"):
        fm.addfont(str(f.resolve()))
    plt.rcParams.update({
        "font.family": "TeX Gyre Heros", "font.size": 7.2, "axes.labelsize": 7.2, "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8, "legend.fontsize": 6.8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": "#444444",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
        "mathtext.fontset": "custom", "mathtext.rm": "TeX Gyre Heros",
        "mathtext.it": "TeX Gyre Heros:italic", "mathtext.bf": "TeX Gyre Heros:bold"})


def letter(fig, x, y, s, title):
    fig.text(x, y, s, fontsize=8.5, fontweight="bold", ha="left", va="bottom")
    fig.text(x + 0.022, y, title, fontsize=7.5, fontweight="bold", ha="left", va="bottom")


def slices(fig, rect, roi_img):
    ax = fig.add_axes(rect); ax.set_axis_off()
    bg = datasets.load_mni152_template(resolution=1)
    disp = plotting.plot_roi(roi_img, bg_img=bg, display_mode="z", cut_coords=CUTS, axes=ax, figure=fig,
                             cmap=ListedColormap([C_SVM, C_CNN, C_BOTH]), vmin=0.5, vmax=3.5, alpha=0.95,
                             draw_cross=False, annotate=False, black_bg=False, dim=-0.35, colorbar=False)
    # slice labels and orientation in the body font
    fig.canvas.draw()
    for z, (axn, a) in zip(CUTS, sorted(disp.axes.items())):
        b = a.ax.get_position()
        fig.text((b.x0 + b.x1) / 2, b.y0 - 0.015, f"$z$ = {z}", ha="center", va="top", fontsize=6.6, color=INK2)
    fig.canvas.draw()
    b0 = disp.axes[sorted(disp.axes)[0]].ax.get_position()
    fig.text(b0.x0 + 0.004, b0.y1 - 0.01, "L", ha="left", va="top", fontsize=6.6, color=INK2)
    fig.text(b0.x1 - 0.004, b0.y1 - 0.01, "R", ha="right", va="top", fontsize=6.6, color=INK2)
    return disp


def scatter(ax, data):
    rm = list(csv.DictReader(open(Path(data) / "regions_match_k6.csv")))
    rj = json.load(open(Path(data) / "regions_match_k6.json"))
    x = np.array([float(r["absz"]) for r in rm]); y = np.array([float(r["lc_mean"]) for r in rm]); names = [r["roi"] for r in rm]
    rs = np.array([int(r["rank_svm"]) for r in rm]); rc = np.array([int(r["rank_cam"]) for r in rm])
    ts, tc = rs <= 20, rc <= 20
    xs20 = np.sort(x)[::-1][19]; yc20 = np.sort(y)[::-1][19]
    ax.axvline(xs20, color=REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhline(yc20, color=REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for m, col in ((~ts & ~tc, C_NONE), (ts & ~tc, C_SVM), (~ts & tc, C_CNN), (ts & tc, C_BOTH)):
        ax.scatter(x[m], y[m], s=13 if col == C_NONE else 18, c=col, edgecolors="white" if col == C_NONE else INK,
                   linewidths=0.4, zorder=2 if col == C_NONE else 3)
    for n, (lab, tx, ty, ha) in CALLOUTS.items():
        i = names.index(n)
        ax.annotate(lab, (x[i], y[i]), xytext=(tx, ty), fontsize=6.4, color=INK, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.45, color=INK2, shrinkA=0.5, shrinkB=2.6), zorder=5)
    ax.text(0.98, 0.03, f"Spearman $\\rho$ = {rj['spearman_magnitude']:.2f}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.8, color=INK)
    ax.text(xs20, 0.47, " top 20", ha="left", va="bottom", fontsize=6.2, color=INK2, rotation=90)
    ax.text(2.33, yc20, "top 20", ha="right", va="bottom", fontsize=6.2, color=INK2)
    ax.set_xlabel("SVM forward pattern (mean |$z$|)")
    ax.set_ylabel("CNN Layer-CAM (whole-brain = 1)")
    ax.set_xlim(0.15, 2.35); ax.set_ylim(0.45, 1.95); ax.set_xticks([0.5, 1.0, 1.5, 2.0])
    ax.spines[["top", "right"]].set_visible(False)
    return rj, int((ts & tc).sum()), int((ts & ~tc).sum()), int((~ts & tc).sum())


def main(roi_dir, data, fonts, out):
    setup(fonts); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    roi = nib.load(str(Path(roi_dir) / "roi_top20_categories.nii.gz"))
    fig = plt.figure(figsize=(7.0, 2.35))
    slices(fig, [0.0, 0.2, 0.585, 0.63], roi)
    b = fig.add_axes([0.685, 0.17, 0.30, 0.70])
    rj, nb, ns, nc = scatter(b, data)
    fig.legend(handles=[Line2D([], [], ls="", marker="s", ms=6, mfc=C_BOTH, mec=INK, mew=0.4, label=f"Both methods ({nb})"),
                        Line2D([], [], ls="", marker="s", ms=6, mfc=C_SVM, mec=INK, mew=0.4, label=f"SVM forward pattern only ({ns})"),
                        Line2D([], [], ls="", marker="s", ms=6, mfc=C_CNN, mec=INK, mew=0.4, label=f"CNN Layer-CAM only ({nc})"),
                        Line2D([], [], ls="", marker="o", ms=4, mfc=C_NONE, mec="white", mew=0.4, label="Other ROIs")],
               loc="lower left", bbox_to_anchor=(0.01, -0.01), ncol=4, frameon=False, handlelength=0.9,
               handletextpad=0.35, columnspacing=1.2, fontsize=6.7)
    fig.savefig(out / "fig_agreement.pdf", facecolor="white")
    fig.savefig(out / "fig_agreement.png", facecolor="white", dpi=250)


if __name__ == "__main__":
    main(*sys.argv[1:5])
