"""
Shared style for the four journal figures (round-2 revision).

Every figure is drawn at its PRINTED size, so the font sizes set here are the
sizes the reader sees: the paper includes the PNGs at 0.75-0.8 of an MDPI
single-column \\linewidth (394.36 pt = 5.46 in). Drawing at 1:1 keeps axis
labels at 8.5 pt and tick labels at 8 pt on the page, instead of a 9-10 in
canvas shrunk to ~40 %.

Model colors are FIXED per model across all figures (Okabe-Ito, CVD-safe; run
through the dataviz validator: all-pairs normal-vision floor passes, the worst
CVD pair sits in the 6-8 floor band, so every model also carries a fixed marker
and bar hatch as secondary encoding). Text is always drawn in ink colors, never
in a series color.

Usage:
    from _style import apply_style, MODEL_STYLE, FIG_W_075, save
"""
import os

import matplotlib as mpl

# ---- Page geometry -----------------------------------------------------------
LINEWIDTH_IN = 394.35522 / 72.27          # MDPI \linewidth, inches (5.46)
FIG_W_075 = 0.75 * LINEWIDTH_IN           # figures included at 0.75\linewidth
FIG_W_080 = 0.80 * LINEWIDTH_IN           # fig_ncolor_ladder at 0.8\linewidth
DPI = 300

# ---- Type sizes (pt, at printed size) ---------------------------------------
FS_LABEL = 8.5      # axis labels
FS_TICK = 8.0       # tick labels
FS_ANNOT = 7.5      # value labels / callouts
FS_LEGEND = 7.5
FS_SMALL = 7.0      # secondary annotations (dataset names under ladder rungs)

# ---- Ink and surface ---------------------------------------------------------
INK_PRIMARY = "#1a1a1a"
INK_SECONDARY = "#4d4d4d"
INK_MUTED = "#8a8a8a"
GRID = "#e6e6e6"
SPINE = "#9a9a9a"
SURFACE = "#ffffff"

# ---- Fixed model identity: color, marker, bar hatch -------------------------
# Okabe-Ito hues. The Gaussian process is always blue, the least-squares
# polynomial always vermillion, and so on, in every figure.
MODEL_STYLE = {
    "gaussian_process": dict(label="Gaussian process", color="#0072B2", marker="o", hatch=""),
    "poly3": dict(label="Polynomial (3rd order)", color="#D55E00", marker="s", hatch="////"),
    "svm": dict(label="SVM", color="#009E73", marker="^", hatch="...."),
    "mlp_deep": dict(label="MLP (deep)", color="#CC79A7", marker="v", hatch="\\\\\\\\"),
    "random_forest": dict(label="Random forest", color="#E69F00", marker="X", hatch="xx"),
    # Powell-refined polynomial: the same model family as poly3, so the same
    # vermillion, told apart by a light fill with vermillion hatching.
    "poly3_de00_powell": dict(label=r"Polynomial (3rd order) + $\Delta E_{00}$ loss (Powell)",
                              color="#D55E00", marker="s", hatch="////", facecolor="#fbe3d4"),
    # Corrected classical baseline (degree 4, cube-root/CIELAB fit): yellow
    # diamonds with a dark edge (the caption refers to "yellow diamonds").
    "poly4_cbrt": dict(label="Polynomial (4th order, cube-root fit)", color="#F0E442", marker="D", hatch=""),
}

METRIC = r"CIEDE2000 $\Delta E_{00}$"


def apply_style() -> None:
    """Set the shared rcParams. Call once, before creating the figure."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": FS_TICK,
        "axes.labelsize": FS_LABEL,
        "axes.labelcolor": INK_PRIMARY,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": SURFACE,
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "xtick.color": SPINE,
        "ytick.color": SPINE,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "hatch.linewidth": 0.6,
        "legend.fontsize": FS_LEGEND,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3,
        "figure.facecolor": SURFACE,
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def save(fig, out_path: str) -> None:
    fig.savefig(out_path, dpi=DPI)
    print(f"Wrote: {os.path.relpath(out_path)}")
