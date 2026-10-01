#!/usr/bin/env python3
"""
Figure: the n-colour ladder — median CIEDE2000 error vs number of inks (n = 3, 4, 5, 7).

Line chart with one trajectory per model. poly3 and gaussian_process are the
highlighted trajectories (the paper's message: poly3 degrades sharply as inks
grow while GP holds); svm, mlp_deep, random_forest are muted context lines.
Linear-family models are omitted (off the chart; range noted in the footnote).

HONESTY: each rung of the ladder is an independent dataset / printing system
(PC10 CMY, PC10 CMYK, KCMYG, CMYKOGV, CMYKOGB), not a controlled sweep of one
printer. The within-model trend across systems is the message — stated in the
on-plot subtitle, never only here.

Source (never hand-entered): journal/results/{PC10-CMY,PC10-CMYK,KCMYG-5,
CMYKOGV-7,CMYKOGB-7}/summary.csv
Run with: .venv/bin/python journal/figures/fig_ncolor_ladder.py
Writes: journal/figures/fig_ncolor_ladder.png
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
OUT_PATH = os.path.join(HERE, "fig_ncolor_ladder.png")

# Ladder rungs: (dataset dir, x position). The two 7-ink systems are offset
# around n=7 so both points are visible; they are independent systems, not
# repeats of one printer.
X_OGV = 6.85
X_OGB = 7.15
RUNGS = [
    ("PC10-CMY", 3.0),
    ("PC10-CMYK", 4.0),
    ("KCMYG-5", 5.0),
    ("CMYKOGV-7", X_OGV),
    ("CMYKOGB-7", X_OGB),
]

from _style import (FIG_W_080, FS_ANNOT, FS_LABEL, FS_LEGEND, FS_SMALL, FS_TICK,
                    GRID, INK_MUTED, INK_PRIMARY, INK_SECONDARY, METRIC, MODEL_STYLE,
                    SURFACE, apply_style, save)

# Curated model subset. Highlighted trajectories carry the story; context
# lines show the subset isn't cherry-picked. Colors and markers are the fixed
# per-model identities from _style (same model, same color, every figure).
MODELS = [
    # key, highlighted?
    ("gaussian_process", True),
    ("poly3", True),
    ("svm", False),
    ("mlp_deep", False),
    ("random_forest", False),
]
MODELS = [(k, MODEL_STYLE[k]["label"], MODEL_STYLE[k]["color"], hl) for k, hl in MODELS]
LINEAR_FAMILY = ["ridge", "lasso", "elastic", "pcr", "plsr"]
# Revision R1 (Reviewer 2, minor 2): discrete markers for the CORRECTED classical
# baseline (degree 4, fitted in cube-root/CIELAB space), so the degree-3 XYZ
# curve cannot be misread as the best the polynomial can do. Markers only, no
# line: it is a reference, not one of the uncorrected trajectories.
CORRECTED = "poly4_cbrt"
DIAMOND_FACE = MODEL_STYLE[CORRECTED]["color"]

# Text behind which a curve may pass gets a small surface-colored backing.
LABEL_BOX = dict(boxstyle="square,pad=0.08", facecolor=SURFACE, edgecolor="none", alpha=0.85)


def load_medians() -> tuple[pd.DataFrame, float, float]:
    """Return (medians[model x dataset], linear-family min, max) from summary CSVs."""
    cols = {}
    linear_vals = []
    for dataset, _x in RUNGS:
        s = pd.read_csv(os.path.join(RESULTS_DIR, dataset, "summary.csv")).set_index("model")["median"]
        cols[dataset] = s
        linear_vals.extend(s.reindex(LINEAR_FAMILY).dropna().tolist())
    df = pd.DataFrame(cols)
    missing = [m for m, *_ in MODELS if m not in df.index] + ([] if CORRECTED in df.index else [CORRECTED])
    if missing:
        raise SystemExit(f"models missing from summaries: {missing}")
    return df, min(linear_vals), max(linear_vals)


def spread_labels(ys: list[float], min_gap_log: float = 0.07) -> list[float]:
    """Nudge label y-positions apart in log space so end labels never collide."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    logs = [math.log10(ys[i]) for i in order]
    for k in range(1, len(logs)):
        if logs[k] - logs[k - 1] < min_gap_log:
            logs[k] = logs[k - 1] + min_gap_log
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = 10 ** logs[k]
    return out


def make_figure(df: pd.DataFrame, lin_lo: float, lin_hi: float, out_path: str) -> None:
    xs = [x for _, x in RUNGS]
    datasets = [d for d, _ in RUNGS]

    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_W_080, 3.25))

    # Recessive hairline grid, y only (log decades).
    ax.grid(axis="y", which="major", color=GRID)

    for key, name, color, highlighted in MODELS:
        vals = [df.loc[key, d] for d in datasets]
        marker = MODEL_STYLE[key]["marker"]
        lw = 1.8 if highlighted else 1.0
        alpha = 1.0 if highlighted else 0.75
        ms = 5.0 if highlighted else 3.8
        z = 4 if highlighted else 3
        # Solid trajectory through n = 3, 4, 5 and on to the OGV 7-ink system...
        ax.plot(xs[:4], vals[:4], color=color, linewidth=lw, alpha=alpha,
                solid_capstyle="round", solid_joinstyle="round", zorder=z)
        # ...and a dashed branch from n = 5 to the second, independent 7-ink system.
        ax.plot([xs[2], xs[4]], [vals[2], vals[4]], color=color, linewidth=lw,
                alpha=alpha, linestyle=(0, (3.5, 2.5)), zorder=z)
        # Markers: filled everywhere; the OGB system open (surface-filled) so the
        # two 7-ink points stay tellable-apart beyond the x-offset.
        ax.plot(xs[:4], vals[:4], marker, color=color, markersize=ms, alpha=alpha,
                markeredgecolor=SURFACE, markeredgewidth=0.7, linestyle="none", zorder=z + 0.5)
        ax.plot([xs[4]], [vals[4]], marker, markerfacecolor=SURFACE, markersize=ms,
                alpha=alpha, markeredgecolor=color, markeredgewidth=1.1,
                linestyle="none", zorder=z + 0.5)

    # Corrected polynomial: discrete diamond markers at every rung,
    # offset slightly right of each rung so a diamond never hides a curve's dot.
    corr = [df.loc[CORRECTED, d] for d in datasets]
    xd = [x + 0.17 for x in xs]
    ax.plot(xd, corr, "D", color=INK_PRIMARY, markersize=4.8, markerfacecolor=DIAMOND_FACE,
            markeredgewidth=0.8, linestyle="none", zorder=6)
    # Value label beside each diamond (not above/below, where curve labels sit);
    # on the last rung it goes below, clear of the end labels.
    for i, (x, v) in enumerate(zip(xd, corr)):
        last = i == len(xd) - 1
        ax.annotate(f"{v:.2f}", xy=(x, v),
                    xytext=(0, -6) if last else (5, 0), textcoords="offset points",
                    va="top" if last else "center", ha="center" if last else "left",
                    fontsize=FS_ANNOT - 0.5, color=INK_PRIMARY, bbox=LABEL_BOX, zorder=7)

    # Direct end labels for every series (relief for sub-3:1 hues), spread to
    # avoid collisions, in text ink with the colored line as the identity mark.
    # Highlighted labels carry their CMYKOGB endpoint value inline.
    end_vals = [df.loc[key, "CMYKOGB-7"] for key, *_ in MODELS]
    # The GP label is anchored a little above its endpoint so its leader line
    # slopes up, away from the corrected-baseline diamond just below it
    # (label position only; the plotted values are untouched).
    anchor_ys = [v * (1.12 if key == "gaussian_process" else 1.0)
                 for (key, *_), v in zip(MODELS, end_vals)]
    label_ys = spread_labels(anchor_ys, min_gap_log=0.095)
    for (key, name, color, highlighted), y_end, y_lab in zip(MODELS, end_vals, label_ys):
        text = f"{name}  {y_end:.2f}" if highlighted else name
        ax.annotate(
            text,
            xy=(X_OGB, y_end), xytext=(X_OGB + 0.5, y_lab),
            va="center", ha="left",
            fontsize=FS_ANNOT,
            fontweight="bold" if highlighted else "normal",
            color=INK_PRIMARY if highlighted else INK_SECONDARY,
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.5,
                            shrinkA=1.5, shrinkB=3.5),
            zorder=5,
        )

    # Selective value callouts on the two highlighted trajectories only.
    for key in ("gaussian_process", "poly3"):
        v3 = df.loc[key, "PC10-CMY"]
        ax.annotate(f"{v3:.2f}", xy=(3.0, v3), xytext=(-6, 0),
                    textcoords="offset points", va="center", ha="right",
                    fontsize=FS_ANNOT - 0.5, color=INK_SECONDARY, zorder=5)
        v_ogv = df.loc[key, "CMYKOGV-7"]
        # GP's CMYKOGV label goes below its dot, clear of the adjacent diamond.
        below = key == "gaussian_process"
        ax.annotate(f"{v_ogv:.2f}", xy=(X_OGV, v_ogv), xytext=(-2, -6 if below else 6),
                    textcoords="offset points", va="top" if below else "bottom",
                    ha="center", fontsize=FS_ANNOT - 0.5, color=INK_SECONDARY, zorder=5)

    # Axes.
    ax.set_yscale("log")
    ax.set_ylim(0.03, 12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel(f"Median {METRIC} (log scale)")

    ax.set_xlim(2.7, 10.0)
    ax.set_xticks([3, 4, 5, 7])
    ax.set_xticklabels(["3", "4", "5", "7"], color=INK_PRIMARY)
    ax.set_xlabel("Number of inks $n$", labelpad=13)
    ax.spines["bottom"].set_bounds(2.7, 7.6)
    # Dataset name under each rung, as named in the paper: the honesty channel
    # on the axis itself.
    for x, label, ha, dx in [(3, "PC10", "center", 0), (4, "PC10", "center", 0),
                             (5, "KCMYG-5", "center", 0),
                             (X_OGV, "CMYKOGV-7", "right", -1), (X_OGB, "CMYKOGB-7", "left", 1)]:
        ax.annotate(label, xy=(x, 0), xycoords=("data", "axes fraction"),
                    xytext=(dx, -13), textcoords="offset points",
                    ha=ha, va="top", fontsize=FS_SMALL, color=INK_MUTED)

    # Legend: the two 7-ink line/marker styles and the corrected-baseline
    # diamond (series identity is direct-labeled at the line ends).
    legend_handles = [
        Line2D([], [], color=INK_SECONDARY, marker="o", markersize=4,
               markeredgecolor=SURFACE, markeredgewidth=0.7, linestyle="-",
               linewidth=1.2, label="solid, filled: $n$ = 3, 4, 5 and CMYKOGV-7"),
        Line2D([], [], color=INK_SECONDARY, marker="o", markersize=4,
               markerfacecolor=SURFACE, markeredgecolor=INK_SECONDARY,
               markeredgewidth=1.0, linestyle=(0, (3.5, 2.5)), linewidth=1.2,
               label="dashed, open: CMYKOGB-7 (second 7-ink set)"),
        Line2D([], [], color=INK_PRIMARY, marker="D", markersize=4.8,
               markerfacecolor=DIAMOND_FACE, markeredgewidth=0.8, linestyle="none",
               label="corrected polynomial (4th order, CIELAB fit)"),
    ]
    # Lower right is empty of data (below every curve at n >= 5).
    ax.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(1.0, 0.0),
              fontsize=FS_LEGEND - 0.5, labelcolor=INK_SECONDARY, handlelength=2.4,
              labelspacing=0.3)

    # No in-figure title, subtitle or footnote: the independent-rungs caveat and
    # the linear-family omission are stated in the paper's caption, and the old
    # title asserted a reading ("the Gaussian process holds") that Section 4.7's
    # corrected comparison does not support.
    fig.tight_layout(pad=0.3)
    save(fig, out_path)
    plt.close(fig)


def main() -> None:
    df, lin_lo, lin_hi = load_medians()
    sub = df.loc[[m for m, *_ in MODELS]]
    print("Medians plotted:")
    print(sub.round(3).to_string())
    print(f"Linear-family median range (omitted): {lin_lo:.3f}-{lin_hi:.3f}")
    make_figure(df, lin_lo, lin_hi, OUT_PATH)


if __name__ == "__main__":
    main()
