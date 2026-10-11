#!/usr/bin/env python3
"""
Figure (round-3 revision, replaces the n-colour ladder as Figure 4): median and
95th-percentile CIEDE2000 error per model, one panel per dataset.

Each panel is one dataset and printing system. No line joins different
datasets, and every panel shows the same seven models in the same row order:

    Gaussian process, XYZ             (pre-specified)
    Gaussian process, cube root       (post hoc)
    Polynomial, 3rd order, XYZ        (pre-specified)
    Polynomial, 4th order, cube root  (post hoc)
    SVM, MLP (deep), random forest, XYZ (pre-specified)

Encoding: the marker is the median, the thin bar runs from the median to the
95th percentile (P95), closed by a short tick. Filled markers are models fitted
on XYZ (the pre-specified comparison); hollow markers are fitted in cube-root
space, added after the XYZ results had been obtained (post hoc). Color is the
model family (Gaussian process blue, polynomial vermillion, as in _style);
marker shape tells the polynomial degrees apart.

All panels share one log-scaled x axis, so equal distances mean equal error
ratios in every panel. PC10 represents the three coated printers at n = 3 and
n = 4, as in the text; PC11 and FOGRA51 are in the tables.

Source (never hand-entered): journal/results/{PC10-CMY,PC10-CMYK,KCMYG-5,
CMYKOGV-7,CMYKOGB-7}/summary.csv
Run with: .venv/bin/python journal/figures/fig_per_dataset.py
Writes: journal/figures/fig_per_dataset.png (300 dpi) and .pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import pandas as pd

from _style import (FS_LABEL, FS_LEGEND, GRID, INK_MUTED, INK_PRIMARY,
                    INK_SECONDARY, LINEWIDTH_IN, METRIC, MODEL_STYLE, SURFACE,
                    apply_style, save)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
OUT_PNG = os.path.join(HERE, "fig_per_dataset.png")
OUT_PDF = os.path.join(HERE, "fig_per_dataset.pdf")

# (results dir, panel title)
DATASETS = [
    ("PC10-CMY", "PC10-CMY ($n$ = 3)"),
    ("PC10-CMYK", "PC10-CMYK ($n$ = 4)"),
    ("KCMYG-5", "KCMYG-5 ($n$ = 5)"),
    ("CMYKOGV-7", "CMYKOGV-7 ($n$ = 7)"),
    ("CMYKOGB-7", "CMYKOGB-7 ($n$ = 7)"),
]

BLUE = MODEL_STYLE["gaussian_process"]["color"]
VERMILLION = MODEL_STYLE["poly3"]["color"]

# key, row label, color, marker, hollow (= post-hoc cube-root fit)
MODELS = [
    ("gaussian_process", "Gaussian process", BLUE, "o", False),
    ("gaussian_process_cbrt", "Gaussian process", BLUE, "o", True),
    ("poly3", "Polynomial, 3rd order", VERMILLION, "s", False),
    ("poly4_cbrt", "Polynomial, 4th order", VERMILLION, "D", True),
    ("svm", "SVM", MODEL_STYLE["svm"]["color"], MODEL_STYLE["svm"]["marker"], False),
    ("mlp_deep", "MLP (deep)", MODEL_STYLE["mlp_deep"]["color"],
     MODEL_STYLE["mlp_deep"]["marker"], False),
    ("random_forest", "Random forest", MODEL_STYLE["random_forest"]["color"],
     MODEL_STYLE["random_forest"]["marker"], False),
]

X_LIM = (0.03, 40)


def load() -> dict[str, pd.DataFrame]:
    """Return {dataset: DataFrame[model -> median, p95]} from the summary CSVs."""
    out = {}
    keys = [k for k, *_ in MODELS]
    for ds, _title in DATASETS:
        s = pd.read_csv(os.path.join(RESULTS_DIR, ds, "summary.csv")).set_index("model")
        missing = [k for k in keys if k not in s.index]
        if missing:
            raise SystemExit(f"{ds}: models missing from summary.csv: {missing}")
        sub = s.loc[keys, ["median", "p95"]]
        lo, hi = sub.min().min(), sub.max().max()
        if lo < X_LIM[0] or hi > X_LIM[1]:
            raise SystemExit(f"{ds}: values {lo}-{hi} fall outside the axis {X_LIM}")
        out[ds] = sub
    return out


def row_label(label: str, hollow: bool) -> str:
    return f"{label}, cube root" if hollow else f"{label}, XYZ"


def draw_panel(ax, df: pd.DataFrame, title: str, show_labels: bool) -> None:
    n = len(MODELS)
    ys = list(range(n))[::-1]          # first model on top
    ax.grid(axis="x", which="major", color=GRID)
    for y, (key, _label, color, marker, hollow) in zip(ys, MODELS):
        med, p95 = df.loc[key, "median"], df.loc[key, "p95"]
        ax.plot([med, p95], [y, y], color=color, linewidth=1.0, solid_capstyle="butt",
                zorder=2)
        ax.plot([p95], [y], marker="|", markersize=6, markeredgewidth=1.1,
                color=color, linestyle="none", zorder=2)
        ax.plot([med], [y], marker=marker, markersize=5.2 if marker != "D" else 4.6,
                markerfacecolor=SURFACE if hollow else color,
                markeredgecolor=color, markeredgewidth=1.2 if hollow else 0.6,
                linestyle="none", zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(*X_LIM)
    ax.xaxis.set_major_locator(mticker.FixedLocator([0.1, 1, 10]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(ys)
    if show_labels:
        ax.set_yticklabels([row_label(lab, h) for _k, lab, _c, _m, h in MODELS],
                           color=INK_PRIMARY)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(title, fontsize=FS_LABEL, color=INK_PRIMARY, loc="left", pad=3)


def make_figure(data: dict[str, pd.DataFrame]) -> None:
    apply_style()
    fig, axes = plt.subplots(3, 2, figsize=(LINEWIDTH_IN, 5.6), sharex=True)
    flat = axes.ravel()
    for i, (ds, title) in enumerate(DATASETS):
        draw_panel(flat[i], data[ds], title, show_labels=(i % 2 == 0))
    for ax in flat[:5]:
        ax.xaxis.set_tick_params(labelbottom=True)
    for ax in (flat[4], flat[3]):
        ax.set_xlabel(f"{METRIC} (log scale)")

    # Sixth slot: the legend (encoding only; model names are on the rows).
    leg_ax = flat[5]
    leg_ax.axis("off")
    handles = [
        Line2D([], [], color=INK_SECONDARY, marker="o", markersize=5.2,
               markerfacecolor=INK_SECONDARY, markeredgewidth=0.6, linestyle="none",
               label="Filled: XYZ fit (pre-specified)"),
        Line2D([], [], color=INK_SECONDARY, marker="o", markersize=5.2,
               markerfacecolor=SURFACE, markeredgewidth=1.2, linestyle="none",
               label="Hollow: cube-root fit (post hoc)"),
    ]
    leg_ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.92),
                  fontsize=FS_LEGEND, labelcolor=INK_PRIMARY, handlelength=1.2,
                  labelspacing=0.7, borderaxespad=0)
    # Key glyph: one example row, labeled, in axes coordinates.
    gy, gx0, gx1 = 0.30, 0.12, 0.55
    leg_ax.plot([gx0, gx1], [gy, gy], color=INK_SECONDARY, linewidth=1.0,
                transform=leg_ax.transAxes, clip_on=False)
    leg_ax.plot([gx1], [gy], marker="|", markersize=6, markeredgewidth=1.1,
                color=INK_SECONDARY, transform=leg_ax.transAxes, clip_on=False)
    leg_ax.plot([gx0], [gy], marker="o", markersize=5.2, color=INK_SECONDARY,
                markeredgewidth=0.6, transform=leg_ax.transAxes, clip_on=False)
    for x, text in ((gx0, "median"), (gx1, "95th percentile")):
        leg_ax.text(x, gy - 0.08, text, transform=leg_ax.transAxes, ha="center",
                    va="top", fontsize=FS_LEGEND, color=INK_PRIMARY)

    fig.tight_layout(pad=0.3, h_pad=0.9, w_pad=0.6)
    save(fig, OUT_PNG)
    save(fig, OUT_PDF)
    plt.close(fig)


def main() -> None:
    data = load()
    print("Plotted (median, P95):")
    for ds, _t in DATASETS:
        print(f"== {ds}")
        print(data[ds].to_string(float_format=lambda v: f"{v:.3f}"))
    make_figure(data)


if __name__ == "__main__":
    main()
