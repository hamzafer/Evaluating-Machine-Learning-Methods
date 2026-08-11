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

# Curated model subset. Highlighted trajectories carry the story; context
# lines show the subset isn't cherry-picked. Categorical hues in fixed order
# (validated: CVD-safe adjacent pairs; sub-3:1-contrast hues get direct labels).
MODELS = [
    # key, display name, color, highlighted?
    ("gaussian_process", "Gaussian Process", "#2a78d6", True),
    ("poly3", "Polynomial (3rd order)", "#eb6834", True),
    ("svm", "SVM", "#1baf7a", False),
    ("mlp_deep", "MLP (deep)", "#4a3aa7", False),
    ("random_forest", "Random Forest", "#e87ba4", False),
]
LINEAR_FAMILY = ["ridge", "lasso", "elastic", "pcr", "plsr"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_medians() -> tuple[pd.DataFrame, float, float]:
    """Return (medians[model x dataset], linear-family min, max) from summary CSVs."""
    cols = {}
    linear_vals = []
    for dataset, _x in RUNGS:
        s = pd.read_csv(os.path.join(RESULTS_DIR, dataset, "summary.csv")).set_index("model")["median"]
        cols[dataset] = s
        linear_vals.extend(s.reindex(LINEAR_FAMILY).dropna().tolist())
    df = pd.DataFrame(cols)
    missing = [m for m, *_ in MODELS if m not in df.index]
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

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Recessive hairline grid, y only (log decades + halves).
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    for key, name, color, highlighted in MODELS:
        vals = [df.loc[key, d] for d in datasets]
        lw = 2.4 if highlighted else 1.6
        alpha = 1.0 if highlighted else 0.55
        ms = 8 if highlighted else 6
        # Solid trajectory through n = 3, 4, 5 and on to the OGV 7-ink system...
        ax.plot(xs[:4], vals[:4], color=color, linewidth=lw, alpha=alpha,
                solid_capstyle="round", solid_joinstyle="round", zorder=3)
        # ...and a dashed branch from n = 5 to the second, independent 7-ink system.
        ax.plot([xs[2], xs[4]], [vals[2], vals[4]], color=color, linewidth=lw,
                alpha=alpha, linestyle=(0, (4, 3)), zorder=3)
        # Markers: filled everywhere; the OGB system open (surface-filled) so the
        # two 7-ink points stay tellable-apart beyond the x-offset.
        ax.plot(xs[:4], vals[:4], "o", color=color, markersize=ms, alpha=alpha,
                markeredgecolor=SURFACE, markeredgewidth=1.4, linestyle="none", zorder=4)
        ax.plot([xs[4]], [vals[4]], "o", markerfacecolor=SURFACE, markersize=ms,
                alpha=alpha, markeredgecolor=color, markeredgewidth=1.8,
                linestyle="none", zorder=4)

    # Direct end labels for every series (relief for sub-3:1 hues), spread to
    # avoid collisions, in text ink with the colored line as the identity mark.
    # Highlighted labels carry their CMYKOGB endpoint value inline.
    end_vals = [df.loc[key, "CMYKOGB-7"] for key, *_ in MODELS]
    label_ys = spread_labels(end_vals)
    for (key, name, color, highlighted), y_end, y_lab in zip(MODELS, end_vals, label_ys):
        text = f"{name}  {y_end:.2f}" if highlighted else name
        ax.annotate(
            text,
            xy=(X_OGB, y_end), xytext=(X_OGB + 0.28, y_lab),
            va="center", ha="left",
            fontsize=9.5 if highlighted else 8.5,
            fontweight="bold" if highlighted else "normal",
            color=INK_PRIMARY if highlighted else INK_SECONDARY,
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.7,
                            shrinkA=2, shrinkB=4),
            zorder=5,
        )

    # Selective value callouts on the two highlighted trajectories only.
    for key in ("gaussian_process", "poly3"):
        v3 = df.loc[key, "PC10-CMY"]
        ax.annotate(f"{v3:.2f}", xy=(3.0, v3), xytext=(-10, 0),
                    textcoords="offset points", va="center", ha="right",
                    fontsize=8.5, color=INK_SECONDARY, zorder=5)
        v_ogv = df.loc[key, "CMYKOGV-7"]
        ax.annotate(f"{v_ogv:.2f}", xy=(X_OGV, v_ogv), xytext=(0, 9),
                    textcoords="offset points", va="bottom", ha="center",
                    fontsize=8.5, color=INK_SECONDARY, zorder=5)

    # Axes.
    ax.set_yscale("log")
    ax.set_ylim(0.03, 12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel(r"Median $\Delta E_{00}$ (log scale)", fontsize=10.5, color=INK_PRIMARY)

    ax.set_xlim(2.55, 9.45)
    ax.set_xticks([3, 4, 5, 7])
    ax.set_xticklabels(["3", "4", "5", "7"], fontsize=11, color=INK_PRIMARY)
    ax.set_xlabel("Number of inks  $n$", fontsize=10.5, color=INK_PRIMARY, labelpad=20)
    # Dataset name under each rung — the honesty channel on the axis itself.
    for x, label, ha, dx in [(3, "PC10 CMY", "center", 0), (4, "PC10 CMYK", "center", 0),
                             (5, "KCMYG", "center", 0),
                             (X_OGV, "CMYKOGV", "right", -4), (X_OGB, "CMYKOGB", "left", 4)]:
        ax.annotate(label, xy=(x, 0), xycoords=("data", "axes fraction"),
                    xytext=(dx, -18), textcoords="offset points",
                    ha=ha, va="top", fontsize=7.5, color=INK_MUTED)

    ax.tick_params(axis="both", colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
        ax.spines[spine].set_linewidth(0.8)

    # Legend: the two 7-ink marker styles (series identity is direct-labeled).
    legend_handles = [
        Line2D([], [], color=INK_SECONDARY, marker="o", markersize=7,
               markeredgecolor=SURFACE, markeredgewidth=1.2, linestyle="-",
               linewidth=1.6, label="solid, filled: single system per rung (7 = CMYKOGV)"),
        Line2D([], [], color=INK_SECONDARY, marker="o", markersize=7,
               markerfacecolor=SURFACE, markeredgecolor=INK_SECONDARY,
               markeredgewidth=1.5, linestyle=(0, (4, 3)), linewidth=1.6,
               label="dashed, open: second 7-ink system (CMYKOGB)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False,
              fontsize=8, labelcolor=INK_SECONDARY, handlelength=2.6)

    # Title + honesty subtitle.
    fig.text(0.055, 0.965,
             "Polynomial regression degrades as inks grow; the Gaussian process holds",
             fontsize=13.5, fontweight="bold", color=INK_PRIMARY, ha="left", va="top")
    fig.text(0.055, 0.915,
             "Median $\\Delta E_{00}$ across the n-colour ladder. Each rung is an independent dataset and printing system\n"
             "— not one printer swept over ink counts — so compare each model's own trend, not levels across rungs.",
             fontsize=9, color=INK_SECONDARY, ha="left", va="top", linespacing=1.35)

    # Footnote: what was left out, with the range taken from the same CSVs.
    fig.text(0.055, 0.018,
             f"Linear-family models (ridge, lasso, elastic net, PCR, PLSR) omitted: their medians span "
             f"{lin_lo:.1f}–{lin_hi:.1f} $\\Delta E_{{00}}$ across these datasets.",
             fontsize=7.5, color=INK_MUTED, ha="left", va="bottom")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.845, bottom=0.155)
    fig.savefig(out_path, dpi=220, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def main() -> None:
    df, lin_lo, lin_hi = load_medians()
    sub = df.loc[[m for m, *_ in MODELS]]
    print("Medians plotted:")
    print(sub.round(3).to_string())
    print(f"Linear-family median range (omitted): {lin_lo:.3f}-{lin_hi:.3f}")
    make_figure(df, lin_lo, lin_hi, OUT_PATH)


if __name__ == "__main__":
    main()
