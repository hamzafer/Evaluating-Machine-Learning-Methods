#!/usr/bin/env python3
"""
Figure: median CIEDE2000 error per model at n=3 (CMY) vs n=4 (CMYK) inks, on PC10.

Dumbbell/slope chart showing which models degrade as ink channels increase.

Source (never hand-entered): journal/results/PC10-CMY/summary.csv,
journal/results/PC10-CMYK/summary.csv
Run with: .venv/bin/python journal/figures/fig_n3_vs_n4.py
Writes: journal/figures/fig_n3_vs_n4.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
OUT_PATH = os.path.join(HERE, "fig_n3_vs_n4.png")

CMY_PATH = os.path.join(RESULTS_DIR, "PC10-CMY", "summary.csv")
CMYK_PATH = os.path.join(RESULTS_DIR, "PC10-CMYK", "summary.csv")

from _style import (FIG_W_075, FS_ANNOT, INK_MUTED, INK_PRIMARY, INK_SECONDARY,
                    METRIC, MODEL_STYLE, SURFACE, apply_style, save)

# n = 3 vs n = 4 is told apart by marker fill (open vs filled) in neutral ink, so
# the two ink counts never borrow a model's fixed color. The connecting segment
# of the two highlighted models wears that model's fixed color.
N3_FACE = SURFACE
N4_FACE = INK_PRIMARY

MODEL_DISPLAY_NAMES = {
    "gaussian_process": "Gaussian process",
    "poly3": "Polynomial (3rd order)",
    "svm": "SVM",
    "gradient_boost": "Gradient boosting",
    "mlp_deep": "MLP (deep)",
    "mlp_shallow": "MLP (shallow)",
    "random_forest": "Random forest",
    "knn": "k-NN",
    "decision_tree": "Decision tree",
    "lasso": "Lasso",
    "elastic": "Elastic net",
    "ridge": "Ridge",
    "pcr": "PCR",
    "plsr": "PLSR",
}
# Exactly the 14-model registry: the paper's caption states the figure shows
# the registry models only, so the dE00-refined and fitting-space variants in
# summary.csv are excluded here.

# Models whose n=3 -> n=4 shift gets a direct callout label (selective labeling,
# not a number on every point).
HIGHLIGHT_MODELS = ["gaussian_process", "poly3"]


def load_comparison() -> pd.DataFrame:
    cmy = pd.read_csv(CMY_PATH).set_index("model")["median"].rename("n3")
    cmyk = pd.read_csv(CMYK_PATH).set_index("model")["median"].rename("n4")
    df = pd.concat([cmy, cmyk], axis=1).dropna()
    df = df.loc[[m for m in df.index if m in MODEL_DISPLAY_NAMES]]
    df = df.sort_values("n3", ascending=True)
    # Reverse so the best (lowest n=3 error) model lands at the TOP of the chart.
    df = df.iloc[::-1]
    return df


def make_figure(df: pd.DataFrame, out_path: str) -> None:
    models = df.index.tolist()
    n = len(models)
    y = list(range(n))

    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_W_075, 3.35))

    for yy, model in zip(y, models):
        v3 = df.loc[model, "n3"]
        v4 = df.loc[model, "n4"]
        hl = model in HIGHLIGHT_MODELS
        ax.plot([v3, v4], [yy, yy],
                color=MODEL_STYLE[model]["color"] if hl else "#c4c4c4",
                linewidth=2.2 if hl else 1.4, zorder=2, solid_capstyle="butt")

    ax.scatter(df["n3"], y, s=22, facecolor=N3_FACE, edgecolor=INK_PRIMARY, linewidth=0.8,
               zorder=3, label="$n$ = 3 (CMY)")
    ax.scatter(df["n4"], y, s=22, facecolor=N4_FACE, edgecolor=INK_PRIMARY, linewidth=0.8,
               zorder=3, label="$n$ = 4 (CMYK)")

    # Selective direct callouts for the two highlighted models.
    for model in HIGHLIGHT_MODELS:
        yy = models.index(model)
        v3 = df.loc[model, "n3"]
        v4 = df.loc[model, "n4"]
        ratio = v4 / v3
        label = f"{v3:.3f} → {v4:.3f}  (×{ratio:.1f})"
        ax.annotate(
            label,
            xy=(max(v3, v4), yy), xytext=(6, 0), textcoords="offset points",
            va="center", ha="left", fontsize=FS_ANNOT, color=INK_PRIMARY,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in models])
    for tick, model in zip(ax.get_yticklabels(), models):
        tick.set_color(INK_PRIMARY if model in HIGHLIGHT_MODELS else INK_SECONDARY)
        if model in HIGHLIGHT_MODELS:
            tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, n - 0.3)

    ax.set_xscale("log")
    ax.set_xlim(0.03, 40)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(f"Median {METRIC} (log scale)")

    ax.grid(axis="x", which="major")
    ax.spines["left"].set_visible(False)

    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.64), handletextpad=0.3,
              labelcolor=INK_SECONDARY)

    fig.tight_layout(pad=0.3)
    save(fig, out_path)
    plt.close(fig)


def main() -> None:
    df = load_comparison()
    make_figure(df, OUT_PATH)


if __name__ == "__main__":
    main()
