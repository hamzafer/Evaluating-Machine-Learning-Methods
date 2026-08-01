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

# Fixed categorical color order (Okabe-Ito palette, CVD-safe): baseline (n=3)
# first, then the n=4 comparison point.
COLOR_N3 = "#0072B2"  # blue
COLOR_N4 = "#D55E00"  # vermillion

MODEL_DISPLAY_NAMES = {
    "gaussian_process": "Gaussian Process",
    "poly3": "Polynomial (3rd order)",
    "svm": "SVM",
    "gradient_boost": "Gradient Boosting",
    "mlp_deep": "MLP (Deep)",
    "mlp_shallow": "MLP (Shallow)",
    "random_forest": "Random Forest",
    "knn": "k-NN",
    "decision_tree": "Decision Tree",
    "lasso": "Lasso",
    "elastic": "Elastic Net",
    "ridge": "Ridge",
    "pcr": "PCR",
    "plsr": "PLSR",
}

# Models whose n=3 -> n=4 shift gets a direct callout label (selective labeling,
# not a number on every point).
HIGHLIGHT_MODELS = {"poly3", "gaussian_process"}


def load_comparison() -> pd.DataFrame:
    cmy = pd.read_csv(CMY_PATH).set_index("model")["median"].rename("n3")
    cmyk = pd.read_csv(CMYK_PATH).set_index("model")["median"].rename("n4")
    df = pd.concat([cmy, cmyk], axis=1).dropna()
    df = df.sort_values("n3", ascending=True)
    # Reverse so the best (lowest n=3 error) model lands at the TOP of the chart.
    df = df.iloc[::-1]
    return df


def make_figure(df: pd.DataFrame, out_path: str) -> None:
    models = df.index.tolist()
    n = len(models)
    y = list(range(n))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 7.2))

    for yy, model in zip(y, models):
        v3 = df.loc[model, "n3"]
        v4 = df.loc[model, "n4"]
        ax.plot([v3, v4], [yy, yy], color="#9a9a9a", linewidth=2, zorder=2, solid_capstyle="round")

    ax.scatter(df["n3"], y, s=90, color=COLOR_N3, edgecolor="#3a3a3a", linewidth=0.6,
               zorder=3, label="n = 3 (CMY)")
    ax.scatter(df["n4"], y, s=90, color=COLOR_N4, edgecolor="#3a3a3a", linewidth=0.6,
               zorder=3, label="n = 4 (CMYK)")

    # Selective direct callouts for the two highlighted models.
    for model in HIGHLIGHT_MODELS:
        yy = models.index(model)
        v3 = df.loc[model, "n3"]
        v4 = df.loc[model, "n4"]
        ratio = v4 / v3
        label = f"{v3:.3f} → {v4:.3f}  (×{ratio:.1f})"
        x_text = max(v3, v4) * 1.35
        ax.annotate(
            label,
            xy=(max(v3, v4), yy), xytext=(x_text, yy),
            va="center", ha="left", fontsize=8.8, color="#2a2a2a",
            arrowprops=dict(arrowstyle="-", color="#7a7a7a", linewidth=0.8),
        )

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in models], fontsize=10)
    ax.set_ylim(-0.7, n - 0.3)

    ax.set_xscale("log")
    ax.set_xlim(0.03, 40)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(r"Median $\Delta E_{00}$ (log scale)")

    ax.set_title("Effect of adding K: median color error at n = 3 vs n = 4 inks (PC10)",
                 fontsize=12, pad=12)

    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", which="both", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def main() -> None:
    df = load_comparison()
    make_figure(df, OUT_PATH)


if __name__ == "__main__":
    main()
