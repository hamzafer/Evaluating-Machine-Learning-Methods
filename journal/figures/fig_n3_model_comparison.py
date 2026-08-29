#!/usr/bin/env python3
"""
Figure: median CIEDE2000 color error per model, across the three n=3 (CMY) datasets.

Source (never hand-entered): journal/results/{PC10-CMY,PC11-CMY,FOGRA51-CMY}/summary.csv
Run with: .venv/bin/python journal/figures/fig_n3_model_comparison.py
Writes: journal/figures/fig_n3_model_comparison.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
OUT_PATH = os.path.join(HERE, "fig_n3_model_comparison.png")

DATASETS = ["PC10-CMY", "PC11-CMY", "FOGRA51-CMY"]
DATASET_LABELS = {
    "PC10-CMY": "PC10",
    "PC11-CMY": "PC11",
    "FOGRA51-CMY": "FOGRA51",
}

# Fixed categorical color order (Okabe-Ito palette -- designed and validated for
# color-vision-deficiency safety; assigned once, never re-derived per chart).
DATASET_COLORS = {
    "PC10-CMY": "#0072B2",     # blue
    "PC11-CMY": "#E69F00",     # orange
    "FOGRA51-CMY": "#009E73",  # bluish green
}

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
    "poly3_de00_nm": "Poly3 + $\\Delta E_{00}$ (Nelder–Mead)",
    "poly3_de00_powell": "Poly3 + $\\Delta E_{00}$ (Powell)",
}
# The figure shows exactly the models named above: the 14-model registry plus
# the two dE00-refined variants, matching the paper's caption. The fitting-space
# variants (poly3_cbrt, poly4, poly4_cbrt, gaussian_process_cbrt) added to
# summary.csv by the fairness run are tabulated in the paper, not plotted here.

JND = 1.0


def load_medians() -> pd.DataFrame:
    frames = {}
    for ds in DATASETS:
        path = os.path.join(RESULTS_DIR, ds, "summary.csv")
        df = pd.read_csv(path)
        frames[ds] = df.set_index("model")["median"]
    combined = pd.DataFrame(frames)
    combined = combined.loc[[m for m in combined.index if m in MODEL_DISPLAY_NAMES]]
    # "Overall" rank = mean of each model's median ΔE00 across the three datasets.
    combined["overall"] = combined[DATASETS].mean(axis=1)
    combined = combined.sort_values("overall", ascending=True)
    # Reverse so the best (lowest-error) model lands at the TOP of the barh chart
    # (matplotlib barh draws index 0 at the bottom).
    combined = combined.iloc[::-1]
    return combined


def make_figure(combined: pd.DataFrame, out_path: str) -> None:
    models = combined.index.tolist()
    n = len(models)
    y = list(range(n))
    bar_h = 0.24

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 7.6))

    offsets = {"PC10-CMY": -bar_h, "PC11-CMY": 0.0, "FOGRA51-CMY": bar_h}
    for ds in DATASETS:
        vals = combined[ds].to_numpy()
        ax.barh(
            [yy + offsets[ds] for yy in y],
            vals,
            height=bar_h,
            label=DATASET_LABELS[ds],
            color=DATASET_COLORS[ds],
            edgecolor="#3a3a3a",
            linewidth=0.4,
            zorder=3,
        )

    # Reference line at the just-noticeable difference threshold.
    ax.axvline(JND, color="#555555", linewidth=1.2, linestyle="--", zorder=2)
    ax.text(
        JND, 0.985, "just-noticeable difference",
        transform=ax.get_xaxis_transform(),
        rotation=90, va="top", ha="right", fontsize=8.3, color="#454545",
    )

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in models], fontsize=10)
    ax.set_ylim(-0.6, n - 0.4)

    ax.set_xscale("log")
    ax.set_xlim(0.03, 12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(r"Median $\Delta E_{00}$ (log scale)")

    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", which="both", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9, title="Dataset")

    fig.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def main() -> None:
    combined = load_medians()
    make_figure(combined, OUT_PATH)


if __name__ == "__main__":
    main()
