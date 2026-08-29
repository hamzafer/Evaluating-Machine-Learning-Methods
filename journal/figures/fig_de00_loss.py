#!/usr/bin/env python3
"""
Figure: does optimizing directly for CIEDE2000 (rather than least-squares) reduce
worst-case color error? poly3 (least-squares fit) vs poly3_de00_powell (same cubic
polynomial basis, re-optimized to minimize ΔE00 directly via Powell's method).

Primary series: max ΔE00 (worst-case), grouped bars per dataset.
Secondary annotation: median ΔE00 for each model/dataset, shown as small labels
below the bar group, to make the "median stays flat" point visually explicit
(the Powell fit trades a small amount -- or nothing -- of typical-case accuracy
for a large cut in worst-case error).

Source (never hand-entered):
  journal/results/PC10-CMY/summary.csv
  journal/results/PC11-CMY/summary.csv
  journal/results/FOGRA51-CMY/summary.csv

Run with: .venv/bin/python journal/figures/fig_de00_loss.py
Writes: journal/figures/fig_de00_loss.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
OUT_PATH = os.path.join(HERE, "fig_de00_loss.png")

DATASETS = ["PC10-CMY", "PC11-CMY", "FOGRA51-CMY"]
DATASET_LABELS = {"PC10-CMY": "PC10", "PC11-CMY": "PC11", "FOGRA51-CMY": "FOGRA51"}

MODELS = ["poly3", "poly3_de00_powell"]
MODEL_DISPLAY_NAMES = {
    "poly3": "Poly3 (least-squares fit)",
    "poly3_de00_powell": "Poly3 + ΔE00 loss (Powell)",
}
MODEL_COLORS = {
    "poly3": "#0072B2",             # blue
    "poly3_de00_powell": "#D55E00",  # vermillion (highlight: the optimized variant)
}


def load_data() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        df = pd.read_csv(os.path.join(RESULTS_DIR, ds, "summary.csv")).set_index("model")
        for m in MODELS:
            rows.append({
                "dataset": ds,
                "model": m,
                "max": df.loc[m, "max"],
                "median": df.loc[m, "median"],
            })
    return pd.DataFrame(rows)


def make_figure(data: pd.DataFrame, out_path: str) -> None:
    n_datasets = len(DATASETS)
    n_models = len(MODELS)
    bar_w = 0.32
    x = list(range(n_datasets))
    offsets = [(i - (n_models - 1) / 2) * bar_w for i in range(n_models)]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.4, 7.2))

    for model, off in zip(MODELS, offsets):
        sub = data[data["model"] == model].set_index("dataset").reindex(DATASETS)
        xs = [xi + off for xi in x]
        ax.bar(
            xs, sub["max"], width=bar_w,
            label=MODEL_DISPLAY_NAMES[model], color=MODEL_COLORS[model],
            edgecolor="#3a3a3a", linewidth=0.4, zorder=3,
        )
        for xi, v in zip(xs, sub["max"]):
            ax.text(xi, v + 0.15, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color="#2a2a2a")

    # Worst-case reduction annotation (computed live, not hand-entered) above each dataset group.
    ymax = data["max"].max()
    for xi, ds in zip(x, DATASETS):
        p3 = data[(data.dataset == ds) & (data.model == "poly3")]["max"].iloc[0]
        pw = data[(data.dataset == ds) & (data.model == "poly3_de00_powell")]["max"].iloc[0]
        reduction = (p3 - pw) / p3 * 100
        ax.text(
            xi, ymax * 1.14, f"−{reduction:.0f}% worst-case",
            ha="center", va="bottom", fontsize=9.3, color="#8a3a00", fontweight="bold",
        )
        med_p3 = data[(data.dataset == ds) & (data.model == "poly3")]["median"].iloc[0]
        med_pw = data[(data.dataset == ds) & (data.model == "poly3_de00_powell")]["median"].iloc[0]
        ax.text(
            xi, -ymax * 0.075,
            f"median: {med_p3:.2f} → {med_pw:.2f}",
            ha="center", va="top", fontsize=8.4, color="#555555", style="italic",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=11)
    ax.set_ylabel(r"Max (worst-case) $\Delta E_{00}$")
    ax.set_ylim(-ymax * 0.16, ymax * 1.42)

    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#888888", linewidth=0.8, zorder=2)

    # Legend placed above the plotting area (outside the axes) so it never overlaps
    # the worst-case-reduction annotations sitting above the tallest bars.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
        ncol=2, frameon=True, framealpha=0.95, fontsize=9.5,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=320)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def main() -> None:
    data = load_data()
    print(data)
    make_figure(data, OUT_PATH)


if __name__ == "__main__":
    main()
