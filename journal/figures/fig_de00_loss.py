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
import numpy as np
from decimal import ROUND_HALF_UP, Decimal


def _half_up(v):
    """Round half up from the CSV's decimal string (6.795 -> 6.80, not float 6.79)."""
    return str(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
OUT_PATH = os.path.join(HERE, "fig_de00_loss.png")

DATASETS = ["PC10-CMY", "PC11-CMY", "FOGRA51-CMY"]
DATASET_LABELS = {"PC10-CMY": "PC10", "PC11-CMY": "PC11", "FOGRA51-CMY": "FOGRA51"}

MODELS = ["poly3", "poly3_de00_powell"]
from _style import (FIG_W_075, FS_ANNOT, FS_SMALL, FS_TICK, INK_PRIMARY, INK_SECONDARY,
                    METRIC, MODEL_STYLE, apply_style, save)

# Both bars are the 3rd-order polynomial, so both wear its fixed vermillion:
# the least-squares fit solid, the Powell refinement light-filled and hatched.
# The legend title names the model; the entries name the optimizer.
MODEL_DISPLAY_NAMES = {
    "poly3": "Least-squares fit",
    "poly3_de00_powell": r"$\Delta E_{00}$ loss (Powell)",
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
    bar_w = 0.34
    x = list(range(n_datasets))
    offsets = [(i - (n_models - 1) / 2) * bar_w for i in range(n_models)]

    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_W_075, 2.9))

    for model, off in zip(MODELS, offsets):
        st = MODEL_STYLE[model]
        sub = data[data["model"] == model].set_index("dataset").reindex(DATASETS)
        xs = [xi + off for xi in x]
        solid = model == "poly3"
        ax.bar(
            xs, sub["max"], width=bar_w * 0.94,
            label=MODEL_DISPLAY_NAMES[model],
            facecolor=st["color"] if solid else st["facecolor"],
            edgecolor=st["color"], hatch=None if solid else st["hatch"],
            linewidth=0.8, zorder=3,
        )
        for xi, v in zip(xs, sub["max"]):
            ax.text(xi, v + 0.12, _half_up(v), ha="center", va="bottom",
                    fontsize=FS_ANNOT, color=INK_PRIMARY)

    # Worst-case reduction annotation (computed live, not hand-entered) above each dataset group.
    ymax = data["max"].max()
    for xi, ds in zip(x, DATASETS):
        p3 = data[(data.dataset == ds) & (data.model == "poly3")]["max"].iloc[0]
        pw = data[(data.dataset == ds) & (data.model == "poly3_de00_powell")]["max"].iloc[0]
        reduction = (p3 - pw) / p3 * 100
        ax.text(
            xi, ymax * 1.12, f"−{reduction:.0f}% worst case",
            ha="center", va="bottom", fontsize=FS_ANNOT, color=INK_PRIMARY, fontweight="bold",
        )
        med_p3 = data[(data.dataset == ds) & (data.model == "poly3")]["median"].iloc[0]
        med_pw = data[(data.dataset == ds) & (data.model == "poly3_de00_powell")]["median"].iloc[0]
        # Median pairing, set just under the dataset name.
        ax.annotate(
            f"median {med_p3:.2f} → {med_pw:.2f}",
            xy=(xi, 0), xytext=(0, -15), textcoords="offset points",
            ha="center", va="top", fontsize=FS_SMALL, color=INK_SECONDARY, style="italic",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=FS_TICK + 0.5,
                       color=INK_PRIMARY)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel(f"Maximum {METRIC}")
    ax.set_ylim(0, ymax * 1.24)
    top = 2 * int(np.ceil(ymax / 2))          # even tick at or above the tallest bar
    ax.set_yticks(np.arange(0, top + 1, 2))
    ax.spines["left"].set_bounds(0, top)
    ax.grid(axis="y")

    # Legend above the axes, outside the data, so it never meets the
    # worst-case annotations over the tallest bars.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Polynomial (3rd order):", title_fontsize=7.5,
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               labelcolor=INK_SECONDARY, alignment="left", columnspacing=1.2)

    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.86))
    save(fig, out_path)
    plt.close(fig)


def main() -> None:
    data = load_data()
    print(data)
    make_figure(data, OUT_PATH)


if __name__ == "__main__":
    main()
