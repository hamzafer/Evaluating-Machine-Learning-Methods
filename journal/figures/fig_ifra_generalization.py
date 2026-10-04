#!/usr/bin/env python3
"""
Figure: IFRA newsprint generalization across three evaluation regimes.

Regimes (x-axis groups):
  - within-run:     train and test on the same press run (13 runs)
  - cross-run:      train on one run, test on a different run (156 ordered pairs)
  - leave-one-out:  train on the other 12 runs pooled, test on the held-out run (13 runs)

Model subset: gaussian_process, poly3, svm, mlp_deep.

Aggregation (never hand-entered): for each (regime, model), take the MEDIAN of the
per-run (or per-pair) "median" ΔE00 column -- i.e. a median-of-medians summary.

gaussian_process is shown in all three regimes. History note: under the pre-Plan-10
kernel (WhiteKernel(1e-5) init) the within-run GP fit collapsed to prior-mean
reversion (median-of-medians ~18.8) and was omitted from this figure; the unified GP
config (Plan 10: WhiteKernel(1e-3, bounds (1e-9,1e5)) + n_restarts_optimizer=15)
resolved it, and the within-run CSV rows were regenerated under that config.

Source (never hand-entered):
  journal/results/ifra/within_run.csv
  journal/results/ifra/cross_run.csv
  journal/results/ifra/leave_one_out.csv

Run with: .venv/bin/python journal/figures/fig_ifra_generalization.py
Writes: journal/figures/fig_ifra_generalization.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results", "ifra")
OUT_PATH = os.path.join(HERE, "fig_ifra_generalization.png")

from _style import (FIG_W_075, FS_ANNOT, INK_PRIMARY, INK_SECONDARY, METRIC,
                    MODEL_STYLE, apply_style, save)

MODELS = ["poly3", "svm", "mlp_deep", "gaussian_process"]
# Fixed per-model color + hatch from the shared style (same identity as in
# every other figure: Gaussian process blue, polynomial vermillion, ...).

REGIMES = ["within_run", "cross_run", "leave_one_out"]
REGIME_LABELS = {
    "within_run": "Within-run\n(train & test,\nsame press run)",
    "cross_run": "Cross-run\n(train 1 run,\ntest another)",
    "leave_one_out": "Leave-one-run-out\n(train 12 runs pooled,\ntest held-out run)",
}

def load_regime_medians() -> pd.DataFrame:
    """Return DataFrame indexed by model, columns = regimes, values = median-of-medians."""
    within = pd.read_csv(os.path.join(RESULTS_DIR, "within_run.csv"))
    cross = pd.read_csv(os.path.join(RESULTS_DIR, "cross_run.csv"))
    loo = pd.read_csv(os.path.join(RESULTS_DIR, "leave_one_out.csv"))

    within_agg = within[within["model"].isin(MODELS)].groupby("model")["median"].median()
    cross_agg = cross[cross["model"].isin(MODELS)].groupby("model")["median"].median()
    loo_agg = loo[loo["model"].isin(MODELS)].groupby("model")["median"].median()

    out = pd.DataFrame(index=MODELS)
    out["within_run"] = within_agg.reindex(MODELS)
    out["cross_run"] = cross_agg.reindex(MODELS)
    out["leave_one_out"] = loo_agg.reindex(MODELS)

    # Overall (all-model) pooled medians -- used to phrase the title narrative.
    cross_overall = cross["median"].median()
    loo_overall = loo["median"].median()
    return out, cross_overall, loo_overall


def make_figure(agg: pd.DataFrame, cross_overall: float, loo_overall: float, out_path: str) -> None:
    n_models = len(MODELS)
    n_regimes = len(REGIMES)
    bar_w = 0.2
    x = list(range(n_regimes))

    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_W_075, 3.0))

    offsets = [(i - (n_models - 1) / 2) * bar_w for i in range(n_models)]

    for model, off in zip(MODELS, offsets):
        st = MODEL_STYLE[model]
        vals = agg.loc[model, REGIMES].to_numpy(dtype=float)
        xs = [xi + off for xi in x]
        # White edge = the surface gap between adjacent bars; the hatch is
        # drawn in the edge color, so it shows as white texture on the fill.
        ax.bar(
            xs, vals, width=bar_w,
            label=st["label"], color=st["color"], hatch=st["hatch"] or None,
            edgecolor="white", linewidth=0.8, zorder=3,
        )
        for xi, v in zip(xs, vals):
            if pd.isna(v):
                continue
            ax.text(xi, v + 0.06, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=FS_ANNOT - 0.5, color=INK_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIMES], color=INK_PRIMARY,
                       linespacing=1.15)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel(f"Median-of-medians {METRIC}")
    all_vals = agg[REGIMES].to_numpy(dtype=float).flatten()
    ymax = all_vals[~pd.isna(all_vals)].max()
    ax.set_ylim(0, ymax * 1.08)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(axis="y")

    # Legend above the axes (outside the data), one row per two models.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               labelcolor=INK_SECONDARY, columnspacing=1.4, handlelength=1.8,
               handleheight=0.9)

    # No in-figure title or history footnote: the press-variation narrative and
    # the GP kernel-initialization history are stated in the paper's text.
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.88))
    save(fig, out_path)
    plt.close(fig)


def main() -> None:
    agg, cross_overall, loo_overall = load_regime_medians()
    print(agg)
    print(f"cross_run overall median: {cross_overall:.3f}")
    print(f"leave_one_out overall median: {loo_overall:.3f}")
    make_figure(agg, cross_overall, loo_overall, OUT_PATH)


if __name__ == "__main__":
    main()
