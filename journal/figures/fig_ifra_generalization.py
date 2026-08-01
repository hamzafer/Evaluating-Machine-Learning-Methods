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

gaussian_process is EXCLUDED from the within-run bar only: on this newsprint data its
within-run kernel fit collapses to a noise-floor pathology (median-of-medians ~18.8,
vs ~1-2 for the other three models), which is a known GP kernel misconfiguration on
this dataset rather than a genuine generalization result. GP is included normally for
cross-run and leave-one-out, where it behaves like the other models.

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
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results", "ifra")
OUT_PATH = os.path.join(HERE, "fig_ifra_generalization.png")

MODELS = ["poly3", "svm", "mlp_deep", "gaussian_process"]
MODEL_DISPLAY_NAMES = {
    "poly3": "Polynomial (3rd order)",
    "svm": "SVM",
    "mlp_deep": "MLP (Deep)",
    "gaussian_process": "Gaussian Process",
}

# Fixed categorical color order (Okabe-Ito CVD-safe palette), assigned once per model.
MODEL_COLORS = {
    "poly3": "#0072B2",             # blue
    "svm": "#E69F00",               # orange
    "mlp_deep": "#009E73",          # bluish green
    "gaussian_process": "#D55E00",  # vermillion
}

REGIMES = ["within_run", "cross_run", "leave_one_out"]
REGIME_LABELS = {
    "within_run": "Within-run\n(train & test,\nsame press run)",
    "cross_run": "Cross-run\n(train 1 run,\ntest another)",
    "leave_one_out": "Leave-one-out\n(train 12 runs pooled,\ntest held-out run)",
}

GP_WITHIN_RUN_OMITTED_NOTE = (
    "GP within-run omitted (kernel noise-floor pathology on newsprint; see text)"
)


def load_regime_medians() -> pd.DataFrame:
    """Return DataFrame indexed by model, columns = regimes, values = median-of-medians."""
    within = pd.read_csv(os.path.join(RESULTS_DIR, "within_run.csv"))
    cross = pd.read_csv(os.path.join(RESULTS_DIR, "cross_run.csv"))
    loo = pd.read_csv(os.path.join(RESULTS_DIR, "leave_one_out.csv"))

    within_agg = within[within["model"].isin(MODELS) & (within["model"] != "gaussian_process")] \
        .groupby("model")["median"].median()
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
    bar_w = 0.19
    group_gap = bar_w * 0.15
    x = list(range(n_regimes))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.0, 6.6))

    offsets = [(i - (n_models - 1) / 2) * (bar_w + group_gap) for i in range(n_models)]

    for model, off in zip(MODELS, offsets):
        vals = agg.loc[model, REGIMES].to_numpy(dtype=float)
        xs = [xi + off for xi in x]
        bars = ax.bar(
            xs, vals, width=bar_w,
            label=MODEL_DISPLAY_NAMES[model],
            color=MODEL_COLORS[model],
            edgecolor="#3a3a3a", linewidth=0.4, zorder=3,
        )
        for xi, v in zip(xs, vals):
            if pd.isna(v):
                continue
            ax.text(xi, v + 0.35, f"{v:.2f}", ha="center", va="bottom", fontsize=8.2, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIMES], fontsize=9.5)
    ax.set_ylabel(r"Median-of-medians $\Delta E_{00}$")
    all_vals = agg[REGIMES].to_numpy(dtype=float).flatten()
    ymax = all_vals[~pd.isna(all_vals)].max()
    ax.set_ylim(0, ymax * 1.2)

    ax.set_title(
        "IFRA newsprint: press-to-press variation dominates model choice\n"
        f"cross-run transfer stays high (median ≈ {cross_overall:.1f}) regardless of model; "
        f"pooling 12 runs helps (leave-one-out median ≈ {loo_overall:.1f})",
        fontsize=11.8, pad=14,
    )

    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9, title="Model", ncol=2)

    fig.text(
        0.5, 0.005, GP_WITHIN_RUN_OMITTED_NOTE,
        ha="center", va="bottom", fontsize=8.2, color="#555555", style="italic",
    )

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def main() -> None:
    agg, cross_overall, loo_overall = load_regime_medians()
    print(agg)
    print(f"cross_run overall median: {cross_overall:.3f}")
    print(f"leave_one_out overall median: {loo_overall:.3f}")
    make_figure(agg, cross_overall, loo_overall, OUT_PATH)


if __name__ == "__main__":
    main()
