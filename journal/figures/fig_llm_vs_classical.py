#!/usr/bin/env python3
"""
Figure: median CIEDE2000 error, classical models vs LLM-as-color-predictor, on PC10-CMY.

Source (never hand-entered):
  journal/results/PC10-CMY/summary.csv          (classical models, 5-fold pooled CV; n read
                                                  from the CSV -- 795 since the 12 Aug dedup)
  journal/results/llm/PC10-CMY_summary.csv      (gpt-4o, gpt-4o-mini; "parsed_only" rows,
                                                  single 100-sample holdout, 400 in-context examples)

IMPORTANT METHODOLOGY CAVEAT (also stated in the figure caption): the classical-model
numbers come from 5-fold pooled cross-validation over the whole (deduplicated) dataset,
while the LLM numbers come from a single 100-sample holdout evaluation with 400
in-context examples. These are not directly, statistically equivalent evaluation
protocols -- the comparison here is illustrative of relative magnitude, not a controlled
head-to-head. Note the LLM holdout was drawn before deduplication; it is not re-run here,
so its 100 samples come from the 818-row pool.

Run with: .venv/bin/python journal/figures/fig_llm_vs_classical.py
Writes: journal/figures/fig_llm_vs_classical.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
OUT_PATH = os.path.join(HERE, "fig_llm_vs_classical.png")

CLASSICAL_SUMMARY = os.path.join(RESULTS_DIR, "PC10-CMY", "summary.csv")
LLM_SUMMARY = os.path.join(RESULTS_DIR, "llm", "PC10-CMY_summary.csv")

MODEL_DISPLAY_NAMES = {
    "gaussian_process": "Gaussian Process",
    "poly3": "Polynomial (3rd order)",
    "poly3_de00_nm": "Poly3 + ΔE00 loss (Nelder-Mead)",
    "poly3_de00_powell": "Poly3 + ΔE00 loss (Powell)",
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
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
}

# Classical models: single muted neutral (all one "classical" identity).
CLASSICAL_COLOR = "#5B7FA6"
# LLM models: one shared highlight color, distinct from the classical group.
LLM_COLOR = "#D55E00"  # vermillion (Okabe-Ito, CVD-safe, high contrast vs muted blue)

def caption(n_classical: int) -> str:
    return (
        f"Methodology caveat: classical models use 5-fold pooled cross-validation over "
        f"{n_classical} rows;\n"
        "LLM models (GPT-4o, GPT-4o-mini) use a single 100-sample holdout with 400 in-context\n"
        "examples ('parsed_only' responses). Evaluation protocols are not strictly equivalent."
    )


def load_data() -> pd.DataFrame:
    raw = pd.read_csv(CLASSICAL_SUMMARY)
    # Read n from the summary rather than hardcoding it: the coated sets are
    # exact-deduplicated at load (795 for PC10-CMY since 12 Aug), so a literal
    # here would silently go stale.
    n_classical = int(raw["n"].iloc[0])
    classical = raw[["model", "median"]].copy()
    classical["group"] = "classical"

    llm = pd.read_csv(LLM_SUMMARY)
    llm = llm[llm["variant"] == "parsed_only"][["model_id", "median"]].rename(columns={"model_id": "model"})
    llm["group"] = "llm"

    combined = pd.concat([classical, llm], ignore_index=True)
    combined = combined.sort_values("median", ascending=True).reset_index(drop=True)
    return combined, n_classical


def make_figure(df: pd.DataFrame, out_path: str, n_classical: int) -> None:
    n = len(df)
    y = list(range(n))
    colors = [LLM_COLOR if g == "llm" else CLASSICAL_COLOR for g in df["group"]]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.4, 8.2))

    bars = ax.barh(
        y, df["median"], height=0.62,
        color=colors, edgecolor="#3a3a3a", linewidth=0.4, zorder=3,
    )

    for yi, v in zip(y, df["median"]):
        ax.text(v * 1.06, yi, f"{v:.2f}", va="center", ha="left", fontsize=8.6, color="#2a2a2a")

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in df["model"]], fontsize=9.8)
    ax.set_ylim(-0.7, n - 0.3)
    # Best (lowest error) at top.
    ax.invert_yaxis()

    ax.set_xscale("log")
    xmax = df["median"].max()
    ax.set_xlim(0.03, xmax * 1.6)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(r"Median $\Delta E_{00}$ (log scale)")

    ax.set_title(
        "Classical regression vs LLM-as-color-predictor on PC10 (CMY)\n"
        "GPT-4o lands mid-pack; GPT-4o-mini trails every classical baseline",
        fontsize=12.2, pad=12,
    )

    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", which="both", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=CLASSICAL_COLOR, ec="#3a3a3a", linewidth=0.4),
        plt.Rectangle((0, 0), 1, 1, color=LLM_COLOR, ec="#3a3a3a", linewidth=0.4),
    ]
    ax.legend(
        legend_handles, [f"Classical (5-fold CV, n={n_classical})", "LLM (100-sample holdout)"],
        loc="upper right", frameon=True, framealpha=0.95, fontsize=9,
    )

    fig.text(0.5, 0.005, caption(n_classical), ha="center", va="bottom", fontsize=8.0,
             color="#555555", style="italic")

    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def main() -> None:
    df, n_classical = load_data()
    print(df)
    make_figure(df, OUT_PATH, n_classical)


if __name__ == "__main__":
    main()
