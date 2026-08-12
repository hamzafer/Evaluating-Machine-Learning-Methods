#!/usr/bin/env python3
"""
Plan 11: external benchmark against colourbill's CharData Viewer
(https://chardata.colourbill.com/, W. Li).

CharData's "Estimate" fits one least-squares polynomial (degree auto 2..min(5,#inks),
IRLS-weighted, inks -> Lab) to ALL rows of a dataset and reports RESUBSTITUTION
(training-set) dE00 stats: mean/min/max/std. Our pipeline reports pooled 5-fold-CV
TEST errors. The two protocols are not directly comparable, so this script bridges them:

  step 1  reproduction check  - refit CharData's documented model class (OLS polynomial
          on Lab, auto degree by mean dE00) on the identical data and record how close
          we land to its published fit stats  -> reproduction_check.csv
          (FINDING 11 Aug 2026: not exactly reproducible - CharData's fitter is a
          compiled WASM module; our full-basis OLS fits tighter than its displayed
          stats on 3 of 4 datasets, so its numbers are a conservative reference for
          the polynomial-fit class, not a bit-exact spec.)
  step 2  protocol bridge     - evaluate the SAME model class under OUR CV protocol
          (5-fold, seed 42; GroupKFold on recipe groups for CMYKOGV, matching
          journal/pipeline/evaluate.py)  -> poly_lab_cv.csv
  step 3  comparison table    - colourbill verbatim numbers beside the bridge rows and
          our pipeline's best models (from journal/results/*/summary.csv)
          -> comparison.csv
  step 4  figure              - mean & max dE00, grouped bars per dataset
          -> journal/figures/fig_vs_colourbill.png

Caveat that must accompany any use of this output: colourbill rows are in-sample
goodness-of-fit; CV rows are out-of-sample prediction error. Common metrics are
mean and max dE00 only (colourbill does not report median/p95).

Second, smaller caveat (12 Aug 2026): colourbill's coated-CMYK numbers were read off
its UI for the datasets as-received, i.e. fitted on 1617 rows, whereas our CV rows now
use the 1588 exact-deduplicated rows. colourbill is a compiled external tool we cannot
re-run per-configuration, so the row counts differ slightly between the two series. Each
row's own `n` is carried in comparison.csv; the 29-row difference is immaterial next to
the in-sample/out-of-sample gap above. (For CMYKOGV-7 both a 3534-row and a 3302-row
colourbill fit were captured, so that dataset is matched exactly.)

Sources (never hand-entered):
  journal/results/colourbill/colourbill_fit_stats.csv   (verbatim UI readings, 11 Aug 2026)
  data/cleaned/{APTEC_PC10_CardBoard_2023_v1,APTEC_PC11_CCNB_2023_v1,FOGRA51}.csv
  journal/data/processed/ncolor/CMYKOGV-7.csv
  journal/results/{PC10-CMYK,PC11-CMYK,FOGRA51-CMYK,CMYKOGV-7}/summary.csv

Run with: .venv/bin/python journal/figures/fig_vs_colourbill.py
"""
import os

import colour
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import PolynomialFeatures

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CB_DIR = os.path.join(REPO, "journal", "results", "colourbill")
OUT_FIG = os.path.join(HERE, "fig_vs_colourbill.png")

# Mirrors journal/pipeline/evaluate.py (SEED/FOLDS/make_groups); kept standalone
# like the other figure scripts. Do not change one without the other.
SEED, FOLDS = 42, 5

DATASETS = ["PC10-CMYK", "PC11-CMYK", "FOGRA51-CMYK", "CMYKOGV-7"]
DATASET_LABELS = {
    "PC10-CMYK": "PC10 (CMYK)",
    "PC11-CMYK": "PC11 (CMYK)",
    "FOGRA51-CMYK": "FOGRA51 (CMYK)",
    "CMYKOGV-7": "CMYKOGV (7-ink)",
}


def load_dataset(name, dedup):
    """Return (X inks 0-100, Lab).

    Two data bases are needed, because the two series being bridged were produced
    under different conventions:

      dedup=False -> the file exactly as colourbill's UI saw it. Used for the step-1
        reproduction check on the coated sets, whose whole point is to refit
        colourbill's model class on colourbill's own rows (1617). CMYKOGV-7 is the
        exception there: colourbill was run on both 3534 and 3302 rows and we pair
        with its 3302 one, so step 1 passes dedup=True for that dataset only.
      dedup=True  -> our pipeline's data basis, i.e. datasets.py's dedup_exact.
        Required for the step-2 bridge, which is labelled "the same model class
        under OUR protocol" -- and since 12 Aug our protocol drops byte-identical
        duplicate rows on every coated set (1617 -> 1588; CMYKOGV 3534 -> 3302).

    Keeping step 2 on the un-deduplicated rows would reintroduce into a published
    number the pooled-median double-counting that dedup exists to remove.
    """
    if name == "CMYKOGV-7":
        df = pd.read_csv(os.path.join(REPO, "journal", "data", "processed", "ncolor", "CMYKOGV-7.csv"))
        ink_cols = [f"INK_{i}" for i in range(1, 8)]
    else:
        csv = {
            "PC10-CMYK": "APTEC_PC10_CardBoard_2023_v1.csv",
            "PC11-CMYK": "APTEC_PC11_CCNB_2023_v1.csv",
            "FOGRA51-CMYK": "FOGRA51.csv",
        }[name]
        df = pd.read_csv(os.path.join(REPO, "data", "cleaned", csv))
        ink_cols = ["CMYK_C", "CMYK_M", "CMYK_Y", "CMYK_K"]
    if dedup:  # = datasets.py dedup_exact
        value_cols = ink_cols + ["XYZ_X", "XYZ_Y", "XYZ_Z", "LAB_L", "LAB_A", "LAB_B"]
        df = df.drop_duplicates(subset=value_cols).reset_index(drop=True)
    X = df[ink_cols].to_numpy(float)
    lab = df[["LAB_L", "LAB_A", "LAB_B"]].to_numpy(float)
    return X, lab


def de00(lab_true, lab_pred):
    return colour.difference.delta_E(lab_true, lab_pred, method="CIE 2000")


def fit_poly_lab(Xtr, labtr, Xte, degree):
    """CharData's documented model class, unweighted variant: least-squares
    polynomial (inks -> Lab). Inks scaled to 0-1 for conditioning."""
    pf = PolynomialFeatures(degree=degree, include_bias=True)
    A = pf.fit_transform(Xtr / 100.0)
    coef, *_ = np.linalg.lstsq(A, labtr, rcond=None)
    return pf.transform(Xte / 100.0) @ coef


def make_groups(X):
    """One group per distinct recipe (= evaluate.make_groups)."""
    _, inverse = np.unique(np.round(X, 6), axis=0, return_inverse=True)
    return inverse


def resub_stats(X, lab, max_degree):
    """CharData fitting strategy: degree 2..max, keep best by mean dE00
    (then std, then max); resubstitution stats of the kept model."""
    best = None
    for d in range(2, max_degree + 1):
        de = de00(lab, fit_poly_lab(X, lab, X, d))
        key = (de.mean(), de.std(), de.max())
        if best is None or key < best[0]:
            best = (key, d, de)
    _, d, de = best
    return d, {"mean": de.mean(), "min": de.min(), "max": de.max(), "std": de.std(), "n": de.size}


def cv_stats(X, lab, degree, grouped):
    """Same model class under our CV protocol; pooled per-sample dE00."""
    de = np.empty(len(X))
    if grouped:
        splits = GroupKFold(n_splits=FOLDS).split(X, groups=make_groups(X))
    else:
        splits = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(X)
    for tr, te in splits:
        de[te] = de00(lab[te], fit_poly_lab(X[tr], lab[tr], X[te], degree))
    return {"median": np.median(de), "p95": np.percentile(de, 95),
            "max": de.max(), "mean": de.mean(), "n": de.size}


def main():
    cb = pd.read_csv(os.path.join(CB_DIR, "colourbill_fit_stats.csv"))

    # -- step 1: reproduction check (resubstitution, same data, documented strategy)
    # Data basis = whichever rows colourbill itself fitted, so the comparison is
    # matched: coated CMYK has only a 1617-row colourbill run, CMYKOGV-7 has both
    # 3534 and 3302 and we pair with its 3302 (deduplicated) one.
    repro_rows = []
    for ds in DATASETS:
        X, lab = load_dataset(ds, dedup=(ds == "CMYKOGV-7"))
        max_deg = min(5, X.shape[1])
        deg, st = resub_stats(X, lab, max_deg)
        cb_row = cb[(cb.dataset == ds) & (cb.points_fitted == len(X))].iloc[0]
        # also fit at the degree colourbill reported, for a same-degree line
        de_cbdeg = de00(lab, fit_poly_lab(X, lab, X, int(cb_row.poly_degree)))
        repro_rows.append({
            "dataset": ds, "n": st["n"],
            "cb_degree": cb_row.poly_degree, "our_degree": deg,
            "cb_mean": cb_row.mean_de00, "our_mean": round(st["mean"], 3),
            "our_mean_at_cb_degree": round(de_cbdeg.mean(), 3),
            "cb_max": cb_row.max_de00, "our_max": round(st["max"], 3),
            "our_max_at_cb_degree": round(de_cbdeg.max(), 3),
            "cb_std": cb_row.std_de00, "our_std": round(st["std"], 3),
        })
    repro = pd.DataFrame(repro_rows)
    repro.to_csv(os.path.join(CB_DIR, "reproduction_check.csv"), index=False)
    print("reproduction_check:\n", repro.to_string(index=False))

    # -- step 2: the bridge (same model class, our CV protocol)
    # Bridge uses the degree colourbill itself selected (4 on all four datasets),
    # so the row is literally "colourbill's chosen model class under our protocol".
    # Data basis = OUR pipeline's, i.e. deduplicated everywhere (coated 1617 -> 1588,
    # CMYKOGV unchanged at 3302), so this row follows the protocol it claims to.
    bridge_rows = []
    for ds in DATASETS:
        X, lab = load_dataset(ds, dedup=True)
        deg = int(repro.set_index("dataset").loc[ds, "cb_degree"])
        st = cv_stats(X, lab, deg, grouped=(ds == "CMYKOGV-7"))
        bridge_rows.append({"dataset": ds, "degree": deg,
                            **{k: round(v, 3) for k, v in st.items() if k != "n"},
                            "n": st["n"]})
    bridge = pd.DataFrame(bridge_rows)
    bridge.to_csv(os.path.join(CB_DIR, "poly_lab_cv.csv"), index=False)
    print("\npoly_lab_cv:\n", bridge.to_string(index=False))

    # -- step 3: comparison table
    comp_rows = []
    for ds in DATASETS:
        # points_fitted selects which colourbill run to quote, and is COLOURBILL's
        # own fitted-point count -- not ours. colourbill fitted the coated CMYK
        # sets as-received (1617 rows); we now model 1588 after exact-dedup, and
        # colourbill is an external tool we cannot re-run, so this key stays 1617.
        # Do not "fix" it to 1588: there is no such colourbill run. The row-count
        # difference is noted in the caveat below (it is minor next to the
        # in-sample vs out-of-sample gap the comparison already carries).
        n_cb = {"CMYKOGV-7": 3302}.get(ds, 1617)
        cb_row = cb[(cb.dataset == ds) & (cb.points_fitted == n_cb)].iloc[0]
        comp_rows.append({"dataset": ds, "series": "colourbill poly (in-sample fit)",
                          "protocol": "resubstitution", "model": f"poly{cb_row.poly_degree} (Lab, IRLS)",
                          "median": None, "p95": None,
                          "mean": cb_row.mean_de00, "max": cb_row.max_de00, "n": cb_row.points_fitted})
        b = bridge.set_index("dataset").loc[ds]
        comp_rows.append({"dataset": ds, "series": "same poly class, our CV",
                          "protocol": "5-fold CV", "model": f"poly{int(b.degree)} (Lab, OLS)",
                          "median": b["median"], "p95": b.p95,
                          "mean": b["mean"], "max": b["max"], "n": int(b.n)})
        summ = pd.read_csv(os.path.join(REPO, "journal", "results", ds, "summary.csv")).set_index("model")
        gp = summ.loc["gaussian_process"]
        comp_rows.append({"dataset": ds, "series": "our best model (GP)",
                          "protocol": "5-fold CV", "model": "gaussian_process (XYZ)",
                          "median": gp["median"], "p95": gp.p95,
                          "mean": gp["mean"], "max": gp["max"], "n": int(gp.n)})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(os.path.join(CB_DIR, "comparison.csv"), index=False)
    print("\ncomparison:\n", comp.to_string(index=False))

    # -- step 4: figure (mean + max panels; the only stats colourbill shares)
    series = ["colourbill poly (in-sample fit)", "same poly class, our CV", "our best model (GP)"]
    colors = {series[0]: "#0072B2", series[1]: "#E69F00", series[2]: "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(DATASETS))
    width = 0.26
    for ax, stat, title in zip(axes, ["mean", "max"], ["Mean ΔE00", "Max ΔE00"]):
        for i, s in enumerate(series):
            vals = [comp[(comp.dataset == d) & (comp.series == s)][stat].iloc[0] for d in DATASETS]
            bars = ax.bar(x + (i - 1) * width, vals, width * 0.94, color=colors[s],
                          label=s if stat == "mean" else None)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=7.5, color="#333333")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x, [DATASET_LABELS[d] for d in DATASETS], fontsize=8.5)
        ax.set_ylabel("ΔE00", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("External benchmark: colourbill CharData polynomial fit vs this work "
                 "(in-sample fit vs 5-fold CV — protocols differ; see caption)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_FIG, dpi=200)
    print(f"\nwrote {OUT_FIG}")


if __name__ == "__main__":
    main()
