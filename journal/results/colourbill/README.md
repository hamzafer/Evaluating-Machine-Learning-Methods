# colourbill (CharData Viewer) external-benchmark provenance

All numbers in `colourbill_fit_stats.csv` were read verbatim from the CharData Viewer UI
(https://chardata.colourbill.com/, © William Li, **v1.18.0**, accessed **11 Aug 2026** via
Playwright-driven browser). No number was adjusted. Screenshots of the exact UI blocks are in
`screenshots/`.

## Where each number comes from

For each dataset: left pane **Standard datasets** → click the file → it loads into **Dataset A**
(Explore mode) → the **Estimate** section auto-fits a polynomial model and shows a
"Model fit — degree D, N points" header plus five statistics: **Mean ΔE00, Min ΔE00, Max ΔE00,
Std Dev, Points fitted**. Those five values + the degree are what the CSV records.

Settings used (right-hand Settings pane): ΔE Method = **ΔE00**; Illuminant **D50**, Observer **2°**;
Colorimetry Source "Prefer spectral" (irrelevant — none of these files carry spectral data, so file
LAB is used); Model Weighted = **On** (tool default, IRLS). Weighted **Off** (plain least squares)
was also run for both CMYKOGV rows and produced **identical** displayed statistics (2 dp), so only
the Weighted=On rows are recorded.

`filter_duplicates` column: the tool's "Filter Duplicates" setting (applies at file load).
- CMYKOGV row 1: filter off → "Loaded 3534 rows." fit on 3534.
- CMYKOGV row 2: filter on (median) → "Loaded 3534 rows, reduced to 3302 rows by filtering
  duplicates (median)." — independently matching our ingest dedup (3534 → 3302).
- PC10/PC11/FOGRA51: filter off (all 1617 rows), matching our CMYK experiments (n=1617).
  The loader reported "Found 29 duplicate measurements ... (repeatability ΔE mean 0.00, max 0.00)"
  for all three files.

## What these numbers ARE (methodology caveat — do not drop when citing)

CharData's Estimate statistics are **resubstitution (training-set) errors**: one polynomial
(degree auto-selected 2..min(5, #colorants) by mean ΔE, IRLS-weighted least squares,
colorants → L\*a\*b\* directly) is fitted to **all** loaded rows and evaluated on those same rows.
There is **no train/test split**. Our pipeline numbers are pooled **5-fold cross-validation
test errors** (unseen patches). The two protocols are NOT directly comparable; resubstitution is
systematically optimistic. `journal/figures/fig_vs_colourbill.py` bridges the protocols by
evaluating the same model class (least-squares polynomial → Lab) under both protocols on the same
data (outputs in this folder: `reproduction_check.csv`, `poly_lab_cv.csv`, `comparison.csv`).

**Reproduction finding (11 Aug 2026):** CharData's numbers are *not* exactly reproducible from
its documented strategy — the actual fitter lives in a compiled WASM module (loaded by
`gamut.js`; no readable source). Our full-basis OLS implementation of the documented strategy
fits *tighter* than the tool's displayed stats (e.g. PC10 mean 0.14 vs 0.31; CMYKOGV-7 max 4.81
vs 12.37 at the same degree 4) on 3 of 4 datasets (PC11 max is the exception: ours 1.76 vs 1.64).
So treat colourbill's stats as a conservative external reference for the polynomial-fit class,
not a bit-exact spec; the genuinely like-for-like row in `comparison.csv` is our own
implementation of that class ("same poly class, our CV"). Details per dataset in
`reproduction_check.csv`. Also note the tool auto-selected degree 4 everywhere, while our
implementation's selection rule would pick degree 5 on CMYKOGV-7; the bridge pins degree 4
(colourbill's choice).

**Moving-target note:** the `our best model (GP)` rows in `comparison.csv` are read live from
`journal/results/*/summary.csv`, which the Plan-10 GP re-run was updating on 11 Aug when this
was generated. Re-run `journal/figures/fig_vs_colourbill.py` after Plan 10 lands to refresh
(CSV + figure regenerate from sources; nothing is hand-entered).

## Same-data verification

The tool's standard datasets were downloaded from
`https://chardata.colourbill.com/standard-data/<file>.txt` and compared numerically to ours:
- `APTEC_CMYKOGV_Coated_LinearCTV_2025_M1.txt` = our `journal/data/raw/ncolor/APTEC_CMYKOGV_7clr_xyzlab.txt`:
  all 3534 rows, all numeric columns identical, same order. (colourbill's copy has the corrected
  `NUMBER_OF_SETS 3534` header; the ICC-registry copy we hold says a stale 1624.)
- `APTEC_PC10_CardBoard_2023_v1.txt`, `APTEC_PC11_CCNB_2023_v1.txt`, `FOGRA51.txt` = our
  `data/cleaned/*.csv`: 1617/1617 rows exact match on CMYK + LAB (our XYZ columns are derived).

## ΔE00 cross-check via Compare mode (closes the metric loop)

`PC10_poly4lab_cv_predictions.txt` = CGATS export of the bridge model's pooled 5-fold-CV Lab
predictions for PC10 (same CMYK inkings, predicted Lab). Loaded as Dataset B in Compare mode
against the standard PC10 (Dataset A), colourbill reports (ΔE00): N 1617, **Mean 0.16,
Median 0.11, P90 0.29, P95 0.38, Max 2.41, Std 0.19** — identical at displayed precision to our
colour-science computation (mean 0.157, median 0.110, p95 0.380, max 2.412). The external tool
independently confirms our ΔE00 arithmetic. Verbatim record: `compare_crosscheck.csv`;
screenshots `PC10_compare_*.png`.

## Screenshots

- `CMYKOGV_datasetA_info.png`, `CMYKOGV_weightedOn_estimate.png` — unfiltered (3534) load + fit.
- `CMYKOGV_dedup3302_datasetA_info.png`, `CMYKOGV_dedup3302_estimate.png` — duplicate-filtered load + fit.
- `PC10_datasetA_info.png` / `PC10_estimate.png`, `PC11_...`, `FOGRA51_...` — per-dataset load + fit.
- `PC10_compare_datasetA.png` / `PC10_compare_datasetB.png` / `PC10_compare_predictions_stats.png`
  — Compare-mode ΔE00 cross-check (measured PC10 vs our CV predictions).
