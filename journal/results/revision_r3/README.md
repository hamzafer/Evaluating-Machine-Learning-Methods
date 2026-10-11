# Revision R3: polynomial degree, training vs held-out error

Produced by `journal/pipeline/revision_r3.py` (python 3.12.11, scikit-learn 1.7.2,
numpy 2.3.3, scipy 1.16.2, colour-science 0.4.6; `journal/requirements-frozen.txt`).

```
python -m journal.pipeline.revision_r3 sweep      # degree_sweep_train_vs_heldout.csv + persample_degree_sweep.csv
python -m journal.pipeline.revision_r3 controls   # controls.csv
python -m journal.pipeline.revision_r3 paired     # paired_poly5cbrt_vs_gpcbrt.csv
```

## Protocol

Same as `run.py` / `revision_r1.py`:

- **Data:** `DatasetSpec.load()`, with the same K=0 filter and exact-duplicate removal as the paper.
- **Folds:** read from the published fold files `journal/results/revision_r2/folds/<dataset>.csv`. The script asserts that they equal `evaluate.fold_splits`, which is GroupKFold on recipe groups for CMYKOGV-7, KCMYG-5 and CMYKOGB-7, and seeded shuffled KFold (seed 42) for PC10-CMYK.
- **Scaling:** MinMax scalers are fit on the training fold only.
- **Models:** the paper's own classes.
  - `xyz`: `PolynomialFeatures(d) + LinearRegression`, fitted on scaled XYZ.
  - `cbrt`: `models.CubeRootPolynomial(d)`.
- **Error:** the prediction is inverse-scaled and clipped at 0. ΔE00 is computed on denormalized XYZ (XYZ → Lab D50 → CIEDE2000).

## Columns of `degree_sweep_train_vs_heldout.csv`

One row per (dataset, degree 2..5, space).

| column | definition |
|---|---|
| `heldout_{median,p95,max}` | Statistic of the pooled held-out ΔE00. Every sample is predicted once, by the fold model that did not see it. This is the paper's reporting convention. |
| `heldout_fold_{stat}` | Mean over the 5 folds of the statistic on that fold's held-out rows. |
| `train_{stat}` | Mean over the 5 folds of the statistic on that fold's in-fold training rows. The model is scored on the rows it was fitted to. |
| `gap_{stat}` | Mean over folds of (held-out fold stat − training fold stat), which equals `heldout_fold_{stat} − train_{stat}`. `gap_median` is the definition used for `CMYKOGV-7/degree_sweep.csv` (tab:degree). |
| `terms_per_channel` | Number of polynomial terms including the bias, which is the coefficient count per output channel. |
| `mean_train_rows`, `rows_per_coef` | Mean training-fold size, and that size divided by `terms_per_channel`. |
| `seconds` | Wall time for the 5 folds on the laptop. |

The training set of a fold is about 4× larger than its held-out set. For that reason the max-based gap can be negative at low degree: the max is taken over more rows.

## Other files

- **`persample_degree_sweep.csv`:** `dataset, idx, fold, d{2..5}_{xyz,cbrt}`, giving the held-out ΔE00 for each sample. Here `idx` is the row position after load, the same index as `revision_r1/persample`.
- **`controls.csv`:** checks this run against published results.
  - Degree-3 and degree-4 rows vs `<dataset>/summary.csv` (median/P95/max, 3 dp).
  - Per-sample values vs the R1 per-sample files (max |diff|).
  - CMYKOGV-7 vs `degree_sweep.csv`.
  - All 80 checks pass.
- **`paired_poly5cbrt_vs_gpcbrt.csv`:** a = degree-5 cbrt, b = GP-cbrt, using the per-sample files in `revision_r1/persample`. The test is the one in `revision_r1_tables.paired`, as used for the GP vs poly4-cbrt table:
  - **Test:** two-sided Wilcoxon signed-rank (`zero_method='wilcox'`).
  - **Interval:** percentile bootstrap 95% CI of median(a) − median(b), with B = 10,000 and rng seed 42.
  - **Correction:** Holm across the 4 datasets tested here.
  - **Share:** `frac_a_better` is the share of patches where degree 5 has the lower ΔE00.
