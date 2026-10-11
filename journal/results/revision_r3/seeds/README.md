# Revision R3: fold-seed check of degree 5 against the Gaussian process

Produced by `journal/pipeline/revision_r3_seeds.py` (python 3.12.11, scikit-learn 1.7.2,
numpy 2.3.3, scipy 1.16.2; `journal/requirements-frozen.txt`). Run on the laptop with
four parallel jobs and `VECLIB_MAXIMUM_THREADS=3`.

```
python -m journal.pipeline.revision_r3_seeds jobs     # 46 jobs, one per line (dataset, tag, part)
python -m journal.pipeline.revision_r3_seeds run --dataset KCMYG-5 --tag s0 --part gp
python -m journal.pipeline.revision_r3_seeds merge    # writes the CSVs below
```

## What varies

Only the fold assignment. Data, models, model seeds, scalers and the DE00 path are as in
`revision_r3.py`.

| tag | n>4 sets (grouped) | PC10-CMYK |
|---|---|---|
| `published` | the paper's folds: unshuffled `GroupKFold` on recipe groups (asserted equal to `revision_r2/folds/`) | shuffled `KFold`, seed 42 |
| `s0`..`s3`, `s42` | `GroupKFold(5, shuffle=True, random_state=s)`: recipe groups shuffled, then split into five parts | shuffled `KFold`, seed s (`s42` = `published`) |

The seed set 0/1/2/3/42 is the one of the paper's coated-set shuffle-seed check.

Models: polynomials of degree 2 to 5 in XYZ and cube-root space, the cube-root Gaussian process,
every fixed configuration of the n>4 model table except the XYZ Gaussian process and the two
DE00-refined polynomials (not rerun, cost), and the twelve cube-root variants. PC10-CMYK: the
polynomial sweep and the cube-root Gaussian process only. Tuned models are not rerun.

## Files

- `summary.csv`: dataset, tag, model, pooled held-out median / P95 / max.
- `persample/<dataset>__<tag>.csv`: idx, fold, per-sample DE00 of the polynomial sweep and the cube-root GP.
- `paired.csv`: per tag, a = `poly5_cbrt` or `poly4_cbrt`, b = `gaussian_process_cbrt`; the test of
  `revision_r1_tables.paired` (bootstrap 95% CI of the median difference, B = 10,000, rng seed 42;
  two-sided Wilcoxon), Holm across the four datasets per tag and per `a`.
- `reliability.csv`: per tag and n>4 set, over the fixed configurations of the degree table and the
  n>4 model table run here (23 per set): best median / P95 / max and which model, the GP's ratio to
  each best, whether the GP has the lowest max, and the closest other configuration's worst ratio
  over the three statistics and three sets.
- `controls.csv`: the `published` tag against the published CSVs (every model's median / P95 / max at
  3 dp, per-sample values where published, and the published paired test). All 426 checks pass.
- `_jobs/` holds the raw job outputs and is not committed.
