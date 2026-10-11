# Nested cross-validation of fitting space and degree (revision R3, editor point 1)

Question: the cube-root fitting space (and, for the polynomial, the degree) was chosen
after results had been seen. If each model instead chooses its own configuration from
training data only, does the Gaussian process (GP) still lead the polynomial?

Script: `journal/pipeline/revision_r3_nested.py`.

## Design

- **Outer folds**: the paper's own folds (`journal/results/revision_r2/folds/`), asserted
  equal to `evaluate.fold_splits` (seeded shuffled KFold, seed 42, for the coated sets;
  GroupKFold on recipe groups for the n>4 sets). Scalers are fitted on the outer training
  fold, as in `evaluate.cross_validate`.
- **Inner selection** inside each outer training fold: the selection step of
  `models.InnerTuned` (3-fold GroupKFold on recipe groups, shuffled, seed 42; criterion =
  median CIEDE2000 on denormalized XYZ; ties go to the earlier candidate).
  - Polynomial: degree {2, 3, 4, 5} x fitting space {XYZ, cube root} (8 candidates).
    The degree grid is 2 to 5, as in the paper's degree sweep.
  - GP: fitting space {XYZ, cube root} (2 candidates), the paper's kernel and settings
    (`models.registry()['gaussian_process' | 'gaussian_process_cbrt']`).
- The selected candidate is refitted on the whole outer training fold and scored on the
  outer held-out fold. DE00: XYZ -> Lab D50 -> CIE 2000 (`color.delta_e00`).
- Every candidate is also fitted on every outer training fold, so the fixed
  configurations come from the same harness and are checked against the published numbers.

## Files

| File | Content |
|---|---|
| `persample/<dataset>.csv` | per-patch DE00: `poly_nested`, `gp_nested`, every fixed candidate (`poly_d{2..5}_{xyz,cbrt}`, `gp_xyz`, `gp_cbrt`), outer `fold` |
| `summary.csv` | median / P95 / max per dataset and column of the per-sample file |
| `selections.csv` | per dataset, family and outer fold: the selected candidate and the inner median DE00 of every candidate |
| `paired_nested_gp_vs_poly.csv` | Table-11 test (a = GP, b = polynomial): bootstrap 95% CIs (B = 10,000, seed 42), median difference with CI, Wilcoxon signed-rank, Holm over datasets. `comparison = nested` is the result; `fixed_gp_cbrt_vs_poly4_cbrt` must reproduce `revision_r1/tables/stats_gp_vs_poly4cbrt.csv` (the `paired` step exits with an error if it does not) |
| `controls.csv` | fixed configurations vs published `<dataset>/summary.csv` (3 decimals), vs R1 per-sample files and the R3 degree sweep (max abs difference), and the InnerSelect vs InnerTuned cross-check |
| `tables/nested_cv.tex` | supplementary table fragment built from the CSVs |
| `parts/`, `joblogs/` | per-job outputs and logs |

## How to run

Single-threaded BLAS per job (as for the R1 runs); at most 7 jobs in parallel were used.

```
python -m journal.pipeline.revision_r3_nested jobs > jobs.txt      # 45 GP jobs (dataset x fold) + 9 polynomial jobs
tr '\n' '\0' < jobs.txt | xargs -0 -P 7 -I{} sh -c 'journal/results/revision_r3/nested_cv/runjob.sh {}'
python -m journal.pipeline.revision_r3_nested merge
python -m journal.pipeline.revision_r3_nested controls     # exits 1 on any failed check
python -m journal.pipeline.revision_r3_nested paired     # exits 1 if Table 11 is not reproduced
python -m journal.pipeline.revision_r3_nested latex
```

`runjob.sh` uses `.venv/bin/python` unless `PY` is set. The KCMYG-5 and CMYKOGB-7 jobs
need the restricted datasets.

## Result (from `paired_nested_gp_vs_poly.csv`, `selections.csv`)

- Controls: 293 of 293 checks pass. The fixed configurations reproduce the published
  `summary.csv` values and per-sample files exactly, and the fixed GP-vs-poly4-cbrt pair
  reproduces `revision_r1/tables/stats_gp_vs_poly4cbrt.csv` (max deviation 1e-16; the
  `paired` step asserts this).
- Selections: the GP chose the cube-root space in 42 of 45 outer folds (XYZ in two
  FOGRA51-CMY folds and one FOGRA51-CMYK fold). The polynomial chose degree 5 in every outer
  fold except on CMYKOGB-7 (degree 4), cube root in 43 of 45 (XYZ in two FOGRA51-CMY folds).
- The degree grid is 2 to 5, as in the paper's degree sweep. On the coated sets degree 5 is
  the top of the grid and was chosen in every fold, so the nested polynomial results there
  hold for degrees up to 5 only.
- The nested GP has the lower median on 7 of 9 datasets (all six coated sets and CMYKOGB-7;
  bootstrap CI of the difference excludes zero). The nested polynomial has the lower median on
  CMYKOGV-7 (0.126 vs 0.160, CI excludes zero) and KCMYG-5 (0.742 vs 0.764, CI includes zero),
  with a higher maximum on both (12.038 vs 5.713; 136.611 vs 16.172).
