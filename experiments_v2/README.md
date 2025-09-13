experiments_v2 — Clean pipeline for CMY→XYZ printer characterization

Overview
- Purpose: Re-implement the study with a clean, modular pipeline that evaluates many regressors for predicting XYZ from CMY and computes ΔE00 on denormalized XYZ→Lab (correct handling).
- Key features:
  - Proper denormalization: compute Lab/ΔE00 after inverse-scaling XYZ.
  - Holdout or K-fold evaluation with repeats.
  - Metrics: mean, median, max, 95th percentile (P95), SD, SEM.
  - Consistent CSV outputs per model + summaries per dataset.
  - CLI to select dataset, models, and evaluation mode.

Usage
- Minimal deps: scikit-learn, numpy, pandas, colour-science
- Example (holdout, 1 repeat):
  uv run python experiments_v2/run_experiment.py --dataset PC10 --mode holdout --repeats 1 --out experiments_v2/results
- K-fold (5-fold, 2 repeats):
  uv run python experiments_v2/run_experiment.py --dataset FOGRA --mode kfold --folds 5 --repeats 2 --out experiments_v2/results

Notes
- Datasets are loaded from the repo's `cleaned/` directory.
- Optional K=0 filtering: use `--k-zero-only` to keep only rows with CMYK_K==0.

