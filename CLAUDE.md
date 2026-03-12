# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repo for AIC 2025 paper and NTNU report (IMT4304_10033_Phil) evaluating 14 ML methods for **CMY → XYZ color space regression**. Accuracy is measured using CIEDE2000 (ΔE₀₀). Three datasets: APTEC PC10 (cardboard), APTEC PC11 (coated paper), FOGRA51 (reference standard).

## Running the Pipeline

```bash
# From repo root, activate venv first
source .venv/bin/activate

# Run all 14 models on a dataset (PC10, PC11, or FOGRA)
cd main/cmy2xyz && PYTHONPATH=../.. python3 main.py PC10

# Results save to main/cmy2xyz/results/{dataset}/
# Merge per-model CSVs into a single file:
PYTHONPATH=. python3 -c "from utils.mergeFiles import merge; merge('PC10')"
```

**Important:** The pipeline must run from `main/cmy2xyz/` so result paths resolve correctly. `PYTHONPATH=../..` is needed for imports like `from input.input import get_dataset`.

## Architecture

- **`main/cmy2xyz/main.py`** — Entry point. Calls all 14 model `process()` functions sequentially.
- **`main/cmy2xyz/nn_*.py`** + **`polynomial_regression.py`** — Each model script tests multiple hyperparameter configs, picks the best, and saves results to CSV.
- **`input/input.py`** — Dataset loading. Reads from `data/cleaned/*.csv`, returns column subsets (CMY, CMYK, LAB, XYZ).
- **`utils/`** — Shared utilities: `xyz2lab.py` (XYZ→Lab, D50 white point), `calcError.py` (ΔE₀₀ via `colour-science`), `save_results.py`, `mergeFiles.py`.
- **`analysis/`** — Scripts generating publication figures (Figure 1: all models bar chart, Figure 2: top models comparison, PC10 table).
- **`figures/`** — Committed publication figures and summary CSVs.
- **`data/`** — Raw `.txt` files and `cleaned/` CSVs (3 datasets).

## Key Technical Details

- All models use **90/10 train/test split**, **MinMaxScaler** normalization, **random_state=42**.
- ΔE₀₀ is computed on **normalized XYZ** (not denormalized) — this is how the paper reports results.
- Each model script has identical boilerplate (load data, normalize, split, loop configs, compute errors, save). The pattern is consistent across all 13 `nn_*.py` files.
- 6 of 14 models (Bayesian GP, Random Forest, SVM, SimpleMLP, k-NN, Polynomial) show minor float differences (~0.01–0.05 ΔE) across scikit-learn versions. 8 models reproduce identically.

## Dependencies

```bash
pip install -r requirements.txt
# Contains: numpy, pandas, matplotlib, scikit-learn, colour-science, seaborn
```
