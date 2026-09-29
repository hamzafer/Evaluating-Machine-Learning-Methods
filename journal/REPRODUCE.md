# Reproducing the journal results (technologies-4564760)

Everything in the paper's tables and figures comes from the code in `journal/pipeline/`
and the CSVs in `journal/results/`. This page says where each protocol step lives
and how to rerun it.

## 1. Environment

Python 3.12.11 with the pinned set in `journal/requirements-frozen.txt`
(scikit-learn 1.7.2, NumPy 2.3.3, SciPy 1.16.2). Version drift in scikit-learn moves
results by roughly 0.01 to 0.05 ΔE00, so use the pins.

## 2. Data

The characterization data are third-party and are not redistributed here
(see the paper's Data Availability statement and `docs/DATA.md`).
Place the public sets at the paths in `docs/DATA.md` (FOGRA51 from Fogra; APTEC PC10,
PC11 and CMYKOGV from the ICC characterization data registry), then run the ingest
scripts for the sets that need conversion (`journal.pipeline.ingest_ncolor`,
`journal.pipeline.ingest_ifra`).

## 3. Protocol, step by step

| Step | Where |
|---|---|
| K=0 filter for the CMY conditions (rows kept only if K=0, never a dropped column) | `journal/pipeline/datasets.py`, `DatasetSpec.load` |
| Exact-duplicate removal (same inks and same XYZ+Lab) on the coated sets and CMYKOGV-7 | `journal/pipeline/datasets.py`, `dedup_exact`; rationale in `docs/DATA.md` |
| Recipe groups for the n>4 sets (rows with the same ink vector, rounded to 6 decimals, share a group) | `journal/pipeline/evaluate.py`, `make_groups` |
| Outer folds: seeded shuffled 5-fold KFold (seed 42), or GroupKFold on the recipe groups | `journal/pipeline/evaluate.py`, `fold_splits` |
| Scalers fitted on the training fold only; ΔE00 on denormalized XYZ (D50) | `journal/pipeline/evaluate.py`, `cross_validate`; `journal/pipeline/color.py` |
| Median, 95th percentile, maximum | `journal/pipeline/evaluate.py`, `summarize` |

The exact fold of every analyzed row is published in `journal/results/revision_r2/folds/`
(`idx` = row position after filtering and deduplication, `fold`, and `recipe_group` for the
grouped sets; `SAMPLE_ID` for the publicly available datasets). Regenerate and check with
`python -m journal.pipeline.revision_r2 folds`.

## 4. Rerunning the experiments

```
python -m journal.pipeline.run                       # main model matrix, per dataset
python -m journal.pipeline.run_ifra                  # newsprint experiments
python -m journal.pipeline.revision_r1 --help        # round-1 revision experiments
python -m journal.pipeline.revision_r1_timing        # fit / predict / LUT timing (idle machine)
python -m journal.pipeline.revision_r2 --help        # round-2: optimizer start-point sensitivity
python -m journal.pipeline.revision_r2_footprint     # round-2: memory footprint and latency
```

Per-sample ΔE00 for every (dataset, model) pair is in `journal/results/revision_r1/persample/`
and the one-row summaries in `journal/results/revision_r1/summary/`; round-2 outputs are in
`journal/results/revision_r2/`. Figures are generated from these CSVs by the scripts in
`journal/figures/`.
