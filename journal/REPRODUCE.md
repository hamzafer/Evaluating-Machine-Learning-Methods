# Reproducing the journal results (technologies-4564760)

Everything in the paper's tables and figures comes from the code in `journal/pipeline/`
and the CSVs in `journal/results/`. This page says where each protocol step lives
and how to rerun it.

## 1. Environment

Python 3.12.11 with the pinned set in `journal/requirements-frozen.txt`
(scikit-learn 1.7.2, NumPy 2.3.3, SciPy 1.16.2, colour-science 0.4.6). Version drift in
scikit-learn moves results by roughly 0.01 to 0.05 ΔE00, so use the pins. From the repository
root, with [uv](https://docs.astral.sh/uv/):

```
uv venv --python 3.12.11 .venv
source .venv/bin/activate
uv pip install -r journal/requirements-frozen.txt
python -m pytest journal/pipeline/tests -q
```

or with plain pip (any Python 3.12.11 interpreter):

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r journal/requirements-frozen.txt
python -m pytest journal/pipeline/tests -q
```

Run every command on this page from the repository root with this environment active (the figure
scripts are run from `journal/figures/`, as shown in section 5). Without the datasets, the tests
that load them are skipped with a reason; all other tests, the table builder and the figure scripts
for the paper need no raw data.

The LLM-as-color-predictor track (`journal/llm/`) needs a few more packages:
`uv pip install -r journal/requirements-llm.txt` (this includes the core pins); its tests are
`python -m pytest journal/llm/tests -q`.

## 2. Data

The characterization data are third-party and are not redistributed here
(see the paper's Data Availability statement and `docs/DATA.md`). Only rerunning the
experiments (section 4) needs them; rebuilding the tables and figures (section 5) does not.

**Coated sets (PC10, PC11, FOGRA51).** Download:

- APTEC PC10 and PC11 from the ICC characterization data registry
  (https://registry.color.org/cmyk-registry/aptec_coated_cardboard and
  https://registry.color.org/cmyk-registry/aptec_ccnb): `APTEC_PC10_CardBoard_2023_v1.txt`
  and `APTEC_PC11_CCNB_2023_v1.txt`;
- FOGRA51 from Fogra (https://fogra.org/en/downloads/work-tools/characterisation-data):
  download `F51-52.zip` and take `FOGRA51.txt` (CMYK and Lab, 1,617 patches) from it.

These files hold CMYK and Lab only. Put them in `data/` (any folder works) and run

```
mkdir -p data
python -m journal.pipeline.ingest_coated data/APTEC_PC10_CardBoard_2023_v1.txt \
    data/APTEC_PC11_CCNB_2023_v1.txt data/FOGRA51.txt
```

It adds XYZ (0-100, from the measured Lab with the ICC D50 white X, Y, Z = 96.422, 100, 82.521)
and writes `data/cleaned/<same name>.csv`, the files the loader reads. Keep the file names as
above: the output name is the input name with `.csv`. The rebuilt files load to exactly the values
of the files used for the paper.

**APTEC CMYKOGV (n=7).** Download `APTEC_CMYKOGV_Coated_LinearCTV_2025_M1.txt` from
https://registry.color.org/cmyk-registry/aptec_cmykogvcoated, save it as
`journal/data/raw/ncolor/APTEC_CMYKOGV_7clr_xyzlab.txt` (the name `ingest_ncolor` expects; create
the folder first with `mkdir -p journal/data/raw/ncolor`), and run

```
python -m journal.pipeline.ingest_ncolor
```

It writes `journal/data/processed/ncolor/CMYKOGV-7.csv`. The two restricted n>4 sets (KCMYG-5,
CMYKOGB-7) are skipped with a note when their raw files are absent.

**Restricted sets.** The newsprint runs (`journal.pipeline.ingest_ifra`), KCMYG-5 and CMYKOGB-7
can only be used with their providers' permission (see the Data Availability statement).

After filtering and deduplication the loader gives 795 rows for the CMY conditions, 1,588 for CMYK
and 3,302 for CMYKOGV-7, the rows of the published fold files (`journal/results/revision_r2/folds/`).
With the data in place, `python -m pytest journal/pipeline/tests -q` also
runs the dataset tests that were skipped before.

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
(`idx` = row position after filtering and deduplication, and `fold`; for the publicly available
datasets also `SAMPLE_ID`, and `recipe_group` where folds are grouped). Regenerate them with
`python -m journal.pipeline.revision_r2 folds`, which asserts that its rows match the dataset loader.
It needs the seven public datasets; a restricted set whose data are absent (KCMYG-5, CMYKOGB-7) is
skipped with a note and its published fold file is left unchanged.

## 4. Rerunning the experiments

```
python -m journal.pipeline.run                         # main model matrix, per dataset
python -m journal.pipeline.run_ifra                    # newsprint experiments
python -m journal.pipeline.revision_r1 --help          # round-1 revision experiments
python -m journal.pipeline.revision_r1_timing          # Table 12 timing (idle machine)

# round 2 (Reviewer 2)
python -m journal.pipeline.revision_r2 folds           # fold assignment files (section 3)
python -m journal.pipeline.revision_r2 ifra-pairs      # newsprint run metadata, model-free run comparison
python -m journal.pipeline.revision_r2 optim-cost      # convergence and cost of the direct-DE00 fits
python -m journal.pipeline.revision_r2 jobs            # the 990 start-point jobs, one per line
python -m journal.pipeline.revision_r2 seeds --dataset PC10-CMY --fold 0 --start ols --method powell
python -m journal.pipeline.revision_r2 seeds-merge     # pool the jobs into seeds/summary.csv
python -m journal.pipeline.revision_r2_footprint       # memory footprint and latency (idle machine)
python -m journal.pipeline.revision_r2_footprint --embedded   # degree-4 polynomial as a bare matrix
python -m journal.pipeline.revision_r2_tables          # LaTeX tables from the round-2 CSVs
cd journal/figures && python fig_de00_loss.py          # likewise the other figure scripts
```

`seeds` writes to `journal/results/revision_r2/seeds/`; set `R2_SEEDS_DIR=seeds_laptop` to write
elsewhere (the six coated datasets were run on the laptop, the canonical platform, into
`seeds_laptop/`; the full sweep ran on a second machine into `seeds/`). Timings are
machine-specific.

On the same platform (macOS/arm64), reruns reproduce the reported values exactly, also from the
public files rebuilt as in section 2 (the CMYKOGV-7 file is rebuilt byte for byte; for the coated
files, all 20 models on the six conditions, 120 model/dataset cells, matched in our check). The
iterative and stochastic models (SVM, the MLPs, the tree models and the Powell-refined polynomial)
depend on the data down to the last digit. An earlier
version of `ingest_coated`, whose XYZ differed from the paper's files by at most 3e-14, moved 34 of
these 120 cells, by up to 0.143 ΔE00 in a median and 11.118 in a maximum (the MLPs). Data prepared
in any other way can move these models by similar amounts.

Across platforms, a re-run on Linux/x86-64 with the same pinned environment matched 58 of
96 model/dataset cells exactly and 76 of 96 to within 0.01 ΔE00, with a worst case of 0.108 ΔE00;
the largest differences are in the iteratively optimized neural nets (see the paper's
Reproducibility paragraph).

The KCMYG-5 and CMYKOGB-7 experiments need the restricted datasets, so they can only be rerun
with the providers' permission. Their per-row outputs here contain row positions, folds and errors
only.

Per-sample ΔE00 for every (dataset, model) pair is in `journal/results/revision_r1/persample/`
and the one-row summaries in `journal/results/revision_r1/summary/`; round-2 outputs are in
`journal/results/revision_r2/`. Figures are generated from these CSVs by the scripts in
`journal/figures/` (they use Arial; matplotlib falls back to its default font where it is absent).

## 5. Rebuilding the tables and figures from the CSVs

No raw data is needed for this step: it reads only the tracked results CSVs.

```
python -m journal.pipeline.build_tables --write journal/results/tables   # one fragment per table label
git diff --exit-code journal/results/tables                              # rebuilt tables equal the tracked ones
python -m journal.pipeline.revision_r3_tables                            # round-3 degree tables (main + supplementary)
python -m journal.pipeline.build_tables --check path/to/main.tex         # with the LaTeX source: check every table cell
```

Without the LaTeX source, the first two lines are the check: `--write` rebuilds every table
fragment from the CSVs, and `git diff --exit-code` exits non-zero if any rebuilt fragment differs
from the tracked one. The fragments hold the table rows as printed in the paper.

`build_tables` rebuilds every main-text table that holds model results (`tab:n3`, `tab:n4`,
`tab:de00loss`, `tab:fitspace`, `tab:transforms`, `tab:degree`, `tab:newsprint`, `tab:n7`,
`tab:stats`, `tab:timing`, the "This paper" rows of `tab:prior`, and Appendix A `tab:appbest`)
from the per-dataset `summary.csv` files and the `revision_r1/`, `revision_r2/` and `revision_r3/`
CSVs. Values are read as CSV strings and rounded half up, as the paper prints them. `--write` puts
the rows of each table (between `\midrule` and `\bottomrule`) in `journal/results/tables/`, one file
per label with the colon replaced by an underscore (`tab:n3` is `tab_n3.tex`).
`--check` parses each table of the manuscript by its `\label`, compares every numeric cell with
the rebuilt value, prints matched and mismatched cells per table with the CSV path of each
expected value, and exits non-zero on any mismatch. It also confirms that every CSV it read is
tracked by git.

The round-3 degree sweep itself is rerun with `python -m journal.pipeline.revision_r3 sweep`
(see `journal/results/revision_r3/README.md`).

Figures of the paper (there are no figures in the Supplementary Materials). Each is drawn from
the results CSVs by its own script in `journal/figures/` (the source CSVs are named in each
script's header) and needs no raw data:

| Paper | Script | Output |
|---|---|---|
| Figure 1 (PC10, n=3 vs n=4, 14 methods) | `fig_n3_vs_n4.py` | `fig_n3_vs_n4.png` |
| Figure 2 (direct ΔE00 loss, maximum error) | `fig_de00_loss.py` | `fig_de00_loss.png` |
| Figure 3 (newsprint generalization) | `fig_ifra_generalization.py` | `fig_ifra_generalization.png` (the paper's `fig_newsprint_generalization.png`) |
| Figure 4 (one panel per dataset) | `fig_per_dataset.py` | `fig_per_dataset.png`, `.pdf` |

```
cd journal/figures
python fig_n3_vs_n4.py            # Figure 1
python fig_de00_loss.py           # Figure 2
python fig_ifra_generalization.py # Figure 3 (newsprint)
python fig_per_dataset.py         # Figure 4
```

On macOS/arm64 the PNGs match the paper's figure files byte for byte; the PDF differs only in its
creation-date metadata. Other scripts in the folder draw figures that are not in the paper:

```
python fig_n3_model_comparison.py # CSV only
python fig_ncolor_ladder.py       # CSV only; former Figure 4 (round 2 and earlier)
python fig_llm_vs_classical.py    # CSV only
python fig_vs_colourbill.py       # needs the PC10, PC11, FOGRA51 and CMYKOGV-7 data (section 2)
```
