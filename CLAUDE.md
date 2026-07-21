# CLAUDE.md

Guidance for Claude Code in this repository.

## Project

ML methods for printer color characterization (CMY(K)→XYZ regression, evaluated with CIEDE2000). Produced the AIC 2025 conference paper; now being extended into a journal paper.

**Current goal: MDPI Technologies special-issue paper. Submission deadline 30 Aug 2026.** Full plan: `docs/plans/journal_roadmap.md`. Core questions (Phil Green): (a) can ML match/beat classical methods for n≤4 inks, (b) can AI handle n>4 (CMYKOGV). Supporting: multi-printer generalization (newsprint), direct ΔE00 loss, LLM-as-color-predictor.

## State of the repo (July 2026)

- `main/cmy2xyz/` + `input/` + `utils/` = the **legacy AIC pipeline (v1)**. Its results are the published AIC numbers — keep as reference baseline, do not rebuild on it. Known flaw: ΔE00 computed on normalized XYZ.
- An `experiments_v2/` pipeline existed briefly and was **deliberately deleted** (commit 3dbbc6e). Do not resurrect it: it was CMY-only (3-channel, hardcoded to the three CSV datasets) and its data paths are stale. The journal work gets a **fresh pipeline, built n-channel-generic** (a dataset declares its input channels: 3, 4, or 7 — one code path for all). Useful reference: its corrected-ΔE00 PC10 results are recoverable via `git show 636b056:experiments_v2/results/PC10/`.
- `journal/` = the **new pipeline for the journal paper (v2)**, self-contained: `pipeline/` (n-channel-generic code), `llm/` (LLM-as-color-predictor track, separate since it calls an API rather than fitting a model), `data/raw/` + `data/processed/` (v2-only data — IFRA, future n>4 sets; the shared PC10/PC11/FOGRA51 CSVs stay in top-level `data/cleaned/`), `results/` (one subfolder per dataset), `figures/`.
- AIC paper PDF + presentation: `../Reports/`.

## Data

`data/cleaned/*.csv` — PC10, PC11, FOGRA51: 1,617 rows each, columns `SAMPLE_ID, CMYK_C/M/Y/K, LAB_L/A/B, XYZ_X/Y/Z`. Note: 799 rows have K>0; the AIC paper trained CMY-only by *dropping the K column but keeping those rows* (identical CMY → different XYZ in training). The fresh pipeline must handle K properly.

`journal/data/raw/Ifra-{wb,bb}.zip` — IFRA 2005 newsprint: 43 press runs (13 white-backing + 30 black-backing — **keep wb/bb separate**, per Phil), ECI2002/CGATS format, 1,485 samples each, CMYK + 36-band spectral reflectance 380–730nm. Needs ingestion: parse (latin-1 encoding) + spectral→XYZ via colour-science (D50, 2° observer).

n>4 (CMYKOGV) datasets: not yet received — waiting on Phil.

## Rules for the journal work

- ΔE00 always on **denormalized** XYZ (pred and truth → Lab D50 → `colour.difference.delta_E` CIE 2000).
- Report **median, max, 95th percentile** (not mean/std — errors aren't normal). 2–3 decimals max.
- Polynomial regression capped at 3rd order.
- Fixed seeds, pinned dependency versions (sklearn version drift moves results ~0.01–0.05 ΔE).
- Every experiment writes a results CSV; figures are generated from CSVs, never hand-made.
- Results that look anomalous get investigated before they get reported.

## Environment

`.venv` is fragile (symlinks to system python). Prefer a fresh pinned env (uv) for new work. Installed stack: numpy, pandas, scikit-learn, matplotlib, seaborn, colour-science.
