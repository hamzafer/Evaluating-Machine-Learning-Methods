# CLAUDE.md

Guidance for Claude Code in this repository.

## Git commits

Three distinct parts, three distinct rules:
- **Subject line** (the one-line summary): stays clean, no mention of Claude/the model.
- **Body/description** (the explanatory paragraph): fine to mention the model by name in prose if relevant (e.g. "drafted with Claude Sonnet 5").
- **`Co-Authored-By: Claude ...` trailer**: never add it, in either part — enforced via `.claude/settings.json`'s `attribution` setting, but keep it out even if writing a commit manually.

## Project

ML methods for printer color characterization (CMY(K)→XYZ regression, evaluated with CIEDE2000). Produced the AIC 2025 conference paper; now being extended into a journal paper.

**Current goal: MDPI Technologies special-issue paper. Submission deadline 30 Aug 2026.** Full plan: `docs/plans/journal_roadmap.md`. Core questions (Phil Green): (a) can ML match/beat classical methods for n≤4 inks, (b) can AI handle n>4 (CMYKOGV). Supporting: multi-printer generalization (newsprint), direct ΔE00 loss, LLM-as-color-predictor.

## State of the repo (July 2026)

- `main/cmy2xyz/` + `input/` + `utils/` = the **legacy AIC pipeline (v1), now corrected**. Its original ΔE00 was computed on MinMax-normalized XYZ (~5× understatement; confirmed Jul 2026, acknowledged by Phil — no AIC erratum, journal = new analysis). The one-line fix (`scaler.inverse_transform` before `xyz2lab`) is applied and all results regenerated: HEAD results = the **corrected conference baseline**. The exact published state lives at git tag `aic2025-published`. Second v1 flaw left in place by design: it trains on all 1,617 rows with K dropped (paper text wrongly claims K=0 filtering) — v1 stays frozen as the reference; K is handled properly only in `journal/`.
- An `experiments_v2/` pipeline existed briefly and was **deliberately deleted** (commit 3dbbc6e). Do not resurrect it: CMY-only, stale data paths. Its proper mode cross-verified the v1 fix (6-decimal agreement, Jul 2026); recoverable via `git show 636b056:experiments_v2/` if ever needed. The journal work gets a **fresh pipeline, built n-channel-generic** (a dataset declares its input channels: 3, 4, or 7 — one code path for all).
- `journal/` = the **new pipeline for the journal paper (v2)**, self-contained: `pipeline/` (n-channel-generic code), `llm/` (LLM-as-color-predictor track, separate since it calls an API rather than fitting a model), `data/raw/` + `data/processed/` (v2-only data — IFRA, future n>4 sets; the shared PC10/PC11/FOGRA51 CSVs stay in top-level `data/cleaned/`), `results/` (one subfolder per dataset), `figures/`.
- AIC paper PDF + presentation: `../Reports/`.

## Data

`data/cleaned/*.csv` — PC10, PC11, FOGRA51: 1,617 rows each, columns `SAMPLE_ID, CMYK_C/M/Y/K, LAB_L/A/B, XYZ_X/Y/Z`. Note: 799 rows have K>0; the AIC paper trained CMY-only by *dropping the K column but keeping those rows* (identical CMY → different XYZ in training). The fresh pipeline must handle K properly.

`journal/data/raw/Ifra-wb.zip` — IFRA 2005 newsprint, **white-backing only** (13 press runs × 1,485 samples; CMYK + 36-band spectral 380–730nm). Ingested. **Black-backing is OUT OF SCOPE** (11 Aug decision: a separate 'substrate correction' problem; its zip was moved out of the repo).

n>4 datasets: **all received (11 Aug)** — n=5 KCMYG, n=7 CMYKOGV, n=7 CMYKOGB in `journal/data/raw/ncolor/` (see its README and `docs/DATA.md`). Full n=3/4/5/7 ladder in hand.

## Rules for the journal work

- ΔE00 always on **denormalized** XYZ (pred and truth → Lab D50 → `colour.difference.delta_E` CIE 2000).
- Report **median, max, 95th percentile** (not mean/std — errors aren't normal). 2–3 decimals max.
- Polynomial regression capped at 3rd order.
- Fixed seeds, pinned dependency versions (sklearn version drift moves results ~0.01–0.05 ΔE).
- Every experiment writes a results CSV; figures are generated from CSVs, never hand-made.
- Results that look anomalous get investigated before they get reported.

## Environment

`.venv` is fragile (symlinks to system python). Prefer a fresh pinned env (uv) for new work. Installed stack: numpy, pandas, scikit-learn, matplotlib, seaborn, colour-science.
