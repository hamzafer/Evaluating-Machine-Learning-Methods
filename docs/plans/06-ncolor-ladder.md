# 06 — n>4 Colorant Ladder Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** Ingest the three n>4 datasets and run the full model comparison, producing
the paper's headline: median ΔE00 vs ink-count across **n = 3, 4, 5, 7**, showing whether
classical methods degrade as inks grow while ML holds.

**Architecture:** Reuse the n-channel-generic pipeline. Two of the three files are spectral
(convert via the IFRA `spectral_to_xyz` path); one has native XYZ. Register as DatasetSpecs,
run `journal.pipeline.run`, then a figure script. No new modelling machinery.

**Tech Stack:** existing `journal/pipeline/` (colour-science, scikit-learn).

## Global Constraints
- ΔE00 on denormalized XYZ (D50/2°); median/P95/max; 2–3 decimals; fixed seed 42.
- GP: subsample training folds to <=2000 rows (seed 42) + `n_restarts_optimizer=10` (see Plan 10).
- Report per-dataset, never pooled; datasets differ in source/measurement conditions.
- Anomalous results investigated before reported.

## Datasets (in `journal/data/raw/ncolor/`, see its README)
| Name | Inks | Cols | Patches | Measurement |
|---|---|---|---|---|
| CMYKOGV-7 | 7 (APTEC) | 7CLR_1..7 + XYZ+LAB | 1624 | D50/2°, M1, registry |
| KCMYG-5 | 5 | 5CLR_1..5 + spectral | 2214 | D50/2°, ProfileMaker |
| CMYKOGB-7 | 7 (Apex) | 7CLR_1..7 + spectral | 2000 | D50/2°, ProfileMaker |

---

### Task 1: ProfileMaker/CGATS reader + ingester
**Files:** Create `journal/pipeline/ingest_ncolor.py`; Test `journal/pipeline/tests/test_ncolor.py`.
**Produces:** `ingest()` writing standard-schema CSVs to `journal/data/processed/ncolor/<name>.csv`
with columns `SAMPLE_ID, INK_1..INK_n, XYZ_X/Y/Z, LAB_L/A/B`.

- [ ] Step 1: Failing test — a minimal ProfileMaker-format fixture (LGO* header, `BEGIN_DATA_FORMAT`
  with `nCLR_k` + `nm380..nm730`, one row) parses; `spectral_to_xyz` gives sane XYZ (flat 60% → Y≈60);
  ink columns preserved. Also assert the APTEC file's native XYZ is used directly.
- [ ] Step 2: Run → fails (no module).
- [ ] Step 3: Implement. Reuse `journal.pipeline.ifra.parse_cgats` / `spectral_to_xyz` / `spectral_cols`
  (the `BEGIN_DATA_FORMAT`/`BEGIN_DATA` block is CGATS-like; the `nm###` regex already matches).
  For APTEC: use native XYZ_X/Y/Z + LAB. For the two spectral: derive via `spectral_to_xyz`.
  **Verify ink-value scale** (0–1 vs 0–100) per file and normalize to a consistent 0–100; record what you found.
- [ ] Step 4: Run tests → pass.
- [ ] Step 5: Run `ingest()` on the three real files; assert row counts (1624/2214/2000), no NaNs,
  ink ranges sane, unprinted-patch XYZ_Y plausible. For CMYKOGV cross-check derived-vs-native if both present.
- [ ] Step 6: Commit code + processed CSVs (git add -f) + README note. **No Co-Authored-By trailer.**

### Task 2: Register + run the comparison
**Files:** Modify `journal/pipeline/datasets.py`; Output `journal/results/{KCMYG-5,CMYKOGV-7,CMYKOGB-7}/summary.csv`.

- [ ] Step 1: Register three DatasetSpecs (input_cols = the n ink columns; filter_k_zero=False).
  Tripwire: works against native Lab (APTEC) / spectral-derived Lab; ensure `DatasetSpec.load` passes.
- [ ] Step 2: `.venv/bin/python -m journal.pipeline.run --datasets KCMYG-5 CMYKOGV-7 CMYKOGB-7` (all 14 models),
  foreground, one dataset at a time. GP obeys subsample + n_restarts. Do NOT block on background jobs.
- [ ] Step 3: Sanity/anomaly gate — 14 rows each; flag any model wildly out of family; investigate before reporting.
  Expected: poly3 degrades as n grows; GP strong; linear family poor.
- [ ] Step 4: Commit results.

### Task 3: The degradation figure
**Files:** Create `journal/figures/fig_ncolor_ladder.py` → PNG.

- [ ] Load `dataviz` skill. Median ΔE00 vs n (3,4,5,7) per model family, from the n=3/n=4 (PC10 as the
  CMY/CMYK reference) + the three new summaries. Log y-axis. Highlight poly3 vs GP trajectories.
  Because datasets differ, annotate that points at n=5/7 are independent datasets (not the same printer);
  the *within-model trend* is the message. Generate from CSVs only.
- [ ] Render, Read-back for collisions, commit script + PNG.

### Acceptance
- Three summaries + one ladder figure committed. The 3→4→5→7 trend is stated with per-dataset honesty.
- If GP or any model behaves anomalously at n=7, it is investigated and explained, not hidden.
