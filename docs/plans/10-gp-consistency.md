# 10 — GP Config / Newsprint Anomaly Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** Resolve the IFRA within-run Gaussian Process anomaly (median ~18.8, vs ~1–2 for other
models and ~3 for GP in the same experiment's cross-run/LOO), unify the GP configuration across all
datasets, and re-run so the paper reports GP under one documented config.

**Verified root cause (from the committed 1 Aug investigation — `docs/interpretations/2026-08-first-results.md` §8
and `docs/teaching/phil-meeting-briefing.md`):** it is a **WhiteKernel noise-floor mismatch**, not an
optimizer failure. The kernel uses `WhiteKernel(1e-5)` (assumes near-zero measurement noise, true for the
coated-paper sets whose duplicate patches are byte-identical), but newsprint has genuine press repeatability
of **~0.6–0.8 ΔE00** (measured independently from wb duplicate patches). With the noise floor too tight the
RBF length-scale collapses to its lower bound and predictions revert to the prior mean off-recipe → the ~18.8.

> CORRECTION vs the first draft of this plan: an earlier (reverted) ingestion attempt proposed
> `n_restarts_optimizer=10` as "the fix" and claimed KCMYG-5 showed the same collapse. That work was
> reverted and NOT verified (KCMYG-5 isn't ingested yet). Treat `n_restarts` as a secondary robustness
> knob to TEST, not the primary fix; treat the KCMYG-5 claim as an untested hypothesis.

> REVISION (11 Aug 2026, Task-1 diagnosis — `.superpowers/sdd/10-gp-consistency/task-1-report.md`):
> the noise-floor mismatch is real but acts at *initialization*, not through the bounds: the
> `WhiteKernel(1e-5)` init seeds a local-optimum basin (length-scale collapse, ΔLML ≈ 10,000 nats
> worse) that a single L-BFGS start cannot leave; even the collapsed fit had already raised its
> noise level to ~0.002 within the default bounds. Restarts escape the trap; a raised hard floor
> also fixes IFRA but harms clean coated data (PC10-CMY 0.054→0.091) and is rejected. The KCMYG-5
> collapse claim tested TRUE under `n_restarts=0`. Adopted config (unified, dataset-agnostic):
> `WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-9, 1e5))` + `n_restarts_optimizer=10`,
> later raised to **15** when the Task-3 never-worse gate caught one pooled-LOO cell (marca_133)
> where 10 draws over the widened bounds missed the healthy basin (see Task 3).

**Architecture:** Raise the WhiteKernel noise floor to reflect real measurement repeatability (dataset-aware:
~0 for the coated sets, ~0.5–0.8 ΔE00-equivalent variance for newsprint), re-run GP everywhere, and check
whether the IFRA within-run anomaly resolves. Optionally test `n_restarts_optimizer` as belt-and-suspenders.

**Tech Stack:** `journal/pipeline/` (models.py, run.py).

## Global Constraints
- Any GP config change must be verified **never worse** on datasets that already converged (PC10/PC11/FOGRA51/IFRA cross-run+LOO).
- Determinism: seed 42; GP subsample-to-2000 stays.
- If the IFRA within-run anomaly resolves, update Plan 03 results/figure and soften/remove the footnote — say so honestly. If it does NOT fully resolve, keep the footnote and report what the noise-floor fix did achieve.
- This is partly a modelling choice; document the chosen noise floor and its justification (the ~0.7 ΔE00 wb repeatability).

---

### Task 1: Diagnose + choose the noise floor
- [x] Reproduce the collapse: fit the current GP on one IFRA wb within-run fold; confirm length_scale hits its bound and log-marginal-likelihood is poor (matches the 1 Aug diagnosis). *(Reproduced exactly under the fde1335 config; recorded within_run.csv rows were stale vs the committed registry.)*
- [x] Quantify newsprint measurement variance from wb duplicate-recipe pairs (the ~0.6–0.8 ΔE00 figure) and translate to a sensible WhiteKernel noise_level / bound in normalized-Y units. *(Measured median 0.634 ΔE00 → noise_level ~2e-3 in normalized-Y variance units; the GP's own healthy fits chose ~0.002.)*
- [x] Decide: a single principled WhiteKernel bound that works for all datasets, OR a dataset-aware floor. Record the choice + justification in the report. *(Single dataset-agnostic config chosen — see REVISION note above; dataset-aware floor rejected.)*

### Task 2: Apply + regression-guard test
**Files:** Modify `journal/pipeline/models.py`; Test `journal/pipeline/tests/test_gp_config.py`.
- [x] Failing/guard test: on a synthetic sample with injected noise at the newsprint ratio (~2e-3), the unified GP does NOT collapse
  (length_scale stays off its bound; off-recipe median ΔE00 sane); on a near-noise-free sample it matches the old config within tolerance (never worse). Plus a frozen-config guard.
- [x] Implement the WhiteKernel change (single unified config, no dataset-aware wiring). Run tests → pass. Commit (no trailer). *(725abf0)*

### Task 3: Re-run all GP + reconcile
- [x] `run --models gaussian_process` on every dataset (PC10/PC11/FOGRA51 CMY+CMYK, the 13 IFRA wb specs, KCMYG-5/CMYKOGV-7/CMYKOGB-7). merge-update preserves other rows.
- [x] Diff GP medians before/after; confirm never-worse on the previously-converged sets. *(First pass at n_restarts=10 FAILED on exactly one cell — IFRA-wb-marca_133 LOO 3.247 → 4.876: the widened noise bounds stretch the restart-init draws over 14 decades and 10 draws missed the healthy basin on that one pooled-LOO fit. Coordinator-accepted remedy: n_restarts=15 — same seed keeps the first 10 draws, so the chosen optimum is equal-or-better in LML everywhere; verified to restore marca to 3.247. Full re-run at 15 passed the gate. Details: `.superpowers/sdd/10-gp-consistency/task-2-3-report.md`.)*
- [x] IFRA within-run GP: record the new number. If it drops into family, update Plan 03's figure/table/footnote. If not, keep the footnote and report the partial effect. *(Dropped into family — 0.674–2.141, mostly best-in-class; figure regenerated with GP included and an honest history footnote.)*
- [x] Optionally test `n_restarts_optimizer=10` on top; keep only if it strictly helps. *(Tested in Task 1: restarts are part of the fix — kept, raised to 15 after the LOO gate finding.)*
- [x] Commit updated summaries + a note in the interpretation doc.

### Acceptance
- One documented GP config; before/after showing never-worse; IFRA anomaly resolved (and cleaned up) or its residual honestly reported. Root cause and fix stated correctly (noise-floor mismatch *at initialization* — a local-optimum trap the restarts + neutral init remove; a hard raised floor was tested and rejected).
