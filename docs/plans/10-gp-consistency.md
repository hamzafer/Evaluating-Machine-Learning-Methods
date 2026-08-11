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
- [ ] Reproduce the collapse: fit the current GP on one IFRA wb within-run fold; confirm length_scale hits its bound and log-marginal-likelihood is poor (matches the 1 Aug diagnosis).
- [ ] Quantify newsprint measurement variance from wb duplicate-recipe pairs (the ~0.6–0.8 ΔE00 figure) and translate to a sensible WhiteKernel noise_level / bound in normalized-Y units.
- [ ] Decide: a single principled WhiteKernel bound that works for all datasets, OR a dataset-aware floor. Record the choice + justification in the report.

### Task 2: Apply + regression-guard test
**Files:** Modify `journal/pipeline/models.py`; Test `journal/pipeline/tests/test_gp_config.py`.
- [ ] Failing/guard test: on a synthetic sample with injected noise ~0.7, the raised-floor GP does NOT collapse
  (length_scale stays off its bound; train median ΔE00 sane); on a near-noise-free sample it matches the old config within tolerance (never worse).
- [ ] Implement the WhiteKernel change (and, if chosen, dataset-aware wiring). Run tests → pass. Commit (no trailer).

### Task 3: Re-run all GP + reconcile
- [ ] `run --models gaussian_process` on every dataset (PC10/PC11/FOGRA51 CMY+CMYK, IFRA wb within-run, and the n>4 sets once Plan 06 lands). merge-update preserves other rows.
- [ ] Diff GP medians before/after; confirm never-worse on the previously-converged sets.
- [ ] IFRA within-run GP: record the new number. If it drops into family, update Plan 03's figure/table/footnote. If not, keep the footnote and report the partial effect.
- [ ] Optionally test `n_restarts_optimizer=10` on top; keep only if it strictly helps.
- [ ] Commit updated summaries + a note in the interpretation doc.

### Acceptance
- One documented GP config; before/after showing never-worse; IFRA anomaly resolved (and cleaned up) or its residual honestly reported. Root cause and fix stated correctly (noise floor, not optimizer restarts).
