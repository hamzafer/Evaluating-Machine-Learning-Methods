# 10 — GP Config Consistency Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** Make the Gaussian Process configuration consistent across ALL datasets and re-run,
so the paper reports GP under one config. Motivated by a real finding: on KCMYG-5 (and the IFRA
within-run case) the default single-start optimizer collapsed to a degenerate kernel
(RBF length_scale → lower bound, WhiteKernel noise ballooned) — an optimizer failure, not overfitting.
`n_restarts_optimizer=10` reliably escapes it and is a strict improvement (never worse) where GP
already converged.

**Architecture:** One-line model change (`n_restarts_optimizer=10`) + the existing subsample-to-2000
rule, then re-run GP on every dataset and update summaries + the IFRA anomaly footnote if it resolves.

**Tech Stack:** `journal/pipeline/`.

## Global Constraints
- Change must be verified "never worse" on datasets that already converged (PC10/PC11/FOGRA51/IFRA).
- Determinism: seed 42; GP subsample-to-2000 stays.
- If the IFRA within-run GP anomaly (median ~18.8) resolves, update Plan 03 results, the figure, and remove/soften the footnote — and say so honestly in the paper.

---

### Task 1: Unify GP config + regression-guard test
**Files:** Modify `journal/pipeline/models.py`; Test `journal/pipeline/tests/test_gp_config.py`.

- [ ] Step 1: Failing/guard test — on a small structured sample where the degenerate optimum is reproducible,
  assert the `n_restarts_optimizer=10` GP achieves median train ΔE00 below a sane threshold (escapes the collapse),
  and that on an easy sample it matches the 0-restart result within tolerance (never worse).
- [ ] Step 2: Run → fails (config not yet changed).
- [ ] Step 3: Set `gaussian_process` factory to `GaussianProcessRegressor(kernel=..., normalize_y=True,
  random_state=SEED, n_restarts_optimizer=10)`.
- [ ] Step 4: Run tests → pass. [ ] Step 5: Commit (no trailer).

### Task 2: Re-run all GP + reconcile
**Files:** update `journal/results/*/summary.csv` (GP rows only, via merge-update run.py), `journal/results/ifra/*`.

- [ ] Step 1: `run --models gaussian_process` on every dataset (PC10/PC11/FOGRA51 CMY+CMYK, IFRA wb within-run,
  and the n>4 sets once Plan 06 lands). merge-update preserves other models' rows.
- [ ] Step 2: Diff GP medians before/after; confirm never-worse on the previously-converged sets (expect ~identical).
- [ ] Step 3: Check the IFRA within-run GP number: if it drops from ~18.8 into family (~1–2), record it; update
  Plan 03's figure/table and the anomaly footnote. If it does NOT resolve, keep the footnote and note n_restarts didn't fix it.
- [ ] Step 4: Commit updated summaries + a short note in the interpretation doc.

### Acceptance
- One GP config everywhere; a documented before/after showing never-worse; IFRA anomaly either resolved (and cleaned up) or explicitly still-open.
