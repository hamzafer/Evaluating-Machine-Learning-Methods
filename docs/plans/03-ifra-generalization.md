# 03 — IFRA Cross-Run Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer Phil's "larger/combined datasets" question with three experiments on the IFRA press runs: (A) within-run accuracy, (B) cross-run transfer (train on one press run, test on another), (C) leave-one-run-out combined training.

**Architecture:** One runner script over the registered `IFRA-*` datasets. Experiments B and C need train-on-X/test-on-Y evaluation, which the CV-oriented pipeline doesn't have — add one function `train_test(Xtr, Ytr, Xte, Yte, factory)` to `evaluate.py`.

**Tech Stack:** existing pipeline; depends on plan 02 being complete.

> wb-only: bb was quarantined (invalid CMYK join — chart-layout mismatch, commit 7719a02) and then descoped (out of scope, 11 Aug). Not an appendix/time item — bb is dropped.

## Global Constraints

- wb runs only for the headline experiments (industry norm, per Phil); repeat for bb only if time allows (appendix).
- Model subset to keep runtime sane: `gaussian_process, poly3, svm, mlp_deep` (winner + classical baseline + two mid-field representatives). Full 14-model sweep only for experiment A.
- Same reporting: median/P95/max, pooled per-sample ΔE00.

---

### Task 1: `train_test` in evaluate.py

**Files:**
- Modify: `journal/pipeline/evaluate.py`
- Test: `journal/pipeline/tests/test_evaluate.py` (append)

**Interfaces:**
- Produces: `train_test(Xtr, Ytr, Xte, Yte, model_factory) -> np.ndarray` — fit scalers on the training set only, fit model, return per-sample ΔE00 on the test set (denormalized, clipped at 0), mirroring one fold of `cross_validate`.

- [ ] **Step 1: Write the failing test**

```python
def test_train_test_perfect_linear_map():
    rng = np.random.RandomState(1)
    Xtr, Xte = rng.uniform(0, 100, (200, 3)), rng.uniform(10, 90, (50, 3))
    from journal.pipeline.evaluate import train_test
    from sklearn.linear_model import LinearRegression
    de = train_test(Xtr, Xtr * 0.9, Xte, Xte * 0.9, LinearRegression)
    assert de.shape == (50,) and np.median(de) < 0.05   # exactly learnable map
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest journal/pipeline/tests/test_evaluate.py -v` → `ImportError: train_test`

- [ ] **Step 3: Implement**

```python
def train_test(Xtr, Ytr, Xte, Yte, model_factory) -> np.ndarray:
    """Fit on (Xtr,Ytr), return per-sample DE00 on (Xte,Yte). One fold's logic."""
    sx, sy = MinMaxScaler().fit(Xtr), MinMaxScaler().fit(Ytr)
    model = model_factory()
    model.fit(sx.transform(Xtr), sy.transform(Ytr))
    pred = sy.inverse_transform(np.asarray(model.predict(sx.transform(Xte))))
    return delta_e00(np.clip(pred, 0.0, None), Yte)
```

- [ ] **Step 4: Run to verify it passes**, then **Step 5: Commit**

```bash
git add journal/pipeline/evaluate.py journal/pipeline/tests/test_evaluate.py
git commit -m "journal: train_test evaluation for cross-dataset experiments"
```

### Task 2: The three experiments

**Files:**
- Create: `journal/pipeline/run_ifra.py`
- Output: `journal/results/ifra/{within_run,cross_run,leave_one_out}.csv`

- [ ] **Step 1: Write the runner**

```python
# journal/pipeline/run_ifra.py
"""IFRA generalization experiments (wb runs). Run:
   .venv/bin/python -m journal.pipeline.run_ifra
A: within-run 5-fold CV, all 14 models, every wb run.
B: cross-run pairs — train run i, test run j (i != j), model subset.
C: leave-one-run-out — train on all wb runs but one, test on the held-out run.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import registry as dreg
from .evaluate import cross_validate, summarize, train_test
from .models import registry as mreg

SUBSET = ['gaussian_process', 'poly3', 'svm', 'mlp_deep']
OUT = Path(__file__).resolve().parents[1] / 'results' / 'ifra'


def wb_runs():
    return {k: v for k, v in dreg().items() if k.startswith('IFRA-wb-')}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {name: spec.load() for name, spec in wb_runs().items()}
    names = sorted(runs)

    # A: within-run, all models
    rows = [{'run': n, 'model': m,
             **{k: round(v, 3) for k, v in
                summarize(cross_validate(*runs[n], mreg()[m])).items()}}
            for n in names for m in mreg()]
    pd.DataFrame(rows).to_csv(OUT / 'within_run.csv', index=False)

    # B: cross-run transfer, subset
    rows = [{'train': a, 'test': b, 'model': m,
             **{k: round(v, 3) for k, v in
                summarize(train_test(*runs[a], *runs[b], mreg()[m])).items()}}
            for a in names for b in names if a != b for m in SUBSET]
    pd.DataFrame(rows).to_csv(OUT / 'cross_run.csv', index=False)

    # C: leave-one-run-out, subset
    rows = []
    for held in names:
        Xtr = np.vstack([runs[n][0] for n in names if n != held])
        Ytr = np.vstack([runs[n][1] for n in names if n != held])
        for m in SUBSET:
            s = summarize(train_test(Xtr, Ytr, *runs[held], mreg()[m]))
            rows.append({'held_out': held, 'model': m,
                         **{k: round(v, 3) for k, v in s.items()}})
            print(f"LOO {held} {m}: median={s['median']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'leave_one_out.csv', index=False)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run it** (background; B is 13×12 pairs × 4 models — GP fit per pair dominates; expect ~30–60 min)

Run: `.venv/bin/python -m journal.pipeline.run_ifra`
Expected: three CSVs. If runtime explodes, cut experiment B's GP to a 5-run subsample and note it in the CSV filename.

- [ ] **Step 3: Acceptance criteria / anomaly gate**

- Within-run medians should be worse than PC10 (newsprint is noisier) but the ranking should resemble the coated-paper story; if a model's median is >2× out of family across runs, investigate before reporting.
- Expected narrative signal: cross-run ≫ within-run error (printer-to-printer variation dominates model choice) and LOO combined training landing between the two. Whatever the numbers, they answer Phil's question directly.

- [ ] **Step 4: Commit**

```bash
git add journal/pipeline/run_ifra.py journal/results/ifra/
git commit -m "journal: IFRA within-run, cross-run, and leave-one-out results"
```
