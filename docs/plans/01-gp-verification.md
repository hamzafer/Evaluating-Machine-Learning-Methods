# 01 — GP Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish whether the Gaussian Process headline result (median ΔE00 ≈ 0.05–0.07 on all 6 variants) survives duplicate-aware cross-validation, and quantify the measurement noise floor it should be compared against.

**Architecture:** Extend `journal/pipeline/evaluate.py` with a grouped CV variant (duplicate input recipes never straddle train/test), add a noise-floor computation from duplicate patch pairs, and a small runner that writes one verification CSV.

**Tech Stack:** Python (repo `.venv`), scikit-learn `GroupKFold`, existing pipeline modules.

## Global Constraints

- ΔE00 always on denormalized XYZ (repo rule; `journal/pipeline/color.py` owns this).
- Fixed seed 42; report median / P95 / max, 2–3 decimals.
- Anomalous results get investigated before they get reported.

---

### Task 1: Grouped cross-validation in evaluate.py

**Files:**
- Modify: `journal/pipeline/evaluate.py`
- Test: `journal/pipeline/tests/test_evaluate.py` (create; also `touch journal/pipeline/tests/__init__.py`)

**Interfaces:**
- Produces: `cross_validate(X, Y, model_factory, groups=None) -> np.ndarray` — existing signature gains optional `groups`; when given, uses `GroupKFold(5)` so rows with the same group id land in the same fold. `make_groups(X) -> np.ndarray` returns one id per row (identical input rows share an id).

- [ ] **Step 1: Write the failing test**

```python
# journal/pipeline/tests/test_evaluate.py
import numpy as np
from journal.pipeline.evaluate import make_groups, cross_validate


def test_make_groups_identical_rows_share_id():
    X = np.array([[10, 20, 30], [0, 0, 0], [10, 20, 30], [5, 5, 5]])
    g = make_groups(X)
    assert g[0] == g[2]
    assert len({g[0], g[1], g[3]}) == 3


def test_grouped_cv_keeps_duplicates_out_of_train():
    # y = x with a poisoned duplicate: if duplicates split across folds,
    # 1-NN memorizes the twin and scores ~0 error on it.
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 100, size=(80, 3))
    X[40:] = X[:40]                      # every row duplicated once
    Y = X.copy()
    from sklearn.neighbors import KNeighborsRegressor
    de_plain = cross_validate(X, Y, lambda: KNeighborsRegressor(1))
    de_grouped = cross_validate(X, Y, lambda: KNeighborsRegressor(1),
                                groups=make_groups(X))
    assert np.median(de_plain) < 0.01          # memorization
    assert np.median(de_grouped) > np.median(de_plain)  # grouped CV blocks it
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest journal/pipeline/tests/test_evaluate.py -v`
Expected: FAIL with `ImportError: cannot import name 'make_groups'`

- [ ] **Step 3: Implement**

```python
# add to journal/pipeline/evaluate.py
from sklearn.model_selection import GroupKFold


def make_groups(X: np.ndarray) -> np.ndarray:
    """One group id per distinct input recipe (row), so duplicates co-travel."""
    _, inverse = np.unique(np.round(X, 6), axis=0, return_inverse=True)
    return inverse
```

and change `cross_validate` signature/split to:

```python
def cross_validate(X, Y, model_factory, groups=None) -> np.ndarray:
    de = np.empty(len(X))
    if groups is None:
        splits = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(X)
    else:
        splits = GroupKFold(n_splits=FOLDS).split(X, groups=groups)
    for tr, te in splits:
        ...  # body unchanged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest journal/pipeline/tests/test_evaluate.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add journal/pipeline/evaluate.py journal/pipeline/tests/
git commit -m "journal: grouped CV so duplicate recipes never straddle train/test"
```

### Task 2: Noise floor + verification runner

**Files:**
- Create: `journal/pipeline/verify_gp.py`
- Output: `journal/results/gp_verification.csv`

**Interfaces:**
- Consumes: `make_groups`, `cross_validate(..., groups=...)`, `summarize`, dataset/model registries.
- Produces: CSV with columns `dataset, model, cv, median, p95, max, n` plus rows `model=noise_floor` (ΔE00 between duplicate measurement pairs — print/instrument repeatability, the floor no honest model can beat).

- [ ] **Step 1: Write the runner**

```python
# journal/pipeline/verify_gp.py
"""GP headline verification: plain KFold vs GroupKFold + measurement noise floor.
Run: .venv/bin/python -m journal.pipeline.verify_gp
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .color import delta_e00
from .datasets import registry as dreg
from .evaluate import cross_validate, make_groups, summarize
from .models import registry as mreg

MODELS = ['gaussian_process', 'poly3']          # headline + baseline
OUT = Path(__file__).resolve().parents[1] / 'results' / 'gp_verification.csv'


def noise_floor(X, Y):
    """DE00 between measurements that share an identical input recipe."""
    groups = make_groups(X)
    pairs = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        pairs += [delta_e00(Y[i:i+1], Y[j:j+1])[0] for i, j in combinations(idx, 2)]
    return np.array(pairs)


def main():
    rows = []
    for ds_name, spec in dreg().items():
        X, Y = spec.load()
        groups = make_groups(X)
        nf = noise_floor(X, Y)
        if nf.size:
            rows.append({'dataset': ds_name, 'model': 'noise_floor', 'cv': 'pairs',
                         **{k: round(v, 3) for k, v in summarize(nf).items() if k != 'n'},
                         'n': nf.size})
        for m in MODELS:
            for cv, g in [('kfold', None), ('grouped', groups)]:
                s = summarize(cross_validate(X, Y, mreg()[m], groups=g))
                rows.append({'dataset': ds_name, 'model': m, 'cv': cv,
                             **{k: round(v, 3) for k, v in s.items() if k != 'n'},
                             'n': s['n']})
                print(f"{ds_name:14s} {m:16s} {cv:8s} median={s['median']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m journal.pipeline.verify_gp`
Expected: prints 6 datasets × 2 models × 2 CV modes + writes the CSV; runtime < 5 min.

- [ ] **Step 3: Evaluate acceptance criteria (record verdict in this file)**

- GP grouped-CV median within ~2× of plain-CV median on every variant → headline stands; paper claims "at/near the measurement noise floor" using the noise_floor rows.
- If grouped ≫ plain (e.g. >5×): the 0.05 was duplicate leakage → paper reports the grouped numbers instead. Either way the paper reports grouped CV.
- Sanity: noise_floor median expected ~0.1–0.5 ΔE00. If it's 0 or >2, investigate the duplicate detection before reporting anything.

- [ ] **Step 4: Commit**

```bash
git add journal/pipeline/verify_gp.py journal/results/gp_verification.csv
git commit -m "journal: GP verification — grouped CV + measurement noise floor"
```
