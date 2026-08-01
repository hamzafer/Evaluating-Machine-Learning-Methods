# 05 — Direct ΔE00 Minimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether optimizing model parameters directly on ΔE00 (instead of MSE on XYZ) improves the classical polynomial — and in doing so cover Phil's "other optimizers" item (Nelder-Mead, Powell/SLSQP as the GRG stand-in) in the same experiment.

**Architecture:** Take the 3rd-order polynomial (the paper's classical baseline, 20 terms × 3 outputs = 60 coefficients), initialize from the least-squares solution, then refine the coefficient vector with scipy optimizers whose objective is mean ΔE00 on the training fold. Evaluate with the standard grouped 5-fold CV so numbers are directly comparable to `journal/results/*/summary.csv`.

**Tech Stack:** scipy.optimize (already a sklearn dependency), existing pipeline.

## Global Constraints

- Honest framing (goes in the paper): GP is already at the measurement noise floor, so ΔE00-loss cannot help *it*; the question is whether it improves the *fast, deployable* classical model. Expected gains concentrate in P95/max (ΔE00 weights errors perceptually; MSE on XYZ does not).
- Optimizer budget fixed and reported: `maxiter=2000` Nelder-Mead, `maxiter=200` Powell — runtime per fold must stay < ~2 min.
- Same seeds, same reporting stats.

---

### Task 1: ΔE00-optimized polynomial as a pipeline model

**Files:**
- Create: `journal/pipeline/de00_poly.py`
- Modify: `journal/pipeline/models.py` (register `poly3_de00_nm`, `poly3_de00_powell`)
- Test: `journal/pipeline/tests/test_de00_poly.py`

**Interfaces:**
- Produces: class `DE00Polynomial(method='Nelder-Mead', maxiter=2000)` with sklearn-style `fit(X, Y)` / `predict(X)`; operates in the pipeline's normalized space but computes its training objective in real XYZ via an inverse-transform closure — it receives the fold's `y_scaler` through `set_scaler(sy)` called by a small hook (see Step 3 note).

- [ ] **Step 1: Write the failing test**

```python
# journal/pipeline/tests/test_de00_poly.py
import numpy as np
from journal.pipeline.de00_poly import DE00Polynomial


def test_de00_poly_beats_or_matches_lsq_on_train_objective():
    rng = np.random.RandomState(3)
    X = rng.uniform(0, 1, (300, 3))
    Y = np.stack([20 + 60 * X[:, 0] ** 2, 15 + 70 * X[:, 1], 10 + 50 * X[:, 2] ** 3], axis=1)
    m = DE00Polynomial(maxiter=200)
    m.fit(X, Y)                     # unscaled Y here: scaler defaults to identity
    from journal.pipeline.color import delta_e00
    de_opt = np.mean(delta_e00(m.predict(X), Y))
    de_lsq = np.mean(delta_e00(m.lsq_predict(X), Y))
    assert de_opt <= de_lsq + 1e-9   # refinement never worsens the objective
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError expected.

- [ ] **Step 3: Implement**

```python
# journal/pipeline/de00_poly.py
"""3rd-order polynomial whose coefficients are refined on mean DE00 (scipy)."""
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from .color import delta_e00


class DE00Polynomial:
    def __init__(self, method='Nelder-Mead', maxiter=2000):
        self.method, self.maxiter = method, maxiter
        self._sy = None                      # optional y-scaler (identity if None)

    def set_scaler(self, sy):                # pipeline hook: real-XYZ objective
        self._sy = sy

    def _denorm(self, Yn):
        return self._sy.inverse_transform(Yn) if self._sy is not None else Yn

    def fit(self, X, Y):
        self._pf = PolynomialFeatures(degree=3).fit(X)
        Phi = self._pf.transform(X)
        self._lsq = LinearRegression(fit_intercept=False).fit(Phi, Y)
        w0 = self._lsq.coef_.ravel()
        Ytrue = self._denorm(np.asarray(Y, dtype=float))

        def objective(w):
            pred = Phi @ w.reshape(3, -1).T
            return float(np.mean(delta_e00(
                np.clip(self._denorm(pred), 0.0, None), Ytrue)))

        res = minimize(objective, w0, method=self.method,
                       options={'maxiter': self.maxiter, 'xatol': 1e-6, 'fatol': 1e-6}
                       if self.method == 'Nelder-Mead' else {'maxiter': self.maxiter})
        self._w = res.x if res.fun <= objective(w0) else w0   # never worse than LSQ
        return self

    def predict(self, X):
        return self._pf.transform(X) @ self._w.reshape(3, -1).T

    def lsq_predict(self, X):
        return self._lsq.predict(self._pf.transform(X))
```

Note on the hook: in `evaluate.cross_validate`, after building `sy`, add
`if hasattr(model, 'set_scaler'): model.set_scaler(sy)` — two lines, keeps every
other model untouched.

Registry additions in `models.py`:

```python
        'poly3_de00_nm': lambda: DE00Polynomial(method='Nelder-Mead', maxiter=2000),
        'poly3_de00_powell': lambda: DE00Polynomial(method='Powell', maxiter=200),
```

- [ ] **Step 4: Run test to green.** **Step 5: Commit.**

```bash
git add journal/pipeline/de00_poly.py journal/pipeline/models.py journal/pipeline/evaluate.py journal/pipeline/tests/test_de00_poly.py
git commit -m "journal: DE00-objective polynomial (Nelder-Mead / Powell refinement)"
```

### Task 2: Run and compare

- [ ] Run on the three CMY variants first (60 coeffs; CMYK has 105 — run after if time permits):
  `.venv/bin/python -m journal.pipeline.run --datasets PC10-CMY PC11-CMY FOGRA51-CMY --models poly3 poly3_de00_nm poly3_de00_powell`
- [ ] Acceptance: `poly3_de00_*` median ≤ `poly3` median and (the real question) P95/max reduced. If the optimizer makes things worse on held-out data (overfit to train-fold ΔE00), that is itself a reportable finding — do not hide it.
- [ ] Commit updated summaries: `git add journal/results && git commit -m "journal: DE00-loss polynomial results"`
