from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler

from .color_utils import xyz_to_lab, delta_e00, summarize_errors


@dataclass
class RunResult:
    config: str
    mean: float
    median: float
    max: float
    p95: float
    sd: float


def evaluate_model(X: np.ndarray, Y: np.ndarray, model, *, mode: str = 'holdout', repeats: int = 1, folds: int = 5, random_state: int = 42, deltae_mode: str = 'proper') -> pd.DataFrame:
    """Evaluate a model on (X,Y) with normalization and proper ΔE00 on denormalized XYZ.

    Returns: DataFrame of per-run metrics.
    """
    rng = np.random.RandomState(random_state)
    rows: List[Dict] = []

    for rep in range(repeats):
        if mode == 'holdout':
            Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.1, random_state=rng.randint(0, 1_000_000))
            rows.extend(_fit_eval_once(Xtr, Xte, Ytr, Yte, model, deltae_mode=deltae_mode))
        elif mode == 'kfold':
            kf = KFold(n_splits=folds, shuffle=True, random_state=rng.randint(0, 1_000_000))
            for tr_idx, te_idx in kf.split(X):
                rows.extend(_fit_eval_once(X[tr_idx], X[te_idx], Y[tr_idx], Y[te_idx], model, deltae_mode=deltae_mode))
        else:
            raise ValueError("mode must be 'holdout' or 'kfold'")

    return pd.DataFrame(rows)


def _fit_eval_once(Xtr: np.ndarray, Xte: np.ndarray, Ytr: np.ndarray, Yte: np.ndarray, model, *, deltae_mode: str = 'proper') -> List[Dict]:
    # Scale inputs and outputs to [0,1]
    sx = MinMaxScaler().fit(Xtr)
    sy = MinMaxScaler().fit(Ytr)
    Xtr_s, Xte_s = sx.transform(Xtr), sx.transform(Xte)
    Ytr_s = sy.transform(Ytr)

    # Fit
    est = model
    est.fit(Xtr_s, Ytr_s)

    # Predict and inverse transform to real XYZ
    Yhat_s = est.predict(Xte_s)
    if deltae_mode == 'legacy':
        # Legacy mode: compute ΔE00 on scaler-normalized XYZ (to mirror baseline code)
        lab_pred = xyz_to_lab(Yhat_s)
        lab_true = xyz_to_lab(sy.transform(Yte))
    else:
        Yhat = sy.inverse_transform(Yhat_s)
        Ytrue = Yte
        lab_pred = xyz_to_lab(Yhat)
        lab_true = xyz_to_lab(Ytrue)
    de = delta_e00(lab_pred, lab_true)
    stats = summarize_errors(de)
    return [{
        'Configuration': str(est),
        'Mean Error': stats['mean'],
        'Median Error': stats['median'],
        'Max Error': stats['max'],
        'P95 Error': stats['p95'],
        'Std Dev': stats['sd'],
    }]
