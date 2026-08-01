"""Evaluation: 5-fold CV, scalers fit on the training fold only, ΔE00 on
denormalized XYZ. Every test sample is predicted exactly once, so summary
statistics pool over the full dataset rather than one 10% holdout.
"""
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import MinMaxScaler

from .color import delta_e00

SEED = 42
FOLDS = 5


def make_groups(X: np.ndarray) -> np.ndarray:
    """One group id per distinct input recipe (row), so duplicates co-travel."""
    _, inverse = np.unique(np.round(X, 6), axis=0, return_inverse=True)
    return inverse


def summarize(de: np.ndarray) -> dict:
    """Repo reporting rule: median, max, P95 (mean kept for reference only)."""
    de = np.asarray(de, dtype=float)
    return {
        'median': float(np.median(de)),
        'p95': float(np.percentile(de, 95)),
        'max': float(np.max(de)),
        'mean': float(np.mean(de)),
        'n': int(de.size),
    }


def cross_validate(X: np.ndarray, Y: np.ndarray, model_factory, groups=None) -> np.ndarray:
    """Return pooled per-sample ΔE00 over 5-fold CV."""
    de = np.empty(len(X))
    if groups is None:
        splits = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(X)
    else:
        splits = GroupKFold(n_splits=FOLDS).split(X, groups=groups)
    for tr, te in splits:
        sx = MinMaxScaler().fit(X[tr])
        sy = MinMaxScaler().fit(Y[tr])
        model = model_factory()
        if hasattr(model, 'set_scaler'):
            model.set_scaler(sy)
        model.fit(sx.transform(X[tr]), sy.transform(Y[tr]))
        pred_norm = np.asarray(model.predict(sx.transform(X[te])))
        pred_xyz = sy.inverse_transform(pred_norm)       # back to real 0-100 XYZ
        pred_xyz = np.clip(pred_xyz, 0.0, None)          # tristimulus can't be negative
        de[te] = delta_e00(pred_xyz, Y[te])
    return de


def train_test(Xtr, Ytr, Xte, Yte, model_factory) -> np.ndarray:
    """Fit on (Xtr,Ytr), return per-sample DE00 on (Xte,Yte). One fold's logic."""
    sx = MinMaxScaler().fit(Xtr)
    sy = MinMaxScaler().fit(Ytr)
    model = model_factory()
    if hasattr(model, 'set_scaler'):
        model.set_scaler(sy)
    model.fit(sx.transform(Xtr), sy.transform(Ytr))
    pred = sy.inverse_transform(np.asarray(model.predict(sx.transform(Xte))))
    return delta_e00(np.clip(pred, 0.0, None), Yte)
