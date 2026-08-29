"""Provenance for the Section 4.6 (fitting space) numbers that previously lived
only in docs/research/cube-root-fitting.md: the degree sweep, the transform
sweep, the y^{-4/3} weighting control, and the residual-RMS mechanism table.

Every table in the paper must be readable from a committed results CSV; this
module writes the four that were missing:

  journal/results/CMYKOGV-7/degree_sweep.csv    (tab:degree)
  journal/results/transform_sweep.csv           (tab:transforms)
  journal/results/weighting_control.csv         (the y^-4/3 control paragraph)
  journal/results/residual_rms.csv              (the mechanism paragraph)

Protocol is identical to run.py: DatasetSpec loading (dedup, K=0 filter),
cross_validate (MinMax scalers fit on the training fold, prediction clipped at
zero, dE00 on denormalized XYZ), GroupKFold on the grouped specs.

Weighting-control convention (this WAS the unrecorded detail): each output
channel is fitted separately by weighted least squares with per-row weights
w = clip(y_channel, 1e-6)^(-4/3), i.e. the channel's own first-order
equivalent of fitting cbrt(y_channel). Channel-mean weights give slightly
different numbers (e.g. PC10-CMYK 1.704 instead of 1.551) and a 9-of-9 rather
than 8-of-9 "worse than unweighted" count; the per-channel convention is the
one the paper's numbers come from and the one this script records.

Run with: .venv/bin/python -m journal.pipeline.sweeps
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from .datasets import registry as dataset_registry
from .evaluate import FOLDS, SEED, cross_validate, make_groups
from .color import delta_e00

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', 'results'))

DATASETS = ['PC10-CMY', 'PC11-CMY', 'FOGRA51-CMY',
            'PC10-CMYK', 'PC11-CMYK', 'FOGRA51-CMYK',
            'KCMYG-5', 'CMYKOGV-7', 'CMYKOGB-7']

# name -> (forward, inverse); inverse is applied before clipping at zero.
TRANSFORMS = {
    'identity': (lambda y: y, lambda z: z),
    'sqrt': (lambda y: np.sqrt(np.clip(y, 0.0, None)), lambda z: np.clip(z, 0.0, None) ** 2),
    'cbrt': (lambda y: np.cbrt(np.clip(y, 0.0, None)), lambda z: np.clip(z, 0.0, None) ** 3),
    'pow025': (lambda y: np.clip(y, 0.0, None) ** 0.25, lambda z: np.clip(z, 0.0, None) ** 4),
    'log1p': (lambda y: np.log1p(np.clip(y, 0.0, None)), lambda z: np.expm1(z)),
}


class TransformedTargetPolynomial:
    """OLS polynomial fitted against transform(physical XYZ); prediction is
    inverse-transformed on the way out. transform='cbrt' reproduces
    models.CubeRootPolynomial; 'identity' reproduces plain polyN."""

    def __init__(self, degree=3, transform='identity'):
        self.degree = degree
        self.fwd, self.inv = TRANSFORMS[transform]
        self._scaler = None
        self._model = None

    def set_scaler(self, scaler):
        self._scaler = scaler

    def fit(self, X, y_scaled):
        Y = self._scaler.inverse_transform(y_scaled) if self._scaler is not None else y_scaled
        self._model = make_pipeline(PolynomialFeatures(degree=self.degree), LinearRegression())
        self._model.fit(X, self.fwd(Y))
        return self

    def predict(self, X):
        Y = np.clip(self.inv(self._model.predict(X)), 0.0, None)
        return self._scaler.transform(Y) if self._scaler is not None else Y


class WeightedXYZPolynomial:
    """The explicit control for metric alignment: degree-N OLS on physical XYZ
    with per-row sample weights y^{-4/3}, the first-order equivalent of fitting
    cbrt(y). Each output channel is fitted separately with its own channel's
    weights (see module docstring)."""

    def __init__(self, degree=3, eps=1e-6):
        self.degree, self.eps = degree, eps
        self._scaler = None

    def set_scaler(self, scaler):
        self._scaler = scaler

    def fit(self, X, y_scaled):
        Y = self._scaler.inverse_transform(y_scaled) if self._scaler is not None else y_scaled
        self._pf = PolynomialFeatures(degree=self.degree)
        A = self._pf.fit_transform(X)
        self._coef = np.empty((A.shape[1], Y.shape[1]))
        for c in range(Y.shape[1]):
            sw = np.sqrt(np.clip(Y[:, c], self.eps, None) ** (-4.0 / 3.0))
            self._coef[:, c], *_ = np.linalg.lstsq(A * sw[:, None], Y[:, c] * sw, rcond=None)
        return self

    def predict(self, X):
        Y = np.clip(self._pf.transform(X) @ self._coef, 0.0, None)
        return self._scaler.transform(Y) if self._scaler is not None else Y


def _load(name):
    spec = dataset_registry()[name]
    X, Y = spec.load()
    groups = make_groups(X) if spec.grouped else None
    return X, Y, groups


def degree_sweep():
    """tab:degree — CMYKOGV-7, degrees 2..5, XYZ and cbrt, with the average
    in-fold-training vs held-out median gap for both spaces."""
    X, Y, groups = _load('CMYKOGV-7')
    rows = []
    for degree in (2, 3, 4, 5):
        row = {'dataset': 'CMYKOGV-7', 'degree': degree,
               'terms_per_channel': PolynomialFeatures(degree=degree).fit(X[:1]).n_output_features_}
        for space in ('identity', 'cbrt'):
            de = cross_validate(X, Y, lambda: TransformedTargetPolynomial(degree, space), groups=groups)
            row[f'median_{"xyz" if space == "identity" else "cbrt"}'] = round(float(np.median(de)), 3)
            # train/test gap: median of in-fold-training dE00 vs held-out dE00, mean over folds
            gaps = []
            splits = (GroupKFold(n_splits=FOLDS).split(X, groups=groups) if groups is not None
                      else KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(X))
            for tr, te in splits:
                sx, sy = MinMaxScaler().fit(X[tr]), MinMaxScaler().fit(Y[tr])
                m = TransformedTargetPolynomial(degree, space)
                m.set_scaler(sy)
                m.fit(sx.transform(X[tr]), sy.transform(Y[tr]))

                def de_on(idx):
                    pred = np.clip(sy.inverse_transform(np.asarray(m.predict(sx.transform(X[idx])))), 0.0, None)
                    return delta_e00(pred, Y[idx])
                gaps.append(float(np.median(de_on(te))) - float(np.median(de_on(tr))))
            row[f'gap_{"xyz" if space == "identity" else "cbrt"}'] = round(float(np.mean(gaps)), 3)
        rows.append(row)
        print('degree_sweep:', row)
    out = os.path.join(RESULTS, 'CMYKOGV-7', 'degree_sweep.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out)


def transform_sweep():
    """tab:transforms — degree 3, five target transforms, all nine datasets."""
    rows = []
    for name in DATASETS:
        X, Y, groups = _load(name)
        row = {'dataset': name}
        for t in TRANSFORMS:
            de = cross_validate(X, Y, lambda: TransformedTargetPolynomial(3, t), groups=groups)
            row[t] = round(float(np.median(de)), 3)
        rows.append(row)
        print('transform_sweep:', row)
    out = os.path.join(RESULTS, 'transform_sweep.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out)


def weighting_control():
    """The y^{-4/3} control: weighted degree-3 XYZ fit vs the unweighted one."""
    rows = []
    for name in DATASETS:
        X, Y, groups = _load(name)
        de_plain = cross_validate(X, Y, lambda: TransformedTargetPolynomial(3, 'identity'), groups=groups)
        de_wt = cross_validate(X, Y, lambda: WeightedXYZPolynomial(3), groups=groups)
        rows.append({'dataset': name,
                     'median_unweighted_xyz': round(float(np.median(de_plain)), 3),
                     'median_weighted_xyz': round(float(np.median(de_wt)), 3),
                     'weighted_is_worse': bool(np.median(de_wt) > np.median(de_plain))})
        print('weighting_control:', rows[-1])
    out = os.path.join(RESULTS, 'weighting_control.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out)


def residual_rms():
    """Mechanism table: degree-3 in-sample residual RMS as % of the target's
    standard deviation, per channel, in XYZ and in cbrt space."""
    rows = []
    for name in ('PC10-CMY', 'FOGRA51-CMY', 'CMYKOGV-7'):
        X, Y, _ = _load(name)
        Xs = MinMaxScaler().fit_transform(X)
        for space in ('xyz', 'cbrt'):
            T = Y if space == 'xyz' else np.cbrt(np.clip(Y, 0.0, None))
            model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()).fit(Xs, T)
            resid = T - model.predict(Xs)
            rms = np.sqrt((resid ** 2).mean(axis=0))
            pct = 100.0 * rms / T.std(axis=0)
            rows.append({'dataset': name, 'space': space,
                         'pct_X': round(float(pct[0]), 2),
                         'pct_Y': round(float(pct[1]), 2),
                         'pct_Z': round(float(pct[2]), 2)})
            print('residual_rms:', rows[-1])
    out = os.path.join(RESULTS, 'residual_rms.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out)


if __name__ == '__main__':
    degree_sweep()
    transform_sweep()
    weighting_control()
    residual_rms()
