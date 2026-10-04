"""Revision R2, Reviewer 2 point 1: memory footprint and inference latency of
every model family, and whether a degree-4 polynomial suits an embedded or
real-time raster image processor (RIP).

Extends the R1 timing protocol (revision_r1_timing.py) to every model family
and adds two footprint measures and a single-thread latency column:

  serialized_kb       size of the fitted predictor pickled as scikit-learn stores it
                      (tuning/factory closures stripped; GP keeps its Cholesky factor)
  learned_values      numbers a deployment must store to predict the mean
                      (coefficients, support vectors and dual weights, tree nodes,
                      network weights, GP training inputs and weights, ...)
  us_per_point_1t     prediction cost per input with BLAS/OpenMP limited to one thread

Run on an otherwise idle laptop (canonical platform):
  .venv/bin/python -m journal.pipeline.revision_r2_footprint            # all models
  .venv/bin/python -m journal.pipeline.revision_r2_footprint --embedded # poly4 float32/float64 only
Writes journal/results/revision_r2/footprint.csv and poly4_embedded.csv.
"""
import argparse
import copy
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

from .color import delta_e00
from .datasets import registry as dataset_registry
from .evaluate import fold_splits, make_groups
from .models import cbrt_registry, cbrt_tuned_registry, registry as model_registry

OUT = Path(__file__).resolve().parents[1] / 'results' / 'revision_r2'
MODELS = ('poly3', 'poly3_cbrt', 'poly4', 'poly4_cbrt',
          'gaussian_process', 'gaussian_process_cbrt',
          'svm', 'svm_cbrt_tuned', 'random_forest', 'random_forest_cbrt_tuned',
          'gradient_boost', 'gradient_boost_cbrt_tuned', 'decision_tree',
          'mlp_shallow', 'mlp_deep', 'mlp_deep_cbrt_tuned', 'knn', 'knn_cbrt_tuned',
          'ridge', 'lasso', 'elastic', 'pcr', 'plsr')
REPEATS = 3


def fit_model(spec_name, m, reg):
    spec = dataset_registry()[spec_name]
    X, Y = spec.load()
    tr, te = fold_splits(X, make_groups(X) if spec.grouped else None)[0]
    sx, sy = MinMaxScaler().fit(X[tr]), MinMaxScaler().fit(Y[tr])
    model = reg[m]()
    if hasattr(model, 'set_scaler'):
        model.set_scaler(sy)
    t0 = time.perf_counter()
    model.fit(sx.transform(X[tr]), sy.transform(Y[tr]))
    return model, time.perf_counter() - t0, X.shape[1], len(tr), (sx, sy, X[te], Y[te])


def timed_predict(model, Z, chunk=100_000):
    t0 = time.perf_counter()
    for i in range(0, len(Z), chunk):
        model.predict(Z[i:i + chunk])
    return time.perf_counter() - t0


# ---- footprint ------------------------------------------------------------

def _strip(obj, seen=None):
    """Deep-copied predictor without the (unpicklable, non-deployed) factory
    closures, tuning grids and tuning score tables."""
    obj = copy.deepcopy(obj) if seen is None else obj
    seen = set() if seen is None else seen
    if id(obj) in seen or not hasattr(obj, '__dict__'):
        return obj
    seen.add(id(obj))
    for k in ('factory', 'build', 'grid', 'inner_scores_'):
        if k in obj.__dict__:
            obj.__dict__[k] = None
    def walk(v):
        if hasattr(v, '__dict__'):
            _strip(v, seen)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
    for v in list(obj.__dict__.values()):
        walk(v)
    return obj


def serialized_kb(model) -> float:
    return len(pickle.dumps(_strip(model), protocol=5)) / 1024


def _tree_values(t) -> int:
    # per node: feature, threshold, left, right (internal) + output values (leaves)
    tr = t.tree_
    leaves = int((tr.children_left == -1).sum())
    return 4 * (tr.node_count - leaves) + leaves * int(np.prod(tr.value.shape[1:]))


def learned_values(model) -> int:
    """Numbers needed to predict the mean, per estimator family (recursive)."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    m = model
    if isinstance(m, Pipeline):
        steps = [s for _, s in m.steps]
        if (len(steps) == 2 and isinstance(steps[0], PolynomialFeatures) and steps[0].include_bias
                and isinstance(steps[1], LinearRegression)):
            # the bias column's coefficient and the intercept are one number per channel:
            # the deployable polynomial is the n_terms x 3 coefficient matrix
            return int(steps[1].coef_.size)
        return sum(learned_values(s) for s in steps)
    if isinstance(m, PolynomialFeatures):
        return 0                                   # exponents are implied by the degree
    if isinstance(m, (LinearRegression, Ridge, Lasso, ElasticNet)):
        return int(np.size(m.coef_) + np.size(m.intercept_))
    if isinstance(m, PCA):
        return int(m.components_.size + m.mean_.size)
    if isinstance(m, PLSRegression):
        return int(m.coef_.size + m.intercept_.size + m._x_mean.size + m._x_std.size)
    if isinstance(m, KNeighborsRegressor):
        return int(m._fit_X.size + np.size(m._y))
    if isinstance(m, MultiOutputRegressor):
        return sum(learned_values(e) for e in m.estimators_)
    if isinstance(m, SVR):
        return int(m.support_vectors_.size + m.dual_coef_.size + m.intercept_.size)
    if isinstance(m, DecisionTreeRegressor):
        return _tree_values(m)
    if isinstance(m, RandomForestRegressor):
        return sum(_tree_values(t) for t in m.estimators_)
    if isinstance(m, GradientBoostingRegressor):
        return sum(_tree_values(t) for t in m.estimators_.ravel()) + 1
    if isinstance(m, MLPRegressor):
        return int(sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_))
    if isinstance(m, GaussianProcessRegressor):
        extra = 2 * m._y_train_mean.size if m.normalize_y else 0
        return int(m.X_train_.size + m.alpha_.size + m.kernel_.theta.size + extra)
    # repo wrappers
    for attr in ('_model', '_inner', 'estimator'):
        if getattr(m, attr, None) is not None and attr in m.__dict__:
            # input/output scalers are not counted for any model (like-for-like)
            return learned_values(m.__dict__[attr])
    raise TypeError(f'no footprint rule for {type(m).__name__}')


def cmd_all(args):
    reg = {**model_registry(), **cbrt_registry(), **cbrt_tuned_registry(7)}
    reg4 = {**model_registry(), **cbrt_registry(), **cbrt_tuned_registry(4)}
    rng = np.random.default_rng(42)
    rows = []
    for m in args.models or MODELS:
        fits, preds, preds1, luts = [], [], [], []
        for _ in range(REPEATS):
            model, t_fit, n_in, n_tr, _ = fit_model('CMYKOGV-7', m, reg)
            fits.append(t_fit)
            Z = rng.random((1_000_000, n_in))
            preds.append(timed_predict(model, Z))
            with threadpool_limits(1):
                preds1.append(timed_predict(model, Z[:200_000]) * 5)
            cmyk, _, _, _, _ = fit_model('PC10-CMYK', m, reg4)
            g = np.linspace(0, 1, 17)
            grid = np.stack(np.meshgrid(g, g, g, g, indexing='ij'), -1).reshape(-1, 4)
            luts.append(timed_predict(cmyk, grid))
        per_point = np.median(preds) / 1_000_000
        rows.append({'model': m, 'dataset': 'CMYKOGV-7', 'n_train': n_tr,
                     'fit_s': round(float(np.median(fits)), 3),
                     'predict_1M_s': round(float(np.median(preds)), 3),
                     'us_per_point': round(per_point * 1e6, 3),
                     'us_per_point_1t': round(float(np.median(preds1)), 3),
                     'lut_17pow4_cmyk_s': round(float(np.median(luts)), 3),
                     'lut_9pow7_projected_s': round(per_point * 9 ** 7, 1),
                     'serialized_kb': round(serialized_kb(model), 1),
                     'learned_values': learned_values(model)})
        print(rows[-1], flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT / ('footprint.csv' if not args.models else
                                         f'footprint_{"_".join(args.models)}.csv'), index=False)


def cmd_embedded(args):
    """poly4 on CMYKOGV-7 fold 0 as a bare coefficient matrix: prediction =
    monomials(x) @ W. float64 vs float32 cost (one thread) and accuracy."""
    from sklearn.preprocessing import PolynomialFeatures
    reg = model_registry()
    rows = []
    for m in ('poly4', 'poly4_cbrt'):
        model, _, n_in, _, (sx, sy, Xte, Yte) = fit_model('CMYKOGV-7', m, reg)
        pipe = model if m == 'poly4' else model._model
        pf, lr = pipe.steps[0][1], pipe.steps[1][1]
        W = np.vstack([lr.intercept_[None, :], lr.coef_[:, 1:].T]).astype(np.float64)
        # PolynomialFeatures' first column is the bias; merge it with the intercept
        W[0] += lr.coef_[:, 0]
        pw = pf.powers_[1:]
        # incremental monomials, as an embedded implementation would compute them:
        # every term is an earlier term times one ink value (one multiply per term)
        index = {tuple(p): k for k, p in enumerate(pf.powers_)}
        steps = []
        for k, p in enumerate(pf.powers_[1:], start=1):
            j = int(np.flatnonzero(p)[0])
            q = p.copy(); q[j] -= 1
            steps.append((k, index[tuple(q)], j))
        Zte = sx.transform(Xte)
        rng = np.random.default_rng(7)
        Z = rng.random((1_000_000, n_in))
        for dt in (np.float64, np.float32):
            Wd, Zd = W.astype(dt), Z.astype(dt)

            def predict(Zb, Wd=Wd):
                F = np.empty((len(pw) + 1, len(Zb)), dtype=Wd.dtype)
                F[0] = 1.0
                Zt = Zb.T
                for k, parent, j in steps:
                    np.multiply(F[parent], Zt[j], out=F[k])
                return (Wd.T @ F).T

            with threadpool_limits(1):
                ts = []
                for _ in range(REPEATS):
                    t0 = time.perf_counter()
                    for i in range(0, len(Zd), 20_000):
                        predict(Zd[i:i + 20_000])
                    ts.append(time.perf_counter() - t0)
            out = predict(Zte.astype(dt)).astype(np.float64)
            if m == 'poly4_cbrt':
                xyz = np.clip(out, 0.0, None) ** 3
            else:
                xyz = sy.inverse_transform(out)
            de = delta_e00(np.clip(xyz, 0.0, None), Yte)
            ref = delta_e00(np.clip(sy.inverse_transform(np.asarray(model.predict(Zte))), 0.0, None), Yte)
            n_mono = len(pw) + 1
            rows.append({'model': m, 'dtype': np.dtype(dt).name, 'n_terms_per_channel': n_mono,
                         'coef_values': int(W.size), 'coef_kb': round(W.size * np.dtype(dt).itemsize / 1024, 2),
                         'us_per_point_1t': round(float(np.median(ts)), 3),
                         'mult_add_per_point': int(3 * n_mono + (n_mono - 1)),
                         'fold0_test_median': round(float(np.median(de)), 4),
                         'fold0_test_max': round(float(np.max(de)), 4),
                         'max_abs_de_diff_vs_sklearn': float(np.max(np.abs(de - ref)))})
            print(rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'poly4_embedded.csv', index=False)


def cmd_recount(args):
    """Refit each model once and rewrite only the learned_values column of
    footprint.csv (a counting rule changed; timings are left as measured)."""
    reg = {**model_registry(), **cbrt_registry(), **cbrt_tuned_registry(7)}
    path = OUT / 'footprint.csv'
    f = pd.read_csv(path)
    for i, m in enumerate(f.model):
        model, *_ = fit_model('CMYKOGV-7', m, reg)
        f.loc[i, 'learned_values'] = learned_values(model)
        print(m, int(f.loc[i, 'learned_values']), flush=True)
    f.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embedded', action='store_true')
    ap.add_argument('--recount', action='store_true')
    ap.add_argument('--models', nargs='*')
    args = ap.parse_args()
    cmd_embedded(args) if args.embedded else cmd_recount(args) if args.recount else cmd_all(args)


if __name__ == '__main__':
    main()
