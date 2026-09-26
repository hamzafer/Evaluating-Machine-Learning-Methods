"""Revision R1-5 / R2-4: fit and prediction cost, and the cost of baking a model
into a look-up table (how a characterization model is deployed in a RIP).

Run on an otherwise idle machine (laptop = canonical platform), default BLAS
threading:  .venv/bin/python -m journal.pipeline.revision_r1_timing
Writes journal/results/revision_r1/timing.csv

Protocol: fit on fold 0's training split of CMYKOGV-7 (the largest dataset,
7 inks, 3,302 analyzed rows) exactly as in cross-validation; time predict() on
1,000,000 random ink vectors; fill a real 17^4 = 83,521-node CMYK grid with the
PC10-CMYK model; project a 9^7 = 4,782,969-node seven-ink grid from the
measured per-point rate. Every timing is the median of 3 repeats.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .datasets import registry as dataset_registry
from .evaluate import fold_splits, make_groups
from .models import cbrt_registry, registry as model_registry

OUT = Path(__file__).resolve().parents[1] / 'results' / 'revision_r1' / 'timing.csv'
MODELS = ('poly3', 'poly4', 'poly4_cbrt', 'gaussian_process', 'gaussian_process_cbrt',
          'svm', 'random_forest', 'mlp_deep')
REPEATS = 3


def fit_model(spec_name, m, reg):
    spec = dataset_registry()[spec_name]
    X, Y = spec.load()
    tr, _ = fold_splits(X, make_groups(X) if spec.grouped else None)[0]
    sx, sy = MinMaxScaler().fit(X[tr]), MinMaxScaler().fit(Y[tr])
    model = reg[m]()
    if hasattr(model, 'set_scaler'):
        model.set_scaler(sy)
    t0 = time.perf_counter()
    model.fit(sx.transform(X[tr]), sy.transform(Y[tr]))
    return model, time.perf_counter() - t0, X.shape[1], len(tr)


def timed_predict(model, Z):
    t0 = time.perf_counter()
    model.predict(Z)
    return time.perf_counter() - t0


def main():
    reg = {**model_registry(), **cbrt_registry()}
    rng = np.random.default_rng(42)
    rows = []
    for m in MODELS:
        fits, preds, luts = [], [], []
        for _ in range(REPEATS):
            model, t_fit, n_in, n_tr = fit_model('CMYKOGV-7', m, reg)
            fits.append(t_fit)
            Z = rng.random((1_000_000, n_in))            # inputs in normalized [0,1]
            preds.append(timed_predict(model, Z))
            cmyk, _, _, _ = fit_model('PC10-CMYK', m, reg)
            g = np.linspace(0, 1, 17)
            grid = np.stack(np.meshgrid(g, g, g, g, indexing='ij'), -1).reshape(-1, 4)
            luts.append(timed_predict(cmyk, grid))
        per_point = np.median(preds) / 1_000_000
        rows.append({'model': m, 'dataset': 'CMYKOGV-7', 'n_train': n_tr,
                     'fit_s': round(float(np.median(fits)), 3),
                     'predict_1M_s': round(float(np.median(preds)), 3),
                     'us_per_point': round(per_point * 1e6, 3),
                     'lut_17pow4_cmyk_s': round(float(np.median(luts)), 3),
                     'lut_9pow7_projected_s': round(per_point * 9 ** 7, 1)})
        print(rows[-1], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
