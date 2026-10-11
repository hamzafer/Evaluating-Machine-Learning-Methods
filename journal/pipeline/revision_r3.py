"""Revision R3 (technologies-4564760): polynomial degree 2..5, XYZ vs cube-root
fitting space, held-out AND in-fold training DE00, on the paper's exact folds.

Extends sweeps.degree_sweep (which wrote CMYKOGV-7/degree_sweep.csv: held-out
median + median gap only) to median/P95/max on both sides, more datasets, and
a paired degree-5-cbrt vs GP-cbrt comparison with the Table-11 test.

Protocol (identical to run.py / revision_r1.py):
  - DatasetSpec.load() (K=0 filter for CMY specs, exact dedup where declared).
  - Folds: the published fold files journal/results/revision_r2/folds/<ds>.csv.
    The script asserts they equal evaluate.fold_splits (seeded KFold, or
    GroupKFold on recipe groups for grouped specs) before using them.
  - Per fold: MinMax scalers fit on the training fold; models are the paper's
    own classes: XYZ = PolynomialFeatures(d)+LinearRegression on scaled XYZ
    (models.registry 'polyN'), cbrt = models.CubeRootPolynomial(d).
  - Prediction inverse-scaled and clipped at 0, DE00 on denormalized XYZ
    (color.delta_e00: XYZ -> Lab D50 -> CIEDE2000).

Statistics per (dataset, degree, space):
  heldout_{median,p95,max}       pooled over all samples (each held out once);
                                 the paper's reporting convention.
  heldout_fold_{stat}            mean over the 5 folds of the stat on that fold's
                                 held-out rows.
  train_{stat}                   mean over the 5 folds of the stat on that fold's
                                 in-fold training rows (model scored on the data
                                 it was fitted to).
  gap_{stat}                     mean over folds of (held-out fold stat - training
                                 fold stat) = heldout_fold_{stat} - train_{stat}.
                                 gap_median is the definition used by
                                 sweeps.degree_sweep / tab:degree.
  terms_per_channel              PolynomialFeatures(d) output size (incl. bias).
  rows_per_coef                  mean training-fold size / terms_per_channel.

  python -m journal.pipeline.revision_r3 sweep    # degree sweep + per-sample
  python -m journal.pipeline.revision_r3 controls # vs published numbers
  python -m journal.pipeline.revision_r3 paired   # deg5 cbrt vs GP cbrt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from .color import delta_e00
from .datasets import registry as dataset_registry
from .evaluate import fold_splits, make_groups
from .models import CubeRootPolynomial
from .revision_r1_tables import SEED, holm, paired, persample as r1_persample

RES = Path(__file__).resolve().parents[1] / 'results'
OUT = RES / 'revision_r3'
DATASETS = ('CMYKOGV-7', 'KCMYG-5', 'CMYKOGB-7', 'PC10-CMYK')
DEGREES = (2, 3, 4, 5)
SPACES = ('xyz', 'cbrt')
STATS = {'median': np.median, 'p95': lambda v: np.percentile(v, 95), 'max': np.max}


def factory(degree, space):
    if space == 'cbrt':
        return lambda: CubeRootPolynomial(degree=degree)
    return lambda: make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())


def published_folds(name, X, groups):
    """Splits from the published fold file, checked against fold_splits."""
    f = pd.read_csv(RES / 'revision_r2' / 'folds' / f'{name}.csv')
    assert len(f) == len(X) and (f.idx.to_numpy() == np.arange(len(X))).all(), name
    fold = f.fold.to_numpy()
    splits = []
    for i, (tr, te) in enumerate(fold_splits(X, groups)):
        te_pub = np.flatnonzero(fold == i)
        assert np.array_equal(np.sort(te), te_pub), f'{name} fold {i} differs from published'
        splits.append((np.flatnonzero(fold != i), te_pub))
    return splits, fold


def run_one(X, Y, splits, degree, space):
    de_pooled = np.empty(len(X))
    per_fold = []
    for tr, te in splits:
        sx, sy = MinMaxScaler().fit(X[tr]), MinMaxScaler().fit(Y[tr])
        model = factory(degree, space)()
        if hasattr(model, 'set_scaler'):
            model.set_scaler(sy)
        model.fit(sx.transform(X[tr]), sy.transform(Y[tr]))

        def de_on(idx):
            pred = sy.inverse_transform(np.asarray(model.predict(sx.transform(X[idx]))))
            return delta_e00(np.clip(pred, 0.0, None), Y[idx])

        de_te, de_tr = de_on(te), de_on(tr)
        de_pooled[te] = de_te
        per_fold.append({'n_train': len(tr),
                         **{f'te_{k}': float(f(de_te)) for k, f in STATS.items()},
                         **{f'tr_{k}': float(f(de_tr)) for k, f in STATS.items()}})
    return de_pooled, pd.DataFrame(per_fold)


def cmd_sweep(args):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, ps_frames, timing = [], [], []
    for name in args.datasets:
        spec = dataset_registry()[name]
        X, Y = spec.load()
        groups = make_groups(X) if spec.grouped else None
        splits, fold = published_folds(name, X, groups)
        ps = pd.DataFrame({'dataset': name, 'idx': np.arange(len(X)), 'fold': fold})
        for degree in DEGREES:
            terms = PolynomialFeatures(degree=degree).fit(X[:1]).n_output_features_
            for space in SPACES:
                t0 = time.perf_counter()
                de, pf = run_one(X, Y, splits, degree, space)
                secs = time.perf_counter() - t0
                r = {'dataset': name, 'n': len(X), 'n_inputs': X.shape[1],
                     'cv': 'GroupKFold' if groups is not None else 'KFold(shuffle,seed=42)',
                     'degree': degree, 'space': space, 'terms_per_channel': terms,
                     'mean_train_rows': float(pf.n_train.mean()),
                     'rows_per_coef': float(pf.n_train.mean() / terms)}
                for k, f in STATS.items():
                    r[f'heldout_{k}'] = float(f(de))
                for k in STATS:
                    r[f'heldout_fold_{k}'] = float(pf[f'te_{k}'].mean())
                for k in STATS:
                    r[f'train_{k}'] = float(pf[f'tr_{k}'].mean())
                for k in STATS:
                    r[f'gap_{k}'] = float((pf[f'te_{k}'] - pf[f'tr_{k}']).mean())
                r['seconds'] = round(secs, 2)
                rows.append(r)
                ps[f'd{degree}_{space}'] = np.round(de, 6)
                print(f"{name:10s} d{degree} {space:4s} held {r['heldout_median']:.3f}/"
                      f"{r['heldout_p95']:.3f}/{r['heldout_max']:.3f} train {r['train_median']:.3f}/"
                      f"{r['train_p95']:.3f}/{r['train_max']:.3f} gap_med {r['gap_median']:+.3f} "
                      f"rows/coef {r['rows_per_coef']:.2f} ({secs:.1f}s)", flush=True)
        ps_frames.append(ps)
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c.startswith(('heldout', 'train_', 'gap_'))]
    df[num] = df[num].round(6)
    df['rows_per_coef'] = df['rows_per_coef'].round(3)
    df.to_csv(OUT / 'degree_sweep_train_vs_heldout.csv', index=False)
    pd.concat(ps_frames, ignore_index=True).to_csv(OUT / 'persample_degree_sweep.csv', index=False)
    print('wrote', OUT / 'degree_sweep_train_vs_heldout.csv')


def cmd_paired(args):
    """Table-11 test: per-sample paired, a = poly5 cbrt, b = GP cbrt (R1 per-sample
    files). Wilcoxon signed-rank (two-sided, zero_method='wilcox'), percentile
    bootstrap 95% CI of median(a)-median(b) (B=10000, rng seed 42), Holm across
    the datasets tested here."""
    ps = pd.read_csv(OUT / 'persample_degree_sweep.csv')
    rows = []
    for name in args.datasets:
        gp = r1_persample(name, 'gaussian_process_cbrt')
        a = ps[ps.dataset == name].sort_values('idx')['d5_cbrt'].to_numpy()
        if gp is None:
            print('no GP per-sample file for', name)
            continue
        assert len(a) == len(gp), name
        rng = np.random.default_rng(SEED)
        r = {'dataset': name, 'n': len(a), 'a': 'poly5_cbrt', 'b': 'gaussian_process_cbrt',
             'a_median': float(np.median(a)), 'b_median': float(np.median(gp))}
        r.update(paired(a, gp, rng))
        r['frac_a_equal'] = float(np.mean(a == gp))
        rows.append(r)
    df = pd.DataFrame(rows)
    df['p_holm'] = holm(df.p)
    df.to_csv(OUT / 'paired_poly5cbrt_vs_gpcbrt.csv', index=False)
    print(df.to_string())


def cmd_controls(args):
    """Controls: rows of this sweep vs (a) published <ds>/summary.csv poly3/poly4
    (xyz + cbrt) at 3 decimals, (b) CMYKOGV-7/degree_sweep.csv (tab:degree),
    (c) max abs difference of per-sample DE00 vs the R1 per-sample files."""
    sw = pd.read_csv(OUT / 'degree_sweep_train_vs_heldout.csv')
    ps = pd.read_csv(OUT / 'persample_degree_sweep.csv')
    rows = []
    for name in args.datasets:
        pub = pd.read_csv(RES / name / 'summary.csv').set_index('model')
        for d in (3, 4):
            for space in SPACES:
                model = f'poly{d}' + ('_cbrt' if space == 'cbrt' else '')
                r = sw[(sw.dataset == name) & (sw.degree == d) & (sw.space == space)].iloc[0]
                for k in STATS:
                    rows.append({'dataset': name, 'check': f'summary.csv {model} {k}',
                                 'published': round(float(pub.loc[model, k]), 3),
                                 'this_run': round(float(r[f'heldout_{k}']), 3)})
                a = ps[ps.dataset == name].sort_values('idx')[f'd{d}_{space}'].to_numpy()
                b = r1_persample(name, model)
                if b is not None:
                    rows.append({'dataset': name, 'check': f'R1 persample {model} max|diff|',
                                 'published': 0.0, 'this_run': round(float(np.max(np.abs(a - b))), 6)})
    if 'CMYKOGV-7' in args.datasets:
        ref = pd.read_csv(RES / 'CMYKOGV-7' / 'degree_sweep.csv')
        for _, q in ref.iterrows():
            for sp in SPACES:
                r = sw[(sw.dataset == 'CMYKOGV-7') & (sw.degree == q.degree) & (sw.space == sp)].iloc[0]
                for col, src in ((f'median_{sp}', 'heldout_median'), (f'gap_{sp}', 'gap_median')):
                    rows.append({'dataset': 'CMYKOGV-7', 'check': f'degree_sweep.csv d{q.degree} {col}',
                                 'published': round(float(q[col]), 3), 'this_run': round(float(r[src]), 3)})
    df = pd.DataFrame(rows)
    df['pass'] = (df.published - df.this_run).abs() < 1e-9
    df.to_csv(OUT / 'controls.csv', index=False)
    print(df.to_string())
    print('ALL PASS' if df['pass'].all() else 'CONTROL FAILURE')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('sweep', 'paired', 'controls'))
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    args = ap.parse_args()
    {'sweep': cmd_sweep, 'paired': cmd_paired, 'controls': cmd_controls}[args.cmd](args)


if __name__ == '__main__':
    main()
