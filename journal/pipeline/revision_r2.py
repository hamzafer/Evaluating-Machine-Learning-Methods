"""Revision R2 experiments (MDPI Technologies, technologies-4564760, round 2).

Reviewer 2, point 3: start-point ("seed") sensitivity and convergence of the
direct DE00 refinement of the cubic polynomial (Powell, Nelder-Mead), against
the closed-form least-squares fit it starts from.

One job = one (dataset, fold, start, method) fit, so the 990 jobs run in parallel:

  python -m journal.pipeline.revision_r2 seeds --dataset PC10-CMY --fold 0 --start n05s1 --method powell
  python -m journal.pipeline.revision_r2 seeds-merge

Starts (coefficient vector w0 handed to scipy.optimize.minimize):
  ols      the least-squares solution (what the paper reports; same-platform baseline)
  nXXsK    ols * (1 + XX/100 * N(0,1)), elementwise, noise seed K (XX in 01, 05, 10; K in 0..2)
  zero     all coefficients zero (a naive start with no least-squares information)

Budgets are the paper's: Powell maxiter=200; Nelder-Mead gets at most the objective
evaluations the OLS-start Powell fit used in the same fold in revision R1
(journal/results/revision_r1/optim_folds/<dataset>__fold<i>.json), so every start
of one fold runs Nelder-Mead on the same budget (scipy stops at, in practice exactly at, maxfev).

Outputs under journal/results/revision_r2/seeds/:
  jobs/<dataset>__f<i>__<start>__<method>.csv   idx, de00 (that fold's test samples)
  jobs/<...>.json   nfev, nit, seconds, message, train objective at start/end/OLS, trace
  summary.csv       one row per (dataset, start, method): pooled median/p95/max over 5 folds
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from .color import delta_e00
from .datasets import registry as dataset_registry
from .evaluate import fold_splits, make_groups, summarize

ROOT = Path(__file__).resolve().parents[1] / 'results'
import os
# R2_SEEDS_DIR: the laptop cross-check writes to seeds_laptop/, masters to seeds/
OUT = ROOT / 'revision_r2' / os.environ.get('R2_SEEDS_DIR', 'seeds')
R1_FOLDS = ROOT / 'revision_r1' / 'optim_folds'
NINE = ('PC10-CMY', 'PC11-CMY', 'FOGRA51-CMY', 'PC10-CMYK', 'PC11-CMYK',
        'FOGRA51-CMYK', 'KCMYG-5', 'CMYKOGV-7', 'CMYKOGB-7')
STARTS = ('ols', 'zero') + tuple(f'n{lvl:02d}s{k}' for lvl in (1, 5, 10) for k in range(3))
METHODS = ('powell', 'nm')
N_CHECKPOINTS = 80


def start_vector(w_ols: np.ndarray, start: str) -> np.ndarray:
    if start == 'ols':
        return w_ols.copy()
    if start == 'zero':
        return np.zeros_like(w_ols)
    lvl, k = int(start[1:3]), int(start[4:])
    rng = np.random.default_rng(1000 + k)     # same noise draw for every dataset/fold size
    return w_ols * (1.0 + lvl / 100.0 * rng.standard_normal(w_ols.size))


def fit_one(Xtr, Ytr_n, sy, start, method, nm_budget):
    """Refine a cubic polynomial on mean DE00 from the given start. Returns
    (weights, info). Mirrors DE00Polynomial (journal/pipeline/de00_poly.py)
    except for the start vector and the recorded trace."""
    pf = PolynomialFeatures(degree=3).fit(Xtr)
    Phi = pf.transform(Xtr)
    w_ols = LinearRegression(fit_intercept=False).fit(Phi, Ytr_n).coef_.ravel()
    Ytrue = sy.inverse_transform(np.asarray(Ytr_n, dtype=float))

    def mean_de(w):
        pred = Phi @ w.reshape(3, -1).T
        return float(np.mean(delta_e00(np.clip(sy.inverse_transform(pred), 0.0, None), Ytrue)))

    trace, best, count = [], [np.inf], [0]
    t0 = time.perf_counter()

    def objective(w):
        f = mean_de(w)
        count[0] += 1
        if f < best[0]:
            best[0] = f
        trace.append((count[0], best[0], time.perf_counter() - t0))
        return f

    w0 = start_vector(w_ols, start)
    f_start = mean_de(w0)
    if method == 'powell':
        opts = {'maxiter': 200}
        res = minimize(objective, w0, method='Powell', options=opts)
    else:
        opts = {'maxiter': 10**9, 'xatol': 1e-6, 'fatol': 1e-6, 'maxfev': nm_budget}
        res = minimize(objective, w0, method='Nelder-Mead', options=opts)
    secs = time.perf_counter() - t0
    w = res.x if res.fun <= f_start else w0          # never worse than its own start
    # log-spaced checkpoints of the best-so-far training objective
    tr = np.array(trace)
    idx = np.unique(np.clip(np.round(np.logspace(0, np.log10(len(tr)), N_CHECKPOINTS)) - 1,
                            0, len(tr) - 1).astype(int))
    info = {'nfev': int(res.nfev), 'nit': int(getattr(res, 'nit', -1)),
            'seconds': round(secs, 2), 'message': str(res.message),
            'success': bool(res.success), 'n_coef': int(w0.size),
            'train_obj_ols': round(mean_de(w_ols), 6), 'train_obj_start': round(f_start, 6),
            'train_obj_end': round(mean_de(w), 6),
            'trace': [[int(tr[i, 0]), round(float(tr[i, 1]), 6), round(float(tr[i, 2]), 3)]
                      for i in idx]}
    return pf, w, info


def cmd_seeds(args):
    spec = dataset_registry()[args.dataset]
    X, Y = spec.load()
    groups = make_groups(X) if spec.grouped else None
    tr, te = fold_splits(X, groups)[args.fold]
    sx, sy = MinMaxScaler().fit(X[tr]), MinMaxScaler().fit(Y[tr])
    Xtr, Xte, Ytr = sx.transform(X[tr]), sx.transform(X[te]), sy.transform(Y[tr])
    r1 = json.loads((R1_FOLDS / f'{args.dataset}__fold{args.fold}.json').read_text())
    pf, w, info = fit_one(Xtr, Ytr, sy, args.start, args.method, r1['powell_nfev'])
    pred = np.clip(sy.inverse_transform(pf.transform(Xte) @ w.reshape(3, -1).T), 0.0, None)
    de = delta_e00(pred, Y[te])
    tag = f'{args.dataset}__f{args.fold}__{args.start}__{args.method}'
    (OUT / 'jobs').mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'idx': te, 'de00': np.round(de, 6)}).to_csv(OUT / 'jobs' / f'{tag}.csv', index=False)
    info.update({'dataset': args.dataset, 'fold': args.fold, 'start': args.start,
                 'method': args.method, 'nm_budget': r1['powell_nfev']})
    (OUT / 'jobs' / f'{tag}.json').write_text(json.dumps(info))
    print(f"{tag}: nfev={info['nfev']} {info['seconds']}s train {info['train_obj_start']:.3f}"
          f"->{info['train_obj_end']:.3f} (ols {info['train_obj_ols']:.3f}) "
          f"test median={np.median(de):.3f} [{info['message']}]", flush=True)


def cmd_seeds_merge(args):
    rows, missing = [], []
    for d in NINE:
        n = len(dataset_registry()[d].load()[0])
        for s in STARTS:
            for m in METHODS:
                tags = [f'{d}__f{i}__{s}__{m}' for i in range(5)]
                if not all((OUT / 'jobs' / f'{t}.csv').exists() for t in tags):
                    missing.append(f'{d} {s} {m}')
                    continue
                parts = pd.concat([pd.read_csv(OUT / 'jobs' / f'{t}.csv') for t in tags]).sort_values('idx')
                assert len(parts) == n and (parts.idx.to_numpy() == np.arange(n)).all()
                infos = [json.loads((OUT / 'jobs' / f'{t}.json').read_text()) for t in tags]
                st = summarize(parts.de00.to_numpy())
                rows.append({'dataset': d, 'start': s, 'method': m,
                             **{k: round(v, 3) for k, v in st.items() if k != 'n'}, 'n': st['n'],
                             'seconds_total': round(sum(i['seconds'] for i in infos), 1),
                             'seconds_max_fold': round(max(i['seconds'] for i in infos), 1),
                             'nfev_mean': round(float(np.mean([i['nfev'] for i in infos])), 0),
                             # Nelder-Mead is budget-capped by design, so its scipy 'success' flag is
                             # not a convergence test; reported for Powell only
                             'folds_converged': (sum(i['success'] for i in infos) if m == 'powell' else ''),
                             'train_obj_end_mean': round(float(np.mean([i['train_obj_end'] for i in infos])), 4),
                             'train_obj_ols_mean': round(float(np.mean([i['train_obj_ols'] for i in infos])), 4)})
    pd.DataFrame(rows).to_csv(OUT / 'summary.csv', index=False)
    print(f'wrote {OUT / "summary.csv"} ({len(rows)} rows); missing {len(missing)}')
    for x in missing[:20]:
        print('  missing', x)


PUBLIC_IDS = ('PC10-CMY', 'PC11-CMY', 'FOGRA51-CMY', 'PC10-CMYK', 'PC11-CMYK',
              'FOGRA51-CMYK', 'CMYKOGV-7')     # datasets whose source is publicly available


def cmd_folds(args):
    """Reviewer 2 point 5: publish the exact fold assignment of every analyzed row
    (after the K=0 filter and exact-duplicate removal), so the protocol can be
    checked against a copy of the source data. SAMPLE_ID is given for the public
    datasets only; the restricted ones get the row position alone."""
    out = ROOT / 'revision_r2' / 'folds'
    out.mkdir(parents=True, exist_ok=True)
    for d in NINE:
        spec = dataset_registry()[d]
        X, _ = spec.load()
        df = pd.read_csv(spec.csv)
        if spec.filter_k_zero:
            df = df[df['CMYK_K'] == 0].reset_index(drop=True)
        if spec.dedup_exact:
            cols = list(spec.input_cols) + list(spec.target_cols) + list(spec.lab_cols)
            df = df.drop_duplicates(subset=cols).reset_index(drop=True)
        assert np.array_equal(df.loc[:, list(spec.input_cols)].to_numpy(dtype=float), X)
        groups = make_groups(X) if spec.grouped else None
        fold = np.empty(len(X), dtype=int)
        for i, (_, te) in enumerate(fold_splits(X, groups)):
            fold[te] = i
        rows = {'idx': np.arange(len(X))}
        if d in PUBLIC_IDS:
            rows['SAMPLE_ID'] = df['SAMPLE_ID'].to_numpy()
        rows['fold'] = fold
        if groups is not None and d in PUBLIC_IDS:
            rows['recipe_group'] = groups   # restricted sets: row position and fold only
        pd.DataFrame(rows).to_csv(out / f'{d}.csv', index=False)
        print(f'{d:13s} n={len(X):5d} folds={np.bincount(fold).tolist()} '
              f'{"grouped" if groups is not None else "seeded KFold"}', flush=True)


def cmd_ifra_pairs(args):
    """Reviewer 2 point 2: what the 13 newsprint runs are, and whether the
    cross-run error is already present in the measured data (model-free).
    Only one title has two runs, so the same-title rows are a single pair:
    indicative, not a category statistic. Uses only what the data record:
    the raw file headers (chart, instrument, creation date) and the file names
    (one newspaper title per run, one title, TagesA, present twice).

    Writes journal/results/revision_r2/newsprint/
      runs.csv         run, title, created, instrument, chart, print_conditions
      measured_pairs.csv  model-free: DE00 between the two runs' MEASURED colors, per pair
      model_pairs.csv  condition (B) cross-run DE00 split into same-title vs different-title
    """
    import re
    import zipfile
    from itertools import combinations
    out = ROOT / 'revision_r2' / 'newsprint'
    out.mkdir(parents=True, exist_ok=True)
    raw = ROOT.parent / 'data' / 'raw' / 'Ifra-wb.zip'
    runs = []
    with zipfile.ZipFile(raw) as z:
        for name in sorted(z.namelist()):
            head = z.read(name).decode('latin-1').split('BEGIN_DATA\n')[0]
            field = lambda k: (re.search(rf'^{k}\s+"?([^"\n]*)"?', head, re.M) or [None, ''])[1].strip()
            stem = name.replace('.txt', '')
            # PRINT_CONDITIONS is empty in all 13 headers, so it is not exported
            runs.append({'run': f'IFRA-wb-{stem}-CMYK', 'title': stem.split('_')[0],
                         'created': field('CREATED').split('"')[0].strip(),
                         'instrument': field('INSTRUMENTATION'),
                         'chart': head.splitlines()[0].strip()})
    runs = pd.DataFrame(runs)
    runs.to_csv(out / 'runs.csv', index=False)
    title = dict(zip(runs.run, runs.title))
    # model-free: measured color difference between runs, patch by patch
    reg = dataset_registry()
    meas = {r: reg[r].load() for r in runs.run}
    rows = []
    for a, b in combinations(runs.run, 2):
        (Xa, Ya), (Xb, Yb) = meas[a], meas[b]
        assert np.array_equal(Xa, Xb), 'runs must share the chart layout'
        de = delta_e00(Ya, Yb)
        rows.append({'run_a': a, 'run_b': b, 'same_title': title[a] == title[b],
                     'median': round(float(np.median(de)), 3),
                     'p95': round(float(np.percentile(de, 95)), 3),
                     'max': round(float(np.max(de)), 3)})
    mp = pd.DataFrame(rows)
    mp.to_csv(out / 'measured_pairs.csv', index=False)
    # condition (B) model errors, same-title vs different-title
    cr = pd.read_csv(ROOT / 'ifra' / 'cross_run.csv')
    cr['same_title'] = [title[a] == title[b] for a, b in zip(cr.train, cr.test)]
    summ = []
    for (m, same), g in cr.groupby(['model', 'same_title']):
        summ.append({'model': m, 'same_title': same, 'n_pairs': len(g),
                     'median_of_medians': round(float(g['median'].median()), 3),
                     'min_median': round(float(g['median'].min()), 3),
                     'max_median': round(float(g['median'].max()), 3)})
    for same, g in mp.groupby('same_title'):
        summ.append({'model': 'measured (no model)', 'same_title': same, 'n_pairs': len(g),
                     'median_of_medians': round(float(g['median'].median()), 3),
                     'min_median': round(float(g['median'].min()), 3),
                     'max_median': round(float(g['median'].max()), 3)})
    pd.DataFrame(summ).to_csv(out / 'model_pairs.csv', index=False)
    print(runs.to_string(index=False))
    print(pd.DataFrame(summ).to_string(index=False))


def cmd_optim_cost(args):
    """Reviewer 2 point 3: convergence and compute of the R1 direct-DE00 fits
    (canonical laptop runs, journal/results/revision_r1/optim_folds) next to the
    closed-form least-squares fit they start from (timed here, median of 5)."""
    rows = []
    for d in NINE:
        spec = dataset_registry()[d]
        X, Y = spec.load()
        tr, _ = fold_splits(X, make_groups(X) if spec.grouped else None)[0]
        sx, sy = MinMaxScaler().fit(X[tr]), MinMaxScaler().fit(Y[tr])
        Xtr, Ytr = sx.transform(X[tr]), sy.transform(Y[tr])
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            Phi = PolynomialFeatures(degree=3).fit_transform(Xtr)
            LinearRegression(fit_intercept=False).fit(Phi, Ytr)
            ts.append(time.perf_counter() - t0)
        infos = [json.loads((R1_FOLDS / f'{d}__fold{i}.json').read_text()) for i in range(5)]
        rows.append({'dataset': d, 'n_coef': infos[0]['n_coef'],
                     'ols_fit_ms': round(float(np.median(ts)) * 1e3, 2),
                     'powell_s_median_fold': round(float(np.median([f['powell_seconds'] for f in infos])), 1),
                     'powell_s_max_fold': round(float(max(f['powell_seconds'] for f in infos)), 1),
                     'powell_nfev_median': int(np.median([f['powell_nfev'] for f in infos])),
                     'powell_nit_median': int(np.median([f['powell_nit'] for f in infos])),
                     'powell_converged_folds': sum('successfully' in f['powell_message'] for f in infos),
                     'nm_s_median_fold': round(float(np.median([f['nm_seconds'] for f in infos])), 1),
                     'nm_budget_exhausted_folds': sum('Maximum number' in f['nm_message'] for f in infos),
                     'powell_messages': '; '.join(sorted({f['powell_message'] for f in infos}))})
    out = ROOT / 'revision_r2' / 'optim_cost.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).drop(columns='powell_messages').to_string(index=False))
    print(f'wrote {out}')


def jobs_list(args):
    """Print the full job list (one line per job), heaviest datasets first."""
    order = ('CMYKOGV-7', 'CMYKOGB-7', 'KCMYG-5', 'PC10-CMYK', 'PC11-CMYK', 'FOGRA51-CMYK',
             'PC10-CMY', 'PC11-CMY', 'FOGRA51-CMY')
    for d in order:
        for i in range(5):
            for s in STARTS:
                for m in METHODS:
                    print(f'seeds --dataset {d} --fold {i} --start {s} --method {m}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('seeds')
    p.add_argument('--dataset', required=True); p.add_argument('--fold', type=int, required=True)
    p.add_argument('--start', required=True, choices=STARTS)
    p.add_argument('--method', required=True, choices=METHODS)
    sub.add_parser('seeds-merge')
    sub.add_parser('jobs')
    sub.add_parser('folds')
    sub.add_parser('ifra-pairs')
    sub.add_parser('optim-cost')
    args = ap.parse_args()
    {'seeds': cmd_seeds, 'seeds-merge': cmd_seeds_merge, 'jobs': jobs_list, 'folds': cmd_folds, 'ifra-pairs': cmd_ifra_pairs, 'optim-cost': cmd_optim_cost}[args.cmd](args)


if __name__ == '__main__':
    main()
