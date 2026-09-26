"""Revision R1 experiments (MDPI Technologies, technologies-4564760).

One driver, one job per (experiment, dataset) so jobs run in parallel. Every
job writes per-sample DE00 CSVs plus a one-row summary CSV; `aggregate` builds
the tables from those files only.

  python -m journal.pipeline.revision_r1 cv    --dataset PC10-CMY --models gaussian_process_cbrt poly4_cbrt
  python -m journal.pipeline.revision_r1 optim --dataset PC10-CMY
  python -m journal.pipeline.revision_r1 loo   --held IFRA-wb-Age_64a_wb-CMYK
  python -m journal.pipeline.revision_r1 timing
  python -m journal.pipeline.revision_r1 aggregate

Outputs under journal/results/revision_r1/:
  persample/<dataset>/<model>.csv   idx, de00   (pooled CV, every sample once)
  summary/<dataset>__<model>.csv    median, p95, max, mean, n, seconds, fold info
  loo/persample/<held>__<model>.csv  and loo/summary/<held>__<model>.csv
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import registry as dataset_registry
from .de00_poly import DE00Polynomial
from .evaluate import cross_validate, fold_splits, make_groups, summarize, train_test
from .models import (FitSubsampled, cbrt_registry, registry as model_registry,
                     tuned_registry)
from .runlog import append as log_run

OUT = Path(__file__).resolve().parents[1] / 'results' / 'revision_r1'
NINE = ('PC10-CMY', 'PC11-CMY', 'FOGRA51-CMY', 'PC10-CMYK', 'PC11-CMYK',
        'FOGRA51-CMYK', 'KCMYG-5', 'CMYKOGV-7', 'CMYKOGB-7')


def all_models(n_inputs: int) -> dict:
    return {**model_registry(), **cbrt_registry(), **tuned_registry(n_inputs)}


def _write_persample(path: Path, de: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'idx': np.arange(len(de)), 'de00': np.round(de, 6)}).to_csv(path, index=False)


def _write_summary(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)


def cmd_cv(args):
    spec = dataset_registry()[args.dataset]
    X, Y = spec.load()
    groups = make_groups(X) if spec.grouped else None
    reg = all_models(X.shape[1])
    for m in args.models:
        folds = []

        def hook(i, model, folds=folds):
            info = {'fold': i}
            if getattr(model, 'best_params_', None) is not None:
                info['best_params'] = {k: (list(v) if isinstance(v, tuple) else v)
                                       for k, v in model.best_params_.items()}
            if hasattr(model, 'nfev_'):
                info['nfev'] = model.nfev_
            folds.append(info)

        t0 = time.time()
        de = cross_validate(X, Y, reg[m], groups=groups, fold_hook=hook)
        secs = time.time() - t0
        s = summarize(de)
        _write_persample(OUT / 'persample' / args.dataset / f'{m}.csv', de)
        _write_summary(OUT / 'summary' / f'{args.dataset}__{m}.csv',
                       {'dataset': args.dataset, 'model': m,
                        **{k: round(v, 3) for k, v in s.items() if k != 'n'}, 'n': s['n'],
                        'seconds': round(secs, 1), 'folds': json.dumps(folds)})
        log_run('revision_r1.py', '5fold-grouped' if spec.grouped else '5fold-kfold',
                args.dataset, m, s, secs, notes='revision R1')
        print(f"{args.dataset:13s} {m:24s} median={s['median']:.3f} p95={s['p95']:.3f} "
              f"max={s['max']:.3f} [{secs:.0f}s]", flush=True)


def cmd_optim(args):
    """R1-4: per fold, Powell (budget as published: maxiter=200), then Nelder-Mead
    given exactly the objective evaluations Powell used in that fold."""
    from sklearn.preprocessing import MinMaxScaler
    from .color import delta_e00
    spec = dataset_registry()[args.dataset]
    X, Y = spec.load()
    groups = make_groups(X) if spec.grouped else None
    de = {'poly3_de00_powell_r1': np.empty(len(X)), 'poly3_de00_nm_eqfev': np.empty(len(X))}
    folds, t_pow, t_nm = [], 0.0, 0.0
    only = getattr(args, 'fold', None)
    for i, (tr, te) in enumerate(fold_splits(X, groups)):
        if only is not None and i != only:
            continue
        sx, sy = MinMaxScaler().fit(X[tr]), MinMaxScaler().fit(Y[tr])
        Xtr, Xte, Ytr = sx.transform(X[tr]), sx.transform(X[te]), sy.transform(Y[tr])
        info = {'fold': i}
        for name, make in (('poly3_de00_powell_r1',
                            lambda: DE00Polynomial(method='Powell', maxiter=200)),
                           ('poly3_de00_nm_eqfev',
                            lambda: DE00Polynomial(method='Nelder-Mead', maxiter=10**9,
                                                   maxfev=info['powell_nfev']))):
            t0 = time.time()
            m = make()
            m.set_scaler(sy)
            m.fit(Xtr, Ytr)
            pred = np.clip(sy.inverse_transform(np.asarray(m.predict(Xte))), 0.0, None)
            de[name][te] = delta_e00(pred, Y[te])
            dt = time.time() - t0
            key = 'powell' if 'powell' in name else 'nm'
            info[f'{key}_nfev'], info[f'{key}_nit'] = m.nfev_, m.nit_
            info[f'{key}_seconds'], info[f'{key}_message'] = round(dt, 1), m.message_
            if key == 'powell':
                t_pow += dt
            else:
                t_nm += dt
        info['n_coef'] = m.n_coef_
        folds.append(info)
        print(f"{args.dataset} fold {i}: powell nfev={info['powell_nfev']} "
              f"({info['powell_seconds']}s)  nm nfev={info['nm_nfev']} ({info['nm_seconds']}s) "
              f"[{info['nm_message']}]", flush=True)
        if only is not None:          # fold job: write the partial and stop
            part = OUT / 'optim_folds' / f'{args.dataset}__fold{i}.csv'
            part.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'idx': te, 'powell': np.round(de['poly3_de00_powell_r1'][te], 6),
                          'nm': np.round(de['poly3_de00_nm_eqfev'][te], 6)}).to_csv(part, index=False)
            (part.with_suffix('.json')).write_text(json.dumps(info))
            return
    for name, secs in (('poly3_de00_powell_r1', t_pow), ('poly3_de00_nm_eqfev', t_nm)):
        s = summarize(de[name])
        _write_persample(OUT / 'persample' / args.dataset / f'{name}.csv', de[name])
        _write_summary(OUT / 'summary' / f'{args.dataset}__{name}.csv',
                       {'dataset': args.dataset, 'model': name,
                        **{k: round(v, 3) for k, v in s.items() if k != 'n'}, 'n': s['n'],
                        'seconds': round(secs, 1), 'folds': json.dumps(folds)})
        log_run('revision_r1.py', 'optim-eqfev', args.dataset, name, s, secs, notes='revision R1-4')
        print(f"{args.dataset:13s} {name:24s} median={s['median']:.3f} p95={s['p95']:.3f} "
              f"max={s['max']:.3f} [{secs:.0f}s]", flush=True)


def cmd_optim_merge(args):
    """Combine the 5 fold jobs of `optim --fold i` into the standard outputs."""
    spec = dataset_registry()[args.dataset]
    X, _ = spec.load()
    parts = [pd.read_csv(OUT / 'optim_folds' / f'{args.dataset}__fold{i}.csv') for i in range(5)]
    infos = [json.loads((OUT / 'optim_folds' / f'{args.dataset}__fold{i}.json').read_text())
             for i in range(5)]
    allp = pd.concat(parts).sort_values('idx')
    assert len(allp) == len(X) and (allp.idx.to_numpy() == np.arange(len(X))).all()
    for name, col, key in (('poly3_de00_powell_r1', 'powell', 'powell'),
                           ('poly3_de00_nm_eqfev', 'nm', 'nm')):
        de = allp[col].to_numpy()
        secs = sum(f[f'{key}_seconds'] for f in infos)
        s = summarize(de)
        _write_persample(OUT / 'persample' / args.dataset / f'{name}.csv', de)
        _write_summary(OUT / 'summary' / f'{args.dataset}__{name}.csv',
                       {'dataset': args.dataset, 'model': name,
                        **{k: round(v, 3) for k, v in s.items() if k != 'n'}, 'n': s['n'],
                        'seconds': round(secs, 1), 'folds': json.dumps(infos)})
        log_run('revision_r1.py', 'optim-eqfev', args.dataset, name, s, secs, notes='revision R1-4')
        print(f"{args.dataset:13s} {name:24s} median={s['median']:.3f} p95={s['p95']:.3f} "
              f"max={s['max']:.3f} [{secs:.0f}s summed over folds]", flush=True)


LOO_MODELS = ('gaussian_process', 'svm', 'mlp_deep', 'poly3')
SUB_MODELS = ('svm', 'mlp_deep', 'poly3')          # R2-7 matched-subsample control


def cmd_loo(args):
    """Newsprint leave-one-run-out with per-sample output (R1-3), plus the R2-7
    control: SVM / deep MLP / poly3 fitted on the SAME 2,000-row subsample the
    GP sees (FitSubsampled, identical cap and seed -> identical rows)."""
    ds = dataset_registry()
    wb = sorted(k for k in ds if k.startswith('IFRA-wb-'))
    runs = {n: ds[n].load() for n in wb}
    held = args.held
    Xtr = np.vstack([runs[n][0] for n in wb if n != held])
    Ytr = np.vstack([runs[n][1] for n in wb if n != held])
    base = model_registry()
    jobs = {m: base[m] for m in LOO_MODELS}
    jobs.update({f'{m}_sub2000': (lambda f=base[m]: FitSubsampled(f(), cap=2000))
                 for m in SUB_MODELS})
    for m in (args.models or list(jobs)):
        t0 = time.time()
        de = train_test(Xtr, Ytr, *runs[held], jobs[m])
        secs = time.time() - t0
        s = summarize(de)
        _write_persample(OUT / 'loo' / 'persample' / f'{held}__{m}.csv', de)
        _write_summary(OUT / 'loo' / 'summary' / f'{held}__{m}.csv',
                       {'held_out': held, 'model': m, 'n_train': len(Xtr),
                        **{k: round(v, 3) for k, v in s.items() if k != 'n'}, 'n': s['n'],
                        'seconds': round(secs, 1)})
        log_run('revision_r1.py', 'ifra-loo', f'held={held}', m, s, secs, notes='revision R1')
        print(f"LOO {held} {m:18s} median={s['median']:.3f} [{secs:.0f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('cv'); p.add_argument('--dataset', required=True); p.add_argument('--models', nargs='+', required=True)
    p = sub.add_parser('optim'); p.add_argument('--dataset', required=True); p.add_argument('--fold', type=int)
    p = sub.add_parser('optim-merge'); p.add_argument('--dataset', required=True)
    p = sub.add_parser('loo'); p.add_argument('--held', required=True); p.add_argument('--models', nargs='*')
    args = ap.parse_args()
    {'cv': cmd_cv, 'optim': cmd_optim, 'optim-merge': cmd_optim_merge, 'loo': cmd_loo}[args.cmd](args)


if __name__ == '__main__':
    main()
