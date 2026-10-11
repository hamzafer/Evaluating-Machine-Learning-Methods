"""Revision R3, editor point 1: nested cross-validation of the fitting space
(XYZ or cube root) and, for the polynomial, the degree.

Question: when each model chooses its own configuration from training data
only, does the Gaussian process still lead the polynomial?

Protocol (the paper's own code, folds and seeds):
  - DatasetSpec.load() (K=0 filter for CMY specs, exact dedup where declared).
  - Outer folds: the published fold files (journal/results/revision_r2/folds/),
    asserted equal to evaluate.fold_splits (seeded KFold for the coated sets,
    GroupKFold on recipe groups for the n>4 sets), via revision_r3.published_folds.
  - Per outer fold: MinMax scalers fitted on the outer training fold (as in
    evaluate.cross_validate). Inside that training fold, the selection step of
    models.InnerTuned (3-fold GroupKFold on recipe groups, shuffle, seed 42;
    criterion = median CIEDE2000 on denormalized XYZ) picks
      polynomial: degree {2,3,4,5} x space {xyz, cbrt}   (8 candidates)
      GP:         space {xyz, cbrt}, the paper's kernel   (2 candidates)
    The selected candidate is refitted on the whole outer training fold and
    scored on the outer held-out fold (DE00: XYZ -> Lab D50 -> CIE 2000).
  - Candidates are the paper's model classes: polyN = PolynomialFeatures(N) +
    LinearRegression on scaled XYZ, CubeRootPolynomial(N); GP = models.registry
    'gaussian_process' / 'gaussian_process_cbrt' (FitSubsampled cap 2000).

Every candidate is also fitted on every outer training fold, so the fixed
(non-nested) configurations come out of the same harness and are checked
against the published numbers (controls).

Jobs (each writes under journal/results/revision_r3/nested_cv/parts/):
  python -m journal.pipeline.revision_r3_nested poly --dataset CMYKOGV-7
  python -m journal.pipeline.revision_r3_nested gp   --dataset CMYKOGV-7 --fold 0
  python -m journal.pipeline.revision_r3_nested jobs     # one job per line
  python -m journal.pipeline.revision_r3_nested merge    # per-sample, summary, selections
  python -m journal.pipeline.revision_r3_nested controls # vs published numbers
  python -m journal.pipeline.revision_r3_nested paired   # nested GP vs nested poly
  python -m journal.pipeline.revision_r3_nested latex    # supplementary table fragment
Run jobs with single-threaded BLAS (see README in the output folder).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.preprocessing import MinMaxScaler

from .color import delta_e00
from .datasets import registry as dataset_registry
from .evaluate import make_groups
from .models import InnerTuned, registry as model_registry
from .revision_r1_tables import SEED, boot_ci, holm, paired, published, persample as r1_persample
from .revision_r3 import factory as poly_factory, published_folds

RES = Path(__file__).resolve().parents[1] / 'results'
OUT = RES / 'revision_r3' / 'nested_cv'
PARTS = OUT / 'parts'
NINE = ('PC10-CMY', 'PC11-CMY', 'FOGRA51-CMY', 'PC10-CMYK', 'PC11-CMYK',
        'FOGRA51-CMYK', 'KCMYG-5', 'CMYKOGV-7', 'CMYKOGB-7')
# run order: the n>4 sets and PC10-CMYK first (task priority), then the rest
ORDER = ('CMYKOGV-7', 'KCMYG-5', 'CMYKOGB-7', 'PC10-CMYK', 'PC11-CMYK',
         'FOGRA51-CMYK', 'PC10-CMY', 'PC11-CMY', 'FOGRA51-CMY')
POLY_GRID = {'degree': [2, 3, 4, 5], 'space': ['xyz', 'cbrt']}
GP_GRID = {'space': ['xyz', 'cbrt']}


def poly_build(degree, space):
    return poly_factory(degree, space)()


def gp_build(space):
    return model_registry()['gaussian_process_cbrt' if space == 'cbrt' else 'gaussian_process']()


def tag(params):
    return (f"d{params['degree']}_" if 'degree' in params else 'gp_') + params['space']


class InnerSelect(InnerTuned):
    """models.InnerTuned's selection step without the final refit: same inner
    splits, candidate order, scaler hook and criterion (median DE00 on
    denormalized XYZ, ties to the earlier grid entry). The caller fits every
    candidate on the outer training fold anyway (controls), so the refit of the
    winner is taken from those fits instead of being repeated. For the
    polynomial, `cmd_poly` checks this against InnerTuned itself."""

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        splits = list(GroupKFold(n_splits=self.n_inner, shuffle=True, random_state=self.seed)
                      .split(X, groups=make_groups(X)))
        Ytrue = np.clip(self._scaler.inverse_transform(y), 0.0, None)
        scores = []
        for params in ParameterGrid(self.grid):
            de = np.empty(len(X))
            for tr, te in splits:
                m = self._make(params).fit(X[tr], y[tr])
                pred = np.clip(self._scaler.inverse_transform(np.asarray(m.predict(X[te]))), 0.0, None)
                de[te] = delta_e00(pred, Ytrue[te])
            scores.append((float(np.median(de)), params))
        scores.sort(key=lambda t: t[0])
        self.inner_scores_ = scores
        self.best_params_ = scores[0][1]
        return self


def _load(name):
    spec = dataset_registry()[name]
    X, Y = spec.load()
    groups = make_groups(X) if spec.grouped else None
    splits, fold = published_folds(name, X, groups)
    return X, Y, splits, fold


def _outer(X, Y, tr, te, build, grid, check_inner_tuned=False):
    """One outer fold: fit every candidate on the training fold (held-out DE00
    per candidate), run the inner selection, return both."""
    sx, sy = MinMaxScaler().fit(X[tr]), MinMaxScaler().fit(Y[tr])
    Xtr, Ytr, Xte = sx.transform(X[tr]), sy.transform(Y[tr]), sx.transform(X[te])
    t0 = time.perf_counter()
    de = {}
    for params in ParameterGrid(grid):
        m = build(**params)
        if hasattr(m, 'set_scaler'):
            m.set_scaler(sy)
        m.fit(Xtr, Ytr)
        pred = np.clip(sy.inverse_transform(np.asarray(m.predict(Xte))), 0.0, None)
        de[tag(params)] = delta_e00(pred, Y[te])
    t_fixed = time.perf_counter() - t0
    t0 = time.perf_counter()
    sel = InnerSelect(build, grid)
    sel.set_scaler(sy)
    sel.fit(Xtr, Ytr)
    t_inner = time.perf_counter() - t0
    best = tag(sel.best_params_)
    info = {'selected': best,
            'inner_median': {tag(p): s for s, p in sel.inner_scores_},
            'seconds_fixed': round(t_fixed, 1), 'seconds_inner': round(t_inner, 1)}
    if check_inner_tuned:
        # the unmodified InnerTuned must select the same candidate and its refit
        # must predict exactly what the fixed fit of that candidate predicts
        it = InnerTuned(build, grid)
        it.set_scaler(sy)
        it.fit(Xtr, Ytr)
        pred = np.clip(sy.inverse_transform(np.asarray(it.predict(Xte))), 0.0, None)
        info['innertuned_same_choice'] = tag(it.best_params_) == best
        info['innertuned_max_abs_diff'] = float(np.max(np.abs(delta_e00(pred, Y[te]) - de[best])))
    return de, best, info


def _write_part(stem, te, de, best, info):
    PARTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({'idx': te, **{k: np.round(v, 6) for k, v in de.items()},
                       'nested': np.round(de[best], 6)})
    df.to_csv(PARTS / f'{stem}.csv', index=False)
    (PARTS / f'{stem}.json').write_text(json.dumps(info, indent=1))


def cmd_poly(args):
    X, Y, splits, _ = _load(args.dataset)
    for i, (tr, te) in enumerate(splits):
        de, best, info = _outer(X, Y, tr, te, poly_build, POLY_GRID, check_inner_tuned=True)
        info.update({'dataset': args.dataset, 'family': 'poly', 'fold': i})
        _write_part(f'{args.dataset}__poly__fold{i}', te, de, best, info)
        print(f"{args.dataset} poly fold {i}: selected {best} "
              f"(InnerTuned agrees: {info['innertuned_same_choice']}, "
              f"max|diff| {info['innertuned_max_abs_diff']:.2e})", flush=True)


def cmd_gp(args):
    X, Y, splits, _ = _load(args.dataset)
    tr, te = splits[args.fold]
    de, best, info = _outer(X, Y, tr, te, gp_build, GP_GRID)
    info.update({'dataset': args.dataset, 'family': 'gp', 'fold': args.fold})
    _write_part(f'{args.dataset}__gp__fold{args.fold}', te, de, best, info)
    print(f"{args.dataset} gp fold {args.fold}: selected {best} inner {info['inner_median']} "
          f"[{info['seconds_fixed']}s fixed, {info['seconds_inner']}s inner]", flush=True)


def cmd_jobs(args):
    # slowest first: GP jobs on the largest sets lead, the cheap poly jobs fill in
    gp = [f'gp --dataset {d} --fold {i}' for d in ORDER for i in range(5)]
    poly = [f'poly --dataset {d}' for d in ORDER]
    print('\n'.join(gp[:15] + poly + gp[15:]))


def _family(name, fam, n):
    parts = [pd.read_csv(PARTS / f'{name}__{fam}__fold{i}.csv') for i in range(5)]
    infos = [json.loads((PARTS / f'{name}__{fam}__fold{i}.json').read_text()) for i in range(5)]
    df = pd.concat(parts).sort_values('idx').reset_index(drop=True)
    assert len(df) == n and (df.idx.to_numpy() == np.arange(n)).all(), (name, fam)
    return df, infos


def _stats(v):
    return {'median': float(np.median(v)), 'p95': float(np.percentile(v, 95)), 'max': float(np.max(v))}


def cmd_merge(args):
    rows, sel_rows = [], []
    for name in args.datasets:
        X, _, _, fold = _load(name)
        if not all((PARTS / f'{name}__{f}__fold{i}.csv').exists()
                   for f in ('poly', 'gp') for i in range(5)):
            raise SystemExit(f'incomplete job outputs for {name}: run its poly and gp jobs first')
        pol, pinfo = _family(name, 'poly', len(X))
        gp, ginfo = _family(name, 'gp', len(X))
        ps = pd.DataFrame({'idx': np.arange(len(X)), 'fold': fold,
                           'poly_nested': pol.nested, 'gp_nested': gp.nested})
        for c in pol.columns:
            if c not in ('idx', 'nested'):
                ps[f'poly_{c}'] = pol[c]
        for c in ('gp_xyz', 'gp_cbrt'):
            ps[c] = gp[c]
        (OUT / 'persample').mkdir(parents=True, exist_ok=True)
        ps.to_csv(OUT / 'persample' / f'{name}.csv', index=False)
        for col in ps.columns[2:]:
            rows.append({'dataset': name, 'n': len(X), 'model': col,
                         **{k: round(v, 6) for k, v in _stats(ps[col]).items()}})
        for fam, infos in (('poly', pinfo), ('gp', ginfo)):
            for f in infos:
                sel_rows.append({'dataset': name, 'family': fam, 'fold': f['fold'],
                                 'selected': f['selected'],
                                 **{f'inner_median_{k}': round(v, 6) for k, v in f['inner_median'].items()},
                                 'seconds_fixed': f['seconds_fixed'],
                                 'seconds_inner': f['seconds_inner']})
        print(f"{name}: nested poly {_stats(ps.poly_nested)} | nested GP {_stats(ps.gp_nested)}")
    pd.DataFrame(rows).to_csv(OUT / 'summary.csv', index=False)
    pd.DataFrame(sel_rows).to_csv(OUT / 'selections.csv', index=False)
    print('wrote', OUT / 'summary.csv', OUT / 'selections.csv')


def _ps(name):
    # a requested dataset without merged output is an error, never a silent skip,
    # so a partial run cannot pass the controls or enter the paired table
    p = OUT / 'persample' / f'{name}.csv'
    if not p.exists():
        raise SystemExit(f'no merged per-sample file for {name}: run merge first')
    return pd.read_csv(p)


def cmd_controls(args):
    """(a) fixed configs from this harness vs published <ds>/summary.csv at
    3 decimals (poly3/poly4 xyz+cbrt, GP xyz+cbrt); (b) max |diff| of
    per-sample DE00 vs the R1 per-sample files; (c) degrees 2-5 vs the R3
    degree sweep where it exists; (d) the poly InnerTuned cross-check."""
    sweep = pd.read_csv(RES / 'revision_r3' / 'persample_degree_sweep.csv')
    rows = []
    for name in args.datasets:
        ps = _ps(name)
        pairs = [('poly3', 'poly_d3_xyz'), ('poly3_cbrt', 'poly_d3_cbrt'),
                 ('poly4', 'poly_d4_xyz'), ('poly4_cbrt', 'poly_d4_cbrt'),
                 ('gaussian_process', 'gp_xyz'), ('gaussian_process_cbrt', 'gp_cbrt')]
        for model, col in pairs:
            pub = published(name, model)
            st = _stats(ps[col])
            for k in ('median', 'p95', 'max'):
                rows.append({'dataset': name, 'check': f'summary.csv {model} {k}',
                             'published': round(float(pub[k]), 3), 'this_run': round(st[k], 3),
                             'pass': round(float(pub[k]), 3) == round(st[k], 3)})
            b = r1_persample(name, model)
            d = float(np.max(np.abs(ps[col].to_numpy() - b)))
            rows.append({'dataset': name, 'check': f'R1 persample {model} max|diff|',
                         'published': 0.0, 'this_run': d, 'pass': d <= 1e-6})
        sw = sweep[sweep.dataset == name].sort_values('idx')
        if len(sw):
            for d in (2, 3, 4, 5):
                for sp in ('xyz', 'cbrt'):
                    diff = float(np.max(np.abs(ps[f'poly_d{d}_{sp}'].to_numpy() - sw[f'd{d}_{sp}'].to_numpy())))
                    rows.append({'dataset': name, 'check': f'R3 degree sweep d{d}_{sp} max|diff|',
                                 'published': 0.0, 'this_run': diff, 'pass': diff <= 1e-6})
        for i in range(5):
            f = json.loads((PARTS / f'{name}__poly__fold{i}.json').read_text())
            ok = f['innertuned_same_choice'] and f['innertuned_max_abs_diff'] == 0.0
            rows.append({'dataset': name, 'check': f'fold {i} InnerSelect == InnerTuned (choice, DE00)',
                         'published': 0.0, 'this_run': f['innertuned_max_abs_diff'], 'pass': ok})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'controls.csv', index=False)
    bad = df[~df['pass']]
    print(f"{len(df)} checks, {len(bad)} failed")
    if len(bad):
        print(bad.to_string())
    return 0 if bad.empty else 1


def cmd_paired(args):
    """Same test as the paper's Table 11 (revision_r1_tables.table_stats):
    a = GP, b = polynomial (negative difference = GP lower), bootstrap CIs
    (B=10000, rng seed 42, same draw order), Wilcoxon signed-rank two-sided,
    Holm over the datasets present. Run for nested GP vs nested poly, and for
    the published fixed pair (gp_cbrt vs poly_d4_cbrt) as a check that this
    reproduces stats_gp_vs_poly4cbrt.csv."""
    out = []
    for label, ga, pb in (('nested', 'gp_nested', 'poly_nested'),
                          ('fixed_gp_cbrt_vs_poly4_cbrt', 'gp_cbrt', 'poly_d4_cbrt')):
        rows = []
        for name in args.datasets:
            ps = _ps(name)
            gp, pol = ps[ga].to_numpy(), ps[pb].to_numpy()
            rng = np.random.default_rng(SEED)
            r = {'comparison': label, 'dataset': name, 'n': len(gp)}
            for nm, x in (('gp', gp), ('poly', pol)):
                p95 = lambda v, axis=None: np.percentile(v, 95, axis=axis)
                r[f'{nm}_median'] = float(np.median(x))
                r[f'{nm}_median_lo'], r[f'{nm}_median_hi'] = boot_ci(x, np.median, rng)
                r[f'{nm}_p95'] = float(np.percentile(x, 95))
                r[f'{nm}_p95_lo'], r[f'{nm}_p95_hi'] = boot_ci(x, p95, rng)
                r[f'{nm}_max'] = float(np.max(x))
            r.update(paired(gp, pol, rng))
            rows.append(r)
        df = pd.DataFrame(rows)
        df['p_holm'] = holm(df.p)
        out.append(df)
    df = pd.concat(out, ignore_index=True)
    df.to_csv(OUT / 'paired_nested_gp_vs_poly.csv', index=False)
    # check: the fixed pair must reproduce the paper's Table 11 CSV (all nine only)
    ref = pd.read_csv(RES / 'revision_r1' / 'tables' / 'stats_gp_vs_poly4cbrt.csv').set_index('dataset')
    fx = df[df.comparison == 'fixed_gp_cbrt_vs_poly4_cbrt'].set_index('dataset')
    if len(fx) == len(ref):
        cols = ['gp_median', 'poly_median', 'median_diff', 'median_diff_lo', 'median_diff_hi',
                'frac_a_better', 'wilcoxon_W', 'p', 'p_holm']
        dev = (fx[cols] - ref.loc[fx.index, cols]).abs().max().max()
        print('fixed pair vs stats_gp_vs_poly4cbrt.csv, max abs deviation:', dev)
    cols = ['comparison', 'dataset', 'gp_median', 'poly_median', 'gp_p95', 'poly_p95', 'gp_max',
            'poly_max', 'median_diff', 'median_diff_lo', 'median_diff_hi', 'frac_a_better', 'p_holm']
    print(df[cols].round(4).to_string())


LABEL = {'PC10-CMY': 'PC10-CMY', 'PC11-CMY': 'PC11-CMY', 'FOGRA51-CMY': 'FOGRA51-CMY',
         'PC10-CMYK': 'PC10-CMYK', 'PC11-CMYK': 'PC11-CMYK', 'FOGRA51-CMYK': 'FOGRA51-CMYK',
         'KCMYG-5': 'KCMYG-5', 'CMYKOGV-7': 'CMYKOGV-7', 'CMYKOGB-7': 'CMYKOGB-7'}


def _r(x):
    # half-up rounding to 3 decimals, as the paper's tables do
    from decimal import Decimal, ROUND_HALF_UP
    v = str(Decimal(repr(float(x))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))
    return v.replace('-', '$-$')    # sign kept on -0.000 (an interval end just below 0)


def _choices(sel, name, fam):
    s = sel[(sel.dataset == name) & (sel.family == fam)].sort_values('fold').selected
    short = {'xyz': 'X', 'cbrt': 'C'}
    out = []
    for v in s:
        if v.startswith('gp_'):
            out.append(short[v[3:]])
        else:
            d, sp = v.split('_')
            out.append(d[1:] + short[sp])
    return ' '.join(out)


def cmd_latex(args):
    pr = pd.read_csv(OUT / 'paired_nested_gp_vs_poly.csv')
    pr = pr[pr.comparison == 'nested'].set_index('dataset')
    sel = pd.read_csv(OUT / 'selections.csv')
    lines = [r'\begin{tabular}{@{}lccccccccc@{}}', r'\toprule',
             r' & \multicolumn{3}{c}{\textbf{Nested GP}} & \multicolumn{3}{c}{\textbf{Nested polynomial}} & '
             r'\textbf{Difference} & \multicolumn{2}{c}{\textbf{Choice per outer fold}} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){9-10}',
             r'\textbf{Dataset} & \textbf{Median} & \textbf{P95} & \textbf{Max} & \textbf{Median} & '
             r'\textbf{P95} & \textbf{Max} & \textbf{of medians [95\% CI]} & \textbf{GP} & '
             r'\textbf{Polynomial} \\', r'\midrule']
    for name in NINE:
        if name not in pr.index:
            continue
        r = pr.loc[name]
        lines.append(f"{LABEL[name]} & {_r(r.gp_median)} & {_r(r.gp_p95)} & {_r(r.gp_max)} & "
                     f"{_r(r.poly_median)} & {_r(r.poly_p95)} & {_r(r.poly_max)} & "
                     f"{_r(r.median_diff)} [{_r(r.median_diff_lo)}, {_r(r.median_diff_hi)}] & "
                     f"{_choices(sel, name, 'gp')} & {_choices(sel, name, 'poly')} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    (OUT / 'tables' / 'nested_cv.tex').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('poly'); p.add_argument('--dataset', required=True)
    p = sub.add_parser('gp'); p.add_argument('--dataset', required=True); p.add_argument('--fold', type=int, required=True)
    sub.add_parser('jobs')
    for c in ('merge', 'controls', 'paired', 'latex'):
        p = sub.add_parser(c); p.add_argument('--datasets', nargs='+', default=list(NINE))
    args = ap.parse_args()
    return {'poly': cmd_poly, 'gp': cmd_gp, 'jobs': cmd_jobs, 'merge': cmd_merge,
            'controls': cmd_controls, 'paired': cmd_paired, 'latex': cmd_latex}[args.cmd](args) or 0


if __name__ == '__main__':
    raise SystemExit(main())
