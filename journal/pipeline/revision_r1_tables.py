"""Revision R1: build every new table from the per-sample / summary CSVs written
by revision_r1.py. Tables are generated from CSVs only (repo rule).

  python -m journal.pipeline.revision_r1_tables            # all tables that have data
Writes journal/results/revision_r1/tables/*.csv (+ .tex fragments).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RES = Path(__file__).resolve().parents[1] / 'results'
R1 = RES / 'revision_r1'
OUT = R1 / 'tables'
NINE = ('PC10-CMY', 'PC11-CMY', 'FOGRA51-CMY', 'PC10-CMYK', 'PC11-CMYK',
        'FOGRA51-CMYK', 'KCMYG-5', 'CMYKOGV-7', 'CMYKOGB-7')
B = 10_000
SEED = 42


def persample(ds, model):
    p = R1 / 'persample' / ds / f'{model}.csv'
    return pd.read_csv(p).de00.to_numpy() if p.exists() else None


def summary(ds, model):
    p = R1 / 'summary' / f'{ds}__{model}.csv'
    return pd.read_csv(p).iloc[0] if p.exists() else None


def published(ds, model):
    df = pd.read_csv(RES / ds / 'summary.csv')
    r = df[df.model == model]
    return r.iloc[0] if len(r) else None


def _chunks(n, b=B, size=500):
    done = 0
    while done < b:
        k = min(size, b - done)
        yield k
        done += k


def boot_ci(x, stat, rng, b=B):
    """Percentile bootstrap 95% CI of stat(x), resampling samples with replacement."""
    n = len(x)
    vals = np.concatenate([stat(x[rng.integers(0, n, size=(k, n))], axis=1) for k in _chunks(n, b)])
    return np.percentile(vals, [2.5, 97.5])


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def paired(a, b, rng):
    """a, b: per-sample errors of two models on the same samples."""
    d = a - b
    parts = []
    for k in _chunks(len(d)):
        idx = rng.integers(0, len(d), size=(k, len(d)))
        parts.append(np.median(a[idx], axis=1) - np.median(b[idx], axis=1))
    dmed = np.concatenate(parts)
    w = wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
    return {'median_diff': float(np.median(a) - np.median(b)),
            'median_diff_lo': float(np.percentile(dmed, 2.5)),
            'median_diff_hi': float(np.percentile(dmed, 97.5)),
            'frac_a_better': float(np.mean(d < 0)),
            'wilcoxon_W': float(w.statistic), 'p': float(w.pvalue)}


def table_stats():
    """R1-3: GP vs the corrected polynomial on every dataset."""
    rows = []
    for ds in NINE:
        gp, pol = persample(ds, 'gaussian_process_cbrt'), persample(ds, 'poly4_cbrt')
        if gp is None or pol is None:
            continue
        rng = np.random.default_rng(SEED)
        r = {'dataset': ds, 'n': len(gp)}
        for name, x in (('gp', gp), ('poly', pol)):
            r[f'{name}_median'] = float(np.median(x))
            r[f'{name}_median_lo'], r[f'{name}_median_hi'] = boot_ci(x, np.median, rng)
            p95 = lambda v, axis=None: np.percentile(v, 95, axis=axis)
            r[f'{name}_p95'] = float(np.percentile(x, 95))
            r[f'{name}_p95_lo'], r[f'{name}_p95_hi'] = boot_ci(x, p95, rng)
        r.update(paired(gp, pol, rng))
        # control: per-sample file reproduces the published summary
        pub = published(ds, 'gaussian_process_cbrt')
        r['gp_matches_published'] = bool(pub is not None and round(r['gp_median'], 3) == round(pub['median'], 3))
        pub = published(ds, 'poly4_cbrt')
        r['poly_matches_published'] = bool(pub is not None and round(r['poly_median'], 3) == round(pub['median'], 3))
        rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df['p_holm'] = holm(df.p)
    df.to_csv(OUT / 'stats_gp_vs_poly4cbrt.csv', index=False)
    return df


def loo_ps(held, model):
    p = R1 / 'loo' / 'persample' / f'{held}__{model}.csv'
    return pd.read_csv(p).de00.to_numpy() if p.exists() else None


def table_loo():
    """R1-3 (newsprint) + R2-7 (matched 2,000-row control)."""
    helds = sorted(p.name.split('__')[0] for p in (R1 / 'loo' / 'summary').glob('*__gaussian_process.csv')) \
        if (R1 / 'loo' / 'summary').exists() else []
    if not helds:
        return None
    models = ('gaussian_process', 'svm', 'mlp_deep', 'poly3', 'svm_sub2000', 'mlp_deep_sub2000', 'poly3_sub2000')
    per_run = []
    for h in helds:
        row = {'held_out': h}
        for m in models:
            x = loo_ps(h, m)
            row[m] = float(np.median(x)) if x is not None else np.nan
        per_run.append(row)
    pr = pd.DataFrame(per_run)
    pr.to_csv(OUT / 'loo_per_run_medians.csv', index=False)
    rng = np.random.default_rng(SEED)
    rows = []
    for m in models:
        v = pr[m].to_numpy()
        ok = ~np.isnan(v)
        rows.append({'model': m, 'runs': int(ok.sum()), 'median_of_medians': float(np.median(v[ok])) if ok.any() else np.nan})
    mm = pd.DataFrame(rows)
    # paired tests vs GP: (i) over the 13 run medians, (ii) pooled per sample
    tests = []
    for m in ('svm', 'mlp_deep', 'poly3', 'svm_sub2000', 'mlp_deep_sub2000', 'poly3_sub2000'):
        a, b = pr['gaussian_process'].to_numpy(), pr[m].to_numpy()
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() < 5:
            continue
        w_runs = wilcoxon(a[ok], b[ok])
        ga = np.concatenate([loo_ps(h, 'gaussian_process') for h in pr.held_out[ok]])
        gb = np.concatenate([loo_ps(h, m) for h in pr.held_out[ok]])
        pp = paired(ga, gb, rng)
        tests.append({'gp_vs': m, 'runs': int(ok.sum()),
                      'gp_better_runs': int(np.sum(a[ok] < b[ok])),
                      'p_runs_wilcoxon': float(w_runs.pvalue),
                      'pooled_n': len(ga), 'pooled_median_diff': pp['median_diff'],
                      'pooled_median_diff_lo': pp['median_diff_lo'], 'pooled_median_diff_hi': pp['median_diff_hi'],
                      'pooled_frac_gp_better': pp['frac_a_better'], 'p_pooled_wilcoxon': pp['p']})
    t = pd.DataFrame(tests)
    if len(t):
        t['p_runs_holm'] = holm(t.p_runs_wilcoxon)
        t['p_pooled_holm'] = holm(t.p_pooled_wilcoxon)
    mm.to_csv(OUT / 'loo_median_of_medians.csv', index=False)
    t.to_csv(OUT / 'loo_tests.csv', index=False)
    return mm, t


FIXED = ('ridge', 'lasso', 'elastic', 'pcr', 'plsr', 'knn', 'svm', 'decision_tree',
         'random_forest', 'gradient_boost', 'mlp_shallow', 'mlp_deep')


def table_cbrt_all():
    rows = []
    for ds in NINE:
        for m in ('poly3', 'poly4') + FIXED + ('gaussian_process',):
            base = published(ds, m)
            cb = summary(ds, f'{m}_cbrt') if m in FIXED else published(ds, f'{m}_cbrt')
            if base is None or cb is None:
                continue
            rows.append({'dataset': ds, 'model': m,
                         'xyz_median': base['median'], 'xyz_p95': base['p95'], 'xyz_max': base['max'],
                         'cbrt_median': cb['median'], 'cbrt_p95': cb['p95'], 'cbrt_max': cb['max'],
                         'median_ratio_xyz_over_cbrt': round(base['median'] / cb['median'], 3)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'cbrt_all.csv', index=False)
    return df


def table_tuned():
    rows = []
    for ds in NINE:
        for m in FIXED:
            base, tu = published(ds, m), summary(ds, f'{m}_tuned')
            if base is None or tu is None:
                continue
            folds = json.loads(tu['folds'])
            params = [json.dumps(f.get('best_params'), sort_keys=True) for f in folds]
            modal = max(set(params), key=params.count)
            rows.append({'dataset': ds, 'model': m,
                         'fixed_median': base['median'], 'fixed_p95': base['p95'], 'fixed_max': base['max'],
                         'tuned_median': tu['median'], 'tuned_p95': tu['p95'], 'tuned_max': tu['max'],
                         'modal_params': modal, 'folds_agreeing': params.count(modal),
                         'seconds': tu['seconds']})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'tuned.csv', index=False)
    return df


def table_optim():
    rows = []
    for ds in NINE:
        pw, nm = summary(ds, 'poly3_de00_powell_r1'), summary(ds, 'poly3_de00_nm_eqfev')
        if pw is None or nm is None:
            continue
        folds = json.loads(pw['folds'])
        pub_pw, pub_nm, ls = published(ds, 'poly3_de00_powell'), published(ds, 'poly3_de00_nm'), published(ds, 'poly3')
        rows.append({'dataset': ds, 'n_coef': folds[0]['n_coef'],
                     'nfev_per_fold_mean': float(np.mean([f['powell_nfev'] for f in folds])),
                     'nm_nfev_per_fold_mean': float(np.mean([f['nm_nfev'] for f in folds])),
                     'ls_median': ls['median'], 'ls_p95': ls['p95'], 'ls_max': ls['max'],
                     'powell_median': pw['median'], 'powell_p95': pw['p95'], 'powell_max': pw['max'],
                     'powell_seconds': pw['seconds'],
                     'nm_eqfev_median': nm['median'], 'nm_eqfev_p95': nm['p95'], 'nm_eqfev_max': nm['max'],
                     'nm_eqfev_seconds': nm['seconds'],
                     'nm_published_median': pub_nm['median'], 'nm_published_p95': pub_nm['p95'], 'nm_published_max': pub_nm['max'],
                     'powell_reproduces_published': bool(round(pw['median'], 3) == pub_pw['median']
                                                         and round(pw['max'], 3) == pub_pw['max'])})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'optim_eqfev.csv', index=False)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (('stats', table_stats), ('loo', table_loo), ('cbrt_all', table_cbrt_all),
                     ('tuned', table_tuned), ('optim', table_optim)):
        r = fn()
        print(f'== {name}: ' + ('no data yet' if r is None else 'written'))
        if r is not None:
            with pd.option_context('display.width', 250, 'display.max_columns', 40):
                print(r if not isinstance(r, tuple) else r[0].to_string() + '\n' + r[1].to_string())


if __name__ == '__main__':
    main()
