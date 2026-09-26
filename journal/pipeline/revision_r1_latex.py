"""Revision R1: LaTeX fragments for the manuscript, generated from CSVs only.

  python -m journal.pipeline.revision_r1_latex
Reads journal/results/revision_r1/{summary,tables}/ and the published
journal/results/<dataset>/summary.csv; writes
journal/results/revision_r1/tables/tex/*.tex. Nothing here is hand-entered.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / 'results'
R1 = RES / 'revision_r1'
TAB = R1 / 'tables'
TEX = TAB / 'tex'
NINE = ('PC10-CMY', 'PC11-CMY', 'FOGRA51-CMY', 'PC10-CMYK', 'PC11-CMYK',
        'FOGRA51-CMYK', 'KCMYG-5', 'CMYKOGV-7', 'CMYKOGB-7')
SHORT = {'PC10-CMY': 'PC10\\textsubscript{3}', 'PC11-CMY': 'PC11\\textsubscript{3}',
         'FOGRA51-CMY': 'FOGRA51\\textsubscript{3}', 'PC10-CMYK': 'PC10\\textsubscript{4}',
         'PC11-CMYK': 'PC11\\textsubscript{4}', 'FOGRA51-CMYK': 'FOGRA51\\textsubscript{4}',
         'KCMYG-5': 'KCMYG\\textsubscript{5}', 'CMYKOGV-7': 'OGV\\textsubscript{7}',
         'CMYKOGB-7': 'OGB\\textsubscript{7}'}
NAMES = {'poly3': 'Polynomial (3rd order)', 'poly4': 'Polynomial (4th order)',
         'ridge': 'Ridge', 'lasso': 'Lasso', 'elastic': 'Elastic Net', 'pcr': 'PCR',
         'plsr': 'PLSR', 'knn': '$k$-NN', 'svm': 'SVM', 'decision_tree': 'Decision Tree',
         'random_forest': 'Random Forest', 'gradient_boost': 'Gradient Boosting',
         'mlp_shallow': 'MLP (Shallow)', 'mlp_deep': 'MLP (Deep)',
         'gaussian_process': 'Gaussian Process'}
FIXED = ('ridge', 'lasso', 'elastic', 'pcr', 'plsr', 'knn', 'svm', 'decision_tree',
         'random_forest', 'gradient_boost', 'mlp_shallow', 'mlp_deep')


def f3(x):
    return '--' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.3f}'


def pub(ds):
    return pd.read_csv(RES / ds / 'summary.csv').set_index('model')


def r1(ds, m):
    p = R1 / 'summary' / f'{ds}__{m}.csv'
    return pd.read_csv(p).iloc[0] if p.exists() else None


def write(name, text):
    TEX.mkdir(parents=True, exist_ok=True)
    (TEX / f'{name}.tex').write_text(text)
    print(f'--- {name}.tex\n{text}')


def main_table_rows():
    """Rows for Tables 3, 4 and 10: SVR and deep MLP fitted in cube-root space."""
    groups = {'n3': ('PC10-CMY', 'PC11-CMY', 'FOGRA51-CMY'),
              'n4': ('PC10-CMYK', 'PC11-CMYK', 'FOGRA51-CMYK'),
              'n7': ('KCMYG-5', 'CMYKOGV-7', 'CMYKOGB-7')}
    for g, dss in groups.items():
        lines = []
        for m, label in (('svm_cbrt', 'SVM (cbrt)'), ('mlp_deep_cbrt', 'MLP (Deep, cbrt)')):
            cells = []
            for ds in dss:
                r = r1(ds, m)
                cells += [f3(r['median']), f3(r['p95']), f3(r['max'])] if r is not None else ['--'] * 3
            lines.append(f'{label}$^\\ddagger$ & ' + ' & '.join(cells) + ' \\\\')
        write(f'rows_{g}', '\n'.join(lines))


def appendix_cbrt():
    """Table A1: median in cube-root space for every non-GP method (XYZ in the main tables)."""
    hdr = ' & '.join(SHORT[d] for d in NINE)
    lines = [f'\\textbf{{Method}} & {hdr} \\\\', '\\midrule']
    order = ('poly3', 'poly4') + FIXED
    for m in order:
        cells = []
        for ds in NINE:
            if m in ('poly3', 'poly4'):
                p = pub(ds)
                cells.append(f"{f3(p.loc[m, 'median'])} $\\to$ {f3(p.loc[m + '_cbrt', 'median'])}")
            else:
                r = r1(ds, f'{m}_cbrt')
                cells.append(f"{f3(pub(ds).loc[m, 'median'])} $\\to$ {f3(r['median']) if r is not None else '--'}")
        lines.append(f'{NAMES[m]} & ' + ' & '.join(cells) + ' \\\\')
    cells = [f"{f3(pub(ds).loc['gaussian_process', 'median'])} $\\to$ {f3(pub(ds).loc['gaussian_process_cbrt', 'median'])}" for ds in NINE]
    lines += ['\\midrule', 'Gaussian Process & ' + ' & '.join(cells) + ' \\\\']
    write('app_cbrt', '\n'.join(lines))


def appendix_tuned():
    """Table A2: fixed -> tuned median; Table A3: best effort (tuned + cbrt)."""
    hdr = ' & '.join(SHORT[d] for d in NINE)
    lines = [f'\\textbf{{Method}} & {hdr} \\\\', '\\midrule']
    for m in FIXED:
        cells = []
        for ds in NINE:
            r = r1(ds, f'{m}_tuned')
            cells.append(f"{f3(pub(ds).loc[m, 'median'])} $\\to$ {f3(r['median']) if r is not None else '--'}")
        lines.append(f'{NAMES[m]} & ' + ' & '.join(cells) + ' \\\\')
    write('app_tuned', '\n'.join(lines))
    lines = [f'\\textbf{{Method}} & {hdr} \\\\', '\\midrule']
    for m in ('svm', 'mlp_deep', 'gradient_boost', 'random_forest', 'knn'):
        cells = []
        for ds in NINE:
            r = r1(ds, f'{m}_cbrt_tuned')
            cells.append(f3(r['median']) if r is not None else '--')
        lines.append(f'{NAMES[m]} (tuned, cbrt) & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\midrule', 'Gaussian Process (cbrt) & ' + ' & '.join(
        f3(pub(ds).loc['gaussian_process_cbrt', 'median']) for ds in NINE) + ' \\\\']
    write('app_best_effort', '\n'.join(lines))
    # modal hyperparameters per method (for the appendix text)
    rows = []
    for m in FIXED:
        for ds in NINE:
            r = r1(ds, f'{m}_tuned')
            if r is None:
                continue
            ps = [json.dumps(f.get('best_params'), sort_keys=True) for f in json.loads(r['folds'])]
            rows.append({'model': m, 'dataset': ds, 'modal': max(set(ps), key=ps.count),
                         'agree': ps.count(max(set(ps), key=ps.count))})
    if rows:
        pd.DataFrame(rows).to_csv(TAB / 'tuned_modal_params.csv', index=False)


def stats_table():
    p = TAB / 'stats_gp_vs_poly4cbrt.csv'
    if not p.exists():
        return
    df = pd.read_csv(p).set_index('dataset')
    lines = []
    for ds in NINE:
        if ds not in df.index:
            continue
        r = df.loc[ds]
        pv = r['p_holm']
        ptxt = '$<10^{-3}$' if pv < 1e-3 else f'{pv:.3f}'
        lines.append(f"{ds} & {f3(r['gp_median'])} [{f3(r['gp_median_lo'])}, {f3(r['gp_median_hi'])}] & "
                     f"{f3(r['poly_median'])} [{f3(r['poly_median_lo'])}, {f3(r['poly_median_hi'])}] & "
                     f"{f3(r['gp_p95'])} [{f3(r['gp_p95_lo'])}, {f3(r['gp_p95_hi'])}] & "
                     f"{f3(r['poly_p95'])} [{f3(r['poly_p95_lo'])}, {f3(r['poly_p95_hi'])}] & "
                     f"{100 * r['frac_a_better']:.0f}\\% & {ptxt} \\\\")
    write('stats', '\n'.join(lines))


def optim_table():
    p = TAB / 'optim_eqfev.csv'
    if not p.exists():
        return
    df = pd.read_csv(p).set_index('dataset')
    lines = []
    for ds in NINE:
        if ds not in df.index:
            continue
        r = df.loc[ds]
        lines.append(f"{ds} & {int(r['n_coef'])} & {r['nfev_per_fold_mean'] / 1000:.0f}k & "
                     f"{f3(r['powell_median'])} & {f3(r['powell_p95'])} & {f3(r['powell_max'])} & "
                     f"{f3(r['nm_eqfev_median'])} & {f3(r['nm_eqfev_p95'])} & {f3(r['nm_eqfev_max'])} \\\\")
    write('optim', '\n'.join(lines))


def loo_table():
    mm, t = TAB / 'loo_median_of_medians.csv', TAB / 'loo_tests.csv'
    if not mm.exists():
        return
    mm = pd.read_csv(mm).set_index('model')
    t = pd.read_csv(t).set_index('gp_vs') if t.exists() else pd.DataFrame()
    lines = []
    for m, label in (('svm', 'SVM'), ('mlp_deep', 'MLP (Deep)'), ('poly3', 'Polynomial (3rd order)')):
        full, sub = mm.loc[m, 'median_of_medians'], mm.loc[f'{m}_sub2000', 'median_of_medians']
        a = t.loc[m] if m in t.index else None
        b = t.loc[f'{m}_sub2000'] if f'{m}_sub2000' in t.index else None
        fmt = lambda x: '--' if x is None else (f"{int(x['gp_better_runs'])}/{int(x['runs'])}, "
                                                f"$p={x['p_runs_holm']:.2f}$")
        lines.append(f"{label} & {f3(full)} & {fmt(a)} & {f3(sub)} & {fmt(b)} \\\\")
    write('loo', f"GP reference: {f3(mm.loc['gaussian_process', 'median_of_medians'])}\n" + '\n'.join(lines))


def timing_table():
    p = R1 / 'timing.csv'
    if not p.exists():
        return
    df = pd.read_csv(p)
    lab = {'poly3': 'Polynomial (3rd order)', 'poly4': 'Polynomial (4th order)',
           'poly4_cbrt': 'Polynomial (4th order, cbrt)', 'gaussian_process': 'Gaussian Process',
           'gaussian_process_cbrt': 'Gaussian Process (cbrt)', 'svm': 'SVM',
           'random_forest': 'Random Forest', 'mlp_deep': 'MLP (Deep)'}
    lines = [f"{lab[r.model]} & {r.fit_s:.2f} & {r.predict_1M_s:.2f} & {r.us_per_point:.2f} & "
             f"{r.lut_17pow4_cmyk_s:.2f} & {r.lut_9pow7_projected_s:.0f} \\\\" for r in df.itertuples()]
    write('timing', '\n'.join(lines))


if __name__ == '__main__':
    for fn in (main_table_rows, appendix_cbrt, appendix_tuned, stats_table, optim_table, loo_table, timing_table):
        try:
            fn()
        except Exception as e:           # partial data during the run: report, continue
            print(f'!! {fn.__name__}: {type(e).__name__}: {e}')
