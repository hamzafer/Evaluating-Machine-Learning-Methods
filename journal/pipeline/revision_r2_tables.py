"""LaTeX tables for the round-2 revision, generated from the round-2 CSVs only.

  python -m journal.pipeline.revision_r2_tables
Writes journal/results/revision_r2/tables/*.tex
"""
from pathlib import Path

import pandas as pd

R2 = Path(__file__).resolve().parents[1] / 'results' / 'revision_r2'
OUT = R2 / 'tables'

NAMES = {
    'poly3': 'Polynomial (3rd order)', 'poly3_cbrt': 'Polynomial (3rd order, cbrt)',
    'poly4': 'Polynomial (4th order)', 'poly4_cbrt': 'Polynomial (4th order, cbrt)',
    'gaussian_process': 'Gaussian Process', 'gaussian_process_cbrt': 'Gaussian Process (cbrt)',
    'svm': 'SVM', 'svm_cbrt_tuned': 'SVM (cbrt, tuned)',
    'random_forest': 'Random Forest', 'random_forest_cbrt_tuned': 'Random Forest (cbrt, tuned)',
    'gradient_boost': 'Gradient Boosting', 'gradient_boost_cbrt_tuned': 'Gradient Boosting (cbrt, tuned)',
    'decision_tree': 'Decision Tree', 'mlp_shallow': 'MLP (Shallow)', 'mlp_deep': 'MLP (Deep)',
    'mlp_deep_cbrt_tuned': 'MLP (Deep, cbrt, tuned)', 'knn': '$k$-NN', 'knn_cbrt_tuned': '$k$-NN (cbrt, tuned)',
    'ridge': 'Ridge', 'lasso': 'Lasso', 'elastic': 'Elastic Net', 'pcr': 'PCR', 'plsr': 'PLSR',
}


def sig3(x):
    """Three significant figures, as the R1 timing table (no scientific notation)."""
    if x < 0.001:
        return '$<$0.001'
    # three significant figures, but never more than the CSV's three decimals
    import math
    decimals = min(3, max(0, 2 - math.floor(math.log10(x))))
    return f'{x:.{decimals}f}'


def thousands(x):
    return f'{int(round(x)):,}'.replace(',', '{,}')


def footprint_rows(models=None):
    f = pd.read_csv(R2 / 'footprint.csv')
    if models:
        f = f.set_index('model').loc[list(models)].reset_index()
    return f


def table_timing_columns():
    """The two footprint columns appended to main-text Table 12 (same 8 rows)."""
    rows = ('poly3', 'poly4', 'poly4_cbrt', 'gaussian_process', 'gaussian_process_cbrt',
            'svm', 'random_forest', 'mlp_deep')
    f = footprint_rows(rows)
    lines = [f"{NAMES[r.model]} & {thousands(r.serialized_kb) if r.serialized_kb >= 100 else f'{r.serialized_kb:.1f}'}"
             f" & {thousands(r.learned_values)} \\\\" for r in f.itertuples()]
    return '\n'.join(lines)


def table_s4():
    f = footprint_rows()
    head = (r'\textbf{Model} & \textbf{Fit (s)} & \textbf{Predict ($\mu$s)} & \textbf{1 thread ($\mu$s)}'
            r' & \textbf{$17^4$ grid (s)} & \textbf{Stored (KB)} & \textbf{Learned values} \\')
    body = []
    for r in f.itertuples():
        kb = thousands(r.serialized_kb) if r.serialized_kb >= 100 else f'{r.serialized_kb:.1f}'
        body.append(f'{NAMES[r.model]} & {sig3(r.fit_s)} & {sig3(r.us_per_point)} & '
                    f'{sig3(r.us_per_point_1t)} & {sig3(r.lut_17pow4_cmyk_s)} & {kb} & '
                    f'{thousands(r.learned_values)} \\\\')
    return head + '\n\\midrule\n' + '\n'.join(body)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'timing_footprint_columns.tex').write_text(table_timing_columns() + '\n')
    (OUT / 'table_s4_footprint.tex').write_text(table_s4() + '\n')
    for p in sorted(OUT.glob('*.tex')):
        print(f'== {p.name}\n{p.read_text()}')


if __name__ == '__main__':
    main()
