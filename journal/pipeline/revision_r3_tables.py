"""Revision R3: LaTeX tables for the degree sweep, generated from the CSV only.

  python -m journal.pipeline.revision_r3_tables

Reads journal/results/revision_r3/degree_sweep_train_vs_heldout.csv (written by
journal.pipeline.revision_r3 sweep) and, for the Gaussian process reference row,
journal/results/<dataset>/summary.csv. Writes

  journal/results/revision_r3/tables/tab_degree.tex                 (main text, tab:degree)
  journal/results/revision_r3/tables/tab_s_degree_train_heldout.tex (Supplementary table)

Every number is read from a CSV cell as a string and rounded half up with
Decimal, as the paper's tables are; nothing is typed by hand.
"""
import csv
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / 'results'
R3 = RES / 'revision_r3'
SWEEP = R3 / 'degree_sweep_train_vs_heldout.csv'
OUT = R3 / 'tables'

MAIN_DATASETS = ('CMYKOGV-7', 'KCMYG-5', 'CMYKOGB-7')
SUPP_DATASETS = ('CMYKOGV-7', 'KCMYG-5', 'CMYKOGB-7', 'PC10-CMYK')
DEGREES = (2, 3, 4, 5)
SPACES = ('xyz', 'cbrt')
SPACE_LABEL = {'xyz': 'XYZ', 'cbrt': 'cbrt'}
STATS = ('median', 'p95', 'max')
OUTPUT_CHANNELS = 3   # X, Y, Z: one coefficient vector per channel


def half_up(value, places=3):
    """Round a CSV string (or Decimal) half up to `places` decimals, as the paper does."""
    q = Decimal(1).scaleb(-places)
    return str(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))


def signed(value, places=3):
    """Half-up rounding with an explicit LaTeX sign, as the paper prints gaps ($+$0.029)."""
    s = half_up(value, places)
    if s.startswith('-'):
        return '$-$' + s[1:]
    return '$+$' + s


def rows_per_coefficient(row):
    """Mean training-fold size / (3 x terms per channel), the paper's definition (2.7 at
    degree 4 on CMYKOGV-7). The CSV's rows_per_coef column divides by terms per channel
    only, i.e. per output channel."""
    d = Decimal(row['mean_train_rows']) / (OUTPUT_CHANNELS * Decimal(row['terms_per_channel']))
    return half_up(d, 1)


def load_sweep(path=SWEEP):
    with open(path, newline='') as f:
        return {(r['dataset'], int(r['degree']), r['space']): r for r in csv.DictReader(f)}


def load_summary(dataset):
    with open(RES / dataset / 'summary.csv', newline='') as f:
        return {r['model']: r for r in csv.DictReader(f)}


def table_degree(sweep):
    """Main-text table: held-out median/P95/max, degree 2..5, XYZ and cube root, n>4 sets."""
    lines = [
        r'\begin{table}[H]',
        r'\caption{Polynomial degree sweep on the three $n>4$ datasets. Held-out $\Delta E_{00}$ '
        r'(median, 95th percentile and maximum) for degrees 2 to 5, fitted on XYZ and on '
        r'$\sqrt[3]{\mathrm{XYZ}}$ (cbrt). All degrees are fitted with the same grouped 5-fold '
        r'cross-validation folds, and every sample is predicted once, by the fold model that did '
        r'not see it. Rows/coef.: mean training-fold size divided by the $3 \times$ '
        r'terms-per-channel coefficients fitted. The Gaussian process rows (one fixed configuration, '
        r'same folds) are for reference. Source: '
        r'\texttt{journal/results/revision\_r3/degree\_sweep\_train\_vs\_heldout.csv}.\label{tab:degree}}',
        r'\begin{adjustwidth}{-\extralength}{0cm}',
        r'\centering',
        r'\footnotesize',
        r'\begin{tabular}{@{}lcccccccc@{}}',
        r'\toprule',
        r'& & & \multicolumn{3}{c}{\textbf{XYZ}} & \multicolumn{3}{c}{\textbf{cbrt}} \\',
        r'\cmidrule(lr){4-6}\cmidrule(lr){7-9}',
        r'\textbf{Degree} & \textbf{Terms/channel} & \textbf{Rows/coef.} & \textbf{Median} & '
        r'\textbf{P95} & \textbf{Max} & \textbf{Median} & \textbf{P95} & \textbf{Max} \\',
    ]
    for ds in MAIN_DATASETS:
        lines += [r'\midrule', rf'\multicolumn{{9}}{{@{{}}l}}{{\emph{{{ds}}}}} \\']
        for d in DEGREES:
            first = sweep[(ds, d, 'xyz')]
            assert first['terms_per_channel'] == sweep[(ds, d, 'cbrt')]['terms_per_channel']
            cells = [str(d), first['terms_per_channel'], rows_per_coefficient(first)]
            for sp in SPACES:
                r = sweep[(ds, d, sp)]
                cells += [half_up(r[f'heldout_{s}']) for s in STATS]
            lines.append(' & '.join(cells) + r' \\')
        summ = load_summary(ds)
        cells = ['Gaussian process', '--', '--']
        for model in ('gaussian_process', 'gaussian_process_cbrt'):
            cells += [half_up(summ[model][s]) for s in STATS]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{adjustwidth}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def table_s_degree(sweep):
    """Supplementary table: training vs held-out median/P95/max and the gaps."""
    lines = [
        r'\begin{table}[H]',
        r'\caption{Polynomial degree sweep, training against held-out $\Delta E_{00}$, degrees 2 to 5, '
        r'fitted on XYZ and on $\sqrt[3]{\mathrm{XYZ}}$ (cbrt), with the folds of the main text. '
        r'Training: the statistic over the training rows of each fold (the model scored on the data '
        r'it was fitted to), averaged over the five folds. Held-out: the statistic of the pooled '
        r'held-out errors, every sample predicted once (the convention of every table in the paper). '
        r'Gap: held-out minus training statistic within each fold, averaged over the five folds. '
        r'Because the held-out column is pooled and the training column is a fold mean, the gap is '
        r'not exactly the difference of the two columns. A training fold is about four times larger '
        r'than its held-out fold, so a maximum-based gap can be negative. Source: '
        r'\texttt{journal/results/revision\_r3/degree\_sweep\_train\_vs\_heldout.csv}.'
        r'\label{tab:s_degree_train_heldout}}',
        r'\begin{adjustwidth}{-\extralength}{0cm}',
        r'\centering',
        r'\scriptsize',
        r'\begin{tabular}{@{}llccccccccc@{}}',
        r'\toprule',
        r'& & \multicolumn{3}{c}{\textbf{Training}} & \multicolumn{3}{c}{\textbf{Held-out}} '
        r'& \multicolumn{3}{c}{\textbf{Gap}} \\',
        r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}',
        r'\textbf{Degree} & \textbf{Space} & \textbf{Median} & \textbf{P95} & \textbf{Max} & '
        r'\textbf{Median} & \textbf{P95} & \textbf{Max} & \textbf{Median} & \textbf{P95} & '
        r'\textbf{Max} \\',
    ]
    for ds in SUPP_DATASETS:
        if not any(k[0] == ds for k in sweep):
            continue
        lines += [r'\midrule', rf'\multicolumn{{11}}{{@{{}}l}}{{\emph{{{ds}}}}} \\']
        for d in DEGREES:
            for sp in SPACES:
                r = sweep[(ds, d, sp)]
                cells = [str(d), SPACE_LABEL[sp]]
                cells += [half_up(r[f'train_{s}']) for s in STATS]
                cells += [half_up(r[f'heldout_{s}']) for s in STATS]
                cells += [signed(r[f'gap_{s}']) for s in STATS]
                lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{adjustwidth}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def main():
    sweep = load_sweep()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in (('tab_degree', table_degree(sweep)),
                       ('tab_s_degree_train_heldout', table_s_degree(sweep))):
        (OUT / f'{name}.tex').write_text(text)
        print(f'wrote {OUT / name}.tex')


if __name__ == '__main__':
    main()
