"""Rebuild the paper's results tables from the public CSVs, and check a manuscript against them.

  python -m journal.pipeline.build_tables --write journal/results/tables
  python -m journal.pipeline.build_tables --check path/to/main.tex

Every table that holds model results is rebuilt from tracked CSVs under
journal/results/ (the per-dataset summary.csv files and the revision_r1/r2/r3
CSVs). Nothing is typed by hand. Values are read as CSV strings and rounded
half up with Decimal, the paper's convention; aggregates (newsprint medians
across runs and press pairs, rows per coefficient) are computed in Decimal
from those strings.

--write writes one LaTeX fragment per table label (the rows between \\midrule
and \\bottomrule, ready to paste into the paper's tabular), named after the label.

--check finds each table in the manuscript by its \\label, parses its rows and
compares every numeric cell, row by row, with the rebuilt value. It prints, per
table, the number of numeric cells, how many match, and every mismatch with
the paper value, the CSV value and the CSV path. Rows the paper has but no CSV
explains (for example the literature rows of tab:prior) are listed as untraced.
It exits non-zero if any cell or bold mark mismatches, a CSV row is missing
from the paper, or a paper row outside tab:prior has no CSV source.
"""
import argparse
import csv
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

JOURNAL = Path(__file__).resolve().parents[1]
REPO = JOURNAL.parent
RES = JOURNAL / 'results'

NINE = ('PC10-CMY', 'PC11-CMY', 'FOGRA51-CMY', 'PC10-CMYK', 'PC11-CMYK',
        'FOGRA51-CMYK', 'KCMYG-5', 'CMYKOGV-7', 'CMYKOGB-7')
STATS = ('median', 'p95', 'max')

READ = set()          # every CSV path read, checked against `git ls-files` in --check


# ----------------------------------------------------------------------------- rounding

def half_up(value, places=3):
    """Round a CSV string (or Decimal) half up to `places` decimals, as the paper does."""
    q = Decimal(1).scaleb(-places)
    return str(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))


def sig3(value):
    """Three significant figures, never more than three decimals (the timing table)."""
    d = Decimal(str(value))
    if d == 0:
        return '0'
    exp = d.adjusted()                          # floor(log10(|d|))
    places = min(3, max(0, 2 - exp))
    return half_up(d, places)


def thousands(value):
    """Integer, half up, with a LaTeX thousands separator (31{,}460)."""
    n = int(Decimal(str(value)).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    return f'{n:,}'.replace(',', '{,}')


def signed(value, places=3):
    s = half_up(value, places)
    return '$-$' + s[1:] if s.startswith('-') else '$+$' + s


# ----------------------------------------------------------------------------- sources

def rel(path):
    return str(Path(path).resolve().relative_to(REPO))


def read_csv(path):
    path = Path(path)
    READ.add(path.resolve())
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


@dataclass
class Cell:
    """One table cell: the LaTeX text the table prints and, for each number in it, its source."""
    text: str
    sources: list = field(default_factory=list)     # one 'path [row, column] = raw' per number
    numeric: bool = True
    paths: list = field(default_factory=list)       # CSV files the numbers come from

    @staticmethod
    def blank(text=''):
        return Cell(text, [], numeric=False)


def summary(ds):
    path = RES / ds / 'summary.csv'
    return {r['model']: (r, path) for r in read_csv(path)}


def r1_summary(ds, model):
    path = RES / 'revision_r1' / 'summary' / f'{ds}__{model}.csv'
    return read_csv(path)[0], path


def cell(row_path, column, fmt=half_up, bold=False):
    row, path = row_path
    raw = row[column]
    key = row.get('model') or row.get('dataset') or ''
    text = fmt(raw)
    if bold:
        text = rf'\textbf{{{text}}}'
    return Cell(text, [f'{rel(path)} [{key}, {column}] = {raw}'], paths=[path])


def derived(value, text, how, paths):
    return Cell(text, [f'{how} = {value} from ' + ', '.join(rel(p) for p in paths)], paths=list(paths))


# ----------------------------------------------------------------------------- table model

@dataclass
class Row:
    labels: list            # leading label cells (LaTeX)
    cells: list             # Cell objects
    section: str = ''       # LaTeX of the \multicolumn section line this row sits under
    rule_before: bool = False


@dataclass
class Table:
    label: str
    ncols: int
    rows: list
    bold_checked: bool = False   # True: the bold markup is part of the CSV-derived content
    sorted_note: str = ''

    def render(self):
        out, section = [], None
        for r in self.rows:
            if r.section != section:
                if section is not None:
                    out.append(r'\midrule')
                if r.section:
                    out.append(rf'\multicolumn{{{self.ncols}}}{{@{{}}l}}{{{r.section}}} \\')
                section = r.section
            elif r.rule_before:
                out.append(r'\midrule')
            out.append(' & '.join(list(r.labels) + [c.text for c in r.cells]) + r' \\')
        return '\n'.join(out) + '\n'


# ----------------------------------------------------------------------------- labels

MODEL_LABEL = {
    'gaussian_process_cbrt': r'Gaussian Process (cbrt)$^\dagger$',
    'gaussian_process': 'Gaussian Process',
    'poly4_cbrt': r'Polynomial (4th order, cbrt)$^\dagger$',
    'poly4': r'Polynomial (4th order)$^\dagger$',
    'poly3_cbrt': r'Polynomial (3rd order, cbrt)$^\dagger$',
    'poly3': 'Polynomial (3rd order)',
    'svm_cbrt': r'SVM (cbrt)$^\ddagger$',
    'mlp_deep_cbrt': r'MLP (Deep, cbrt)$^\ddagger$',
    'svm': 'SVM', 'gradient_boost': 'Gradient Boosting', 'mlp_deep': 'MLP (Deep)',
    'mlp_shallow': 'MLP (Shallow)', 'random_forest': 'Random Forest', 'knn': '$k$-NN',
    'decision_tree': 'Decision Tree', 'lasso': 'Lasso', 'plsr': 'PLSR', 'elastic': 'Elastic Net',
    'ridge': 'Ridge', 'pcr': 'PCR',
    'poly3_de00_powell': r'Poly3 + $\Delta E_{00}$ (Powell)',
    'poly3_de00_nm_eqfev': r'Poly3 + $\Delta E_{00}$ (Nelder--Mead, equal evaluations)',
}
R1_ROWS = ('svm_cbrt', 'mlp_deep_cbrt')          # cube-root rows from revision_r1/summary


def optim_eqfev():
    path = RES / 'revision_r1' / 'tables' / 'optim_eqfev.csv'
    return {r['dataset']: (r, path) for r in read_csv(path)}


def model_stats(ds, model):
    """(row, path, column prefix) for one model on one dataset, wherever the paper takes it from."""
    if model in R1_ROWS:
        row, path = r1_summary(ds, model)
        return (row, path), ''
    if model == 'poly3_de00_nm_eqfev':
        return optim_eqfev()[ds], 'nm_eqfev_'
    return summary(ds)[model], ''


def results_table(label, datasets, extra=(), drop=('poly3_de00_powell', 'poly3_de00_nm')):
    """Tables 3, 4 and 10: every model, median/P95/max, sorted by the first dataset's median."""
    models = [m for m in summary(datasets[0]) if m not in drop] + list(R1_ROWS) + list(extra)
    def first_median(m):
        (row, _), prefix = model_stats(datasets[0], m)
        return Decimal(row[prefix + 'median'])
    models.sort(key=first_median)                     # stable: ties keep CSV order
    rows = []
    for m in models:
        cells = []
        for ds in datasets:
            src, prefix = model_stats(ds, m)
            cells += [cell(src, prefix + s) for s in STATS]
        rows.append(Row([MODEL_LABEL[m]], cells))
    return Table(label, 1 + 3 * len(datasets), rows,
                 sorted_note=f'sorted by {datasets[0]} median')


def tab_n3():
    return results_table('tab:n3', NINE[0:3])


def tab_n4():
    return results_table('tab:n4', NINE[3:6])


def tab_n7():
    return results_table('tab:n7', NINE[6:9], extra=('poly3_de00_powell', 'poly3_de00_nm_eqfev'),
                         drop=('poly3_de00_powell', 'poly3_de00_nm'))


def tab_de00loss():
    rows = []
    for cond, dss in ((r'\emph{$n=3$ (CMY)}', NINE[0:3]), (r'\emph{$n=4$ (CMYK)}', NINE[3:6])):
        for m, lab in (('poly3', 'Poly3 (least squares)'),
                       ('poly3_de00_nm_eqfev', MODEL_LABEL['poly3_de00_nm_eqfev']),
                       ('poly3_de00_powell', MODEL_LABEL['poly3_de00_powell'])):
            cells = []
            for ds in dss:
                src, prefix = model_stats(ds, m)
                cells += [cell(src, prefix + s) for s in STATS]
            rows.append(Row([lab], cells, section=cond))
    return Table('tab:de00loss', 10, rows)


def tab_fitspace():
    pairs = ('poly3', 'poly3_cbrt', 'poly4', 'poly4_cbrt', 'gaussian_process', 'gaussian_process_cbrt')
    rows = []
    for stat, sec in (('median', r'\emph{Median $\Delta E_{00}$}'), ('max', r'\emph{Maximum $\Delta E_{00}$}')):
        for ds in NINE:
            s = summary(ds)
            vals = [Decimal(half_up(s[m][0][stat])) for m in pairs]
            best = min(vals) if stat == 'median' else None
            rows.append(Row([ds], [cell(s[m], stat, bold=(v == best)) for m, v in zip(pairs, vals)],
                            section=sec))
    return Table('tab:fitspace', 7, rows, bold_checked=True)


def tab_transforms():
    path = RES / 'transform_sweep.csv'
    cols = ('identity', 'sqrt', 'cbrt', 'pow025', 'log1p')
    rows = []
    for r in read_csv(path):
        vals = [Decimal(half_up(r[c])) for c in cols]
        rows.append(Row([r['dataset']], [cell((r, path), c, bold=(v == min(vals))) for c, v in zip(cols, vals)]))
    return Table('tab:transforms', 6, rows, bold_checked=True)


def _sweep():
    path = RES / 'revision_r3' / 'degree_sweep_train_vs_heldout.csv'
    return {(r['dataset'], int(r['degree']), r['space']): (r, path) for r in read_csv(path)}, path


def _rows_per_coef(src):
    """Mean training-fold size / (3 x terms per channel): the paper's 'rows per coefficient'."""
    r, path = src
    v = Decimal(r['mean_train_rows']) / (3 * Decimal(r['terms_per_channel']))
    return derived(v, half_up(v, 1), f"mean_train_rows / (3 x terms_per_channel) "
                                     f"[{r['dataset']}, d{r['degree']}]", [path])


def tab_degree():
    """Round-3 layout: degrees 2-5 x {XYZ, cbrt} x the n>4 sets, held-out median/P95/max."""
    sweep, _ = _sweep()
    rows = []
    for ds in ('CMYKOGV-7', 'KCMYG-5', 'CMYKOGB-7'):
        sec = rf'\emph{{{ds}}}'
        for d in (2, 3, 4, 5):
            x = sweep[(ds, d, 'xyz')]
            cells = [cell(x, 'terms_per_channel', fmt=str), _rows_per_coef(x)]
            for sp in ('xyz', 'cbrt'):
                cells += [cell(sweep[(ds, d, sp)], f'heldout_{s}') for s in STATS]
            rows.append(Row([str(d)], cells, section=sec))
        s = summary(ds)
        cells = [Cell.blank('--'), Cell.blank('--')]
        for m in ('gaussian_process', 'gaussian_process_cbrt'):
            cells += [cell(s[m], st) for st in STATS]
        rows.append(Row(['Gaussian process'], cells, section=sec))
    return Table('tab:degree', 9, rows)


def tab_degree_r2():
    """The round-2 layout (CMYKOGV-7 only; median XYZ, median cbrt, terms, rows/coef, gap cbrt)."""
    path = RES / 'CMYKOGV-7' / 'degree_sweep.csv'
    sweep, _ = _sweep()
    rows = []
    for r in read_csv(path):
        d = int(r['degree'])
        src = (r, path)
        rows.append(Row([str(d)], [cell(src, 'median_xyz'), cell(src, 'median_cbrt'),
                                   cell(src, 'terms_per_channel', fmt=str),
                                   _rows_per_coef(sweep[('CMYKOGV-7', d, 'xyz')]),
                                   cell(src, 'gap_cbrt', fmt=signed)]))
    return Table('tab:degree', 6, rows)


def _newsprint_dir():
    # the newsprint results folder is the one holding within_run.csv
    return next(RES.glob('*/within_run.csv')).parent


def tab_newsprint():
    nd = _newsprint_dir()
    within, cross, loo = (nd / 'within_run.csv', nd / 'cross_run.csv', nd / 'leave_one_out.csv')
    w, c, l = read_csv(within), read_csv(cross), read_csv(loo)
    mm_path = RES / 'revision_r1' / 'tables' / 'loo_median_of_medians.csv'
    mm = {r['model']: (r, mm_path) for r in read_csv(mm_path)}

    def med(rows, model, n_expected, path, what):
        vals = [Decimal(r['median']) for r in rows if r['model'] == model]
        assert len(vals) == n_expected, (model, len(vals))
        v = statistics.median(vals)
        return derived(v, half_up(v), f'median of {len(vals)} {what} medians [{model}]', [path])

    rows = []
    for m, lab in (('gaussian_process', 'Gaussian Process'), ('svm', 'SVM'),
                   ('poly3', 'Polynomial (3rd order)'), ('mlp_deep', 'MLP (Deep)')):
        rows.append(Row([lab], [med(w, m, 13, within, 'per-run'), med(c, m, 156, cross, 'press-pair'),
                                med(l, m, 13, loo, 'held-out-run')]))
    sec = r"\emph{Fitted on the Gaussian process's 2{,}000-row subsample (condition C only)}"
    for m, lab in (('svm', 'SVM'), ('poly3', 'Polynomial (3rd order)'), ('mlp_deep', 'MLP (Deep)')):
        rows.append(Row([lab], [Cell.blank('--'), Cell.blank('--'),
                                cell(mm[f'{m}_sub2000'], 'median_of_medians')], section=sec))
    return Table('tab:newsprint', 4, rows)


def tab_stats():
    path = RES / 'revision_r1' / 'tables' / 'stats_gp_vs_poly4cbrt.csv'
    rows = []
    for r in read_csv(path):
        src = f"{rel(path)} [{r['dataset']}, %s] = %s"

        def ci(a):
            cols = (a, a + '_lo', a + '_hi')
            return Cell(f'{half_up(r[cols[0]])} [{half_up(r[cols[1]])}, {half_up(r[cols[2]])}]',
                        [src % (k, r[k]) for k in cols], paths=[path])
        frac = Decimal(r['frac_a_better']) * 100
        pct = Cell(f"{half_up(frac, 0)}\\%", [src % ('frac_a_better', r['frac_a_better'])], paths=[path])
        p = Decimal(r['p_holm'])
        ptxt = r'$<10^{-22}$' if p < Decimal('1e-22') else half_up(p)
        pcell = Cell(ptxt, [src % ('p_holm', r['p_holm'])], numeric=False, paths=[path])
        rows.append(Row([r['dataset']], [ci('gp_median'), ci('poly_median'), ci('gp_p95'), ci('poly_p95'),
                                         pct, pcell]))
    return Table('tab:stats', 7, rows)


def tab_timing():
    tpath = RES / 'revision_r1' / 'timing.csv'
    fpath = RES / 'revision_r2' / 'footprint.csv'
    fp = {r['model']: (r, fpath) for r in read_csv(fpath)}
    lab = {'poly3': 'Polynomial (3rd order)', 'poly4': 'Polynomial (4th order)',
           'poly4_cbrt': 'Polynomial (4th order, cbrt)', 'gaussian_process': 'Gaussian Process',
           'gaussian_process_cbrt': 'Gaussian Process (cbrt)', 'svm': 'SVM',
           'random_forest': 'Random Forest', 'mlp_deep': 'MLP (Deep)'}
    rows = []
    for r in read_csv(tpath):
        src = (r, tpath)
        f = fp[r['model']]
        kb = Decimal(f[0]['serialized_kb'])
        rows.append(Row([lab[r['model']]], [
            cell(src, 'fit_s', fmt=sig3), cell(src, 'us_per_point', fmt=sig3),
            cell(src, 'lut_17pow4_cmyk_s', fmt=sig3), cell(src, 'lut_9pow7_projected_s', fmt=sig3),
            cell(f, 'serialized_kb', fmt=thousands if kb >= 100 else (lambda v: half_up(v, 1))),
            cell(f, 'learned_values', fmt=thousands)]))
    return Table('tab:timing', 7, rows)


def tab_prior():
    """Only the 'This paper' rows: GP (cbrt) / degree-4 polynomial (cbrt), median and mean."""
    def pair(ds, stat):
        s = summary(ds)
        a, b = s['gaussian_process_cbrt'], s['poly4_cbrt']
        return Cell(f'{half_up(a[0][stat])} / {half_up(b[0][stat])}',
                    [f"{rel(a[1])} [gaussian_process_cbrt, {stat}] = {a[0][stat]}",
                     f"{rel(b[1])} [poly4_cbrt, {stat}] = {b[0][stat]}"], paths=[a[1]])
    e = Cell.blank
    rows = []
    for stat, err in (('median', r'$\Delta E_{00}$ median'), ('mean', r'$\Delta E_{00}$ mean')):
        first = stat == 'median'
        rows.append(Row([r'This paper$^{c}$' if first else '',
                         'GP / polynomial, degree 4 (both cube root); PC10, KCMYG, CMYKOGV' if first else '',
                         r'see $^{c}$' if first else '', err],
                        [pair('PC10-CMY', stat), pair('PC10-CMYK', stat), pair('KCMYG-5', stat),
                         pair('CMYKOGV-7', stat), e()]))
    for stat, err in (('median', r'$\Delta E_{00}$ median'), ('mean', r'$\Delta E_{00}$ mean')):
        first = stat == 'median'
        rows.append(Row([r'This paper$^{c}$' if first else '', 'as above; CMYKOGB' if first else '',
                         r'see $^{c}$' if first else '', err],
                        [e(), e(), e(), pair('CMYKOGB-7', stat), e()]))
    t = Table('tab:prior', 9, rows)
    t.nlabels = 4
    return t


def tab_appbest():
    rows = []
    for m, lab in (('svm', 'SVM'), ('mlp_deep', 'MLP (Deep)'), ('gradient_boost', 'Gradient Boosting'),
                   ('random_forest', 'Random Forest'), ('knn', '$k$-NN')):
        rows.append(Row([f'{lab}, tuned, cbrt'],
                        [cell(r1_summary(ds, f'{m}_cbrt_tuned'), 'median') for ds in NINE]))
    rows.append(Row(['Gaussian Process (cbrt)'],
                    [cell(summary(ds)['gaussian_process_cbrt'], 'median') for ds in NINE], rule_before=True))
    return Table('tab:appbest', 10, rows)


BUILDERS = {
    'tab:n3': tab_n3, 'tab:n4': tab_n4, 'tab:de00loss': tab_de00loss, 'tab:fitspace': tab_fitspace,
    'tab:transforms': tab_transforms, 'tab:degree': tab_degree, 'tab:newsprint': tab_newsprint,
    'tab:n7': tab_n7, 'tab:stats': tab_stats, 'tab:timing': tab_timing, 'tab:prior': tab_prior,
    'tab:appbest': tab_appbest,
}
ALTERNATES = {'tab:degree': [tab_degree_r2]}       # older layouts still accepted by --check


# ----------------------------------------------------------------------------- manuscript parsing

def _strip_comments(tex):
    return '\n'.join(re.sub(r'(?<!\\)%.*$', '', line) for line in tex.split('\n'))


def _split_top(s, sep='&'):
    parts, depth, cur, i = [], 0, [], 0
    while i < len(s):
        ch = s[i]
        if ch == '\\' and i + 1 < len(s):
            cur.append(s[i:i + 2])
            i += 2
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    parts.append(''.join(cur))
    return [p.strip() for p in parts]


def _brace_args(s, n):
    """The first n top-level {...} groups of s (for \\multicolumn{n}{spec}{text})."""
    args, i = [], 0
    while len(args) < n:
        i = s.index('{', i)
        depth, j = 0, i
        while True:
            if s[j] == '\\':
                j += 2
                continue
            depth += {'{': 1, '}': -1}.get(s[j], 0)
            if depth == 0:
                break
            j += 1
        args.append(s[i + 1:j])
        i = j + 1
    return args


RULE = re.compile(r'^\s*(\\(toprule|midrule|bottomrule|addlinespace|hline)\b|\\cmidrule(\([^)]*\))?\{[^}]*\})\s*')


def paper_table(tex, label):
    """Body rows of the tabular that follows \\label{label}: a list of (section, cells)."""
    tex = _strip_comments(tex)
    at = tex.find(rf'\label{{{label}}}')
    if at < 0:
        return None
    m = re.compile(r'\\begin\{tabularx?\}').search(tex, at)
    end = tex.find(r'\end{tabular', m.end())
    body = tex[m.end():end]
    body = body[body.find(r'\midrule') + len(r'\midrule'):]
    body = body[:body.find(r'\bottomrule')] if r'\bottomrule' in body else body
    rows, section = [], ''
    for chunk in re.split(r'\\\\', body):
        s = chunk.strip()
        while True:
            m2 = RULE.match(s)
            if not m2:
                break
            s = s[m2.end():].strip()
        if not s:
            continue
        if s.startswith(r'\multicolumn') and len(_split_top(s)) == 1:
            section = _brace_args(s, 3)[2]
            continue
        rows.append((section, _split_top(s)))
    return rows


def norm_label(s):
    s = re.sub(r'\$\^\{?\\?(dagger|ddagger|[a-z])\}?\$', '', s)
    s = s.replace('{,}', ',')
    s = re.sub(r'\\(textbf|emph|textsubscript)\b', '', s)
    return re.sub(r'[\s${}\\]', '', s).lower()


NUM = re.compile(r'[+-]?\d+(?:\.\d+)?')


def norm_cell(s):
    """(numbers, bold, text) of a cell as printed."""
    bold = r'\textbf' in s
    s = re.sub(r'\\textbf\{([^{}]*)\}', r'\1', s)
    s = s.replace('{,}', '').replace('$+$', '+').replace('$-$', '-').replace(r'\%', '%')
    s = ' '.join(s.split())
    return NUM.findall(s), bold, s


# ----------------------------------------------------------------------------- check

@dataclass
class Result:
    label: str
    n: int = 0
    matched: int = 0
    mismatches: list = field(default_factory=list)
    untraced: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _keyed(rows, nlabels):
    out, carry = [], [''] * nlabels
    for section, labels, payload in rows:
        labels = [lab if lab.strip() else carry[i] for i, lab in enumerate(labels)]
        carry = labels
        out.append(((norm_label(section), tuple(norm_label(x) for x in labels)), labels, payload))
    return out


def check_table(table, prows):
    nl = getattr(table, 'nlabels', 1)
    res = Result(table.label)
    built = _keyed([(r.section, r.labels, r) for r in table.rows], nl)
    paper = _keyed([(sec, cells[:nl], cells) for sec, cells in prows], nl)
    by_key = {k: r for k, _, r in built}
    by_label = {}
    for (sec, lab), _, r in built:
        by_label.setdefault(lab, []).append(r)
    used = set()
    for (sec, lab), labels, cells in paper:
        r = by_key.get((sec, lab))
        if r is None and len(by_label.get(lab, [])) == 1:
            r = by_label[lab][0]
        if r is None:
            res.untraced.append(' | '.join(labels))
            continue
        used.add(id(r))
        pcells = cells[nl:]
        if len(pcells) != len(r.cells):
            res.mismatches.append((' / '.join(x for x in (sec, labels[0]) if x), '(row)',
                                   f'{len(pcells)} cells', f'{len(r.cells)} cells', 'column count differs'))
            continue
        for j, (pc, bc) in enumerate(zip(pcells, r.cells)):
            col = j + 1
            pnums, pbold, ptext = norm_cell(pc)
            bnums, bbold, btext = norm_cell(bc.text)
            where = (r.section and norm_cell(r.section)[2] + ' / ' or '') + ' '.join(labels)
            if not bc.numeric:
                if ptext != btext:
                    res.mismatches.append((where, col, ptext, btext, '; '.join(bc.sources) or 'text'))
                continue
            if len(pnums) != len(bnums):
                res.n += max(len(pnums), len(bnums))
                res.mismatches.append((where, col, ptext, btext, '; '.join(bc.sources)))
                continue
            for k, (a, b) in enumerate(zip(pnums, bnums)):
                res.n += 1
                if a == b:
                    res.matched += 1
                else:
                    src = bc.sources[k] if k < len(bc.sources) else '; '.join(bc.sources)
                    res.mismatches.append((where, col, a, b, src))
            if table.bold_checked and pbold != bbold:
                res.mismatches.append((where, col, f'{"bold" if pbold else "plain"}: {ptext}',
                                       f'{"bold" if bbold else "plain"} (CSV best-value rule)',
                                       '; '.join(bc.sources)))
    for (_, _), labels, r in built:
        if id(r) not in used:
            res.missing.append(' | '.join(labels))
    if table.sorted_note:              # ties may be ordered either way; only a descent is flagged
        firsts = [Decimal(norm_cell(cells[nl])[0][0]) for _, _, cells in paper if norm_cell(cells[nl])[0]]
        if any(b < a for a, b in zip(firsts, firsts[1:])):
            res.notes.append(f'rows are not {table.sorted_note}')
    return res


def check(tex_path):
    tex = Path(tex_path).read_text()
    results = []
    for label, builder in BUILDERS.items():
        prows = paper_table(tex, label)
        if prows is None:
            r = Result(label)
            r.notes.append('label not found in the manuscript')
            results.append(r)
            continue
        width = max((len(c) for _, c in prows), default=0)
        candidates = [builder] + ALTERNATES.get(label, [])
        tables = [b() for b in candidates]
        table = next((t for t in tables if t.ncols == width), tables[0])
        r = check_table(table, prows)
        if table is not tables[0]:
            r.notes.append(f'checked against the older layout ({candidates[tables.index(table)].__name__})')
        results.append(r)
    return results


def untracked(paths):
    try:
        out = subprocess.run(['git', 'ls-files', '--', *[str(p) for p in paths]], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.split('\n')
    except (OSError, subprocess.CalledProcessError):
        return None
    tracked = {(REPO / p).resolve() for p in out if p}
    return sorted(rel(p) for p in paths if p not in tracked)


def print_report(results):
    print(f"{'table':<16}{'cells':>7}{'matched':>9}{'mismatched':>12}{'untraced rows':>15}")
    for r in results:
        print(f'{r.label:<16}{r.n:>7}{r.matched:>9}{len(r.mismatches):>12}{len(r.untraced):>15}')
    for r in results:
        if not (r.mismatches or r.untraced or r.missing or r.notes):
            continue
        print(f'\n== {r.label}')
        for where, col, paper, csvv, src in r.mismatches:
            print(f'  MISMATCH  row "{where}", column {col}: paper {paper}, CSV {csvv}  <- {src}')
        for u in r.untraced:
            print(f'  untraced  paper row with no CSV source: {u}')
        for u in r.missing:
            print(f'  missing   CSV row not in the paper table: {u}')
        for n in r.notes:
            print(f'  note      {n}')


def _source_name(path):
    path = Path(path).resolve()
    if path.parent == (RES / 'revision_r1' / 'summary').resolve():
        return 'journal/results/revision_r1/summary/<dataset>__<model>.csv'
    if path.parent == _newsprint_dir().resolve():
        return f'<newsprint results folder>/{path.name}'
    return rel(path)


def write(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for label, builder in BUILDERS.items():
        t = builder()
        sources = sorted({_source_name(p) for r in t.rows for c in r.cells for p in c.paths})
        head = (f'% {label}: generated by journal/pipeline/build_tables.py from the CSVs; do not edit.\n'
                f'% Rows between \\midrule and \\bottomrule. Sources: {", ".join(sources)}\n')
        name = label.replace('tab:', 'tab_')
        (outdir / f'{name}.tex').write_text(head + t.render())
        print(f'wrote {outdir / name}.tex')


UNTRACED_OK = {'tab:prior'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', metavar='MAIN_TEX', help='check every table cell of a manuscript')
    ap.add_argument('--write', metavar='OUTDIR', help='write one LaTeX fragment per table label')
    a = ap.parse_args(argv)
    if not (a.check or a.write):
        ap.error('give --check and/or --write')
    if a.write:
        write(a.write)
    status = 0
    if a.check:
        results = check(a.check)
        print_report(results)
        bad = untracked(READ)
        if bad:
            print('\nWARNING: read CSVs that git does not track: ' + ', '.join(bad))
            status = 1
        elif bad is None:
            print('\n(git not available: tracked-file check skipped)')
        else:
            print(f'\nAll {len(READ)} CSVs read are tracked by git.')
        # Every paper row must be traced to a CSV, except the literature rows of the
        # prior-work table (values read from the cited papers, as its caption says).
        if any(r.mismatches or r.missing or (r.untraced and r.label not in UNTRACED_OK)
               for r in results):
            status = 1
    return status


if __name__ == '__main__':
    sys.exit(main())
