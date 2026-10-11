"""build_tables: half-up rounding, and that --check finds every rebuilt cell and catches a changed one."""
import pytest

from journal.pipeline import build_tables as bt


def test_half_up_rounds_the_csv_string_not_the_float():
    assert bt.half_up('2.0805') == '2.081'          # float formatting gives 2.080
    assert bt.half_up('0.0005') == '0.001'
    assert bt.half_up('0.16') == '0.160'
    assert bt.half_up('3.8735') == '3.874'
    assert bt.sig3('121.5') == '122' and bt.sig3('2.2') == '2.20' and bt.sig3('0.008') == '0.008'
    assert bt.signed('-7.274259') == '$-$7.274' and bt.signed('0.029352') == '$+$0.029'
    assert bt.thousands('31459.7') == '31{,}460'


def _tex(table):
    return (f'\\begin{{table}}\\caption{{x\\label{{{table.label}}}}}\n'
            f'\\begin{{tabular}}{{{"l" * table.ncols}}}\n\\toprule\nheader \\\\\n\\midrule\n'
            f'{table.render()}\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n')


@pytest.mark.parametrize('label', sorted(bt.BUILDERS))
def test_rendered_table_checks_clean(label):
    table = bt.BUILDERS[label]()
    res = bt.check_table(table, bt.paper_table(_tex(table), label))
    assert res.n > 0 and res.matched == res.n
    assert not res.mismatches and not res.untraced and not res.missing


def test_a_changed_cell_is_reported_with_its_source():
    table = bt.tab_transforms()
    tex = _tex(table)
    first = table.rows[0].cells[0].text                  # PC10-CMY, identity
    tex = tex.replace(f'& {first} &', '& 9.999 &', 1)
    res = bt.check_table(table, bt.paper_table(tex, 'tab:transforms'))
    assert len(res.mismatches) == 1
    where, col, paper, csv_value, source = res.mismatches[0]
    assert (col, paper, csv_value) == (1, '9.999', first)
    assert 'transform_sweep.csv' in source
