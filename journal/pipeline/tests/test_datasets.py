import numpy as np
import pandas as pd

from journal.pipeline.color import xyz_to_lab
from journal.pipeline.datasets import DatasetSpec, registry
from journal.pipeline.evaluate import make_groups

INKS = ('INK_1', 'INK_2', 'INK_3')


def _synth_csv(path):
    """4 rows: [0] and [1] byte-identical; [2] same inks but XYZ off by 1e-6
    (near-identical -> must survive exact dedup); [3] distinct."""
    inks = np.array([[10., 20., 30.],
                     [10., 20., 30.],
                     [10., 20., 30.],
                     [50., 60., 70.]])
    xyz = np.array([[30., 40., 50.],
                    [30., 40., 50.],
                    [30., 40., 50.000001],
                    [10., 12., 14.]])
    lab = xyz_to_lab(xyz)
    df = pd.DataFrame(inks, columns=list(INKS))
    df.insert(0, 'SAMPLE_ID', np.arange(1, 5))
    df[['XYZ_X', 'XYZ_Y', 'XYZ_Z']] = xyz
    df[['LAB_L', 'LAB_A', 'LAB_B']] = lab
    df.to_csv(path, index=False)
    return path


def _spec(csv, **kw):
    return DatasetSpec(name='synth', csv=csv, input_cols=INKS,
                       filter_k_zero=False, **kw)


def test_dedup_exact_drops_byte_identical_rows_only(tmp_path):
    csv = _synth_csv(tmp_path / 's.csv')
    X, Y = _spec(csv, dedup_exact=True).load()
    assert len(X) == 3                          # only the exact twin dropped
    assert (Y[:, 2] == 50.000001).any()         # near-identical row survives


def test_dedup_defaults_off(tmp_path):
    csv = _synth_csv(tmp_path / 's.csv')
    X, _ = _spec(csv).load()
    assert len(X) == 4
    spec = _spec(csv)
    assert spec.dedup_exact is False and spec.grouped is False


def test_grouped_flag_surfaces_groups(tmp_path):
    csv = _synth_csv(tmp_path / 's.csv')
    spec = _spec(csv, dedup_exact=True, grouped=True)
    assert spec.grouped is True
    X, _ = spec.load()
    g = make_groups(X)                          # what run.py passes when grouped
    assert g[0] == g[1]                         # same recipe co-travels
    assert g[0] != g[2]                         # distinct recipe separate


def test_registry_ncolor_specs():
    reg = registry()
    for name, n in (('KCMYG-5', 5), ('CMYKOGV-7', 7), ('CMYKOGB-7', 7)):
        spec = reg[name]
        assert spec.input_cols == tuple(f'INK_{i}' for i in range(1, n + 1))
        assert spec.filter_k_zero is False and spec.grouped is True
        assert spec.dedup_exact is (name == 'CMYKOGV-7')
    # existing specs untouched: plain KFold, no dedup
    assert reg['PC10-CMYK'].grouped is False and reg['PC10-CMYK'].dedup_exact is False


def test_cmykogv_effective_rows():
    # CSV keeps all 3534 rows as received; exact dedup at load -> 3302.
    X, Y = registry()['CMYKOGV-7'].load()
    assert X.shape == (3302, 7) and Y.shape == (3302, 3)
