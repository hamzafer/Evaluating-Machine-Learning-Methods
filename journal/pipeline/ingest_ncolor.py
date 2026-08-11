"""Ingest the three n>4 colorant datasets into standard-schema CSVs.

Run: .venv/bin/python -m journal.pipeline.ingest_ncolor

Raw files (journal/data/raw/ncolor/, see its README):

- KCMYG_5clr_spectral.txt   (2214 patches, 5 inks, spectral nm380..nm730)
- APTEC_CMYKOGV_7clr_xyzlab.txt (3534 patches, 7 inks, native XYZ + Lab, D50/2 M1)
- Apex_CMYKOGB_7clr_spectral.txt (2000 patches, 7 inks, spectral nm380..nm730)

All three carry a CGATS-style BEGIN_DATA_FORMAT/BEGIN_DATA block, so
journal.pipeline.ifra.parse_cgats reads them as-is (the two ProfileMaker
files' LGO* header keys are simply ignored by the regex-based parser, and
ifra.spectral_cols already matches the `nm380` band naming).

Colour source per file:
- spectral files -> ifra.spectral_to_xyz (D50/2 deg, XYZ on 0-100), as for IFRA;
- the APTEC file has no spectral bands: its native XYZ/Lab columns are used
  directly (they are D50/2 M1 per the file header; assert_lab_roundtrip
  verifies XYZ and Lab are mutually consistent on the 0-100 scale).

Output schema: SAMPLE_ID, INK_1..INK_n, XYZ_X/Y/Z, LAB_L/A/B. Ink channel
order = the file's nCLR_k column order. Ink values normalized to 0-100
(all three raw files already use 0-100 -- verified at ingest time by
`_ink_scale`, which rescales only if a file's inks turn out to be 0-1).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .color import assert_lab_roundtrip
from .ifra import parse_cgats, spectral_cols, spectral_to_xyz

RAW = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'ncolor'
OUT = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'ncolor'

# output name -> (raw file, expected ink count, expected patches)
# NOTE: the APTEC file's NUMBER_OF_SETS header says 1624 (the figure the raw
# README and plan originally quoted) but its BEGIN_DATA block really holds
# 3534 rows: contiguous SAMPLE_ID 1..3534, one chart (no seam at 1624 --
# rows 1-1624 and 1625-3534 share only 145 ink combos), 3302 unique ink
# combos, and duplicated combos carry byte-identical XYZ/Lab. The header
# count is stale metadata; all 3534 measured rows are kept verbatim.
FILES = {
    'KCMYG-5': ('KCMYG_5clr_spectral.txt', 5, 2214),
    'CMYKOGV-7': ('APTEC_CMYKOGV_7clr_xyzlab.txt', 7, 3534),
    'CMYKOGB-7': ('Apex_CMYKOGB_7clr_spectral.txt', 7, 2000),
}

_INK_PAT = re.compile(r'^(\d+)CLR_(\d+)$', re.I)


def ink_cols(df: pd.DataFrame) -> list:
    """Return the nCLR_k ink columns in the file's channel order (k ascending).

    Asserts the declared n matches the number of channel columns found.
    """
    found = []
    for c in df.columns:
        m = _INK_PAT.match(c)
        if m:
            found.append((int(m.group(2)), int(m.group(1)), c))
    assert found, 'no nCLR_k ink columns found'
    n_declared = {n for _, n, _ in found}
    assert len(n_declared) == 1, f'mixed nCLR prefixes: {n_declared}'
    cols = [c for _, _, c in sorted(found)]
    assert len(cols) == n_declared.pop(), \
        f'declared n does not match {len(cols)} channel columns'
    return cols


def _ink_scale(ink: np.ndarray) -> float:
    """Detect the ink-value scale: 100.0 if already 0-100, 1.0 if 0-1.

    Every chart contains full-tone (100%) patches, so max ~1 means the
    fractional scale; anything above 1.5 means percent. All-zero ink data is
    scale-ambiguous but needs no rescaling, so it reports percent (identity).
    """
    mx = float(np.nanmax(ink))
    assert 0 <= mx <= 100.5, f'ink values out of any known range (max={mx})'
    return 1.0 if 0 < mx <= 1.5 else 100.0


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Parsed raw frame -> standard schema:
    SAMPLE_ID, INK_1..INK_n, XYZ_X/Y/Z, LAB_L/A/B.

    Spectral files get XYZ/Lab derived via spectral_to_xyz; files without
    spectral bands must already carry native XYZ_X/Y/Z + LAB_L/A/B, which are
    used directly (never converted).
    """
    inks = ink_cols(df)
    if spectral_cols(df):
        df = spectral_to_xyz(df)
    else:
        missing = {'XYZ_X', 'XYZ_Y', 'XYZ_Z', 'LAB_L', 'LAB_A', 'LAB_B'} - set(df.columns)
        assert not missing, f'no spectral bands and no native colour columns: {missing}'

    id_col = 'SAMPLE_ID' if 'SAMPLE_ID' in df.columns else 'SampleID'
    out = pd.DataFrame({'SAMPLE_ID': df[id_col].astype(int)})
    ink = df[inks].to_numpy(dtype=float) * (100.0 / _ink_scale(df[inks].to_numpy(dtype=float)))
    for k in range(len(inks)):
        out[f'INK_{k + 1}'] = ink[:, k]
    for c in ('XYZ_X', 'XYZ_Y', 'XYZ_Z', 'LAB_L', 'LAB_A', 'LAB_B'):
        out[c] = df[c].to_numpy(dtype=float)
    return out


def ingest() -> dict:
    """Ingest all three raw files; write OUT/<name>.csv; return {name: df}."""
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, (fname, n_inks, n_rows) in FILES.items():
        raw = parse_cgats(RAW / fname)
        scale = _ink_scale(raw[ink_cols(raw)].to_numpy(dtype=float))
        df = standardize(raw)

        assert len(df) == n_rows, f'{name}: expected {n_rows} rows, got {len(df)}'
        assert list(df.columns)[1:n_inks + 1] == [f'INK_{k + 1}' for k in range(n_inks)]
        assert not df.isna().any().any(), f'{name}: NaNs in output'
        ink = df.filter(like='INK_').to_numpy()
        assert ink.min() >= 0 and ink.max() <= 100.5, f'{name}: ink range off'
        # unprinted patches (all inks 0) must look like paper white
        white_y = df.loc[(ink == 0).all(axis=1), 'XYZ_Y']
        assert len(white_y) > 0 and white_y.between(60, 100).all(), \
            f'{name}: unprinted-patch Y implausible: {sorted(white_y)}'
        # tripwire: the output XYZ must reproduce the output Lab (0-100, D50/2)
        assert_lab_roundtrip(df[['XYZ_X', 'XYZ_Y', 'XYZ_Z']].to_numpy(),
                             df[['LAB_L', 'LAB_A', 'LAB_B']].to_numpy(), name)

        df.to_csv(OUT / f'{name}.csv', index=False)
        print(f'  {name}: {len(df)} rows, {n_inks} inks (raw ink scale 0-{scale:.0f}), '
              f'paper-white Y={white_y.median():.2f} -> {OUT / (name + ".csv")}')
        result[name] = df
    return result


if __name__ == '__main__':
    ingest()
