"""IFRA 2005 newsprint: parsing (wb=CGATS, bb=plain-header) + spectral -> XYZ/Lab
(D50, 2 deg).

Two genuinely different raw formats (see journal/data/processed/ifra/README.md,
Task-1 inventory):

- wb: real CGATS/ECI2002 text (BEGIN_DATA_FORMAT/END_DATA_FORMAT keywords).
  Columns: SAMPLE_ID, SAMPLE_NAME, CMYK_C/M/Y/K, SPECTRAL_NM_380..730. No XYZ/Lab.
- bb: NOT CGATS. 4-line key/value preamble, blank line, one tab-delimited
  header row (`SampleID  XYZ-X ... CIE-Lab-b  380nm...730nm`), blank line,
  then data rows. No CMYK columns at all. Some bb files (README "Group B")
  lack the XYZ-X/Y/Z columns and start straight from CIE-Lab-L.

Both readers return native columns as found in the file, verbatim. Derived
XYZ_X/XYZ_Y/XYZ_Z (0-100 scale) and LAB_L/LAB_A/LAB_B are added by
spectral_to_xyz, uniformly for every run (amendment #4) -- never trust a
native XYZ/Lab column as pipeline output, only as a cross-check.
"""
import re
from pathlib import Path

import colour
import numpy as np
import pandas as pd

from .color import xyz_to_lab

_CMF = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
_D50 = colour.SDS_ILLUMINANTS['D50']


def parse_cgats(path: Path) -> pd.DataFrame:
    """Parse a wb-style genuine-CGATS IFRA file into a DataFrame.

    Columns are exactly the file's BEGIN_DATA_FORMAT field list (SAMPLE_ID,
    SAMPLE_NAME, CMYK_C/M/Y/K, SPECTRAL_NM_380..730).
    """
    text = Path(path).read_text(encoding='latin-1')
    fmt = re.search(r'BEGIN_DATA_FORMAT\s*\n(.*?)\nEND_DATA_FORMAT', text, re.S)
    cols = fmt.group(1).split()
    data = re.search(r'BEGIN_DATA\s*\n(.*?)\nEND_DATA', text, re.S).group(1)
    rows = [ln.split() for ln in data.strip().splitlines()]
    df = pd.DataFrame(rows, columns=cols)
    return df.apply(pd.to_numeric, errors='ignore')


def _strip_trailing_empty(fields: list) -> list:
    """Drop trailing empty strings from a tab-split line.

    Real bb files end each header/data line with a trailing tab (an
    unnamed/empty final field) -- both the header row and the data rows do
    this consistently, so stripping independently on each keeps the field
    counts aligned.
    """
    while fields and fields[-1] == '':
        fields.pop()
    return fields


def parse_bb(path: Path) -> pd.DataFrame:
    """Parse a bb-style plain-header IFRA file into a DataFrame.

    Not CGATS: no BEGIN_DATA_FORMAT block. Field names come from the first
    tab-delimited line starting with "SampleID" (after a 4-line key/value
    preamble + blank line); data rows follow after a blank line, running to
    EOF. Columns are exactly as found in the file's header row (e.g.
    `SampleID, XYZ-X, XYZ-Y, XYZ-Z, CIE-Lab-L, CIE-Lab-a, CIE-Lab-b,
    380nm..730nm` for the majority "Group A" sub-format, or the same minus
    XYZ-X/Y/Z for the XYZ-less "Group B" sub-format). No CMYK columns are
    present in bb files (amendment #3) -- those must be joined in separately
    from a wb-derived SAMPLE_ID -> CMYK table.
    """
    text = Path(path).read_text(encoding='latin-1')
    lines = text.split('\n')
    header_idx = next(
        i for i, ln in enumerate(lines) if ln.strip().lower().startswith('sampleid'))
    cols = _strip_trailing_empty(lines[header_idx].rstrip('\r').split('\t'))

    rows = []
    for ln in lines[header_idx + 1:]:
        ln = ln.rstrip('\r')
        if not ln.strip():
            continue
        rows.append(_strip_trailing_empty(ln.split('\t')))

    df = pd.DataFrame(rows, columns=cols)
    return df.apply(pd.to_numeric, errors='ignore')


def spectral_cols(df: pd.DataFrame) -> list:
    """Return this DataFrame's spectral-band columns, sorted 380->730 nm.

    Matches both raw naming conventions found across wb and bb:
    `SPECTRAL_NM_380` (wb, CGATS) and `380nm` (bb, plain-header).
    """
    pat_prefix = re.compile(r'(?:SPECTRAL_)?NM_?(\d{3})$', re.I)   # ..._NM_380
    pat_suffix = re.compile(r'^(\d{3})NM$', re.I)                  # 380nm
    found = []
    for c in df.columns:
        m = pat_prefix.search(c) or pat_suffix.search(c)
        if m:
            found.append((int(m.group(1)), c))
    return [c for _, c in sorted(found)]


def spectral_to_xyz(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived XYZ_X/XYZ_Y/XYZ_Z (0-100) and LAB_L/LAB_A/LAB_B columns,
    integrated from the 36 spectral bands under D50/2 deg. Any native
    XYZ/Lab columns already on df (bb only) are left untouched -- they are a
    cross-check, not the pipeline output.
    """
    cols = spectral_cols(df)
    assert len(cols) == 36, f"expected 36 bands, found {len(cols)}"
    wl = np.arange(380, 740, 10)
    out = df.copy()
    xyz = np.array([
        colour.sd_to_XYZ(
            colour.SpectralDistribution(dict(zip(wl, row)), name='r'),
            cmfs=_CMF, illuminant=_D50)
        for row in df[cols].to_numpy(dtype=float)])
    out[['XYZ_X', 'XYZ_Y', 'XYZ_Z']] = xyz
    out[['LAB_L', 'LAB_A', 'LAB_B']] = xyz_to_lab(xyz)
    return out
