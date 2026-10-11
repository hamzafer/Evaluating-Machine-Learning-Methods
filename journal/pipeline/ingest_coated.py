"""Convert the public coated characterization sets (PC10, PC11, FOGRA51) to the CSVs the loader reads.

Run from the repository root, one or more raw files at a time:

    python -m journal.pipeline.ingest_coated data/APTEC_PC10_CardBoard_2023_v1.txt \
        data/APTEC_PC11_CCNB_2023_v1.txt data/FOGRA51.txt

The originators' files (ICC characterization data registry for APTEC PC10 and PC11,
Fogra for FOGRA51) are CGATS text with SAMPLE_ID, CMYK_C/M/Y/K and LAB_L/A/B only.
This adds XYZ on the 0-100 scale, computed from the measured Lab with the ICC D50 white
(X, Y, Z = 0.96422, 1, 0.82521), and writes data/cleaned/<raw file stem>.csv with the
columns SAMPLE_ID, CMYK_C/M/Y/K, LAB_L/A/B, XYZ_X/Y/Z. Rows are kept as received;
the K=0 filter and the exact-duplicate removal happen at load (datasets.py).
"""
import sys
from pathlib import Path

import colour
import numpy as np

from .color import assert_lab_roundtrip
from .datasets import CLEANED, _CSV
from .ifra import parse_cgats

ICC_D50_XYZ = np.array([0.96422, 1.0, 0.82521])
COLUMNS = ['SAMPLE_ID', 'CMYK_C', 'CMYK_M', 'CMYK_Y', 'CMYK_K', 'LAB_L', 'LAB_A', 'LAB_B']


def convert(raw_path: Path):
    """Parse one raw CGATS file and return it with XYZ_X/Y/Z (0-100) added."""
    df = parse_cgats(raw_path)
    missing = set(COLUMNS) - set(df.columns)
    assert not missing, f'{raw_path.name}: missing columns {sorted(missing)}'
    df = df[COLUMNS].copy()
    lab = df[['LAB_L', 'LAB_A', 'LAB_B']].to_numpy(dtype=float)
    xyz = colour.Lab_to_XYZ(lab, colour.XYZ_to_xy(ICC_D50_XYZ)) * 100.0
    df[['XYZ_X', 'XYZ_Y', 'XYZ_Z']] = xyz
    assert_lab_roundtrip(xyz, lab, raw_path.stem)
    return df


def main(paths) -> None:
    if not paths:
        raise SystemExit(__doc__)
    expected = {p.name for p in _CSV.values()}
    CLEANED.mkdir(parents=True, exist_ok=True)
    for p in map(Path, paths):
        out = CLEANED / f'{p.stem}.csv'
        if out.name not in expected:
            print(f'  note: the loader expects one of {sorted(expected)}; '
                  f'{out.name} will not be read unless renamed')
        df = convert(p)
        # full precision: importing colour sets numpy's legacy print mode, which
        # would otherwise write 12 significant digits
        df.to_csv(out, index=False, float_format='%.17g')
        print(f'  {p.name}: {len(df)} rows -> {out}')


if __name__ == '__main__':
    main(sys.argv[1:])
