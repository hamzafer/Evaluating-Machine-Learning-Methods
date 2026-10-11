"""Convert the public coated characterization sets (PC10, PC11, FOGRA51) to the CSVs the loader reads.

Run from the repository root, one or more raw files at a time:

    python -m journal.pipeline.ingest_coated data/APTEC_PC10_CardBoard_2023_v1.txt \
        data/APTEC_PC11_CCNB_2023_v1.txt data/FOGRA51.txt

The originators' files (ICC characterization data registry for APTEC PC10 and PC11,
Fogra for FOGRA51) are CGATS text with SAMPLE_ID, CMYK_C/M/Y/K and LAB_L/A/B only.
This adds XYZ on the 0-100 scale, computed from the measured Lab with the ICC D50 white
(X, Y, Z = 96.422, 100, 82.521), and writes data/cleaned/<raw file stem>.csv with the
columns SAMPLE_ID, CMYK_C/M/Y/K, LAB_L/A/B, XYZ_X/Y/Z. Rows are kept as received;
the K=0 filter and the exact-duplicate removal happen at load (datasets.py).

The conversion is written out (lab_to_xyz100) with the same floating-point operations
as the script that made the CSVs used for the paper, and values are written in their
shortest exact form, so the rebuilt files load to the same numbers bit for bit.
"""
import sys
from pathlib import Path

import numpy as np

from .color import assert_lab_roundtrip
from .datasets import CLEANED, _CSV
from .ifra import parse_cgats

ICC_D50_XYZ_100 = np.array([96.422, 100.0, 82.521])
CIE_EPSILON, CIE_KAPPA = 216 / 24389, 24389 / 27
COLUMNS = ['SAMPLE_ID', 'CMYK_C', 'CMYK_M', 'CMYK_Y', 'CMYK_K', 'LAB_L', 'LAB_A', 'LAB_B']


def lab_to_xyz100(lab: np.ndarray) -> np.ndarray:
    """CIE 1976 Lab -> XYZ (0-100) for the ICC D50 white: f**3 * white, with the
    CIE linear segment below epsilon (no patch of PC10, PC11 or FOGRA51 reaches it)."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16) / 116
    f = np.column_stack((a / 500 + fy, fy, fy - b / 200))
    cube = f ** 3
    return np.where(cube > CIE_EPSILON, cube, (116 * f - 16) / CIE_KAPPA) * ICC_D50_XYZ_100


def convert(raw_path: Path):
    """Parse one raw CGATS file and return it with XYZ_X/Y/Z (0-100) added."""
    df = parse_cgats(raw_path)
    missing = set(COLUMNS) - set(df.columns)
    assert not missing, f'{raw_path.name}: missing columns {sorted(missing)}'
    df = df[COLUMNS].copy()
    lab = df[['LAB_L', 'LAB_A', 'LAB_B']].to_numpy(dtype=float)
    xyz = lab_to_xyz100(lab)
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
        # shortest exact form (repr), as pandas writes by default; set explicitly
        # because importing colour switches numpy to a 12-digit legacy print mode
        df.to_csv(out, index=False, float_format=lambda v: repr(float(v)))
        print(f'  {p.name}: {len(df)} rows -> {out}')


if __name__ == '__main__':
    main(sys.argv[1:])
