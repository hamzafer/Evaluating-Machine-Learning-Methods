"""Batch-ingest IFRA 2005 newsprint wb+bb runs into processed CSVs.

Run: .venv/bin/python -m journal.pipeline.ingest_ifra

Follows the AMENDMENT in docs/plans/02-ifra-ingestion.md (post-Task-1
inventory), which overrides the plan's original Task-3 sketch:

1. wb (13 files) is genuine CGATS -> parse_cgats. bb (30 raw files) is a
   different, non-CGATS plain-header format -> parse_bb.
2. The 5 bb `lab*`-prefixed files are duplicate re-exports of Lab columns
   already present in a sibling bb file -- skipped, no new data. True bb
   run count: 25.
3. bb carries no CMYK columns at all. The SAMPLE_ID -> CMYK chart table is
   built from one wb file (byte-identical across all 13 -- confirmed in the
   Task-1 README) and joined into every bb run on SampleID. Every id must
   match; zero unmatched ids is asserted per file.
4. XYZ/Lab are derived from the spectral bands uniformly for ALL runs (wb
   and bb), never taken from a native column. Where a bb file also ships a
   native XYZ column (bb "Group A"), the native value is used only as a
   cross-check: delta_e00(derived, native) is computed via
   journal.pipeline.color.delta_e00 (not some other formula), and the
   median/max per file is printed. If any file's median exceeds 1.0 ΔE00,
   ingestion aborts -- that would mean our spectral integration disagrees
   with the instrument's own colorimetry by more than expected tolerance.
5. Output columns are exactly the standard schema shared with
   data/cleaned/*.csv: SAMPLE_ID, CMYK_C/M/Y/K, LAB_L/A/B, XYZ_X/Y/Z.
"""
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .color import delta_e00
from .ifra import parse_bb, parse_cgats, spectral_to_xyz

RAW = Path(__file__).resolve().parents[1] / 'data' / 'raw'
OUT = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'ifra'
KEEP = ['SAMPLE_ID', 'CMYK_C', 'CMYK_M', 'CMYK_Y', 'CMYK_K',
        'LAB_L', 'LAB_A', 'LAB_B', 'XYZ_X', 'XYZ_Y', 'XYZ_Z']

EXPECTED_SAMPLES = 1485
MEDIAN_ABORT_THRESHOLD = 1.0  # ΔE00 -- amendment #4


def _extract(zip_path: Path, dest: Path) -> list:
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return sorted(p for p in dest.iterdir() if p.is_file())


def _is_bb_duplicate(path: Path) -> bool:
    """The 5 `lab*`-prefixed bb files are duplicate Lab-only re-exports of a
    sibling bb file -- amendment #2. Never inferred from filename pattern
    alone elsewhere in this module (amendment #6), but this is the one place
    the raw README explicitly names the prefix as the skip signal."""
    return path.stem.lower().startswith('lab')


def _build_cmyk_table(wb_file: Path) -> pd.DataFrame:
    """SAMPLE_ID -> CMYK chart table from a single wb file. Confirmed
    byte-identical across all 13 wb files in the Task-1 inventory, so any one
    of them is a valid source of truth."""
    df = parse_cgats(wb_file)
    table = df[['SAMPLE_ID', 'CMYK_C', 'CMYK_M', 'CMYK_Y', 'CMYK_K']].copy()
    table['SAMPLE_ID'] = table['SAMPLE_ID'].astype(int)
    return table


def _process_wb(f: Path, out_dir: Path) -> int:
    df = spectral_to_xyz(parse_cgats(f))
    assert len(df) == EXPECTED_SAMPLES, f'{f.name}: expected {EXPECTED_SAMPLES} rows, got {len(df)}'
    df[KEEP].to_csv(out_dir / f'{f.stem}.csv', index=False)
    return len(df)


def _process_bb(f: Path, out_dir: Path, cmyk_table: pd.DataFrame) -> tuple:
    raw = parse_bb(f)
    df = spectral_to_xyz(raw)  # derived XYZ_X/Y/Z + LAB_L/A/B, native cols untouched

    df['SampleID'] = df['SampleID'].astype(int)
    merged = df.merge(cmyk_table, left_on='SampleID', right_on='SAMPLE_ID',
                       how='left', validate='one_to_one')
    n_unmatched = merged['CMYK_C'].isna().sum()
    assert n_unmatched == 0, f'{f.name}: {n_unmatched} unmatched SampleID(s) in CMYK join'
    assert len(merged) == EXPECTED_SAMPLES, f'{f.name}: expected {EXPECTED_SAMPLES} rows, got {len(merged)}'

    cross_check = None
    if {'XYZ-X', 'XYZ-Y', 'XYZ-Z'}.issubset(raw.columns):
        derived = merged[['XYZ_X', 'XYZ_Y', 'XYZ_Z']].to_numpy(dtype=float)
        native = merged[['XYZ-X', 'XYZ-Y', 'XYZ-Z']].to_numpy(dtype=float)
        de = delta_e00(derived, native)
        median, mx = float(np.median(de)), float(np.max(de))
        cross_check = (median, mx)
        print(f'  bb/{f.stem}: cross-check ΔE00(derived,native) median={median:.3f} max={mx:.3f}')
        if median > MEDIAN_ABORT_THRESHOLD:
            raise AssertionError(
                f'{f.name}: median ΔE00(derived,native) = {median:.3f} > '
                f'{MEDIAN_ABORT_THRESHOLD} -- aborting ingestion, spectral '
                f'integration disagrees with native XYZ beyond tolerance.')

    merged[KEEP].to_csv(out_dir / f'{f.stem}.csv', index=False)
    return len(merged), cross_check


def main():
    tmp_root = Path(tempfile.mkdtemp(prefix='ifra_ingest_'))
    try:
        wb_dir = OUT / 'wb'
        bb_dir = OUT / 'bb'
        wb_dir.mkdir(parents=True, exist_ok=True)
        bb_dir.mkdir(parents=True, exist_ok=True)

        wb_files = _extract(RAW / 'Ifra-wb.zip', tmp_root / 'wb')
        bb_files = _extract(RAW / 'Ifra-bb.zip', tmp_root / 'bb')

        assert len(wb_files) == 13, f'expected 13 wb files, found {len(wb_files)}'
        bb_real = [f for f in bb_files if not _is_bb_duplicate(f)]
        bb_skipped = [f for f in bb_files if _is_bb_duplicate(f)]
        assert len(bb_skipped) == 5, f'expected 5 bb duplicate files, found {len(bb_skipped)}'
        assert len(bb_real) == 25, f'expected 25 real bb files, found {len(bb_real)}'

        cmyk_table = _build_cmyk_table(wb_files[0])

        print(f'Processing {len(wb_files)} wb files...')
        for f in sorted(wb_files):
            n = _process_wb(f, wb_dir)
            print(f'  wb/{f.stem}: {n} samples')

        print(f'Skipping {len(bb_skipped)} bb duplicate (lab*) files: '
              f'{sorted(p.name for p in bb_skipped)}')
        print(f'Processing {len(bb_real)} real bb files...')
        cross_checks = {}
        for f in sorted(bb_real):
            n, cc = _process_bb(f, bb_dir, cmyk_table)
            print(f'  bb/{f.stem}: {n} samples')
            if cc is not None:
                cross_checks[f.stem] = cc

        if cross_checks:
            medians = [m for m, _ in cross_checks.values()]
            maxes = [x for _, x in cross_checks.values()]
            print(f'\nCross-check summary over {len(cross_checks)} bb files with '
                  f'native XYZ: median ΔE00 range [{min(medians):.3f}, {max(medians):.3f}], '
                  f'max ΔE00 range [{min(maxes):.3f}, {max(maxes):.3f}]')
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == '__main__':
    main()
