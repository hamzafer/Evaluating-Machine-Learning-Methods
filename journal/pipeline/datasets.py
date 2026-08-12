"""Dataset registry. A dataset declares its input channels (n-channel-generic):
the same code path serves 3-channel (CMY), 4-channel (CMYK), and future n>4 sets.

Design rule (agreed with Phil, Jul 2026): never drop an input column while
keeping its rows. CMY experiments use only K=0 rows; CMYK experiments use all
rows with K as an input.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .color import assert_lab_roundtrip

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEANED = REPO_ROOT / 'data' / 'cleaned'

_CSV = {
    'PC10': CLEANED / 'APTEC_PC10_CardBoard_2023_v1.csv',
    'PC11': CLEANED / 'APTEC_PC11_CCNB_2023_v1.csv',
    'FOGRA51': CLEANED / 'FOGRA51.csv',
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str                    # e.g. 'PC10-CMY'
    csv: Path
    input_cols: tuple            # declares n: 3, 4, or more
    filter_k_zero: bool          # True for CMY variants, False for CMYK. Raw row counts
                                 # are 818/1617; the coated specs also set dedup_exact,
                                 # so their EFFECTIVE counts are 795/1588.
    target_cols: tuple = ('XYZ_X', 'XYZ_Y', 'XYZ_Z')
    lab_cols: tuple = ('LAB_L', 'LAB_A', 'LAB_B')
    # Drop byte-identical duplicate rows (same inks AND same XYZ+Lab) at load.
    # The source CSV keeps every row as received; dedup is a modelling choice —
    # identical pairs would otherwise leak across CV folds and double-count in
    # the pooled median. On for every coated n<=4 set and CMYKOGV-7; off for
    # IFRA, whose duplicate recipes differ in value (genuine repeatability).
    dedup_exact: bool = False
    # Opt-in (Plan 06): run.py passes groups=make_groups(X) into cross_validate
    # (GroupKFold), so repeated recipes co-travel. The coated n<=4 and IFRA specs
    # stay on plain KFold (plan-01 verified equivalence there).
    grouped: bool = False

    def load(self):
        df = pd.read_csv(self.csv)
        if self.filter_k_zero:
            df = df[df['CMYK_K'] == 0].reset_index(drop=True)
        if self.dedup_exact:
            value_cols = list(self.input_cols) + list(self.target_cols) + list(self.lab_cols)
            df = df.drop_duplicates(subset=value_cols).reset_index(drop=True)
        X = df.loc[:, list(self.input_cols)].to_numpy(dtype=float)
        Y = df.loc[:, list(self.target_cols)].to_numpy(dtype=float)
        # Tripwire: the dataset's own XYZ must reproduce its measured Lab.
        assert_lab_roundtrip(Y, df.loc[:, list(self.lab_cols)].to_numpy(dtype=float), self.name)
        return X, Y


def registry() -> dict:
    specs = {}
    # Coated sets: exact-dedup at load, same policy as CMYKOGV-7. Their duplicate
    # recipes are byte-identical (an upstream averaging/duplication artifact, not
    # repeated measurement), so a test row's twin could sit in the training fold
    # under plain KFold -- measured leakage, see
    # docs/research/cv-leakage-2026-08-12.md. Dropping the twins removes both the
    # leakage and the pooled-median double-counting; 818 -> 795 (CMY),
    # 1617 -> 1588 (CMYK). No duplicate recipe survives, so grouped CV is
    # unnecessary by construction and the seeded shuffled KFold stays.
    for ds, csv in _CSV.items():
        specs[f'{ds}-CMY'] = DatasetSpec(
            name=f'{ds}-CMY', csv=csv,
            input_cols=('CMYK_C', 'CMYK_M', 'CMYK_Y'), filter_k_zero=True,
            dedup_exact=True)
        specs[f'{ds}-CMYK'] = DatasetSpec(
            name=f'{ds}-CMYK', csv=csv,
            input_cols=('CMYK_C', 'CMYK_M', 'CMYK_Y', 'CMYK_K'), filter_k_zero=False,
            dedup_exact=True)

    # IFRA 2005 newsprint (wb/bb kept separate per Phil). One DatasetSpec per
    # press-run CSV produced by journal.pipeline.ingest_ifra.
    # NOT deduplicated, deliberately: IFRA's duplicate recipes carry genuinely
    # DIFFERING measurements (real press repeatability, ~0.6-0.8 dE00), which is
    # signal about the press, not an artifact. Dropping them would destroy it.
    ifra_root = REPO_ROOT / 'journal' / 'data' / 'processed' / 'ifra'
    for backing in ('wb', 'bb'):
        for csv in sorted((ifra_root / backing).glob('*.csv')):
            name = f'IFRA-{backing}-{csv.stem}-CMYK'
            specs[name] = DatasetSpec(
                name=name, csv=csv,
                input_cols=('CMYK_C', 'CMYK_M', 'CMYK_Y', 'CMYK_K'),
                filter_k_zero=False)

    # n>4 colorant ladder (Plan 06), produced by journal.pipeline.ingest_ncolor.
    # All three use grouped CV (Apex's 20 genuine paper-white repeats co-travel;
    # catches any residual same-recipe rows). CMYKOGV-7's CSV keeps all 3534
    # rows as received (averaged export with redundant repeats); exact-dedup at
    # load drops the byte-identical ones -> 3302 effective rows, zero info loss.
    ncolor_root = REPO_ROOT / 'journal' / 'data' / 'processed' / 'ncolor'
    for ds, n_inks, dedup in (('KCMYG-5', 5, False),
                              ('CMYKOGV-7', 7, True),
                              ('CMYKOGB-7', 7, False)):
        specs[ds] = DatasetSpec(
            name=ds, csv=ncolor_root / f'{ds}.csv',
            input_cols=tuple(f'INK_{i}' for i in range(1, n_inks + 1)),
            filter_k_zero=False, dedup_exact=dedup, grouped=True)
    return specs
