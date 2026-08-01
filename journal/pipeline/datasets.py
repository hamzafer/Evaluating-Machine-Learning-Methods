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
    filter_k_zero: bool          # True for CMY variants (818 rows), False for CMYK (1617)
    target_cols: tuple = ('XYZ_X', 'XYZ_Y', 'XYZ_Z')
    lab_cols: tuple = ('LAB_L', 'LAB_A', 'LAB_B')

    def load(self):
        df = pd.read_csv(self.csv)
        if self.filter_k_zero:
            df = df[df['CMYK_K'] == 0].reset_index(drop=True)
        X = df.loc[:, list(self.input_cols)].to_numpy(dtype=float)
        Y = df.loc[:, list(self.target_cols)].to_numpy(dtype=float)
        # Tripwire: the dataset's own XYZ must reproduce its measured Lab.
        assert_lab_roundtrip(Y, df.loc[:, list(self.lab_cols)].to_numpy(dtype=float), self.name)
        return X, Y


def registry() -> dict:
    specs = {}
    for ds, csv in _CSV.items():
        specs[f'{ds}-CMY'] = DatasetSpec(
            name=f'{ds}-CMY', csv=csv,
            input_cols=('CMYK_C', 'CMYK_M', 'CMYK_Y'), filter_k_zero=True)
        specs[f'{ds}-CMYK'] = DatasetSpec(
            name=f'{ds}-CMYK', csv=csv,
            input_cols=('CMYK_C', 'CMYK_M', 'CMYK_Y', 'CMYK_K'), filter_k_zero=False)
    return specs
