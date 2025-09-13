from __future__ import annotations

from pathlib import Path
import pandas as pd


DATA_FILES = {
    'PC10': 'APTEC_PC10_CardBoard_2023_v1.csv',
    'PC11': 'APTEC_PC11_CCNB_2023_v1.csv',
    'FOGRA': 'FOGRA51.csv',
}


def cleaned_dir() -> Path:
    # repo_root/cleaned
    here = Path(__file__).resolve()
    return here.parents[2] / 'cleaned'


def load_dataset(name: str, *, k_zero_only: bool = False) -> pd.DataFrame:
    name = name.upper()
    if name not in DATA_FILES:
        raise ValueError(f"Unknown dataset '{name}'. Expected one of {list(DATA_FILES)}")
    path = cleaned_dir() / DATA_FILES[name]
    df = pd.read_csv(path)
    if k_zero_only and 'CMYK_K' in df.columns:
        df = df[df['CMYK_K'] == 0]
    return df.reset_index(drop=True)


def select_columns(df: pd.DataFrame, *, x_cols=('CMYK_C','CMYK_M','CMYK_Y'), y_cols=('XYZ_X','XYZ_Y','XYZ_Z')):
    X = df.loc[:, x_cols].astype(float).to_numpy()
    Y = df.loc[:, y_cols].astype(float).to_numpy()
    return X, Y

