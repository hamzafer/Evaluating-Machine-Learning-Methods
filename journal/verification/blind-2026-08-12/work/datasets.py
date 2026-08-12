"""Load each raw file into a standard structure: ink inputs (0-100), XYZ100 target,
plus whatever native fields are available for consistency checking."""
import numpy as np
import pandas as pd
from parse import parse_cgats_like
from colorimetry import Lab_to_XYZ100, XYZ100_to_Lab, spectral_to_XYZ100

DATA_DIR = "/home/user1/blind_verify/data"

SPECTRAL_WL = np.arange(380, 731, 10)


class DS:
    def __init__(self, name, ink_cols, df, header, meta, kind):
        self.name = name
        self.ink_cols = ink_cols
        self.df = df
        self.header = header
        self.meta = meta
        self.kind = kind  # 'lab_only', 'xyz_lab_native', 'spectral_only'
        self.X = df[ink_cols].to_numpy(dtype=float)  # 0-100 scale, as-is

        self.native_Lab = None
        self.native_XYZ = None
        self.spectral = None
        self.wavelengths = None

        if kind == "lab_only":
            self.native_Lab = df[["LAB_L", "LAB_A", "LAB_B"]].to_numpy(dtype=float)
            self.XYZ = Lab_to_XYZ100(self.native_Lab)
        elif kind == "xyz_lab_native":
            self.native_XYZ = df[["XYZ_X", "XYZ_Y", "XYZ_Z"]].to_numpy(dtype=float)
            self.native_Lab = df[["LAB_L", "LAB_A", "LAB_B"]].to_numpy(dtype=float)
            self.XYZ = self.native_XYZ.copy()
        elif kind == "spectral_only":
            spec_cols = [c for c in df.columns if c.lower().startswith(("nm", "spectral_nm"))]
            self.spectral = df[spec_cols].to_numpy(dtype=float)
            self.wavelengths = SPECTRAL_WL
            assert self.spectral.shape[1] == len(SPECTRAL_WL), (self.name, self.spectral.shape)
            self.XYZ = spectral_to_XYZ100(self.spectral, self.wavelengths)
        else:
            raise ValueError(kind)

        self.XYZ = np.clip(self.XYZ, 0, None)  # physical tristimulus can't be negative
        self.Lab = XYZ100_to_Lab(self.XYZ)


def load_pc10():
    hl, df, meta = parse_cgats_like(f"{DATA_DIR}/APTEC_PC10_CardBoard_2023_v1.txt")
    return DS("PC10", ["CMYK_C", "CMYK_M", "CMYK_Y", "CMYK_K"], df, hl, meta, "lab_only")


def load_pc11():
    hl, df, meta = parse_cgats_like(f"{DATA_DIR}/APTEC_PC11_CCNB_2023_v1.txt")
    return DS("PC11", ["CMYK_C", "CMYK_M", "CMYK_Y", "CMYK_K"], df, hl, meta, "lab_only")


def load_fogra51():
    hl, df, meta = parse_cgats_like(f"{DATA_DIR}/FOGRA51.txt")
    return DS("FOGRA51", ["CMYK_C", "CMYK_M", "CMYK_Y", "CMYK_K"], df, hl, meta, "lab_only")


def load_kcmyg():
    hl, df, meta = parse_cgats_like(f"{DATA_DIR}/KCMYG_5clr_spectral.txt")
    return DS("KCMYG5", ["5CLR_1", "5CLR_2", "5CLR_3", "5CLR_4", "5CLR_5"], df, hl, meta, "spectral_only")


def load_cmykogv():
    hl, df, meta = parse_cgats_like(f"{DATA_DIR}/APTEC_CMYKOGV_7clr_xyzlab.txt")
    return DS("CMYKOGV7", [f"7CLR_{i}" for i in range(1, 8)], df, hl, meta, "xyz_lab_native")


def load_cmykogb():
    hl, df, meta = parse_cgats_like(f"{DATA_DIR}/Apex_CMYKOGB_7clr_spectral.txt")
    return DS("CMYKOGB7", [f"7CLR_{i}" for i in range(1, 8)], df, hl, meta, "spectral_only")


def load_ifra_age():
    hl, df, meta = parse_cgats_like(f"{DATA_DIR}/Age_64a_wb.txt")
    return DS("IFRA_Age64a", ["CMYK_C", "CMYK_M", "CMYK_Y", "CMYK_K"], df, hl, meta, "spectral_only")


def load_ifra_pressj():
    hl, df, meta = parse_cgats_like(f"{DATA_DIR}/PressJ_158_wb.txt")
    return DS("IFRA_PressJ158", ["CMYK_C", "CMYK_M", "CMYK_Y", "CMYK_K"], df, hl, meta, "spectral_only")


ALL_LOADERS = {
    "PC10": load_pc10,
    "PC11": load_pc11,
    "FOGRA51": load_fogra51,
    "KCMYG5": load_kcmyg,
    "CMYKOGV7": load_cmykogv,
    "CMYKOGB7": load_cmykogb,
    "IFRA_Age64a": load_ifra_age,
    "IFRA_PressJ158": load_ifra_pressj,
}


def load_all():
    return {k: v() for k, v in ALL_LOADERS.items()}


if __name__ == "__main__":
    for name, ds in load_all().items():
        print(name, ds.X.shape, ds.XYZ.shape, "declared_sets=", ds.meta["n_declared_sets"], "actual_rows=", ds.meta["n_actual_rows"])
