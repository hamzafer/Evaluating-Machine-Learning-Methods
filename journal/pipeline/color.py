"""Color math for the journal pipeline. XYZ is ALWAYS on the 0-100 scale here."""
import numpy as np
import colour

# D50, 2-degree observer (CIE 15) — xy chromaticity for colour-science
_D50_xy = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']


def xyz_to_lab(xyz_100: np.ndarray) -> np.ndarray:
    """XYZ (0-100 scale) -> CIELAB, D50/2deg."""
    return colour.XYZ_to_Lab(np.asarray(xyz_100, dtype=float) / 100.0, illuminant=_D50_xy)


def delta_e00(xyz_pred_100: np.ndarray, xyz_true_100: np.ndarray) -> np.ndarray:
    """Per-sample CIEDE2000 between two sets of XYZ values on the 0-100 scale."""
    return colour.difference.delta_E(
        xyz_to_lab(xyz_true_100), xyz_to_lab(xyz_pred_100), method='CIE 2000')


def assert_lab_roundtrip(xyz_100: np.ndarray, lab_measured: np.ndarray,
                         name: str, tol_median: float = 0.6) -> None:
    """Tripwire: converting the dataset's own XYZ must reproduce its measured LAB.

    Guards against scale mistakes (e.g. normalized XYZ fed to the Lab
    conversion — the AIC 2025 flaw). Median |dLab| across the dataset must be
    small; instrument rounding in the CSVs keeps it from being exactly 0.
    """
    lab = xyz_to_lab(xyz_100)
    err = np.median(np.abs(lab - lab_measured))
    if err > tol_median:
        raise AssertionError(
            f"{name}: XYZ->Lab roundtrip median |dLab| = {err:.3f} > {tol_median}. "
            f"XYZ is not on the expected 0-100 scale or white point is wrong.")
