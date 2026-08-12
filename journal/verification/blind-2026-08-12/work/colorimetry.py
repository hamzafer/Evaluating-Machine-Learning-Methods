import numpy as np
import colour

_XY_D50_2 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
_CMFS = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
_ILLUM_SD = colour.SDS_ILLUMINANTS["D50"]


def Lab_to_XYZ100(Lab):
    Lab = np.asarray(Lab, dtype=float)
    XYZ = colour.Lab_to_XYZ(Lab, illuminant=_XY_D50_2)
    return XYZ * 100.0


def XYZ100_to_Lab(XYZ100):
    XYZ100 = np.asarray(XYZ100, dtype=float)
    Lab = colour.XYZ_to_Lab(XYZ100 / 100.0, illuminant=_XY_D50_2)
    return Lab


def spectral_to_XYZ100(reflectance, wavelengths):
    """reflectance: (n_samples, n_wl) array in [0,1]. wavelengths: (n_wl,) array, e.g. 380..730 step 10.
    Returns (n_samples, 3) XYZ on 0-100 scale, D50/2-degree, ASTM E308 method.
    Uses colour.msds_to_XYZ batch API (verified bit-identical to per-sample sd_to_XYZ, ~1000x faster)."""
    reflectance = np.asarray(reflectance, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    msds = colour.MultiSpectralDistributions(reflectance.T, domain=wavelengths)
    XYZ = colour.msds_to_XYZ(msds, cmfs=_CMFS, illuminant=_ILLUM_SD, method="ASTM E308")
    return np.atleast_2d(XYZ)


def delta_e00(Lab1, Lab2):
    Lab1 = np.asarray(Lab1, dtype=float)
    Lab2 = np.asarray(Lab2, dtype=float)
    return colour.difference.delta_E_CIE2000(Lab1, Lab2)
