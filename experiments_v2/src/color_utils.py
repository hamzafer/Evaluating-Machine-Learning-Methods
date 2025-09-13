from __future__ import annotations

import numpy as np

try:
    from colour.difference import delta_E
except Exception:  # pragma: no cover
    from colour import delta_E  # type: ignore


def d50() -> np.ndarray:
    return np.array([96.422, 100.0, 82.521], dtype=float)


def xyz_to_lab(xyz: np.ndarray, whitepoint: np.ndarray | None = None) -> np.ndarray:
    """Convert XYZ (absolute) to Lab using the same formula as the repo (D50)."""
    rwhite = d50() if whitepoint is None else whitepoint
    XYZ = np.array(xyz, dtype=float) / rwhite
    threshold = (6 / 29) ** 3
    above = XYZ > threshold
    XYZ[above] = XYZ[above] ** (1 / 3)
    XYZ[~above] = (XYZ[~above] * (841 / 108)) + (4 / 29)

    L = (116 * XYZ[:, 1]) - 16
    a = 500 * (XYZ[:, 0] - XYZ[:, 1])
    b = 200 * (XYZ[:, 1] - XYZ[:, 2])
    return np.column_stack((L, a, b))


def delta_e00(lab_pred: np.ndarray, lab_true: np.ndarray) -> np.ndarray:
    """Compute CIEDE2000 per-sample errors."""
    return delta_E(lab_true, lab_pred, method='CIE 2000')


def summarize_errors(de: np.ndarray) -> dict:
    de = np.asarray(de, dtype=float)
    return {
        'mean': float(np.mean(de)),
        'median': float(np.median(de)),
        'max': float(np.max(de)),
        'p95': float(np.percentile(de, 95)),
        'sd': float(np.std(de, ddof=1)) if de.size > 1 else 0.0,
    }
