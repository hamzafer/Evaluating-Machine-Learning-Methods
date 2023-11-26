import numpy as np


def d50():
    # Default D50 reference white in XYZ
    return np.array([96.422, 100.0, 82.521])


def xyz2lab(XYZ, XYZn=None):
    """
    Convert Lab color space values to XYZ color space.
    """
    if XYZn is None:
        rwhite = d50()
    else:
        rwhite = XYZn

    XYZ = XYZ / rwhite
    XYZ[XYZ > (6 / 29) ** 3] = XYZ[XYZ > (6 / 29) ** 3] ** (1 / 3)
    XYZ[XYZ <= (6 / 29) ** 3] = (XYZ[XYZ <= (6 / 29) ** 3] * (841 / 108)) + (4 / 29)

    L = (116 * XYZ[:, 1]) - 16
    a = 500 * (XYZ[:, 0] - XYZ[:, 1])
    b = 200 * (XYZ[:, 1] - XYZ[:, 2])

    Lab = np.column_stack((L, a, b))

    return Lab
