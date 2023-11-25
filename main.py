import numpy as np

def d50():
    # Default D50 reference white in XYZ
    return np.array([96.422, 100.0, 82.521])

def lab2xyz(Lab, XYZn=None):
    """
    Convert Lab color space values to XYZ color space.
    """
    if XYZn is None:
        rwhite = d50()
    else:
        rwhite = XYZn

    k = 0.008856 ** (1 / 3)

    L, a, b = Lab[:, 0], Lab[:, 1], Lab[:, 2]
    Xn, Yn, Zn = rwhite

    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    X = fx ** 3 * Xn
    Y = fy ** 3 * Yn
    Z = fz ** 3 * Zn

    p = fx < k
    q = fy < k
    r = fz < k

    X[p] = ((fx[p] - 16 / 116) / 7.787) * Xn
    Y[q] = ((fy[q] - 16 / 116) / 7.787) * Yn
    Z[r] = ((fz[r] - 16 / 116) / 7.787) * Zn

    XYZ = np.column_stack((X, Y, Z))

    return XYZ

def print_hi(name):
    print(f'Hi, {name}')  # Greeting message

# Test the lab2xyz function
if __name__ == '__main__':
    print_hi('PyCharm')

    # Example LAB values
    Lab_values = np.array([[48.02, 72.76, 7.22]])

    # Convert LAB to XYZ
    XYZ_values = lab2xyz(Lab_values)

    print("Converted XYZ values:", XYZ_values)
