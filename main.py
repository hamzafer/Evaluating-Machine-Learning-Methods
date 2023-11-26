import numpy as np
from lab2xyz import lab2xyz

if __name__ == '__main__':

    # Example LAB values
    Lab_values = np.array([[48.02, 72.76, 7.22]])

    # Convert LAB to XYZ
    XYZ_values = lab2xyz(Lab_values)

    print("Converted XYZ values:", XYZ_values)
