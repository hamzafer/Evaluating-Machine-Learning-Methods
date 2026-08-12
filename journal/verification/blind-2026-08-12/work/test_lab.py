import numpy as np
import colour

xy_D50 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]

# PC10 row 27 CMYK 0/0/0/0 -> LAB 95.08 1.04 -1.06 (near paper white)
Lab = np.array([95.08, 1.04, -1.06])
XYZ = colour.Lab_to_XYZ(Lab, illuminant=xy_D50)
print("XYZ (0-1 scale?):", XYZ)
print("XYZ*100:", XYZ*100)

Lab_back = colour.XYZ_to_Lab(XYZ, illuminant=xy_D50)
print("Lab back:", Lab_back)

# check delta E 2000 API
Lab2 = Lab + np.array([0.1,0.1,0.1])
de = colour.difference.delta_E_CIE2000(Lab, Lab2)
print("dE00 tiny diff:", de)
