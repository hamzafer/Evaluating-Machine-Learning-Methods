import numpy as np

from utils.lab2xyz import lab2xyz
from utils.xyz2lab import xyz2lab

# Given LAB values
lab = np.array([[48.02, 72.76, 7.22]])

# Convert LAB to XYZ
xyz = lab2xyz(lab)

# Convert the obtained XYZ back to LAB
converted_lab = xyz2lab(xyz)

# Compare the results
print("Original Lab:", lab)
print("Converted Lab:", converted_lab)
