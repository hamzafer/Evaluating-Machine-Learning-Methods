import numpy as np
import colour

print(colour.__version__)

wl = np.arange(380, 731, 10)
print(len(wl), wl[0], wl[-1])

# fake reflectance = 1 everywhere -> should give white point XYZ (Y=100)
refl = np.ones_like(wl, dtype=float)
sd = colour.SpectralDistribution(dict(zip(wl, refl)))

cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
illum = colour.SDS_ILLUMINANTS['D50']

for method in ['ASTM E308', 'Integration']:
    try:
        XYZ = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illum, method=method)
        print(method, XYZ)
    except Exception as e:
        print(method, 'FAILED', e)

# D50 2-degree white point tristimulus values commonly used
print('CCS D50 2deg', colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'])
xy = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']
XYZ_w = colour.xy_to_XYZ(xy) * 100
print('XYZ_w scaled', XYZ_w)
