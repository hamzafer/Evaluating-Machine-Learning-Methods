import numpy as np
import colour
import time

wl = np.arange(380, 731, 10)
cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
illum = colour.SDS_ILLUMINANTS["D50"]

rng = np.random.default_rng(0)
n=2000
R = rng.uniform(0,1,size=(n, len(wl)))

# MultiSpectralDistributions expects data shape (n_wl, n_samples) with wavelengths as index
msds = colour.MultiSpectralDistributions(R.T, domain=wl)

t0=time.time()
XYZ_batch = colour.msds_to_XYZ(msds, cmfs=cmfs, illuminant=illum, method="ASTM E308")
t1=time.time()
print("batch time", t1-t0, XYZ_batch.shape)

t0=time.time()
XYZ_ref = np.empty((20,3))
for i in range(20):
    sd = colour.SpectralDistribution(dict(zip(wl, R[i])))
    XYZ_ref[i] = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illum, method="ASTM E308")
t1=time.time()
print("loop time for 20", t1-t0)

print("max abs diff (first 20):", np.max(np.abs(XYZ_batch[:20]-XYZ_ref)))
print(XYZ_batch[:3])
print(XYZ_ref[:3])
