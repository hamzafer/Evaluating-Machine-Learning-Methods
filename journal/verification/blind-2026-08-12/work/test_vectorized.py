import numpy as np
import colour
import time

wl = np.arange(380, 731, 10)
cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
illum = colour.SDS_ILLUMINANTS["D50"]

cmfs_vals = cmfs.copy().align(colour.SpectralShape(380,730,10)).values  # (n_wl,3)
illum_vals = illum.copy().align(colour.SpectralShape(380,730,10)).values  # (n_wl,)

k = 100.0 / np.sum(illum_vals * cmfs_vals[:,1])
W = illum_vals[:,None] * cmfs_vals  # (n_wl,3)

rng = np.random.default_rng(0)
R = rng.uniform(0,1,size=(2000, len(wl)))

t0=time.time()
XYZ_vec = k * (R @ W)
t1=time.time()
print("vectorized time", t1-t0)

# compare against per-sample sd_to_XYZ for a subset
t0=time.time()
XYZ_ref = np.empty((20,3))
for i in range(20):
    sd = colour.SpectralDistribution(dict(zip(wl, R[i])))
    XYZ_ref[i] = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illum, method="ASTM E308")
t1=time.time()
print("per-sample time for 20:", t1-t0)

print("max abs diff:", np.max(np.abs(XYZ_vec[:20]-XYZ_ref)))
print(XYZ_vec[:3])
print(XYZ_ref[:3])
