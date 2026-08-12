import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from datasets import load_all
from cv import run_cv_model

ds = load_all()["CMYKOGV7"]
X = ds.X
XYZ = ds.XYZ

Xr = np.round(X, 6)
_, first_idx = np.unique(Xr, axis=0, return_index=True)
first_idx = np.sort(first_idx)
print("deduped n =", len(first_idx))

Xd = X[first_idx]
XYZd = XYZ[first_idx]

for m in ["poly3", "knn", "svm"]:
    metrics, dE, pred = run_cv_model(Xd, XYZd, m)
    print(m, metrics)
