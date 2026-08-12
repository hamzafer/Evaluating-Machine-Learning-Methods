import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import MinMaxScaler
from datasets import load_all
from cv import make_group_folds, fit_predict_gp
from colorimetry import XYZ100_to_Lab, delta_e00

ds = load_all()["CMYKOGV7"]
X = ds.X; XYZ = ds.XYZ
Xr = np.round(X, 6)
_, first_idx = np.unique(Xr, axis=0, return_index=True)
first_idx = np.sort(first_idx)
Xd = X[first_idx]; XYZd = XYZ[first_idx]
print("n=", len(Xd))

def worker(fold):
    os.environ["OMP_NUM_THREADS"]="1"
    row_fold, n_groups = make_group_folds(Xd)
    test_idx = np.where(row_fold==fold)[0]
    train_idx = np.where(row_fold!=fold)[0]
    xs = MinMaxScaler().fit(Xd[train_idx]); ys = MinMaxScaler().fit(XYZd[train_idx])
    Xtr = xs.transform(Xd[train_idx]); Xte = xs.transform(Xd[test_idx]); ytr = ys.transform(XYZd[train_idx])
    pred_scaled = fit_predict_gp(Xtr, ytr, Xte, seed=42)
    pred_xyz = ys.inverse_transform(pred_scaled)
    return fold, test_idx, pred_xyz

y_pred_all = np.empty_like(XYZd)
with ProcessPoolExecutor(max_workers=5) as ex:
    futs = [ex.submit(worker, f) for f in range(5)]
    for fut in as_completed(futs):
        fold, test_idx, pred_xyz = fut.result()
        y_pred_all[test_idx] = pred_xyz
        print("fold done", fold)

y_pred_all = np.clip(y_pred_all, 0, None)
Lab_true = XYZ100_to_Lab(XYZd)
Lab_pred = XYZ100_to_Lab(y_pred_all)
dE = delta_e00(Lab_true, Lab_pred)
print("gp dedup:", dict(median=float(np.median(dE)), p95=float(np.percentile(dE,95)), max=float(np.max(dE)), n=len(dE)))
