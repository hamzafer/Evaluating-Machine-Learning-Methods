import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from variants import build_variants
from cv import build_poly3, build_knn, fit_predict_svm, evaluate_pooled_dE00

variants = {(n, v): (X, XYZ) for n, v, X, XYZ in build_variants()}


def run_ungrouped(X, XYZ, model_name, seed=42, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred_all = np.empty_like(XYZ)
    for train_idx, test_idx in kf.split(X):
        xs = MinMaxScaler().fit(X[train_idx])
        ys = MinMaxScaler().fit(XYZ[train_idx])
        Xtr = xs.transform(X[train_idx])
        Xte = xs.transform(X[test_idx])
        ytr = ys.transform(XYZ[train_idx])
        if model_name == "poly3":
            m = build_poly3()
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
        elif model_name == "knn":
            m = build_knn()
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
        elif model_name == "svm":
            pred = fit_predict_svm(Xtr, ytr, Xte)
        pred_xyz = ys.inverse_transform(pred)
        y_pred_all[test_idx] = pred_xyz
    metrics, _ = evaluate_pooled_dE00(XYZ, y_pred_all)
    return metrics


for dsname in [("PC10", "CMY"), ("PC10", "CMYK"), ("PC11", "CMY"), ("FOGRA51", "CMY")]:
    X, XYZ = variants[dsname]
    print("===", dsname, "===")
    for m in ["poly3", "knn", "svm"]:
        met = run_ungrouped(X, XYZ, m)
        med = met["median"]
        p95 = met["p95"]
        mx = met["max"]
        print(f"  ungrouped {m:6s} median={med:.3f} p95={p95:.3f} max={mx:.3f}")
