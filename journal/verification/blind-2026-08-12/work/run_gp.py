import os
import sys
import csv
import time
import json

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import MinMaxScaler

from variants import build_variants
from cv import make_group_folds, fit_predict_gp
from colorimetry import XYZ100_to_Lab, delta_e00

OUT_CSV = "/home/user1/blind_verify/blind_results.csv"
PRED_DIR = "/home/user1/blind_verify/work/gp_preds"
os.makedirs(PRED_DIR, exist_ok=True)


def _worker(args):
    name, variant, fold, X, XYZ = args
    os.environ["OMP_NUM_THREADS"] = "1"
    row_fold, n_groups = make_group_folds(X)
    test_idx = np.where(row_fold == fold)[0]
    train_idx = np.where(row_fold != fold)[0]
    x_scaler = MinMaxScaler().fit(X[train_idx])
    y_scaler = MinMaxScaler().fit(XYZ[train_idx])
    Xtr = x_scaler.transform(X[train_idx])
    Xte = x_scaler.transform(X[test_idx])
    ytr = y_scaler.transform(XYZ[train_idx])
    t0 = time.time()
    pred_scaled = fit_predict_gp(Xtr, ytr, Xte, seed=42)
    dt = time.time() - t0
    pred_xyz = y_scaler.inverse_transform(pred_scaled)
    return name, variant, fold, test_idx, pred_xyz, dt


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    variants = build_variants()
    tasks = []
    for name, variant, X, XYZ in variants:
        for fold in range(5):
            tasks.append((name, variant, fold, X, XYZ))

    # sort so the known-slowest (largest train fold) tasks go first -> better load balancing
    tasks.sort(key=lambda t: -t[3].shape[0])

    results_by_variant = {}  # (name,variant) -> {fold: (test_idx, pred_xyz)}
    xyz_by_variant = {(n, v): XYZ for n, v, X, XYZ in variants}

    print(f"Launching {len(tasks)} GP fold-fits with {max_workers} workers", flush=True)
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        done_count = 0
        for fut in as_completed(futs):
            name, variant, fold, test_idx, pred_xyz, dt = fut.result()
            results_by_variant.setdefault((name, variant), {})[fold] = (test_idx, pred_xyz)
            done_count += 1
            elapsed = time.time() - t_start
            print(f"[{done_count}/{len(tasks)}] {name} {variant} fold={fold} took {dt:.1f}s (elapsed {elapsed:.0f}s)", flush=True)

            key = (name, variant)
            if len(results_by_variant[key]) == 5:
                XYZ = xyz_by_variant[key]
                y_pred_all = np.empty_like(XYZ)
                for f, (ti, px) in results_by_variant[key].items():
                    y_pred_all[ti] = px
                y_pred_all_clipped = np.clip(y_pred_all, 0, None)
                Lab_true = XYZ100_to_Lab(XYZ)
                Lab_pred = XYZ100_to_Lab(y_pred_all_clipped)
                dE = delta_e00(Lab_true, Lab_pred)
                metrics = dict(
                    median=float(np.median(dE)), p95=float(np.percentile(dE, 95)),
                    max=float(np.max(dE)), n=len(dE),
                )
                print(f">>> COMPLETE {name} {variant} gaussian_process: {metrics}", flush=True)
                write_header = not os.path.exists(OUT_CSV)
                with open(OUT_CSV, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(["dataset", "variant", "model", "median", "p95", "max", "n"])
                    w.writerow([name, variant, "gaussian_process",
                                round(metrics["median"], 3), round(metrics["p95"], 3),
                                round(metrics["max"], 3), metrics["n"]])
                np.save(f"{PRED_DIR}/{name}_{variant}_gp_pred.npy", y_pred_all)

    print("ALL GP DONE", flush=True)


if __name__ == "__main__":
    main()
