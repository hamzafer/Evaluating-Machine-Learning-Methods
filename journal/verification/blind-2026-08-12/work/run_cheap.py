import csv
import time
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from variants import build_variants
from cv import run_cv_model

OUT = "/home/user1/blind_verify/blind_results.csv"
MODELS = ["poly3", "svm", "knn"]


def main():
    variants = build_variants()
    write_header = not os.path.exists(OUT)
    f = open(OUT, "a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["dataset", "variant", "model", "median", "p95", "max", "n"])
        f.flush()

    for name, variant, X, XYZ in variants:
        for model in MODELS:
            t0 = time.time()
            metrics, dE, pred = run_cv_model(X, XYZ, model)
            dt = time.time() - t0
            writer.writerow(
                [name, variant, model, round(metrics["median"], 3), round(metrics["p95"], 3), round(metrics["max"], 3), metrics["n"]]
            )
            f.flush()
            print(f"{name:16s} {variant:6s} {model:18s} median={metrics['median']:.3f} p95={metrics['p95']:.3f} max={metrics['max']:.3f} n={metrics['n']} ({dt:.1f}s)", flush=True)
    f.close()


if __name__ == "__main__":
    main()
