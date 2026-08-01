"""IFRA generalization experiments (wb runs). Run:
   .venv/bin/python -m journal.pipeline.run_ifra
A: within-run 5-fold CV, all 14 models, every wb run.
B: cross-run pairs — train run i, test run j (i != j), model subset.
C: leave-one-run-out — train on all wb runs but one, test on the held-out run.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import registry as dreg
from .evaluate import cross_validate, summarize, train_test
from .models import registry as mreg

SUBSET = ['gaussian_process', 'poly3', 'svm', 'mlp_deep']
OUT = Path(__file__).resolve().parents[1] / 'results' / 'ifra'


def wb_runs():
    return {k: v for k, v in dreg().items() if k.startswith('IFRA-wb-')}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {name: spec.load() for name, spec in wb_runs().items()}
    names = sorted(runs)

    # A: within-run, all models
    rows = []
    for n in names:
        for m in mreg():
            s = summarize(cross_validate(*runs[n], mreg()[m]))
            rows.append({'run': n, 'model': m,
                         **{k: round(v, 3) for k, v in s.items()}})
            print(f"A within-run {n} {m}: median={s['median']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'within_run.csv', index=False)

    # B: cross-run transfer, subset
    rows = []
    for a in names:
        for b in names:
            if a == b:
                continue
            for m in SUBSET:
                s = summarize(train_test(*runs[a], *runs[b], mreg()[m]))
                rows.append({'train': a, 'test': b, 'model': m,
                             **{k: round(v, 3) for k, v in s.items()}})
                print(f"B cross-run {a}->{b} {m}: median={s['median']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'cross_run.csv', index=False)

    # C: leave-one-run-out, subset
    rows = []
    for held in names:
        Xtr = np.vstack([runs[n][0] for n in names if n != held])
        Ytr = np.vstack([runs[n][1] for n in names if n != held])
        for m in SUBSET:
            s = summarize(train_test(Xtr, Ytr, *runs[held], mreg()[m]))
            rows.append({'held_out': held, 'model': m,
                         **{k: round(v, 3) for k, v in s.items()}})
            print(f"C LOO {held} {m}: median={s['median']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'leave_one_out.csv', index=False)


if __name__ == '__main__':
    main()
