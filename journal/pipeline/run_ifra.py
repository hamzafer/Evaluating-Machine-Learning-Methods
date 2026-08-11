"""IFRA generalization experiments (wb runs). Run:
   .venv/bin/python -m journal.pipeline.run_ifra
A: within-run 5-fold CV, all 14 models, every wb run.
B: cross-run pairs — train run i, test run j (i != j), model subset.
C: leave-one-run-out — train on all wb runs but one, test on the held-out run.

Partial re-runs (Plan 10): --models / --parts / --runs restrict the sweep;
results merge-update into the existing CSVs (rows for the exact
run/pair/model combinations just computed are replaced, all others kept),
so a chunked or model-subset re-run never clobbers a fuller result set.
"""
import argparse
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


def merge_write(rows, path, keys):
    """Replace rows matching the just-computed key tuples; keep all others."""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        stale = old.set_index(keys).index.isin(new.set_index(keys).index)
        new = pd.concat([old[~stale], new], ignore_index=True)
    new.to_csv(path, index=False)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=None, choices=list(mreg()),
                    help='restrict all parts to these models '
                         '(default: A=all 14, B/C=the 4-model SUBSET)')
    ap.add_argument('--parts', nargs='*', default=['A', 'B', 'C'],
                    choices=['A', 'B', 'C'])
    ap.add_argument('--runs', nargs='*', default=None,
                    help="chunking filter: A's run / B's train run / C's "
                         'held-out run (default: all wb runs)')
    args = ap.parse_args()
    models_a = args.models if args.models else list(mreg())
    models_bc = args.models if args.models else SUBSET

    OUT.mkdir(parents=True, exist_ok=True)
    runs = {name: spec.load() for name, spec in wb_runs().items()}
    names = sorted(runs)
    picked = args.runs if args.runs else names
    for p in picked:
        assert p in names, f'unknown wb run {p}'

    # A: within-run
    if 'A' in args.parts:
        rows = []
        for n in picked:
            for m in models_a:
                s = summarize(cross_validate(*runs[n], mreg()[m]))
                rows.append({'run': n, 'model': m,
                             **{k: round(v, 3) for k, v in s.items()}})
                print(f"A within-run {n} {m}: median={s['median']:.3f}", flush=True)
        merge_write(rows, OUT / 'within_run.csv', ['run', 'model'])

    # B: cross-run transfer
    if 'B' in args.parts:
        rows = []
        for a in picked:
            for b in names:
                if a == b:
                    continue
                for m in models_bc:
                    s = summarize(train_test(*runs[a], *runs[b], mreg()[m]))
                    rows.append({'train': a, 'test': b, 'model': m,
                                 **{k: round(v, 3) for k, v in s.items()}})
                    print(f"B cross-run {a}->{b} {m}: median={s['median']:.3f}", flush=True)
        merge_write(rows, OUT / 'cross_run.csv', ['train', 'test', 'model'])

    # C: leave-one-run-out
    if 'C' in args.parts:
        rows = []
        for held in picked:
            Xtr = np.vstack([runs[n][0] for n in names if n != held])
            Ytr = np.vstack([runs[n][1] for n in names if n != held])
            for m in models_bc:
                s = summarize(train_test(Xtr, Ytr, *runs[held], mreg()[m]))
                rows.append({'held_out': held, 'model': m,
                             **{k: round(v, 3) for k, v in s.items()}})
                print(f"C LOO {held} {m}: median={s['median']:.3f}", flush=True)
        merge_write(rows, OUT / 'leave_one_out.csv', ['held_out', 'model'])


if __name__ == '__main__':
    main()
