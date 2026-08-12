"""GP headline verification: plain KFold vs GroupKFold + measurement noise floor.
Run: .venv/bin/python -m journal.pipeline.verify_gp

INTERPRETATION NOTE (supersedes commit cfc4bb3's message): the coated sets'
noise_floor rows used to read 0.000 because every duplicate-recipe pair in
those CSVs is byte-identical (XYZ equal to full float precision) — an upstream
averaging/duplication artifact, NOT evidence of perfect measurement
repeatability. Those rows were INVALID as a noise estimate; the coated datasets
contain no internal repeatability information. The paper must cite typical
spectrophotometer repeatability from literature (~0.1-0.3 dE00) instead, and
must not claim "at the measurement noise floor" from this CSV.

Since the 12 Aug dedup fix the coated specs drop those byte-identical twins at
load, so they emit no noise_floor row at all — the misleading 0.000 rows are
gone rather than merely annotated. IFRA keeps its duplicates (they genuinely
differ) and its noise_floor rows remain meaningful.

What the coated grouped-vs-kfold rows now mean has changed. After dedup every
coated row is its own group, so GroupKFold performs no grouping: it simply
produces a different (deterministic, unshuffled) partition than
KFold(shuffle=True, seed=42). The two therefore do NOT coincide — they differ by
~0.000-0.008 dE00 — but that residual is ordinary partition-to-partition
variation, not leakage, and it is far smaller than the seed-to-seed spread of the
same models. Read these rows as a different-partition robustness check on the
coated sets, and as a genuine leakage check only on IFRA, where real duplicate
recipes still exist.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .color import delta_e00
from .datasets import registry as dreg
from .evaluate import cross_validate, make_groups, summarize
from .models import registry as mreg

MODELS = ['gaussian_process', 'poly3']          # headline + baseline
OUT = Path(__file__).resolve().parents[1] / 'results' / 'gp_verification.csv'


def noise_floor(X, Y):
    """DE00 between measurements that share an identical input recipe."""
    groups = make_groups(X)
    pairs = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        pairs += [delta_e00(Y[i:i+1], Y[j:j+1])[0] for i, j in combinations(idx, 2)]
    return np.array(pairs)


def main():
    reg = dreg()
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=list(reg), choices=list(reg),
                    help='subset to recompute; other datasets keep their existing '
                         'CSV rows verbatim (merge-update, as in run.py)')
    args = ap.parse_args()

    rows = []
    for ds_name in args.datasets:
        spec = reg[ds_name]
        X, Y = spec.load()
        groups = make_groups(X)
        nf = noise_floor(X, Y)
        if nf.size:
            rows.append({'dataset': ds_name, 'model': 'noise_floor', 'cv': 'pairs',
                         **{k: round(v, 3) for k, v in summarize(nf).items() if k != 'n'},
                         'n': nf.size})
        for m in MODELS:
            for cv, g in [('kfold', None), ('grouped', groups)]:
                s = summarize(cross_validate(X, Y, mreg()[m], groups=g))
                rows.append({'dataset': ds_name, 'model': m, 'cv': cv,
                             **{k: round(v, 3) for k, v in s.items() if k != 'n'},
                             'n': s['n']})
                print(f"{ds_name:14s} {m:16s} {cv:8s} median={s['median']:.3f}", flush=True)
    new = pd.DataFrame(rows)
    if OUT.exists():
        # Merge-update: replace rows for datasets just recomputed, keep the rest
        # byte-identical. A partial --datasets run must never clobber the CSV.
        old = pd.read_csv(OUT)
        new = pd.concat([old[~old.dataset.isin(new.dataset)], new], ignore_index=True)
    new.to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
