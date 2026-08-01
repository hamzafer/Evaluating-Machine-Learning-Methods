"""GP headline verification: plain KFold vs GroupKFold + measurement noise floor.
Run: .venv/bin/python -m journal.pipeline.verify_gp

INTERPRETATION NOTE (supersedes commit cfc4bb3's message): the noise_floor
rows are 0.000 because every duplicate-recipe pair in these CSVs is
byte-identical (XYZ equal to full float precision) — an upstream
averaging/duplication artifact, NOT evidence of perfect measurement
repeatability. These rows are therefore INVALID as a noise estimate; the
datasets contain no internal repeatability information. The paper must cite
typical spectrophotometer repeatability from literature (~0.1-0.3 dE00)
instead, and must not claim "at the measurement noise floor" from this CSV.
The grouped-vs-plain CV comparison is unaffected and remains the valid result.
"""
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
    rows = []
    for ds_name, spec in dreg().items():
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
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
