"""Run journal experiments.

  python -m journal.pipeline.run                      # all datasets, all models
  python -m journal.pipeline.run --datasets PC10-CMY  # subset
Writes journal/results/<dataset>/summary.csv (one row per model).
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from .datasets import registry as dataset_registry
from .evaluate import cross_validate, summarize
from .models import registry as model_registry

RESULTS = Path(__file__).resolve().parents[1] / 'results'


def main():
    ap = argparse.ArgumentParser()
    ds_reg, m_reg = dataset_registry(), model_registry()
    ap.add_argument('--datasets', nargs='*', default=list(ds_reg), choices=list(ds_reg))
    ap.add_argument('--models', nargs='*', default=list(m_reg), choices=list(m_reg))
    args = ap.parse_args()

    for ds_name in args.datasets:
        spec = ds_reg[ds_name]
        X, Y = spec.load()
        rows = []
        for m_name in args.models:
            t0 = time.time()
            de = cross_validate(X, Y, m_reg[m_name])
            stats = summarize(de)
            rows.append({'model': m_name, **{k: round(v, 3) for k, v in stats.items()
                                             if k != 'n'}, 'n': stats['n']})
            print(f"{ds_name:14s} {m_name:16s} median={stats['median']:7.3f} "
                  f"p95={stats['p95']:7.3f} max={stats['max']:8.3f} "
                  f"[{time.time()-t0:5.1f}s]", flush=True)
        out_dir = RESULTS / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values('median').to_csv(out_dir / 'summary.csv', index=False)
        print(f"wrote {out_dir/'summary.csv'}", flush=True)


if __name__ == '__main__':
    main()
