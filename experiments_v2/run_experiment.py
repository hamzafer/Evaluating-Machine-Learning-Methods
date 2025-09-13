from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd

from experiments_v2.src.data import load_dataset, select_columns
from experiments_v2.src.models import registry
from experiments_v2.src.evaluate import evaluate_model


def main():
    ap = argparse.ArgumentParser(description="Run CMY→XYZ regression experiments with proper ΔE00 evaluation.")
    ap.add_argument('--dataset', choices=['PC10','PC11','FOGRA'], required=True)
    ap.add_argument('--mode', choices=['holdout','kfold'], default='holdout')
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--models', nargs='*', default=['random_forest','gradient_boost','knn','bayesian_gp','ridge','lasso','elastic','decision_tree','mlp_shallow','mlp_deep','svm','pcr','plsr','poly3'])
    ap.add_argument('--out', type=Path, default=Path('experiments_v2/results'))
    ap.add_argument('--deltae-mode', choices=['proper','legacy'], default='proper', help='ΔE00 computation: proper (denormalized XYZ) or legacy (on normalized XYZ).')
    ap.add_argument('--k-zero-only', action='store_true', help='Filter to K==0 rows only (if CMYK_K present).')
    args = ap.parse_args()

    out_dir = args.out / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset, k_zero_only=args.k_zero_only)
    X, Y = select_columns(df)

    reg = registry()
    summary_rows = []

    for key in args.models:
        ms = reg[key]
        print(f"Evaluating {ms.name} on {args.dataset} ({args.mode}, repeats={args.repeats})…")
        runs = evaluate_model(X, Y, ms.factory(), mode=args.mode, repeats=args.repeats, folds=args.folds, deltae_mode=args.deltae_mode)
        runs_path = out_dir / f"{key}_runs.csv"
        runs.to_csv(runs_path, index=False)

        # Summaries: best (min mean) and run-mean (averaged over runs)
        best = runs.sort_values('Mean Error', ascending=True).iloc[0]
        run_mean = runs[['Mean Error','Median Error','Max Error','P95 Error','Std Dev']].mean().to_dict()

        summary_rows.append({
            'Algorithm': ms.name,
            'Agg': 'best',
            'Mean': best['Mean Error'],
            'Median': best['Median Error'],
            'Max': best['Max Error'],
            'P95': best['P95 Error'],
            'SD': best['Std Dev'],
        })
        summary_rows.append({
            'Algorithm': ms.name,
            'Agg': 'run-mean',
            **{k: run_mean[k] for k in ['Mean Error','Median Error','Max Error','P95 Error','Std Dev']},
        })

    summary = pd.DataFrame(summary_rows)
    summary.rename(columns={'Mean Error':'Mean','Median Error':'Median','Max Error':'Max','P95 Error':'P95','Std Dev':'SD'}, inplace=True, errors='ignore')
    summary.to_csv(out_dir / 'summary.csv', index=False)
    print(f"Wrote: {out_dir/'summary.csv'}")


if __name__ == '__main__':
    main()
