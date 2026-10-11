"""Revision R3, editor point 3: is the degree-5 vs Gaussian process result robust
to the fold assignment?

The paper's folds are fixed: on the n>4 sets, unshuffled GroupKFold on the recipe
groups; on the coated sets, shuffled KFold with seed 42. This script reruns the
round-3 comparison with the fold assignment varied over the paper's shuffle-seed
set 0/1/2/3/42 (the same set as the coated-set check in the Reproducibility
paragraph) and keeps everything else fixed (data, models, model seeds, scalers,
DE00 path).

Fold conditions per dataset ("tag"):
  published   the paper's folds (evaluate.fold_splits), asserted equal to the
              published fold file revision_r2/folds/<ds>.csv. This is the control:
              every result under it must equal the published value.
  s0..s3,s42  grouped sets: GroupKFold(5, shuffle=True, random_state=s), which
              shuffles the recipe groups and splits them into five near-equal
              parts (scikit-learn >= 1.6); repeated recipes still share a fold.
              KFold sets (PC10-CMYK): KFold(5, shuffle=True, random_state=s);
              s42 is then the published split and is not rerun (merge copies it).

Model parts (one job = dataset x tag x part):
  gp     the cube-root Gaussian process (registry 'gaussian_process_cbrt'), the
         expensive one (about 10 to 17 min per dataset and tag).
  cheap  polynomials of degree 2..5 in XYZ and cube-root space (revision_r3.factory),
         every fixed configuration of the n>4 model table (models.registry) except
         the XYZ Gaussian process and the two DE00-refined polynomials, and the
         twelve cube-root variants (models.cbrt_registry). Not rerun (cost): the XYZ
         Gaussian process, poly3_de00_nm, poly3_de00_powell and the tuned models.
  poly   only the polynomial sweep (used for PC10-CMYK, where only degree 4/5 vs
         the Gaussian process is asked).

  python -m journal.pipeline.revision_r3_seeds jobs            # one job per line
  python -m journal.pipeline.revision_r3_seeds run --dataset KCMYG-5 --tag s0 --part gp
  python -m journal.pipeline.revision_r3_seeds merge           # summary, paired, reliability, controls

Outputs (journal/results/revision_r3/seeds/):
  summary.csv      dataset, tag, model, median, p95, max (pooled held-out DE00)
  persample/<ds>__<tag>.csv  idx, fold and per-sample DE00 of the polynomial sweep and
                   the cube-root GP (the inputs of the paired test)
  paired.csv       per tag: a = poly5 cbrt (and poly4 cbrt), b = GP cbrt; the
                   Table-11 test of revision_r1_tables.paired (bootstrap B=10000,
                   rng seed 42; Wilcoxon), Holm across the four datasets per tag
  reliability.csv  per tag and n>4 dataset: best median / P95 / max over the fixed
                   configurations run here, the GP's ratio to each best, and the
                   closest other configuration's worst ratio
  controls.csv     the published tag against the published CSVs
Job outputs go to seeds/_jobs/ (not committed; merge reads them).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from .color import delta_e00
from .datasets import registry as dataset_registry
from .evaluate import FOLDS, fold_splits, make_groups
from .models import cbrt_registry, registry as model_registry
from .revision_r1_tables import SEED as BOOT_SEED, holm, paired, persample as r1_persample
from .revision_r3 import DEGREES, SPACES, factory as poly_factory, published_folds

RES = Path(__file__).resolve().parents[1] / 'results'
OUT = RES / 'revision_r3' / 'seeds'
JOBS = OUT / '_jobs'
NGT4 = ('CMYKOGV-7', 'KCMYG-5', 'CMYKOGB-7')
DATASETS = NGT4 + ('PC10-CMYK',)
SEEDS = (0, 1, 2, 3, 42)
TAGS = ('published',) + tuple(f's{s}' for s in SEEDS)
STATS = {'median': np.median, 'p95': lambda v: np.percentile(v, 95), 'max': np.max}
GP = 'gaussian_process_cbrt'
SKIPPED = ('gaussian_process', 'poly3_de00_nm', 'poly3_de00_powell')
# Table n7 (n>4 model table): the fixed configurations of models.registry plus the
# two cube-root ones it shows (svm, deep MLP). The degree table adds the sweep.
N7_EXTRA_CBRT = ('svm_cbrt', 'mlp_deep_cbrt')


def sweep_name(degree, space):
    return f'poly{degree}' + ('_cbrt' if space == 'cbrt' else '')


def poly_models():
    return {sweep_name(d, s): poly_factory(d, s) for d in DEGREES for s in SPACES}


def cheap_models():
    reg = model_registry()
    out = poly_models()
    for name, f in reg.items():
        if name in SKIPPED or name == GP or name in out:
            continue
        out[name] = f
    out.update(cbrt_registry())
    return out


def part_models(part):
    if part == 'gp':
        return {GP: model_registry()[GP]}
    if part == 'poly':
        return poly_models()
    return cheap_models()


def splits_for(name, tag):
    """(X, Y, splits, fold vector) for a dataset under a fold condition."""
    spec = dataset_registry()[name]
    X, Y = spec.load()
    groups = make_groups(X) if spec.grouped else None
    if tag == 'published':
        splits, fold = published_folds(name, X, groups)
        return X, Y, splits, fold
    seed = int(tag[1:])
    if groups is None:
        if seed == 42:
            raise SystemExit(f'{name} {tag} is the published split; run tag published')
        it = KFold(n_splits=FOLDS, shuffle=True, random_state=seed).split(X)
    else:
        it = GroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed).split(X, groups=groups)
    splits = [(tr, te) for tr, te in it]
    fold = np.empty(len(X), dtype=int)
    for i, (_, te) in enumerate(splits):
        fold[te] = i
    if groups is not None:   # repeated recipes must share a fold
        assert (pd.Series(fold).groupby(groups).nunique() == 1).all(), name
    return X, Y, splits, fold


def cv(X, Y, splits, model_factory):
    """evaluate.cross_validate with explicit splits (same steps, same order)."""
    from sklearn.preprocessing import MinMaxScaler
    de = np.empty(len(X))
    for tr, te in splits:
        sx = MinMaxScaler().fit(X[tr])
        sy = MinMaxScaler().fit(Y[tr])
        model = model_factory()
        if hasattr(model, 'set_scaler'):
            model.set_scaler(sy)
        model.fit(sx.transform(X[tr]), sy.transform(Y[tr]))
        pred = sy.inverse_transform(np.asarray(model.predict(sx.transform(X[te]))))
        de[te] = delta_e00(np.clip(pred, 0.0, None), Y[te])
    return de


def job_path(name, tag, part):
    return JOBS / f'{name}__{tag}__{part}.csv'


def cmd_jobs(args):
    for name in DATASETS:
        parts = ('poly', 'gp') if name == 'PC10-CMYK' else ('cheap', 'gp')
        for tag in TAGS:
            if name == 'PC10-CMYK' and tag == 's42':
                continue
            for part in parts:
                print(f'--dataset {name} --tag {tag} --part {part}')


def cmd_run(args):
    JOBS.mkdir(parents=True, exist_ok=True)
    X, Y, splits, fold = splits_for(args.dataset, args.tag)
    out = pd.DataFrame({'idx': np.arange(len(X)), 'fold': fold})
    secs = {}
    for m, f in part_models(args.part).items():
        t0 = time.perf_counter()
        out[m] = np.round(cv(X, Y, splits, f), 6)
        secs[m] = round(time.perf_counter() - t0, 1)
        print(f'{args.dataset} {args.tag} {m}: median {np.median(out[m]):.3f} '
              f'p95 {np.percentile(out[m], 95):.3f} max {out[m].max():.3f} ({secs[m]}s)', flush=True)
    p = job_path(args.dataset, args.tag, args.part)
    out.to_csv(p.with_suffix('.tmp'), index=False)
    p.with_suffix('.tmp').rename(p)
    p.with_suffix('.json').write_text(json.dumps(secs))


def load_tag(name, tag):
    """Wide per-sample frame of one dataset/tag (all parts), or None if incomplete."""
    src = 'published' if (name == 'PC10-CMYK' and tag == 's42') else tag
    frames = [pd.read_csv(p) for p in sorted(JOBS.glob(f'{name}__{src}__*.csv'))]
    if not frames:
        return None
    df = frames[0]
    for f in frames[1:]:
        assert (f.idx.values == df.idx.values).all() and (f.fold.values == df.fold.values).all()
        df = df.merge(f.drop(columns='fold'), on='idx')
    return df


def published_value(name, model):
    """Published (median, p95, max) for a model: summary.csv, then R1 summaries,
    then the R3 degree sweep (degrees 2 and 5)."""
    s = pd.read_csv(RES / name / 'summary.csv').set_index('model')
    if model in s.index:
        return s.loc[model, ['median', 'p95', 'max']].astype(float).to_numpy(), f'{name}/summary.csv'
    p = RES / 'revision_r1' / 'summary' / f'{name}__{model}.csv'
    if p.exists():
        r = pd.read_csv(p).iloc[0]
        return r[['median', 'p95', 'max']].astype(float).to_numpy(), f'revision_r1/summary/{p.name}'
    sw = pd.read_csv(RES / 'revision_r3' / 'degree_sweep_train_vs_heldout.csv')
    d, space = int(model[4]), ('cbrt' if model.endswith('_cbrt') else 'xyz')
    r = sw[(sw.dataset == name) & (sw.degree == d) & (sw.space == space)].iloc[0]
    return r[['heldout_median', 'heldout_p95', 'heldout_max']].astype(float).to_numpy(), \
        'revision_r3/degree_sweep_train_vs_heldout.csv'


def published_persample(name, model):
    ps = pd.read_csv(RES / 'revision_r3' / 'persample_degree_sweep.csv')
    ps = ps[ps.dataset == name].sort_values('idx')
    if model.startswith('poly') and model[4].isdigit() and 'de00' not in model:
        d, space = model[4], ('cbrt' if model.endswith('_cbrt') else 'xyz')
        return ps[f'd{d}_{space}'].to_numpy()
    return r1_persample(name, model)


def r3(x):
    return round(float(x), 3)


def cmd_merge(args):
    summary, controls, persample_frames = [], [], {}
    for name in DATASETS:
        for tag in TAGS:
            df = load_tag(name, tag)
            if df is None:
                print('missing', name, tag)
                continue
            models = [c for c in df.columns if c not in ('idx', 'fold')]
            for m in models:
                v = df[m].to_numpy()
                summary.append({'dataset': name, 'tag': tag, 'model': m, 'n': len(v),
                                **{k: float(f(v)) for k, f in STATS.items()}})
            keep = ['idx', 'fold'] + [c for c in models if c == GP or
                                      (c.startswith('poly') and 'de00' not in c)]
            persample_frames[(name, tag)] = df[keep]
            if tag != 'published':
                continue
            for m in models:
                pub, src = published_value(name, m)
                this = [r3(f(df[m].to_numpy())) for f in STATS.values()]
                for k, a, b in zip(STATS, pub, this):
                    controls.append({'dataset': name, 'model': m, 'check': f'{k} vs {src}',
                                     'published': r3(a), 'this_run': b,
                                     'pass': r3(a) == b})
                ref = published_persample(name, m)
                if ref is not None:
                    d = float(np.max(np.abs(df.sort_values('idx')[m].to_numpy() - ref)))
                    controls.append({'dataset': name, 'model': m, 'check': 'per-sample max|diff|',
                                     'published': 0.0, 'this_run': round(d, 6), 'pass': d <= 1e-6})
    OUT.mkdir(parents=True, exist_ok=True)
    sm = pd.DataFrame(summary)
    sm[list(STATS)] = sm[list(STATS)].round(6)
    sm.to_csv(OUT / 'summary.csv', index=False)
    (OUT / 'persample').mkdir(exist_ok=True)
    for (name, tag), f in persample_frames.items():
        f.to_csv(OUT / 'persample' / f'{name}__{tag}.csv', index=False)

    # paired: poly5 cbrt (and poly4 cbrt) against GP cbrt, Holm across datasets per tag
    prs = []
    for tag in TAGS:
        rows = []
        for name in DATASETS:
            df = persample_frames.get((name, tag))
            if df is None or GP not in df:
                continue
            for a_name in ('poly5_cbrt', 'poly4_cbrt'):
                a, b = df[a_name].to_numpy(), df[GP].to_numpy()
                r = {'tag': tag, 'dataset': name, 'n': len(a), 'a': a_name, 'b': GP,
                     'a_median': float(np.median(a)), 'b_median': float(np.median(b))}
                r.update(paired(a, b, np.random.default_rng(BOOT_SEED)))
                rows.append(r)
        if rows:
            t = pd.DataFrame(rows)
            for a_name in t.a.unique():
                sel = t.a == a_name
                t.loc[sel, 'p_holm'] = holm(t.loc[sel, 'p'])
            prs.append(t)
    pr = pd.concat(prs, ignore_index=True)
    pr['ci_includes_zero'] = (pr.median_diff_lo <= 0) & (pr.median_diff_hi >= 0)
    pr.to_csv(OUT / 'paired.csv', index=False)
    pub = pd.read_csv(RES / 'revision_r3' / 'paired_poly5cbrt_vs_gpcbrt.csv')
    for _, q in pub.iterrows():
        r = pr[(pr.tag == 'published') & (pr.dataset == q.dataset) & (pr.a == 'poly5_cbrt')]
        if len(r):
            r = r.iloc[0]
            for col in ('median_diff', 'median_diff_lo', 'median_diff_hi', 'frac_a_better', 'p', 'p_holm'):
                ok = np.isclose(float(r[col]), float(q[col]), rtol=1e-9, atol=1e-12)
                controls.append({'dataset': q.dataset, 'model': 'poly5_cbrt vs GP',
                                 'check': f'paired {col} vs paired_poly5cbrt_vs_gpcbrt.csv',
                                 'published': float(q[col]), 'this_run': float(r[col]), 'pass': bool(ok)})

    # reliability over the fixed configurations of tab:degree + tab:n7 run here
    n7 = set(model_registry()) - set(SKIPPED) | set(N7_EXTRA_CBRT) | set(poly_models())
    rel = []
    for tag in TAGS:
        sub = sm[(sm.tag == tag) & sm.dataset.isin(NGT4) & sm.model.isin(n7)]
        if sub.dataset.nunique() < len(NGT4) or not (sub.model == GP).any():
            continue
        worst = {}   # model -> worst ratio to best over stats and datasets
        for name in NGT4:
            d = sub[sub.dataset == name].set_index('model')
            best = {k: d[k].min() for k in STATS}
            arg = {k: d[k].idxmin() for k in STATS}
            ratio = d[list(STATS)] / pd.Series(best)
            for m, rr in ratio.iterrows():
                worst[m] = max(worst.get(m, 0.0), float(rr.max()))
            rel.append({'tag': tag, 'dataset': name, 'n_configs': len(d),
                        **{f'best_{k}': best[k] for k in STATS},
                        **{f'best_{k}_model': arg[k] for k in STATS},
                        **{f'gp_{k}_ratio': float(ratio.loc[GP, k]) for k in STATS},
                        'gp_lowest_max': bool(arg['max'] == GP)})
        others = {m: w for m, w in worst.items() if m != GP}
        close = min(others, key=others.get)
        for r in rel:
            if r['tag'] == tag:
                r['closest_other'] = close
                r['closest_other_worst_ratio'] = others[close]
                r['n_other_within_3x'] = int(sum(w <= 3 for w in others.values()))
                r['poly5_cbrt_worst_ratio'] = others['poly5_cbrt']
    rl = pd.DataFrame(rel)
    rl.round(6).to_csv(OUT / 'reliability.csv', index=False)

    ct = pd.DataFrame(controls)
    ct.to_csv(OUT / 'controls.csv', index=False)
    print(f'controls: {int(ct["pass"].sum())}/{len(ct)} pass')
    print(ct[~ct['pass']].to_string())
    return 0 if ct['pass'].all() else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('jobs', 'run', 'merge'))
    ap.add_argument('--dataset', choices=DATASETS)
    ap.add_argument('--tag', choices=TAGS)
    ap.add_argument('--part', choices=('gp', 'cheap', 'poly'))
    args = ap.parse_args()
    return {'jobs': cmd_jobs, 'run': cmd_run, 'merge': cmd_merge}[args.cmd](args) or 0


if __name__ == '__main__':
    raise SystemExit(main())
