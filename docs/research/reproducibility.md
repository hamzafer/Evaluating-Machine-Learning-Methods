# Reproducibility annex (11–12 Aug 2026)

Two robustness checks run on the colourlab machine (Linux x86_64, 24-core; identical pinned
env: python 3.12.11, sklearn 1.7.2, numpy 2.3.3, scipy 1.16.2). The laptop (macOS arm64)
is the canonical platform for all paper-reported numbers.

## 1. Cross-platform replication (full 16-model matrix, 9 datasets)
Same code, data, seeds, and package versions; different OS/architecture (BLAS: Apple
Accelerate vs OpenBLAS). Both platforms are bit-reproducible against themselves; the
divergence is deterministic library math. Max |Δ median ΔE00| per dataset (x86 vs the
committed arm64 values at d0cf8c2-era config; GP rows excluded — config changed in plan 10,
final-config GP mirror ran separately):

| dataset | max abs. median delta | worst model |
|---|---|---|
| PC10-CMY | 0.112 | mlp_shallow |
| PC10-CMYK | 0.141 | mlp_deep |
| PC11-CMY | 0.022 | random_forest |
| PC11-CMYK | 0.020 | mlp_shallow |
| FOGRA51-CMY | 0.032 | mlp_deep |
| FOGRA51-CMYK | 0.036 | mlp_deep |
| KCMYG-5 | 0.063 | random_forest |
| CMYKOGV-7 | 0.415 | mlp_deep |
| CMYKOGB-7 | 0.177 | plsr |

Reading: closed-form/kernel methods reproduce to ~0.03; iterative optimizers (MLP lbfgs)
drift most, up to 0.42 at n=7. Pinned environments alone do NOT give bit-reproducibility
across platforms — the paper's numbers are additionally pinned to one platform.

## 2. CV seed sensitivity (5 seeds x 6 coated datasets x 5 models, 150 runs)
`journal/results/robustness/seed_sweep_REMOTE_x86.csv` (KFold shuffle seed varied
0/1/2/3/42; grouped n>4 sets excluded — GroupKFold is deterministic). Max std of the
median across seeds, per model: gaussian_process 0.0013, poly3 0.0091, svm 0.0253,
knn 0.0586, mlp_deep 0.1034. Fold assignment is not a meaningful source of variance
at the 2-3 decimals we report (MLPs excepted at the 0.1 level).

**Caveat on the GP row (gate finding, 12 Aug):** this sweep ran BEFORE plan 10's unified GP
config landed, so its `gaussian_process` rows use the old kernel (`WhiteKernel(1e-5)`,
n_restarts=10) — e.g. PC10-CMY seed 42 reads 0.054, whereas the final config gives 0.044.
The non-GP rows are unaffected (those models never changed). **Resolved:** the GP-only
re-sweep under the final config (30 runs, `seed_sweep_GP_finalconfig_REMOTE_x86.csv`) gives
max std of the median across the 5 seeds = **0.0013** — identical to the figure above, so the
"GP is the most seed-stable model" claim holds under the final config. Per-dataset spread is
0.0000–0.0013 (e.g. PC10-CMY 0.042–0.044, FOGRA51-CMYK 0.067 on all five seeds). Quote GP
seed-stability from that file rather than from the mixed-model sweep.

## 3. Final-config GP mirror across platforms (12 Aug)
The plan-10 GP config was re-run on the remote x86_64 box and compared to the committed
arm64 values, model `gaussian_process`, all nine non-IFRA specs:

| dataset | x86_64 | arm64 (committed) | delta |
|---|---|---|---|
| PC10-CMY / PC11-CMY | 0.044 / 0.044 | 0.044 / 0.044 | exact |
| PC10-CMYK / PC11-CMYK | 0.056 / 0.056 | 0.056 / 0.056 | exact |
| FOGRA51-CMY / -CMYK | 0.056 / 0.067 | 0.056 / 0.067 | exact |
| CMYKOGB-7 | 1.280 | 1.280 | exact |
| KCMYG-5 | 0.851 | 0.867 | −0.016 |
| CMYKOGV-7 | 0.236 | 0.249 | −0.013 |

Seven of nine reproduce exactly; the two multi-ink sets differ by ≤0.016 ΔE00 — the
restarted log-marginal-likelihood optimizer can settle on marginally different optima when
the linear-algebra path differs, while remaining deterministic on each platform. Note this
is an order of magnitude tighter than the MLP divergence in §1, i.e. the GP results the
paper leans on are the *least* platform-sensitive of the family. The IFRA GP mirror
(within/cross/LOO) was still running when this table was written; spot values agreed exactly
(GratN_90 within-run 0.674, Mbd_103a 0.721).

Provenance: run on the remote (exploratory platform) 11-12 Aug; scripts: /tmp/seed_sweep.py
(remote) + three tmux replication sessions; coordinator-verified deltas via git-diff on the
remote clone.
