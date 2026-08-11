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
The non-GP rows are unaffected (those models never changed). A GP-only re-sweep under the
final config is recorded separately in
`journal/results/robustness/seed_sweep_GP_finalconfig_REMOTE_x86.csv`; quote the GP
seed-stability figure only from that file.

Provenance: run on the remote (exploratory platform) 11-12 Aug; scripts: /tmp/seed_sweep.py
(remote) + three tmux replication sessions; coordinator-verified deltas via git-diff on the
remote clone.
