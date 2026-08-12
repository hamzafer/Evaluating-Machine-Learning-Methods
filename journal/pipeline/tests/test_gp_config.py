"""Guard tests for the unified GP config (Plan 10).

Root cause of the IFRA within-run anomaly (Task-1 diagnosis, 11 Aug 2026): the
old `WhiteKernel(1e-5)` init seeded a local-optimum basin in which the RBF
length_scale collapses to its lower bound and off-recipe predictions revert to
the prior mean (~18-20 median dE00 on newsprint). The unified config —
`WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-9, 1e5))` +
`n_restarts_optimizer=15` (15, not 10: with the widened bounds, 10 restart
draws missed the healthy basin on the noisiest pooled-LOO fit, IFRA
marca_133; the fixed seed keeps the first 10 draws, so 15 is equal-or-better
in LML everywhere) — starts from a neutral noise level and must:
  1. NOT collapse on data with genuine measurement noise at the newsprint
     scale (noise-to-signal variance ratio ~2e-3);
  2. stay never-worse on near-noise-free (coated-paper-like) data vs the old
     kernel;
  3. be exactly the frozen config (this is the regression guard for the
     paper's "one GP config for every dataset" claim).

Synthetic data keeps runtimes small (a few hundred rows per fit).
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from journal.pipeline.evaluate import train_test
from journal.pipeline.models import SEED, FitSubsampled, registry

NEWSPRINT_NOISE_LEVEL = 2e-3   # measured wb duplicate-pair ratio (Task-1, cb 2)


def smooth_ink_to_xyz(X):
    """Smooth printer-like map [0,1]^3 -> XYZ-ish (roughly 0-100 per channel)."""
    c, m, y = X[:, 0], X[:, 1], X[:, 2]
    x = 95 * (1 - 0.8 * c) * (1 - 0.5 * m) * (1 - 0.3 * y)
    yy = 100 * (1 - 0.6 * c) * (1 - 0.7 * m) * (1 - 0.2 * y)
    z = 80 * (1 - 0.3 * c) * (1 - 0.4 * m) * (1 - 0.85 * y)
    return np.column_stack([x, yy, z])


def make_synthetic(noise_level, n_recipes=150, seed=0):
    """Duplicated recipes + Gaussian XYZ noise at a chosen noise-to-signal
    variance ratio (the units WhiteKernel.noise_level ends up in after the
    pipeline's MinMax + normalize_y scalings — both variance-ratio invariant).
    """
    rng = np.random.default_rng(seed)
    recipes = rng.uniform(0, 1, size=(n_recipes, 3))
    X = np.repeat(recipes, 2, axis=0)              # every recipe measured twice
    Y = smooth_ink_to_xyz(X)
    if noise_level > 0:
        sigma = np.sqrt(noise_level * Y.var(axis=0))
        Y = Y + rng.normal(0, 1, size=Y.shape) * sigma
    # held-out queries strictly off-recipe (the regime where collapse shows)
    Xte = rng.uniform(0.05, 0.95, size=(200, 3))
    Yte = smooth_ink_to_xyz(Xte)
    return X, Y, Xte, Yte


def fitted_rbf_length_scale(model):
    """Unwrap FitSubsampled -> GaussianProcessRegressor -> fitted RBF."""
    gpr = model.estimator if isinstance(model, FitSubsampled) else model
    k = gpr.kernel_.k1        # kernel = (Constant * RBF) + WhiteKernel
    assert isinstance(k.k2, RBF)
    return float(k.k2.length_scale)


def old_config_gp():
    """The pre-Plan-10 kernel (WhiteKernel(1e-5), default bounds)."""
    return FitSubsampled(GaussianProcessRegressor(
        kernel=ConstantKernel() * RBF() + WhiteKernel(1e-5),
        normalize_y=True, n_restarts_optimizer=10, random_state=SEED))


def test_unified_config_is_frozen():
    """Regression guard: the exact Plan-10 unified config, nothing else."""
    model = registry()['gaussian_process']()
    assert isinstance(model, FitSubsampled) and model.cap == 2000
    gpr = model.estimator
    assert isinstance(gpr, GaussianProcessRegressor)
    assert gpr.normalize_y is True
    assert gpr.n_restarts_optimizer == 15
    assert gpr.random_state == SEED
    white = gpr.kernel.k2
    assert isinstance(white, WhiteKernel)
    assert white.noise_level == 1e-3
    assert tuple(white.noise_level_bounds) == (1e-9, 1e5)
    assert isinstance(gpr.kernel.k1.k2, RBF)
    # Pin the amplitude term as well: swapping ConstantKernel for something
    # else would otherwise slip past this freeze test (gate finding L6).
    assert isinstance(gpr.kernel.k1.k1, ConstantKernel)


def test_no_collapse_on_newsprint_scale_noise():
    """Injected noise at the measured newsprint ratio (~2e-3): the unified GP
    must keep a macroscopic length_scale and predict off-recipe queries far
    better than prior-mean reversion would."""
    X, Y, Xte, Yte = make_synthetic(NEWSPRINT_NOISE_LEVEL)
    model = registry()['gaussian_process']()
    de = train_test(X, Y, Xte, Yte, registry()['gaussian_process'])
    # collapse baseline: predict the training-mean XYZ everywhere
    from journal.pipeline.color import delta_e00
    de_mean = delta_e00(np.tile(Y.mean(axis=0), (len(Xte), 1)), Yte)
    assert np.median(de) < 3.0                       # sane, in-family error
    assert np.median(de) < 0.25 * np.median(de_mean)  # nowhere near reversion
    # and the fitted kernel itself is healthy (refit to inspect; same seed)
    from sklearn.preprocessing import MinMaxScaler
    sx, sy = MinMaxScaler().fit(X), MinMaxScaler().fit(Y)
    model.fit(sx.transform(X), sy.transform(Y))
    assert fitted_rbf_length_scale(model) > 1e-2     # off the 1e-5 bound


def test_never_worse_on_noise_free_data():
    """Byte-identical duplicates, no noise (coated-paper regime): the unified
    config must match or beat the old kernel within tolerance."""
    X, Y, Xte, Yte = make_synthetic(0.0)
    de_new = train_test(X, Y, Xte, Yte, registry()['gaussian_process'])
    de_old = train_test(X, Y, Xte, Yte, old_config_gp)
    assert np.median(de_new) <= np.median(de_old) + 0.05
