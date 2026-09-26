"""Model registry: the 14 methods of the AIC study, input-dimension-agnostic.

Configs match the AIC paper's best configs where those were sane; the two that
were degenerate there (Lasso/ElasticNet at alpha=1.0 -> constant predictor,
SVR epsilon=0.1 -> tube covers 10% of the target range) use standard sensible
values instead, noted below. Every stochastic model is seeded.
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .de00_poly import DE00Polynomial

SEED = 42


class FitSubsampled:
    """Fit-time subsampling wrapper (Plan 10's unified GP config): if a
    training fold exceeds `cap` rows, fit the inner estimator on a fixed-seed
    random subsample of `cap` rows; predict is untouched. Lives here so
    evaluate.py stays model-agnostic — only the GP is wrapped (cubic fit cost).
    """

    def __init__(self, estimator, cap=2000, seed=SEED):
        self.estimator, self.cap, self.seed = estimator, cap, seed

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        if len(X) > self.cap:
            keep = np.random.RandomState(self.seed).choice(len(X), self.cap, replace=False)
            X, y = X[keep], y[keep]
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)


class CubeRootPolynomial:
    """Polynomial fitted to the CUBE ROOT of physical XYZ, not to XYZ itself.

    Motivation (surfaced by an LLM during the plan-09 equation experiment, then
    tested here properly): CIELAB is defined through a cube root of XYZ, so
    CIEDE2000 errors are roughly linear in cube-root space. Fitting there aligns
    the least-squares objective with the metric the paper actually reports,
    without any change to the evaluation protocol.

    `evaluate.cross_validate` hands models MinMax-scaled targets, so this uses
    the existing `set_scaler` hook to recover physical XYZ before taking the
    root, and re-applies the scaler on the way out.
    """

    def __init__(self, degree=3):
        self.degree = degree
        self._scaler = None
        self._model = None

    def set_scaler(self, scaler):
        self._scaler = scaler

    def fit(self, X, y_scaled):
        Y = self._scaler.inverse_transform(y_scaled) if self._scaler is not None else y_scaled
        self._model = make_pipeline(PolynomialFeatures(degree=self.degree), LinearRegression())
        self._model.fit(X, np.cbrt(np.clip(Y, 0.0, None)))
        return self

    def predict(self, X):
        Y = np.clip(self._model.predict(X), 0.0, None) ** 3
        return self._scaler.transform(Y) if self._scaler is not None else Y


class CubeRootTarget:
    """Fit ANY estimator against cbrt(physical XYZ) and cube the prediction back.

    Generalisation of CubeRootPolynomial, so the fitting-space question can be
    asked of every model rather than only the polynomial. Symmetry matters here:
    having found that the polynomial baseline was handicapped by fitting in XYZ,
    the same correction must be offered to its competitors before any comparison
    between them is fair.
    """

    def __init__(self, factory, rescale=False):
        # rescale (revision R1, R2-1): MinMax-scale the cube-root target on the
        # training fold before the inner fit, so a scale-sensitive estimator
        # (SVR epsilon, MLP, Ridge alpha) sees targets on the same [0, 1] range
        # it sees in XYZ mode; the transform is then the ONLY difference. The
        # original GP variant keeps rescale=False (normalize_y already handles it).
        self.factory, self.rescale = factory, rescale
        self._scaler = None
        self._inner = None
        self._ts = None

    def set_scaler(self, scaler):
        self._scaler = scaler

    def fit(self, X, y_scaled):
        Y = self._scaler.inverse_transform(y_scaled) if self._scaler is not None else y_scaled
        T = np.cbrt(np.clip(Y, 0.0, None))
        if self.rescale:
            self._ts = MinMaxScaler().fit(T)
            T = self._ts.transform(T)
        self._inner = self.factory()
        if hasattr(self._inner, 'set_scaler') and self.rescale:
            # an inner tuner scores candidates on physical XYZ: hand it the map
            # from its (scaled cube-root) targets back to XYZ
            self._inner.set_scaler(_CbrtInverse(self._ts))
        self._inner.fit(X, T)
        return self

    def predict(self, X):
        T = np.asarray(self._inner.predict(X))
        if self.rescale:
            T = self._ts.inverse_transform(T)
        Y = np.clip(T, 0.0, None) ** 3
        return self._scaler.transform(Y) if self._scaler is not None else Y


class _CbrtInverse:
    """Scaler-like adapter: scaled cube-root targets -> physical XYZ."""

    def __init__(self, ts):
        self.ts = ts

    def inverse_transform(self, T):
        return np.clip(self.ts.inverse_transform(np.asarray(T)), 0.0, None) ** 3


class InnerTuned:
    """Hyperparameter tuning INSIDE the training fold (revision R1-1).

    Grid search over `grid` with a 3-fold inner split of the outer training
    fold (grouped on duplicate recipes, seeded). Selection criterion is the
    paper's own metric: median CIEDE2000 on denormalized XYZ, via the same
    set_scaler hook the cube-root models use. The winner is refitted on the
    whole training fold; the outer test fold is never seen during tuning.
    """

    def __init__(self, build, grid, n_inner=3, seed=SEED):
        self.build, self.grid, self.n_inner, self.seed = build, grid, n_inner, seed
        self._scaler = None
        self.best_params_ = None
        self.inner_scores_ = None

    def set_scaler(self, scaler):
        self._scaler = scaler

    def _make(self, params):
        m = self.build(**params)
        if hasattr(m, 'set_scaler'):
            m.set_scaler(self._scaler)
        return m

    def fit(self, X, y):
        from .color import delta_e00
        from .evaluate import make_groups
        X, y = np.asarray(X), np.asarray(y)
        splits = list(GroupKFold(n_splits=self.n_inner, shuffle=True, random_state=self.seed)
                      .split(X, groups=make_groups(X)))
        Ytrue = np.clip(self._scaler.inverse_transform(y), 0.0, None)
        scores = []
        for params in ParameterGrid(self.grid):
            de = np.empty(len(X))
            for tr, te in splits:
                m = self._make(params).fit(X[tr], y[tr])
                pred = np.clip(self._scaler.inverse_transform(np.asarray(m.predict(X[te]))), 0.0, None)
                de[te] = delta_e00(pred, Ytrue[te])
            scores.append((float(np.median(de)), params))
        scores.sort(key=lambda t: t[0])
        self.inner_scores_ = scores
        self.best_params_ = scores[0][1]
        self._model = self._make(self.best_params_).fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


def registry() -> dict:
    return {
        'poly3': lambda: make_pipeline(PolynomialFeatures(degree=3), LinearRegression()),
        'poly3_cbrt': lambda: CubeRootPolynomial(degree=3),
        # Fairness controls for the fitting-space finding (23 Aug): the same
        # correction offered to the polynomial's competitors, plus the degree
        # lever Phil's 3rd-order cap currently forbids.
        'poly4_cbrt': lambda: CubeRootPolynomial(degree=4),
        'poly4': lambda: make_pipeline(PolynomialFeatures(degree=4), LinearRegression()),
        'gaussian_process_cbrt': lambda: CubeRootTarget(
            lambda: FitSubsampled(GaussianProcessRegressor(
                kernel=ConstantKernel() * RBF() + WhiteKernel(
                    noise_level=1e-3, noise_level_bounds=(1e-9, 1e5)),
                normalize_y=True, n_restarts_optimizer=15, random_state=SEED))),
        'ridge': lambda: Ridge(alpha=0.5, random_state=SEED),
        'lasso': lambda: Lasso(alpha=1e-3, random_state=SEED),          # AIC's 1.0 was degenerate
        'elastic': lambda: ElasticNet(alpha=1e-3, l1_ratio=0.5, random_state=SEED),  # ditto
        # PCA keeps all components: on designed targets variance splits evenly
        # across channels ('mle' dropped one and cratered — see journal notes),
        # so at n<=4 PCR reduces to Ridge in a rotated basis.
        'pcr': lambda: make_pipeline(PCA(), Ridge(alpha=0.5)),
        'plsr': lambda: PLSRegression(n_components=3),
        'knn': lambda: KNeighborsRegressor(n_neighbors=5, weights='uniform'),
        'svm': lambda: MultiOutputRegressor(
            SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.01)),    # AIC's eps=0.1 was degenerate
        'decision_tree': lambda: DecisionTreeRegressor(random_state=SEED),
        'random_forest': lambda: RandomForestRegressor(n_estimators=200, max_depth=15, random_state=SEED),
        'gradient_boost': lambda: MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=SEED)),
        # Plan 10 unified GP config (one config for every dataset): neutral
        # noise init 1e-3 (the old 1e-5 init seeded a length-scale-collapse
        # basin on newsprint/KCMYG), lower bound widened to 1e-9 so clean
        # coated data can fit noise below 1e-5; restarts escape bad basins.
        # n_restarts=15 (not 10): with the widened bounds the restart inits
        # span 14 decades of noise level, and 10 draws missed the healthy
        # basin on the noisiest pooled-LOO fit (IFRA marca_133); 15 recovers
        # it, and with the fixed seed the first 10 draws are unchanged, so
        # the chosen optimum is equal-or-better in LML everywhere.
        'gaussian_process': lambda: FitSubsampled(GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF()
                   + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-9, 1e5)),
            normalize_y=True, n_restarts_optimizer=15, random_state=SEED)),
        'mlp_shallow': lambda: MLPRegressor(hidden_layer_sizes=(64,), solver='lbfgs',
                                            max_iter=2000, random_state=SEED),
        'mlp_deep': lambda: MLPRegressor(hidden_layer_sizes=(64, 64, 64), solver='lbfgs',
                                         max_iter=2000, random_state=SEED),
        'poly3_de00_nm': lambda: DE00Polynomial(method='Nelder-Mead', maxiter=2000),
        'poly3_de00_powell': lambda: DE00Polynomial(method='Powell', maxiter=200),
    }


# ---------------------------------------------------------------------------
# Revision R1 additions. Kept out of registry() so the 16-model matrix and
# every existing summary.csv stay exactly as published.
# ---------------------------------------------------------------------------

# The 12 non-GP methods that had no cube-root variant (poly3/poly4 already do).
CBRT_BASES = ('ridge', 'lasso', 'elastic', 'pcr', 'plsr', 'knn', 'svm',
              'decision_tree', 'random_forest', 'gradient_boost',
              'mlp_shallow', 'mlp_deep')


def cbrt_registry() -> dict:
    """R2-1: every non-GP method fitted against cbrt(XYZ), same config otherwise."""
    base = registry()
    return {f'{m}_cbrt': (lambda f=base[m]: CubeRootTarget(f, rescale=True))
            for m in CBRT_BASES}


def tuning_grids(n_inputs: int) -> dict:
    """R1-1: (builder, grid) per non-closed-form method. Small grids that bracket
    each fixed configuration; PLSR/PCR components span 1..n inputs."""
    comps = list(range(1, n_inputs + 1))
    return {
        'ridge': (lambda alpha: Ridge(alpha=alpha, random_state=SEED),
                  {'alpha': [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 10.0]}),
        'lasso': (lambda alpha: Lasso(alpha=alpha, max_iter=10000, random_state=SEED),
                  {'alpha': [1e-5, 1e-4, 1e-3, 1e-2]}),
        'elastic': (lambda alpha, l1_ratio: ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                                       max_iter=10000, random_state=SEED),
                    {'alpha': [1e-5, 1e-4, 1e-3, 1e-2], 'l1_ratio': [0.2, 0.5, 0.8]}),
        'pcr': (lambda n_components, alpha: make_pipeline(PCA(n_components=n_components),
                                                          Ridge(alpha=alpha)),
                {'n_components': comps, 'alpha': [1e-3, 0.5, 10.0]}),
        'plsr': (lambda n_components: PLSRegression(n_components=n_components),
                 {'n_components': comps}),
        'knn': (lambda n_neighbors, weights: KNeighborsRegressor(n_neighbors=n_neighbors,
                                                                 weights=weights),
                {'n_neighbors': [1, 3, 5, 7, 10, 15], 'weights': ['uniform', 'distance']}),
        'svm': (lambda C, gamma, epsilon: MultiOutputRegressor(
                    SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)),
                {'C': [1.0, 10.0, 100.0, 1000.0], 'gamma': ['scale', 1.0, 10.0],
                 'epsilon': [0.001, 0.01]}),
        'decision_tree': (lambda max_depth, min_samples_leaf: DecisionTreeRegressor(
                              max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                              random_state=SEED),
                          {'max_depth': [None, 10, 15, 20], 'min_samples_leaf': [1, 2, 5]}),
        'random_forest': (lambda max_depth, max_features: RandomForestRegressor(
                              n_estimators=200, max_depth=max_depth, max_features=max_features,
                              random_state=SEED),
                          {'max_depth': [None, 15, 25], 'max_features': [1.0, 'sqrt']}),
        'gradient_boost': (lambda n_estimators, learning_rate, max_depth: MultiOutputRegressor(
                               GradientBoostingRegressor(n_estimators=n_estimators,
                                                         learning_rate=learning_rate,
                                                         max_depth=max_depth, random_state=SEED)),
                           {'n_estimators': [200, 500], 'learning_rate': [0.05, 0.1],
                            'max_depth': [3, 5, 7]}),
        'mlp_shallow': (lambda hidden_layer_sizes, alpha: MLPRegressor(
                            hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, solver='lbfgs',
                            max_iter=2000, random_state=SEED),
                        {'hidden_layer_sizes': [(32,), (64,), (128,)],
                         'alpha': [1e-5, 1e-4, 1e-3]}),
        'mlp_deep': (lambda hidden_layer_sizes, alpha: MLPRegressor(
                         hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, solver='lbfgs',
                         max_iter=2000, random_state=SEED),
                     {'hidden_layer_sizes': [(64, 64, 64), (128, 128, 128)],
                      'alpha': [1e-4, 1e-3]}),
    }


def tuned_registry(n_inputs: int) -> dict:
    return {f'{m}_tuned': (lambda b=b, g=g: InnerTuned(b, g))
            for m, (b, g) in tuning_grids(n_inputs).items()}



CBRT_TUNED_BASES = ('svm', 'mlp_deep', 'gradient_boost', 'random_forest', 'knn')


def cbrt_tuned_registry(n_inputs: int) -> dict:
    """Best effort per competitor: tuned inside the fold AND fitted in cube-root
    space (the combination of R1-1 and R2-1) for the five nonlinear competitors."""
    g = tuning_grids(n_inputs)
    return {f'{m}_cbrt_tuned': (lambda b=g[m][0], gr=g[m][1]:
                                CubeRootTarget(lambda: InnerTuned(b, gr), rescale=True))
            for m in CBRT_TUNED_BASES}
