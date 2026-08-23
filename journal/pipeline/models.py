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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
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

    def __init__(self, factory):
        self.factory = factory
        self._scaler = None
        self._inner = None

    def set_scaler(self, scaler):
        self._scaler = scaler

    def fit(self, X, y_scaled):
        Y = self._scaler.inverse_transform(y_scaled) if self._scaler is not None else y_scaled
        self._inner = self.factory()
        self._inner.fit(X, np.cbrt(np.clip(Y, 0.0, None)))
        return self

    def predict(self, X):
        Y = np.clip(self._inner.predict(X), 0.0, None) ** 3
        return self._scaler.transform(Y) if self._scaler is not None else Y


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
