"""Model registry: the 14 methods of the AIC study, input-dimension-agnostic.

Configs match the AIC paper's best configs where those were sane; the two that
were degenerate there (Lasso/ElasticNet at alpha=1.0 -> constant predictor,
SVR epsilon=0.1 -> tube covers 10% of the target range) use standard sensible
values instead, noted below. Every stochastic model is seeded.
"""
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

SEED = 42


def registry() -> dict:
    return {
        'poly3': lambda: make_pipeline(PolynomialFeatures(degree=3), LinearRegression()),
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
        'gaussian_process': lambda: GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF() + WhiteKernel(1e-5),
            normalize_y=True, random_state=SEED),
        'mlp_shallow': lambda: MLPRegressor(hidden_layer_sizes=(64,), solver='lbfgs',
                                            max_iter=2000, random_state=SEED),
        'mlp_deep': lambda: MLPRegressor(hidden_layer_sizes=(64, 64, 64), solver='lbfgs',
                                         max_iter=2000, random_state=SEED),
    }
