from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor


@dataclass
class ModelSpec:
    name: str
    factory: Callable[[], object]


def registry() -> Dict[str, ModelSpec]:
    return {
        'random_forest': ModelSpec('Random Forest', lambda: RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)),
        'gradient_boost': ModelSpec('Gradient Boosting', lambda: MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42))),
        'knn': ModelSpec('k-NN', lambda: KNeighborsRegressor(n_neighbors=5, weights='uniform')),
        'bayesian_gp': ModelSpec('Bayesian (Gaussian Process)', lambda: MultiOutputRegressor(GaussianProcessRegressor(random_state=42))),
        'ridge': ModelSpec('Ridge Regression', lambda: Ridge(alpha=1.0, random_state=42)),
        'lasso': ModelSpec('Lasso Regression', lambda: Lasso(alpha=1.0, random_state=42)),
        'elastic': ModelSpec('Elastic Net', lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)),
        'decision_tree': ModelSpec('Decision Tree', lambda: MultiOutputRegressor(__import__('sklearn.tree').tree.DecisionTreeRegressor(random_state=42))),
        'mlp_shallow': ModelSpec('MLP (Shallow Network)', lambda: MLPRegressor(hidden_layer_sizes=(20,), activation='relu', solver='lbfgs', max_iter=1000, random_state=42)),
        'mlp_deep': ModelSpec('Deep Learning (Neural Network)', lambda: MLPRegressor(hidden_layer_sizes=(50, 30, 10), activation='relu', solver='adam', max_iter=500, random_state=42)),
        'svm': ModelSpec('SVM (RBF Kernel)', lambda: MultiOutputRegressor(SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1))),
        'pcr': ModelSpec('PCR', lambda: make_pipeline(PCA(n_components=3), Ridge(alpha=1.0, random_state=42))),
        'plsr': ModelSpec('PLSR', lambda: PLSRegression(n_components=3)),
        'poly3': ModelSpec('Polynomial Regression (3rd)', lambda: make_pipeline(__import__('sklearn.preprocessing').preprocessing.PolynomialFeatures(degree=3), LinearRegression())),
    }

