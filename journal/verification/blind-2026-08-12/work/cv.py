import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from colorimetry import XYZ100_to_Lab, delta_e00

SEED = 42
N_SPLITS = 5


def make_group_folds(X, seed=SEED, n_splits=N_SPLITS, round_dp=6):
    """Assign each row to one of n_splits folds such that rows with an identical
    ink recipe (rounded to round_dp decimals) always land in the same fold.
    Deterministic given seed."""
    Xr = np.round(X, round_dp)
    # unique recipe -> integer group id, in a stable (lexicographic) order
    _, inverse = np.unique(Xr, axis=0, return_inverse=True)
    n_groups = inverse.max() + 1
    rng = np.random.RandomState(seed)
    shuffled_groups = rng.permutation(n_groups)
    group_fold = np.empty(n_groups, dtype=int)
    for fold_idx, grp_ids in enumerate(np.array_split(shuffled_groups, n_splits)):
        group_fold[grp_ids] = fold_idx
    row_fold = group_fold[inverse]
    return row_fold, n_groups


def build_poly3():
    return make_pipeline(PolynomialFeatures(degree=3), LinearRegression())


def build_knn():
    return KNeighborsRegressor(n_neighbors=5, weights="uniform")


def fit_predict_svm(X_train, y_train, X_test):
    """One independent SVR per output column, as specified."""
    preds = np.empty((X_test.shape[0], y_train.shape[1]))
    for j in range(y_train.shape[1]):
        svr = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.01)
        svr.fit(X_train, y_train[:, j])
        preds[:, j] = svr.predict(X_test)
    return preds


def fit_predict_gp(X_train, y_train, X_test, seed=SEED, max_train=2000):
    if X_train.shape[0] > max_train:
        rng = np.random.RandomState(seed)
        sub = rng.choice(X_train.shape[0], size=max_train, replace=False)
        X_train = X_train[sub]
        y_train = y_train[sub]
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(
        noise_level=1e-3, noise_level_bounds=(1e-9, 1e5)
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=15, random_state=seed
    )
    gp.fit(X_train, y_train)
    return gp.predict(X_test)


def evaluate_pooled_dE00(y_true_all, y_pred_all):
    y_pred_all = np.clip(y_pred_all, 0, None)
    Lab_true = XYZ100_to_Lab(y_true_all)
    Lab_pred = XYZ100_to_Lab(y_pred_all)
    dE = delta_e00(Lab_true, Lab_pred)
    return dict(
        median=float(np.median(dE)),
        p95=float(np.percentile(dE, 95)),
        max=float(np.max(dE)),
        n=len(dE),
    ), dE


def run_cv_model(X, XYZ, model_name, seed=SEED, n_splits=N_SPLITS):
    row_fold, n_groups = make_group_folds(X, seed=seed, n_splits=n_splits)
    y_pred_all = np.empty_like(XYZ)
    for f in range(n_splits):
        test_idx = np.where(row_fold == f)[0]
        train_idx = np.where(row_fold != f)[0]
        x_scaler = MinMaxScaler().fit(X[train_idx])
        y_scaler = MinMaxScaler().fit(XYZ[train_idx])
        Xtr = x_scaler.transform(X[train_idx])
        Xte = x_scaler.transform(X[test_idx])
        ytr = y_scaler.transform(XYZ[train_idx])

        if model_name == "poly3":
            model = build_poly3()
            model.fit(Xtr, ytr)
            pred_scaled = model.predict(Xte)
        elif model_name == "knn":
            model = build_knn()
            model.fit(Xtr, ytr)
            pred_scaled = model.predict(Xte)
        elif model_name == "svm":
            pred_scaled = fit_predict_svm(Xtr, ytr, Xte)
        elif model_name == "gaussian_process":
            pred_scaled = fit_predict_gp(Xtr, ytr, Xte, seed=seed)
        else:
            raise ValueError(model_name)

        pred_xyz = y_scaler.inverse_transform(pred_scaled)
        y_pred_all[test_idx] = pred_xyz

    metrics, dE = evaluate_pooled_dE00(XYZ, y_pred_all)
    metrics["n_groups"] = n_groups
    return metrics, dE, y_pred_all
