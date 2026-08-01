import numpy as np
from journal.pipeline.evaluate import make_groups, cross_validate, train_test


def test_make_groups_identical_rows_share_id():
    X = np.array([[10, 20, 30], [0, 0, 0], [10, 20, 30], [5, 5, 5]])
    g = make_groups(X)
    assert g[0] == g[2]
    assert len({g[0], g[1], g[3]}) == 3


def test_grouped_cv_keeps_duplicates_out_of_train():
    # y = x with a poisoned duplicate: if duplicates split across folds,
    # 1-NN memorizes the twin and scores ~0 error on it.
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 100, size=(80, 3))
    X[40:] = X[:40]                      # every row duplicated once
    Y = X.copy()
    from sklearn.neighbors import KNeighborsRegressor
    de_plain = cross_validate(X, Y, lambda: KNeighborsRegressor(1))
    de_grouped = cross_validate(X, Y, lambda: KNeighborsRegressor(1),
                                groups=make_groups(X))
    assert np.median(de_plain) < 0.01          # memorization
    assert np.median(de_grouped) > np.median(de_plain)  # grouped CV blocks it


def test_train_test_perfect_linear_map():
    rng = np.random.RandomState(1)
    Xtr, Xte = rng.uniform(0, 100, (200, 3)), rng.uniform(10, 90, (50, 3))
    from sklearn.linear_model import LinearRegression
    de = train_test(Xtr, Xtr * 0.9, Xte, Xte * 0.9, LinearRegression)
    assert de.shape == (50,) and np.median(de) < 0.05   # exactly learnable map
