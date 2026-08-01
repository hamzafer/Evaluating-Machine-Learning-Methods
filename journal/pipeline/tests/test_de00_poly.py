import numpy as np
from journal.pipeline.de00_poly import DE00Polynomial


def test_de00_poly_beats_or_matches_lsq_on_train_objective():
    rng = np.random.RandomState(3)
    X = rng.uniform(0, 1, (300, 3))
    Y = np.stack([20 + 60 * X[:, 0] ** 2, 15 + 70 * X[:, 1], 10 + 50 * X[:, 2] ** 3], axis=1)
    m = DE00Polynomial(maxiter=200)
    m.fit(X, Y)                     # unscaled Y here: scaler defaults to identity
    from journal.pipeline.color import delta_e00
    de_opt = np.mean(delta_e00(m.predict(X), Y))
    de_lsq = np.mean(delta_e00(m.lsq_predict(X), Y))
    assert de_opt <= de_lsq + 1e-9   # refinement never worsens the objective
