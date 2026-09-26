"""3rd-order polynomial whose coefficients are refined on mean DE00 (scipy)."""
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from .color import delta_e00


class DE00Polynomial:
    def __init__(self, method='Nelder-Mead', maxiter=2000, maxfev=None):
        # maxfev (revision R1-4): cap on objective evaluations, so optimizers
        # can be compared at equal compute. None keeps the original behaviour.
        self.method, self.maxiter, self.maxfev = method, maxiter, maxfev
        self._sy = None                      # optional y-scaler (identity if None)

    def set_scaler(self, sy):                # pipeline hook: real-XYZ objective
        self._sy = sy

    def _denorm(self, Yn):
        return self._sy.inverse_transform(Yn) if self._sy is not None else Yn

    def fit(self, X, Y):
        self._pf = PolynomialFeatures(degree=3).fit(X)
        Phi = self._pf.transform(X)
        self._lsq = LinearRegression(fit_intercept=False).fit(Phi, Y)
        w0 = self._lsq.coef_.ravel()
        Ytrue = self._denorm(np.asarray(Y, dtype=float))

        def objective(w):
            pred = Phi @ w.reshape(3, -1).T
            return float(np.mean(delta_e00(
                np.clip(self._denorm(pred), 0.0, None), Ytrue)))

        opts = ({'maxiter': self.maxiter, 'xatol': 1e-6, 'fatol': 1e-6}
                if self.method == 'Nelder-Mead' else {'maxiter': self.maxiter})
        if self.maxfev is not None:
            opts['maxfev'] = self.maxfev
        res = minimize(objective, w0, method=self.method, options=opts)
        self.nfev_, self.nit_, self.message_ = int(res.nfev), int(getattr(res, 'nit', -1)), str(res.message)
        self.n_coef_ = int(w0.size)
        self._w = res.x if res.fun <= objective(w0) else w0   # never worse than LSQ
        return self

    def predict(self, X):
        return self._pf.transform(X) @ self._w.reshape(3, -1).T

    def lsq_predict(self, X):
        return self._lsq.predict(self._pf.transform(X))
