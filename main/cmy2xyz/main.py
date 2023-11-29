from nn_RandomForestRegressor import process as nn_RandomForestRegressor
from nn_logistic_lbfgs import process as nn_logistic_lbfgs
from polynomial_regression import process as polynomial_regression

nn_RandomForestRegressor('PC10', 'CMY', 'XYZ', visualize=False)
nn_logistic_lbfgs('PC10', 'CMY', 'XYZ', visualize=False)
polynomial_regression('PC10', 'CMY', 'XYZ', 3, visualize=False)
