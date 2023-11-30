from nn_Bayesian import process as nn_Bayesian
from nn_DecisionTree import process as nn_DecisionTree
from nn_DeepLearning import process as nn_DeepLearning
from nn_Elastic import process as nn_Elastic
from nn_GradientBoost import process as nn_GradientBoost
from nn_Lasso import process as nn_Lasso
from nn_PCR import process as nn_PCR
from nn_PLSRegression import process as nn_PLSRegression
from nn_RandomForestRegressor import process as nn_RandomForestRegressor
from nn_RidgeRegression import process as nn_RidgeRegression
from nn_SVM import process as nn_SVM
from nn_k_Nearest import process as nn_k_Nearest
from nn_logistic_lbfgs import process as nn_logistic_lbfgs
from polynomial_regression import process as polynomial_regression

# Constants
DATASET_NAME = 'PC10'
INPUT_TYPE = 'CMY'
OUTPUT_TYPE = 'XYZ'

# Machine Learning Algorithms with PC10 dataset
nn_RandomForestRegressor(DATASET_NAME, INPUT_TYPE, OUTPUT_TYPE, visualize=False)
nn_logistic_lbfgs(DATASET_NAME, INPUT_TYPE, OUTPUT_TYPE, visualize=False)
nn_Bayesian(DATASET_NAME, INPUT_TYPE, OUTPUT_TYPE, visualize=False)
nn_DecisionTree(DATASET_NAME, INPUT_TYPE, OUTPUT_TYPE, visualize=False)
nn_DeepLearning(DATASET_NAME, INPUT_TYPE, OUTPUT_TYPE, visualize=False)
nn_Elastic(DATASET_NAME, INPUT_TYPE, OUTPUT_TYPE, visualize=False)
nn_k_Nearest(DATASET_NAME, INPUT_TYPE, OUTPUT_TYPE, visualize=False)
nn_Lasso(DATASET_NAME, INPUT_TYPE, OUTPUT_TYPE, visualize=False)
nn_PCR(DATASET_NAME, INPUT_TYPE, OUTPUT_TYPE, visualize=False)
nn_PLSRegression(DATASET_NAME, INPUT_TYPE, OUTPUT_TYPE, visualize=False)
nn_RidgeRegression(DATASET_NAME, INPUT_TYPE, OUTPUT_TYPE, visualize=False)

# MultiOutputRegressor Algorithms with PC10 dataset
nn_GradientBoost(DATASET_NAME, INPUT_TYPE, OUTPUT_TYPE, visualize=False)
nn_SVM(DATASET_NAME, INPUT_TYPE, OUTPUT_TYPE, visualize=False)

# Polynomial Regression with PC10 dataset
for degree in range(1, 11):
    polynomial_regression(DATASET_NAME, INPUT_TYPE, OUTPUT_TYPE, degree, visualize=False)
