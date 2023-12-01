import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_paths):
    dataframes = {model: pd.read_csv(file_path) for model, file_path in file_paths.items()}
    return dataframes


def extract_best_results(dataframes):
    best_results = {model: df.iloc[-1] for model, df in dataframes.items()}
    return best_results


def plot_comparison_chart(best_results, bar_width=0.25):
    # Extracting the performance metrics
    mean_errors = []
    median_errors = []
    max_errors = []
    for model, result in best_results.items():
        mean_errors.append(result['Mean Error'])
        median_errors.append(result['Median Error'])
        max_errors.append(result['Max Error'])

    # Model names for x-axis
    model_names = list(best_results.keys())
    pos = list(range(len(model_names)))

    # Plotting
    plt.figure(figsize=(20, 10))
    plt.bar([p - bar_width for p in pos], mean_errors, width=bar_width, label='Mean Error', color='blue')
    plt.bar(pos, median_errors, width=bar_width, label='Median Error', color='green')
    plt.bar([p + bar_width for p in pos], max_errors, width=bar_width, label='Max Error', color='red')

    plt.xticks([p for p in pos], model_names, rotation=45, ha='right')
    plt.xlabel('Model')
    plt.ylabel('Error')
    plt.title('Comparison of Best Performance Metrics Across Models (with Polynomial Regression at the end)')
    plt.legend()
    plt.tight_layout()
    plt.show()


# File paths for your CSV files
file_paths = {
    'Bayesian': 'path/to/nn_Bayesian_results.csv',
    'DecisionTree': 'path/to/nn_DecisionTree_results.csv',
    'DeepLearning': 'path/to/nn_DeepLearning_results.csv',
    'ElasticNet': 'path/to/nn_Elastic_results.csv',
    'GradientBoost': 'path/to/nn_GradientBoost_results.csv',
    'KNearest': 'path/to/nn_k_Nearest_results.csv',
    'Lasso': 'path/to/nn_Lasso_results.csv',
    'PCR': 'path/to/nn_PCR_results.csv',
    'PLSRegression': 'path/to/nn_PLSRegression_results.csv',
    'RandomForestRegressor': 'path/to/nn_RandomForestRegressor_results.csv',
    'RidgeRegression': 'path/to/nn_RidgeRegression_results.csv',
    'SimpleMLP': 'path/to/nn_SimpleMLP_results.csv',
    'SVM': 'path/to/nn_SVM_results.csv',
    'PolynomialRegression': 'path/to/polynomial_regression_results.csv',
}

# Load data, extract best results, and plot the chart
dataframes = load_data(file_paths)
best_results = extract_best_results(dataframes)
plot_comparison_chart(best_results)
