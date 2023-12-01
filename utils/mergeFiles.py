import pandas as pd
import os


def merge_and_append_csv_files(base_path, file_names, output_file):
    # Initialize an empty DataFrame to hold the appended data
    appended_data = pd.DataFrame()

    # Loop through the list of CSV files
    for file_name in file_names:
        # Construct the full file path
        file_path = os.path.join(base_path, file_name)
        # Read the CSV file
        try:
            data = pd.read_csv(file_path)
            # Append to the appended DataFrame
            appended_data = pd.concat([appended_data, data], ignore_index=True)
        except FileNotFoundError as e:
            print(f"File not found: {file_path}")
            continue

    # Save the appended data to a new CSV file
    appended_data.to_csv(output_file, index=False)
    print(f'Appended data saved to {output_file}')


# Define the base path where the CSV files are located
# Adjust the base_path according to where the CSV files are located relative to the mergeFiles.py script
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'main', 'cmy2xyz', 'results', 'PC10'))

# List of CSV file names
file_names = [
    'nn_Bayesian_results.csv',
    'nn_DecisionTree_results.csv',
    'nn_DeepLearning_results.csv',
    'nn_Elastic_results.csv',
    'nn_GradientBoost_results.csv',
    'nn_k_Nearest_results.csv',
    'nn_Lasso_results.csv',
    'nn_PCR_results.csv',
    'nn_PLSRegression_results.csv',
    'nn_RandomForestRegressor_results.csv',
    'nn_RidgeRegression_results.csv',
    'nn_SimpleMLP_results.csv',
    'nn_SVM_results.csv',
    'polynomial_regression_results.csv',
]

# Output file name
output_file = os.path.join(base_path, 'appended_results.csv')

# Call the function to merge and append CSV files
merge_and_append_csv_files(base_path, file_names, output_file)
