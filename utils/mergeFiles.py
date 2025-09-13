import pandas as pd
import os
import sys


def merge_and_append_csv_files(base_path, file_names, output_file):
    # Initialize an empty DataFrame to hold the appended data
    appended_data = pd.DataFrame()

    # Loop through the list of CSV files
    for file_name in file_names:
        # Extract the algorithm name from the file name
        algorithm_name = file_name.replace('_results.csv', '').replace('nn_', '').replace('polynomial_', '')
        # Construct the full file path
        file_path = os.path.join(base_path, file_name)
        # Read the CSV file
        try:
            data = pd.read_csv(file_path)
            # Insert a new column with the algorithm name
            data['Algorithm'] = algorithm_name
            # Ensure new metrics columns exist even if missing (older runs)
            if 'P95 Error' not in data.columns:
                data['P95 Error'] = float('nan')
            if 'Std Dev' not in data.columns:
                data['Std Dev'] = float('nan')
            # Append to the appended DataFrame
            appended_data = pd.concat([appended_data, data], ignore_index=True)
        except FileNotFoundError as e:
            print(f"File not found: {file_path}")
            continue
        except pd.errors.ParserError:
            # Handle inconsistent CSV rows (mixed column counts). Fallback parser.
            try:
                data = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
                data['Algorithm'] = algorithm_name
                if 'P95 Error' not in data.columns:
                    data['P95 Error'] = float('nan')
                if 'Std Dev' not in data.columns:
                    data['Std Dev'] = float('nan')
                appended_data = pd.concat([appended_data, data], ignore_index=True)
            except Exception as e2:
                print(f"Failed to parse {file_path}: {e2}")
                continue

    # Save the appended data to a new CSV file
    appended_data.to_csv(output_file, index=False)
    print(f'Appended data saved to {output_file}')


def _parse_dataset_arg() -> str:
    if len(sys.argv) >= 2 and sys.argv[1] in {'PC10', 'PC11', 'FOGRA'}:
        return sys.argv[1]
    return os.environ.get('DATASET_NAME', 'PC11')

DATASET_NAME = _parse_dataset_arg()
# Define the base path where the CSV files are located
# Adjust the base_path according to where the CSV files are located relative to the mergeFiles.py script
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'main', 'cmy2xyz', 'results', DATASET_NAME))

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
output_file = os.path.join(base_path, f"{DATASET_NAME}_append_results.csv")

if __name__ == "__main__":
    # Call the function to merge and append CSV files
    merge_and_append_csv_files(base_path, file_names, output_file)
