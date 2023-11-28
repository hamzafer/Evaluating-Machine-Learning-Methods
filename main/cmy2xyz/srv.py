# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from input.input import get_dataset
from utils.save_results import save_results_to_CSV
from utils.xyz2lab import xyz2lab
from visual.vis_lab import visualize_lab_values

# Function to normalize the data
def normalize_data(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# Main function to run color mapping
def run_color_mapping():
    results = pd.DataFrame(columns=['Configuration', 'Mean Error', 'Median Error', 'Max Error'])

    # Load your CMY and XYZ data as numpy arrays
    input_cmy = get_dataset('PC10', 'CMY')
    output_xyz = get_dataset('PC10', 'XYZ')

    # Normalize the data
    input_cmy_norm = normalize_data(input_cmy)
    output_xyz_norm = normalize_data(output_xyz)

    # Split the dataset
    input_train, input_test, output_train, output_test = train_test_split(input_cmy_norm, output_xyz_norm, test_size=0.1, random_state=42)

    # SVR configurations
    configurations = [
        {'model': SVR, 'kernel': 'linear', 'C': 1},
        {'model': SVR, 'kernel': 'poly', 'C': 1, 'degree': 2},
        {'model': SVR, 'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
        {'model': SVR, 'kernel': 'sigmoid', 'C': 1, 'gamma': 'scale'}
    ]

    # Train and evaluate each configuration
    for index, config in enumerate(configurations):
        model = config['model'](kernel=config['kernel'], C=config['C'], degree=config.get('degree', 3), gamma=config.get('gamma', 'scale'))

        model.fit(input_train, output_train)
        output_pred_norm = model.predict(input_test)

        output_pred_lab = xyz2lab(output_pred_norm)
        output_test_lab = xyz2lab(output_test)

        errors = np.sqrt(np.sum((output_pred_lab - output_test_lab) ** 2, axis=1))
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        max_error = np.max(errors)

        results.loc[index] = [str(config), mean_error, median_error, max_error]
        visualize_lab_values(output_test_lab, output_pred_lab)

    best_result = results.loc[results['Mean Error'].idxmin()]
    best_result_df = pd.DataFrame([best_result], index=['Best Configuration'])
    final_results = pd.concat([results, best_result_df])

    print("All Results:")
    print(final_results)

    csv_file_path = save_results_to_CSV(final_results, script_name=__file__)
    print(f"Results saved to '{csv_file_path}'")