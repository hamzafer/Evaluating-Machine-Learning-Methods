import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from input.input import get_dataset
from utils.save_results import save_results_to_CSV
from utils.xyz2lab import xyz2lab
from visual.vis_lab import visualize_lab_values


def process(dataset_name, input_type, output_type, visualize=False):
    results = pd.DataFrame(columns=['Configuration', 'Mean Error', 'Median Error', 'Max Error'])

    input_data = get_dataset(dataset_name, input_type)
    output_data = get_dataset(dataset_name, output_type)

    # Normalize the data to the range [0, 1]
    scaler = MinMaxScaler()
    input_cmy_norm = scaler.fit_transform(input_data)
    output_xyz_norm = scaler.fit_transform(output_data)

    # Split the dataset into training and testing sets (90% train, 10% test)
    input_train, input_test, output_train, output_test = train_test_split(input_cmy_norm, output_xyz_norm,
                                                                          test_size=0.1,
                                                                          random_state=42)

    # Deep Learning configurations (using MLPRegressor as an example of a neural network)
    configurations = [{'model': MLPRegressor, 'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam',
                       'random_state': 42},
                      {'model': MLPRegressor, 'hidden_layer_sizes': (50, 50), 'activation': 'tanh', 'solver': 'sgd',
                       'random_state': 42}, ]

    # Train and evaluate each configuration, which now includes Random Forest
    for index, config in enumerate(configurations):
        model = MLPRegressor(hidden_layer_sizes=config['hidden_layer_sizes'], activation=config['activation'], solver=config['solver'], random_state=config['random_state'])

        # Train the model
        model.fit(input_train, output_train)

        # Predict the output from the test input
        output_pred = model.predict(input_test)

        # Convert predicted XYZ to LAB
        output_pred_lab = xyz2lab(output_pred)  # Replace with your actual conversion after denormalization

        # Convert true XYZ to LAB for the test set
        output_test_lab = xyz2lab(output_test)

        # Calculate the Euclidean distance (error) between the predicted and true LAB values
        errors = np.sqrt(np.sum((output_pred_lab - output_test_lab) ** 2, axis=1))

        # Output the mean Euclidean error
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        max_error = np.max(errors)

        # Add results to DataFrame
        results.loc[index] = [str(config), mean_error, median_error, max_error]

        if visualize:
            # Visualize the predicted vs true LAB values
            visualize_lab_values(output_test_lab, output_pred_lab)

    # Identify the best configuration based on the lowest mean error
    best_result = results.loc[results['Mean Error'].idxmin()]

    # Add the best configuration as a new row or a separate section in the DataFrame
    best_result_df = pd.DataFrame([best_result], index=['Best Configuration'])
    final_results = pd.concat([results, best_result_df])

    # Print all results and the best configuration
    print("All Results:")
    print(final_results)

    # Use the updated function to save the results, now including the best configuration
    csv_file_path = save_results_to_CSV(final_results, dataset_name, script_name=__file__)

    print(f"Results saved to '{csv_file_path}'")
