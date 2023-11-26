import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from input.input import get_dataset
from utils.save_results import save_results_to_excel
from utils.xyz2lab import xyz2lab
from visual.plot_loss import plot_loss
from visual.vis_lab import visualize_lab_values


# Normalize the data to be in the range [0, 1]
def normalize_data(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


results = pd.DataFrame(columns=['Configuration', 'Mean Error', 'Median Error', 'Max Error'])

# Assuming `input_cmy` is your CMY data and `output_lab` is your XYZ data as numpy arrays
# Dummy data for demonstration; replace these with your actual data
input_cmy = get_dataset('PC10', 'CMY')
output_xyz = get_dataset('PC10', 'XYZ')

# Normalize the data
input_cmy_norm = normalize_data(input_cmy)
output_xyz_norm = normalize_data(output_xyz)

# Split the dataset into training and testing sets (90% train, 10% test)
input_train, input_test, output_train, output_test = train_test_split(input_cmy_norm, output_xyz_norm, test_size=0.1,
                                                                      random_state=42)

configurations = [
    {'hidden_layer_sizes': (6,), 'activation': 'logistic', 'solver': 'lbfgs', 'max_iter': 1000},
    {'hidden_layer_sizes': (6,), 'activation': 'logistic', 'solver': 'adam', 'max_iter': 1000},
    {'hidden_layer_sizes': (6,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 1000},
    {'hidden_layer_sizes': (6,), 'activation': 'tanh', 'solver': 'sgd', 'max_iter': 1000},
    {'hidden_layer_sizes': (10,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 200},
    {'hidden_layer_sizes': (50, 30, 10), 'activation': 'tanh', 'solver': 'sgd', 'max_iter': 500},
    {'hidden_layer_sizes': (10,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 200},
    {'hidden_layer_sizes': (20, 10), 'activation': 'relu', 'solver': 'adam', 'max_iter': 300},
    {'hidden_layer_sizes': (30, 20, 10), 'activation': 'relu', 'solver': 'adam', 'max_iter': 400},
    {'hidden_layer_sizes': (20,), 'activation': 'tanh', 'solver': 'adam', 'max_iter': 200},
    {'hidden_layer_sizes': (20,), 'activation': 'logistic', 'solver': 'adam', 'max_iter': 200},
    {'hidden_layer_sizes': (20,), 'activation': 'relu', 'solver': 'sgd', 'max_iter': 500},
    {'hidden_layer_sizes': (20,), 'activation': 'relu', 'solver': 'lbfgs', 'max_iter': 200},
    {'hidden_layer_sizes': (20,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 1000}
]

# Loop over each configuration
for index, config in enumerate(configurations):
    mlp = MLPRegressor(
        hidden_layer_sizes=config['hidden_layer_sizes'],
        activation=config['activation'],
        solver=config['solver'],
        max_iter=config['max_iter'],
        random_state=42
    )
    # Train the neural network
    mlp.fit(input_train, output_train)

    # Predict the output from the test input
    output_pred = mlp.predict(input_test)

    # Convert predicted XYZ to LAB
    output_pred_lab = xyz2lab(output_pred)

    xyz_data = output_test[['XYZ_X', 'XYZ_Y', 'XYZ_Z']].values
    # Convert true XYZ to LAB for the test set
    output_test_lab = xyz2lab(xyz_data)

    # Calculate the Euclidean distance (error) between the predicted and true LAB values
    errors = np.sqrt(np.sum((output_pred_lab - output_test_lab) ** 2, axis=1))

    # Output the mean Euclidean error
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)

    # Add results to DataFrame
    results.loc[index] = [str(config), mean_error, median_error, max_error]

    # Visualize the predicted vs true LAB values
    visualize_lab_values(output_test_lab, output_pred_lab)
    if config['solver'] in ['sgd', 'adam']:
        loss_values = mlp.loss_curve_
        losses = plot_loss(loss_values)


# Identify the best configuration based on the lowest mean error
best_result = results.loc[results['Mean Error'].idxmin()]
# Add the best configuration as a new row or a separate section in the DataFrame
best_result_df = pd.DataFrame([best_result], index=['Best Configuration'])
final_results = pd.concat([results, best_result_df])

# Print all results and the best configuration
print("All Results:")
print(final_results)

# Use the updated function to save the results, now including the best configuration
excel_file_path = save_results_to_excel(final_results, script_name=__file__)

print(f"Results saved to '{excel_file_path}'")
