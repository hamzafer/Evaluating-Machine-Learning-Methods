import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from input.input import get_dataset
from utils.xyz2lab import xyz2lab


# Normalize the data to be in the range [0, 1]
def normalize_data(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


# Assuming `input_cmy` is your CMY data and `output_xyz` is your XYZ data as numpy arrays
# Dummy data for demonstration; replace these with your actual data
input_cmy = get_dataset('PC10', 'CMY')
output_xyz = get_dataset('PC10', 'XYZ')

# Normalize the data
input_cmy_norm = normalize_data(input_cmy)
output_xyz_norm = normalize_data(output_xyz)

# Split the dataset into training and testing sets (90% train, 10% test)
input_train, input_test, output_train, output_test = train_test_split(input_cmy_norm, output_xyz_norm, test_size=0.1,
                                                                      random_state=42)

# Create the neural network model
mlp = MLPRegressor(hidden_layer_sizes=(6,), activation='logistic', solver='lbfgs', max_iter=1000, random_state=42)

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
print(f"Mean Euclidean error: {mean_error}")
