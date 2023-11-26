import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from input.input import get_dataset
from utils.save_results import save_results_to_excel
from utils.xyz2lab import xyz2lab

# Assume dummy CMY and XYZ data
# Let's generate dummy data with 1617 samples
# For the inputs (CMY), we use random numbers to simulate the colorant values
# For the outputs (XYZ), we'll use random numbers to simulate the device measurements
np.random.seed(0)  # For reproducibility
cmy_data = get_dataset('PC10', 'CMY')
xyz_data = get_dataset('PC10', 'XYZ')

# Normalize the data to the range [0, 1]
scaler = MinMaxScaler()
cmy_data_normalized = scaler.fit_transform(cmy_data)
xyz_data_normalized = scaler.fit_transform(xyz_data)

# Split the data into training and testing sets
cmy_train, cmy_test, xyz_train, xyz_test = train_test_split(cmy_data_normalized, xyz_data_normalized, test_size=0.1, random_state=42)

# Create a pipeline that creates polynomial features and then applies linear regression
model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

# Train the model
model.fit(cmy_train, xyz_train)

# Predict using the test data
xyz_pred = model.predict(cmy_test)

# Convert predicted and test XYZ values to Lab
lab_pred = xyz2lab(xyz_pred)
lab_test = xyz2lab(xyz_test)

# Calculate the Euclidean error between the predicted and actual Lab values
errors = np.linalg.norm(lab_pred - lab_test, axis=1)

# Output the mean Euclidean error
mean_error = np.mean(errors)
median_error = np.median(errors)
max_error = np.max(errors)
print('Mean Euclidean error:', mean_error)
print('Median Euclidean error:', median_error)
print('Max Euclidean error:', max_error)

results = pd.DataFrame(columns=['Mean Euclidean error', 'Median Euclidean error', 'Max Euclidean error'])
results.loc[0] = [mean_error, median_error, max_error]
# Use the new function to save the results
excel_file_path = save_results_to_excel(results, script_name=__file__)

print(f"Results saved to '{excel_file_path}'")
