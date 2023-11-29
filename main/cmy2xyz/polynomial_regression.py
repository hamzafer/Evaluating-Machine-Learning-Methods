import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from input.input import get_dataset
from utils.save_results import save_results_to_CSV
from utils.xyz2lab import xyz2lab


def process(dataset_name, input_type, output_type, degree, visualize=False):
    input_data = get_dataset(dataset_name, input_type)
    output_data = get_dataset(dataset_name, output_type)

    # Normalize the data to the range [0, 1]
    scaler = MinMaxScaler()
    input_cmy_norm = scaler.fit_transform(input_data)
    output_xyz_norm = scaler.fit_transform(output_data)

    # Split the data into training and testing sets
    cmy_train, cmy_test, xyz_train, xyz_test = train_test_split(input_cmy_norm, output_xyz_norm, test_size=0.1,
                                                                random_state=42)

    # Create a pipeline that creates polynomial features and then applies linear regression
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())

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

    # Create the results DataFrame
    results = pd.DataFrame(columns=['Polynomial Degree', 'Mean Euclidean Error', 'Median Euclidean Error', 'Max Euclidean Error'])
    results.loc[0] = [degree, mean_error, median_error, max_error]

    # Save results
    csv_file_path = save_results_to_CSV(results, dataset_name, script_name=__file__)

    print(f"Results saved to '{csv_file_path}'")
