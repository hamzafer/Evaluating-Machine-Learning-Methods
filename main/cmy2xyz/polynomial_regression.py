import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from input.input import get_dataset
from utils.calcError import error
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

    # Calculate the error between the predicted and actual Lab values
    errors = error(lab_pred, lab_test)

    # Aggregate error statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)
    p95_error = np.percentile(errors, 95)
    std_error = np.std(errors, ddof=1)
    print('Mean error:', mean_error)
    print('Median error:', median_error)
    print('Max error:', max_error)

    # Create the results DataFrame
    results = pd.DataFrame(columns=['Configuration', 'Mean Error', 'Median Error', 'Max Error', 'P95 Error', 'Std Dev'])
    results.loc[0] = [degree, mean_error, median_error, max_error, p95_error, std_error]

    # Save results: start a fresh file at degree 1, then append for others
    append_flag = False if degree == 1 else True
    csv_file_path = save_results_to_CSV(results, dataset_name, script_name=__file__, append=append_flag)

    print(f"Results saved to '{csv_file_path}'")
