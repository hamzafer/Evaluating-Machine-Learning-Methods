import os


def save_results_to_CSV(results, results_dir='results', script_name=None):
    """
    Saves the given DataFrame to a CSV file within the specified results directory.

    :param results: DataFrame containing the results to save.
    :param results_dir: The directory where the results CSV file will be saved.
    :param script_name: The name of the script, used to create the CSV filename. If not provided, defaults to 'results'.
    """
    # Create the results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # If no script_name is provided, use a default filename
    if script_name is None:
        script_name = 'results'
    else:
        # Remove the '.py' if it is included in the script_name
        script_name = os.path.splitext(os.path.basename(script_name))[0]

    # Construct the filename and the full path
    results_filename = f"{script_name}_results.csv"
    csv_file_path = os.path.join(results_dir, results_filename)

    # Save the DataFrame to a CSV file
    results.to_csv(csv_file_path, index=False)

    # Return the path where the file was saved
    return csv_file_path