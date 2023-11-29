import os


def save_results_to_CSV(dataframe, dataset_name, script_name, append=False):
    # Create a directory path for results
    directory_path = os.path.join('results', dataset_name)
    os.makedirs(directory_path, exist_ok=True)

    # Remove the file extension from script_name if present
    base_script_name = os.path.splitext(os.path.basename(script_name))[0]

    # Construct the file path
    file_name = f"{base_script_name}_results.csv"
    file_path = os.path.join(directory_path, file_name)

    if append and os.path.exists(file_path):
        # Append without header if file exists and appending is specified
        dataframe.to_csv(file_path, mode='a', header=False, index=False)
    else:
        # Write new file or overwrite if append is False
        dataframe.to_csv(file_path, index=False)

    return file_path
