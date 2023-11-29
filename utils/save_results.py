import os


def save_results_to_CSV(dataframe, dataset_name, script_name, additional_info=''):
    # Create a directory path for results
    directory_path = os.path.join('results', dataset_name)
    os.makedirs(directory_path, exist_ok=True)

    # Remove the file extension from script_name if present
    base_script_name = os.path.splitext(os.path.basename(script_name))[0]

    # If there's additional info, append it to the filename
    if additional_info:
        file_name = f"{base_script_name}_{additional_info}_results.csv"
    else:
        file_name = f"{base_script_name}_results.csv"

    file_path = os.path.join(directory_path, file_name)

    # Save the DataFrame to CSV
    dataframe.to_csv(file_path, index=False)
    return file_path
