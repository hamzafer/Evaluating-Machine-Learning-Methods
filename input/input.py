import os

import pandas as pd


def get_dataset(dataset_code, subset_type):
    # Get the absolute path of the current script
    current_script_path = os.path.abspath(__file__)

    # Get the directory containing the current script
    current_dir = os.path.dirname(current_script_path)

    # Define the relative path to the cleaned directory from the current script
    cleaned_dir = os.path.join(current_dir, '..', 'cleaned')

    # Define dataset paths using the cleaned directory
    datasets = {
        'PC10': os.path.join(cleaned_dir, 'APTEC_PC10_CardBoard_2023_v1.csv'),
        'PC11': os.path.join(cleaned_dir, 'APTEC_PC11_CCNB_2023_v1.csv'),
    }

    # Check if the dataset code is valid
    if dataset_code not in datasets:
        raise ValueError("Invalid dataset code. Choose 'PC10' or 'PC11'.")

    # Read the selected dataset
    data = pd.read_csv(datasets[dataset_code])

    # Define column subsets
    columns = {
        'CMY': ['CMYK_C', 'CMYK_M', 'CMYK_Y'],
        'CMYK': ['CMYK_C', 'CMYK_M', 'CMYK_Y', 'CMYK_K'],
        'LAB': ['LAB_L', 'LAB_A', 'LAB_B'],
        'XYZ': ['XYZ_X', 'XYZ_Y', 'XYZ_Z']
    }

    # Select and return the relevant columns
    if subset_type in columns:
        return data[columns[subset_type]]
    else:
        raise ValueError("Invalid subset type. Choose from 'CMY', 'CMYK', 'LAB', 'XYZ'.")

# Example usage
# dataset1 = get_dataset('PC10', 'LAB')
# dataset2 = get_dataset('PC11', 'LAB')
# print(dataset1, dataset2)
