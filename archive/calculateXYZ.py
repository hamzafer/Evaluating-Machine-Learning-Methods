import numpy as np
import pandas as pd

from utils.lab2xyz import lab2xyz


# Function to convert LAB to XYZ
def convert_lab_to_xyz(df):
    # Convert each LAB row to XYZ and store in new columns within the same DataFrame
    for index, row in df.iterrows():
        # Create a 2D array with a single LAB color value
        lab_value = np.array([[row['LAB_L'], row['LAB_A'], row['LAB_B']]])
        # Convert and extract the XYZ value
        xyz_value = lab2xyz(lab_value)[0]
        # Create new columns for XYZ in the DataFrame
        df.at[index, 'XYZ_X'] = xyz_value[0]
        df.at[index, 'XYZ_Y'] = xyz_value[1]
        df.at[index, 'XYZ_Z'] = xyz_value[2]

    return df


# Read the CSV file
file_path = '/Users/stan/PycharmProjects/ColorProject/cleaned/FOGRA51.csv'
fogra51_df = pd.read_csv(file_path)

# Convert the DataFrame
updated_df = convert_lab_to_xyz(fogra51_df)

# Save the updated DataFrame to the same CSV file (or choose a new file to keep both versions)
output_file_path = '/cleaned/FOGRA51.csv'
updated_df.to_csv(output_file_path, index=False)
