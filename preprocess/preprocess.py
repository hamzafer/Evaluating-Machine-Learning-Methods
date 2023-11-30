import pandas as pd

from utils.lab2xyz import lab2xyz


def convert_lab_to_xyz(data_df):
    # Convert LAB to XYZ
    lab_data = data_df[['LAB_L', 'LAB_A', 'LAB_B']].apply(pd.to_numeric, errors='coerce')
    xyz_data = lab2xyz(lab_data.values)
    data_df['XYZ_X'] = xyz_data[:, 0]
    data_df['XYZ_Y'] = xyz_data[:, 1]
    data_df['XYZ_Z'] = xyz_data[:, 2]

    return data_df
