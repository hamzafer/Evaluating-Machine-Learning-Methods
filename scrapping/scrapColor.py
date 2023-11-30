import os
from io import StringIO

import pandas as pd
import requests

from utils.lab2xyz import lab2xyz


def download_and_convert_to_csv(file_name, base_url='https://color.org/chardata/'):
    file_url = f"{base_url}{file_name}"
    response = requests.get(file_url)

    if response.status_code == 200:
        # Use StringIO to treat the string content as a file
        content = StringIO(response.content.decode('utf-8'))
        df = pd.read_csv(content, delimiter="\t")

        # Assuming the data format and data sections are labeled in the file
        data_format = df[df.iloc[:, 0] == "BEGIN_DATA_FORMAT"].index[0] + 1
        data_start = df[df.iloc[:, 0] == "BEGIN_DATA"].index[0] + 1
        data_end = df[df.iloc[:, 0] == "END_DATA"].index[0]

        # Extract fields and data
        fields = df.iloc[data_format].values.tolist()
        data = df.iloc[data_start:data_end].values.tolist()

        # Create a new dataframe with the correct fields
        data_df = pd.DataFrame(data, columns=fields)

        # Convert LAB to XYZ
        lab_data = data_df[['LAB_L', 'LAB_A', 'LAB_B']].apply(pd.to_numeric, errors='coerce')
        xyz_data = lab2xyz(lab_data.values)
        data_df['XYZ_X'] = xyz_data[:, 0]
        data_df['XYZ_Y'] = xyz_data[:, 1]
        data_df['XYZ_Z'] = xyz_data[:, 2]

        # Define the cleaned directory
        cleaned_dir = '../cleaned'
        if not os.path.exists(cleaned_dir):
            os.makedirs(cleaned_dir)

        # Save to CSV in the cleaned directory
        csv_file_name = os.path.join(cleaned_dir, file_name.replace('.txt', '.csv'))
        data_df.to_csv(csv_file_name, index=False)

        return csv_file_name
    else:
        print(f'Failed to download the file. Status code: {response.status_code}')
        return None


# Replace 'file_name' with the actual file name you wish to download
file_name = 'APTEC_PC10_CardBoard_2023_v1.txt'
csv_file = download_and_convert_to_csv(file_name)

if csv_file:
    print(f'CSV file with XYZ data is ready for use: {csv_file}')
