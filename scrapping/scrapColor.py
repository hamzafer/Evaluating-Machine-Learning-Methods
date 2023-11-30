import requests
import csv


def download_file(file_name, base_url='https://color.org/chardata/'):
    # Construct the URL for the file
    file_url = f"{base_url}{file_name}"

    # Send a request to the URL
    response = requests.get(file_url)

    # Check if the request was successful
    if response.status_code == 200:
        content = response.content.decode('utf-8')

        # Pre-process the content to extract fields and data
        lines = content.splitlines()
        fields_section = False
        data_section = False
        fields = []
        data = []
        for line in lines:
            if 'BEGIN_DATA_FORMAT' in line:
                fields_section = True
            elif 'END_DATA_FORMAT' in line:
                fields_section = False
            elif fields_section and 'SAMPLE_ID' in line:
                fields = line.strip().split('\t')
            elif 'BEGIN_DATA' in line:
                data_section = True
            elif 'END_DATA' in line:
                data_section = False
            elif data_section:
                data.append(line.strip().split('\t'))

        # Write the fields and data to a CSV file
        csv_file_name = file_name.replace('.txt', '.csv')
        with open(csv_file_name, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            for entry in data:
                csvwriter.writerow(entry)

        print(f'Download and conversion to CSV complete: {csv_file_name}')
        return csv_file_name
    else:
        print(f'Failed to download the file. Status code: {response.status_code}')
        return None


# Replace 'file_name' with the actual file name you wish to download
file_name = 'APTEC_PC10_CardBoard_2023_v1.txt'
csv_file = download_file(file_name)

if csv_file:
    # If file was downloaded and converted successfully, you can now use the CSV file
    print(f'CSV file is ready for use: {csv_file}')
