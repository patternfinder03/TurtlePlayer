# This script is used to filter and save filenames of stock files based on specific criteria.
# It identifies valid stock files within a specified date range, optionally excluding files with NaN values or negative price data.
# The valid filenames are then saved in a JSON file with additional metadata.

import pandas as pd
import os
import json
from tqdm import tqdm


save_directory = './actual/valid_stocks.json'
directory = './actual/data'


def save_stock_filenames_in_timespan_daily(start='20100104', end='20201231', exclude_nan=True, exclude_negative=True):
    """
    Save the filenames of stock files that meet specified criteria into a JSON file.

    This function filters stock files based on their coverage of a specific date range and optionally excludes files with NaN values
    or negative price data. The filtered filenames are then saved in a JSON file.

    Args:
        start (str): The start date in 'YYYYMMDD' format. Defaults to '20100104'.
        end (str): The end date in 'YYYYMMDD' format. Defaults to '20201231'.
        exclude_nan (bool): Whether to exclude files with NaN values. Defaults to True.
        exclude_negative (bool): Whether to exclude files with negative values in 'PRC' and 'OPENPRC'. Defaults to True.

    """

    # Clear the JSON file at the start
    with open(save_directory, 'w') as json_file:
        json_file.write("[]")

    # Get the number of timesteps for AAPL to use as benchmark
    aapl_path = os.path.join(directory, 'AAPL.csv')
    aapl_data = pd.read_csv(aapl_path)
    aapl_data['date'] = aapl_data['date'].astype(str).str.replace('Day_', '')
    filtered_aapl_data = aapl_data[(aapl_data['date'] >= start) & (aapl_data['date'] <= end)]
    num_timesteps = len(filtered_aapl_data)

    # Loop over every file in the directory
    valid_files = []
    for filename in tqdm(os.listdir(directory), desc="Finding Valid Stocks"):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            data['date'] = data['date'].astype(str)

            if start in data['date'].values and end in data['date'].values:
                filtered_data = data[(data['date'] >= start) & (data['date'] <= end)]

                if len(filtered_data) == num_timesteps:
                    if not exclude_nan or not data[['PRC', 'OPENPRC', 'Close', 'Low', 'High', 'Open']].isnull().any(
                            axis=1).any():
                        if not exclude_negative or (data['PRC'].ge(0).all() and data['OPENPRC'].ge(0).all()) and (
                                data['High'].ge(0).all()) and (data['Low'].ge(0).all()):
                            # If passes all conditions add to valid files
                            valid_files.append(filename)

    # Save the list of filenames to the JSON file
    with open(save_directory, 'w') as json_file:
        json_data = {
            'valid_files': valid_files,
            'start_date': start,
            'end_date': end,
            'num_timesteps': num_timesteps,
            'exclude_nan': exclude_nan,
            'exclude_positive': exclude_negative,
            'num_stocks': len(valid_files)
        }
        json.dump(json_data, json_file, indent=4)
        
        
if __name__ == '__main__':
    save_stock_filenames_in_timespan_daily()