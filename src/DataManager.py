"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import pandas as pd
from math import ceil


class DataManager:
    def __init__(self, *,
                 file_path: str,
                 date_start: str,
                 series_length: int,
                 n_series: int,
                 norm_method: str,
                 window_size: int,
                 jump_size: int) -> None:
        norm_methods = ['none', 'min-max', 'z-score', 'z-score-shifted']

        if norm_method not in norm_methods:
            raise ValueError(f'Wrong normalization method, valid choices are: {norm_methods}')

        try:
            data = pd.read_csv(file_path)

        except FileNotFoundError:
            raise FileNotFoundError(f"Please download the dataset and save it as '{file_path}' to use this script")

        data['x'] = pd.to_datetime(data['x'])

        self.data = self.filter(data, date_start, series_length, n_series)
        self.data_split = self.split(self.data, window_size, jump_size)

        # Normalized independently for each window
        self.data_split_norm = {}

        for window_start, df in self.data_split.items():
            self.data_split_norm[window_start] = self.normalize(df, norm_method)

        # Normalized with regards to the full data
        # self.data_norm = self.normalize(self.data, norm_method)
        # self.data_norm_split = self.split(self.data_norm, window_size, jump_size)

    @staticmethod
    def filter(df: pd.DataFrame,
               date_start: str,
               series_length: int,
               n_series: int) -> pd.DataFrame:

        return df.loc[df['x'] >= date_start].head(series_length).iloc[:, : n_series + 1]

    @staticmethod
    def normalize(df: pd.DataFrame,
                  norm_method: str) -> pd.DataFrame:
        dates_col = pd.DataFrame(df['x'])
        data = df[df.columns[1:]]

        if norm_method == 'min-max':
            data = (data - data.min()) / (data.max() - data.min())

        elif norm_method == 'z-score':
            data = (data - data.mean()) / data.std()

        elif norm_method == 'z-score-shifted':
            data = (data - data.mean()) / data.std()
            data += abs(data.min())

        return dates_col.join(data)

    @staticmethod
    def split(df: pd.DataFrame,
              window_size: int,
              jump_size: int) -> pd.DataFrame:
        data_dict = {}

        for i in range(ceil(len(df) / jump_size)):
            index = i * jump_size
            data_dict[df.iloc[index]['x'].strftime('%Y-%m-%d')] = \
                df[index: index + window_size]

        return data_dict
