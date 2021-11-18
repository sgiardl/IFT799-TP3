"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from math import ceil, floor
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)

    except FileNotFoundError:
        raise FileNotFoundError(f"Please download the dataset and save it as '{filepath}' to use this script")

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

    data = data.fillna(0)  # Replace nan with 0 when data.std() == nan (1 unique value in data window)
    return dates_col.join(data)


def save_dataframe(data: pd.DataFrame, *,
                   filename: str) -> None:
    data.to_csv(f'csv/{filename}.csv', index=False)

    with pd.option_context('max_colwidth', 1000):
        with open(f'latex/{filename}.txt', 'w') as file:
            file.write(data.to_latex(index=False, escape=False))


def split_data(data: pd.DataFrame,
              window_size: int,
              jump_size: int,
          *,
          normalize_windows: bool = True,
          norm_method: str = 'z-score-shifted') -> pd.DataFrame:

    data_list = []

    # else:
    for i in range(floor(len(data) / jump_size)):  # floor or ceil, to choose, but ceil may give datasets with singleton
        data_window = data[i * jump_size:i * jump_size + window_size]
        data_normalied = normalize(data_window, norm_method)
        data_list.append(data_normalied if normalize_windows else data_window)

    return data_list
