"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import pandas as pd
import numpy as np
from math import ceil


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)

    except FileNotFoundError:
        raise FileNotFoundError(f"Please download the dataset and save it as '{filepath}' to use this script")


def split_data(df: pd.DataFrame,
               window_size: int) -> dict:
    data_list = []

    for i in range(ceil(len(df) / window_size)):
        data_list.append(df[i * window_size:(i + 1) * window_size])

    return data_list
