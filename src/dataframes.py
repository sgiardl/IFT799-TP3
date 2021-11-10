"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)

    except FileNotFoundError:
        raise FileNotFoundError(f"Please download the dataset and save it as '{filepath}' to use this script")
