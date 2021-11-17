"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.DataManager import DataManager


class PlottingManager:
    def __init__(self, *,
                 data_manager: DataManager) -> None:
        self.data_manager = data_manager
        self.y_top_margin = 0.05

    def plot_time_series(self) -> None:
        df_dict = {
            'Value (raw)': self.data_manager.data,
            'Value (normalized)': self.data_manager.data_norm
        }

        for xlabel, df in df_dict.items():
            self.plot_dataframe(df=df, xlabel=xlabel)

    def plot_dataframe(self, *,
                       df: pd.DataFrame,
                       xlabel: str) -> None:
        plt.clf()

        df.plot(x='x',
                y=df.columns[1:],
                xlabel='Time',
                ylabel=xlabel,
                xlim=(df['x'].min(), df['x'].max()),
                ylim=(0, (1 + self.y_top_margin) * df[df.columns[1:]].max().max()),
                legend=False,
                fontsize=9)

        plt.tight_layout()
        plt.show()
