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
                 data_manager: DataManager,
                 y_top_margin: float = 0.05) -> None:
        self.data_manager = data_manager

        self.date_min = self.data_manager.data['x'].min()
        self.date_max = self.data_manager.data['x'].max()

        self.y_top_margin = y_top_margin

    def plot_full_time_series(self) -> None:
        df_dict = {
            'Value (raw)': self.data_manager.data,
            # 'Value (normalized)': self.data_manager.data_norm
        }

        for xlabel, df in df_dict.items():
            self.plot_full_dataframe(df=df, xlabel=xlabel)

    def plot_full_dataframe(self, *,
                            df: pd.DataFrame,
                            xlabel: str) -> None:
        plt.clf()

        df.plot(x='x',
                y=df.columns[1:],
                xlabel='Time',
                ylabel=xlabel,
                xlim=(self.date_min, self.date_max),
                ylim=(0, (1 + self.y_top_margin) * df[df.columns[1:]].max().max()),
                legend=False,
                fontsize=9)

        plt.tight_layout()
        plt.show()

    def plot_all_windows_for_series(self, *,
                                    series_name: str) -> None:
        plt.clf()

        fig, ax = plt.subplots()

        val_max = 0

        for df in self.data_manager.data_split_norm.values():
            df.plot(ax=ax,
                    x='x',
                    y=series_name)

            if df[series_name].max() > val_max:
                val_max = df[series_name].max()

        legend = ax.get_legend()
        legend.remove()

        plt.xlabel('Time')
        plt.ylabel(f'Series: {series_name}, Value (normalized)')
        plt.xlim((self.date_min, self.date_max))
        plt.ylim((0, (1 + self.y_top_margin) * val_max))

        plt.tight_layout()
        plt.show()

    def plot_all_series_for_window(self,
                                   window_start: str) -> None:
        plt.clf()

        fig, ax = plt.subplots()

        df = self.data_manager.data_split_norm[window_start]

        for series in df.columns[1:]:
            df.plot(ax=ax,
                    x='x',
                    y=series)

        legend = ax.get_legend()
        legend.remove()

        plt.xlabel('Time')
        plt.ylabel(f'Window Start: {window_start}, Value (normalized)')
        plt.xlim((df['x'].min(), df['x'].max()))
        plt.ylim((0, (1 + self.y_top_margin) *
                 df[df.columns[1:]].max().max()))

        plt.tight_layout()
        plt.show()
