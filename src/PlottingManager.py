"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import matplotlib.pyplot as plt

from src.DataManager import DataManager


class PlottingManager:
    def __init__(self, *,
                 data_manager: DataManager) -> None:
        self.data_manager = data_manager

    def plot_time_series(self) -> None:
        self.data_manager.data_filtered.plot()
        plt.show()
