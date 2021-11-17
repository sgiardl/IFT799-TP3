"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from src.DataManager import DataManager
from src.PlottingManager import PlottingManager

if __name__ == '__main__':
    data_manager = DataManager(file_path='data/res_2000.csv',
                               date_start='2006-01-01',
                               series_length=1260,
                               n_series=100,
                               norm_method='z-score-shifted',
                               window_size=21,
                               jump_size=10)

    plotting_manager = PlottingManager(data_manager=data_manager)

    plotting_manager.plot_full_time_series()
    plotting_manager.plot_windows(series_name='FPX1')
