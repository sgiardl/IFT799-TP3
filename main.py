"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from tqdm import tqdm

from src.DataManager import DataManager
from src.PlottingManager import PlottingManager
from src.FCM import FCM
from src.KMeans import KMeans

if __name__ == '__main__':
    plot_clusters = False

    data_manager = DataManager(file_path='data/res_2000.csv',
                               date_start='2000-01-01',
                               series_length=5221,
                               n_series=287,
                               norm_method='z-score-shifted',
                               window_size=21,
                               jump_size=10)

    plotting_manager = PlottingManager()

    plotting_manager.plot_full_time_series(data_manager=data_manager)
    plotting_manager.plot_all_windows_for_series(data_manager=data_manager,
                                                 series_name='FPX1')
    plotting_manager.plot_all_series_for_window(data_manager=data_manager,
                                                window_start='2000-12-29')

    clustering_methods = {
        'K-Means': KMeans(),
        'FCM': FCM(),
        'DTW': KMeans(metric='dtw')
    }

    for name, method in clustering_methods.items():
        for window_start, window in tqdm(data_manager.data_split_norm.items(),
                                         desc=f'{name}: Finding optimal clusters...'):
            method(window_start=window_start, window=window)

            if plot_clusters:
                plotting_manager.plot_clustering_results(method=name,
                                                         results=method.results_optimal[window_start],
                                                         window_start=window_start,
                                                         window=window,
                                                         c=method.results_optimal[window_start]['c'])

        method.analyze_results()

        plotting_manager.plot_largest_cluster_curve(method=name,
                                                    df=method.largest_cluster_curve)
        plotting_manager.plot_rand_curve(method=name,
                                         df=method.rand_curve)
