"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import pandas as pd

from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score


class KMeans:
    def __init__(self,
                 c_min: int = 2,
                 c_max: int = 10,
                 max_iter: int = 50,
                 tol: float = 1e-6,
                 metric: str = 'euclidean',
                 seed: int = 54288) -> None:
        self.c_min = c_min
        self.c_max = c_max
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric
        self.seed = seed

        self.results = {}
        self.results_optimal = {}

    def run_k_means(self,
                    window_start: str,
                    window: pd.DataFrame) -> None:
        series_list = []

        for col in window.columns[1:]:
            series_list.append(window[col].to_list())

        x = to_time_series_dataset(series_list)

        results = {}
        silhouette_dict = {}

        for c in range(self.c_min, self.c_max + 1):
            km = TimeSeriesKMeans(n_clusters=c,
                                  max_iter=self.max_iter,
                                  tol=self.tol,
                                  metric=self.metric,
                                  random_state=self.seed).fit(x)

            km.silhouette = silhouette_score(x, km.labels_,
                                             metric=self.metric)
            results[f'{c=}'] = km
            silhouette_dict[f'{c=}'] = km.silhouette

        self.results[window_start] = results

        optimal_c = max(silhouette_dict, key=silhouette_dict.get)

        self.results_optimal[window_start] = self.results[window_start][optimal_c]
