"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from collections import Counter
import pandas as pd
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score

from src.ClusteringMethod import ClusteringMethod


class KMeans(ClusteringMethod):
    def __init__(self, *,
                 max_iter: int = 50,
                 tol: float = 1e-6,
                 c_min: int = 2,
                 c_max: int = 10,
                 metric: str = 'euclidean',
                 seed: int = 54288) -> None:
        super().__init__(max_iter=max_iter,
                         tol=tol,
                         c_min=c_min,
                         c_max=c_max,
                         metric=metric,
                         seed=seed)

    def __call__(self,
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

            memberships = km.labels_

            count = Counter(memberships.tolist())
            largest_cluster = count.most_common(1)[0]

            silhouette = silhouette_score(x,
                                          memberships,
                                          metric=self.metric)
            silhouette_dict[f'{c=}'] = silhouette

            results[f'{c=}'] = {'km': km,
                                'c': c,
                                'silhouette': silhouette,
                                'cluster_centers': km.cluster_centers_.squeeze(),
                                'memberships': memberships,
                                'largest_cluster': largest_cluster[0],
                                'largest_cluster_size': largest_cluster[1]}

        self.results[window_start] = results

        optimal_c = max(silhouette_dict, key=silhouette_dict.get)

        self.results_optimal[window_start] = self.results[window_start][optimal_c]
