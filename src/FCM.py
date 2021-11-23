"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from collections import Counter
import numpy as np
import pandas as pd
from skfuzzy.cluster import cmeans
from sklearn.metrics import silhouette_score

from src.ClusteringMethod import ClusteringMethod


class FCM(ClusteringMethod):
    def __init__(self, *,
                 m: int = 2,
                 max_iter: int = 1000,
                 tol: float = 0.005,
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

        self.m = m

    def run_fcm(self,
                window_start: str,
                window: pd.DataFrame) -> None:
        data = window[window.columns[1:]].to_numpy()

        results = {}
        silhouette_dict = {}

        for c in range(self.c_min, self.c_max + 1):
            cntr, u, u0, d, jm, p, fpc = cmeans(data=data,
                                                c=c,
                                                m=self.m,
                                                error=self.tol,
                                                maxiter=self.max_iter,
                                                metric=self.metric,
                                                init=None,
                                                seed=self.seed)

            cm = {'cntr': cntr, 'u': u, 'u0': u0, 'd': d, 'jm': jm, 'p': p, 'fpc': fpc}

            memberships = np.argmax(u, axis=0)

            count = Counter(memberships.tolist())
            largest_cluster = count.most_common(1)[0]

            silhouette = silhouette_score(data.T,
                                          memberships,
                                          metric=self.metric)
            silhouette_dict[f'{c=}'] = silhouette

            results[f'{c=}'] = {'cm': cm,
                                'c': c,
                                'silhouette': silhouette,
                                'cluster_centers': cntr,
                                'memberships': memberships,
                                'largest_cluster': largest_cluster[0],
                                'largest_cluster_size': largest_cluster[1]}

        self.results[window_start] = results

        optimal_c = max(silhouette_dict, key=silhouette_dict.get)

        self.results_optimal[window_start] = self.results[window_start][optimal_c]
