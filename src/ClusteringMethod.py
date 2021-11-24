"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import rand_score, adjusted_rand_score


class ClusteringMethod(ABC):
    def __init__(self, *,
                 max_iter: int,
                 tol: float,
                 c_min: int,
                 c_max: int,
                 metric: str,
                 seed: int) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.c_min = c_min
        self.c_max = c_max
        self.metric = metric
        self.seed = seed

        self.results = {}
        self.results_optimal = {}
        self.largest_cluster_curve = None
        self.rand_curve = None

    @abstractmethod
    def __call__(self) -> None:
        return

    def analyze_results(self) -> None:
        self.get_largest_cluster_curve()
        self.get_rand_curve()

    def get_largest_cluster_curve(self) -> None:
        columns = ['window_start', 'largest_cluster_size']
        df = pd.DataFrame(columns=columns)

        for window_start, results_dict in self.results_optimal.items():
            df = df.append({columns[0]: window_start,
                            columns[1]: results_dict[columns[1]]},
                           ignore_index=True)

        df[columns[0]] = pd.to_datetime(df[columns[0]])

        self.largest_cluster_curve = df

    def get_rand_curve(self) -> None:
        columns = ['window_start', 'rand_score', 'adjusted_rand_score']
        df = pd.DataFrame(columns=columns)

        dates = sorted(list(self.results_optimal.keys()))

        for i, (window_start, results_dict) in enumerate(self.results_optimal.items()):
            if i > 0:
                rand = rand_score(self.results_optimal[dates[i - 1]]['memberships'],
                                  self.results_optimal[dates[i]]['memberships'])

                rand_adj = adjusted_rand_score(self.results_optimal[dates[i - 1]]['memberships'],
                                               self.results_optimal[dates[i]]['memberships'])

                df = df.append({columns[0]: window_start,
                                columns[1]: rand,
                                columns[2]: rand_adj},
                               ignore_index=True)

        df[columns[0]] = pd.to_datetime(df[columns[0]])

        self.rand_curve = df
