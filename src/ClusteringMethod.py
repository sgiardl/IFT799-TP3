"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from abc import ABC
import pandas as pd


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

    def get_largest_cluster_curve(self) -> None:
        columns = ['window_start', 'largest_cluster_size']
        df = pd.DataFrame(columns=columns)

        for window_start, results_dict in self.results_optimal.items():
            df = df.append({columns[0]: window_start,
                            columns[1]: results_dict[columns[1]]},
                           ignore_index=True)

        df[columns[0]] = pd.to_datetime(df[columns[0]])

        self.largest_cluster_curve = df
