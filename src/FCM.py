"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import numpy as np
import pandas as pd
from skfuzzy.cluster import cmeans
from sklearn.metrics import silhouette_score


class FCM:
    def __init__(self,
                 c_min: int = 2,
                 c_max: int = 10,
                 m: int = 2,
                 error: float = 0.005,
                 maxiter: int = 1000,
                 metric: str = 'euclidean',
                 seed: int = 54288) -> None:

        self.c_min = c_min
        self.c_max = c_max
        self.m = m
        self.error = error
        self.maxiter = maxiter
        self.metric = metric
        self.seed = seed

        self.results = {}
        self.results_optimal = {}

    def run_fcm(self,
                window_start: str,
                window: pd.DataFrame) -> None:
        data = self.process_data(window)

        results = {}
        silhouette_dict = {}

        for c in range(self.c_min, self.c_max + 1):
            cntr, u, u0, d, jm, p, fpc = cmeans(data=data,
                                                c=c,
                                                m=self.m,
                                                error=self.error,
                                                maxiter=self.maxiter,
                                                metric=self.metric,
                                                init=None,
                                                seed=self.seed)

            cm = {'c': c, 'cntr': cntr, 'u': u, 'u0': u0, 'd': d, 'jm': jm, 'p': p, 'fpc': fpc,
                  'memberships': np.argmax(u, axis=0),
                  'silhouette': silhouette_score(data.T,
                                                 np.argmax(u, axis=0),
                                                 metric=self.metric)}

            silhouette_dict[f'{c=}'] = cm['silhouette']

            results[f'{c=}'] = cm

        self.results[window_start] = results

        optimal_c = max(silhouette_dict, key=silhouette_dict.get)

        self.results_optimal[window_start] = self.results[window_start][optimal_c]

    @staticmethod
    def process_data(df: pd.DataFrame) -> np.array:
        data = np.zeros((len(df), len(df.columns) - 1))

        for i, (_, row) in enumerate(df.iterrows()):
            for j, col in enumerate(df.columns[1:]):
                data[i, j] = row[col]

        return data
