"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from skfuzzy.cluster import cmeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np


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

    def run_fcm(self,
                window_start: str,
                window: pd.DataFrame) -> None:
        data = self.process_data(window)

        results = {'cntr': {}, 'u': {}, 'u0': {}, 'd': {},
                   'jm': {}, 'p': {}, 'fpc': {},
                   'membership': {}, 'silhouette': {},
                   'c_optimal': None}

        for c in range(self.c_min, self.c_max + 1):
            cntr, u, u0, d, jm, p, fpc = cmeans(data=data,
                                                c=c,
                                                m=self.m,
                                                error=self.error,
                                                maxiter=self.maxiter,
                                                metric=self.metric,
                                                init=None,
                                                seed=self.seed)

            results['cntr'][f'{c=}'] = cntr
            results['u'][f'{c=}'] = u
            results['u0'][f'{c=}'] = u0
            results['d'][f'{c=}'] = d
            results['jm'][f'{c=}'] = jm
            results['p'][f'{c=}'] = p
            results['fpc'][f'{c=}'] = fpc
            results['membership'][f'{c=}'] = np.argmax(u, axis=0)
            results['silhouette'][f'{c=}'] = silhouette_score(data.T,
                                                              results['membership'][f'{c=}'],
                                                              metric=self.metric)
            results['c_optimal'] = int(max(results['silhouette'],
                                           key=results['silhouette'].get)[2:])

        self.results[window_start] = results

    @staticmethod
    def process_data(df: pd.DataFrame) -> np.array:
        x_pts = []
        y_pts = []

        dates_dict = {}

        for i, date in enumerate(df['x']):
            dates_dict[date] = i

        series = df.columns[1:].to_list()

        for _, row in df.iterrows():
            for col in series:
                x_pts.append(dates_dict[row['x']])
                y_pts.append(row[col])

        return np.vstack((np.array(x_pts), np.array(y_pts)))
