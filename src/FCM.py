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

        self.fcm_results_all = {}

    def run_fcm(self,
                window_start: str,
                window: pd.DataFrame) -> None:
        data = self.process_data(window)

        fcm_results = {'cntr': {}, 'u': {}, 'u0': {}, 'd': {},
                       'jm': {}, 'p': {}, 'fpc': {},
                       'membership': {}, 'silhouette': {}}

        for c in range(self.c_min, self.c_max + 1):
            cntr, u, u0, d, jm, p, fpc = cmeans(data=data,
                                                c=c,
                                                m=self.m,
                                                error=self.error,
                                                maxiter=self.maxiter,
                                                metric=self.metric,
                                                init=None,
                                                seed=self.seed)

            fcm_results['cntr'][f'{c=}'] = cntr
            fcm_results['u'][f'{c=}'] = u
            fcm_results['u0'][f'{c=}'] = u0
            fcm_results['d'][f'{c=}'] = d
            fcm_results['jm'][f'{c=}'] = jm
            fcm_results['p'][f'{c=}'] = p
            fcm_results['fpc'][f'{c=}'] = fpc
            fcm_results['membership'][f'{c=}'] = np.argmax(u, axis=0)
            fcm_results['silhouette'][f'{c=}'] = silhouette_score(data.T,
                                                                  fcm_results['membership'][f'{c=}'],
                                                                  metric=self.metric)

        self.fcm_results_all[window_start] = fcm_results

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
