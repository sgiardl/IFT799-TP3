"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from math import ceil
import matplotlib.pyplot as plt
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

        self.plot_results(window_start, window, self.results_optimal[window_start].n_clusters)

    def plot_results(self,
                     window_start: str,
                     window: pd.DataFrame,
                     c: int) -> None:
        plt.clf()

        results = self.results[window_start][f'c={c}']

        n_rows = ceil(c ** (1 / 2))
        n_cols = ceil(c / n_rows)

        fig, axes = plt.subplots(n_rows, n_cols)

        k = 0

        for i, row in enumerate(axes):
            if c > 2:
                for j, col in enumerate(row):
                    try:
                        self.plot_cluster(axes[i, j], window, results, k)
                        k += 1

                    except IndexError:
                        axes[i, j].set_axis_off()
            else:
                self.plot_cluster(axes[i], window, results, k)
                k += 1

        plt.suptitle(f'{window_start=}, n_clusters={c}, silhouette={results.silhouette:.4f}')

        plt.tight_layout()

        plt.show()


    @staticmethod
    def plot_cluster(ax,
                     window: pd.DataFrame,
                     results: TimeSeriesKMeans,
                     cluster_n: int,
                     color_members: str = 'blue',
                     color_center: str = 'red') -> None:
        for col, membership in zip(window.columns[1:], results.labels_):
            if membership == cluster_n:
                ax.plot(window[col].to_list(), color=color_members, alpha=0.3)

        ax.plot(results.cluster_centers_[cluster_n, :, 0], color=color_center)

        ax.set_title(f'Cluster {cluster_n + 1}')

    # @staticmethod
    # def process_data(df: pd.DataFrame) -> np.array:

