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

from src.DataManager import DataManager


class PlottingManager:
    def __init__(self, *,
                 y_top_margin: float = 0.05) -> None:
        self.y_margin = y_top_margin

    def plot_full_time_series(self, *,
                              data_manager: DataManager) -> None:
        df_dict = {
            'Value (raw)': data_manager.data,
            # 'Value (normalized)': self.data_manager.data_norm
        }

        for xlabel, df in df_dict.items():
            self.plot_full_dataframe(df=df, xlabel=xlabel)

    def plot_full_dataframe(self, *,
                            df: pd.DataFrame,
                            xlabel: str) -> None:
        date_min = df['x'].min()
        date_max = df['x'].max()

        plt.clf()

        df.plot(x='x',
                y=df.columns[1:],
                xlabel='Time',
                ylabel=xlabel,
                xlim=(date_min, date_max),
                ylim=(0, (1 + self.y_margin) * df[df.columns[1:]].max().max()),
                legend=False,
                fontsize=9)

        plt.tight_layout()
        plt.show()

    def plot_all_windows_for_series(self, *,
                                    data_manager: DataManager,
                                    series_name: str) -> None:
        date_min = data_manager.data['x'].min()
        date_max = data_manager.data['x'].max()

        plt.clf()

        fig, ax = plt.subplots()

        val_max = 0

        for df in data_manager.data_split_norm.values():
            df.plot(ax=ax,
                    x='x',
                    y=series_name)

            if df[series_name].max() > val_max:
                val_max = df[series_name].max()

        legend = ax.get_legend()
        legend.remove()

        plt.xlabel('Time')
        plt.ylabel(f'Series: {series_name}, Value (normalized)')
        plt.xlim((date_min, date_max))
        plt.ylim((0, (1 + self.y_margin) * val_max))

        plt.tight_layout()
        plt.show()

    def plot_all_series_for_window(self, *,
                                   data_manager: DataManager,
                                   window_start: str) -> None:
        plt.clf()

        fig, ax = plt.subplots()

        df = data_manager.data_split_norm[window_start]

        for series in df.columns[1:]:
            df.plot(ax=ax,
                    x='x',
                    y=series)

        legend = ax.get_legend()
        legend.remove()

        plt.xlabel('Time')
        plt.ylabel(f'Window Start: {window_start}, Value (normalized)')
        plt.xlim((df['x'].min(), df['x'].max()))
        plt.ylim((0, (1 + self.y_margin) *
                  df[df.columns[1:]].max().max()))

        plt.tight_layout()
        plt.show()

    def plot_clustering_results(self, *,
                                method: str,
                                results: dict,
                                window_start: str,
                                window: pd.DataFrame,
                                c: int) -> None:
        plt.clf()

        n_rows = ceil(c ** (1 / 2))
        n_cols = ceil(c / n_rows)

        fig, axes = plt.subplots(n_rows, n_cols)

        k = 0

        for i, row in enumerate(axes):
            if c > 2:
                for j, col in enumerate(row):
                    try:
                        self.plot_cluster(ax=axes[i, j],
                                          window=window,
                                          results=results,
                                          cluster_n=k)
                        k += 1

                    except IndexError:
                        axes[i, j].set_axis_off()
            else:
                self.plot_cluster(ax=axes[i],
                                  window=window,
                                  results=results,
                                  cluster_n=k)
                k += 1

        plt.suptitle(f'{method}: {window_start=}, n_clusters={c}, silhouette={results["silhouette"]:.4f}')

        plt.tight_layout()

        plt.show()

    def plot_cluster(self, *,
                     ax,
                     window: pd.DataFrame,
                     results: dict,
                     cluster_n: int,
                     color_members: str = 'blue',
                     color_center: str = 'red') -> None:
        memberships = results['memberships']
        cluster_center = results['cluster_centers'][cluster_n, :]

        for col, membership in zip(window.columns[1:], memberships):
            if membership == cluster_n:
                ax.plot(window[col].to_list(), color=color_members, alpha=0.3)

        ax.plot(cluster_center, color=color_center)

        ax.set_xlim((0, len(window) - 1))
        ax.set_ylim((0, (1 + self.y_margin) *
                     window[window.columns[1:]].max().max()))

        ax.set_title(f'c={cluster_n}, n_members={memberships.tolist().count(cluster_n)}')

    def plot_largest_cluster_curve(self, *,
                                   method: str,
                                   df: pd.DataFrame) -> None:
        plt.clf()

        plt.plot(df['window_start'], df['largest_cluster_size'])

        plt.title(method)
        plt.xlabel('Time')
        plt.ylabel('Largest Cluster Size')
        plt.xlim((df['window_start'].min(), df['window_start'].max()))
        plt.ylim((df['largest_cluster_size'].min(),
                  (1 + self.y_margin) * df['largest_cluster_size'].max()))

        plt.tight_layout()
        plt.show()

    def plot_rand_curve(self, *,
                        method: str,
                        df: pd.DataFrame) -> None:
        plt.clf()

        plt.plot(df['window_start'], df['rand_score'], label='rand_score')
        plt.plot(df['window_start'], df['adjusted_rand_score'], label='adjusted_rand_score')

        plt.title(method)
        plt.xlabel('Time')
        plt.ylabel('Rand Score')
        plt.xlim((df['window_start'].min(), df['window_start'].max()))
        ymin = df[df.columns[1:]].min().min()
        plt.ylim(((1 - self.y_margin) * ymin if ymin > 0 else (1 + self.y_margin) * ymin,
                  (1 + self.y_margin) * df[df.columns[1:]].max().max()))
        plt.legend()

        plt.tight_layout()
        plt.show()
