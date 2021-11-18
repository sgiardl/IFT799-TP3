"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from src.clustering import fit_clusters
from src.DataManager import load_data, split_data
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # Get complete dataset
    data = load_data('data/res_2000.csv')

    # Get data splits
    windows_datasets = split_data(data, window_size=20, jump_size=10)

    list_date = []
    list_km_max = []
    list_km_nb = []

    results = pd.DataFrame(columns={'Date',
                                 'kmeans-n_cluster',
                                 'fcm-n_cluster'})

    for windows_data in windows_datasets:

        proccessed_data = windows_data[windows_data.columns[1:]]

        # clustering with kmeans
        km_results = fit_clusters(proccessed_data.T, cluster_algo='KMeans')
        km_labels = km_results['labels']
        km_nb_max = np.bincount(km_labels).max()
        # km_results['nb_clusters']

        # clustering with FCmeans
        fcm_results = fit_clusters(proccessed_data.T, cluster_algo='FCM')
        fcm_labels = fcm_results['labels']
        fcm_nb_max = np.bincount(fcm_labels).max()

        results.loc[len(results.index)] = [windows_data['x'].min(), km_nb_max, fcm_nb_max]

    results['Date'] = pd.to_datetime(results['Date'])

    """import matplotlib.pyplot as plt
    plt.plot(list_date, list_km_max)
    #plt.plot(list_date, list_km_nb)
    plt.xticks(rotation=90)
    plt.show()"""
