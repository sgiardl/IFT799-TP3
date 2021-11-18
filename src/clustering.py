"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""


from src.constants import max_num_clusters
from fcmeans import FCM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def fit_clusters(data, cluster_algo, nb_clusters=None):
    if nb_clusters is not None:
        return find_num_clusters(data, cluster_algo, nb_clusters)
    else:
        # Define parameters to find the good number of clusters
        max_silhouette = -1  # Ranges between -1 and 1
        best_cluster = None
        possible_cluster = None

        for iter_num_clusters in range(2, max_num_clusters):
            possible_cluster = find_num_clusters(data, cluster_algo, iter_num_clusters)
            if possible_cluster['silhouette'] > max_silhouette:
                max_silhouette = possible_cluster['silhouette']
                best_cluster = possible_cluster
        return best_cluster

def find_num_clusters(data, cluster_algo, nb_clusters, metric="euclidean"):
    if cluster_algo == 'FCM':
        data = data.to_numpy()  # FCM algorithm takes numpy arrays as input
        clusterer = FCM(n_clusters=nb_clusters)
        clusterer.fit(data)

    elif cluster_algo == 'KMeans':
        clusterer = KMeans(n_clusters=nb_clusters).fit(data)

    # outputs
    clusterer_labels = clusterer.predict(data)

    # silhouette
    silhouette_avg = silhouette_score(data, clusterer_labels, metric=metric)

    # results
    results = {
        'labels': clusterer_labels,
        'silhouette': silhouette_avg,
        'nb_clusters': nb_clusters
    }
    return results
