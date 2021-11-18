
import numpy as np
from src.clustering import fit_clusters

from fcmeans import FCM
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score


n_samples = 3000

X = np.concatenate((
    np.random.normal((-2, -2), size=(n_samples, 2)),
    np.random.normal((2, 2), size=(n_samples, 2))
))
test = fit_clusters(X)
"""
for nb_cluster in range(2, 7):
    fcm = FCM(n_clusters=nb_cluster)
    fcm.fit(X)

    # outputs
    fcm_centers = fcm.centers
    fcm_labels = fcm.predict(X)

    # silhouette
    silhouette_avg = silhouette_score(X, fcm_labels, metric="euclidean")
    print(f"For {nb_cluster} clusters, score = {silhouette_avg}")

    # plot result
    f, axes = plt.subplots(1, 2, figsize=(11,5))
    axes[0].scatter(X[:,0], X[:,1], alpha=.1)
    axes[1].scatter(X[:,0], X[:,1], c=fcm_labels, alpha=.1)
    axes[1].scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='w')

plt.show()
"""