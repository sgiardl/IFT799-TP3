"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import skfuzzy


class FCM:
    def __init__(self) -> None:
        self.n_clusters_min = 2
        self.n_clusters_max = 10
