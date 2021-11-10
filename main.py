"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from src.dataframes import load_data

if __name__ == '__main__':
    data = load_data('data/res_2000.csv')

    print(data)
