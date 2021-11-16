"""
IFT799 - Science des données
TP3
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from src.dataframes import load_data, split_data

if __name__ == '__main__':
    data = load_data('data/res_2000.csv')

    data_split = split_data(data, 21)

    print(data_split)
