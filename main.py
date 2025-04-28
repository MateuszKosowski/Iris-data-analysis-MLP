from mlp import MLP
# import pandas as pd
# import numpy as np


def main():
    # Wczytaj plik csv
    # file_test = './data/data3_test.csv'
    # file_train = './data/data3_train.csv'
    #
    # # Dane do trenowania
    # data_train = pd.read_csv(file_train, header=None, sep=',')
    # data_train.columns = ['Dlugosc kielicha', 'Szerokosc kielicha', 'Dlugosc platka', 'Szerokosc platka', 'Gatunek']
    #
    # # Dane do testowania
    # data_test = pd.read_csv(file_test, header=None, sep=',')
    # data_test.columns = ['Dlugosc kielicha', 'Szerokosc kielicha', 'Dlugosc platka', 'Szerokosc platka', 'Gatunek']

    print("--- Testowanie Inicjalizacji i Propagacji w Przód MLP ---")
    simple_structure = [2, 3, 1]
    print(f"  Tworzenie prostej sieci o strukturze: {simple_structure}")
    simple_mlp = MLP(layer_sizes_array=simple_structure, use_bias=True)

    simple_input = [0.6, 0.1]
    print(f"  Testowanie predict dla wejścia: {simple_input}")
    simple_output = simple_mlp.forward_pass(simple_input)

    print(f"  Wyjście prostej sieci (przed treningiem): {simple_output}")

if __name__ == "__main__":
    main()