from mlp import MLP
from menu import get_network_config_from_user
import pandas as pd
import numpy as np


def main():
    #Wczytaj plik csv
    file_test = './data/data3_test.csv'
    file_train = './data/data3_train.csv'

    # Dane do trenowania
    data_train = pd.read_csv(file_train, header=None, sep=',')
    data_train.columns = ['Dlugosc kielicha', 'Szerokosc kielicha', 'Dlugosc platka', 'Szerokosc platka', 'Gatunek']
    train_features = data_train[['Dlugosc kielicha', 'Szerokosc kielicha', 'Dlugosc platka', 'Szerokosc platka']].values
    train_labels = data_train['Gatunek'].values

    # Dane do testowania
    data_test = pd.read_csv(file_test, header=None, sep=',')
    data_test.columns = ['Dlugosc kielicha', 'Szerokosc kielicha', 'Dlugosc platka', 'Szerokosc platka', 'Gatunek']
    test_features = data_test[['Dlugosc kielicha', 'Szerokosc kielicha', 'Dlugosc platka', 'Szerokosc platka']].values
    test_labels = data_test['Gatunek'].values

    # Rozmiar zbioru treningowego
    num_input_features = train_features.shape[1]
    num_output_classes = len(np.unique(train_labels))
    print(f"\nParametry sieci ustalone na podstawie danych:")
    print(f"  Liczba cech wejściowych: {num_input_features}")
    print(f"  Liczba klas wyjściowych: {num_output_classes}")

    # Parametry sieci od usera
    architecture, bias_flag = get_network_config_from_user(num_input_features, num_output_classes)

    print("\n--- Tworzenie Sieci ---")
    mlp_network = MLP(layer_sizes_array=architecture, use_bias=bias_flag)

    print(f"\nPrzetwarzanie danych testowych ({len(test_features)} próbek) przez nienauczoną sieć...")

    test_results = [] # Lista do przechowywania wyników

    for i in range(len(test_features)):
        input_sample = test_features[i]
        original_label = test_labels[i]

        output = mlp_network.forward_pass(input_sample)

        test_results.append(output)

        if i < 8:
            print("--------------------------------------")
            print(f"  Próbka testowa {i + 1}:")
            print(f"    Wejście : {np.round(input_sample, 2)}")
            print(f"    Cel (prawidołowa klasa): {original_label}")
            print(f"    Wyjście : {np.round(output, 4)}")  # Wyniki przed treningiem
            print(f"    Klasa wyjściowa: {np.argmax(output)}")  # Klasa z najwyższym prawdopodobieństwem

    print(f"\nPrzetworzono wszystkie {len(test_features)} próbek testowych.")


if __name__ == "__main__":
    main()