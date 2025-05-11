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

    # Train labels jako wektory
    train_labels_vector = np.zeros((len(train_labels), num_output_classes))
    for j, label in enumerate(train_labels):
        train_labels_vector[j, label] = 1

    num_epochs = 1000  # Zdefiniuj liczbę epok treningu
    num_samples = len(train_features)

    print(f"\n--- Rozpoczęcie Treningu ({num_epochs} epok) ---")

    for epoch in range(num_epochs):

        permutation = np.random.permutation(num_samples) # Losowe przetasowanie danych
        shuffled_train_features = train_features[permutation] # Przetasowanie cech
        shuffled_train_labels_vector = train_labels_vector[permutation] # Przetasowanie etykiet

        total_epoch_error = 0.0

        for i in range(num_samples):
            input_sample = shuffled_train_features[i]
            target_label_vector = shuffled_train_labels_vector[i]

            # Krok 1: Przetwarzanie danych przez sieć (forward pass)
            current_outputs = mlp_network.forward_pass(input_sample)

            # Krok 2: Obliczanie błędów i aktualizacja wag (backward pass)
            mlp_network.backward_pass(target_label_vector)

            #Oblicz błąd dla tej próbki i dodaj do błędu epoki
            error_for_sample = mlp_network.calculate_mse(current_outputs, target_label_vector)
            total_epoch_error += error_for_sample

        #Wyświetl średni błąd dla epoki
        average_epoch_error = total_epoch_error / num_samples
        if (epoch + 1) % 10 == 0:  # Wyświetl co 10 epok
            print(
                f"Epoka {epoch + 1}/{num_epochs} zakończona, MSE: {average_epoch_error:.6f}")

    print("\n--- Zakończono Trening ---")

    test_labels_vector = np.zeros((len(test_labels), num_output_classes))
    for j, label in enumerate(test_labels):
        test_labels_vector[j, label] = 1

    for i in range(45):
        input_sample = test_features[i]
        original_label = test_labels_vector[i]

        print("--------------------------------------")
        print(f"  Próbka testowa {i + 1}:")
        print(f"    Wejście : {np.round(input_sample, 2)}")
        print(f"    Cel (prawidołowa klasa): {original_label}")
        print(f"    Wyjście : {np.round(mlp_network.forward_pass(input_sample), 4)}")

if __name__ == "__main__":
    main()