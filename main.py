from pandas.io.stata import precision_loss_doc

from mlp import MLP
from menu import get_network_config_from_user, mode_menu, save_mlp_to_file, load_mlp_from_file, how_much_echos, \
    shuffle_data_menu, use_momentum_menu, epochs_or_precision, give_precision
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

    mlp_network = None
    confusion_matrix = None

    is_mlp_trained = False
    is_mlp_created = False

    while True:

        mode = mode_menu()

        if mode == "create":
            # Parametry sieci od usera
            architecture, bias_flag = get_network_config_from_user(num_input_features, num_output_classes)
            print("\n--- Tworzenie Sieci ---")
            mlp_network = MLP(layer_sizes_array=architecture, use_bias=bias_flag)
            is_mlp_created = True

        # Tryb działania sieci
        elif mode == "learn":
            if is_mlp_created:

                choice = epochs_or_precision()
                if choice == "epoch":
                    num_epochs = how_much_echos()
                elif choice == "precision":
                    precision = give_precision()
                    num_epochs = 1000

                num_samples = len(train_features)
                shuffle_data = shuffle_data_menu()
                use_momentum = use_momentum_menu()




                # Train labels jako wektory
                train_labels_vector = np.zeros((len(train_labels),
                                                num_output_classes))  # Tworzona jest nowa tablica z zerami o wymiarach (len(train_labels), num_output_classes) - 105x3
                for j, label in enumerate(train_labels):  # iteracja dla każdej etykiety w train_labels
                    train_labels_vector[j, label] = 1  # Ustawienie wartości 1 w odpowiedniej kolumnie dla danej etykiety

                if choice == "epoch":

                    print(f"\n--- Rozpoczęcie Treningu ({num_epochs} epok) ---")

                    for epoch in range(num_epochs):

                        if shuffle_data:
                            permutation = np.random.permutation(num_samples)  # Zwraca losową permutację indeksów
                            shuffled_train_features = train_features[permutation]  # Nowa tablica z przetasowanymi danymi
                            shuffled_train_labels_vector = train_labels_vector[permutation]  # Przetasowanie etykiet w ten sam sposób aby nie stracić informacji o etykietach

                        total_epoch_error = 0.0  # Błąd epoki

                        for i in range(num_samples):

                            if shuffle_data:
                                input_sample = shuffled_train_features[i]  # Próbka wejściowa
                                target_label_vector = shuffled_train_labels_vector[i]  # Wektor etykiet docelowych
                            else:
                                input_sample = train_features[i]
                                target_label_vector = train_labels_vector[i]

                            # Krok 1: Przetwarzanie danych przez sieć (forward pass)
                            current_outputs = mlp_network.forward_pass(input_sample)

                            # Krok 2: Obliczanie błędów i aktualizacja wag (backward pass)
                            mlp_network.backward_pass(target_label_vector, use_momentum)

                            # Oblicz błąd dla tej próbki i dodaj do błędu epoki
                            error_for_sample = mlp_network.calculate_mse(current_outputs, target_label_vector)
                            total_epoch_error += error_for_sample

                        # Wyświetl średni błąd dla epoki
                        average_epoch_error = total_epoch_error / num_samples

                        if (epoch + 1) % 10 == 0:  # Wyświetl co 10 epok
                            print(f"Epoka {epoch + 1}/{num_epochs} zakończona, MSE: {average_epoch_error:.6f}")
                            with open("mse_log.txt", "a") as f:
                                f.write(f"{epoch + 1},{average_epoch_error:.6f}\n")

                elif choice == "precision":
                    print(f"\n--- Rozpoczęcie Treningu ({precision} MSE) ---")

                    average_epoch_error = 2.0
                    counter = 0
                    while  average_epoch_error > precision and counter < num_epochs:
                        total_epoch_error = 0.0
                        counter += 1
                        if shuffle_data:
                            permutation = np.random.permutation(num_samples)  # Zwraca losową permutację indeksów
                            shuffled_train_features = train_features[
                                permutation]  # Nowa tablica z przetasowanymi danymi
                            shuffled_train_labels_vector = train_labels_vector[
                                permutation]  # Przetasowanie etykiet w ten sam sposób aby nie stracić informacji o etykietach

                        for i in range(num_samples):
                            if shuffle_data:
                                input_sample = shuffled_train_features[i]  # Próbka wejściowa
                                target_label_vector = shuffled_train_labels_vector[i]  # Wektor etykiet docelowych
                            else:
                                input_sample = train_features[i]
                                target_label_vector = train_labels_vector[i]

                            # Krok 1: Przetwarzanie danych przez sieć (forward pass)
                            current_outputs = mlp_network.forward_pass(input_sample)

                            # Krok 2: Obliczanie błędów i aktualizacja wag (backward pass)
                            mlp_network.backward_pass(target_label_vector, use_momentum)

                            # Oblicz błąd dla tej próbki i dodaj do błędu epoki
                            error_for_sample = mlp_network.calculate_mse(current_outputs, target_label_vector)
                            total_epoch_error += error_for_sample

                            # Wyświetl średni błąd dla epoki
                        average_epoch_error = total_epoch_error / num_samples

                        if (counter + 1) % 10 == 0:  # Wyświetl co 10 epok
                            print(f"Epoka {counter + 1}/{num_epochs} zakończona, MSE: {average_epoch_error:.6f}")
                            with open("mse_log.txt", "a") as f:
                                f.write(f"{counter + 1},{average_epoch_error:.6f}\n")


                is_mlp_trained = True
                print("\n--- Zakończono Trening ---")
            else:
                print("Sieć nie została stworzona!")

        # Tryb testowy
        elif mode == "test":
            if is_mlp_created:
                # Inicjalizacja macierzy pomyłek
                confusion_matrix = np.zeros((num_output_classes, num_output_classes), dtype=int)

                test_labels_vector = np.zeros((len(test_labels), num_output_classes))
                for j, label in enumerate(test_labels):
                    test_labels_vector[j, label] = 1

                print("\n--- Rozpoczęcie Testowania ---")
                with open("test_log.txt", "w") as log_file:
                    log_file.write("Wejscie;Blad calkowity;Wzorzec oczekiwany;Bledy na wyjsciach;Wyjscia neuronow wyjsciowych;Wagi neuronow wyjsciowych;Wyjscia neuronow ukrytych;Wagi neuronow ukrytych\n")
                    for i in range(len(test_features)):
                        input_sample = test_features[i]
                        true_label_numeric = test_labels[i]
                        original_label_vector = test_labels_vector[i]  # Wektor one-hot do wyświetlania

                        # Forward pass z zapisem wyjść warstw
                        hidden_outputs = []
                        current_outputs = input_sample
                        for layer in mlp_network.layers[:-1]:
                            layer_outputs = np.array([neuron.calculate_output(current_outputs) for neuron in layer.neurons])
                            hidden_outputs.append(layer_outputs)
                            current_outputs = layer_outputs
                        # Warstwa wyjściowa
                        output_layer = mlp_network.layers[-1]
                        output_probabilities = np.array([neuron.calculate_output(current_outputs) for neuron in output_layer.neurons])
                        predicted_label_numeric = np.argmax(output_probabilities)

                        # Aktualizacja macierzy pomyłek
                        confusion_matrix[true_label_numeric, predicted_label_numeric] += 1

                        # Obliczenia błędów
                        error_vector = output_probabilities - original_label_vector
                        total_error = np.mean((output_probabilities - original_label_vector) ** 2)

                        # Wagi neuronów wyjściowych
                        output_weights = [neuron.weights.tolist() for neuron in output_layer.neurons]
                        # Wagi neuronów ukrytych (od ostatniej do pierwszej warstwy ukrytej)
                        hidden_weights = []
                        for layer in reversed(mlp_network.layers[:-1]):
                            hidden_weights.append([neuron.weights.tolist() for neuron in layer.neurons])

                        # Logowanie do pliku
                        log_file.write(
                            f"{np.round(input_sample, 3).tolist()};"
                            f"{total_error:.6f};"
                            f"{original_label_vector.tolist()};"
                            f"{np.round(error_vector, 6).tolist()};"
                            f"{np.round(output_probabilities, 6).tolist()};"
                            f"{output_weights};"
                            f"{[np.round(h, 6).tolist() for h in hidden_outputs]};"
                            f"{hidden_weights}\n"
                        )

                        output_rounded = np.round(output_probabilities, 3)
                        print("--------------------------------------")
                        print(f"  Próbka testowa {i + 1}:")
                        print(f"    Wejście : {np.round(input_sample, 2)}")
                        print(f"    Cel (prawidłowa klasa - one-hot): {original_label_vector}")
                        print(f"    Cel (prawidłowa klasa - numeryczna): {true_label_numeric}")
                        print(f"    Wyjście (prawdopodobieństwa): {output_rounded}")
                        print(f"    Przewidziana klasa (numeryczna): {predicted_label_numeric}")

                print("\n--- Zakończono Testowanie ---")

                # Obliczanie i wyświetlanie statystyk
                print("\n--- Wyniki Testowania ---")

                # Liczba poprawnie sklasyfikowanych obiektów
                correctly_classified_total = np.trace(confusion_matrix)
                print(
                    f"Łączna liczba poprawnie sklasyfikowanych obiektów: {correctly_classified_total} z {len(test_features)}")

                print("\nLiczba poprawnie sklasyfikowanych obiektów w rozbiciu na klasy:")
                for class_idx in range(num_output_classes):
                    correctly_classified_class = confusion_matrix[class_idx, class_idx]
                    total_in_class = np.sum(confusion_matrix[class_idx, :])
                    print(f"  Klasa {class_idx}: {correctly_classified_class} z {total_in_class}")

                # Macierz pomyłek
                print("\nMacierz pomyłek (wiersze: prawdziwe klasy, kolumny: przewidziane klasy):")
                print(confusion_matrix)

                # Precision, Recall, F-measure
                print("\nMetryki oceny (Precision, Recall, F-measure) dla każdej klasy:")
                for class_idx in range(num_output_classes):
                    TP = confusion_matrix[class_idx, class_idx]  # True Positives
                    FP = np.sum(confusion_matrix[:, class_idx]) - TP  # False Positives
                    FN = np.sum(confusion_matrix[class_idx, :]) - TP  # False Negatives

                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                    f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                    print(f"  Klasa {class_idx}:")
                    print(f"    Precision: {precision:.4f}")
                    print(f"    Recall:    {recall:.4f}")
                    print(f"    F-measure: {f_measure:.4f}")

            else:
                print("Sieć nie została stworzona!")

        elif mode == "save":
            if is_mlp_created and is_mlp_trained:
                save_mlp_to_file(mlp_network)
            else:
                print("Sieć nie istnieje lub nie została stworzona!")

        elif mode == "load":
            loaded_mlp = load_mlp_from_file()
            if loaded_mlp:
                mlp_network = loaded_mlp
                is_mlp_created = True
                is_mlp_trained = True

        elif mode == "exit":
            print("\n--- Zakończenie ---")
            break

if __name__ == "__main__":
    main()