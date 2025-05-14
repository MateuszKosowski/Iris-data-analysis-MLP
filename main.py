from autoencoder import create_autoencoder_mlp, train_autoencoder, get_hidden_layer_outputs
from mlp import MLP
from menu import get_network_config_from_user, mode_menu, save_mlp_to_file, load_mlp_from_file, how_much_echos, \
    shuffle_data_menu, use_momentum_menu, epochs_or_precision, give_precision
from learn import learn
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
    num_input_features = train_features.shape[1] # Ile jest cech
    num_output_classes = len(np.unique(train_labels)) # Ile jest klas irysów
    print(f"\nParametry sieci ustalone na podstawie danych:")
    print(f"  Liczba cech wejściowych: {num_input_features}")
    print(f"  Liczba klas wyjściowych: {num_output_classes}")

    # Train labels jako wektory
    train_labels_vector = np.zeros((len(train_labels), num_output_classes))
    for j, label in enumerate(train_labels):
        train_labels_vector[j, label] = 1

    # Test label jako wektory
    test_labels_one_hot = np.zeros((len(test_labels), num_output_classes))
    for j, label_val in enumerate(test_labels):
        test_labels_one_hot[j, int(label_val)] = 1

    # Flagi
    mlp_network = None
    is_mlp_created = False


    # ------ Główna pętla ------
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
                    precision = 0.0
                    num_epochs = how_much_echos()
                elif choice == "precision":
                    precision = give_precision()
                    num_epochs = 1000

                num_samples = len(train_features)
                shuffle_data_flag = shuffle_data_menu()
                use_momentum_flag = use_momentum_menu()
                learn(choice, num_epochs, precision, shuffle_data_flag, use_momentum_flag, mlp_network, train_features, train_labels_vector, num_samples)

            else:
                print("Sieć nie została stworzona!")


        # Tryb testowy
        elif mode == "test":
            if is_mlp_created:

                # Inicjalizacja macierzy pomyłek
                confusion_matrix = np.zeros((num_output_classes, num_output_classes), dtype=int) # Tablica 3x3

                print("\n--- Rozpoczęcie Testowania ---")

                # Nazwa pliku logu
                log_filename = "test_details_log.csv"  #

                try:
                    with open(log_filename, "w") as log_file:
                        # --- Przygotowanie Nagłówka dla pliku CSV ---
                        header_parts = [
                            "Wzorzec_Wejsciowy", # wzorca wejściowego
                            "Blad_Calkowity_MSE_Wzorca", # popełnionego przez sieć błędu dla całego wzorca
                            "Pozadany_Wzorzec_Odpowiedzi", # pożądanego wzorca odpowiedzi
                            "Bledy_Na_Wyjsciach_Ostatniej_Warstwy",  # błędów popełnionych na poszczególnych wyjściach sieci
                            "Wartosci_Wyjsciowe_Neuronow_Wyjsciowych" # wartości wyjściowych neuronów wyjściowych
                        ]

                        # Nagłówki dla wag i biasów neuronów wyjściowych
                        output_layer_obj = mlp_network.layers[-1]
                        output_layer_number = len(mlp_network.layers)
                        for neuron_idx in range(output_layer_obj.num_neurons):
                            for weight_idx in range(output_layer_obj.num_input_per_neuron):
                                header_parts.append(f"Waga_Warstwa_{output_layer_number}_Neuron_{neuron_idx}_Polaczenie_{weight_idx}") # wag neuronów wyjściowych - POŁĄCZENIA
                                if mlp_network.use_bias:
                                    header_parts.append(f"Bias_Warstwa_{output_layer_number}_Neuron_{neuron_idx}") # wag neuronów wyjściowych - BIAS

                        # Nagłówki dla wyjść, wag i biasów neuronów ukrytych
                        # Kolejność: od warstw dalszych (bliżej wyjścia) do bliższych wejścia
                        for layer_idx in range(len(mlp_network.layers) - 2, -1, -1):  # Od ostatniej ukrytej do pierwszej ukrytej - argumenty: start, stop, step
                            hidden_layer_obj = mlp_network.layers[layer_idx]
                            # Wyjścia neuronów ukrytych
                            header_parts.append(f"Wartosci_Wyjsciowe_Warstwy_Ukrytej_{len(mlp_network.layers) - 1 - layer_idx}")  # wartości wyjściowych neuronów ukrytych - od 1
                            # Wagi i biasy neuronów ukrytych
                            for n_idx in range(hidden_layer_obj.num_neurons):
                                for w_idx in range(hidden_layer_obj.num_input_per_neuron):
                                    header_parts.append(f"Waga_Warstywy_Ukrytej_{len(mlp_network.layers) - 1 - layer_idx}_Neuron_{n_idx}_Polaczenie_{w_idx}")
                                if mlp_network.use_bias:
                                    header_parts.append(f"Bias_Warstywy_Ukrytej_{len(mlp_network.layers) - 1 - layer_idx}_Neuron_{n_idx}")

                        log_file.write("\t".join(header_parts) + "\n")  # Użyj tab jako separatora dla CSV

                        # --- Pętla po próbkach testowych ---
                        for i in range(len(test_features)):
                            input_sample = test_features[i]
                            true_label_numeric = int(test_labels[i])
                            target_one_hot_vector = test_labels_one_hot[i] # zmienna

                            # KROK 1: forward pass
                            output_probabilities = mlp_network.forward_pass(input_sample)
                            predicted_label_numeric = np.argmax(output_probabilities) # Największa wartość z listy

                            # Aktualizacja macierzy pomyłek
                            confusion_matrix[true_label_numeric, predicted_label_numeric] += 1 # Wiesze to prawdziwy gatunek, kolumny to predykcja

                            # --- Zbieranie danych do logowania dla bieżącej próbki ---

                            # 1. Wzorzec wejściowy
                            log_data_for_sample = [str(np.round(input_sample, 4).tolist())]

                            # 2. Popełniony przez sieć błąd dla całego wzorca (MSE)
                            mse_for_sample = mlp_network.calculate_mse(output_probabilities, target_one_hot_vector)
                            log_data_for_sample.append(f"{mse_for_sample:.6f}")

                            # 3. Pożądany wzorzec odpowiedzi (one-hot)
                            log_data_for_sample.append(str(target_one_hot_vector.tolist()))

                            # 4. Błędy popełnione na poszczególnych wyjściach sieci (rzeczywiste minus cel)
                            error_vector = output_probabilities - target_one_hot_vector
                            log_data_for_sample.append(str(np.round(error_vector, 6).tolist()))

                            # 5. Wartości wyjściowe neuronów wyjściowych
                            log_data_for_sample.append(str(np.round(output_probabilities, 6).tolist()))


                            # 6. Wagi i biasy neuronów wyjściowych
                            for neuron in output_layer_obj.neurons:
                                # Jeśli test jest po treningu, to są to wagi po ostatniej aktualizacji.
                                log_data_for_sample.append(str(np.round(neuron.weights, 6).tolist()))
                                if mlp_network.use_bias:
                                    log_data_for_sample.append(f"{neuron.bias:.6f}")

                            # 7. Wartości wyjściowe neuronów ukrytych
                            # 8. Wagi neuronów ukrytych (połączenia + bias)
                            # Kolejność: od warstw dalszych (bliżej wyjścia) do bliższych wejścia
                            for layer_idx in range(len(mlp_network.layers) - 2, -1, -1):
                                hidden_layer_obj = mlp_network.layers[layer_idx]

                                # 7. Wartości wyjściowe neuronów ukrytych (aktywacje tej warstwy)
                                # .outputs jest ustawiane w Layer.calculate_outputs(), które jest częścią MLP.forward_pass()
                                log_data_for_sample.append(str(np.round(hidden_layer_obj.outputs, 6).tolist()))

                                # 8. Wagi i biasy neuronów ukrytych tej warstwy
                                for neuron in hidden_layer_obj.neurons:
                                    log_data_for_sample.append(str(np.round(neuron.weights, 6).tolist()))
                                    if mlp_network.use_bias:
                                        log_data_for_sample.append(f"{neuron.bias:.6f}")

                            # Zapisz wiersz danych do pliku CSV
                            log_file.write("\t".join(log_data_for_sample) + "\n")

                            # Pokaż kilka pierwszych/ostatnich
                            if i < 5 or i >= len(test_features) - 5:
                                output_rounded = np.round(output_probabilities, 3)
                                print("--------------------------------------")
                                print(f"  Próbka testowa {i + 1}:")
                                print(f"    Wejście : {np.round(input_sample, 2)}")
                                print(f"    Cel (one-hot): {target_one_hot_vector}")
                                print(f"    Cel (numeryczna): {true_label_numeric}")
                                print(f"    Wyjście (prawdopodobieństwa): {output_rounded}")
                                print(f"    Przewidziana klasa (numeryczna): {predicted_label_numeric}")
                                print(f"    MSE próbki: {mse_for_sample:.6f}")

                    print(f"\nSzczegółowe wyniki testowania zapisano do pliku: {log_filename}")
                    print("--- Zakończono Testowanie ---")

                except IOError:
                    print(f"BŁĄD: Nie można zapisać do pliku logu '{log_filename}'. Sprawdź uprawnienia.")
                except Exception as e:
                    print(f"Wystąpił nieoczekiwany błąd podczas testowania i logowania: {e}")
                    import traceback
                    traceback.print_exc()


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
                    tp = confusion_matrix[class_idx, class_idx]  # True Positives
                    fp = np.sum(confusion_matrix[:, class_idx]) - tp  # False Positives
                    fn = np.sum(confusion_matrix[class_idx, :]) - tp  # False Negatives

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                    print(f"  Klasa {class_idx}:")
                    print(f"    Precision: {precision:.4f}")
                    print(f"    Recall:    {recall:.4f}")
                    print(f"    F-measure: {f_measure:.4f}")

            else:
                print("Sieć nie została stworzona!")


        elif mode == "save":
            if is_mlp_created:
                save_mlp_to_file(mlp_network)
            else:
                print("Sieć nie istnieje!")


        elif mode == "load":
            loaded_mlp = load_mlp_from_file()
            if loaded_mlp:
                mlp_network = loaded_mlp
                is_mlp_created = True


        elif mode == "autoencoder":

            print(f"\n---- Nauka z Bias ----")
            mlp_network = create_autoencoder_mlp()
            train_autoencoder(mlp_network, 300, 0.6, False, 0, "mse_log_autoencoder_bias.txt")
            hidde_layer_outputs = get_hidden_layer_outputs(mlp_network)
            for i in range(len(hidde_layer_outputs)):
                print(f"Wartość wyjściowa neuronów ukrytych, po nauce, dla paternu {i + 1}: {hidde_layer_outputs[i]}")

            print(f"\n---- Nauka bez Bias ----")
            mlp_network = create_autoencoder_mlp(False)
            train_autoencoder(mlp_network, 300, 0.6, False, 0, "mse_log_autoencoder_without_bias.txt")
            hidde_layer_outputs = get_hidden_layer_outputs(mlp_network)
            for i in range(len(hidde_layer_outputs)):
                print(f"Wartość wyjściowa neuronów ukrytych, po nauce, dla paternu {i + 1}: {hidde_layer_outputs[i]}")

            print(f"\n---- Szybkość nauki (ile epok do MSE < 0.02) ----")



        elif mode == "exit":
            print("\n--- Zakończenie ---")
            break

if __name__ == "__main__":
    main()