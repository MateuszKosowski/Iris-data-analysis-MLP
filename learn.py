import numpy as np


def learn(choice, num_epochs, precision, shuffle_data, use_momentum, mlp_network, train_features, train_labels_vector, num_samples = None, filename='mse_log_train.txt'):

    #Czyszczenie pliku do zapisu
    with open(filename, "w") as f:
        f.write("")

    # Przygotowanie danych
    if shuffle_data:
        permutation = np.random.permutation(num_samples)
        train_features_to_use = train_features[permutation]
        train_label_vector_to_use = train_labels_vector[permutation]
    else:
        train_features_to_use = train_features
        train_label_vector_to_use = train_labels_vector

    if choice == "epoch":
        print(f"\n--- Rozpoczęcie Treningu ({num_epochs} epok) ---")
        for epoch in range(num_epochs):
            average_epoch_error = learn_one_epoch(num_samples, num_epochs, use_momentum, mlp_network, train_features_to_use, train_label_vector_to_use, epoch)

        # Zapisz ostatnią epokę - jeśli nie była już zapisana
        if num_epochs % 10 != 0:
            print(f"Epoka {num_epochs}/{num_epochs} zakończona, MSE: {average_epoch_error:.6f} (koniec treningu)")
            with open(filename, "a") as f:
                f.write(f"{num_epochs},{average_epoch_error:.6f}\n")

    elif choice == "precision":
        print(f"\n--- Rozpoczęcie Treningu ({precision} MSE) ---")
        average_epoch_error = 2.0
        epoch = 0
        while average_epoch_error > precision and epoch < num_epochs:
            epoch += 1 # counter do liczenia która epoka
            average_epoch_error = learn_one_epoch(num_samples, num_epochs, use_momentum, mlp_network, train_features_to_use, train_label_vector_to_use, epoch, filename=filename)

        # Zapisz ostatnią epokę - jeśli nie była już zapisana
        if (epoch + 1) % 10 != 0:
            print(f"Epoka {epoch + 1}/{num_epochs} zakończona, MSE: {average_epoch_error:.6f} (koniec treningu)")
            with open(filename, "a") as f:
                f.write(f"{epoch + 1},{average_epoch_error:.6f}\n")


    print("\n--- Zakończono Trening ---")



def learn_one_epoch(num_samples, num_epochs, use_momentum, mlp_network, train_features_to_use, train_label_vector_to_use, epoch = 0, filename='mse_log_train.txt'):
    total_epoch_error = 0.0

    for i in range(num_samples):
        # Cechy wejściowe, Wynik
        input_sample = train_features_to_use[i]
        target_label_vector = train_label_vector_to_use[i]

        # Krok 1: Przetwarzanie danych przez sieć (forward pass)
        current_outputs = mlp_network.forward_pass(input_sample)

        # Krok 2: Obliczanie błędów i aktualizacja wag (backward pass)
        mlp_network.backward_pass(target_label_vector, use_momentum)

        # Oblicz błąd dla tej próbki i dodaj do błędu epoki
        error_for_sample = mlp_network.calculate_mse(current_outputs, target_label_vector)
        total_epoch_error += error_for_sample

    # Wyświetl średni błąd dla epoki
    average_epoch_error = total_epoch_error / num_samples

    # Zapis co 10 epoka: jaki mse
    if (epoch + 1) % 10 == 0:
        print(f"Epoka {epoch + 1}/{num_epochs} zakończona, MSE: {average_epoch_error:.6f}")
        with open(filename, "a") as f:
            f.write(f"{epoch + 1},{average_epoch_error:.6f}\n")

    return average_epoch_error