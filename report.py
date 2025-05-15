import numpy as np

from autoencoder import create_autoencoder_mlp, train_autoencoder, autoencoder_patterns, get_hidden_layer_outputs, \
    train_and_test_autoencoder
from learn import learn_one_epoch
from mlp import MLP
from matplotlib import pyplot as plt

def report(train_features, test_features, train_labels_vector, test_labels_vector, train_labels, test_labels):

    mlp_instance = MLP(layer_sizes_array=(4, 2, 3, 3), use_bias=True, learning_rate=0.2, momentum=0.2)

    # Przygotowanie danych
    np.random.seed(50)  # Ustawienie ziarna dla powtarzalności permutacji
    permutation = np.random.permutation(len(train_features))
    train_features_to_use = train_features[permutation]
    train_label_vector_to_use = train_labels_vector[permutation]

    # train_features_to_use = train_features
    # train_label_vector_to_use = train_labels_vector

    report_mlp(
        mlp_instance,
        "1",
        train_features_to_use,
        test_features,
        train_label_vector_to_use,
        test_labels_vector,
        train_labels,
        test_labels,
        use_momentum=True
    )

    make_plot("1")

    return mlp_instance

def report_autoencoder():
    mlp_instance = create_autoencoder_mlp(use_bias=True, architecture=(4, 2, 4))
    report_number = "2"
    learning_rate = 0.6
    momentum_coeff = 0
    epochs = 1000
    use_momentum = False

    train_error, test_error = train_and_test_autoencoder(
        mlp_instance,
        train_patterns=autoencoder_patterns,
        test_patterns=autoencoder_patterns,
        epochs=epochs,
        learning_rate=learning_rate,
        use_momentum=use_momentum,
        momentum_coeff=momentum_coeff,
        log_filename_train=f"./reports/Rep{report_number}_train.txt",
        log_filename_test=f"./reports/Rep{report_number}_test.txt",
        shuffle_patterns=True,
        target_mse_for_stop=None
    )

    print("Wyjścia warstwy ukrytej:")
    hidde_layer_outputs = get_hidden_layer_outputs(mlp_instance)
    for i in range(len(hidde_layer_outputs)):
        print(f"Wartość wyjściowa neuronów ukrytych, po nauce, dla paternu {i + 1}: {hidde_layer_outputs[i]}")

    make_plot("2")


def report_mlp(
        mlp_instance,
        report_number,
        train_features,
        test_features,
        train_labels_vector,
        test_labels_vector,
        train_labels,
        test_labels,
        precision=0.02,
        num_epochs=1000,
        use_momentum=False
):
    average_epoch_error = 2.0
    avg_test_epoch_mse = 2.0
    epoch = 0
    filename = "./reports/Rep" + report_number

    # Czyszczenie pliku do zapisu
    with open(filename + "_train.txt", "w") as f:
        f.write("")
    with open(filename + "_test.txt", "w") as f:
        f.write("")

    while average_epoch_error > precision and epoch < num_epochs:
        epoch += 1  # counter do liczenia która epoka
        average_epoch_error = learn_one_epoch(
            len(train_features),
            num_epochs,
            use_momentum,
            mlp_instance,
            train_features,
            train_labels_vector,
            epoch,
            filename + "_train.txt"
        )

        test_sum_mse = 0.0

        # --- Pętla po próbkach testowych ---
        for i in range(len(test_features)):
            input_sample = test_features[i]
            target_one_hot_vector = test_labels_vector[i]  # zmienna

            output_probabilities = mlp_instance.forward_pass(input_sample)

            mse_for_sample = mlp_instance.calculate_mse(output_probabilities, target_one_hot_vector)

            test_sum_mse += mse_for_sample

        avg_test_epoch_mse = test_sum_mse / len(test_features)

        # Zapisz bład do pliku co 10 epokę
        if (epoch + 1) % 10 == 0:
            with open(filename + "_test.txt", "a") as f:
                f.write(f"{epoch + 1},{avg_test_epoch_mse:.6f}\n")

    # Zapisz ostatnią epokę - jeśli nie była już zapisana
    if (epoch + 1) % 10 != 0:
        with open(filename + "_train.txt", "a") as f:
            f.write(f"{epoch + 1},{average_epoch_error:.6f}\n")
        with open(filename + "_test.txt", "a") as f:
            f.write(f"{epoch + 1},{avg_test_epoch_mse:.6f}\n")

def make_plot(report_number):
    train_file = f"./reports/Rep{report_number}_train.txt"
    test_file = f"./reports/Rep{report_number}_test.txt"

    def read_data(filename):
        epochs = []
        errors = []
        with open(filename, "r") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        epochs.append(int(parts[0]))
                        errors.append(float(parts[1]))
        return epochs, errors

    train_epochs, train_errors = read_data(train_file)
    test_epochs, test_errors = read_data(test_file)

    plt.figure(figsize=(8, 5))
    plt.plot(train_epochs, train_errors, label="Błąd treningowy", color="blue")
    plt.plot(test_epochs, test_errors, label="Błąd testowy", color="orange")
    plt.xlabel("Numer epoki")
    plt.ylabel("Współczynnik błędu")
    plt.title("Błąd treningowy i testowy w kolejnych epokach")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()