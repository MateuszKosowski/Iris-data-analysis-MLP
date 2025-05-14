import numpy as np
import random
from mlp import MLP

autoencoder_patterns = (
    (np.array([1,0,0,0]), np.array([1,0,0,0])),
    (np.array([0,1,0,0]), np.array([0,1,0,0])),
    (np.array([0,0,1,0]), np.array([0,0,1,0])),
    (np.array([0,0,0,1]), np.array([0,0,0,1]))
)

# Architektura autoenkodera
autoencoder_architecture = (4, 2, 4) # 4 wejścia, 2 ukryte, 4 wyjścia

def create_autoencoder_mlp(use_bias=True, architecture=autoencoder_architecture):
    return MLP(layer_sizes_array=architecture, use_bias=use_bias)

def train_autoencoder(mlp_instance, epochs, learning_rate, use_momentum, momentum_coeff, patterns=autoencoder_patterns, shuffle_patterns=True, target_mse_for_stop=None):
    print(f"\n--- Trening Autoenkodera ---")
    print(f"Architektura: {mlp_instance.layer_sizes_array}, Bias: {mlp_instance.use_bias}")
    print(f"LR: {learning_rate}, Momentum: {momentum_coeff if use_momentum else 'Brak'}, Epoki: {epochs}")
    print(f"Liczba wzorców: {len(patterns)}")



    # ustawienie momentum i wartośći dla każdego neuronu
    for layer in mlp_instance.layers:
        for neuron in layer.neurons:
            neuron.learning_rate = learning_rate
            neuron.momentum_param = momentum_coeff

    for epoch in range(epochs):
        current_patterns = list(patterns)
        if shuffle_patterns:
            random.shuffle(current_patterns)

        total_epoch_error = 0.0

        for input_pattern, target_pattern in current_patterns:

            # Forward pass
            output = mlp_instance.forward_pass(input_pattern)

            # Backward pass
            mlp_instance.backward_pass(target_pattern, use_momentum)  # Przekaż flagę use_momentum

            # Oblicz błąd dla wzorca
            error_for_sample = mlp_instance.calculate_mse(output, target_pattern)
            total_epoch_error += error_for_sample

        average_epoch_error = total_epoch_error / len(patterns)

        if (epoch + 1) % 10 == 0:  # Loguj co 10 epok dla autoenkodera
            print(f"Epoka {epoch + 1}/{epochs}, MSE: {average_epoch_error:.8f}")

        if target_mse_for_stop is not None and average_epoch_error <= target_mse_for_stop:
            print(f"Osiągnięto docelowy błąd MSE {target_mse_for_stop} w epoce {epoch + 1}.")
            break

    print(f"Trening zakończony. Końcowe MSE: {average_epoch_error:.8f}")
    return average_epoch_error


def get_hidden_layer_outputs(mlp_instance, patterns = autoencoder_patterns):
    """Zwraca aktywacje warstwy ukrytej dla podanych wzorców."""
    hidden_outputs_all_patterns = []
    # Założenie: warstwa ukryta to pierwsza warstwa przetwarzająca, czyli self.layers[0]
    if not mlp_instance.layers:
        print("Błąd: Sieć nie ma warstw.")
        return None

    hidden_layer_index = 0  # Pierwsza warstwa przetwarzająca to warstwa ukryta

    for input_pattern, _ in patterns:
        mlp_instance.forward_pass(input_pattern)  # Wykonaj forward pass, aby zaktualizować .outputs w warstwach
        # Odczytaj .outputs z odpowiedniej warstwy ukrytej
        hidden_activations = mlp_instance.layers[hidden_layer_index].outputs.copy()
        hidden_outputs_all_patterns.append(hidden_activations)
    return hidden_outputs_all_patterns