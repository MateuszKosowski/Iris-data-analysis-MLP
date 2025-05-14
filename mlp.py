import numpy as np
from layer import Layer

class MLP:
    def __init__(self, layer_sizes_array, use_bias=True, learning_rate=0.02, momentum=0.9):
        self.layers = []
        self.num_layers = len(layer_sizes_array) # Liczba warstw (ukryte + wyjściowa)
        self.layer_sizes_array = layer_sizes_array # Tablica z liczbą neuronów wejść w każdej warstwie
        self.use_bias = use_bias

        print(f"Tworzenie sieci MLP o strukturze: {layer_sizes_array}, Użycie biasu: {use_bias}")

        # Tworzymy warstwy
        for i in range(self.num_layers - 1):
            num_neurons_in_layer = layer_sizes_array[i + 1]  # Liczba neuronów w TEJ tworzonej warstwie
            num_inputs_per_neuron = layer_sizes_array[i]  # Liczba wejść = liczba neuronów w POPRZEDNIEJ warstwie

            print(f"  Tworzenie warstwy {i + 1} ({num_inputs_per_neuron} wejść,  {num_neurons_in_layer} neuronów)")

            # Utwórz obiekt Layer i dodaj do listy

            if i == self.num_layers - 2:
                # Ostatnia warstwa - ustawiamy flagę, że to warstwa wyjściowa
                layer = Layer(num_neurons=num_neurons_in_layer,
                              num_input_per_neuron=num_inputs_per_neuron,
                              use_bias=self.use_bias,
                              learning_rate=learning_rate,
                              is_last_layer=True,
                              momentum=momentum)

            else:
                layer = Layer(num_neurons=num_neurons_in_layer,
                              num_input_per_neuron=num_inputs_per_neuron,
                              use_bias=self.use_bias,
                              learning_rate=learning_rate,
                              momentum=momentum)

            self.layers.append(layer)

        print("Sieć MLP utworzona.")

    # Przekazywanie wag do warstwy wyjściowej
    def forward_pass(self, inputs):
        # Sprawdzenie, czy liczba wejść zgadza się z pierwszą warstwą
        if len(inputs) != self.layer_sizes_array[0]:
            raise ValueError(f"Liczba wejść musi być równa liczbie neuronów w pierwszej warstwie ({self.layer_sizes_array[0]}).")

        current_outputs = np.array(inputs)

        # Przechodzimy przez wszystkie warstwy
        for i, layer in enumerate(self.layers):
            #print(f"  Przetwarzanie przez warstwę {i+1}...")
            current_outputs = layer.calculate_outputs(current_outputs)

        return current_outputs

    def backward_pass(self, target_outputs_vector, use_momentum):

        # --- Krok 1: Warstwa wyjściowa ---

        output_layer = self.layers[-1] # Ostatnia warstwa w liście

        for i, neuron in enumerate(output_layer.neurons):
            neuron.calculate_delta(target_outputs_vector[i])  # Przekazujemy odpowiedni element celu
            neuron.calculate_gradient()  # Obliczamy gradienty wag dla tego neuronu
            neuron.update_weights(use_momentum)  # Aktualizujemy wagi i bias tego neuronu

        # --- Krok 2: Warstwy ukryte (od końca do początku) ---

        for layer_idx in range(len(self.layers) - 2, -1, -1): # Iteracja od przedostatniej do pierwszej warstwy, składnia: for i in range(start, end, step)

            current_hidden_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]  # Warstwa bliżej wyjścia

            for j_idx, neuron_hj in enumerate(current_hidden_layer.neurons): # j_idx - indeks neuronu w warstwie ukrytej

                propagated_error_to_hj = 0.0 # Zmienna do przechowywania błędu propagowanego do neuronu ukrytego

                for k_idx, neuron_k_next in enumerate(next_layer.neurons): # k_idx - indeks neuronu w warstwie wyjściowej
                    propagated_error_to_hj += neuron_k_next.delta * neuron_k_next.last_weights[j_idx] # musi być last_weight, bo aktualizacja nastąpiła już w warstwie wyjściowej

                neuron_hj.calculate_delta(propagated_error_to_hj)
                neuron_hj.calculate_gradient()
                neuron_hj.update_weights(use_momentum)  # aktualizujemy wagi tej warstwy ukrytej


    def calculate_mse(self, current_outputs, target_outputs):
        # Obliczanie błędu średniokwadratowego
        mse = np.mean((current_outputs - target_outputs) ** 2)
        return mse
