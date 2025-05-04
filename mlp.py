import numpy as np
from layer import Layer

class MLP:
    def __init__(self, layer_sizes_array, use_bias=True):
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
            layer = Layer(num_neurons=num_neurons_in_layer,
                          num_input_per_neuron=num_inputs_per_neuron,
                          use_bias=self.use_bias)
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