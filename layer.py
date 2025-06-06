import numpy as np
from neuron import Neuron

class Layer:
    # Konstruktor
    def __init__(self, num_neurons, num_input_per_neuron, use_bias, learning_rate, is_last_layer=False, momentum=0.9):
        self.num_neurons = num_neurons
        self.num_input_per_neuron = num_input_per_neuron
        # Tworzymy listę neuronów
        self.neurons = [Neuron(num_input_per_neuron, use_bias, learning_rate, is_last_layer, momentum) for _ in range(num_neurons)]
        # Inicjalizujemy wyjścia neuronów
        self.outputs = np.zeros(num_neurons) # array z zerami o długości num_neurons
        self.is_last_layer = is_last_layer

    # Zwraca wyjścia neuronów
    def calculate_outputs(self, inputs):
        if len(inputs) != self.num_input_per_neuron:
            raise ValueError(f"Liczba wejść musi być równa liczbie wejść na neuron ({self.num_input_per_neuron}).")

        # Obliczamy wyjścia dla każdego neuronu
        for i, neuron in enumerate(self.neurons):
            self.outputs[i] = neuron.calculate_output(inputs)

        return self.outputs
