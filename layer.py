import numpy as np
from neuron import Neuron

class Layer:
    # Konstruktor
    def __init__(self, num_neurons, num_input_per_neuron, use_bias):
        self.num_neurons = num_neurons
        self.num_input_per_neuron = num_input_per_neuron
        # Tworzymy listę neuronów
        self.neurons = [Neuron(num_input_per_neuron, use_bias) for _ in range(num_neurons)]
        # Inicjalizujemy wyjścia neuronów
        self.outputs = np.zeros(num_neurons) # array z zerami o długości num_neurons

    # Zwraca wyjścia neuronów
    def calculate_outputs(self, inputs):
        if len(inputs) != self.num_input_per_neuron:
            raise ValueError(f"Liczba wejść musi być równa liczbie wejść na neuron ({self.num_input_per_neuron}).")

        # Obliczamy wyjścia dla każdego neuronu
        for i, neuron in enumerate(self.neurons):
            self.outputs[i] = neuron.calculate_output(inputs)

        return self.outputs

    # Ustawia wagi neuronów
    def set_weights(self, weights):
        if len(weights) != self.num_neurons:
            raise ValueError(f"Liczba wag musi być równa liczbie neuronów ({self.num_neurons}).")

        for i, neuron in enumerate(self.neurons):
            neuron.set_weights(weights[i])