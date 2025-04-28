import numpy as np

class Neuron:

    # Konstruktor klasy Neuron
    def __init__(self, num_inputs, use_bias): # self oznacza instancję klasy w Pythonie, jest to konwencja

        if not isinstance(num_inputs, int) or num_inputs < 1:
            raise ValueError("Liczba wejść musi być dodatnią liczbą całkowitą.")

        self.num_inputs = num_inputs
        self.weights = (np.random.rand(num_inputs) * 2) - 1
        if use_bias:
            self.bias = (np.random.rand() * 2) - 1
        else:
            self.bias = 0


    #Sigmoid - zwraca wartość między 0 a 1. Po zsumowaniu, wag i biasu mówi na ile % neuron jest aktywowany, czyli np na ile % jest to pies lub nie.
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def calculate_output(self, inputs):
        # Sprawdzenie, czy liczba wejść się zgadza
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Neuron oczekuje {self.num_inputs} wejść, a otrzymał {len(inputs)}.")

        # 1. Oblicz sumę ważoną: (w1*x1 + w2*x2 + ... + wn*xn)
        weighted_sum = np.dot(inputs, self.weights)

        # 2. Dodaj bias
        total_input = weighted_sum + self.bias

        # 3. Zastosuj funkcję aktywacji
        output = self._sigmoid(total_input)

        return output

    def set_weights(self, weights):
        if len(weights) != self.num_inputs:
            raise ValueError(f"Liczba wag musi być równa liczbie wejść ({self.num_inputs}).")
        self.weights = weights
