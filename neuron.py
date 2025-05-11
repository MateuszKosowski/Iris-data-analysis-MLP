import numpy as np

class Neuron:

    # Konstruktor klasy Neuron
    def __init__(self, num_inputs, use_bias, learning_rate, is_last_layer=False): # self oznacza instancję klasy w Pythonie, jest to konwencja

        if not isinstance(num_inputs, int) or num_inputs < 1:
            raise ValueError("Liczba wejść musi być dodatnią liczbą całkowitą.")

        self.num_inputs = num_inputs
        self.weights = (np.random.rand(num_inputs) * 2) - 1
        if use_bias:
            self.bias = (np.random.rand() * 2) - 1
        else:
            self.bias = 0

        # Do propagacji wstecznej
        self.learning_rate = learning_rate
        self.last_inputs_to_neuron = None  # Wejścia, które otrzymał ostatnio neuron
        self.last_sigmoid_input = None  # z = (inputs * weights) + bias - Nie używane, ale można by użyć do debugowania
        self.last_sigmoid_output = None  # a = sigmoid(z)
        self.delta = 0.0  # Miejsce na przechowywanie obliczonej delty tego neuronu
        self.gradient = None  # Miejsce na przechowywanie gradientu tego neuronu
        self.is_last_layer = is_last_layer  # Flaga, czy neuron jest w ostatniej warstwie
        self.weights_velocity = np.zeros_like(num_inputs)  # Wektory prędkości dla wag
        self.bias_velocity = 0  # Prędkość dla biasu
        self.momentum_param = 0.9 # Parametr momentum


    #Sigmoid - zwraca wartość między 0 a 1. Po zsumowaniu, wag i biasu mówi na ile % neuron jest aktywowany, czyli np na ile % jest to pies lub nie.
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def calculate_output(self, inputs):
        # Sprawdzenie, czy liczba wejść się zgadza
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Neuron oczekuje {self.num_inputs} wejść, a otrzymał {len(inputs)}.")

        # Zapisanie ostatnich wejść do neuronu
        self.last_inputs_to_neuron = inputs

        # 1. Oblicz sumę ważoną: (w1*x1 + w2*x2 + ... + wn*xn)
        weighted_sum = np.dot(inputs, self.weights)

        # 2. Dodaj bias
        total_input = weighted_sum + self.bias

        # 3. Zastosuj funkcję aktywacji
        self.last_sigmoid_input = total_input # Zapisanie ostatniego wejścia do funkcji aktywacji
        output = self._sigmoid(total_input)
        self.last_sigmoid_output = output # Zapisanie ostatniego wyjścia z funkcji aktywacji

        return output

    def calculate_delta(self, target_or_propagated_value):
        if self.is_last_layer:
            error = self.last_sigmoid_output - target_or_propagated_value  # Obliczenie błędu = wyjście neuronu - cel
            self.delta = error * self.calculate_derivative_of_sigmoid()
        else:
            self.delta = target_or_propagated_value * self.calculate_derivative_of_sigmoid()

    def calculate_derivative_of_sigmoid(self):
        # Obliczenie pochodnej funkcji sigmoidalnej
        return self.last_sigmoid_output * (1 - self.last_sigmoid_output)

    def calculate_gradient(self):
        # Obliczenie gradientu
        self.gradient = self.delta * self.last_inputs_to_neuron

    def update_weights(self):
        # Aktualizacja wag
        # self.weights += self.learning_rate * self.gradient
        self.weights_velocity = (self.momentum_param * self.weights_velocity) - (self.learning_rate * self.gradient)
        self.weights += self.weights_velocity
        # Aktualizacja biasu
        # self.bias += self.learning_rate * self.delta
        self.bias_velocity = (self.momentum_param * self.bias_velocity) - (self.learning_rate * self.delta)
        self.bias += self.bias_velocity