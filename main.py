from neuron import *
#import pandas as pd


def main():
    # # Wczytaj plik csv
    # file_test = './data/data3_test.csv'
    # file_train = './data/data3_train.csv'
    #
    # # Dane do trenowania
    # data_train = pd.read_csv(file_train, header=None, sep=',')
    # data_train.columns = ['Dlugosc kielicha', 'Szerokosc kielicha', 'Dlugosc platka', 'Szerokosc platka', 'Gatunek']
    #
    # # Dane do testowania
    # data_test = pd.read_csv(file_test, header=None, sep=',')
    # data_test.columns = ['Dlugosc kielicha', 'Szerokosc kielicha', 'Dlugosc platka', 'Szerokosc platka', 'Gatunek']

    # Testowy neuron
    neuron_is_a_dog = Neuron(4, 0.5)
    print("Neuron:")
    print("Liczba wejść:", neuron_is_a_dog.num_inputs)
    print("Wagi:", neuron_is_a_dog.weights)
    print("Bias:", neuron_is_a_dog.bias)

    # Przykładowe dane wejściowe
    how_many_legs = 4
    avg_weight = 10
    have_a_fin = 0
    is_barking = 1
    inputs = [how_many_legs, avg_weight, have_a_fin, is_barking]

    # Obliczanie wyjścia neuronu
    output = neuron_is_a_dog.calculate_output(inputs)
    print("Wyjście neuronu z losowymi wagami:", output)

    neuron_is_a_dog.set_weights([0.2, 0.3, -0.99, 0.9])

    # Obliczanie wyjścia neuronu po zmianie wag
    output = neuron_is_a_dog.calculate_output(inputs)
    print("Wyjście neuronu po zmianie wag:", output)

if __name__ == "__main__":
    main()