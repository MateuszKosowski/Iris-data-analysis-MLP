
# Metoda pomocnicza do wczytywania dodatniej liczby całkowitej od użytkownika
def get_positive_integer_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            else:
                print("Wartość musi być dodatnią liczbą całkowitą. Spróbuj ponownie.")
        except ValueError:
            print("Nieprawidłowe dane. Wprowadź liczbę całkowitą.")


def get_network_config_from_user():
    print("--- Konfiguracja sieci MLP ---")

    # Ile inputów
    num_inputs = get_positive_integer_input("Podaj liczbę WEJŚĆ: ")

    # Ile warstw neuronów ukrytych
    while True:
        try:
            num_hidden_layers = int(input("Podaj liczbę warstw UKRYTYCH (0 lub więcej): "))
            if num_hidden_layers >= 0:
                break
            else:
                print("Liczba warstw ukrytych nie może być ujemna. Spróbuj ponownie.")
        except ValueError:
            print("Nieprawidłowe dane. Wprowadź liczbę całkowitą.")

    # Liczba neuronów w każdej warstwie ukrytej
    hidden_layer_sizes = []
    for i in range(num_hidden_layers):
        size = get_positive_integer_input(f"Podaj liczbę neuronów w warstwie ukrytej nr {i + 1}: ")
        hidden_layer_sizes.append(size)

    # Liczba neuronów w warstwie wyjściowej
    num_outputs = 3 # Bo są 3 rodzaje irysów

    # Użycie biasu
    use_bias = None
    while use_bias is None:
        bias_choice = input("Czy używać biasu w neuronach przetwarzających? (tak/nie): ").strip().lower()
        if bias_choice in ['t', 'tak', 'yes', 'y']:
            use_bias = True
        elif bias_choice in ['n', 'nie', 'no']:
            use_bias = False
        else:
            print("Nieprawidłowy wybór. Wpisz 'tak' lub 'nie'.")

    # Składanie architektury w listę
    layer_sizes_array = [num_inputs] + hidden_layer_sizes + [num_outputs]

    print("\n--- Podsumowanie Konfiguracji ---")
    print(f"Architektura sieci: WEJŚĆIA - liczba neuronów w warstwach ukrytych - liczba neuronów w warstwie WYJŚCIOWE: {layer_sizes_array}")
    print(f"Użycie biasu: {'Tak' if use_bias else 'Nie'}")

    return layer_sizes_array, use_bias

