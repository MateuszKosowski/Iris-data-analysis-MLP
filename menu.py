import pickle

# Metody pomocnicze - walidacja danych wejściowych
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

def get_non_negative_integer_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value >= 0:
                return value
            else:
                print("Wartość nie może być ujemna. Spróbuj ponownie.")
        except ValueError:
            print("Nieprawidłowe dane. Wprowadź liczbę całkowitą.")


# Główne menu
def get_network_config_from_user(num_inputs, num_outputs):

    print("\n--- Konfiguracja Warstw Ukrytych i Biasu ---")

    # Ile warstw neuronów ukrytych
    num_hidden_layers = get_non_negative_integer_input("Podaj liczbę warstw UKRYTYCH (0 lub więcej): ")

    # Liczba neuronów w każdej warstwie ukrytej
    hidden_layer_sizes = []
    for i in range(num_hidden_layers):
        size = get_positive_integer_input(f"Podaj liczbę neuronów w warstwie ukrytej nr {i + 1}: ")
        hidden_layer_sizes.append(size)

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

def mode_menu():
    print("\n--- Co chcesz zrobić? ---")
    print("1. Ucz sieć")
    print("2. Testuj sieć")
    print("3. Zapisz model sieci")
    print("4. Wczytaj model sieci")
    print("5. Utwórz model sieci")
    print("6. Tryb autoencoder")
    print("7. Do sprawozdania")
    print("8. Wyjdź")

    while True:
        choice = input("Wybierz opcję (1-8): ").strip()
        if choice == '1':
            return "learn"
        elif choice == '2':
            return "test"
        elif choice == '3':
            return "save"
        elif choice == '4':
            return "load"
        elif choice == '5':
            return "create"
        elif choice == '6':
            return "autoencoder"
        elif choice == '7':
            return "report"
        elif choice == '8':
            return "exit"
        else:
            print("Nieprawidłowy wybór. Wpisz numer od 1 do 8.")


def how_much_echos():
    while True:
        try:
            num_epochs_str = input("Podaj liczbę epok (1-1000) liczba całkowita: ")
            num_epochs = int(num_epochs_str)
            if 1 <= num_epochs <= 1000:
                return num_epochs
            else:
                print("Liczba epok musi być w zakresie od 1 do 1000. Spróbuj ponownie.")
        except ValueError:
            print("Nieprawidłowe dane. Wprowadź liczbę całkowitą. Spróbuj ponownie.")

def shuffle_data_menu():
    shuffle_data = None
    while shuffle_data is None:
        shuffle_data = input("Czy pomieszać dane? (tak/nie): ").strip().lower()
        if shuffle_data in ['t', 'tak', 'yes', 'y']:
            shuffle_data = True
            return shuffle_data
        elif shuffle_data in ['n', 'nie', 'no']:
            shuffle_data = False
            return shuffle_data
        else:
            print("Nieprawidłowy wybór. Wpisz 'tak' lub 'nie'.")

def set_learning_rate_menu():
    learning_rate = None
    while learning_rate is None:
        try:
            learning_rate_str = input("Podaj współczynnik uczenia (0.01 - 0.5): ")
            learning_rate_val = float(learning_rate_str)
            if 0.01 <= learning_rate_val <= 0.5:
                learning_rate = learning_rate_val
                return learning_rate
            else:
                print("Wartość musi być w zakresie od 0.01 do 0.5. Spróbuj ponownie.")
                learning_rate = None  # Resetuj, aby pętla kontynuowała
        except ValueError:
            print("Nieprawidłowe dane. Wprowadź liczbę zmiennoprzecinkową. Spróbuj ponownie.")
            learning_rate = None # Resetuj, aby pętla kontynuowała


def set_momentum_value_menu():
    momentum = None
    while momentum is None:
        try:
            momentum_str = input("Podaj wartość momentum (0.1 - 0.99): ")
            momentum_val = float(momentum_str)
            if 0.1 <= momentum_val <= 0.99:
                momentum = momentum_val
                return momentum
            else:
                print("Wartość musi być w zakresie od 0.1 do 0.99. Spróbuj ponownie.")
                momentum = None  # Resetuj, aby pętla kontynuowała
        except ValueError:
            print("Nieprawidłowe dane. Wprowadź liczbę zmiennoprzecinkową. Spróbuj ponownie.")
            momentum = None # Resetuj, aby pętla kontynuowała


# Zapis do pliku - serializacja obiketu - zapis w postaci bajtów
def save_mlp_to_file(mlp):
    filename = input("Podaj nazwę pliku do zapisu modelu MLP (np. model.pkl): ")
    try:
        with open(filename, 'wb') as f:
            pickle.dump(mlp, f)
        print(f"Model MLP został pomyślnie zapisany do pliku '{filename}'.")
    except Exception as e:
        print(f"Wystąpił błąd podczas zapisywania modelu: {e}")

# Wczytaj z pliku
def load_mlp_from_file():
    filename = input("Podaj nazwę pliku do wczytania modelu MLP (np. model.pkl): ")
    try:
        with open(filename, 'rb') as f:
            mlp = pickle.load(f)
        print(f"Model MLP został pomyślnie wczytany z pliku '{filename}'.")
        return mlp
    except FileNotFoundError:
        print(f"Błąd: Plik '{filename}' nie został znaleziony.")
        return None
    except Exception as e:
        print(f"Wystąpił błąd podczas wczytywania modelu: {e}")
        return None

def use_momentum_menu():
    use_momentum = None
    while use_momentum is None:
        choice = input("Czy chcesz użyć momentum podczas uczenia? (tak/nie): ").strip().lower()
        if choice in ['t', 'tak', 'yes', 'y']:
            use_momentum = True
            return use_momentum
        elif choice in ['n', 'nie', 'no']:
            use_momentum = False
            return use_momentum
        else:
            print("Nieprawidłowy wybór. Wpisz 'tak' lub 'nie'.")

def epochs_or_precision():
    choice = None
    while choice not in ['e', 'epoka', 'epoch', 'p', 'precyzja', 'precision']:
        choice = input("Warunek zakończenia nauki (epoka/precyzja): ").strip().lower()
        if choice in ['e', 'epoka', 'epoch']:
            choice = "epoch"
            return choice
        elif choice in ['p', 'precyzja', 'precision']:
            choice = "precision"
            return choice
        else:
            print("Nieprawidłowy wybór. Wpisz 'e' lub 'p'.")

def give_precision():
    precision = None
    while precision is None:
        try:
            precision_str = input("Podaj docelową wartość błędu MSE (od 0.001 do 0.999): ")
            precision_val = float(precision_str)
            if 0.001 <= precision_val <= 0.999:
                precision = precision_val
                return precision
            else:
                print("Wartość MSE musi być w zakresie od 0.001 do 0.999. Spróbuj ponownie.")
                precision = None  # Resetuj, aby pętla kontynuowała
        except ValueError:
            print("Nieprawidłowe dane. Wprowadź liczbę zmiennoprzecinkową. Spróbuj ponownie.")
            precision = None # Resetuj, aby pętla kontynuowała



