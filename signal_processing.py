import pickle
import matplotlib.pyplot as plt
from numpy import abs, arange, interp, ceil, int64, square, where, trapz
from numpy.fft import fft, fftfreq
import os
import sys
from PIL import Image
from plugin_wrapper import WaveformClassificationPlugin

def load_data(file_path):
    """
    #### Summary
    Ładuje dane z pliku pickle.

    #### Args:
     - file_path (str): Ścieżka do pliku pickle.

    #### Returns:
     - dict: Załadowane dane.
    """
    try:
        with open(file_path, 'rb') as f:    # Otwarcie pliku 
            data = pickle.load(f)           # wczytanie danych
        return data
    except:
        return None

def calculate_timestamps(signal, time_start, sampling_frequency):
    """
    #### Summary
    Oblicza znaczniki czasowe dla sygnału na podstawie częstotliwości próbkowania.

    #### Args:
     - signal (numpy.ndarray): Dane sygnału.
     - time_start (int): Czas rozpoczęcia pomiaru.
     - sampling_frequency (float): Częstotliwość próbkowania sygnału.

    #### Returns:
     - numpy.ndarray: Tablica znaczników czasowych.
    """
    # Tworzę wektor czasu na podstawie częstotliwości próbkowania
    time_length = len(signal)
    time = arange(time_length) / sampling_frequency  # czas w sekundach
    time = time + time_start  # dodanie czasu początkowego
    return time

    # period = 1 / sampling_frequency                     # Oblicznie okresu na podstawie częstotliwości
    # time_stop = time_start + len(signal) * period       # Obliczenie końcowego znacznika czasowego
    # timestamps = arange(time_start, time_stop, period)  # Obliczenie znaczników czasowych za pomocą numpy.arrange
    # return timestamps

def interpolate_signal(data, current_timestamps, target_period):
    """
    #### Summary
    Interpoluje sygnał do nowej częstotliwości próbkowania.

    #### Args:
     - data (numpy.ndarray): Dane sygnału.
     - current_timestamps (numpy.ndarray): Znaczniki czasowe oryginalnego sygnału.
     - target_period (float): Docelowy okres próbkowania.

    Returns:
     - tuple: Tablica nowych znaczników czasowych i interpolowany sygnał.
    """
    target_timestamps = arange(current_timestamps[0], current_timestamps[-1], target_period)    # Obliczenie docelowych znaczników czasowych za pomocą numpy.arrange
    interpolated_signal = interp(target_timestamps, current_timestamps, data)                   # Interpolowanie sygnału na obliczonych znacznikach czasowych za pomocą numpy.interp
    return target_timestamps, interpolated_signal

def plot_signal(timestamps, signal, title, xlabel, ylabel, show_plot = True, save_path = None):
    """
    #### Summary
    Rysuje sygnał w dziedzinie czasu.
    
    #### Args:
     - timestamps (numpy.ndarray): Znaczniki czasowe sygnału.
     - signal (numpy.ndarray): Dane sygnału.
     - title (str): Tytuł wykresu.
     - xlabel (str): Etykieta osi X.
     - ylabel (str): Etykieta osi Y.
     - show_plot (bool): Pokaż wykres.
     - save_path (str): Ścieżka do zapisania wykresu.
    """
    plt.figure(figsize=(8, 6))      # Inicjalizacja figury
    plt.plot(timestamps, signal)    # Wykres dla sygnału
    plt.title(title)                # Dodatnie tytułu
    plt.xlabel(xlabel)              # Dodanie opisu osi X
    plt.ylabel(ylabel)              # Dodanie opisu osi Y
    if show_plot:                   # Jeśli trzeba pokaż wykres
        plt.show()
    if save_path != None:                           # Jeśli podana ścieżka do pliku, zapisz wykres pod nią
        output_dir = os.path.dirname(save_path)     # Stwórz folder jeśli nie istnieje
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(save_path)
    plt.close()                     # Zamknięcie figury

def plot_fft(signal_fft, signal_fft_frequency, sampling_frequency, title, xlabel, ylabel, show_plot = True, save_path = None):
    """
    #### Summary
    Rysuje FFT sygnału w dziedzinie częstotliwości.

    #### Args:
     - signal_fft (numpy.ndarray): fft sygnału.
     - signal_fft_frequency (numpy.ndarray): częstotliwość dla fft.
     - sampling_frequency (float): Częstotliwość próbkowania sygnału.
     - title (str): Tytuł wykresu.
     - xlabel (str): Etykieta osi X.
     - ylabel (str): Etykieta osi Y.
     - show_plot (bool): Pokaż wykres.
     - save_path (str): Ścieżka do zapisania wykresu.
    """
    plt.figure(figsize=(8, 6))              # Inicjalizacja figury
    plt.plot(signal_fft[1], abs(signal_fft[0]))                   # Wykres dla FFT z określoną częstotliwością
    plt.title(title)                        # Dodatnie tytułu
    plt.xlabel(xlabel)                      # Dodanie opisu osi X
    plt.ylabel(ylabel)                      # Dodanie opisu osi Y
    plt.xlim([-signal_fft_frequency/2, sampling_frequency/2])       # Ograniczenie osi X
    if show_plot:                           # Jeśli trzeba pokaż wykres
        plt.show()
    if save_path != None:                           # Jeśli podana ścieżka do pliku, zapisz wykres pod nią
        output_dir = os.path.dirname(save_path)     # Stwórz folder jeśli nie istnieje
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(save_path)
    plt.close()                             # Zamknięcie figury

def plot_psd(signal_psd, signal_psd_frequency, sampling_frequency, x_lim, y_lim, title, xlabel, ylabel, show_plot = True, save_path = None):
    """
    #### Summary
    Rysuje PSD sygnału w dziedzinie częstotliwości.

    #### Args:
     - signal_psd (numpy.ndarray): fft sygnału.
     - signal_psd_frequency (numpy.ndarray): częstotliwość dla fft.
     - sampling_frequency (float): Częstotliwość próbkowania sygnału.
     - x_lim (min, max): Min oraz Max dla osi X
     - y_lim (min, max): Min oraz Max dla osi Y
     - title (str): Tytuł wykresu.
     - xlabel (str): Etykieta osi X.
     - ylabel (str): Etykieta osi Y.
     - show_plot (bool): Pokaż wykres.
     - save_path (str): Ścieżka do zapisania wykresu.
    """
    plt.figure(figsize=(8, 6))              # Inicjalizacja figury
    plt.plot(signal_psd_frequency, signal_psd)                   # Wykres dla PSD z określoną częstotliwością
    plt.title(title)                        # Dodatnie tytułu
    plt.xlabel(xlabel)                      # Dodanie opisu osi X
    plt.ylabel(ylabel)                      # Dodanie opisu osi Y
    plt.xlim(x_lim)                         # Ograniczenie osi X
    plt.ylim(y_lim)                         # Ograniczenie osi Y
    if show_plot:                           # Jeśli trzeba pokaż wykres
        plt.show()
    if save_path != None:                           # Jeśli podana ścieżka do pliku, zapisz wykres pod nią
        output_dir = os.path.dirname(save_path)     # Stwórz folder jeśli nie istnieje
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(save_path)
    plt.close()    

def plot_combined(signal_fft, sampling_frequency, signal, window_ranges, title, xlabel, ylabel, show_plot = True, save_path = None):
    """
    #### Summary
    Rysuje FFT sygnału w dziedzinie częstotliwości, a poniżej rysuje okno dla którego wykonano fft.

    #### Args:
     - `signal_fft` (tuple of numpy.ndarray and numpy.ndarray): Dane sygnału w dziedzinie częstotliwości wraz z odpowiadającą częstotliwością.
     - `sampling_frequency` (float): Częstotliwość próbkowania sygnału.
     - `signal` (numpy.ndarray): Dane sygnału.
     - `window_ranges` (touple(int, int)): Zakresy dla okna dla FFT.
     - `title` (str): Tytuł wykresu.
     - `xlabel` (str): Etykieta osi X.
     - `ylabel` (str): Etykieta osi Y.
     - `show_plot` (bool): Pokaż wykres.
     - `save_path` (str): Ścieżka do zapisania wykresu.
    """
    plt.figure(figsize=(8, 6))          # Inicjalizacja figury 
    plt.subplot(2, 1, 1)                # Pierwszy wykres - fft
    plt.plot(signal_fft[1], abs(signal_fft[0]))        # Wykres dla FFT z poprawną częstotliwością w Hz
    plt.title(title)                    # Dodatnie tytułu
    plt.xlabel(xlabel)                  # Dodanie opisu osi X
    plt.ylabel(ylabel)                  # Dodanie opisu osi Y
    plt.xlim([-sampling_frequency/2, sampling_frequency/2])   # Ograniczenie osi X

    plt.subplot(2, 1, 2)                # Pierwszy wykres - pokazanie okna na sygnale
    plt.plot(signal)                    # Drugi wykres dla wartości sygnału
    signal_max = max(signal)            # Znalezienie max wartości w sygnale
    plt.plot([signal_max if i >= window_ranges[0] and i < window_ranges[1] else 0 for i in range(len(signal))])     # Wykreślenie okna nałożonego na sygnał
    plt.legend(["sygnał", "okno FFT"])  # Dodanie legendy

    if show_plot:                       # Jeśli trzeba pokaż wykres
        plt.show()
    if save_path != None:                           # Jeśli podana ścieżka do pliku, zapisz wykres pod nią
        output_dir = os.path.dirname(save_path)     # Stwórz folder jeśli nie istnieje
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(save_path)
    plt.close()                         # Zamknięcie figury

def calculate_signal_energy(signal):
    """
    #### Summary
    Funkcja obliczająca energię sygnału.

    ##### How?
    `E = sum(e(x) ^ 2)`
    
    #### Args:
     - signal (numpy.ndarray): Dane sygnału.

    #### Returns:
     - float: Obliczona wartość energii sygnału.
    """
    total_energy = 0.0                  # Inicjalizacja sumy energii
    for value in signal:        
        total_energy += value ** 2      # Dodanie energii dla kolejnej próbki 
    return total_energy

def calculate_fft(signal, frequency):
    """
    #### Summary
    Funkcja obliczająca fft.
    
    #### Args:
     - signal (numpy.ndarray): Dane sygnału.
     - frequency (float): Częstotliwość sygnału.

    #### Returns:
     - touple: (Obliczona wartość FFT, wektor częstotliwości dla obliczonego FFT.
    """
    return (fft(signal), fftfreq(len(signal), d=1/frequency))

def calculate_windowed_fft(signal, window_size, step_size, frequency):
    """
    #### Summary
    Funkcja obliczająca fft w sposób okienkowy ze stałą szerokością okna.
    
    #### Args:
     - signal (numpy.ndarray): Dane sygnału.
     - window_size (int): Szerokość okna.
     - step_size (int): Krok okna w każdej iteracji.

    #### Returns:
     - list: Obliczona wartość FFT.
     - list of touples: Lista zawierająca krotki z punktami granicznymi okna.
    """
    signal_length = len(signal)                             # Długość sygnału
    steps_number = int64(ceil(signal_length / step_size))   # Obliczenie ilości kroków, które trzeba przetworzyć dla danego okna
    fft_windows = []                                        # Lista na FFT dla danych okien     
    window_ranges = []                                      # Lista na zakresy dla okien
    for i in range(steps_number):
        start_point = step_size * i                         # Obliczenie początkowego punktu dla okna
        end_point = start_point + window_size               # Obliczenie końcowego punktu dla okna
        if end_point > signal_length - 1:                   # Ograniczenie zakresu, żeby nie wykraczało poza długość sygnału 
            end_point = signal_length - 1
        if start_point >= signal_length - 1:                # Jeśli punk początkowy trafił na koniec sygnału, wóczas kończymy
            break
        data = signal[start_point : end_point]              # Odczytanie danych dla wybranego zakresu
        fft_windows.append(calculate_fft(data, frequency))  # Obliczenie FFT dla wybranego zakresu
        window_ranges.append((start_point, end_point))      # Dodanie zakresów do listy zakresów 
    return fft_windows, window_ranges

def calculate_adaptive_windowed_fft(signal, window_min, window_max, step_size, frequency):
    """
    #### Summary
    Funkcja obliczająca fft w sposób okienkowy z adaptacyjną szerokością okna.
    
    #### Args:
     - signal (numpy.ndarray): Dane sygnału.
     - window_min (int): Minimalna szerokość okna.
     - window_max (int): Maksymalna szerokość okna.
     - step_size (int): Krok okna w każdej iteracji.

    #### Returns:
     - list: Obliczona wartość FFT oraz częstotliwości.
     - list of touples: Lista zawierająca krotki z punktami granicznymi okna.
    """
    signal_length = len(signal)                         # Długość sygnału
    window_size = (window_max + window_min) // 2        # Początkowa wartość szerokości okna
    fft_windows = []                                    # Lista na FFT dla danych okien
    window_ranges = []                                  # Lista na zakresy dla okien
    energy = 0.0                                        # Początkowa wartość energii dla danego zakresu
    previous_energy = 0.0                               # Początkowa wartość energii dla poprzedniego zakresu 
    finish = False                                      # Oznaczenie końca przetwarzania
    start_point = 0                                     # Początkowy punkt dla okna
    end_point = window_size                             # Końcowy punkt dla okna
    while True:
        if finish:                                      # Jeśli znacznik zakończenia obliczeń jest True, kończymy
            break
        start_point = start_point + step_size           # Obliczenie startu okna
        end_point = start_point + window_size           # Obliczenie końca okna
        if start_point >= signal_length - 1:            # Jeśli punk początkowy trafił na koniec sygnału, wóczas kończymy
            break
        if end_point >= signal_length - 1:              # Ograniczenie zakresu, żeby nie wykraczało poza długość sygnału
            end_point = signal_length - 1
            finish = True                               # Osiągnęliśmy koniec sygnału więc koniec przetwarzania    
        data = signal[start_point : end_point]          # Odczytanie danych dla wybranego zakresu
        window_ranges.append((start_point, end_point))  # Dodanie zakresów do listy zakresów 
        fft_windows.append(calculate_fft(data, frequency))         # Obliczenie FFT dla wybranego zakresu
        energy = calculate_signal_energy(data)          # Oblicz energię dla danego zakresu
        
        if energy < 0.95 * previous_energy:             # Sprawdzenie czy energia sygnału się zmniejszyła
            window_size = int(window_size * 1.2)        # Jeśli jest mniejsza, wówczas zwiększ rozmiar okna
        elif energy > 1.05 * previous_energy:           # Sprawdzenie czy energia sygnału wzrosła
            window_size = int(window_size * 0.8)        # Jeśli jest większa, wówczas zmniejsz rozmiar okna
        
        if window_size > window_max:                    # Weryfikacja czy okno nie jest za duże
            window_size = window_max
        elif window_size < window_min:                  # Weryfikacja czy okno nie jest za małe
            window_size = window_min
        
        previous_energy = energy                        # Zapamiętujemy obliczoną energię dla kolejnej iteracji
        
    return fft_windows, window_ranges

def create_gif(image_folder, output_path, duration=0.5):
    output_dir = os.path.dirname(output_path)                                   # Stwórz folder jeśli nie istnieje
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = [f for f in os.listdir(image_folder) if f.endswith('.png')]         # Odczytanie listy plików w folderze - są to zdjęcia z nazwą jak: 0.png
    files.sort(key=lambda f: int(os.path.splitext(f)[0]))                       # Posortowanie plików według nazwy - aby zachować kolejność przy generacji GIF'a
    images = [Image.open(os.path.join(image_folder, file)) for file in files]   # Wczytanie obrazów
    images[0].save(                                                             # Zapisanie obrazów jako GIF
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration * 1000,  # Czas trwania w milisekundach
        loop=0
    )

def calculate_power_spectral_density(signal, frequency):
    """
    #### Summary
    Funkcja obliczająca PSD (power spectral density)
    
    #### Args:
     - signal (numpy.ndarray): Dane sygnału.
     - frequency (float): Częstotliwość próbkowania.

    #### Returns:
     - tuple: Obliczona wartość PSD oraz odpowiadająca jej częstotliwość.
    """
    fft, freq = calculate_fft(signal, frequency)        # Obliczam fft
    N = len(signal)                                     # Obliczam długość sygnału
    psd = 1/(N * frequency) * square(abs(fft))          # Obliczam PSD jako: PSD = A^2/(N*fs), gdzie N to długość sygnału, z którego liczone jest FFT, a fs to to częstotliwość próbkowania
    return (psd, freq)

def main():
    """
    #### Summary
    Główna funkcja programu.
    """
    for_all = True

    # Folder z danymi
    data_folder = 'C:\\Users\\Nati\\Desktop\\DATA_TBI_NKowal\\ICP'

    # Sprawdzenie czy folder istnieje
    if not os.path.exists(data_folder):
        print(f"Folder {data_folder} nie istnieje.")
        exit()

    # Pobieranie listy plików pickle z folderu
    pickle_files = [f for f in os.listdir(data_folder) if f.endswith('.pkl')]

    if not pickle_files:
        print(f"Brak plików pickle w folderze {data_folder}.")
        exit()

    print(f"Znaleziono {len(pickle_files)} plików pickle do analizy.")

    # Docelowa częstotliwość w Hz
    target_ft = 100.0

    # Obliczenie docelowego okresu w sekundach
    target_period = 1 / target_ft
    
    # Przetwarzanie wszystkich plików pickle w folderze
    for file_index, pickle_file in enumerate(pickle_files):
        file_path = os.path.join(data_folder, pickle_file)
        print(f"\nAnalizowanie pliku {file_index+1}/{len(pickle_files)}: {pickle_file}")
        
        # Załadowanie danych z pliku
        data = load_data(file_path)
        if data is None:
            print(f"Nie znaleziono pliku z danymi: {file_path}")
            continue
        
        # Pętla w celu analizy wszystkich sygnałów
        for count, single_signal in enumerate(data):
            # Generowanie unikalnego ID dla sygnału (nazwa pliku bez rozszerzenia + indeks sygnału)
            signal_id = f"{os.path.splitext(pickle_file)[0]}_{count}"
            
            # Wyodrębnienie informacji o sygnale 
            signal_fs = single_signal['fs'] # częstotliwość próbkowania
            signal_data = single_signal['signal'] # dane sygnału
            signal_time_start = single_signal['time_start'] # czas początkowy

            print("-----------------------------------------------------------")
            print(f"Analizowanie sygnału {count} z pliku {pickle_file}:")
            print("- signal sampling frequency: " + str(signal_fs) + " Hz")
            print("- signal samples count N: " + str(len(signal_data)))
            print("- signal length: " + str(int(len(signal_data) / signal_fs)) + " s")
            print("- signal start time: " + str(signal_time_start))
            
            # Obliczenie oryginalnych znaczników czasowych
            print("Obliczanie znaczników czasowych.")
            current_timestamps = calculate_timestamps(signal_data, signal_time_start, signal_fs)
            
            # Interpolacja sygnału
            print("Interpolowanie sygnału.")
            target_timestamps, signal_interpolated = interpolate_signal(signal_data, current_timestamps, target_period)

            # Obliczenie klasycznego fft
            print("Obliczenie FFT.")
            signal_fft, signal_fft_freq = calculate_fft(signal_interpolated, target_ft)

            # Obliczenie okienkowego fft dla sygnałów dłuższych od jednego okna
            print("Obliczenie okienkowego FFT.")
            window_size = int(5 * 60 * target_ft) # okno 5 minutowe dla częstotliwości 100Hz
            step_size = window_size // 3
            if len(signal_data) > window_size:
                windowed_fft, window_ranges = calculate_windowed_fft(signal_data, window_size, step_size, target_ft)

                # Zapisz poszczególne okna jako obrazy
                print("Tworzenie wykresów dla okienkowego FFT.", end='', flush=True)
                for i in range(len(windowed_fft)):
                    print(".", end='', flush=True)
                    plot_combined(windowed_fft[i], target_ft, signal_interpolated, window_ranges[i], "Oryginal Signal 1 in time domain", "frequency", "value", False, f"./img/{signal_id}/window/{i}.png")

                # Stworzenie gif'a na podstawie stworzonych wykresów dla okienkowego FFT
                print("\nTworzenie GIF dla okienkowego FFT.")
                create_gif(f"./img/{signal_id}/window/", f"./resoult/{signal_id}/window_fft.gif", 0.1)

                # Obliczenie adaptacyjnego okienkowego fft
                print("Obliczenie adaptacyjnego okienkowego FFT.")
                window_min_size = int(window_size * 0.5)
                window_max_size = window_size
                step_size = window_min_size // 3
                adaptive_windowed_fft, adaptive_window_ranges = calculate_adaptive_windowed_fft(signal_data, window_min_size, window_max_size, window_max_size, target_ft)

                # Zapisz poszczególne okna jako obrazy
                print("Tworzenie wykresów dla adaptacyjnego okienkowego FFT.", end='', flush=True)
                for i in range(len(adaptive_windowed_fft)):
                    print(".", end='', flush=True)
                    plot_combined(adaptive_windowed_fft[i], target_ft, signal_interpolated, adaptive_window_ranges[i], "Oryginal Signal 1 in time domain", "frequency", "value", False, f"./img/{signal_id}/adaptive_window/{i}.png")
                # Stworzenie gif'a na podstawie stworzonych wykresów dla adaptacyjnego okienkowego FFT
                print("\nTworzenie GIF dla adaptacyjnego okienkowego FFT.")
                create_gif(f"./img/{signal_id}/adaptive_window/", f"./resoult/{signal_id}/adaptive_window_fft.gif", 0.1)
            else:
                print("Sygnał jest za krótki dla wybranego okna")

            # Rysujowanie oryginalnego sygnału
            plot_signal(current_timestamps, signal_data, "Oryginal Signal 1 in time domain", "time", "value", False, f"./resoult/{signal_id}/oryginal_signal.png")
            
            # Rysujowanie interpolowanego sygnału
            plot_signal(target_timestamps, signal_interpolated, "Interpolated Signal 1 in time domain", "time", "value", False, f"./resoult/{signal_id}/interpolated_signal.png")

            # Obliczam PSD
            psd, freq_psd = calculate_power_spectral_density(signal_interpolated, target_ft) # obliczam psd dla interpolowanego sygnału oraz częstotliwość z nim związaną
            plot_psd(psd, freq_psd, target_ft, [0, 0.06], [0, 40], "Wykres PSD dla intepolowanego sygnału", "Częstotliwość [Hz]", "PSD value", False, f"./resoult/{signal_id}/PSD.png")

            # Szukam indeksów w tablicy odpowiadającym początkowi i końcowi zakresu dla poniższych wartości określająctycych zakres
            psd_min = 0.005
            psd_max = 0.05
            psd_min_sample_number = where(freq_psd >= psd_min)[0][0]
            psd_max_sample_number = where(freq_psd <= psd_max)[0][0]
            trapz_integration_resoult = trapz(psd[psd_min_sample_number:psd_max_sample_number], freq_psd[psd_min_sample_number:psd_max_sample_number])
            print("Pole pod krzywą w zakresie [" + str(psd_min) + ", " + str(psd_max) + "] wynosi: " + str(trapz_integration_resoult), flush=True)

            # Uruchom analizę modelem 
            print("Uruchomienie analizy modelem ICPM Waveform Classification Plugin")
            WaveformClassificationPlugin(signal_data, target_timestamps, target_ft, f"./resoult/{signal_id}/", signal_id)
            
            # Jeśli analiza jest tylko dla jednego sygnału w pliku, kończymy
            if not for_all:
                break


# Uruchomienie głównej funkcji skryptu
if __name__ == "__main__":
    main()