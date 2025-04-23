import pandas as pd
import pickle
import matplotlib.pyplot as plt
from numpy import abs, arange, interp, ceil, int64, square, where, trapz, asarray, nanmean, argmax, unique, sum, nan
from numpy.fft import fft, fftfreq
from scipy.stats import pearsonr
from scipy.signal import detrend
import os
from tqdm import tqdm

from ICMPWaveformClassificationPlugin.plugin.pulse_detection.classifier_pipeline import ProcessingPipeline

def load_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except:
        return None


def calculate_timestamps(signal, time_start, sampling_frequency):
    time_length = len(signal)
    time = arange(time_length) / sampling_frequency
    time = time + time_start
    return time


def interpolate_signal(data, current_timestamps, target_period):
    target_timestamps = arange(current_timestamps[0], current_timestamps[-1], target_period)
    interpolated_signal = interp(target_timestamps, current_timestamps, data)
    return target_timestamps, interpolated_signal


def calculate_fft(signal, frequency):
    return fft(signal), fftfreq(len(signal), d=1 / frequency)


def calculate_power_spectral_density(signal, frequency):
    fft, freq = calculate_fft(signal, frequency)
    N = len(signal)
    psd = 1 / (N * frequency) * square(abs(fft))
    return psd, freq


def calculate_slow_wave_power(signal, frequency):
    signal = detrend(signal)
    psd, freq_psd = calculate_power_spectral_density(signal, frequency)

    psd_min = 0.005
    psd_max = 0.05
    mask = (freq_psd >= psd_min) & (freq_psd <= psd_max)
    f_band = freq_psd[mask]
    psd_band = psd[mask]

    slow_wave_power = trapz(psd_band, f_band)
    return slow_wave_power


def calculate_prx(icp_signal, abp_signal, frequency):
    # AK: obliczanie średniego ICP i średniego ABP w 10-sekundowych oknach
    window_size = int(10 * frequency)
    step_size = int(10 * frequency)

    signal_length = len(icp_signal)
    steps_number = int64(ceil(signal_length / step_size))
    mean_icp, mean_abp = [], []
    for i in range(steps_number):
        start_point = step_size * i
        end_point = start_point + window_size
        if end_point > signal_length - 1:
            end_point = signal_length - 1
        if start_point >= signal_length - 1:
            return

        mean_icp.append(nanmean(icp_signal[start_point:end_point]))
        mean_abp.append(nanmean(abp_signal[start_point:end_point]))

    mean_icp = asarray(mean_icp)
    mean_abp = asarray(mean_abp)

    # AK: obliczanie PRx jako współczynnika korelacji między średnim ICP i średnim ABP
    if len(mean_icp) > 2 and len(mean_abp) > 2:
        prx, _ = pearsonr(mean_icp, mean_abp)
        return prx
    else:
        return nan


def calculate_psi(signal, time):
    # AK: użycie modelu do klasyfikacji kształtów ICP
    pipeline = ProcessingPipeline()
    classes, times = pipeline.process_signal(signal, time)
    classification_results = argmax(classes, axis=1)

    non_artefact_mask = classification_results != 4
    non_artefact_classes = classification_results[non_artefact_mask]

    # AK: obliczenie PSI jako średniej ważonej numeru klasy i częstości jej występowania
    class_counts = unique(non_artefact_classes, return_counts=True)
    psi = 0
    if len(non_artefact_classes) > 0:
        sum_count = sum(class_counts[1])
        for c, count in zip(class_counts[0], class_counts[1]):
            psi += (c + 1) * count / sum_count
    return psi

def save_signals_to_csv(timestamps, icp_signal, abp_signal, filename):
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sygnaly_csv")
    
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Utworzono folder: {output_folder}")
            
        # Tworzenie pełnej ścieżki do pliku
        output_path = os.path.join(output_folder, f"{filename}_signals_raw.csv")
        
        # Określenie rozmiaru chunka (partii danych)
        chunk_size = 1000000  # 1 milion wierszy na chunk
        total_rows = len(timestamps)
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        # Zapisz nagłówki
        with open(output_path, 'w') as f:
            f.write('Timestamp,ICP_Signal,ABP_Signal\n')
        
        # Zapisuj dane partiami
        with tqdm(total=total_rows, desc="Zapisywanie surowych danych do CSV") as progress:
            for i in range(0, total_rows, chunk_size):
                end_idx = min(i + chunk_size, total_rows)
                chunk_data = {
                    'Timestamp': timestamps[i:end_idx],
                    'ICP_Signal': icp_signal[i:end_idx],
                    'ABP_Signal': abp_signal[i:end_idx]
                }
                chunk_df = pd.DataFrame(chunk_data)
                
                # Zapisz chunk do pliku (append mode)
                chunk_df.to_csv(output_path, mode='a', header=False, index=False)
                
                # Aktualizuj pasek postępu
                progress.update(end_idx - i)
        
        print(f"Pomyślnie zapisano sygnały do pliku: {output_path}")
        
    except Exception as e:
        print(f"Wystąpił błąd podczas zapisywania pliku CSV: {str(e)}")

def save_analyse_resoults_to_csv(window_timestamps, slow_wave_power, prx, psi, filename):
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sygnaly_csv")
    
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Utworzono folder: {output_folder}")
            
        # Tworzenie pełnej ścieżki do pliku
        output_path = os.path.join(output_folder, f"{filename}_signals_analyse.csv")
        
        # Określenie rozmiaru chunka (partii danych)
        chunk_size = 1000000  # 1 milion wierszy na chunk
        total_rows = len(window_timestamps)
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        # Zapisz nagłówki
        with open(output_path, 'w') as f:
            f.write('Window_Timestamp,Slow_Wave_Power,PRX,PSI\n')
        
        # Zapisuj dane partiami
        with tqdm(total=total_rows, desc="Zapisywanie wyników analizy do CSV") as progress:
            for i in range(0, total_rows, chunk_size):
                end_idx = min(i + chunk_size, total_rows)
                chunk_data = {
                    'Window_Timestamp': window_timestamps[i:end_idx],
                    'Slow_Wave_Power': slow_wave_power[i:end_idx],
                    'PRX': prx[i:end_idx],
                    'PSI': psi[i:end_idx]
                }
                chunk_df = pd.DataFrame(chunk_data)
                
                # Zapisz chunk do pliku (append mode)
                chunk_df.to_csv(output_path, mode='a', header=False, index=False)
                
                # Aktualizuj pasek postępu
                progress.update(end_idx - i)
        
        print(f"Pomyślnie zapisano sygnały do pliku: {output_path}")
        
    except Exception as e:
        print(f"Wystąpił błąd podczas zapisywania pliku CSV: {str(e)}")



def process_file(icp_file_path, abp_file_path): # AK: analiza pojedynczego pliku .pkl
    target_fs = 100.0
    target_period = 1 / target_fs

    # wczytywanie danych ICP i ABP z plików
    data_icp = load_data(icp_file_path)
    if data_icp is None:
        print(f"Nie znaleziono pliku z danymi ICP: {icp_file_path}")
        return

    data_abp = load_data(abp_file_path)
    if data_abp is None:
        print(f"Nie znaleziono pliku z danymi ABP: {abp_file_path}")
        return

    # AK: wczytywanie całych sygnałów oraz interpolacja
    full_timestamps_icp, full_signal_icp = [], []
    with tqdm(total=len(data_icp), desc="Przetwarzanie ICP") as progress_icp:
        for count, single_signal in enumerate(data_icp):
            signal_fs = single_signal['fs']
            signal_data = single_signal['signal']
            signal_time_start = single_signal['time_start']

            current_timestamps = calculate_timestamps(signal_data, signal_time_start, signal_fs)
            target_timestamps, signal_interpolated = interpolate_signal(signal_data, current_timestamps, target_period)
            full_timestamps_icp.extend(target_timestamps)
            full_signal_icp.extend(signal_interpolated)
            progress_icp.update(1)

    full_timestamps_abp, full_signal_abp = [], []
    with tqdm(total=len(data_abp), desc="Przetwarzanie ABP") as progress_abp:
        for count, single_signal in enumerate(data_abp):
            signal_fs = single_signal['fs']
            signal_data = single_signal['signal']
            signal_time_start = single_signal['time_start']

            current_timestamps = calculate_timestamps(signal_data, signal_time_start, signal_fs)
            target_timestamps, signal_interpolated = interpolate_signal(signal_data, current_timestamps, target_period)
            full_timestamps_abp.extend(target_timestamps)
            full_signal_abp.extend(signal_interpolated)
            progress_abp.update(1)

    full_timestamps_icp = asarray(full_timestamps_icp)
    full_signal_icp = asarray(full_signal_icp)
    full_signal_abp = asarray(full_signal_abp)

    # Zapisywanie sygnałów do CSV
    filename = os.path.splitext(os.path.basename(icp_file_path))[0]
    save_signals_to_csv(full_timestamps_icp, full_signal_icp, full_signal_abp, filename)

    # AK: przetwarzanie całego sygnału w oknach 5-minutowych
    window_size = int(5 * 60 * target_fs) # okno 5 minut
    step_size = int(10 * target_fs) # przesunięcie okna o 10 sekund
    full_window_timestamps, full_slow_wave_power, full_prx, full_psi = [], [], [], [] # tablice na wyniki
    if len(full_signal_icp) > window_size and len(full_signal_abp) > window_size:
        signal_length = len(full_signal_icp)
        steps_number = int64(ceil(signal_length / step_size))

        with tqdm(total=steps_number, desc="Przetwarzanie okien czasowych") as progress:
            for i in range(steps_number):
                progress.set_description(f"Okno {i+1}/{steps_number} - Przygotowywanie danych")
                start_point = step_size * i
                end_point = start_point + window_size
                if end_point > signal_length - 1:
                    end_point = signal_length - 1
                if start_point >= signal_length - 1:
                    break

                time_icp = full_timestamps_icp[start_point:end_point]
                data_icp = full_signal_icp[start_point:end_point]
                data_abp = full_signal_abp[start_point:end_point]

                # AK: punkt czasu przypisany danemu oknu
                window_timestamp = full_timestamps_icp[start_point]
                full_window_timestamps.append(window_timestamp)

                # AK: obliczanie mocy fal wolnych (pole pod krzywą PSD)
                progress.set_description(f"Okno {i+1}/{steps_number} - Obliczanie Slow Wave Power")
                slow_wave_power = calculate_slow_wave_power(data_icp, target_fs)
                full_slow_wave_power.append(slow_wave_power)

                # AK: obliczanie PRx
                progress.set_description(f"Okno {i+1}/{steps_number} - Obliczanie PRx")
                prx = calculate_prx(data_icp, data_abp, target_fs)
                full_prx.append(prx)

                # AK: obliczanie PSI
                progress.set_description(f"Okno {i+1}/{steps_number} - Obliczanie PSI")
                psi = calculate_psi(data_icp, time_icp)
                full_psi.append(psi)
                
                # Aktualizacja paska postępu
                progress.update(1)

            f, ax = plt.subplots(4, 1, sharex=True)
            ax[0].plot(full_timestamps_icp, full_signal_icp)
            ax[1].plot(full_window_timestamps, full_slow_wave_power)
            ax[2].plot(full_window_timestamps, full_prx)
            ax[3].plot(full_window_timestamps, full_psi)
            plt.show()

    
    # Zapis wyników analizy do pliku CSV
    save_analyse_resoults_to_csv(full_window_timestamps, full_slow_wave_power, full_prx, full_psi, filename)

def main():
    icp_data_folder = r'C:\Users\Nati\Desktop\DATA_TBI_NKowal\ICP'
    abp_data_folder = r'C:\Users\Nati\Desktop\DATA_TBI_NKowal\ABP'
    if not os.path.exists(icp_data_folder) or not os.path.exists(abp_data_folder):
        print(f"Folder {icp_data_folder} lub {abp_data_folder} nie istnieje.")
        exit()

    icp_pickle_files = [f for f in os.listdir(icp_data_folder) if f.endswith('.pkl')]

    if not icp_pickle_files:
        print(f"Brak plików pickle w folderze {icp_data_folder}.")
        exit()

    print(f"Znaleziono {len(icp_pickle_files)} plików pickle do analizy.")

    for file_index, icp_pickle_file in enumerate(icp_pickle_files):
        icp_file_path = os.path.join(icp_data_folder, icp_pickle_file)

        abp_pickle_file = icp_pickle_file.replace('ICP', 'ABP')
        abp_file_path = os.path.join(abp_data_folder, abp_pickle_file)

        print(f"\nAnalizowanie pliku {file_index + 1}/{len(icp_pickle_files)}: {icp_pickle_file}")
        process_file(icp_file_path, abp_file_path)


if __name__ == "__main__":
    main()