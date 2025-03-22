# Import necessary system libraries
import sys
import os
from pathlib import Path

# Import data manipulation libraries
import numpy as np
import pickle as pkl
import pandas as pd
import datetime
import xlrd
import wfdb

# Import graphing libraries
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the plugin directory to the system path, for us it is in the plugin/pulse_detection directory
plugin_dir = Path(os.getcwd()) / 'ICMPWaveformClassificationPlugin' / 'plugin' / 'pulse_detection'
if str(plugin_dir) not in sys.path:
    sys.path.append(str(plugin_dir))

# Import the necessary plugin modules
from ICMPWaveformClassificationPlugin.plugin.pulse_detection.classifier_pipeline import ProcessingPipeline
from ICMPWaveformClassificationPlugin.plugin.pulse_detection.pulse_detector import Segmenter
from ICMPWaveformClassificationPlugin.plugin.pulse_detection.pulse_classifier import Classifier


def icmp_dateformat_to_unix_time(icmp_time_mark):
    datetime_date = xlrd.xldate_as_datetime(icmp_time_mark, 0)
    datetime_date = datetime_date + datetime.timedelta(hours=1)
    return datetime_date.timestamp()

def save_signal_to_pkl(data, file_path):
    """
    #### Summary
    Zapisuje dane do pliku pickle.

    #### Args:
     - data (dict): Dane do zapisania.
     - file_path (str): Ścieżka do pliku pickle.

    #### Returns:
     - bool: True jeśli zapis się powiódł, False w przeciwnym przypadku.
    """
    try:
        with open(file_path, 'wb') as f:
            pkl.dump(data, f)
        print(f"Dane zostały zapisane do pliku: {file_path}")
        return True
    except Exception as e:
        print(f"Błąd podczas zapisywania danych: {e}")
        return False


def WaveformClassificationPlugin(data, timestamps, signal_fs, folder_path, signal_id):
    # Przypisanie danych do zmiennych używanych w dalszej części kodu
    time = timestamps
    raw_signal = data

    # You can use below, commented code to cut the signal to a specific window, assuming the time vector is in seconds and starts from 0
    # window = 1*60*60
    # raw_signal = raw_signal[time <= window]
    # time = time[time <= window]

    # Create a plotly figure to display interactive signal
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time/3600, y=raw_signal, mode='lines'))
    fig.update_layout(title=f'ICP signal ({signal_id})', xaxis_title='Time [hr]', yaxis_title='ICP [mmHg]')
    fig.show()

    # Check the signal for nans
    nan_count = np.sum(np.isnan(raw_signal))
    print(f"Number of NaNs in the signal: {nan_count}")

    # Process the signal using the pipeline, remember to pass the time vector in seconds and the signal without any NaNs
    pipeline = ProcessingPipeline()
    classes, times = pipeline.process_signal(raw_signal, time)

    print(classes[0:2])

    classification_results = np.argmax(classes, axis=1)

    # Print class detection results
    unique_classes, counts = np.unique(classification_results, return_counts=True)
    for c, count in zip(unique_classes, counts):
        if c != 4:
            print(f"Class {c+1} detected {count} times")
        else:
            print(f"Artefacts detected {count} times")

    # Create a two-panel plotly figure to display the signal and the classification
    # Scatter the classification and give each class a different color

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=time/3600, y=raw_signal, mode='lines', name='ICP signal'), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.array(times)/3600, y=classification_results, mode='markers', marker=dict(color=classification_results, colorscale='Viridis', size=5), name='Classification'), row=2, col=1)
    fig.update_layout(title=f'ICP signal and classification ({signal_id})', xaxis_title='Time [hr]')
    fig.update_yaxes(title_text='ICP [mmHg]', row=1, col=1)
    fig.update_yaxes(title_text='Class', row=2, col=1)
    fig.show()

    def save_results(raw_signal, time, signal_fs, classification_results, classification_times, psi_vector, psi_times, signal_id):
        try:
            # Sprawdź czy folder istnieje, jeśli nie - utwórz go
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Utworzono folder: {folder_path}")
            
            # Generuj nazwę pliku na podstawie signal_id
            file_name = f"wyniki_analizy_{signal_id}.pkl"
            
            # Pełna ścieżka do pliku
            output_file_path = os.path.join(folder_path, file_name)
            
            print(f"Zapisuję dane do pliku: {output_file_path}")
            
            # Przygotowanie danych do zapisu
            results_data = {
                'raw_signal': raw_signal,
                'time': time,
                'signal_fs': signal_fs,
                'classification_results': classification_results,
                'classification_times': classification_times,
                'psi_vector': psi_vector,
                'psi_times': psi_times,
                'original_file': signal_id,
                'processing_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Zapisz dane do pliku
            with open(output_file_path, 'wb') as f:
                pkl.dump(results_data, f)
            
            # Sprawdź czy plik został utworzony
            if os.path.exists(output_file_path):
                print(f"Dane zostały pomyślnie zapisane do pliku: {output_file_path}")
                print(f"Rozmiar pliku: {os.path.getsize(output_file_path)} bajtów")
            else:
                print("Plik nie został utworzony mimo braku błędów!")
                
        except Exception as e:
            print(f"Wystąpił błąd podczas zapisywania danych: {e}")
            import traceback
            traceback.print_exc()

    # # Wywołanie funkcji zapisu
    # save_results(raw_signal, time, signal_fs, classification_results, times, psi_vector, psi_times, signal_id)
    
    # Remove the artefact class from the classification results
    non_artefact_mask = classification_results != 4
    non_artefact_classes = classification_results[non_artefact_mask]
    non_artefact_times = np.array(times)[non_artefact_mask]

    # Use rolling window to calculate PSI
    window_length = 5 * 60
    window_step = 10
    starting_time = non_artefact_times[0]

    psi_vector = []
    psi_times = []

    for win_start in np.arange(starting_time, non_artefact_times[-1] - window_length, window_step):
        # Get the classes in the time window
        win_end = win_start + window_length
        win_mask = (non_artefact_times >= win_start) & (non_artefact_times < win_end)
        win_classes = non_artefact_classes[win_mask]

        # Calculate the PSI
        class_counts = np.unique(win_classes, return_counts=True)
        psi = 0
        if len(win_classes) > 0:
            sum_count = np.sum(class_counts[1])
            for c, count in zip(class_counts[0], class_counts[1]):
                psi += (c+1) * count / sum_count

        # Append the PSI to the vector
        psi_vector.append(psi)
        psi_times.append(win_start + window_length / 2)


    # Plot the ICP and PSI vectors
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=time/3600, y=raw_signal, mode='lines', name='ICP signal'), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.array(psi_times)/3600, y=psi_vector, mode='lines', name='PSI'), row=2, col=1)
    fig.update_layout(title=f'ICP signal and PSI ({signal_id})', xaxis_title='Time [hr]')
    fig.update_yaxes(title_text='ICP [mmHg]', row=1, col=1)
    fig.update_yaxes(title_text='PSI', row=2, col=1)
    fig.show()

    # Zapisz dane
    save_results(raw_signal, time, signal_fs, classification_results, times, psi_vector, psi_times, signal_id)