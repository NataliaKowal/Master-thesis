# Master's Thesis: Joint Analysis of Slow Waves and Pulse Waveform of Intracranial Pressure

> **Status:** in progress

This repository contains the code for my master's thesis. It processes long recordings of **intracranial pressure (ICP)** and **arterial blood pressure (ABP)** from patients after traumatic brain injury and calculates three indices over time, so that slow waves and the pulse waveform of ICP can be analysed together.

## What the code calculates

The signals are analysed in **5-minute windows moved every 10 seconds**. For each window the script calculates:

| Index | What it describes | How it is calculated |
|---|---|---|
| **Slow wave power** | Strength of slow ICP oscillations | Area under the power spectral density of the detrended ICP in the 0.005–0.05 Hz band |
| **PRx** (pressure reactivity index) | Cerebrovascular pressure reactivity | Pearson correlation between 10-second mean values of ICP and ABP |
| **PSI** (pulse shape index) | Shape of the ICP pulse waveform | Weighted mean of pulse classes assigned by a neural network classifier, with artefacts excluded |

The pulse classification uses [ICMPWaveformClassificationPlugin](https://github.com/CMataczynski/ICMPWaveformClassificationPlugin), included as a git submodule.

## Processing pipeline

1. Load ICP and ABP recordings from `.pkl` files (each file contains many signal segments with their own sampling frequency and start time).
2. Interpolate every segment to a common sampling frequency of 100 Hz and join the segments into one continuous signal.
3. Save the raw, interpolated signals to CSV.
4. Calculate slow wave power, PRx and PSI in sliding windows.
5. Plot ICP together with the three indices and save the results to CSV.

## Repository structure

| File | Description |
|---|---|
| `signal_processing.py` | Main script: loading, interpolation, windowed analysis, plots and CSV export |
| `plugin_wrapper.py` | Helper for running the waveform classification plugin on a single signal, with interactive Plotly plots of the classes and PSI |
| `ICMPWaveformClassificationPlugin/` | Git submodule with the pulse waveform classifier |

## Getting started

### 1. Clone the repository with the submodule

```sh
git clone --recurse-submodules https://github.com/NataliaKowal/Master-thesis.git
```

If you have already cloned it without submodules:

```sh
git submodule update --init --recursive
```

### 2. Create the environment

The project uses an Anaconda environment defined in the plugin:

```sh
cd ICMPWaveformClassificationPlugin
conda env create -f environment.yml
conda activate ICMPlugin
pip install -r requirements-cpu.txt
cd ..
```

### 3. Set the data folders and run

Open `signal_processing.py` and set the ICP and ABP data folders in `main()` to where your `.pkl` files are stored. Then run:

```sh
python signal_processing.py
```

The results are saved to the `Sygnaly_csv/` folder:

- `<file>_signals_raw.csv`: interpolated ICP and ABP signals
- `<file>_signals_analyse.csv`: slow wave power, PRx and PSI for each window

## Data

The patient recordings are not included in this repository because of their size and because they are medical data. To get access to the data, please contact the author.

## Author

Natalia Kowal
