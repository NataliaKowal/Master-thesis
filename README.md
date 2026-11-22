# Master Thesis

This is my master thesis (in progress).

## Title

> [!NOTE]
> Joint analysis of slow waves and pulse waveform of intracranial pressure

# Project Description

This Python project processes signals stored in a .pkl file, interpolates them to a specified sampling frequency, and then displays both the time-domain signal and its frequency spectrum. The project uses the pickle, pathlib, matplotlib, and numpy libraries.

The file I worked on, "2aHc688_ICP.pkl," contains 172 signals and is too large to share on GitHub. To obtain the .pkl file, please contact the author of the work. Below are example results for signal number 100.

## Requirements

> [!IMPORTANT]
> You need to init and update git submodules

```sh
git submodule update --init --recursive
```

To use this code use anaconda environment with installed required packages.

Create new anaconda environment using Anaconda Prompt and commands
```sh
cd ICMPWaveformClassificationPlugin
conda env create -f environment.yml
```

Activate environment
```sh
conda activate ICMPlugin
```

Install required packages
```sh
pip install -r requirements-cpu.txt
```

To run scrip
```sh
python signal_processing.py
```
