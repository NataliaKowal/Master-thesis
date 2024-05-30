import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from numpy import abs, arange, interp
from numpy.fft import fft

target_ft = 100.0               # desired sampling frequency in Hz
target_period = 1 / target_ft   # desired sampling period

with open('data/2aHc688_ICP.pkl', 'rb') as f:
    data = pickle.load(f)

    signal_1_data = data[1]['signal']                   # signal data
    signal_1_fs = data[1]['fs']                         # sampling frequency
    signal_1_invalid_flag = data[1]['invalid_flag']     # invalid flag
    signal_1_time_start = data[1]['time_start']         # time start

    period = 1 / signal_1_fs                                                                # calculate period based on sampling frequency
    current_timestamps = arange(0, len(signal_1_data) * period, period)                     # create array with timestamps for signal_1_data
    target_timestamps = arange(0, len(signal_1_data) * period, target_period)               # create array with interpolated timestamps down to 100Hz for signal_1_data
    signal_1_interpolated = interp(target_timestamps, current_timestamps, signal_1_data)    # interpolate signals over desired timestamps
    signal_1_interpolated_fft = fft(signal_1_interpolated)                                  # make fft of desired timestamps

    plt.figure(figsize = (8, 6))                    # make figure with aspect ratio 8, 6
    plt.plot(current_timestamps, signal_1_data)     # plot oryginal data        
    plt.title("Signal 1 in time domain")            # add title
    plt.xlabel("time")                              # add x axis label
    plt.ylabel("value")                             # add y axis label

    plt.figure(figsize = (8, 6))                            # make figure with aspect ratio 8, 6
    plt.plot(target_timestamps, signal_1_interpolated)      # plot interpolted data          
    plt.title("Interpolated Signal 1 in time domain")       # add title
    plt.xlabel("time")                                      # add x axis label
    plt.ylabel("value")                                     # add y axis label

    plt.figure(figsize = (8, 6))                            # make figure with aspect ratio 8, 6
    plt.plot(abs(signal_1_interpolated_fft))                # plot fft - fft operation resoult are complex numbers, so to plot them we must take absolute value of them
    plt.title("Interpolated Signal 1 in frequency domain")  # add title
    plt.xlabel("frequency")                                 # add x axis label
    plt.ylabel("value")                                     # add y axis label

    plt.show()      # show all plots