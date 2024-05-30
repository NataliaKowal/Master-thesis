import pickle
from pathlib import Path
import matplotlib.pyplot as plt

with open('data/2aHc688_ICP.pkl', 'rb') as f:
    data = pickle.load(f)

    plt.figure(0)

    plt.plot(data[120]['signal'])
    plt.show()