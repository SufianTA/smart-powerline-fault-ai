import matplotlib.pyplot as plt
import numpy as np

def plot_signal(signal, title='Signal'):
    plt.figure()
    plt.plot(signal)
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
