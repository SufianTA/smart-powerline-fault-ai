import numpy as np

def spectral_energy(signal, fs=1024):
    # simple normalized spectral energy feature
    spec = np.fft.rfft(signal)
    return float((np.abs(spec)**2).sum() / len(spec))
