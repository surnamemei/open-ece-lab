import numpy as np
from openece.analysis.signal import dominant_frequency

def test_dominant_frequency():
    fs = 48000
    t = np.arange(fs) / fs
    x = np.sin(2*np.pi*1234*t)
    assert abs(dominant_frequency(x, fs) - 1234) < 1.0
