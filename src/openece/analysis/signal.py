from __future__ import annotations
import numpy as np
from scipy import signal


def fft_spectrum(x, fs: float):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("x must be a 1-D signal with at least two samples")
    if fs <= 0:
        raise ValueError("fs must be positive")
    x0 = x - np.mean(x)
    window = np.hanning(len(x0))
    cg = window.mean()
    spec = np.fft.rfft(x0 * window)
    amp = np.abs(spec) / (len(x0) * cg)
    if len(amp) > 2:
        amp[1:-1] *= 2
    freq = np.fft.rfftfreq(len(x0), 1.0 / fs)
    return freq, amp


def dominant_frequency(x, fs: float) -> float:
    f, a = fft_spectrum(x, fs)
    if len(a) <= 1:
        return 0.0
    return float(f[1 + np.argmax(a[1:])])


def welch_psd(x, fs: float, nperseg: int | None = None):
    x = np.asarray(x, dtype=float)
    if fs <= 0:
        raise ValueError("fs must be positive")
    if nperseg is None:
        nperseg = min(1024, len(x))
    return signal.welch(x, fs=fs, nperseg=nperseg)
