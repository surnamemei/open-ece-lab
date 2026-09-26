from __future__ import annotations
import numpy as np


def bode_from_complex_ratio(frequency_hz, transfer):
    f = np.asarray(frequency_hz, dtype=float)
    h = np.asarray(transfer, dtype=complex)
    if f.ndim != 1 or h.ndim != 1 or len(f) != len(h):
        raise ValueError("frequency_hz and transfer must be equal-length 1-D arrays")
    if np.any(f <= 0):
        raise ValueError("frequencies must be positive")
    mag_db = 20 * np.log10(np.maximum(np.abs(h), np.finfo(float).tiny))
    phase_deg = np.unwrap(np.angle(h)) * 180 / np.pi
    return mag_db, phase_deg


def estimate_cutoff_hz(frequency_hz, mag_db, drop_db: float = 3.0):
    f = np.asarray(frequency_hz, dtype=float)
    m = np.asarray(mag_db, dtype=float)
    target = float(np.max(m) - drop_db)
    peak_idx = int(np.argmax(m))
    for i in range(peak_idx + 1, len(m)):
        if m[i] <= target:
            f1, f2 = f[i - 1], f[i]
            m1, m2 = m[i - 1], m[i]
            if m2 == m1:
                return float(f2)
            alpha = (target - m1) / (m2 - m1)
            return float(np.exp(np.log(f1) + alpha * (np.log(f2) - np.log(f1))))
    return float("nan")
