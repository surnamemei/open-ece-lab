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


def magnitude_to_db(magnitude):
    """20*log10 of a linear magnitude (ratio or amplitude); values must be finite and > 0."""
    m = np.asarray(magnitude, dtype=float)
    if not np.all(np.isfinite(m)) or np.any(m <= 0):
        raise ValueError("linear magnitudes must be finite and greater than zero")
    return 20 * np.log10(m)


def unwrap_phase_deg(phase_deg):
    """Remove 360-degree jumps from a phase curve ordered by frequency.

    The curve is then shifted by a whole number of turns so that its first point lies in
    (-180, 180], making the result independent of the instrument's phase convention
    (e.g. 0..360 degrees).
    """
    p = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(phase_deg, dtype=float))))
    if p.size:
        p = p - 360.0 * np.ceil((p[0] - 180.0) / 360.0)
    return p


def interpolate_at_frequency(frequency_hz, values, target_hz: float) -> float:
    """Interpolate ``values`` linearly in log10(frequency) at ``target_hz``.

    Returns NaN when ``target_hz`` is not finite or lies outside the measured range.
    """
    f = np.asarray(frequency_hz, dtype=float)
    v = np.asarray(values, dtype=float)
    if f.ndim != 1 or v.ndim != 1 or len(f) != len(v) or len(f) < 2:
        raise ValueError("frequency_hz and values must be equal-length 1-D arrays with >=2 samples")
    if np.any(f <= 0) or np.any(np.diff(f) <= 0):
        raise ValueError("frequencies must be positive and strictly increasing")
    if not np.isfinite(target_hz) or target_hz < f[0] or target_hz > f[-1]:
        return float("nan")
    return float(np.interp(np.log10(target_hz), np.log10(f), v))
