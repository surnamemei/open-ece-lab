from __future__ import annotations
import numpy as np
from openece.analysis.electronics import bode_from_complex_ratio, estimate_cutoff_hz


def run_frequency_sweep(generator, scope, start_hz: float, stop_hz: float, points: int, amplitude_vpk: float = 1.0):
    if start_hz <= 0 or stop_hz <= start_hz or points < 3:
        raise ValueError("invalid sweep configuration")
    f = np.geomspace(start_hz, stop_hz, points)
    h = []
    for freq in f:
        generator.set_sine(float(freq), amplitude_vpk)
        h.append(scope.measure_transfer(float(freq)))
    h = np.asarray(h)
    mag_db, phase_deg = bode_from_complex_ratio(f, h)
    cutoff = estimate_cutoff_hz(f, mag_db)
    return {"frequency_hz": f, "transfer": h, "magnitude_db": mag_db, "phase_deg": phase_deg, "cutoff_hz": cutoff}
