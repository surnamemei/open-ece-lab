from __future__ import annotations
import numpy as np


def _first_crossing(t, y, level):
    idx = np.flatnonzero(y >= level)
    return float(t[idx[0]]) if len(idx) else float("nan")


def step_metrics(t, y, reference: float | None = None, settling_band: float = 0.02):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or len(t) != len(y) or len(t) < 5:
        raise ValueError("t and y must be equal-length 1-D arrays with >=5 samples")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    tail = max(3, len(y) // 10)
    baseline = max(3, min(25, len(y) // 100))
    final = float(np.median(y[-tail:]))
    initial = float(np.median(y[:baseline]))
    span = final - initial
    if span <= 0:
        raise ValueError("current MVP expects a positive-going step response")
    y_norm = y - initial
    rise10 = _first_crossing(t, y_norm, 0.1 * span)
    rise90 = _first_crossing(t, y_norm, 0.9 * span)
    peak_idx = int(np.argmax(y))
    peak = float(y[peak_idx])
    overshoot = max(0.0, (peak - final) / abs(span) * 100.0)
    tol = settling_band * abs(span)
    outside = np.flatnonzero(np.abs(y - final) > tol)
    if not len(outside):
        settling_time = float(t[0])
    elif outside[-1] + 1 < len(t):
        settling_time = float(t[outside[-1] + 1])
    else:
        settling_time = float("nan")  # still outside the band at the end of the record: never settled
    result = {
        "initial": initial,
        "final": final,
        "rise_time_10_90_s": float(rise90 - rise10),
        "peak": peak,
        "peak_time_s": float(t[peak_idx]),
        "overshoot_percent": overshoot,
        "settling_time_s": settling_time,
    }
    if reference is not None:
        result["steady_state_error"] = float(reference - final)
        result["steady_state_error_percent"] = float((reference - final) / reference * 100.0) if reference != 0 else float("nan")
    return result
