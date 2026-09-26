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


def waveform_statistics(x) -> dict[str, float]:
    """Time-domain summary of a signal, in the signal's own unit (crest factor is a ratio)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 1:
        raise ValueError("x must be a non-empty 1-D signal")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN or infinite values")
    x_min, x_max = float(np.min(x)), float(np.max(x))
    rms = float(np.sqrt(np.mean(x * x)))
    peak = max(abs(x_min), abs(x_max))
    return {
        "mean": float(np.mean(x)),
        "rms": rms,
        "ac_rms": float(np.std(x)),
        "minimum": x_min,
        "maximum": x_max,
        "peak_to_peak": x_max - x_min,
        "peak_abs": peak,
        "crest_factor": peak / rms if rms > 0 else float("nan"),
    }


def sampling_uniformity(t) -> tuple[float, int, float]:
    """Mean step of a time axis plus the index and relative deviation of its worst step.

    The mean step is (t[-1] - t[0]) / (N - 1). ``index`` is the sample that ends the step which
    deviates most from the mean step.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("t must be a 1-D array with at least two samples")
    if not np.all(np.isfinite(t)):
        raise ValueError("t contains NaN or infinite values")
    mean_step = (t[-1] - t[0]) / (len(t) - 1)
    if mean_step <= 0:
        raise ValueError("t must increase")
    deviation = np.abs(np.diff(t) - mean_step) / mean_step
    worst = int(np.argmax(deviation))
    return float(mean_step), worst + 1, float(deviation[worst])


def sample_rate_from_time(t, max_step_deviation: float = 0.1) -> float:
    """Sample rate (Hz) of a uniformly sampled time axis in seconds.

    Every step must be within ``max_step_deviation`` (relative) of the mean step. The default
    tolerates timestamps printed with limited precision but rejects missing, repeated or
    out-of-order samples.
    """
    mean_step, index, deviation = sampling_uniformity(t)
    if deviation > max_step_deviation:
        raise ValueError(
            f"time axis is not uniformly sampled: the step ending at sample {index} deviates "
            f"{deviation:.1%} from the mean step of {mean_step:.6g} s"
        )
    return 1.0 / mean_step
