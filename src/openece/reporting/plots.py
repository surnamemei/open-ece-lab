"""Figures for run reports. Presentation only: every plotted quantity is computed elsewhere.

Figures use matplotlib's object-oriented API (``matplotlib.figure.Figure``) and never touch
pyplot's global state or an interactive backend, so they render headless on Windows and Linux.
matplotlib is imported lazily to keep CLI start-up fast.
"""
from __future__ import annotations

import io
import math

import numpy as np

_MAX_PLOT_POINTS = 20_000


def figure_to_png(fig, dpi: int = 110) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    return buffer.getvalue()


def bode_figure(frequency_hz, magnitude_db, phase_deg=None, *, cutoff_hz=math.nan, max_gain_db=math.nan,
                drop_db=3.0, magnitude_unit="dB", title=""):
    fig, axes = _figure(2 if phase_deg is not None else 1, row_height=2.9)
    ax = axes[0]
    ax.semilogx(frequency_hz, magnitude_db, marker=".", markersize=3, linewidth=1.2, label="magnitude")
    if math.isfinite(max_gain_db):
        ax.axhline(max_gain_db - drop_db, linestyle="--", linewidth=0.9, color="0.45", label=f"max - {drop_db:g} dB")
    if math.isfinite(cutoff_hz):
        for axis in axes:
            axis.axvline(cutoff_hz, linestyle=":", linewidth=1.3, color="C3")
        ax.plot([], [], linestyle=":", color="C3", label=f"cutoff {cutoff_hz:.4g} Hz")
    ax.set_ylabel(_plain(f"Magnitude ({magnitude_unit})"), parse_math=False)
    ax.legend(loc="best", fontsize="small")
    if phase_deg is not None:
        axes[1].semilogx(frequency_hz, phase_deg, marker=".", markersize=3, linewidth=1.2, color="C1")
        axes[1].set_ylabel("Phase (deg)")
    axes[-1].set_xlabel("Frequency (Hz)")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
    _title(fig, title)
    return fig


def step_figure(time_s, response, *, final, band_abs, peak, peak_time_s, settling_time_s,
                reference=math.nan, response_unit=None, settling_band=0.02, title=""):
    fig, (ax,) = _figure(1, row_height=3.8)
    ax.plot(time_s, response, linewidth=1.0, label="response")
    ax.axhline(final, linewidth=0.9, color="0.35", label="final value")
    if math.isfinite(band_abs):
        ax.axhspan(final - band_abs, final + band_abs, alpha=0.15, color="C2",
                   label=f"settling band (+/-{settling_band * 100:g} %)")
    if math.isfinite(reference):
        ax.axhline(reference, linestyle="--", linewidth=0.9, color="C2", label="reference")
    if math.isfinite(peak) and math.isfinite(peak_time_s):
        ax.plot([peak_time_s], [peak], marker="v", color="C3", linestyle="none", label="peak")
    if math.isfinite(settling_time_s):
        ax.axvline(settling_time_s, linestyle=":", color="C3", label=f"settled at {settling_time_s:.4g} s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(_plain(f"Response ({response_unit or 'unit not specified'})"), parse_math=False)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    _title(fig, title)
    return fig


def signal_figure(time_s, signal, frequency_hz, amplitude, psd_frequency_hz, psd, *,
                  dominant_hz=math.nan, unit=None, title=""):
    fig, axes = _figure(3, row_height=2.6)
    label = unit or "unit not specified"
    n = len(signal)
    if math.isfinite(dominant_hz) and dominant_hz > 0 and n > 1:
        fs = (n - 1) / (time_s[-1] - time_s[0])
        shown = int(np.clip(10 * fs / dominant_hz, 200, 5000))
    else:
        shown = 5000
    shown = min(shown, n)
    axes[0].plot(time_s[:shown], signal[:shown], linewidth=0.9)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(_plain(f"Signal ({label})"), parse_math=False)
    axes[0].set_title(f"Waveform (first {shown} of {n} samples)", fontsize="small")

    tiny = np.finfo(float).tiny
    f, a = _max_hold(np.asarray(frequency_hz)[1:], np.asarray(amplitude)[1:])
    axes[1].semilogx(f, 20 * np.log10(np.maximum(a, tiny)), linewidth=0.9)
    if math.isfinite(dominant_hz):
        axes[1].axvline(dominant_hz, linestyle=":", color="C3", label=f"dominant {dominant_hz:.6g} Hz")
        axes[1].legend(loc="best", fontsize="small")
    axes[1].set_ylabel(_plain(f"Amplitude (dB re 1 {label})"), parse_math=False)
    axes[1].set_xlabel("Frequency (Hz)")

    pf, pv = _max_hold(np.asarray(psd_frequency_hz)[1:], np.asarray(psd)[1:])
    axes[2].semilogx(pf, 10 * np.log10(np.maximum(pv, tiny)), linewidth=0.9, color="C1")
    axes[2].set_ylabel(_plain(f"PSD (dB re 1 {label}^2/Hz)"), parse_math=False)
    axes[2].set_xlabel("Frequency (Hz)")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
    _title(fig, title)
    return fig


def _figure(rows: int, row_height: float, width: float = 8.0):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, row_height * rows), layout="constrained")
    return fig, list(fig.subplots(rows, 1, squeeze=False)[:, 0])


def _title(fig, title: str) -> None:
    if title:
        fig.suptitle(_plain(title), parse_math=False)


def _plain(text: str) -> str:
    """Text from files or users: displayable as-is (no mathtext; undecodable file-name bytes escaped)."""
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _max_hold(x: np.ndarray, y: np.ndarray):
    """Thin long curves for display, keeping the maximum of each bucket so peaks stay visible."""
    if len(x) <= _MAX_PLOT_POINTS:
        return x, y
    k = int(math.ceil(len(x) / _MAX_PLOT_POINTS))
    m = (len(x) // k) * k
    return x[:m].reshape(-1, k)[:, 0], y[:m].reshape(-1, k).max(axis=1)
