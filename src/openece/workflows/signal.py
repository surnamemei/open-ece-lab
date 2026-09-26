"""Signal analysis (level statistics, dominant frequency, spectrum, PSD) of a sampled waveform."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ..analysis.signal import dominant_frequency, fft_spectrum, sampling_uniformity, waveform_statistics, welch_psd
from ..errors import AnalysisError
from ..io import ColumnRole, Dataset, resolve_columns
from ..io.columns import TIME_NAMES
from .common import column_values, file_source, integer, mapping_dict, number, time_seconds
from .outcome import AnalysisOutcome, DataTable, Measurement

ANALYSIS = "signal"
PARAMETERS = ("nperseg",)
MIN_SAMPLES = 16

_STATISTICS = {
    "mean": "mean (DC) value",
    "rms": "RMS value including DC",
    "ac_rms": "RMS value with the mean removed (standard deviation)",
    "minimum": "minimum value",
    "maximum": "maximum value",
    "peak_to_peak": "maximum - minimum",
    "peak_abs": "largest absolute value",
}


def analyze_signal(
    dataset: Dataset,
    *,
    columns: Mapping[str, str | None] | None = None,
    time_unit: str | None = None,
    sample_rate_hz: float | None = None,
    nperseg: int | None = None,
    max_step_deviation: float = 0.1,
    title: str | None = None,
) -> AnalysisOutcome:
    """Analyse one uniformly sampled channel.

    The sample rate comes from ``sample_rate_hz`` if given, else from the file (WAV), else from a
    uniformly sampled time column. With several candidate channels, the ``signal`` column must be
    chosen explicitly (``columns={"signal": "ch2"}``).
    """
    if sample_rate_hz is not None:
        sample_rate_hz = number(sample_rate_hz, "sample_rate_hz", positive=True)
    max_step_deviation = number(max_step_deviation, "max_step_deviation", positive=True)
    fs_known = sample_rate_hz is not None or dataset.sample_rate_hz is not None
    roles = (
        ColumnRole("time", "time", TIME_NAMES, required=not fs_known, accepts_datetime=True),
        ColumnRole("signal", "signal", by_elimination=True),
    )
    mapping = resolve_columns(dataset, roles, columns)
    warnings = list(dataset.warnings)
    where = dataset.source.path
    x = column_values(dataset, mapping["signal"], "signal")
    if len(x) < MIN_SAMPLES:
        raise AnalysisError(f"{where}: signal analysis needs at least {MIN_SAMPLES} samples, got {len(x)}")

    t = None
    if sample_rate_hz is not None:
        fs = sample_rate_hz
        time_info = {"source": "explicit sample rate", "sample_rate_hz": fs}
        if dataset.sample_rate_hz is not None and dataset.sample_rate_hz != fs:
            warnings.append(f"the file's sample rate of {dataset.sample_rate_hz:g} Hz was overridden with {fs:g} Hz")
        if "time" in mapping:
            warnings.append(f"time column {mapping['time'].column.name!r} was ignored because a sample rate was given")
    elif dataset.sample_rate_hz is not None and "time" not in mapping:
        fs = dataset.sample_rate_hz
        time_info = {"source": "file sample rate", "sample_rate_hz": fs}
    else:
        t, time_info = time_seconds(dataset, mapping["time"], time_unit, warnings)
        try:
            mean_step, index, deviation = sampling_uniformity(t)
        except ValueError as exc:
            raise AnalysisError(f"{where}: time column {mapping['time'].column.name!r}: {exc}") from exc
        if deviation > max_step_deviation:
            raise AnalysisError(
                f"{where}: time column {mapping['time'].column.name!r} is not uniformly sampled: the step ending at "
                f"{dataset.locate(index)} deviates {deviation:.1%} from the mean step of {mean_step:.6g} s. "
                "Resample the data, or give the sample rate explicitly to ignore the time column.",
                hint="sample_rate",
            )
        fs = 1.0 / mean_step
        time_info.update({"sample_rate_hz": fs, "max_step_deviation": deviation})

    n = len(x)
    nperseg = min(1024, n) if nperseg is None else integer(nperseg, "nperseg", minimum=8)
    if nperseg > n:
        raise AnalysisError(f"nperseg ({nperseg}) cannot exceed the number of samples ({n})", hint="nperseg")

    stats = waveform_statistics(x)
    no_ac = stats["ac_rms"] <= 1e-12 * stats["peak_abs"]  # constant up to floating-point residue
    f0 = math.nan if no_ac else dominant_frequency(x, fs)
    freq, amplitude = fft_spectrum(x, fs)
    psd_freq, psd = welch_psd(x, fs, nperseg=nperseg)
    unit = mapping["signal"].column.unit
    results = {
        "dominant_frequency_hz": Measurement(
            f0, "Hz", "frequency of the largest spectral peak (Hann-windowed FFT, DC excluded)",
            note="not available: the signal is constant (no AC content)" if no_ac else None),
        "frequency_resolution_hz": Measurement(
            fs / n, "Hz", "FFT bin spacing (the dominant frequency is quantised to this)"),
        **{name: Measurement(stats[name], unit, text) for name, text in _STATISTICS.items()},
        "crest_factor": Measurement(stats["crest_factor"], "1", "peak_abs / rms"),
    }
    title = title or f"Signal analysis of {dataset.source.path}"
    time_axis = t if t is not None else np.arange(n) / fs
    psd_unit = f"{unit}^2/Hz" if unit else "(signal unit)^2/Hz"
    return AnalysisOutcome(
        analysis=ANALYSIS,
        title=title,
        source=file_source(dataset),
        parameters={
            "sample_rate_hz": fs, "samples": n, "duration_s": n / fs, "nperseg": nperseg,
            "columns": mapping_dict(mapping), "time": time_info,
        },
        results=results,
        warnings=tuple(warnings),
        arrays={"time_s": time_axis, "signal": x, "frequency_hz": freq, "amplitude": amplitude,
                "psd_frequency_hz": psd_freq, "psd": psd},
        tables={"psd.csv": DataTable("Welch power spectral density", {"frequency (Hz)": psd_freq, f"psd ({psd_unit})": psd})},
        figures=_figures(time_axis, x, freq, amplitude, psd_freq, psd, f0, unit, title),
    )


def _figures(t, x, freq, amplitude, psd_freq, psd, f0, unit, title):
    def build():
        from ..reporting.plots import signal_figure

        figure = signal_figure(t, x, freq, amplitude, psd_freq, psd, dominant_hz=f0, unit=unit, title=title)
        return {"signal.png": ("waveform, amplitude spectrum and power spectral density", figure)}

    return build
