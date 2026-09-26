"""Deterministic synthetic-data tests for the numerical helpers (new in v0.1 and Gate 0)."""
import math

import numpy as np
import pytest

from openece.analysis.control import step_metrics
from openece.analysis.electronics import (
    bode_from_complex_ratio, estimate_cutoff_hz, interpolate_at_frequency, magnitude_to_db, unwrap_phase_deg,
)
from openece.analysis.signal import (
    fft_spectrum, sample_rate_from_time, sampling_uniformity, waveform_statistics, welch_psd,
)
from openece.instruments.mock import MockOscilloscope, MockRCPlant, MockSignalGenerator
from openece.measurements.frequency_sweep import run_frequency_sweep


def rc_response(f, fc):
    return 1.0 / (1.0 + 1j * f / fc)


def test_waveform_statistics_of_a_sine_with_dc():
    fs = 10_000
    t = np.arange(fs) / fs  # exactly 50 periods of 50 Hz
    x = 0.2 + 1.5 * np.sin(2 * np.pi * 50 * t)
    s = waveform_statistics(x)
    assert s["mean"] == pytest.approx(0.2, abs=1e-12)
    assert s["ac_rms"] == pytest.approx(1.5 / math.sqrt(2), rel=1e-9)
    assert s["rms"] == pytest.approx(math.sqrt(0.2**2 + 1.5**2 / 2), rel=1e-9)
    assert (s["minimum"], s["maximum"]) == pytest.approx((-1.3, 1.7), abs=1e-9)
    assert s["peak_to_peak"] == pytest.approx(3.0, abs=1e-9)
    assert s["peak_abs"] == pytest.approx(1.7, abs=1e-9)
    assert s["crest_factor"] == pytest.approx(1.7 / s["rms"])


def test_waveform_statistics_rejects_non_finite_values():
    with pytest.raises(ValueError, match="NaN"):
        waveform_statistics([0.0, float("nan")])


def test_sample_rate_tolerates_printed_timestamps():
    t = np.round(np.arange(1000) / 48_000, 6)  # 6 decimals: steps alternate 20/21 us
    assert sample_rate_from_time(t) == pytest.approx(48_000, rel=1e-4)


def test_sample_rate_rejects_a_missing_sample():
    t = np.delete(np.arange(100) / 1000.0, 40)
    _, index, deviation = sampling_uniformity(t)
    assert index == 40 and deviation > 0.5
    with pytest.raises(ValueError, match="not uniformly sampled"):
        sample_rate_from_time(t)


def test_sampling_uniformity_rejects_decreasing_time():
    with pytest.raises(ValueError, match="increase"):
        sampling_uniformity([3.0, 2.0, 1.0])


def test_magnitude_to_db():
    np.testing.assert_allclose(magnitude_to_db([1.0, 10.0, 0.5]), [0.0, 20.0, -6.020599913279624])
    with pytest.raises(ValueError):
        magnitude_to_db([1.0, 0.0])


def test_unwrap_phase_deg():
    np.testing.assert_allclose(unwrap_phase_deg([170.0, -170.0, -150.0]), [170.0, 190.0, 210.0])


def test_interpolated_phase_at_rc_cutoff_is_minus_45_degrees():
    fc = 1591.5494309189535
    f = np.geomspace(100, 100e3, 121)
    phase = np.degrees(np.angle(rc_response(f, fc)))
    assert interpolate_at_frequency(f, phase, fc) == pytest.approx(-45.0, abs=0.05)
    assert math.isnan(interpolate_at_frequency(f, phase, 10.0))
    assert math.isnan(interpolate_at_frequency(f, phase, float("nan")))


def test_bode_from_complex_ratio_at_rc_cutoff():
    mag_db, phase_deg = bode_from_complex_ratio([1000.0], rc_response(np.array([1000.0]), 1000.0))
    assert mag_db[0] == pytest.approx(-3.0103, abs=1e-4)
    assert phase_deg[0] == pytest.approx(-45.0)


def test_estimate_cutoff_on_analytic_rc():
    f = np.geomspace(10, 1e6, 201)
    mag_db = 20 * np.log10(np.abs(rc_response(f, 4700.0)))
    # The cutoff is defined at exactly 3.000 dB below the maximum; a first-order RC reaches
    # -3.000 dB at fc * sqrt(10**0.3 - 1), i.e. 0.24 % below 1 / (2 pi R C).
    assert estimate_cutoff_hz(f, mag_db) == pytest.approx(4700.0 * math.sqrt(10**0.3 - 1), rel=5e-4)
    assert estimate_cutoff_hz(f, mag_db, drop_db=10 * math.log10(2)) == pytest.approx(4700.0, rel=5e-4)


def test_fft_spectrum_amplitude_of_an_on_bin_sine():
    fs, n = 1024.0, 1024
    x = 0.7 * np.sin(2 * np.pi * 64 * np.arange(n) / fs)
    f, amp = fft_spectrum(x, fs)
    k = int(np.argmax(amp))
    assert f[k] == 64.0
    assert amp[k] == pytest.approx(0.7, rel=1e-6)  # symmetric Hann window: tiny leakage


def test_welch_psd_integrates_to_the_variance():
    x = np.random.default_rng(1).normal(0.0, 0.3, 65536)
    f, psd = welch_psd(x, 1000.0)
    assert np.sum(psd) * (f[1] - f[0]) == pytest.approx(np.var(x), rel=0.05)


def test_step_metrics_second_order_overshoot():
    zeta, wn = 0.5, 10.0
    t = np.linspace(0, 5, 50001)
    wd = wn * math.sqrt(1 - zeta**2)
    y = 1 - np.exp(-zeta * wn * t) / math.sqrt(1 - zeta**2) * np.sin(wd * t + math.acos(zeta))
    m = step_metrics(t, y, reference=1.0)
    assert m["overshoot_percent"] == pytest.approx(100 * math.exp(-math.pi * zeta / math.sqrt(1 - zeta**2)), abs=0.05)
    assert m["peak_time_s"] == pytest.approx(math.pi / wd, abs=1e-3)
    assert m["steady_state_error"] == pytest.approx(0.0, abs=1e-6)


def test_step_metrics_rejects_negative_steps():
    t = np.linspace(0, 1, 100)
    with pytest.raises(ValueError, match="positive-going"):
        step_metrics(t, 1 - (1 - np.exp(-t / 0.1)))


def test_frequency_sweep_rejects_invalid_configuration():
    plant = MockRCPlant(noise_std=0.0)
    with pytest.raises(ValueError):
        run_frequency_sweep(MockSignalGenerator(), MockOscilloscope(plant), 1000, 100, 10)


def test_mock_instruments_declare_that_they_are_simulated():
    plant = MockRCPlant()
    for instrument in (MockSignalGenerator(), MockOscilloscope(plant)):
        info = instrument.describe()
        assert info.backend == "mock" and info.simulated is True
    assert plant.seed == 5305


def test_step_metrics_reports_nan_when_the_response_never_settles():
    t = np.linspace(-0.1, 1, 1101)  # 0.1 s of pre-step baseline, then a first-order step at t = 0
    y = np.where(t >= 0, 1 - np.exp(-np.clip(t, 0, None) / 0.05), 0.0)
    y[-1] += 0.1  # the final sample is outside the band: the record ends before settling
    assert math.isnan(step_metrics(t, y, settling_band=0.02)["settling_time_s"])
    y[-1] -= 0.1
    assert step_metrics(t, y, settling_band=0.02)["settling_time_s"] == pytest.approx(0.05 * math.log(50), abs=2e-3)


def test_unwrap_phase_deg_starts_in_the_principal_range():
    np.testing.assert_allclose(unwrap_phase_deg([356.4, 350.0, 315.0, 271.0]), [-3.6, -10.0, -45.0, -89.0])
    np.testing.assert_allclose(unwrap_phase_deg([-190.0, -200.0]), [170.0, 160.0])
    np.testing.assert_allclose(unwrap_phase_deg([180.0, 179.0]), [180.0, 179.0])
