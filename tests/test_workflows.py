"""Workflow behaviour on synthetic data with known answers (units, ordering, failure modes)."""
import math

import numpy as np
import pytest

from openece.errors import AmbiguousColumnError, AnalysisError, ConfigurationError, MissingColumnError
from openece.io import load_csv
from openece.validation import Requirement
from openece.workflows import analyze_bode, analyze_signal, analyze_step, apply_requirements

EXACT_3DB = math.sqrt(10**0.3 - 1)  # where a first-order low-pass is exactly 3.000 dB down, relative to fc


def rc_rows(f_hz, fc):
    h = 1.0 / (1.0 + 1j * f_hz / fc)
    return np.abs(h), np.angle(h)


def csv_text(header, columns, fmt="{:.10g}"):
    rows = [",".join(fmt.format(v) for v in row) for row in zip(*columns)]
    return header + "\n" + "\n".join(rows) + "\n"


def test_bode_converts_khz_linear_magnitude_and_radians(write_text):
    f_hz = np.geomspace(10, 100e3, 121)
    mag, phase = rc_rows(f_hz, 1000.0)
    ds = load_csv(write_text("sweep.csv", csv_text("freq (kHz),gain (V/V),phase (rad)", [f_hz / 1e3, mag, phase])))
    outcome = analyze_bode(ds)
    r = outcome.results
    assert r["cutoff_hz"].value == pytest.approx(1000.0 * EXACT_3DB, rel=1e-3)
    assert r["cutoff_hz"].unit == "Hz"
    assert r["phase_at_cutoff_deg"].value == pytest.approx(-math.degrees(math.atan(EXACT_3DB)), abs=0.05)
    assert r["max_gain_db"].value == pytest.approx(0.0, abs=1e-3) and r["max_gain_db"].unit == "dB"
    units = outcome.parameters["units"]
    assert units["frequency"] == {"unit_used": "kHz", "scale_to_hz": 1000.0}
    assert units["magnitude"]["interpreted_as"] == "linear"
    assert units["phase"]["scale_to_deg"] == pytest.approx(180 / math.pi)
    assert list(outcome.tables["bode_data.csv"].columns) == ["frequency (Hz)", "magnitude (dB)", "phase (deg)"]


def test_bode_descending_sweep_is_reversed_with_a_warning(write_text):
    f_hz = np.geomspace(100e3, 10, 81)
    mag, _ = rc_rows(f_hz, 2000.0)
    ds = load_csv(write_text("down.csv", csv_text("frequency_hz,magnitude_db", [f_hz, 20 * np.log10(mag)])))
    outcome = analyze_bode(ds)
    assert outcome.results["cutoff_hz"].value == pytest.approx(2000.0 * EXACT_3DB, rel=2e-3)
    assert any("reversed" in w for w in outcome.warnings)


@pytest.mark.parametrize("text, message", [
    ("f (Hz),m (dB)\n10,0\n20,-1\n20,-2\n40,-3\n", "not monotonic.*line 4"),
    ("f (Hz),m (dB)\n0,0\n20,-1\n40,-3\n", "non-positive.*line 2"),
    ("f (Hz),m (V/V)\n10,1\n20,-0.5\n40,0.2\n", "value <= 0 at line 3"),
    ("f (Hz),m (dB)\n10,0\n20,-1\n", "at least 3"),
])
def test_bode_input_errors_point_at_the_file(write_text, text, message):
    with pytest.raises(AnalysisError, match=message):
        analyze_bode(load_csv(write_text("bad.csv", text)))


def test_bode_units_are_never_guessed_for_magnitude_or_phase(write_text):
    ds = load_csv(write_text("nounits.csv", "freq,gain,phase\n10,1,-1\n100,0.9,-10\n1000,0.5,-60\n10000,0.1,-85\n"))
    with pytest.raises(AnalysisError, match="dB or linear") as err:
        analyze_bode(ds)
    assert err.value.hint == "magnitude_unit"
    with pytest.raises(AnalysisError, match="phase column 'phase' has no unit") as err:
        analyze_bode(ds, magnitude_unit="linear")
    assert err.value.hint == "phase_unit"
    outcome = analyze_bode(ds, magnitude_unit="linear", phase_unit="deg")
    assert any("assuming Hz" in w for w in outcome.warnings)  # frequency defaults to Hz, visibly


def test_bode_without_a_crossing_warns_and_fails_its_requirement(write_text):
    f_hz = np.geomspace(10, 1000, 41)
    high_pass = f_hz / np.sqrt(f_hz**2 + 100.0**2)
    ds = load_csv(write_text("hp.csv", csv_text("f (Hz),m (V/V)", [f_hz, high_pass])))
    outcome = apply_requirements(analyze_bode(ds), [Requirement("cutoff_hz", 1, 1e6)])
    assert math.isnan(outcome.results["cutoff_hz"].value)
    assert any("highest frequency" in w for w in outcome.warnings)
    assert outcome.overall == "FAIL" and "not determined" in outcome.checks[0].detail


def test_bode_rejects_time_domain_recordings(tmp_path):
    from scipy.io import wavfile
    wavfile.write(tmp_path / "x.wav", 8000, np.zeros(100, dtype=np.int16))
    from openece.io import load_wav
    with pytest.raises(AnalysisError, match="time-domain"):
        analyze_bode(load_wav(tmp_path / "x.wav"))


def first_order_step(t, tau=0.2):
    return np.where(t >= 0, 1 - np.exp(-np.clip(t, 0, None) / tau), 0.0)


def test_step_time_in_milliseconds_is_converted(write_text):
    t = np.linspace(-0.1, 2.0, 2101)
    ds = load_csv(write_text("ms.csv", csv_text("time (ms),y (V)", [t * 1000, first_order_step(t)])))
    outcome = analyze_step(ds)
    assert outcome.results["rise_time_10_90_s"].value == pytest.approx(2.1972 * 0.2, abs=2e-3)
    assert outcome.results["final"].unit == "V"
    assert outcome.parameters["time"]["scale_to_s"] == 1e-3
    assert math.isnan(outcome.results["steady_state_error"].value)
    assert "no reference" in outcome.results["steady_state_error"].note


def test_step_from_a_single_column_and_a_sample_rate(write_text):
    t = np.arange(0, 2000) / 1000.0
    ds = load_csv(write_text("y.csv", csv_text("y", [first_order_step(t - 0.1)])))
    with pytest.raises(MissingColumnError):
        analyze_step(ds)
    outcome = analyze_step(ds, sample_rate_hz=1000.0, step_time_s=0.1, reference=1.0)
    assert outcome.results["rise_time_10_90_s"].value == pytest.approx(0.439, abs=3e-3)
    assert outcome.results["steady_state_error"].value == pytest.approx(0.0, abs=1e-3)
    assert outcome.parameters["time"]["source"] == "explicit sample rate"


def test_step_time_shifts_reported_times(write_text):
    t = np.linspace(0, 3, 3001)
    ds = load_csv(write_text("late.csv", csv_text("t,y", [t, first_order_step(t - 1.0)])))
    base = analyze_step(ds).results["settling_time_s"].value
    shifted = analyze_step(ds, step_time_s=1.0).results["settling_time_s"].value
    assert base - shifted == pytest.approx(1.0)


@pytest.mark.parametrize("text, message", [
    ("t,y\n0,0\n1,0.5\n1,0.8\n2,0.9\n3,1\n", "not strictly increasing at line 4"),
    ("t,y\n0,0\n1,0.5\n2,\n3,0.9\n4,1\n", "blank value.*line 4"),
    ("t,y\n0,1\n1,0.8\n2,0.5\n3,0.2\n4,0\n5,0\n", "positive-going"),
])
def test_step_input_errors(write_text, text, message):
    with pytest.raises(AnalysisError, match=message):
        analyze_step(load_csv(write_text("bad.csv", text)))


def test_step_parameter_validation(write_text):
    ds = load_csv(write_text("ok.csv", "t,y\n0,0\n1,1\n2,1\n3,1\n4,1\n"))
    for kwargs in ({"settling_band": 0.0}, {"settling_band": 1.5}, {"reference": "abc"}, {"sample_rate_hz": -5}):
        with pytest.raises(ConfigurationError):
            analyze_step(ds, **kwargs)


def test_step_accepts_iso_timestamps(write_text):
    stamps = [f"2026-09-26T10:00:{s:06.3f}" for s in np.arange(0, 3, 0.01)]
    y = first_order_step(np.arange(0, 3, 0.01) - 0.5, tau=0.2)
    text = "timestamp,y\n" + "".join(f"{s},{v:.6f}\n" for s, v in zip(stamps, y))
    outcome = analyze_step(load_csv(write_text("log.csv", text)), step_time_s=0.5)
    assert outcome.results["rise_time_10_90_s"].value == pytest.approx(0.44, abs=0.011)
    assert "ISO 8601" in outcome.parameters["time"]["note"]


def test_signal_from_csv_uses_the_time_column(write_text):
    fs = 20_000
    t = np.arange(4000) / fs
    ds = load_csv(write_text("sig.csv", csv_text("time (s),v (V)", [t, 0.3 * np.sin(2 * np.pi * 1000 * t)])))
    outcome = analyze_signal(ds)
    assert outcome.parameters["sample_rate_hz"] == pytest.approx(fs, rel=1e-9)
    assert outcome.results["dominant_frequency_hz"].value == pytest.approx(1000.0, abs=5.0)
    assert outcome.results["ac_rms"].value == pytest.approx(0.3 / math.sqrt(2), rel=1e-3)
    assert outcome.results["ac_rms"].unit == "V" and outcome.results["crest_factor"].unit == "1"


def test_signal_rejects_non_uniform_time_with_the_line(write_text):
    t = np.delete(np.arange(200) / 1000.0, 50)
    ds = load_csv(write_text("gap.csv", csv_text("t,v", [t, np.sin(t)])))
    with pytest.raises(AnalysisError, match="not uniformly sampled.*line 52") as err:
        analyze_signal(ds)
    assert err.value.hint == "sample_rate"
    assert analyze_signal(ds, sample_rate_hz=1000.0).warnings  # the ignored time column is reported


def test_signal_parameter_validation(write_text):
    ds = load_csv(write_text("s.csv", csv_text("v", [np.sin(np.arange(64))])))
    with pytest.raises(AnalysisError, match="nperseg"):
        analyze_signal(ds, sample_rate_hz=100.0, nperseg=128)
    with pytest.raises(ConfigurationError):
        analyze_signal(ds, sample_rate_hz=100.0, nperseg=4)
    with pytest.raises(AnalysisError, match="at least 16"):
        analyze_signal(load_csv(write_text("short.csv", "v\n1\n2\n3\n")), sample_rate_hz=10.0)


def test_a_step_that_never_settles_fails_its_requirement(write_text):
    t = np.linspace(-0.1, 1, 1101)
    y = first_order_step(t, tau=0.05)
    y[-1] += 0.2
    outcome = apply_requirements(analyze_step(load_csv(write_text("u.csv", csv_text("time (s),y (V)", [t, y])))),
                                 [Requirement("settling_time_s", None, 0.4)])
    settling = outcome.results["settling_time_s"]
    assert math.isnan(settling.value) and "did not settle" in settling.note
    assert outcome.overall == "FAIL"


def test_phase_in_the_0_to_360_convention(write_text):
    f_hz = np.geomspace(10, 100e3, 81)
    mag, phase = rc_rows(f_hz, 1000.0)
    ds = load_csv(write_text("p360.csv", csv_text("f (Hz),m (V/V),phase (deg)", [f_hz, mag, np.degrees(phase) % 360])))
    phase_at_cutoff = analyze_bode(ds).results["phase_at_cutoff_deg"].value
    assert phase_at_cutoff == pytest.approx(-math.degrees(math.atan(EXACT_3DB)), abs=0.1)


def test_non_monotonic_sweep_reports_the_offending_line(write_text):
    with pytest.raises(AnalysisError, match="line 6"):
        analyze_bode(load_csv(write_text("w.csv", "f (Hz),m (dB)\n100,0\n200,-1\n400,-3\n800,-7\n100,-12\n")))


def test_unusable_phase_column_is_reported_but_does_not_block(write_text):
    ds = load_csv(write_text("t.csv", "f (Hz),m (dB),phase\n10,0,-1°\n100,-0.5,-8°\n1000,-3,-45°\n"
                                      "10000,-20,-84°\n"))
    outcome = analyze_bode(ds)
    assert outcome.results["cutoff_hz"].value == pytest.approx(1000.0, rel=1e-3)
    assert math.isnan(outcome.results["phase_at_cutoff_deg"].value)
    assert any("looks like phase data but cannot be used" in w for w in outcome.warnings)


def test_a_constant_signal_has_no_dominant_frequency(write_text):
    outcome = analyze_signal(load_csv(write_text("dc.csv", csv_text("v (V)", [np.full(256, 1.25)]))), sample_rate_hz=1000.0)
    result = outcome.results["dominant_frequency_hz"]
    assert math.isnan(result.value) and "constant" in result.note


def test_annotations_are_not_units(write_text):
    t = np.arange(64) / 1000.0
    ds = load_csv(write_text("ch.csv", csv_text("time (s),Voltage (CH1)", [t, np.sin(2 * np.pi * 125 * t)])))
    assert analyze_signal(ds).results["rms"].unit is None  # reported as 'not specified', never 'CH1'
    samples = load_csv(write_text("n.csv", "Time (samples),y\n0,0\n1,0.5\n2,0.8\n3,1\n4,1\n5,1\n"))
    with pytest.raises(AnalysisError, match="not a recognised time unit") as err:
        analyze_step(samples)  # never silently read as seconds
    assert err.value.hint == "time_unit"


def test_a_time_like_single_letter_does_not_hide_an_ambiguous_signal(write_text):
    ds = load_csv(write_text("temp.csv", "sample,T (degC)\n" + "".join(f"{i},{20 + i % 3}\n" for i in range(40))))
    with pytest.raises(AmbiguousColumnError):
        analyze_signal(ds, sample_rate_hz=10.0)
