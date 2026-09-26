"""The committed example datasets must be exactly what scripts/generate_examples.py produces.

Values are compared numerically at the files' printed precision so the check is robust to
last-digit differences in floating-point libraries between platforms.
"""
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from openece.io import load_csv, load_wav

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_csv_examples_match_the_generator(tmp_path, generator):
    for name, text, tolerances in [
        ("rc_sweep.csv", generator.rc_sweep_csv(), {"Frequency (Hz)": 1e-3, "Magnitude (dB)": 1e-4, "Phase (deg)": 1e-3}),
        ("motor_step.csv", generator.motor_step_csv(), {"time (s)": 1e-9, "speed (rpm)": 0.011}),
    ]:
        fresh = tmp_path / name
        fresh.write_text(text, encoding="utf-8")
        committed, expected = load_csv(EXAMPLES / name), load_csv(fresh)
        assert committed.column_names == expected.column_names == list(tolerances)
        assert committed.metadata["comments"] == expected.metadata["comments"]
        for column, atol in tolerances.items():
            np.testing.assert_allclose(committed.find_column(column).values, expected.find_column(column).values,
                                       rtol=1e-6, atol=atol, err_msg=f"{name}: {column}")


def test_wav_example_matches_the_generator(tmp_path, generator):
    fresh = tmp_path / "signal.wav"
    fresh.write_bytes(generator.signal_wav_bytes())
    rate_a, committed = wavfile.read(EXAMPLES / "signal.wav")
    rate_b, expected = wavfile.read(fresh)
    assert rate_a == rate_b == generator.SIGNAL_SAMPLE_RATE_HZ
    assert committed.dtype == expected.dtype == np.int16
    assert np.max(np.abs(committed.astype(int) - expected.astype(int))) <= 1  # at most one LSB
    assert load_wav(EXAMPLES / "signal.wav").metadata["info"] == load_wav(fresh).metadata["info"]


def test_examples_declare_that_they_are_synthetic():
    for name in ("rc_sweep.csv", "motor_step.csv"):
        assert any("SYNTHETIC DATA" in c for c in load_csv(EXAMPLES / name).metadata["comments"])
    assert "SYNTHETIC DATA" in load_wav(EXAMPLES / "signal.wav").metadata["comments"][0]


def test_generator_refuses_to_overwrite_without_force(tmp_path, generator, capsys):
    assert generator.main(["--out-dir", str(tmp_path)]) == 0
    first = (tmp_path / "rc_sweep.csv").read_bytes()
    assert generator.main(["--out-dir", str(tmp_path)]) == 1
    assert "refusing to overwrite" in capsys.readouterr().err
    assert generator.main(["--out-dir", str(tmp_path), "--force"]) == 0
    assert (tmp_path / "rc_sweep.csv").read_bytes() == first  # deterministic
