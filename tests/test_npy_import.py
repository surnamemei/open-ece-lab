import numpy as np
import pytest

from openece.errors import DataImportError, MissingColumnError
from openece.io import detect_format, load_file, load_npy
from openece.workflows import analyze_signal, analyze_step


def test_one_dimensional_array_needs_an_explicit_sample_rate(tmp_path):
    fs = 1000.0
    np.save(tmp_path / "x.npy", np.sin(2 * np.pi * 50 * np.arange(1000) / fs))
    ds = load_npy(tmp_path / "x.npy")
    assert ds.column_names == ["column_1"]
    assert ds.metadata == {"format": "npy", "shape": [1000], "dtype": "float64"}
    with pytest.raises(MissingColumnError) as err:
        analyze_signal(ds)
    assert err.value.role == "time"
    outcome = analyze_signal(ds, sample_rate_hz=fs)
    assert outcome.results["dominant_frequency_hz"].value == pytest.approx(50.0)


def test_two_dimensional_array_columns(tmp_path):
    np.save(tmp_path / "x.npy", np.arange(12, dtype=np.int32).reshape(6, 2))
    ds = load_npy(tmp_path / "x.npy")
    assert ds.column_names == ["column_1", "column_2"]
    np.testing.assert_array_equal(ds.columns[1].values, [1, 3, 5, 7, 9, 11])
    assert ds.columns[1].values.dtype == np.float64


def test_structured_array_uses_field_names(tmp_path):
    t = np.linspace(0, 2, 2001)
    data = np.zeros(t.size, dtype=[("time", "f8"), ("y", "f8")])
    data["time"], data["y"] = t, 1 - np.exp(-t / 0.2)
    np.save(tmp_path / "step.npy", data)
    outcome = analyze_step(load_npy(tmp_path / "step.npy"))
    assert outcome.parameters["columns"]["time"]["column"] == "time"
    assert outcome.results["rise_time_10_90_s"].value == pytest.approx(2.1972 * 0.2, abs=0.01)


def test_channels_first_orientation_is_rejected(tmp_path):
    np.save(tmp_path / "x.npy", np.zeros((2, 100)))
    with pytest.raises(DataImportError, match="transposed"):
        load_npy(tmp_path / "x.npy")


def test_object_arrays_are_refused(tmp_path):
    np.save(tmp_path / "x.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(DataImportError, match="security"):
        load_npy(tmp_path / "x.npy")


def test_complex_arrays_are_refused(tmp_path):
    np.save(tmp_path / "x.npy", np.ones(4, dtype=complex))
    with pytest.raises(DataImportError, match="complex"):
        load_npy(tmp_path / "x.npy")


def test_non_npy_content_is_reported(tmp_path):
    (tmp_path / "x.npy").write_text("not numpy")
    with pytest.raises(DataImportError, match="not a NumPy .npy file"):
        load_npy(tmp_path / "x.npy")


def test_format_dispatch(tmp_path):
    assert detect_format("a.CSV") == "csv" and detect_format("a.tsv") == "csv" and detect_format("b.wav") == "wav"
    with pytest.raises(DataImportError, match="extension") as err:
        detect_format("data.xlsx")
    assert err.value.hint == "format"
    np.save(tmp_path / "x.npy", np.zeros(4))
    with pytest.raises(DataImportError, match="do not apply"):
        load_file(tmp_path / "x.npy", csv_options={"delimiter": ";"})
    assert load_file(tmp_path / "x.npy", csv_options={"delimiter": None}).column_names == ["column_1"]
