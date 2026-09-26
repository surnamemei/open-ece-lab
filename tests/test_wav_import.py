import io
import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from openece.errors import AmbiguousColumnError, DataImportError
from openece.io import load_file, load_wav
from openece.workflows import analyze_signal

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def write_wav(path, rate, data):
    wavfile.write(path, rate, data)
    return path


def tone(fs, n, freq, amplitude):
    return amplitude * np.sin(2 * np.pi * freq * np.arange(n) / fs)


def test_mono_int16_is_scaled_to_full_scale(tmp_path):
    path = write_wav(tmp_path / "m.wav", 48000, np.array([0, 16384, -32768, 32767], dtype=np.int16))
    ds = load_wav(path)
    assert ds.column_names == ["ch1"]
    assert ds.columns[0].unit == "FS"
    np.testing.assert_array_equal(ds.columns[0].values, [0.0, 0.5, -1.0, 32767 / 32768])
    assert ds.sample_rate_hz == 48000.0
    meta = ds.metadata
    assert (meta["channels"], meta["channel_layout"], meta["frames"]) == (1, "mono", 4)
    assert meta["encoding"] == "PCM 16-bit"
    assert meta["duration_s"] == pytest.approx(4 / 48000)
    assert ds.locate(2) == "sample 2"


def test_stereo_channels_stay_separate(tmp_path):
    left, right = tone(8000, 800, 440, 0.5), tone(8000, 800, 1000, 0.25)
    path = write_wav(tmp_path / "s.wav", 8000, np.stack([left, right], axis=1).astype(np.float32))
    ds = load_wav(path)
    assert ds.column_names == ["ch1", "ch2"]
    assert "ch1 = left" in ds.metadata["channel_layout"]
    np.testing.assert_allclose(ds.find_column("ch1").values, left.astype(np.float32))
    np.testing.assert_allclose(ds.find_column("ch2").values, right.astype(np.float32))
    assert ds.metadata["encoding"] == "IEEE float 32-bit"


def test_stereo_analysis_requires_an_explicit_channel(tmp_path):
    left, right = tone(8000, 8000, 440, 0.5), tone(8000, 8000, 1000, 0.25)
    ds = load_wav(write_wav(tmp_path / "s.wav", 8000, np.stack([left, right], axis=1).astype(np.float32)))
    with pytest.raises(AmbiguousColumnError) as err:
        analyze_signal(ds)
    assert err.value.candidates == ("ch1", "ch2")
    assert err.value.hint == "signal"
    right_outcome = analyze_signal(ds, columns={"signal": "ch2"})
    assert right_outcome.results["dominant_frequency_hz"].value == pytest.approx(1000.0, abs=1.0)
    assert analyze_signal(ds, columns={"signal": "ch1"}).results["dominant_frequency_hz"].value == pytest.approx(440.0, abs=1.0)


def test_24_bit_pcm(tmp_path):
    values = [0, 2**23 - 1, -2**23, 2**22]
    path = tmp_path / "p24.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(96000)
        w.writeframes(b"".join(v.to_bytes(3, "little", signed=True) for v in values))
    ds = load_wav(path)
    np.testing.assert_allclose(ds.columns[0].values, [0.0, (2**23 - 1) / 2**23, -1.0, 0.5])
    assert ds.metadata["encoding"] == "PCM 24-bit"


def test_unsigned_8_bit_pcm(tmp_path):
    ds = load_wav(write_wav(tmp_path / "u8.wav", 8000, np.array([0, 128, 255], dtype=np.uint8)))
    np.testing.assert_array_equal(ds.columns[0].values, [-1.0, 0.0, 127 / 128])
    assert ds.metadata["encoding"] == "PCM 8-bit"


def test_metadata_reaches_the_analysis_and_record(tmp_path):
    ds = load_file(write_wav(tmp_path / "t.wav", 44100, tone(44100, 44100, 1000, 0.5).astype(np.float32)))
    outcome = analyze_signal(ds)
    assert outcome.parameters["time"] == {"source": "file sample rate", "sample_rate_hz": 44100.0}
    assert outcome.results["rms"].unit == "FS"
    assert outcome.results["dominant_frequency_hz"].value == pytest.approx(1000.0, abs=1.0)
    assert outcome.source["metadata"]["encoding"] == "IEEE float 32-bit"


def test_truncated_file_is_loaded_with_a_visible_warning(tmp_path):
    buffer = io.BytesIO()
    wavfile.write(buffer, 8000, np.zeros(1000, dtype=np.int16))
    data = buffer.getvalue()
    path = tmp_path / "cut.wav"
    path.write_bytes(data[: len(data) // 2])
    ds = load_wav(path)
    assert any("EOF" in w for w in ds.warnings)


@pytest.mark.parametrize("content, message", [
    (b"just some text", "not a WAV file"),
    (b"", "not a WAV file"),
])
def test_invalid_files(tmp_path, content, message):
    path = tmp_path / "bad.wav"
    path.write_bytes(content)
    with pytest.raises(DataImportError, match=message):
        load_wav(path)


def test_file_without_samples(tmp_path):
    with pytest.raises(DataImportError, match="no samples"):
        load_wav(write_wav(tmp_path / "empty.wav", 8000, np.zeros(0, dtype=np.int16)))


def test_riff_info_comment_is_exposed():
    ds = load_wav(EXAMPLES / "signal.wav")
    assert ds.metadata["comments"] and "SYNTHETIC" in ds.metadata["comments"][0]
    assert ds.metadata["info"]["ISFT"].startswith("OpenECE Lab")
