import math

import pytest

from openece.units import (
    UnitError, frequency_scale_to_hz, magnitude_kind, normalize_name, parse_header, phase_scale_to_degrees,
    time_scale_to_seconds,
)


@pytest.mark.parametrize("header, base, unit", [
    ("Time (s)", "Time", "s"),
    ("time [ms]", "time", "ms"),
    ("Time(us)", "Time", "us"),
    ("Channel A (V)", "Channel A", "V"),
    ("Phase (°)", "Phase", "°"),
    ("frequency_hz", "frequency", "Hz"),
    ("gain_db", "gain", "dB"),
    ("speed_rpm", "speed", "rpm"),
    ("V(out)", "V(out)", None),   # LTspice node name, not a unit
    ("I(R1)", "I(R1)", None),
    ("value_1", "value_1", None),
    ("plain", "plain", None),
])
def test_parse_header(header, base, unit):
    parsed = parse_header(header)
    assert (parsed.base, parsed.unit) == (base, unit)


def test_normalize_name():
    assert normalize_name("  Elapsed-Time ") == "elapsed_time"
    assert normalize_name("TIME") == "time"


@pytest.mark.parametrize("unit, scale", [
    ("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("µs", 1e-6), ("μs", 1e-6), ("ns", 1e-9), ("min", 60.0),
    ("MS", 1e-3),
])
def test_time_scale(unit, scale):
    assert time_scale_to_seconds(unit) == scale


def test_time_scale_unknown_unit():
    with pytest.raises(UnitError, match="samples"):
        time_scale_to_seconds("samples")


@pytest.mark.parametrize("unit, scale", [
    ("Hz", 1.0), ("hz", 1.0), ("kHz", 1e3), ("khz", 1e3), ("MHz", 1e6), ("GHz", 1e9), ("mHz", 1e-3),
    ("rad/s", 1 / (2 * math.pi)),
])
def test_frequency_scale(unit, scale):
    assert frequency_scale_to_hz(unit) == pytest.approx(scale)


def test_lowercase_mhz_is_ambiguous():
    with pytest.raises(UnitError, match="ambiguous"):
        frequency_scale_to_hz("mhz")


def test_phase_scale():
    assert phase_scale_to_degrees("deg") == 1.0
    assert phase_scale_to_degrees("°") == 1.0
    assert phase_scale_to_degrees("rad") == pytest.approx(180 / math.pi)
    with pytest.raises(UnitError):
        phase_scale_to_degrees("grad")


@pytest.mark.parametrize("unit, expected", [
    ("dB", ("db", "dB")), ("dBV", ("db", "dBV")), ("V/V", ("linear", "dB")), ("V", ("linear", "dB re 1 V")),
])
def test_magnitude_kind(unit, expected):
    assert magnitude_kind(unit) == expected


def test_magnitude_kind_unknown():
    with pytest.raises(UnitError):
        magnitude_kind("furlongs")
