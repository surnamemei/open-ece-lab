"""Unit annotations in column headers and conversions to the units the analyses use.

Headers may carry a unit as ``name (unit)``, ``name [unit]`` or, for a short whitelist of
unambiguous units, as a ``name_unit`` suffix (``time_s``, ``frequency_hz``, ``gain_db``).
Parenthesised text glued to the name is only treated as a unit when it is a known unit, so
LTspice-style names such as ``V(out)`` or ``I(R1)`` stay intact.

Analyses work in seconds, hertz, decibels and degrees. The helpers below convert explicitly and
raise :class:`UnitError` instead of guessing when a unit is unknown or ambiguous.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass


class UnitError(ValueError):
    """A unit is unknown, or ambiguous, for the requested quantity."""


@dataclass(frozen=True)
class HeaderUnit:
    base: str
    unit: str | None


_TIME_SCALE = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "ms": 1e-3, "msec": 1e-3,
    "us": 1e-6, "µs": 1e-6, "μs": 1e-6, "usec": 1e-6,
    "ns": 1e-9, "ps": 1e-12,
    "min": 60.0, "h": 3600.0, "hr": 3600.0,
}
_TWO_PI = 2.0 * math.pi
_FREQUENCY_SCALE_EXACT = {
    "Hz": 1.0, "HZ": 1.0, "kHz": 1e3, "KHz": 1e3, "KHZ": 1e3,
    "MHz": 1e6, "MHZ": 1e6, "GHz": 1e9, "GHZ": 1e9, "mHz": 1e-3,
}
_FREQUENCY_SCALE_CASELESS = {"hz": 1.0, "khz": 1e3, "ghz": 1e9, "rad/s": 1.0 / _TWO_PI, "rad/sec": 1.0 / _TWO_PI}
_PHASE_SCALE = {
    "deg": 1.0, "degree": 1.0, "degrees": 1.0, "°": 1.0,
    "rad": 180.0 / math.pi, "radian": 180.0 / math.pi, "radians": 180.0 / math.pi,
}
_LINEAR_RATIO_UNITS = {"v/v", "a/a", "linear", "lin", "ratio", "abs", "1"}
_AMPLITUDE_UNITS = {"v", "mv", "uv", "µv", "μv", "vpp", "vpk", "vrms", "a", "ma", "ua", "µa"}
_OTHER_UNITS = {"%", "rpm", "w", "mw", "ohm", "ω", "Ω", "°c", "degc", "k", "pa", "n", "nm", "m", "mm", "fs"}

# Units recognised in "Name(unit)" (no space) headers.
_KNOWN_UNITS = (
    {u.lower() for u in _TIME_SCALE}
    | {u.lower() for u in _FREQUENCY_SCALE_EXACT}
    | set(_FREQUENCY_SCALE_CASELESS)
    | set(_PHASE_SCALE)
    | _LINEAR_RATIO_UNITS
    | _AMPLITUDE_UNITS
    | _OTHER_UNITS
    | {"db", "dbv", "dbu", "dbm", "dbfs"}
)
# Units recognised as a "name_unit" suffix, mapped to their conventional spelling.
_SUFFIX_UNITS = {
    "s": "s", "ms": "ms", "us": "us", "ns": "ns",
    "hz": "Hz", "khz": "kHz", "mhz": "mhz", "ghz": "GHz",
    "db": "dB", "dbv": "dBV", "dbm": "dBm", "deg": "deg", "rad": "rad",
    "v": "V", "mv": "mV", "uv": "uV", "rpm": "rpm", "pct": "%", "percent": "%",
}

_TRAILING_BRACKET = re.compile(
    r"^(?P<base>.*?)(?P<gap>\s*)(?:\((?P<paren>[^()]*)\)|\[(?P<square>[^\[\]]*)\])\s*$"
)
_UNIT_SUFFIX = re.compile(r"^(?P<base>.+?)_(?P<unit>[A-Za-z%]+)$")


def parse_header(header: str) -> HeaderUnit:
    """Split a column header into a base name and an optional unit."""
    text = header.strip()
    match = _TRAILING_BRACKET.match(text)
    if match and match.group("base").strip():
        base = match.group("base").strip()
        if match.group("square") is not None:
            unit = match.group("square").strip()
            if unit:
                return HeaderUnit(base, unit)
        else:
            unit = match.group("paren").strip()
            if unit and (match.group("gap") or unit.lower() in _KNOWN_UNITS):
                return HeaderUnit(base, unit)
    match = _UNIT_SUFFIX.match(text)
    if match and match.group("unit").lower() in _SUFFIX_UNITS:
        return HeaderUnit(match.group("base"), _SUFFIX_UNITS[match.group("unit").lower()])
    return HeaderUnit(text, None)


def normalize_name(name: str) -> str:
    """Normalise a name for matching: case-folded, runs of punctuation/space become ``_``."""
    return re.sub(r"[\W_]+", "_", name.casefold()).strip("_")


def time_scale_to_seconds(unit: str) -> float:
    """Factor that converts values in ``unit`` to seconds."""
    key = unit.strip()
    scale = _TIME_SCALE.get(key, _TIME_SCALE.get(key.lower()))
    if scale is None:
        raise UnitError(f"unknown time unit {unit!r} (supported: s, ms, us, ns, ps, min, h)")
    return scale


def frequency_scale_to_hz(unit: str) -> float:
    """Factor that converts values in ``unit`` to hertz (``rad/s`` is converted to Hz)."""
    key = unit.strip()
    if key in _FREQUENCY_SCALE_EXACT:
        return _FREQUENCY_SCALE_EXACT[key]
    low = key.lower()
    if low == "mhz":
        raise UnitError(f"frequency unit {unit!r} is ambiguous: write 'MHz' (1e6 Hz) or 'mHz' (1e-3 Hz)")
    if low in _FREQUENCY_SCALE_CASELESS:
        return _FREQUENCY_SCALE_CASELESS[low]
    raise UnitError(f"unknown frequency unit {unit!r} (supported: Hz, kHz, MHz, GHz, mHz, rad/s)")


def phase_scale_to_degrees(unit: str) -> float:
    """Factor that converts values in ``unit`` to degrees."""
    scale = _PHASE_SCALE.get(unit.strip().lower())
    if scale is None:
        raise UnitError(f"unknown phase unit {unit!r} (supported: deg, rad)")
    return scale


def magnitude_kind(unit: str) -> tuple[str, str]:
    """Classify a magnitude unit.

    Returns ``("db", label)`` for decibel units (dB, dBV, dBm, ...) or ``("linear", label)`` for
    linear ratios/amplitudes, where ``label`` is the unit of the values once expressed in dB.
    """
    text = unit.strip()
    low = text.lower()
    if low.startswith("db"):
        return "db", text
    if low in _LINEAR_RATIO_UNITS:
        return "linear", "dB"
    if low in _AMPLITUDE_UNITS:
        return "linear", f"dB re 1 {text}"
    raise UnitError(
        f"cannot interpret magnitude unit {unit!r}: expected a decibel unit (dB, dBV, ...) "
        "or a linear ratio/amplitude (V/V, V, ...)"
    )
