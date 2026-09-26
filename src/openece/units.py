"""Unit annotations in column headers and conversions to the units the analyses use.

Headers may carry a unit as ``name [unit]``, ``name (unit)`` or, for a short whitelist of
unambiguous units, as a ``name_unit`` suffix (``time_s``, ``frequency_hz``, ``gain_db``).
Parenthesised text is a unit only when it is a recognised unit (``m/s^2`` style compounds of
recognised units included). Other parenthesised text after a space, such as ``Voltage (CH1)``,
is kept as an *annotation* and never reported as a unit; glued text such as LTspice's
``V(out)`` or ``I(R1)`` stays part of the name.

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
    annotation: str | None = None  # parenthesised text that is not a recognised unit, e.g. "CH1"


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
_OTHER_UNITS = {
    "%", "ppm", "rpm", "rps", "w", "mw", "kw", "va", "j", "wh", "kwh", "ah", "mah",
    "kv", "vac", "vdc", "na", "ohm", "kohm", "mohm", "\u03c9", "\u2126", "k\u03c9", "m\u03c9",
    "f", "uf", "\u00b5f", "nf", "pf", "h", "mh", "uh", "\u00b5h",
    "\u00b0c", "degc", "\u00b0f", "degf", "k", "pa", "kpa", "mpa", "bar", "mbar", "psi",
    "n", "kn", "nm", "m", "mm", "cm", "km", "um", "\u00b5m", "kg", "g", "mg", "l", "ml",
    "lx", "lux", "fs", "lsb", "counts", "a.u.",
}

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

_EXPONENT = re.compile(r"(\^?-?\d+|[\u00b2\u00b3])$")
_TRAILING_BRACKET = re.compile(
    r"^(?P<base>.*?)(?P<gap>\s*)(?:\((?P<paren>[^()]*)\)|\[(?P<square>[^\[\]]*)\])\s*$"
)
_UNIT_SUFFIX = re.compile(r"^(?P<base>.+?)_(?P<unit>[A-Za-z%]+)$")


def parse_header(header: str) -> HeaderUnit:
    """Split a column header into a base name, an optional unit and an optional annotation."""
    text = header.strip()
    match = _TRAILING_BRACKET.match(text)
    if match and match.group("base").strip():
        base = match.group("base").strip()
        if match.group("square") is not None:
            unit = match.group("square").strip()
            if unit:
                return HeaderUnit(base, unit)
        else:
            inner = match.group("paren").strip()
            if inner and is_known_unit(inner):
                return HeaderUnit(base, inner)
            if inner and match.group("gap"):
                return HeaderUnit(base, None, inner)
    match = _UNIT_SUFFIX.match(text)
    if match:
        written = match.group("unit")
        if written in _FREQUENCY_SCALE_EXACT:  # keep the case: MHz and mHz differ by 1e9
            return HeaderUnit(match.group("base"), written)
        if written.lower() in _SUFFIX_UNITS:
            return HeaderUnit(match.group("base"), _SUFFIX_UNITS[written.lower()])
    return HeaderUnit(text, None)


def is_known_unit(text: str) -> bool:
    """True for recognised units and products/quotients of them (``m/s^2``, ``N*m``, ``1/s``)."""
    return all(_known_atom(atom.strip()) for atom in re.split(r"[/*\u00b7]", text.strip().lower()))


def _known_atom(atom: str) -> bool:
    stripped = _EXPONENT.sub("", atom)  # "s^2" -> "s", "m\u00b2" -> "m"
    return atom in _KNOWN_UNITS or (bool(stripped) and stripped in _KNOWN_UNITS)


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
