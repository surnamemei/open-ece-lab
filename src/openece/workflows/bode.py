"""Bode (frequency-response) analysis of swept data from a file or an instrument sweep."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ..analysis.electronics import estimate_cutoff_hz, interpolate_at_frequency, magnitude_to_db, unwrap_phase_deg
from ..errors import AnalysisError, ConfigurationError
from ..io import ColumnRole, Dataset, resolve_columns
from ..io.columns import FREQUENCY_NAMES, MAGNITUDE_NAMES, PHASE_NAMES
from ..io.dataset import Column
from ..units import UnitError, frequency_scale_to_hz, magnitude_kind, phase_scale_to_degrees
from .common import column_values, file_source, mapping_dict, number, unit_scale
from .outcome import AnalysisOutcome, DataTable, Measurement

ANALYSIS = "bode"
PARAMETERS = ("drop_db",)
ROLES = (
    ColumnRole("frequency", "frequency", FREQUENCY_NAMES),
    ColumnRole("magnitude", "magnitude", MAGNITUDE_NAMES, by_elimination=True),
    ColumnRole("phase", "phase", PHASE_NAMES, required=False),
)


def analyze_bode(
    dataset: Dataset,
    *,
    columns: Mapping[str, str | None] | None = None,
    frequency_unit: str | None = None,
    magnitude_unit: str | None = None,
    phase_unit: str | None = None,
    drop_db: float = 3.0,
    title: str | None = None,
) -> AnalysisOutcome:
    """Analyse frequency / magnitude (/ phase) columns.

    ``magnitude_unit`` is ``"db"`` or ``"linear"`` and is required when the magnitude header
    carries no unit; the phase unit is likewise never assumed. Frequencies default to Hz (with a
    warning) when the header has no unit.
    """
    drop_db = number(drop_db, "drop_db", positive=True)
    where = dataset.source.path
    if dataset.sample_rate_hz is not None:
        raise AnalysisError(
            f"{where}: this is a time-domain recording ({dataset.metadata.get('format', 'sampled')} data); "
            "Bode analysis needs a table of frequency and magnitude values. Use the signal analysis for waveforms."
        )
    mapping = resolve_columns(dataset, ROLES, columns)
    warnings = list(dataset.warnings)

    f_col = mapping["frequency"].column
    f_scale, f_unit = unit_scale(f_col, frequency_unit, quantity="frequency", convert=frequency_scale_to_hz,
                                 default="Hz", hint="frequency_unit", warnings=warnings, where=where)
    frequency = column_values(dataset, mapping["frequency"], "frequency") * f_scale

    m_col = mapping["magnitude"].column
    kind, magnitude_label = _magnitude_kind(m_col, magnitude_unit, warnings, where)
    magnitude = column_values(dataset, mapping["magnitude"], "magnitude")
    if kind == "linear":
        bad = np.flatnonzero(magnitude <= 0)
        if bad.size:
            raise AnalysisError(
                f"{where}: magnitude column {m_col.name!r} is treated as linear but has a value <= 0 at "
                f"{dataset.locate(int(bad[0]))}; if it is already in dB, set the magnitude unit to 'db'",
                hint="magnitude_unit",
            )
        magnitude_db = magnitude_to_db(magnitude)
    else:
        magnitude_db = magnitude.copy()

    phase = None
    units = {
        "frequency": {"unit_used": f_unit, "scale_to_hz": f_scale},
        "magnitude": {"interpreted_as": kind, "result_unit": magnitude_label},
    }
    if "phase" in mapping:
        p_col = mapping["phase"].column
        p_scale, p_unit = unit_scale(p_col, phase_unit, quantity="phase", convert=phase_scale_to_degrees,
                                     default=None, hint="phase_unit", warnings=warnings, where=where)
        phase = column_values(dataset, mapping["phase"], "phase") * p_scale
        units["phase"] = {"unit_used": p_unit, "scale_to_deg": p_scale}

    order = _frequency_order(frequency, dataset, f_col, warnings)
    frequency, magnitude_db = frequency[order], magnitude_db[order]
    if phase is not None:
        phase = unwrap_phase_deg(phase[order])

    results, result_warnings = bode_results(frequency, magnitude_db, phase, drop_db=drop_db,
                                            magnitude_unit=magnitude_label)
    warnings += result_warnings
    title = title or f"Bode analysis of {dataset.source.path}"
    table = {"frequency (Hz)": frequency, f"magnitude ({magnitude_label})": magnitude_db}
    if phase is not None:
        table["phase (deg)"] = phase
    return AnalysisOutcome(
        analysis=ANALYSIS,
        title=title,
        source=file_source(dataset),
        parameters={"drop_db": drop_db, "columns": mapping_dict(mapping), "units": units},
        results=results,
        warnings=tuple(warnings),
        arrays={"frequency_hz": frequency, "magnitude_db": magnitude_db, **({"phase_deg": phase} if phase is not None else {})},
        tables={"bode_data.csv": DataTable(
            "frequency response used for the analysis (sorted by frequency, magnitude in dB, phase unwrapped)", table)},
        figures=bode_figures(frequency, magnitude_db, phase, results, drop_db, magnitude_label, title),
    )


def bode_results(frequency_hz, magnitude_db, phase_deg=None, *, drop_db: float = 3.0,
                 magnitude_unit: str = "dB") -> tuple[dict[str, Measurement], list[str]]:
    """Cutoff, gain maximum and phase at cutoff for a sweep sorted by increasing frequency."""
    f = np.asarray(frequency_hz, dtype=float)
    m = np.asarray(magnitude_db, dtype=float)
    cutoff = estimate_cutoff_hz(f, m, drop_db)
    peak = int(np.argmax(m))
    results = {
        "cutoff_hz": Measurement(
            cutoff, "Hz",
            f"-{drop_db:g} dB cutoff: first crossing of (max gain - {drop_db:g} dB) above the gain maximum, "
            "log-frequency interpolation",
        ),
        "max_gain_db": Measurement(float(m[peak]), magnitude_unit, "maximum magnitude (reference level for the cutoff)"),
        "max_gain_frequency_hz": Measurement(float(f[peak]), "Hz", "frequency of the maximum magnitude"),
    }
    if phase_deg is not None:
        phase_at_cutoff = interpolate_at_frequency(f, phase_deg, cutoff)
        results["phase_at_cutoff_deg"] = Measurement(
            phase_at_cutoff, "deg", "phase at the cutoff frequency (unwrapped, log-frequency interpolation)")
    else:
        results["phase_at_cutoff_deg"] = Measurement(
            math.nan, "deg", "phase at the cutoff frequency", note="not available: the data has no phase column")
    warnings = []
    if not math.isfinite(cutoff):
        if peak == len(m) - 1:
            warnings.append("the magnitude maximum is at the highest frequency, so no cutoff above it can be found")
        else:
            warnings.append(
                f"no -{drop_db:g} dB crossing was found above the gain maximum at {f[peak]:.6g} Hz: the sweep may not "
                "extend far enough, or the response is not low-pass (only the upper cutoff is estimated in v0.1)"
            )
    return results, warnings


def _magnitude_kind(column: Column, override: str | None, warnings: list[str], where: str) -> tuple[str, str]:
    if override is not None:
        kind = override.strip().lower()
        if kind not in ("db", "linear"):
            raise ConfigurationError(f"magnitude unit must be 'db' or 'linear', got {override!r}", hint="magnitude_unit")
        if column.unit:
            try:
                header_kind, label = magnitude_kind(column.unit)
            except UnitError:
                header_kind, label = None, None
            if header_kind == kind:
                return kind, label
            warnings.append(f"magnitude unit {override!r} was given explicitly; the header says {column.unit!r}")
        return kind, "dB"
    if column.unit is None:
        raise AnalysisError(
            f"{where}: cannot tell whether magnitude column {column.name!r} is in dB or linear units: add a unit to the "
            "header (e.g. 'gain (dB)' or 'gain (V/V)') or specify the magnitude unit (db or linear)",
            hint="magnitude_unit",
        )
    try:
        return magnitude_kind(column.unit)
    except UnitError as exc:
        raise AnalysisError(f"{where}: magnitude column {column.name!r}: {exc}", hint="magnitude_unit") from exc


def _frequency_order(frequency: np.ndarray, dataset: Dataset, column: Column, warnings: list[str]) -> np.ndarray:
    where = dataset.source.path
    if len(frequency) < 3:
        raise AnalysisError(f"{where}: a Bode analysis needs at least 3 frequency points, got {len(frequency)}")
    bad = np.flatnonzero(frequency <= 0)
    if bad.size:
        raise AnalysisError(
            f"{where}: frequency column {column.name!r} has a non-positive value at {dataset.locate(int(bad[0]))}; "
            "Bode analysis needs f > 0 Hz (remove any DC row)"
        )
    steps = np.diff(frequency)
    if np.all(steps > 0):
        return np.arange(len(frequency))
    if np.all(steps < 0):
        warnings.append("the sweep runs from high to low frequency; it was reversed for analysis")
        return np.arange(len(frequency))[::-1]
    against = np.flatnonzero(steps <= 0) if frequency[-1] > frequency[0] else np.flatnonzero(steps >= 0)
    first = int(against[0]) + 1
    raise AnalysisError(
        f"{where}: frequency column {column.name!r} is not monotonic (repeated or out-of-order value at "
        f"{dataset.locate(first)}); sort the sweep and remove duplicate frequencies"
    )


def bode_figures(frequency, magnitude_db, phase, results, drop_db, magnitude_label, title):
    def build():
        from ..reporting.plots import bode_figure

        figure = bode_figure(
            frequency, magnitude_db, phase, cutoff_hz=results["cutoff_hz"].value,
            max_gain_db=results["max_gain_db"].value, drop_db=drop_db, magnitude_unit=magnitude_label, title=title,
        )
        return {"bode.png": ("Bode magnitude and phase", figure)}

    return build
