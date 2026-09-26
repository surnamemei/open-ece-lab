"""Helpers shared by the file-based workflows: parameter checks, units and column values."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..errors import AnalysisError, ConfigurationError
from ..io import ColumnAssignment, Dataset
from ..io.dataset import Column
from ..units import UnitError, time_scale_to_seconds


def number(value: Any, name: str, *, positive: bool = False, minimum: float | None = None,
           maximum: float | None = None, exclusive: bool = False) -> float:
    """Validate a numeric parameter and return it as float."""
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a number, got {value!r}", hint=name)
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ConfigurationError(f"{name} must be a number, got {value!r}", hint=name) from None
    if not math.isfinite(result):
        raise ConfigurationError(f"{name} must be finite, got {value!r}", hint=name)
    if positive and result <= 0:
        raise ConfigurationError(f"{name} must be greater than zero, got {result:g}", hint=name)
    if minimum is not None and (result <= minimum if exclusive else result < minimum):
        raise ConfigurationError(f"{name} must be {'>' if exclusive else '>='} {minimum:g}, got {result:g}", hint=name)
    if maximum is not None and (result >= maximum if exclusive else result > maximum):
        raise ConfigurationError(f"{name} must be {'<' if exclusive else '<='} {maximum:g}, got {result:g}", hint=name)
    return result


def integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}, got {value!r}", hint=name)
    return int(value)


def column_values(dataset: Dataset, assignment: ColumnAssignment, label: str) -> np.ndarray:
    """Values of an assigned column; blanks and non-finite values are errors that point at the file."""
    col = assignment.column
    where = dataset.source.path
    if col.blank_rows:
        raise AnalysisError(
            f"{where}: {label} column {col.name!r} has {len(col.blank_rows)} blank value(s), "
            f"first at {dataset.locate(col.blank_rows[0])} (a truncated or incomplete row?)"
        )
    bad = np.flatnonzero(~np.isfinite(col.values))
    if bad.size:
        raise AnalysisError(
            f"{where}: {label} column {col.name!r} contains non-finite values (NaN/inf), first at {dataset.locate(int(bad[0]))}"
        )
    return col.values


def unit_scale(column: Column, override: str | None, *, quantity: str, convert: Callable[[str], float],
               default: str | None, hint: str, warnings: list[str], where: str) -> tuple[float, str]:
    """Scale factor from the column's unit (or an explicit override) to the analysis unit.

    With neither a header unit nor an override, ``default`` is assumed and a warning is added;
    if ``default`` is None that is an error instead.
    """
    unit = override or column.unit
    if unit is None:
        if default is None:
            raise AnalysisError(
                f"{where}: {quantity} column {column.name!r} has no unit in its header; specify the {quantity} unit explicitly",
                hint=hint,
            )
        warnings.append(f"{quantity} column {column.name!r} has no unit in its header; assuming {default}")
        return convert(default), f"{default} (assumed)"
    if override and column.unit and override != column.unit:
        warnings.append(f"{quantity} unit {override!r} was given explicitly; the header says {column.unit!r}")
    try:
        return convert(unit), unit
    except UnitError as exc:
        raise AnalysisError(f"{where}: {quantity} column {column.name!r}: {exc}", hint=hint) from exc


def time_seconds(dataset: Dataset, assignment: ColumnAssignment, time_unit: str | None,
                 warnings: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Time column converted to seconds, plus a description for the record."""
    column = assignment.column
    scale, unit = unit_scale(column, time_unit, quantity="time", convert=time_scale_to_seconds,
                             default="s", hint="time_unit", warnings=warnings, where=dataset.source.path)
    t = column_values(dataset, assignment, "time") * scale
    info = {"source": "time column", **assignment.to_dict(), "unit_used": unit, "scale_to_s": scale}
    if column.note:
        info["note"] = column.note
    return t, info


def file_source(dataset: Dataset) -> dict[str, Any]:
    return {**dataset.source.to_dict(), "metadata": dict(dataset.metadata)}


def mapping_dict(mapping: Mapping[str, ColumnAssignment]) -> dict[str, Any]:
    return {role: assignment.to_dict() for role, assignment in mapping.items()}
