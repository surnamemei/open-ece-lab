"""Map file columns to the roles an analysis needs, without guessing.

For each role, in order:

1. an explicit mapping always wins;
2. otherwise a column whose name (ignoring case and unit annotations) is one of the role's
   recognised names, e.g. ``time``, ``timestamp`` or ``t`` for the time role. Single-letter
   names match case-sensitively (``T`` is usually a temperature, not time);
3. otherwise, for roles that allow it, the single numeric column no other role has claimed.

More than one candidate raises :class:`AmbiguousColumnError`; no candidate for a required role
raises :class:`MissingColumnError`. An optional role whose only name match is unusable is left
unassigned (the workflow reports it) rather than blocking the analysis. How each column was chosen is returned so that it can be
recorded in the run record and shown to the user.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..errors import AmbiguousColumnError, ColumnMappingError, MissingColumnError
from ..units import normalize_name, parse_header
from .dataset import Column, Dataset, SkippedColumn

TIME_NAMES = ("time", "t", "timestamp", "time_stamp", "seconds", "sec", "secs", "elapsed", "elapsed_time")
FREQUENCY_NAMES = ("frequency", "freq", "f", "frequencies")
MAGNITUDE_NAMES = ("magnitude", "mag", "gain")
PHASE_NAMES = ("phase", "phi", "angle", "arg")

EXPLICIT = "explicit"
BY_NAME = "recognised name"
BY_ELIMINATION = "only remaining numeric column"


@dataclass(frozen=True)
class ColumnRole:
    key: str
    description: str
    names: tuple[str, ...] = ()
    required: bool = True
    by_elimination: bool = False
    accepts_datetime: bool = False


@dataclass(frozen=True)
class ColumnAssignment:
    role: str
    column: Column
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {"column": self.column.name, "unit_in_file": self.column.unit, "selected_by": self.method}


def match_by_name(dataset: Dataset, role: ColumnRole) -> tuple[list[Column], list[SkippedColumn]]:
    """Columns (usable and skipped) whose names are recognised for ``role``."""
    names = set(role.names)
    return (
        [c for c in dataset.columns if _name_matches(c.name, names)],
        [s for s in dataset.skipped if _name_matches(s.name, names)],
    )


def resolve_columns(
    dataset: Dataset,
    roles: Sequence[ColumnRole],
    explicit: Mapping[str, str | None] | None = None,
) -> dict[str, ColumnAssignment]:
    """Assign a column to every role (optional roles may stay unassigned)."""
    source = dataset.source.path
    explicit = {k: v for k, v in (explicit or {}).items() if v is not None}
    by_key = {r.key: r for r in roles}
    unknown = sorted(set(explicit) - set(by_key))
    if unknown:
        raise ColumnMappingError(f"unknown column role(s) {unknown}; this analysis uses {sorted(by_key)}")
    assigned: dict[str, ColumnAssignment] = {}

    def claimed(column: Column) -> ColumnAssignment | None:
        return next((a for a in assigned.values() if a.column is column), None)

    def claim(role: ColumnRole, column: Column, method: str) -> None:
        other = claimed(column)
        if other is not None:
            raise ColumnMappingError(
                f"{source}: column {column.name!r} cannot be both the {by_key[other.role].description} "
                f"and the {role.description} column",
                role=role.key,
            )
        if column.kind == "datetime" and not role.accepts_datetime:
            raise ColumnMappingError(
                f"{source}: column {column.name!r} holds date-time values and cannot be the {role.description} column",
                role=role.key,
            )
        assigned[role.key] = ColumnAssignment(role.key, column, method)

    for role in roles:
        if role.key in explicit:
            claim(role, dataset.find_column(str(explicit[role.key]), role=role.key), EXPLICIT)

    for role in roles:
        if role.key in assigned or not role.names:
            continue
        columns, skipped = match_by_name(dataset, role)
        columns = [c for c in columns if claimed(c) is None]
        candidates = [c.name for c in columns] + [s.name for s in skipped]
        if len(candidates) > 1:
            raise AmbiguousColumnError(
                f"{source}: cannot choose the {role.description} column automatically: "
                f"{len(candidates)} columns have a recognised name ({_quoted(candidates)}). Specify it explicitly.",
                role=role.key, candidates=candidates, available=dataset.column_names,
            )
        if skipped and not role.required:
            continue  # optional role: the workflow warns that the column could not be used
        if skipped:
            raise ColumnMappingError(
                f"{source}: column {skipped[0].name!r} looks like the {role.description} column "
                f"but cannot be used: {skipped[0].reason}",
                role=role.key, hint=skipped[0].hint,
            )
        if columns:
            claim(role, columns[0], BY_NAME)

    # A missing name-only role is the root problem: report it before elimination, whose
    # candidates depend on the other roles being assigned.
    _require(dataset, [r for r in roles if not r.by_elimination], assigned)
    for role in roles:
        if role.key in assigned or not role.by_elimination:
            continue
        remaining = [c for c in dataset.columns if c.kind == "numeric" and claimed(c) is None]
        if len(remaining) > 1:
            names = [c.name for c in remaining]
            layout = ""
            if len(names) > dataset.n_rows:
                layout = (f" The file has more columns than data rows ({dataset.n_rows}); samples must be in rows "
                          "(one column per signal), so transpose data that was saved as a row.")
            raise AmbiguousColumnError(
                f"{source}: cannot choose the {role.description} column automatically: "
                f"{len(names)} candidate columns ({_quoted(names)}). Specify it explicitly.{layout}",
                role=role.key, candidates=names, available=dataset.column_names,
            )
        if remaining:
            claim(role, remaining[0], BY_ELIMINATION)

    _require(dataset, roles, assigned)
    return assigned


def _require(dataset: Dataset, roles: Sequence[ColumnRole], assigned: Mapping[str, ColumnAssignment]) -> None:
    for role in roles:
        if role.required and role.key not in assigned:
            recognised = f" (recognised names: {', '.join(role.names)})" if role.names else ""
            unusable = "".join(f" Column {s.name!r} is not usable: {s.reason}." for s in dataset.skipped)
            raise MissingColumnError(
                f"{dataset.source.path}: no {role.description} column found{recognised}. "
                f"Available numeric columns: {_quoted(dataset.column_names)}.{unusable} Specify it explicitly.",
                role=role.key, available=dataset.column_names,
            )


def _name_matches(name: str, names: set[str]) -> bool:
    for candidate in (parse_header(name).base.strip(), name.strip()):
        if len(candidate) == 1:
            if candidate in names:  # exact case for single letters
                return True
        elif normalize_name(candidate) in names:
            return True
    return False


def _quoted(names: Sequence[str], limit: int = 10) -> str:
    if not names:
        return "(none)"
    shown = ", ".join(repr(n) for n in names[:limit])
    return shown if len(names) <= limit else f"{shown} and {len(names) - limit} more"
