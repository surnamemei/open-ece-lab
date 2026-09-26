"""Typed analysis outcomes: results with explicit units, plus validation against requirements."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..errors import ConfigurationError
from ..validation import CheckResult, Requirement, evaluate_requirement, overall_status


@dataclass(frozen=True)
class Measurement:
    """One result. ``unit`` is explicit: ``"1"`` means dimensionless, ``None`` means unknown."""

    value: float
    unit: str | None
    description: str
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"value": self.value, "unit": self.unit, "description": self.description}
        if self.note:
            data["note"] = self.note
        return data


@dataclass(frozen=True, eq=False)
class DataTable:
    """Columns written as a CSV artifact of a run."""

    description: str
    columns: Mapping[str, np.ndarray]


@dataclass(frozen=True, eq=False)
class AnalysisOutcome:
    """Everything an analysis produced, before (or without) persisting it as a run.

    ``figures`` is called only when the run is saved; it returns
    ``{file name: (description, matplotlib Figure)}``.
    """

    analysis: str
    title: str
    source: Mapping[str, Any]
    parameters: Mapping[str, Any]
    results: Mapping[str, Measurement]
    warnings: tuple[str, ...] = ()
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)
    tables: Mapping[str, DataTable] = field(default_factory=dict)
    figures: Callable[[], Mapping[str, tuple[str, Any]]] | None = None
    checks: tuple[CheckResult, ...] = ()
    spec: Mapping[str, Any] | None = None

    @property
    def overall(self) -> str:
        return overall_status(self.checks)

    @property
    def simulated(self) -> bool:
        return self.source.get("simulated") is True

    def validation_dict(self) -> dict[str, Any]:
        return {"overall": self.overall, "spec": self.spec, "checks": [c.to_dict() for c in self.checks]}


def apply_requirements(outcome: AnalysisOutcome, requirements: Iterable[Requirement], *, spec=None) -> AnalysisOutcome:
    """Evaluate requirements against the outcome's results (returns a new outcome)."""
    requirements = list(requirements)
    names = [r.name for r in requirements]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigurationError(f"requirement(s) {duplicates} are defined more than once")
    unknown = [n for n in names if n not in outcome.results]
    if unknown:
        raise ConfigurationError(
            f"requirement(s) {unknown} do not match any result of the {outcome.analysis} analysis; "
            f"available results: {', '.join(outcome.results)}",
            hint="require",
        )
    checks = tuple(
        evaluate_requirement(r, outcome.results[r.name].value, outcome.results[r.name].unit) for r in requirements
    )
    spec_info = spec.to_dict() if hasattr(spec, "to_dict") else spec
    return replace(outcome, checks=checks, spec=spec_info)
