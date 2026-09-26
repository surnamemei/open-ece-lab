"""PASS/FAIL validation of named results against inclusive limits."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .errors import ConfigurationError

PASS = "PASS"
FAIL = "FAIL"
NOT_EVALUATED = "NOT_EVALUATED"


def check_range(name: str, value: float, minimum: float | None = None, maximum: float | None = None):
    passed = True
    if minimum is not None: passed &= value >= minimum
    if maximum is not None: passed &= value <= maximum
    return {"name": name, "value": float(value), "min": minimum, "max": maximum, "passed": bool(passed)}


@dataclass(frozen=True)
class Requirement:
    """Inclusive acceptance limits for one named result (in that result's unit)."""

    name: str
    min: float | None = None
    max: float | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError("a requirement needs a non-empty result name")
        for bound in (self.min, self.max):
            if bound is not None and not math.isfinite(bound):
                raise ConfigurationError(f"requirement {self.name!r}: limits must be finite numbers")
        if self.min is None and self.max is None:
            raise ConfigurationError(f"requirement {self.name!r} needs a minimum and/or a maximum")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ConfigurationError(f"requirement {self.name!r}: minimum {self.min:g} is greater than maximum {self.max:g}")

    def describe(self) -> str:
        if self.min is not None and self.max is not None:
            return f"[{self.min:g}, {self.max:g}]"
        return f">= {self.min:g}" if self.min is not None else f"<= {self.max:g}"


@dataclass(frozen=True)
class CheckResult:
    name: str
    value: float
    unit: str | None
    min: float | None
    max: float | None
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "value": self.value, "unit": self.unit, "min": self.min, "max": self.max,
            "status": PASS if self.passed else FAIL, "detail": self.detail,
        }


def parse_requirement(text: str) -> Requirement:
    """Parse ``NAME=MIN:MAX``; either bound may be empty, e.g. ``overshoot_percent=:10``."""
    name, sep, limits = text.partition("=")
    if not sep or ":" not in limits:
        raise ConfigurationError(
            f"invalid requirement {text!r}: expected NAME=MIN:MAX, e.g. cutoff_hz=1450:1750 or overshoot_percent=:10",
            hint="require",
        )
    low, _, high = limits.partition(":")
    return Requirement(name.strip(), _parse_bound(low, text), _parse_bound(high, text))


def requirements_from_mapping(mapping: Any, *, source: str = "requirements") -> list[Requirement]:
    """Requirements from a ``{name: {min: ..., max: ...}}`` mapping, as used in YAML files."""
    if not isinstance(mapping, Mapping):
        raise ConfigurationError(f"{source}: 'requirements' must map result names to {{min, max}} limits")
    requirements = []
    for name, limits in mapping.items():
        if not isinstance(limits, Mapping):
            raise ConfigurationError(f"{source}: requirement {name!r} must be a mapping with 'min' and/or 'max'")
        unknown = sorted(set(limits) - {"min", "max"})
        if unknown:
            raise ConfigurationError(f"{source}: requirement {name!r} has unknown key(s) {unknown} (allowed: min, max)")
        requirements.append(Requirement(
            str(name),
            _coerce_limit(limits.get("min"), f"{source}: requirement {name!r} min"),
            _coerce_limit(limits.get("max"), f"{source}: requirement {name!r} max"),
        ))
    return requirements


def evaluate_requirement(requirement: Requirement, value: float, unit: str | None = None) -> CheckResult:
    """Check one value; a value that is not finite (not determined) never passes."""
    value = float(value)
    passed = check_range(requirement.name, value, requirement.min, requirement.max)["passed"]
    if not math.isfinite(value):
        detail = "not determined (no finite value)"
    elif requirement.min is not None and value < requirement.min:
        detail = f"below minimum {requirement.min:g}"
    elif requirement.max is not None and value > requirement.max:
        detail = f"above maximum {requirement.max:g}"
    else:
        detail = "within limits"
    return CheckResult(requirement.name, value, unit, requirement.min, requirement.max, passed, detail)


def overall_status(checks: Iterable[CheckResult]) -> str:
    checks = list(checks)
    if not checks:
        return NOT_EVALUATED
    return PASS if all(c.passed for c in checks) else FAIL


def _parse_bound(text: str, original: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise ConfigurationError(f"invalid limit {text!r} in requirement {original!r}", hint="require") from None


def _coerce_limit(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigurationError(f"{where} must be a number, got {value!r}")
    try:
        return float(value)  # YAML 1.1 reads '1e-3' (no dot) as a string; accept it here
    except (TypeError, ValueError):
        raise ConfigurationError(f"{where} must be a number, got {value!r}") from None
