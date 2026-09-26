from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..errors import ConfigurationError
from ..validation import Requirement, requirements_from_mapping

REQUIRED = {"name", "stimulus", "requirements"}
SPEC_KEYS = {"name", "description", "analysis", "parameters", "requirements"}


def load_recipe(path):
    """Load a measurement recipe (YAML mapping with at least name, stimulus and requirements)."""
    return read_recipe(path)[0]


def read_recipe(path) -> tuple[Mapping, bytes]:
    """Like :func:`load_recipe`, plus the exact bytes that were parsed (for the record's checksum)."""
    data, raw = read_yaml(path)
    if not isinstance(data, Mapping):
        raise ConfigurationError(f"{path}: a recipe must be a YAML mapping")
    missing = REQUIRED - set(data or {})
    if missing:
        raise ConfigurationError(f"recipe missing fields: {sorted(missing)}")
    return data, raw


@dataclass(frozen=True)
class Spec:
    """Acceptance specification: requirements plus optional default analysis parameters."""

    name: str
    requirements: tuple[Requirement, ...]
    description: str = ""
    analysis: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    path: str | None = None
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description, "analysis": self.analysis,
            "parameters": dict(self.parameters), "path": self.path, "sha256": self.sha256,
        }


def load_spec(path) -> Spec:
    """Load a YAML spec: ``name``, ``requirements`` and optional ``description``, ``analysis``, ``parameters``."""
    data, raw = read_yaml(path)
    if not isinstance(data, Mapping):
        raise ConfigurationError(f"{path}: a spec must be a YAML mapping")
    unknown = sorted(set(data) - SPEC_KEYS)
    if unknown:
        raise ConfigurationError(f"{path}: unknown key(s) {unknown} (allowed: {', '.join(sorted(SPEC_KEYS))})")
    if "requirements" not in data:
        raise ConfigurationError(f"{path}: a spec needs a 'requirements' section")
    parameters = data.get("parameters") or {}
    if not isinstance(parameters, Mapping) or not all(isinstance(k, str) for k in parameters):
        raise ConfigurationError(f"{path}: 'parameters' must map parameter names to values")
    analysis = data.get("analysis")
    if analysis is not None and not isinstance(analysis, str):
        raise ConfigurationError(f"{path}: 'analysis' must be a string such as 'bode'")
    return Spec(
        name=str(data.get("name") or Path(path).stem),
        requirements=tuple(requirements_from_mapping(data["requirements"], source=str(path))),
        description=str(data.get("description") or ""),
        analysis=analysis,
        parameters=dict(parameters),
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def read_yaml(path) -> tuple[Any, bytes]:
    """Parse a UTF-8 YAML file; returns the data and the raw bytes (for hashing)."""
    p = Path(path)
    if not p.is_file():
        raise ConfigurationError(f"file not found: {p}")
    raw = p.read_bytes()
    try:
        return yaml.safe_load(raw.decode("utf-8-sig")), raw
    except UnicodeDecodeError as exc:
        raise ConfigurationError(f"{p}: not valid UTF-8 text") from exc
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}" if mark is not None else ""
        problem = getattr(exc, "problem", None) or str(exc)
        raise ConfigurationError(f"{p}: invalid YAML{where}: {problem}") from exc
