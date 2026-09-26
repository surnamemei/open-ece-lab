"""User-facing error types.

Every error caused by user input (files, column mappings, units, specs, parameters) derives from
:class:`OpenECEError`. Front-ends (the CLI, a future UI) catch this base class and show the
message without a traceback, so messages must be specific and actionable on their own.

``hint`` names the parameter that resolves the problem (for example ``"time_unit"`` or a column
role such as ``"time"``). It is deliberately front-end neutral: the CLI maps it to a flag such as
``--time-unit``; a UI could map it to a form field.
"""
from __future__ import annotations

from collections.abc import Sequence


class OpenECEError(Exception):
    """Base class for expected, user-facing errors."""

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


class DataImportError(OpenECEError, ValueError):
    """An input file could not be read or interpreted."""


class ColumnMappingError(DataImportError):
    """Columns required by an analysis could not be mapped unambiguously."""

    def __init__(
        self,
        message: str,
        *,
        role: str | None = None,
        candidates: Sequence[str] = (),
        available: Sequence[str] = (),
        hint: str | None = None,
    ):
        super().__init__(message, hint=hint or role)
        self.role = role
        self.candidates = tuple(candidates)
        self.available = tuple(available)


class AmbiguousColumnError(ColumnMappingError):
    """More than one column could fill a role, so an explicit mapping is required."""


class MissingColumnError(ColumnMappingError):
    """No column can fill a required role."""


class ConfigurationError(OpenECEError, ValueError):
    """A recipe, spec, requirement or analysis parameter is invalid."""


class AnalysisError(OpenECEError, ValueError):
    """The data cannot be analysed with the requested parameters."""


class RecordError(OpenECEError):
    """A measurement record could not be created, found or read."""
