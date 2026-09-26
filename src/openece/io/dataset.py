"""Data model returned by every importer: named numeric columns plus provenance and metadata."""
from __future__ import annotations

import difflib
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import AmbiguousColumnError, DataImportError, MissingColumnError
from ..units import normalize_name, parse_header


@dataclass(frozen=True)
class SourceInfo:
    """Where imported data came from; the SHA-256 identifies the exact bytes that were analysed."""

    path: str
    absolute_path: str
    format: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path, fmt: str, data: bytes | None = None) -> SourceInfo:
        p = Path(path)
        if not p.exists():
            raise DataImportError(f"input file not found: {p}")
        if not p.is_file():
            raise DataImportError(f"not a regular file: {p}")
        digest = hashlib.sha256()
        size = 0
        try:
            if data is None:
                with p.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        digest.update(chunk)
                        size += len(chunk)
            else:
                digest.update(data)
                size = len(data)
        except OSError as exc:
            raise DataImportError(f"cannot read {p}: {exc.strerror or exc}") from exc
        return cls(str(path), str(p.resolve()), fmt, size, digest.hexdigest())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "file",
            "path": self.path,
            "absolute_path": self.absolute_path,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, eq=False)
class Column:
    """One numeric column (CSV/NPY) or channel (WAV).

    ``values`` are float64; blank cells are NaN and their row indices are listed in
    ``blank_rows``. ``kind`` is ``"datetime"`` for ISO 8601 text converted to seconds.
    """

    name: str
    values: np.ndarray
    unit: str | None = None
    kind: str = "numeric"
    blank_rows: tuple[int, ...] = ()
    note: str | None = None

    @property
    def base_name(self) -> str:
        return parse_header(self.name).base


@dataclass(frozen=True)
class SkippedColumn:
    """A column that is present in the file but cannot be used as numeric data."""

    name: str
    reason: str
    hint: str | None = None


@dataclass(frozen=True, eq=False)
class Dataset:
    """Imported data: numeric columns, skipped columns, provenance and format metadata.

    ``sample_rate_hz`` is set only when the format defines one (WAV). ``row_lines`` maps row
    indices to 1-based file line numbers for text formats so errors can point at the file.
    """

    source: SourceInfo
    columns: tuple[Column, ...]
    skipped: tuple[SkippedColumn, ...] = ()
    sample_rate_hz: float | None = None
    row_lines: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def n_rows(self) -> int:
        return len(self.columns[0].values) if self.columns else 0

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def locate(self, index: int) -> str:
        """Human-readable location of row ``index`` ("line 12" for text files)."""
        if self.row_lines is not None and 0 <= index < len(self.row_lines):
            return f"line {int(self.row_lines[index])}"
        return f"sample {index}"

    def find_column(self, key: str, *, role: str | None = None) -> Column:
        """Look up a column by exact name, case-insensitive name, or unique base name."""
        for col in self.columns:
            if col.name == key:
                return col
        self._reject_skipped(key, lambda name: name == key)
        folded = key.casefold()
        matches = [c for c in self.columns if c.name.casefold() == folded]
        if len(matches) == 1:
            return matches[0]
        self._reject_skipped(key, lambda name: name.casefold() == folded)
        wanted = normalize_name(key)
        matches = [
            c for c in self.columns
            if wanted and wanted in (normalize_name(c.name), normalize_name(c.base_name))
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = [c.name for c in matches]
            raise AmbiguousColumnError(
                f"{self.source.path}: {key!r} matches several columns: {_quoted(names)}; use the full column name",
                role=role, candidates=names, available=self.column_names,
            )
        all_names = self.column_names + [s.name for s in self.skipped]
        close = difflib.get_close_matches(key, all_names, n=3, cutoff=0.6)
        suggestion = f" Did you mean {_quoted(close)}?" if close else ""
        raise MissingColumnError(
            f"{self.source.path}: column {key!r} not found. Available columns: {_quoted(all_names)}.{suggestion}",
            role=role, available=self.column_names,
        )

    def _reject_skipped(self, key: str, predicate) -> None:
        for skipped in self.skipped:
            if predicate(skipped.name):
                raise DataImportError(
                    f"{self.source.path}: column {skipped.name!r} cannot be used: {skipped.reason}",
                    hint=skipped.hint,
                )


def _quoted(names) -> str:
    return ", ".join(repr(n) for n in names) if names else "(none)"
