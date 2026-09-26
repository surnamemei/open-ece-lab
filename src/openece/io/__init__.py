"""File import: CSV/TSV, WAV and NPY into a common :class:`Dataset`.

This package only reads and describes data. It must not import analysis, workflow, reporting,
instrument or CLI code (enforced by ``tests/test_architecture.py``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import DataImportError
from .columns import ColumnAssignment, ColumnRole, resolve_columns
from .csv_import import load_csv
from .dataset import Column, Dataset, SkippedColumn, SourceInfo
from .npy_import import load_npy
from .wav_import import load_wav

FORMATS_BY_SUFFIX = {".csv": "csv", ".tsv": "csv", ".txt": "csv", ".dat": "csv", ".wav": "wav", ".npy": "npy"}
FORMATS = ("csv", "wav", "npy")

__all__ = [
    "Column", "ColumnAssignment", "ColumnRole", "Dataset", "SkippedColumn", "SourceInfo",
    "FORMATS", "detect_format", "load_csv", "load_file", "load_npy", "load_wav", "resolve_columns",
]


def detect_format(path: str | Path) -> str:
    """Format name from the file extension."""
    suffix = Path(path).suffix.lower()
    try:
        return FORMATS_BY_SUFFIX[suffix]
    except KeyError:
        known = ", ".join(sorted(FORMATS_BY_SUFFIX))
        raise DataImportError(
            f"cannot tell the format of {Path(path).name!r} from its extension (supported: {known}); "
            "specify the format explicitly",
            hint="format",
        ) from None


def load_file(path: str | Path, *, format: str | None = None, csv_options: dict[str, Any] | None = None) -> Dataset:
    """Load any supported file. ``csv_options`` are passed to :func:`load_csv`."""
    fmt = format or detect_format(path)
    options = {k: v for k, v in (csv_options or {}).items() if v is not None}
    if fmt == "csv":
        return load_csv(path, **options)
    if options:
        raise DataImportError(
            f"CSV import options ({', '.join(sorted(options))}) do not apply to {fmt.upper()} files", hint="format"
        )
    if fmt == "wav":
        return load_wav(path)
    if fmt == "npy":
        return load_npy(path)
    raise DataImportError(f"unsupported format {fmt!r} (supported: {', '.join(FORMATS)})", hint="format")
