"""NumPy ``.npy`` import.

Supported layouts (samples always run along axis 0):

- 1-D numeric array: one column, ``column_1``;
- 2-D numeric array of shape (samples, columns): ``column_1`` .. ``column_N``;
- 1-D structured array: one column per numeric field, named after the field.

``.npy`` files carry no units and no sample rate, so time information must come from a time
column or be supplied explicitly. ``timedelta64`` values are converted to seconds and
``datetime64`` values to seconds since the first value (their units are part of the dtype).
The header's declared size is checked against the file before anything is allocated, and
pickled/object arrays are refused (they can execute code).
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

from ..errors import DataImportError
from .dataset import Column, Dataset, SkippedColumn, SourceInfo

_NPY_MAGIC = b"\x93NUMPY"


def load_npy(path: str | Path) -> Dataset:
    """Load a ``.npy`` array as named numeric columns."""
    path = Path(path)
    source, raw = SourceInfo.read(path, "npy")
    if not raw.startswith(_NPY_MAGIC):
        raise DataImportError(f"{path.name}: not a NumPy .npy file (missing NPY header)", hint="format")
    _check_declared_size(raw, path.name)
    try:
        array = np.load(io.BytesIO(raw), allow_pickle=False)
    except ValueError as exc:
        if "allow_pickle" in str(exc):
            raise DataImportError(
                f"{path.name}: contains Python objects, which are not loaded for security reasons; "
                "save a numeric (or structured numeric) array instead"
            ) from exc
        raise DataImportError(f"{path.name}: cannot read .npy data ({exc})") from exc
    except (OSError, EOFError, MemoryError) as exc:
        raise DataImportError(f"{path.name}: cannot read .npy data ({type(exc).__name__}: {exc})") from exc

    metadata = {"format": "npy", "shape": list(array.shape), "dtype": str(array.dtype)}
    if array.ndim == 0 or array.ndim > 2:
        raise DataImportError(f"{path.name}: expected a 1-D or 2-D array, got shape {array.shape}")
    if array.size == 0:
        raise DataImportError(f"{path.name}: the array is empty")
    columns: list[Column] = []
    skipped: list[SkippedColumn] = []
    if array.dtype.names:
        if array.ndim != 1:
            raise DataImportError(f"{path.name}: structured arrays must be 1-D, got shape {array.shape}")
        for field in array.dtype.names:
            values = array[field]
            item = _column(field, values) if values.ndim == 1 else SkippedColumn(
                field, f"field dtype {values.dtype} is not a scalar")
            (columns if isinstance(item, Column) else skipped).append(item)
    else:
        if array.ndim == 1:
            array = array[:, np.newaxis]
        rows, cols = array.shape
        if cols > rows:
            raise DataImportError(
                f"{path.name}: the array has more columns than rows ({rows} x {cols}); samples must run along "
                "axis 0 (shape samples x columns). Save the transposed array (array.T)."
            )
        items = [_column(f"column_{k + 1}", array[:, k]) for k in range(cols)]
        if any(isinstance(item, SkippedColumn) for item in items):
            raise DataImportError(
                f"{path.name}: unsupported dtype {array.dtype}; save real-valued data "
                "(for complex data, store magnitude/phase or real/imaginary parts as separate columns)"
            )
        columns = items
    if not columns:
        raise DataImportError(f"{path.name}: no numeric data found")
    return Dataset(source=source, columns=tuple(columns), skipped=tuple(skipped), metadata=metadata)


def _column(name: str, values: np.ndarray) -> Column | SkippedColumn:
    kind = values.dtype.kind
    if kind in "biuf":
        return Column(name, values.astype(np.float64))
    if kind == "m":  # timedelta64 (numpy counts it as an integer type; its unit is in the dtype)
        seconds = (values / np.timedelta64(1, "s")).astype(np.float64)
        return Column(name, seconds, "s", note=f"{values.dtype} converted to seconds")
    if kind == "M":  # datetime64
        seconds = ((values - values[0]) / np.timedelta64(1, "s")).astype(np.float64)
        return Column(name, seconds, "s", "datetime", note=f"{values.dtype} converted to seconds since {values[0]}")
    return SkippedColumn(name, f"dtype {values.dtype} is not a real numeric scalar")


def _check_declared_size(raw: bytes, name: str) -> None:
    """Reject files whose header promises more (or less) data than they contain."""
    buffer = io.BytesIO(raw)
    try:
        version = npy_format.read_magic(buffer)
        if version == (1, 0):
            shape, _, dtype = npy_format.read_array_header_1_0(buffer)
        elif version == (2, 0):
            shape, _, dtype = npy_format.read_array_header_2_0(buffer)
        else:
            return  # newer header versions are validated by np.load
    except ValueError as exc:
        raise DataImportError(f"{name}: invalid .npy header ({exc})") from exc
    if dtype.hasobject:
        return  # refused by np.load(allow_pickle=False) with its own message
    expected = math.prod(shape) * dtype.itemsize
    available = len(raw) - buffer.tell()
    if expected != available:
        raise DataImportError(
            f"{name}: the header declares an array of shape {tuple(shape)} and dtype {dtype} ({expected} bytes) "
            f"but the file holds {available} bytes of data; the file is corrupted or truncated"
        )
