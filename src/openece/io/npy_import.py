"""NumPy ``.npy`` import.

Supported layouts (samples always run along axis 0):

- 1-D numeric array: one column, ``column_1``;
- 2-D numeric array of shape (samples, columns): ``column_1`` .. ``column_N``;
- 1-D structured array: one column per numeric field, named after the field.

``.npy`` files carry no units and no sample rate, so time information must come from a time
column or be supplied explicitly. Pickled/object arrays are refused (they can execute code).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..errors import DataImportError
from .dataset import Column, Dataset, SkippedColumn, SourceInfo

_NPY_MAGIC = b"\x93NUMPY"


def load_npy(path: str | Path) -> Dataset:
    """Load a ``.npy`` array as named numeric columns."""
    path = Path(path)
    source = SourceInfo.from_path(path, "npy")
    with path.open("rb") as fh:
        if fh.read(len(_NPY_MAGIC)) != _NPY_MAGIC:
            raise DataImportError(f"{path.name}: not a NumPy .npy file (missing NPY header)", hint="format")
    try:
        array = np.load(path, allow_pickle=False)
    except ValueError as exc:
        if "allow_pickle" in str(exc):
            raise DataImportError(
                f"{path.name}: contains Python objects, which are not loaded for security reasons; "
                "save a numeric (or structured numeric) array instead"
            ) from exc
        raise DataImportError(f"{path.name}: cannot read .npy data ({exc})") from exc
    except (OSError, EOFError) as exc:
        raise DataImportError(f"{path.name}: cannot read .npy data ({exc})") from exc

    metadata = {"format": "npy", "shape": list(array.shape), "dtype": str(array.dtype)}
    columns: list[Column] = []
    skipped: list[SkippedColumn] = []
    if array.dtype.names:
        if array.ndim != 1:
            raise DataImportError(f"{path.name}: structured arrays must be 1-D, got shape {array.shape}")
        for field in array.dtype.names:
            values = array[field]
            if values.ndim == 1 and _is_real_numeric(values.dtype):
                columns.append(Column(field, values.astype(np.float64)))
            else:
                skipped.append(SkippedColumn(field, f"field dtype {values.dtype} is not a real numeric scalar"))
    else:
        if not _is_real_numeric(array.dtype):
            raise DataImportError(
                f"{path.name}: unsupported dtype {array.dtype}; save real-valued data "
                "(for complex data, store magnitude/phase or real/imaginary parts as separate columns)"
            )
        if array.ndim == 1:
            array = array[:, np.newaxis]
        elif array.ndim != 2:
            raise DataImportError(f"{path.name}: expected a 1-D or 2-D array, got shape {array.shape}")
        rows, cols = array.shape
        if cols > rows:
            raise DataImportError(
                f"{path.name}: the array has more columns than rows ({rows} x {cols}); samples must run along "
                "axis 0 (shape samples x columns). Save the transposed array (array.T)."
            )
        columns = [Column(f"column_{k + 1}", array[:, k].astype(np.float64)) for k in range(cols)]
    if not columns:
        raise DataImportError(f"{path.name}: no numeric data found")
    if len(columns[0].values) == 0:
        raise DataImportError(f"{path.name}: the array is empty")
    return Dataset(source=source, columns=tuple(columns), skipped=tuple(skipped), metadata=metadata)


def _is_real_numeric(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.bool_)
