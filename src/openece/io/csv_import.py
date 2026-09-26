"""Delimited-text (CSV/TSV) import.

Every detection decision is recorded in ``Dataset.metadata`` so it ends up in the run record:

- lines starting with the comment prefix (default ``#``) are skipped and kept as comments;
- the delimiter is detected among comma, semicolon and tab (then whitespace) unless given;
- the first non-comment row is the header unless all of its cells are numeric;
- a row directly under the header whose cells are all short non-numeric tokens is read as a
  units row (PicoScope exports ``Time,Channel A`` followed by ``(ms),(V)``);
- a column is numeric when every non-blank cell parses as a number. ISO 8601 date-time text is
  converted to seconds since its first value. Anything else is reported as a skipped column
  with the first offending line; nothing is silently coerced or dropped.
"""
from __future__ import annotations

import codecs
import csv
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from ..errors import DataImportError
from ..units import parse_header
from .dataset import Column, Dataset, SkippedColumn, SourceInfo

_DELIMITER_ALIASES = {
    "comma": ",", "semicolon": ";", "tab": "\t", "\\t": "\t", "whitespace": "whitespace", "space": "whitespace",
}
_DELIMITER_LABELS = {",": "comma", ";": "semicolon", "\t": "tab", "whitespace": "whitespace"}
_CANDIDATE_DELIMITERS = (",", ";", "\t")
# A consistent tab or semicolon split beats a comma split: commas are then decimal commas.
_DELIMITER_PRIORITY = {"\t": 2, ";": 1, ",": 0}
_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),  # before UTF-16: same first bytes
    (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16"),
)
_INTEGER = re.compile(r"^[+-]?\d+$")
_SNIFF_ROWS = 50
_MAX_COMMENTS = 100
_MAX_UNIT_LENGTH = 16
_DECIMAL_COMMA = re.compile(r"^[+-]?\d+,\d+(?:[eE][+-]?\d+)?$")
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def load_csv(
    path: str | Path,
    *,
    delimiter: str | None = None,
    decimal: str = ".",
    skip_rows: int = 0,
    header: bool | None = None,
    comment: str | None = "#",
    encoding: str | None = None,
) -> Dataset:
    """Load a delimited text file into a :class:`Dataset`.

    ``delimiter`` may be a single character or one of ``comma``, ``semicolon``, ``tab``,
    ``whitespace``; ``None`` detects it. ``header=None`` detects whether a header row exists.
    ``skip_rows`` skips that many physical lines first (for metadata preambles).
    """
    path = Path(path)
    name = path.name
    delimiter = _normalize_delimiter(delimiter)
    if decimal not in (".", ","):
        raise DataImportError(f"decimal separator must be '.' or ',', got {decimal!r}", hint="decimal")
    if delimiter == "," and decimal == ",":
        raise DataImportError("the delimiter and the decimal separator cannot both be ','", hint="delimiter")
    if isinstance(skip_rows, bool) or not isinstance(skip_rows, int) or skip_rows < 0:
        raise DataImportError(f"skip_rows must be a non-negative integer, got {skip_rows!r}", hint="skip_rows")
    if header not in (None, True, False):
        raise DataImportError(f"header must be True, False or None (auto), got {header!r}")

    source, raw = SourceInfo.read(path, "csv")
    text, used_encoding, warnings = _decode(raw, encoding, name)

    comments: list[str] = []
    content: list[tuple[int, str]] = []
    for lineno, line in enumerate(_LINE_BREAK.split(text), start=1):
        if lineno <= skip_rows:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if comment and stripped.startswith(comment):
            comments.append(stripped[len(comment):].strip())
            continue
        content.append((lineno, line))
    if not content:
        raise DataImportError(f"{name}: no data found (the file is empty or contains only comments and blank lines)")

    if delimiter is None:
        delimiter = _detect_delimiter([t for _, t in content[:_SNIFF_ROWS]], decimal)
    rows = _split_rows(content, delimiter, name)
    if not rows:
        raise DataImportError(f"{name}: no data found (all rows are empty)")

    first_line, first_cells = rows[0]
    if header is None:
        filled = [c for c in first_cells if c]
        header = not all(_parse_number(c, decimal) is not None for c in filled)
    if header:
        header_cells, data_rows, header_line = first_cells, rows[1:], first_line
    else:
        header_cells, data_rows, header_line = [""] * len(first_cells), rows, None
    if not data_rows:
        raise DataImportError(f"{name}: the file has a header (line {header_line}) but no data rows")

    units_cells: list[str] | None = None
    units_line: int | None = None
    if header and len(data_rows) >= 2 and _looks_like_units_row(data_rows[0][1], data_rows[1][1], decimal):
        units_line, units_cells = data_rows[0]
        data_rows = data_rows[1:]
        listed = ", ".join(c for c in units_cells if c)
        warnings.append(f"line {units_line} was read as a units row ({listed})")

    ncols = len(header_cells)
    data_rows = [(lineno, _fit_width(cells, ncols, lineno, header_line or first_line, name)) for lineno, cells in data_rows]
    if units_cells is not None:
        units_cells = (units_cells + [""] * ncols)[:ncols]
    row_lines = np.array([lineno for lineno, _ in data_rows], dtype=np.int64)

    columns: list[Column] = []
    skipped: list[SkippedColumn] = []
    seen: set[str] = set()
    for k in range(ncols):
        cells = [cells[k] for _, cells in data_rows]
        if header:
            col_name = header_cells[k].strip()
            if not col_name:
                if not any(cells):
                    continue  # empty column created by a trailing delimiter
                col_name = f"column_{k + 1}"
                warnings.append(f"column {k + 1} has no header; it is named {col_name!r}")
        else:
            col_name = f"column_{k + 1}"
        unique = _unique_name(col_name, seen)
        if unique != col_name:
            warnings.append(f"duplicate column name {col_name!r}; the later column is named {unique!r}")
        col_name = unique

        unit = _clean_unit(units_cells[k]) if units_cells else None
        unit = unit or parse_header(col_name).unit
        values, blanks, bad, first_bad = _parse_numbers(cells, decimal)
        filled = len(cells) - len(blanks)
        if filled == 0:
            skipped.append(SkippedColumn(col_name, "the column is empty"))
        elif bad == 0:
            columns.append(Column(col_name, values, unit, "numeric", tuple(blanks)))
        else:
            converted = _parse_datetimes(cells) if bad == filled and not blanks else None
            if converted is not None:
                seconds, origin = converted
                columns.append(Column(
                    col_name, seconds, "s", "datetime",
                    note=f"ISO 8601 date-time converted to seconds since {origin}",
                ))
            else:
                skipped.append(_describe_bad_column(col_name, cells, row_lines, bad, filled, first_bad, decimal))

    if (not header and delimiter == "," and decimal == "." and len(columns) == 2 and not skipped
            and all(_INTEGER.match(c) for _, cells in data_rows for c in cells if c)):
        warnings.append(
            f"{name} has no header and two integer columns; if the values use a decimal comma "
            "(e.g. '2,309' meaning 2.309), set the decimal separator to ','"
        )
    if not columns:
        details = "; ".join(f"{s.name!r}: {s.reason}" for s in skipped) or "no columns"
        if any(s.hint == "decimal" for s in skipped):
            advice, hint = "Set the decimal separator to ',' for this file.", "decimal"
        else:
            advice = ("If the file starts with a metadata preamble, skip those lines (skip_rows) "
                      "or prefix them with '#'.")
            hint = "skip_rows"
        raise DataImportError(f"{name}: no numeric columns found ({details}). {advice}", hint=hint)

    metadata = {
        "format": "csv",
        "encoding": used_encoding,
        "delimiter": _DELIMITER_LABELS.get(delimiter, delimiter),
        "decimal": decimal,
        "skip_rows": skip_rows,
        "header_line": header_line,
        "units_line": units_line,
        "first_data_line": int(row_lines[0]),
        "last_data_line": int(row_lines[-1]),
        "rows": len(data_rows),
        "comments": comments[:_MAX_COMMENTS],
    }
    if len(comments) > _MAX_COMMENTS:
        metadata["comments_truncated"] = len(comments)
    return Dataset(
        source=source,
        columns=tuple(columns),
        skipped=tuple(skipped),
        sample_rate_hz=None,
        row_lines=row_lines,
        metadata=metadata,
        warnings=tuple(warnings),
    )


def _normalize_delimiter(delimiter: str | None) -> str | None:
    if delimiter is None:
        return None
    resolved = _DELIMITER_ALIASES.get(delimiter.lower(), delimiter)
    if resolved in ('"', "\r", "\n"):
        raise DataImportError(f"invalid delimiter {delimiter!r}", hint="delimiter")
    if resolved != "whitespace" and len(resolved) != 1:
        raise DataImportError(
            f"invalid delimiter {delimiter!r}: use a single character or one of comma, semicolon, tab, whitespace",
            hint="delimiter",
        )
    return resolved


def _decode(raw: bytes, encoding: str | None, name: str) -> tuple[str, str, list[str]]:
    if encoding:
        try:
            text = raw.decode(encoding)
            return (text[1:] if text.startswith("\ufeff") else text), encoding, []
        except LookupError as exc:
            raise DataImportError(f"unknown text encoding {encoding!r}", hint="encoding") from exc
        except UnicodeDecodeError as exc:
            raise DataImportError(
                f"{name}: cannot be decoded as {encoding} ({exc.reason} at byte {exc.start})", hint="encoding"
            ) from exc
    for bom, codec in _BOMS:
        if raw.startswith(bom):
            try:
                return raw.decode(codec), codec, []
            except UnicodeDecodeError as exc:
                raise DataImportError(
                    f"{name}: cannot be decoded as {codec.upper()} ({exc.reason}); the file may be truncated",
                    hint="encoding",
                ) from exc
    if b"\x00" in raw[:4096]:
        raise DataImportError(
            f"{name}: does not look like a text file (it contains NUL bytes); check the file type", hint="format"
        )
    try:
        return raw.decode("utf-8-sig"), "utf-8", []
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1", [
            f"{name} is not valid UTF-8 and was decoded as Latin-1; set the encoding explicitly if "
            "non-ASCII text (units such as µs or °) looks wrong"
        ]


def _detect_delimiter(lines: list[str], decimal: str) -> str:
    candidates = [d for d in _CANDIDATE_DELIMITERS if not (decimal == "," and d == ",")]
    best, best_score = None, None
    for delim in candidates:
        try:
            counts = [len(cells) for cells in csv.reader(lines, delimiter=delim)]
        except csv.Error:
            continue  # e.g. a whole line exceeds the csv field limit when split this way
        if not counts or max(counts) <= 1:
            continue
        mode = max(set(counts), key=counts.count)
        score = (min(counts) == max(counts), counts.count(mode) / len(counts), _DELIMITER_PRIORITY[delim], mode)
        if best_score is None or score > best_score:
            best, best_score = delim, score
    if best is not None:
        return best
    widths = [len(line.split()) for line in lines]
    if max(widths) > 1 and min(widths) == max(widths):
        return "whitespace"
    return candidates[0]  # a single column: the delimiter does not matter


def _split_rows(content: list[tuple[int, str]], delimiter: str, name: str) -> list[tuple[int, list[str]]]:
    rows: list[tuple[int, list[str]]] = []
    if delimiter == "whitespace":
        for lineno, text in content:
            rows.append((lineno, text.split()))
        return rows
    try:
        reader = csv.reader((text for _, text in content), delimiter=delimiter, skipinitialspace=True)
    except (TypeError, ValueError, csv.Error) as exc:
        raise DataImportError(f"invalid delimiter {delimiter!r}: {exc}", hint="delimiter") from exc
    try:
        for index, cells in enumerate(reader):
            if reader.line_num != index + 1:
                raise DataImportError(
                    f"{name}: line {content[index][0]}: unbalanced quote (quoted fields spanning several lines are not supported)"
                )
            cells = [c.strip() for c in cells]
            if any(cells):
                rows.append((content[index][0], cells))
    except csv.Error as exc:
        lineno = content[min(max(reader.line_num, 1), len(content)) - 1][0]  # the line being read
        raise DataImportError(f"{name}: line {lineno}: {exc}; check the delimiter", hint="delimiter") from exc
    return rows


def _fit_width(cells: list[str], ncols: int, lineno: int, reference_line: int, name: str) -> list[str]:
    if len(cells) > ncols:
        if any(cells[ncols:]):
            raise DataImportError(
                f"{name}: line {lineno} has {len(cells)} fields but line {reference_line} has {ncols}. "
                "Check the delimiter and decimal separator, or whether the file has a preamble to skip.",
                hint="delimiter",
            )
        return cells[:ncols]
    if len(cells) < ncols:
        return cells + [""] * (ncols - len(cells))
    return cells


def _looks_like_units_row(cells: list[str], next_cells: list[str], decimal: str) -> bool:
    filled = [c for c in cells if c]
    return (
        bool(filled)
        and all(_parse_number(c, decimal) is None and len(c) <= _MAX_UNIT_LENGTH for c in filled)
        and any(_parse_number(c, decimal) is not None for c in next_cells if c)
    )


def _clean_unit(cell: str) -> str | None:
    text = cell.strip()
    if len(text) >= 2 and text[0] + text[-1] in ("()", "[]"):
        text = text[1:-1].strip()
    return text or None


def _unique_name(name: str, seen: set[str]) -> str:
    candidate, suffix = name, 2
    while candidate in seen:
        candidate = f"{name}_{suffix}"
        suffix += 1
    seen.add(candidate)
    return candidate


def _parse_number(cell: str, decimal: str) -> float | None:
    try:
        return float(cell.replace(",", ".") if decimal == "," else cell)
    except ValueError:
        return None


def _parse_numbers(cells: list[str], decimal: str) -> tuple[np.ndarray, list[int], int, int]:
    """Parse a column. Returns values (NaN for blanks/bad cells), blank rows, bad count, first bad row."""
    out = [float("nan")] * len(cells)
    blanks: list[int] = []
    bad, first_bad = 0, -1
    comma = decimal == ","
    for i, cell in enumerate(cells):
        if not cell:
            blanks.append(i)
            continue
        try:
            out[i] = float(cell.replace(",", ".") if comma else cell)
        except ValueError:
            bad += 1
            if first_bad < 0:
                first_bad = i
    return np.asarray(out, dtype=np.float64), blanks, bad, first_bad


def _parse_datetimes(cells: list[str]) -> tuple[np.ndarray, str] | None:
    try:
        stamps = [datetime.fromisoformat(c) for c in cells]
    except ValueError:
        return None
    if len({s.tzinfo is None for s in stamps}) > 1:
        return None
    origin = stamps[0]
    return np.array([(s - origin).total_seconds() for s in stamps], dtype=np.float64), origin.isoformat()


def _describe_bad_column(name, cells, row_lines, bad, filled, first_bad, decimal) -> SkippedColumn:
    line = int(row_lines[first_bad])
    example = cells[first_bad]
    if bad == filled:
        reason = f"it is not numeric (for example {example!r} at line {line})"
    else:
        reason = f"{bad} of {filled} values are not numeric (first at line {line}: {example!r})"
    if decimal == "." and _DECIMAL_COMMA.match(example):
        return SkippedColumn(name, reason + "; the values look like they use a decimal comma", hint="decimal")
    return SkippedColumn(name, reason)
