"""Persistent, append-only measurement records (run provenance) stored as JSON.

Every run gets its own directory under a runs root, named after its run ID::

    runs/20260926T170412Z-bode-3f9a2c1d/
        record.json     machine-readable record (schema "openece.run-record", version 1)
        report.html     human-readable report
        *.png, *.csv    figures and data artifacts

Run directories are created exclusively and every file is opened in exclusive-create mode, so
an existing run can never be overwritten. ``record.json`` is written last and atomically; a run
directory without it is incomplete. Non-finite numbers are stored as ``null`` so the files are
strict JSON.
"""
from __future__ import annotations

import copy
import functools
import json
import math
import os
import platform
import re
import secrets
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from . import __version__
from .errors import RecordError

SCHEMA = "openece.run-record"
SCHEMA_VERSION = 1
RECORD_FILENAME = "record.json"
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def to_jsonable(value: Any) -> Any:
    """Convert records to plain JSON types; NaN/inf become ``None``."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    raise TypeError(f"cannot store a {type(value).__name__} in a run record")


@dataclass(frozen=True)
class Artifact:
    """A file produced by a run; ``path`` is relative to the run directory."""

    path: str
    kind: str
    media_type: str
    description: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "kind": self.kind, "media_type": self.media_type, "description": self.description}


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    created_utc: str
    analysis: Mapping[str, Any]
    source: Mapping[str, Any]
    results: Mapping[str, Any]
    validation: Mapping[str, Any]
    artifacts: tuple[Artifact, ...]
    software: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
    invocation: Mapping[str, Any] | None = None
    status: str = "completed"

    @property
    def overall(self) -> str:
        return str(self.validation.get("overall", "NOT_EVALUATED"))

    def artifact(self, kind: str) -> Artifact | None:
        return next((a for a in self.artifacts if a.kind == kind), None)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable({
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "status": self.status,
            "analysis": self.analysis,
            "source": self.source,
            "results": self.results,
            "validation": self.validation,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "warnings": list(self.warnings),
            "software": self.software,
            "invocation": self.invocation,
        })

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunRecord:
        if not isinstance(data, Mapping) or data.get("schema") != SCHEMA:
            raise RecordError("not an OpenECE run record (missing schema 'openece.run-record')")
        version = data.get("schema_version")
        if not isinstance(version, int) or version < 1 or version > SCHEMA_VERSION:
            raise RecordError(
                f"unsupported record schema_version {version!r} (this software reads up to {SCHEMA_VERSION})"
            )
        try:
            return cls(
                run_id=data["run_id"],
                created_utc=data["created_utc"],
                analysis=data["analysis"],
                source=data["source"],
                results=data["results"],
                validation=data["validation"],
                artifacts=tuple(Artifact(**a) for a in data.get("artifacts", [])),
                software=data.get("software", {}),
                warnings=tuple(data.get("warnings", [])),
                invocation=data.get("invocation"),
                status=data.get("status", "completed"),
            )
        except (KeyError, TypeError) as exc:
            raise RecordError(f"malformed run record: {exc}") from exc


def load_record(path: str | Path) -> RunRecord:
    """Read a ``record.json`` file (or the record inside a run directory)."""
    p = Path(path)
    if p.is_dir():
        p = p / RECORD_FILENAME
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RecordError(f"no run record at {p}") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordError(f"cannot read run record {p}: {exc}") from exc
    return RunRecord.from_dict(data)


def _random_token() -> str:
    return secrets.token_hex(4)


def make_run_id(analysis: str, created: datetime) -> str:
    """``YYYYMMDDTHHMMSSZ-<analysis>-<8 hex>``: sortable, file-name safe on Windows and Linux."""
    return f"{created.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{analysis}-{_random_token()}"


@dataclass(frozen=True)
class RunDirectory:
    """A freshly created run directory. Files can be added but never replaced."""

    run_id: str
    path: Path
    created: datetime

    def write_bytes(self, name: str, data: bytes) -> str:
        if not _FILE_NAME.match(name) or name.startswith(RECORD_FILENAME):
            raise ValueError(f"invalid artifact file name {name!r}")
        target = self.path / name
        try:
            with open(target, "xb") as fh:
                fh.write(data)
        except FileExistsError:
            raise RecordError(f"refusing to overwrite existing file {target}") from None
        except OSError as exc:
            raise RecordError(f"cannot write {target}: {exc.strerror or exc}") from exc
        return PurePosixPath(name).as_posix()

    def write_text(self, name: str, text: str) -> str:
        return self.write_bytes(name, text.encode("utf-8"))

    def write_record(self, record: RunRecord) -> Path:
        final = self.path / RECORD_FILENAME
        partial = self.path / (RECORD_FILENAME + ".partial")
        if final.exists():
            raise RecordError(f"refusing to overwrite existing record {final}")
        try:
            with open(partial, "xb") as fh:
                fh.write(record.to_json().encode("utf-8"))
            os.replace(partial, final)
        except OSError as exc:
            raise RecordError(f"cannot write run record {final}: {exc.strerror or exc}") from exc
        return final


class RunStore:
    """A directory of run directories."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def create_run(self, analysis: str, *, now: datetime | None = None) -> RunDirectory:
        if not _SLUG.match(analysis):
            raise ValueError(f"invalid analysis name for a run ID: {analysis!r}")
        created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RecordError(f"cannot create runs directory {self.root}: {exc.strerror or exc}") from exc
        for _ in range(16):
            run_id = make_run_id(analysis, created)
            try:
                (self.root / run_id).mkdir()
            except FileExistsError:
                continue  # never reuse an existing run directory
            except OSError as exc:
                raise RecordError(f"cannot create run directory in {self.root}: {exc.strerror or exc}") from exc
            return RunDirectory(run_id, self.root / run_id, created)
        raise RecordError(f"could not allocate a unique run directory in {self.root}")

    def run_ids(self) -> list[str]:
        """IDs of complete runs (directories containing a record), oldest first."""
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / RECORD_FILENAME).is_file())

    def resolve(self, run_id: str) -> Path:
        """Directory of a run given its ID or a unique prefix of it."""
        ids = self.run_ids()
        if run_id in ids:
            return self.root / run_id
        matches = [i for i in ids if i.startswith(run_id)] if run_id else []
        if len(matches) == 1:
            return self.root / matches[0]
        if matches:
            raise RecordError(f"run ID prefix {run_id!r} is ambiguous: {', '.join(matches)}")
        raise RecordError(f"no run {run_id!r} in {self.root}")

    def load(self, run_id: str) -> RunRecord:
        return load_record(self.resolve(run_id))

    def records(self) -> tuple[list[RunRecord], list[str]]:
        """All readable records (oldest first) plus messages for unreadable ones."""
        records, problems = [], []
        for run_id in self.run_ids():
            try:
                records.append(load_record(self.root / run_id))
            except RecordError as exc:
                problems.append(f"{run_id}: {exc}")
        return records, problems


def software_info() -> dict[str, Any]:
    """Versions of OpenECE Lab, Python and key dependencies, plus the git revision if available."""
    return copy.deepcopy(_software_info())


@functools.lru_cache(maxsize=1)
def _software_info() -> dict[str, Any]:
    dependencies = {}
    for dist in ("numpy", "scipy", "matplotlib", "PyYAML"):
        try:
            dependencies[dist] = importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            dependencies[dist] = None
    info: dict[str, Any] = {
        "name": "open-ece-lab",
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dependencies,
    }
    git = _git_revision(Path(__file__).resolve().parent)
    if git is not None:
        info["git"] = git
    return info


def _git_revision(package_dir: Path) -> dict[str, Any] | None:
    """Commit of a source checkout; ``dirty`` flags uncommitted changes in the package itself."""
    if {"site-packages", "dist-packages"} & set(package_dir.parts):
        return None  # an installed copy: the enclosing repository (if any) says nothing about it
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=package_dir, capture_output=True, text=True, timeout=5, check=False
        )
        if head.returncode != 0:
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "."],
            cwd=package_dir, capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return {"commit": head.stdout.strip(), "dirty": dirty}
