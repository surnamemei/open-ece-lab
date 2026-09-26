"""Persist an analysis outcome as a run: data tables, figures, HTML report, then the JSON record."""
from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..records import Artifact, RunRecord, RunStore, software_info
from ..reporting import render_run_report
from .outcome import AnalysisOutcome, DataTable

REPORT_FILENAME = "report.html"


@dataclass(frozen=True)
class SavedRun:
    record: RunRecord
    directory: Path

    @property
    def record_path(self) -> Path:
        return self.directory / "record.json"

    @property
    def report_path(self) -> Path:
        return self.directory / REPORT_FILENAME


def save_run(outcome: AnalysisOutcome, store: RunStore, *, invocation: Mapping[str, Any] | None = None,
             now: datetime | None = None) -> SavedRun:
    """Create a new run directory and write every artifact and the record. Never overwrites."""
    run = store.create_run(outcome.analysis, now=now)
    artifacts: list[Artifact] = []
    origin = _origin_comment(outcome)
    for filename, table in outcome.tables.items():
        text = _table_csv(table, [f"OpenECE Lab run {run.run_id}: {table.description}", origin])
        artifacts.append(Artifact(run.write_text(filename, text), "data", "text/csv", table.description))

    pngs: dict[str, bytes] = {}
    if outcome.figures is not None:
        from ..reporting.plots import figure_to_png

        for filename, (description, figure) in outcome.figures().items():
            pngs[filename] = figure_to_png(figure)
            artifacts.append(Artifact(run.write_bytes(filename, pngs[filename]), "figure", "image/png", description))

    artifacts.append(Artifact(REPORT_FILENAME, "report", "text/html", "human-readable report"))
    record = RunRecord(
        run_id=run.run_id,
        created_utc=run.created.isoformat(),
        analysis={"type": outcome.analysis, "title": outcome.title, "parameters": outcome.parameters},
        source=outcome.source,
        results={name: m.to_dict() for name, m in outcome.results.items()},
        validation=outcome.validation_dict(),
        artifacts=tuple(artifacts),
        software=software_info(),
        warnings=outcome.warnings,
        invocation=invocation,
    )
    run.write_text(REPORT_FILENAME, render_run_report(record.to_dict(), pngs))
    run.write_record(record)
    return SavedRun(record, run.path)


def _origin_comment(outcome: AnalysisOutcome) -> str:
    if outcome.simulated:
        return "SIMULATED DATA from a mock instrument backend - not a physical measurement"
    source = outcome.source
    if source.get("kind") == "file":
        return f"derived from {source.get('path')} (sha256 {source.get('sha256')})"
    return f"source: {source.get('kind', 'unknown')}"


def _table_csv(table: DataTable, comments: list[str]) -> str:
    buffer = io.StringIO()
    for comment in comments:
        buffer.write(f"# {comment}\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(list(table.columns))
    for row in zip(*table.columns.values()):
        writer.writerow([repr(float(v)) for v in row])
    return buffer.getvalue()
