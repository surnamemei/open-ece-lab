import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import openece
from openece import records
from openece.errors import RecordError
from openece.records import RECORD_FILENAME, RunRecord, RunStore, load_record, software_info, to_jsonable
from openece.validation import Requirement
from openece.workflows import AnalysisOutcome, DataTable, Measurement, apply_requirements, save_run

FIXED_TIME = datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc)
REQUIRED_KEYS = {
    "schema", "schema_version", "run_id", "created_utc", "status", "analysis", "source", "results",
    "validation", "artifacts", "warnings", "software", "invocation",
}


def make_outcome(**overrides):
    fields = dict(
        analysis="bode",
        title="Unit-test run",
        source={"kind": "file", "path": "in.csv", "sha256": "0" * 64, "format": "csv"},
        parameters={"drop_db": 3.0, "columns": {"frequency": {"column": "f", "selected_by": "explicit"}}},
        results={
            "cutoff_hz": Measurement(1591.5, "Hz", "cutoff"),
            "phase_at_cutoff_deg": Measurement(math.nan, "deg", "phase", note="no phase column"),
        },
        warnings=("a warning",),
        tables={"data.csv": DataTable("test table", {"f (Hz)": np.array([1.0, 2.0]), "m (dB)": np.array([0.0, -3.0])})},
    )
    fields.update(overrides)
    return AnalysisOutcome(**fields)


def test_record_contains_every_required_field(tmp_path):
    outcome = apply_requirements(make_outcome(), [Requirement("cutoff_hz", 1450, 1750)])
    saved = save_run(outcome, RunStore(tmp_path / "runs"), invocation={"argv": ["analyze-bode", "in.csv"]})
    text = saved.record_path.read_text(encoding="utf-8")
    data = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))
    assert REQUIRED_KEYS <= set(data)
    assert data["schema"] == "openece.run-record" and data["schema_version"] == 1
    assert data["run_id"] == saved.directory.name
    assert datetime.fromisoformat(data["created_utc"]).tzinfo is not None
    assert data["analysis"]["type"] == "bode" and data["analysis"]["parameters"]["drop_db"] == 3.0
    assert data["source"]["path"] == "in.csv"
    assert data["results"]["cutoff_hz"] == {"value": 1591.5, "unit": "Hz", "description": "cutoff"}
    assert data["results"]["phase_at_cutoff_deg"]["value"] is None  # NaN is stored as null
    assert data["validation"]["overall"] == "PASS"
    assert data["validation"]["checks"][0]["status"] == "PASS"
    assert data["software"]["version"] == openece.__version__
    assert data["warnings"] == ["a warning"]
    assert data["invocation"] == {"argv": ["analyze-bode", "in.csv"]}
    for artifact in data["artifacts"]:
        assert (saved.directory / artifact["path"]).is_file()
    assert {a["kind"] for a in data["artifacts"]} == {"data", "report"}
    assert (saved.directory / "data.csv").read_text().startswith(f"# OpenECE Lab run {saved.record.run_id}")


def test_runs_with_colliding_ids_never_overwrite(tmp_path, monkeypatch):
    tokens = iter(["aaaa0000", "aaaa0000", "bbbb1111"])
    monkeypatch.setattr(records, "_random_token", lambda: next(tokens))
    store = RunStore(tmp_path)
    first = save_run(make_outcome(), store, now=FIXED_TIME)
    before = first.record_path.read_bytes()
    second = save_run(make_outcome(title="second"), store, now=FIXED_TIME)
    assert first.directory != second.directory
    assert second.directory.name.endswith("bbbb1111")
    assert first.record_path.read_bytes() == before
    assert [r.analysis["title"] for r in store.records()[0]] == ["Unit-test run", "second"]


def test_store_gives_up_instead_of_reusing_a_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(records, "_random_token", lambda: "cafe0000")
    store = RunStore(tmp_path)
    store.create_run("bode", now=FIXED_TIME)
    with pytest.raises(RecordError, match="unique run directory"):
        store.create_run("bode", now=FIXED_TIME)


def test_run_directory_files_are_write_once(tmp_path):
    run = RunStore(tmp_path).create_run("step")
    run.write_text("a.csv", "1")
    with pytest.raises(RecordError, match="overwrite"):
        run.write_text("a.csv", "2")
    assert (run.path / "a.csv").read_text() == "1"
    record = RunRecord(run.run_id, run.created.isoformat(), {"type": "step"}, {}, {}, {"overall": "NOT_EVALUATED"}, (), {})
    run.write_record(record)
    with pytest.raises(RecordError, match="overwrite"):
        run.write_record(record)
    for name in ("../escape.txt", "sub/dir.txt", RECORD_FILENAME):
        with pytest.raises(ValueError):
            run.write_text(name, "x")


def test_run_ids_are_sortable_and_file_name_safe(tmp_path):
    run = RunStore(tmp_path).create_run("signal", now=FIXED_TIME)
    assert run.run_id.startswith("20260926T120000Z-signal-")
    assert all(ch.isalnum() or ch in "-_" for ch in run.run_id)
    with pytest.raises(ValueError):
        RunStore(tmp_path).create_run("Bad Name")


def test_load_by_prefix_and_errors(tmp_path):
    store = RunStore(tmp_path)
    a = save_run(make_outcome(), store)
    b = save_run(make_outcome(), store)
    assert store.load(a.record.run_id[:-2]).run_id == a.record.run_id
    assert load_record(b.directory).run_id == b.record.run_id
    with pytest.raises(RecordError, match="ambiguous"):
        store.load(a.record.run_id[:10])
    with pytest.raises(RecordError, match="no run"):
        store.load("does-not-exist")


def test_incomplete_run_directories_are_not_listed(tmp_path):
    store = RunStore(tmp_path)
    store.create_run("bode")  # no record written
    saved = save_run(make_outcome(), store)
    assert store.run_ids() == [saved.record.run_id]


def test_record_round_trip_and_schema_checks(tmp_path):
    saved = save_run(make_outcome(), RunStore(tmp_path))
    data = json.loads(saved.record_path.read_text(encoding="utf-8"))
    again = RunRecord.from_dict(data)
    assert again.to_dict() == data
    with pytest.raises(RecordError, match="not an OpenECE run record"):
        RunRecord.from_dict({**data, "schema": "other"})
    with pytest.raises(RecordError, match="schema_version"):
        RunRecord.from_dict({**data, "schema_version": 99})
    bad = Path(tmp_path / "broken.json")
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(RecordError, match="cannot read"):
        load_record(bad)


def test_software_info_is_a_fresh_copy():
    info = software_info()
    assert info["version"] == openece.__version__
    assert {"numpy", "scipy", "matplotlib", "PyYAML"} <= set(info["dependencies"])
    info["version"] = "mutated"
    assert software_info()["version"] == openece.__version__


def test_to_jsonable_converts_numpy_and_non_finite_values():
    value = {"a": np.float32(1.5), "b": np.int64(3), "c": np.bool_(True), "d": [np.nan, np.inf, 2.0],
             "e": np.array([1, 2]), "f": Path("x"), "g": (1, None)}
    assert to_jsonable(value) == {"a": 1.5, "b": 3, "c": True, "d": [None, None, 2.0], "e": [1, 2], "f": "x", "g": [1, None]}
    with pytest.raises(TypeError):
        to_jsonable({"x": object()})
