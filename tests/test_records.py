import json
import math
import os
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
    shared = os.path.commonprefix([a.record.run_id, b.record.run_id])  # at least the year
    with pytest.raises(RecordError, match="ambiguous"):
        store.load(shared)
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


def test_default_runs_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENECE_RUNS_DIR", str(tmp_path / "explicit"))
    assert records.default_runs_dir() == tmp_path / "explicit"
    monkeypatch.delenv("OPENECE_RUNS_DIR")
    monkeypatch.setattr(records.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert records.default_runs_dir() == tmp_path / "xdg" / "openece" / "runs"
    monkeypatch.setenv("XDG_DATA_HOME", "relative/path")  # not absolute: ignored per the XDG spec
    assert records.default_runs_dir() == Path.home() / ".local" / "share" / "openece" / "runs"
    monkeypatch.setattr(records.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert records.default_runs_dir() == tmp_path / "local" / "OpenECE" / "runs"
    monkeypatch.setattr(records.sys, "platform", "darwin")
    assert records.default_runs_dir() == Path.home() / "Library" / "Application Support" / "OpenECE" / "runs"


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


def test_malformed_records_are_skipped_not_fatal(tmp_path):
    store = RunStore(tmp_path)
    good, bad = save_run(make_outcome(), store), save_run(make_outcome(), store)
    data = json.loads(bad.record_path.read_text(encoding="utf-8"))
    data["analysis"] = "bode"  # hand-edited into the wrong type
    bad.record_path.write_text(json.dumps(data), encoding="utf-8")
    found, problems = store.records()
    assert [r.run_id for r in found] == [good.record.run_id]
    assert len(problems) == 1 and "'analysis' must be a JSON object" in problems[0]


def test_a_failing_figure_leaves_no_run_directory(tmp_path):
    def broken():
        raise RuntimeError("plotting failed")
    with pytest.raises(RuntimeError):
        save_run(make_outcome(figures=broken), RunStore(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_undecodable_file_name_bytes_are_stored_escaped(tmp_path):
    name = "bad\udcffname.csv"  # how Python represents a non-UTF-8 byte in a POSIX file name
    saved = save_run(make_outcome(source={"kind": "file", "path": name, "sha256": "0" * 64}), RunStore(tmp_path))
    record = json.loads(saved.record_path.read_bytes().decode("utf-8"))
    assert record["source"]["path"] == "bad\\udcffname.csv"
    assert "bad\\udcffname.csv" in saved.report_path.read_text(encoding="utf-8")


def test_record_rename_is_retried_while_windows_holds_the_file(tmp_path, monkeypatch):
    run = RunStore(tmp_path).create_run("bode")
    record = RunRecord(run.run_id, run.created.isoformat(), {"type": "bode"}, {}, {}, {"overall": "NOT_EVALUATED"}, (), {})
    real_replace, calls = records.os.replace, []

    def flaky(src, dst):
        calls.append(src)
        if len(calls) < 3:
            raise PermissionError("file in use")
        real_replace(src, dst)

    monkeypatch.setattr(records.os, "replace", flaky)
    monkeypatch.setattr(records.time, "sleep", lambda s: None)
    assert run.write_record(record).is_file() and len(calls) == 3
