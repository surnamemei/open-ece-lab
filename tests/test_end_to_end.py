"""End to end: example file -> import -> analysis -> PASS/FAIL -> record + report, through the CLI."""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import openece
from openece.analysis.control import step_metrics
from openece.cli import demo_rc, main
from openece.io import load_csv
from openece.workflows import analyze_bode

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
SPECS = EXAMPLES / "specs"


def run_cli(capsys, *args):
    code = main([str(a) for a in args])
    out, err = capsys.readouterr()
    return code, out, err


def only_run(runs_dir):
    runs = [p for p in Path(runs_dir).iterdir() if p.is_dir()]
    assert len(runs) == 1, runs
    record = json.loads((runs[0] / "record.json").read_text(encoding="utf-8"))
    return runs[0], record


def test_bode_example_full_workflow(tmp_path, capsys, generator):
    source = EXAMPLES / "rc_sweep.csv"
    code, out, err = run_cli(capsys, "analyze-bode", source, "--spec", SPECS / "rc_lowpass.yaml", "--runs-dir", tmp_path)
    assert code == 0, err
    assert "Overall: PASS" in out and "cutoff_hz" in out
    run_dir, record = only_run(tmp_path)

    assert record["source"]["kind"] == "file"
    assert record["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert any("SYNTHETIC DATA" in c for c in record["source"]["metadata"]["comments"])
    cutoff = record["results"]["cutoff_hz"]
    assert cutoff["unit"] == "Hz"
    assert cutoff["value"] == pytest.approx(generator.rc_cutoff_hz() * math.sqrt(10**0.3 - 1), rel=0.01)
    assert record["validation"]["overall"] == "PASS"
    assert record["validation"]["spec"]["name"] == "RC low-pass acceptance"
    assert record["analysis"]["parameters"]["columns"]["magnitude"]["column"] == "Magnitude (dB)"
    assert record["software"]["version"] == openece.__version__
    assert record["invocation"]["argv"][0] == "analyze-bode"

    paths = {a["kind"]: run_dir / a["path"] for a in record["artifacts"]}
    assert paths["figure"].read_bytes().startswith(b"\x89PNG")
    report = paths["report"].read_text(encoding="utf-8")
    assert "PASS" in report and "RC low-pass acceptance" in report and "data:image/png;base64," in report
    assert "SYNTHETIC DATA" in report

    # The data artifact is itself importable and reproduces the result.
    again = analyze_bode(load_csv(paths["data"]))
    assert again.results["cutoff_hz"].value == pytest.approx(cutoff["value"], rel=1e-12)


def test_step_example_matches_the_generating_model(tmp_path, capsys, generator):
    code, out, err = run_cli(capsys, "analyze-step", EXAMPLES / "motor_step.csv", "--spec", SPECS / "motor_step.yaml",
                             "--runs-dir", tmp_path)
    assert code == 0, err
    _, record = only_run(tmp_path)
    t = generator.motor_time_s()
    truth = step_metrics(t, generator.motor_model_rpm(t), reference=generator.MOTOR_SETPOINT_RPM)
    results = {k: v["value"] for k, v in record["results"].items()}
    assert results["final"] == pytest.approx(truth["final"], abs=1.0)
    assert results["rise_time_10_90_s"] == pytest.approx(truth["rise_time_10_90_s"], abs=0.004)
    assert results["overshoot_percent"] == pytest.approx(truth["overshoot_percent"], abs=1.0)
    assert results["settling_time_s"] == pytest.approx(truth["settling_time_s"], abs=0.02)
    assert results["steady_state_error_percent"] == pytest.approx(1.5, abs=0.2)
    assert record["results"]["final"]["unit"] == "rpm"
    assert record["analysis"]["parameters"]["reference"] == 1000
    assert record["validation"]["overall"] == "PASS"


def test_signal_example_matches_the_generating_model(tmp_path, capsys, generator):
    code, out, err = run_cli(capsys, "analyze-signal", EXAMPLES / "signal.wav", "--spec", SPECS / "signal_1khz.yaml",
                             "--runs-dir", tmp_path)
    assert code == 0, err
    run_dir, record = only_run(tmp_path)
    x = generator.signal_model_fs(generator.signal_time_s())
    results = {k: v["value"] for k, v in record["results"].items()}
    assert results["dominant_frequency_hz"] == 1000.0
    assert results["mean"] == pytest.approx(np.mean(x), abs=1e-4)
    assert results["ac_rms"] == pytest.approx(np.std(x), rel=1e-3)
    assert record["results"]["ac_rms"]["unit"] == "FS"
    assert record["analysis"]["parameters"]["sample_rate_hz"] == 48000
    assert record["source"]["metadata"]["encoding"] == "PCM 16-bit"
    assert (run_dir / "psd.csv").is_file() and (run_dir / "signal.png").is_file()
    assert record["validation"]["overall"] == "PASS"


def test_failing_requirement_exits_with_1(tmp_path, capsys):
    code, out, _ = run_cli(capsys, "analyze-step", EXAMPLES / "motor_step.csv", "--spec", SPECS / "motor_step.yaml",
                           "--require", "overshoot_percent=:5", "--runs-dir", tmp_path)
    assert code == 1
    assert "FAIL  overshoot_percent" in out and "Overall: FAIL" in out
    _, record = only_run(tmp_path)
    assert record["validation"]["overall"] == "FAIL"


def test_no_record_writes_nothing(tmp_path, capsys):
    code, out, _ = run_cli(capsys, "analyze-signal", EXAMPLES / "signal.wav", "--no-record", "--runs-dir", tmp_path / "r")
    assert code == 0 and "NOT EVALUATED" in out
    assert not (tmp_path / "r").exists()


def test_user_errors_exit_with_2_and_a_hint(tmp_path, capsys, write_text):
    path = write_text("three.csv", "time,command,speed\n0,0,0\n1,1,0.5\n2,1,0.8\n3,1,0.9\n4,1,1\n")
    code, out, err = run_cli(capsys, "analyze-step", path, "--runs-dir", tmp_path / "r")
    assert code == 2
    assert "cannot choose the response column" in err and "--response-column" in err and "inspect" in err
    assert "Traceback" not in err
    code, out, _ = run_cli(capsys, "analyze-step", path, "--response-column", "speed", "--no-record")
    assert code == 0 and "'speed' (explicit)" in out
    code, _, err = run_cli(capsys, "analyze-bode", tmp_path / "missing.csv")
    assert code == 2 and "not found" in err


def test_inspect_shows_columns_units_and_roles(capsys):
    code, out, _ = run_cli(capsys, "inspect", EXAMPLES / "rc_sweep.csv")
    assert code == 0
    assert "Frequency (Hz)" in out and "Hz" in out and "SYNTHETIC DATA" in out
    assert "frequency  'Frequency (Hz)'" in out and "time       none" in out


def test_runs_list_and_show(tmp_path, capsys):
    run_cli(capsys, "analyze-bode", EXAMPLES / "rc_sweep.csv", "--runs-dir", tmp_path)
    code, out, _ = run_cli(capsys, "runs", "list", "--runs-dir", tmp_path)
    assert code == 0 and "bode" in out and "NOT_EVALUATED" in out
    run_id = next(p.name for p in tmp_path.iterdir())
    code, out, _ = run_cli(capsys, "runs", "show", run_id[:22], "--runs-dir", tmp_path)
    assert code == 0 and json.loads(out)["run_id"] == run_id
    code, _, err = run_cli(capsys, "runs", "show", "nope", "--runs-dir", tmp_path)
    assert code == 2 and "no run" in err


def test_mock_demo_keeps_gate0_result_and_labels_it_simulated(tmp_path, capsys):
    copy = tmp_path / "rc_report.html"
    code, out, err = run_cli(capsys, "demo-rc", "--recipe", EXAMPLES / "recipes" / "rc_lowpass.yaml",
                             "--runs-dir", tmp_path / "runs", "--output", copy)
    assert code == 0, err
    assert "SIMULATED DATA" in out and "1588.62" in out and "Overall: PASS" in out
    run_dir, record = only_run(tmp_path / "runs")
    assert record["results"]["cutoff_hz"]["value"] == pytest.approx(1588.62, abs=0.005)  # Gate 0 value
    source = record["source"]
    assert source["simulated"] is True and source["backend"] == "mock"
    assert all(i["simulated"] is True for i in source["instruments"])
    assert source["plant"]["nominal_cutoff_hz"] == pytest.approx(1591.55, abs=0.01)
    assert (run_dir / "sweep_raw.csv").read_text().splitlines()[1].startswith("# SIMULATED DATA")
    assert "SIMULATED DATA" in copy.read_text(encoding="utf-8")


def test_legacy_demo_rc_function_still_works(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert demo_rc(str(EXAMPLES / "recipes" / "rc_lowpass.yaml"), "rc_report.html") == 0
    assert (tmp_path / "rc_report.html").is_file() and (tmp_path / "runs").is_dir()


def test_python_dash_m_entry_point():
    result = subprocess.run([sys.executable, "-m", "openece", "--version"], capture_output=True, text=True, check=False)
    assert result.returncode == 0 and openece.__version__ in result.stdout


def test_installed_console_script(tmp_path):
    bin_dir = Path(sys.executable).parent
    script = shutil.which("open-ece", path=os.pathsep.join([str(bin_dir), os.environ.get("PATH", "")]))
    if script is None:
        pytest.skip("open-ece console script is not installed in this environment")
    result = subprocess.run([script, "analyze-bode", str(EXAMPLES / "rc_sweep.csv"), "--spec",
                             str(SPECS / "rc_lowpass.yaml"), "--runs-dir", str(tmp_path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "Overall: PASS" in result.stdout
