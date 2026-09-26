import numpy as np

from openece.reporting import render_run_report, write_html_report
from openece.reporting.plots import bode_figure, figure_to_png, signal_figure, step_figure


def test_legacy_gate0_report_still_works(tmp_path):
    path = tmp_path / "r.html"
    overall = write_html_report(path, "RC", [{"name": "cutoff_hz", "value": 1588.6, "min": 1450, "max": 1750, "passed": True}])
    assert overall is True
    assert "OVERALL: PASS" in path.read_text(encoding="utf-8")


def base_record(**changes):
    record = {
        "run_id": "20260926T120000Z-bode-00000000",
        "created_utc": "2026-09-26T12:00:00+00:00",
        "analysis": {"type": "bode", "title": "<script>alert(1)</script>", "parameters": {"drop_db": 3.0}},
        "source": {"kind": "file", "path": "x.csv", "format": "csv", "size_bytes": 10, "sha256": "ab" * 32,
                   "metadata": {"comments": ["<b>comment</b>"]}},
        "results": {"cutoff_hz": {"value": None, "unit": "Hz", "description": "cutoff", "note": "not found"}},
        "validation": {"overall": "NOT_EVALUATED", "spec": None, "checks": []},
        "artifacts": [{"path": "report.html", "kind": "report", "media_type": "text/html", "description": ""}],
        "software": {"version": "0.1.0"},
        "warnings": ["w <1>"],
    }
    record.update(changes)
    return record


def test_run_report_escapes_file_content_and_shows_status():
    html = render_run_report(base_record())
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<b>comment</b>" not in html and "&lt;b&gt;comment&lt;/b&gt;" in html
    assert "NOT EVALUATED" in html and "n/a" in html and "not found" in html
    assert "cannot verify whether it is a physical measurement" in html


def test_run_report_flags_simulated_data():
    record = base_record(source={"kind": "instrument", "backend": "mock", "simulated": True},
                         validation={"overall": "FAIL", "spec": {"name": "s"}, "checks": [
                             {"name": "cutoff_hz", "value": 1.0, "unit": "Hz", "min": 2.0, "max": None,
                              "status": "FAIL", "detail": "below minimum 2"}]})
    html = render_run_report(record, {"bode.png": b"\x89PNG fake"})
    assert "SIMULATED DATA" in html and "not a physical measurement" in html
    assert 'class="badge fail"' in html and "below minimum 2" in html
    assert "data:image/png;base64," in html


def test_figures_render_headless():
    f = np.geomspace(10, 1e4, 50)
    m = -10 * np.log10(1 + (f / 1000) ** 2)
    t = np.linspace(0, 1, 500)
    y = 1 - np.exp(-t / 0.1)
    for fig in (
        bode_figure(f, m, -np.degrees(np.arctan(f / 1000)), cutoff_hz=1000.0, max_gain_db=0.0),
        bode_figure(f, m),
        step_figure(t, y, final=1.0, band_abs=0.02, peak=1.0, peak_time_s=1.0, settling_time_s=0.4, response_unit=None),
        signal_figure(t, np.sin(2 * np.pi * 5 * t), np.linspace(0, 250, 251), np.ones(251), np.linspace(0, 250, 65),
                      np.ones(65), dominant_hz=5.0, unit="V"),
    ):
        assert figure_to_png(fig).startswith(b"\x89PNG")
