"""Presentation of results: HTML reports and figures. Never computes analysis results.

``write_html_report`` is the original Gate 0 minimal report and is kept for compatibility;
``render_run_report`` renders the full report of a persisted run record.
"""
from __future__ import annotations
from pathlib import Path
from html import escape

from .html import render_run_report

__all__ = ["render_run_report", "write_html_report"]

def write_html_report(path, title: str, checks, notes: str = ""):
    rows = "".join(
        f"<tr><td>{escape(c['name'])}</td><td>{c['value']:.6g}</td><td>{c['min'] if c['min'] is not None else ''}</td><td>{c['max'] if c['max'] is not None else ''}</td><td>{'PASS' if c['passed'] else 'FAIL'}</td></tr>"
        for c in checks
    )
    overall = all(c["passed"] for c in checks)
    html = f"""<!doctype html><meta charset='utf-8'><title>{escape(title)}</title>
<style>body{{font-family:system-ui;max-width:900px;margin:40px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbb;padding:8px}}.result{{font-size:1.4rem;font-weight:700}}</style>
<h1>{escape(title)}</h1><p class='result'>OVERALL: {'PASS' if overall else 'FAIL'}</p>
<table><tr><th>Check</th><th>Value</th><th>Min</th><th>Max</th><th>Status</th></tr>{rows}</table><p>{escape(notes)}</p>"""
    Path(path).write_text(html, encoding="utf-8")
    return overall
