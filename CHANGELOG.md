# Changelog

## 0.1.0 (unreleased): first usable version, no hardware required

### Added
- File import (`openece.io`): CSV/TSV/TXT with delimiter, header, units-row, comment,
  decimal-comma and ISO 8601 timestamp handling; WAV (PCM 8/16/24/32-bit, float, any channel
  count, RIFF INFO comments); NPY (1-D, 2-D, structured). All importers return a typed `Dataset`
  with provenance (path, SHA-256) and import metadata.
- Column-role resolution with explicit mapping and ambiguity errors instead of guesses.
- Unit parsing from headers and explicit conversions (`openece.units`).
- Workflows (`openece.workflows`) for Bode, step-response and signal analysis of files, plus the
  mock RC sweep. Results carry explicit units.
- Requirements from YAML specs or `--require NAME=MIN:MAX`, with PASS/FAIL detail.
- Persistent run records (`openece.records`): one folder per run with `record.json`
  (schema `openece.run-record` v1), `report.html`, figures and data tables. Runs are never
  overwritten.
- `open-ece` CLI: `analyze-bode`, `analyze-step`, `analyze-signal`, `inspect`, `runs list/show`,
  `demo-rc`; documented exit codes; `python -m openece`.
- Synthetic example datasets (`examples/rc_sweep.csv`, `motor_step.csv`, `signal.wav`) with specs,
  generated reproducibly by `scripts/generate_examples.py`.
- Tests for import, column mapping, units, WAV/NPY, records, validation, workflows, reporting,
  end-to-end CLI runs, example reproducibility and architecture (import boundaries).
- CI on Ubuntu and Windows with Python 3.11 and 3.13, including the quick-start example.

### Changed
- `openece.reporting` is now a package; `write_html_report` keeps its Gate 0 behaviour.
- `demo-rc` now creates a run record and a report labelled SIMULATED. `--output` still writes a
  copy of the report. The measured mock cutoff is unchanged (1588.62 Hz).
- Instrument base classes gained `describe()` plus `backend`/`simulated` attributes; the mock
  backends declare `simulated = True`. `MockRCPlant` keeps its seed for provenance.
- `load_recipe` raises `ConfigurationError` (a `ValueError`) with clear messages.
- The package version is read from `openece.__version__`.
