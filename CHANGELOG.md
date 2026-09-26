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
  overwritten. They are stored outside the repository by default, in the per-user data directory
  (`OPENECE_RUNS_DIR` or `--runs-dir` override it).
- `open-ece` CLI: `analyze-bode`, `analyze-step`, `analyze-signal`, `inspect`, `runs list/show`,
  `demo-rc`; documented exit codes; `python -m openece`.
- Synthetic example datasets (`examples/rc_sweep.csv`, `motor_step.csv`, `signal.wav`) with specs,
  generated reproducibly by `scripts/generate_examples.py`.
- Tests for import, column mapping, units, WAV/NPY, records, validation, workflows, reporting,
  end-to-end CLI runs, example reproducibility and architecture (import boundaries).
- CI on Ubuntu and Windows with Python 3.11 and 3.13, including the quick-start example.

### Fixed (independent code review before the release candidate)
- The step response settling time was reported as the first time sample (which could PASS a
  limit) when the response never settled within the record; it is now `null` with a note.
- A single-letter column such as `T (degC)` was taken as the time column (with `--sample-rate` the
  wrong column was then analysed); single letters now match case-sensitively.
- Labels such as `Voltage (CH1)` were reported as units; only recognised units are units now, and
  time/frequency are not assumed to be s/Hz when the header carries an unrecognised annotation.
- `freq_MHz` headers lost their case and were rejected as ambiguous.
- Phase data in the 0..360 degree convention gave `phase_at_cutoff_deg` = 315 instead of -45.
- CSV: crashes on very long lines (row-vector files), truncated UTF-16 and quote delimiters; a wrong
  line number for oversized fields; UTF-32 read as UTF-16; a BOM kept with `--encoding utf-8`;
  headerless decimal-comma files mis-split or losing their first row.
- A constant signal reported a dominant frequency of fs/N; it is now `null` with a note.
- `$` in titles or file names crashed plotting (mathtext) and left a partial run folder; figures
  are now rendered before the run folder is created.
- `demo-rc --output` could overwrite any existing file; it now refuses existing files and paths
  inside the runs folder.
- An unusable phase column blocked Bode analysis; it is now reported as a warning.
- A wrong line was reported for a sweep whose last frequency wraps around.
- One malformed record made `runs list` fail; it is now skipped with a warning.
- NPY headers declaring huge shapes caused a MemoryError; `timedelta64` data lost its unit.
- A WAV `data` chunk longer than the file was loaded without a warning.
- Non-UTF-8 file names crashed record writing and console output.
- `runs show` printed non-UTF-8 JSON on legacy Windows consoles; it now prints ASCII-only JSON.
- WAV/NPY/recipe checksums are computed from the same bytes that are parsed; record renames are
  retried briefly if Windows holds the file.

### Changed
- `openece.reporting` is now a package; `write_html_report` keeps its Gate 0 behaviour.
- `demo-rc` now creates a run record and a report labelled SIMULATED. `--output` still writes a
  copy of the report but no longer overwrites an existing file (Gate 0 overwrote `rc_report.html`).
  The measured mock cutoff is unchanged (1588.62 Hz).
- Instrument base classes gained `describe()` plus `backend`/`simulated` attributes; the mock
  backends declare `simulated = True`. `MockRCPlant` keeps its seed for provenance.
- `load_recipe` raises `ConfigurationError` (a `ValueError`) with clear messages.
- The package version is read from `openece.__version__`.
