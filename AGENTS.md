# OpenECE Lab agent rules

## Product goal
Build a hardware-agnostic engineering measurement and validation tool. The same measurement engine must work with mock instruments and future real instruments.

## Architecture rules
- Keep analysis algorithms independent of UI and instrument drivers.
- Keep instrument-specific code behind interfaces in `src/openece/instruments/`.
- Measurement recipes must be data/config driven where practical.
- A real-instrument backend must not change analysis APIs.
- Every numerical feature requires a deterministic test with synthetic data.

### Layers and allowed imports
`tests/test_architecture.py` enforces these rules. Update it only if you are deliberately changing the architecture, and document the change here.

| Package | Responsibility | Must not import |
| --- | --- | --- |
| `openece.analysis` | pure numerical functions: arrays in, numbers/arrays out | any other `openece` layer, matplotlib |
| `openece.io` | read files into `Dataset`, resolve column roles | analysis, instruments, workflows, reporting, records, cli |
| `openece.units` | parse/convert units (pure) | anything in `openece` except errors |
| `openece.instruments` | instrument interfaces (`base.py`) and backends (`mock.py`, future real ones) | io, analysis, workflows, reporting, records, cli |
| `openece.measurements` | procedures that drive instruments through the interfaces | io, workflows, reporting, records, cli |
| `openece.validation`, `openece.recipes` | requirements, specs, recipes | io, analysis, instruments, workflows, reporting, records, cli |
| `openece.records` | run records and the append-only run store | io, analysis, instruments, workflows, reporting, cli |
| `openece.reporting` | presentation only (HTML, figures); never computes results | io, analysis, instruments, workflows, records, cli |
| `openece.workflows` | orchestration: import -> analysis -> validation -> record/report | cli |
| `openece.cli` | argument parsing and printing only | analysis, instruments, measurements, reporting, matplotlib, scipy |

- New analyses: put the numerics in `analysis/` (with synthetic-data tests), the orchestration in `workflows/`, and presentation in `reporting/`. Then add the CLI command and document it in the README.
- A future UI must call `openece.workflows`, just like the CLI. Never put analysis code in a front-end.
- Real instruments get a new module in `instruments/` that implements the existing interfaces. Declare `backend` and `simulated` on every backend class (`simulated = False` only for real hardware).

## Data import rules
- Never silently guess. Ambiguous or missing columns raise `AmbiguousColumnError` or `MissingColumnError` listing the candidates. Explicit mappings always win.
- Never silently coerce or drop data. Non-numeric cells, blanks, NaN, non-monotonic or non-uniform time must produce an error that names the file line or sample. Anything the importer decides (delimiter, units row, unit assumptions) goes into `metadata` or `warnings`.
- Units: magnitude (dB or linear) and phase (deg or rad) are never assumed. Time and frequency may default to s and Hz only with a visible warning.
- User-facing failures raise `openece.errors.OpenECEError` subclasses with complete messages. Set `hint` to the parameter or column role that fixes the problem; the CLI maps it to a flag. Never let a raw `KeyError`/`IndexError` reach the user.

## Records and provenance rules
- Every analysis or measurement run saved through `workflows.save_run` produces `record.json` (schema `openece.run-record`). Bump `SCHEMA_VERSION` in `records.py` for any incompatible change and keep `RunRecord.from_dict` able to read older versions.
- Run folders and their files are created exclusively and never modified afterwards. Never add code that overwrites or deletes a run.
- Every result carries an explicit unit (`"1"` = dimensionless, `None` = unknown). Undeterminable values are NaN in memory and `null` with a note in JSON.
- Record the source file's path and SHA-256, every effective parameter (including how columns were chosen), warnings, and software versions.

## Safety and integrity
- Never claim a mock/simulated result is a physical measurement.
- Default real power outputs to OFF on connection and on exceptions.
- Never silently overwrite raw measurement data.
- Report units explicitly.
- Example datasets in `examples/` are synthetic. Generate them only with `scripts/generate_examples.py`, never edit them by hand, and keep the "SYNTHETIC DATA" label in each file. `tests/test_examples.py` checks that the committed files match the generator.
- Do not change numerical results by hand or loosen tests to make them pass. If a result changes, explain why. `tests/test_end_to_end.py` pins the Gate 0 mock cutoff (1588.62 Hz).
- Do not add vendor SDKs or hardware-specific assumptions until the hardware gate is opened.

## Compatibility
- Python >= 3.11 on Windows and Linux (CI runs both). Use `pathlib`, pass explicit `encoding="utf-8"` for text files, avoid `:` and other reserved characters in generated file names, and keep matplotlib on the object-oriented `Figure` API (no `pyplot`, no GUI backend).
- Avoid new dependencies. The core runtime is numpy, scipy, matplotlib and PyYAML.

## Definition of done
A feature is done when it has: implementation, tests, a documented example, input validation, and a failure mode that gives a useful error.

Before finishing, run `pip install -e ".[dev]"`, then `pytest -q`, then the README quick-start command. Check the generated `record.json` and `report.html`.
