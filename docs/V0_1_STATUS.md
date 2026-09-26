# OpenECE Lab v0.1 status: RELEASE CANDIDATE

| | |
| --- | --- |
| Status | **v0.1 RELEASE CANDIDATE.** Not a final public release (see "Public-release readiness"). |
| Date | 26 September 2026 |
| Verified commit | `fe3caa7` on branch `release/v0.1-rc` (package version `0.1.0`) |
| CI | GitHub Actions [run 36230630303](https://github.com/surnamemei/open-ece-lab/actions/runs/36230630303) on `fe3caa7`: 4/4 jobs passed (Ubuntu and Windows, Python 3.11 and 3.13) |
| Hardware | none required: all verification used files and mock instruments |

## Verification result

The final verification ran on a fresh clone of `fe3caa7`, installed only with the README
instructions. The exact commands are listed at the end of this document.

| # | Check | Result |
| --- | --- | --- |
| 1 | Fresh install with the documented instructions (`python3 -m venv`, `pip install -e ".[dev]"`) | PASS |
| 2 | Full test suite on the supported Python versions | PASS: 277/277 on 3.11.16, 3.12.3 and 3.13.15 |
| 3 | README quick start, run exactly as documented | PASS (output identical to the README) |
| 4 | Complete RC example end to end (file -> import -> Bode -> spec -> record + report) | PASS: cutoff 1594.68 Hz, PASS |
| 5 | Run records are created outside the repository by default | PASS: nothing new in the checkout |
| 6 | Repeated runs do not overwrite earlier records | PASS: first run byte-identical after 7 runs (2 in the same second) |
| 7 | JSON and HTML artifacts checked for inconsistencies (automated cross-check plus manual reading) | PASS: 7/7 run folders consistent |
| 8 | Malformed, empty and ambiguous CSV inputs give useful errors | PASS: 15/15 give exit 2, a specific message and a hint; no tracebacks |
| 9 | WAV mono/stereo behaviour | PASS: mono automatic; stereo requires `--channel`; unknown channel rejected |
| 10 | CLI `--help` and invalid commands | PASS: help exits 0; invalid usage exits 2 with usage text |

The Gate 0 regression value is unchanged: the mock RC demo still measures **1588.62 Hz** and is
labelled SIMULATED in its output, record and report.

### Independent code review

Before the verification, an independent review of the v0.1 changes reported 24 findings. 22
were reproduced and 2 were plausible without a reproduction (Windows file-locking on rename;
hashing and parsing from separate reads). All 24 were fixed, each with a regression test (39 new
tests). The most important were:

- **Wrong PASS results:** a step response that never settled reported the first time sample as its
  settling time. A `T (degC)` column could be taken as time, after which the wrong column was
  analysed. Channel labels such as `Voltage (CH1)` were reported as units.
- **Robustness:** crashes (exit 3) on row-vector CSVs, truncated UTF-16 files, huge NPY headers,
  `$` in titles and non-UTF-8 file names.
- **Data safety:** `demo-rc --output` could overwrite any file, including files inside an earlier
  run.

The full list is in [CHANGELOG.md](../CHANGELOG.md). The final verification also exposed an
intermittent test failure. It was a test bug: a test read "the first entry" of a directory that
also held its input file. It was fixed in `fe3caa7`, and the tests were repeated 15+ times
without failure.

## Implemented functionality

- **File import** (`openece.io`) into a typed `Dataset` with provenance (path, size, SHA-256
  of the exact bytes parsed) and import metadata:
  - CSV/TSV/TXT: delimiter detection (comma, semicolon, tab, whitespace), header and
    PicoScope-style units-row detection, `#` comments kept as provenance, decimal comma
    (`--decimal ,`), ISO 8601 timestamps, UTF-8/UTF-16/UTF-32/Latin-1, and errors that point at
    file lines;
  - WAV: PCM 8/16/24/32-bit and float, any channel count, RIFF INFO comments, and a warning for
    truncated recordings. Amplitudes are in FS (digital full scale);
  - NPY: 1-D, 2-D and structured arrays, `timedelta64`/`datetime64` converted to seconds, header
    size validated, and pickled data refused.
- **Column mapping without guessing:** explicit mapping, then recognised names, then the single
  remaining column. Anything ambiguous or missing is an error naming the candidates and the flag
  to use.
- **Units:** parsed from headers (`name (unit)`, `name [unit]`, `name_unit`) and converted
  explicitly. Every result carries a unit. Magnitude (dB or linear) and phase (deg or rad) are
  never assumed.
- **Analyses:**
  - Bode: -3 dB cutoff, maximum gain, phase at cutoff;
  - step response: rise time, overshoot, settling time, steady-state error;
  - signal: level statistics, dominant frequency, FFT and Welch PSD.
- **Validation:** inclusive limits from YAML specs or `--require NAME=MIN:MAX`, giving PASS, FAIL
  or NOT_EVALUATED. A typo in a requirement name is an error.
- **Run records:** one folder per run with `record.json` (schema `openece.run-record` v1),
  `report.html`, figures and data tables. Records are:
  - append-only: exclusive creation, and no file is ever replaced;
  - strict JSON;
  - stored in the per-user data directory by default (override with `OPENECE_RUNS_DIR` or
    `--runs-dir`).
- **Reports:** self-contained HTML with embedded figures, provenance, a SIMULATED banner for mock
  data, and a "cannot verify physical origin" note for file data.
- **CLI** `open-ece` (alias `openece`, also `python -m openece`):
  - commands `analyze-bode`, `analyze-step`, `analyze-signal`, `inspect`, `runs list`,
    `runs show` and `demo-rc`;
  - exit codes 0 (PASS or not evaluated), 1 (FAIL), 2 (input error) and 3 (internal error).
- **Mock instruments** (signal generator, oscilloscope, power supply) that declare themselves
  simulated, plus the recipe-driven RC sweep.
- **Examples:** synthetic, reproducible datasets (`rc_sweep.csv`, `motor_step.csv`, `signal.wav`)
  with specs. The tests check that the committed files match the generator.

## Architecture summary

```text
cli ─▶ workflows ─▶ io, units            (file import, column roles, units)
                 ├▶ analysis              (pure numerics: numpy/scipy only)
                 ├▶ measurements ─▶ instruments (interfaces + mock backends)
                 ├▶ validation, recipes   (requirements, specs, recipes)
                 ├▶ records               (append-only JSON run store)
                 └▶ reporting             (HTML + matplotlib figures, presentation only)
```

Dependencies point one way. `tests/test_architecture.py` statically enforces the import rules
described in [AGENTS.md](../AGENTS.md); for example, analysis never imports IO, instruments,
reporting or the CLI. A future UI would call `openece.workflows`, exactly as the CLI does.

## Verified platforms and Python versions

| Platform | Python | numpy / scipy / matplotlib | Result |
| --- | --- | --- | --- |
| Linux: Ubuntu 24.04.4 LTS on WSL2 (kernel 6.6.87.2), x86_64 | 3.12.3 (system) | 2.5.3 / 1.18.1 / 3.11.2 | 277 passed; all CLI checks |
| same machine | 3.11.16 (standalone build) | 2.4.6 / 1.17.1 / 3.11.2 | 277 passed |
| same machine | 3.13.15 (standalone build) | 2.5.3 / 1.18.1 / 3.11.2 | 277 passed |
| Windows (`windows-latest`, GitHub Actions) | 3.11, 3.13 | resolved by pip in CI | install, full test suite and README quick start passed (run 36230630303) |
| Ubuntu (`ubuntu-latest`, GitHub Actions) | 3.11, 3.13 | resolved by pip in CI | install, full test suite and README quick start passed (run 36230630303) |
| macOS | none | none | not tested |

The Windows CI jobs confirm the test suite and the quick start. These Windows-specific edge cases
were exercised only by simulation on Linux:

- console output that can't be encoded is escaped rather than crashing;
- `runs show` prints ASCII-only JSON;
- file names are safe on Windows;
- record renames are retried while antivirus or the indexer holds the file;
- `%LOCALAPPDATA%` is the default runs location.

## Known limitations

- **No physical instruments.** Only mock backends exist. File data is recorded with its checksum
  and comments, but its physical origin can't be verified.
- **Bode:**
  - only the upper (low-pass) cutoff is estimated, at exactly max − 3.000 dB (0.24 % below
    1/(2πRC) for a first-order RC; `--drop-db 3.0103` gives the half-power point);
  - the maximum sample is the reference level, so passband noise biases the cutoff slightly;
  - magnitude/phase input only: no complex data and no Vin/Vout amplitude pairs.
- **Step:** positive-going steps only. Times are reported on the file's time axis unless
  `--step-time` is given.
- **Signal:** the dominant frequency is quantised to the FFT bin spacing. No THD/SNR. Input must
  be uniformly sampled; there is no resampling.
- **Import:**
  - no Excel, HDF5, NPZ or vendor binary formats; quoted CSV fields can't span lines; time-of-day
    timestamps are not parsed;
  - units come from a built-in list, and unknown ones are shown as "not specified";
  - headerless integer pairs such as `2,3090` are ambiguous with decimal-comma data (imported as
    two columns, with a warning);
  - real instrument exports (PicoScope, Rigol, WaveForms, ...) have only been emulated, not tested.
- **Runs:** records store the input path and SHA-256 but don't copy the input file. There is no
  run comparison or regression view yet.
- **Packaging:**
  - the wheel builds and contains all packages and both console scripts, but `examples/` is only
    in the repository (the quick start needs a checkout);
  - `pyproject.toml` has no `license` field;
  - the package has not been published to PyPI.

## Deferred features (explicitly out of scope for v0.1)

- **Analog Discovery 3 integration**, and any other real instrument backend.
- **Vendor SDK dependencies** (for example WaveForms/`dwf`, VISA): none are added.
- **Full GUI** (desktop or web). v0.1 has the CLI and HTML reports only.
- **FPGA integration** and FPGA DUT case studies.
- Also deferred:
  - THD/SNR/SINAD;
  - lower (high-pass) and band-pass edge estimation;
  - Vin/Vout amplitude-pair and complex Bode input;
  - resampling of non-uniform data;
  - run-to-run regression comparison;
  - copying inputs into run folders;
  - Excel/HDF5/NPZ import;
  - volt calibration for WAV captures.

## Public-release readiness

**Not ready for a final public release. Ready as a release candidate for internal and friendly-user
testing.** CI is green on Windows and Ubuntu. Remaining before a public release:

1. Validation with real instrument export files (at least one oscilloscope CSV, one network
   analyser export, one sound-card WAV) and with a few external users.
2. Packaging: add `license = "MIT"` to `pyproject.toml`, and decide how examples ship (package
   data or a download). Then test installation from a built wheel outside the repository and
   decide on PyPI publication.
3. Replace the README's `<your repository URL>` placeholder once the repository is public.
4. Keep the record schema (`openece.run-record` v1) stable, or bump it deliberately, before
   external users rely on it.

## Hardware-integration readiness

**The architecture is ready for a first backend, but the instrument interfaces need extending
first. Hardware integration is deferred.**

Already in place:

- analysis never touches instruments;
- measurement procedures (`run_frequency_sweep`) use only the abstract interfaces;
- every backend declares `backend` and `simulated`, and records store them;
- a real backend can reuse the Bode analysis, validation, records and reports unchanged.

Needed before an Analog Discovery 3 (or any real) backend:

1. **Connection lifecycle:** open/close, context manager, device identification (model, serial,
   firmware) in `InstrumentInfo`.
2. **Safe output state:** an explicit output-enable API for signal generators and power supplies,
   OFF on connect and on any exception (an AGENTS.md safety rule that the current `PowerSupply`
   interface can't yet enforce).
3. **Raw acquisition:** a waveform-capture interface (the mock `measure_transfer` hides the
   stimulus/response capture and the amplitude/phase extraction a real scope needs), with the raw
   waveforms saved as run artifacts.
4. **Error and timeout model** for instrument I/O, and recording of instrument settings (ranges,
   sample rates) in run records.
5. **Hardware-in-the-loop tests,** kept separate from the default test suite.

## Exact commands used for final verification

Linux/bash. Verification-only environment: `XDG_DATA_HOME` pointed at a scratch folder, so the
**default** runs location resolved to `<work>/xdg-data/openece/runs` instead of the real home
directory. `OPENECE_RUNS_DIR` was unset. Python 3.11 and 3.13 were standalone interpreters (from
`uv python install`) in a scratch folder.

```bash
export XDG_DATA_HOME="<work>/xdg-data"; unset OPENECE_RUNS_DIR

# 1. Fresh installation (README > Installation)
git clone --branch release/v0.1-rc <repository> open-ece-lab
cd open-ece-lab
git log --oneline -1                      # fe3caa7
python3 -m venv .venv
source .venv/bin/activate
pip install -q -e ".[dev]"
open-ece --version                        # open-ece 0.1.0

# 2. Test suite on each supported Python version
python -m pytest -q -p no:cacheprovider   # 3.12.3: 277 passed
python3.11 -m venv <work>/venv3.11 && <work>/venv3.11/bin/pip install -q -e ".[dev]"
<work>/venv3.11/bin/python -m pytest -q -p no:cacheprovider   # 277 passed
python3.13 -m venv <work>/venv3.13 && <work>/venv3.13/bin/pip install -q -e ".[dev]"
<work>/venv3.13/bin/python -m pytest -q -p no:cacheprovider   # 277 passed

# 3 + 4. README quick start / complete RC example
open-ece analyze-bode examples/rc_sweep.csv --spec examples/specs/rc_lowpass.yaml   # PASS, exit 0

# 5. Records outside the repository
test -e runs || echo "no runs/ in the checkout"
git status --porcelain --ignored | grep -v -E '\.venv/|egg-info|__pycache__|\.pytest_cache'   # (nothing)
ls "$XDG_DATA_HOME/openece/runs"

# README example commands
open-ece analyze-step examples/motor_step.csv --spec examples/specs/motor_step.yaml        # PASS, exit 0
open-ece analyze-signal examples/signal.wav --spec examples/specs/signal_1khz.yaml         # PASS, exit 0
open-ece analyze-step examples/motor_step.csv --reference 1000 --require overshoot_percent=:5   # FAIL, exit 1
open-ece inspect examples/rc_sweep.csv
open-ece demo-rc --recipe examples/recipes/rc_lowpass.yaml                                 # 1588.62 Hz, SIMULATED, PASS
open-ece runs list
open-ece runs show <first 16 characters of the first run ID>

# 6. Repeated runs (sha256 of the first run's record.json + report.html taken before and after)
open-ece analyze-bode examples/rc_sweep.csv --spec examples/specs/rc_lowpass.yaml
open-ece analyze-bode examples/rc_sweep.csv --spec examples/specs/rc_lowpass.yaml
sha256sum "<runs>/<first run>/record.json" "<runs>/<first run>/report.html"   # unchanged

# 7. Artifacts: a cross-check script (session-local, not in the repository) verified for every run
#    folder that record.json is strict JSON with schema v1, that the run ID matches the folder,
#    that the report shows the run ID, title, time, version, every result (value + unit) and every
#    check, that check statuses agree with their limits and the overall status, that the artifact
#    list equals the folder contents with valid PNGs, that the SIMULATED banner matches
#    source.simulated, and that the source SHA-256 matches the file. Then manual reading of
#    record.json and the rendered report text for the RC file run and the mock demo run, plus
#    visual inspection of the figures.

# 8. Malformed / empty / ambiguous CSV inputs (each: exit 2, message, hint, no traceback)
open-ece analyze-step <input>.csv --no-record
#   empty.csv                ''                                          -> no data found
#   comments_only.csv        '# only comments\n# nothing else\n\n'       -> no data found
#   header_only.csv          'time,speed\n'                              -> header but no data rows
#   ragged.csv               'time,speed\n0,0\n0.1,5,7\n0.2,9\n'         -> line 3 has 3 fields but line 1 has 2
#   ambiguous_response.csv   'time,command,speed\n...'                   -> 2 candidate columns ('command', 'speed')
#   ambiguous_time.csv       't,time,speed\n...'                         -> 2 recognised time names ('t', 'time')
#   no_time_column.csv       'x,y\n...'                                  -> no time column found
#   non_numeric.csv          'time,speed\n0,0\n0.1,abc\n...'             -> 'speed' not usable (line 3: 'abc')
#   blank_cell.csv           'time,speed\n0,0\n0.1,0.5\n0.2,\n...'       -> blank value at line 4
#   repeated_time.csv        'time,speed\n0,0\n0.1,0.5\n0.1,0.8\n...'    -> not strictly increasing at line 4
#   decimal_comma.csv        'zeit;wert\n0,0;0,0\n...'                   -> decimal comma; hint --decimal ,
#   preamble.csv             'Model,XYZ-100\nSerial,12345\ntime,speed\n...' -> hint --skip-rows N
#   binary_as.csv            (first 64 bytes of a WAV file)              -> not a text file
#   unbalanced_quote.csv     'a,b\n"1,2\n3,4\n'                          -> line 2: unbalanced quote
open-ece analyze-bode bode_no_units.csv --no-record   # 'freq,gain\n...' -> dB or linear? hint --magnitude-unit

# 9. WAV mono and stereo (stereo.wav: 48 kHz int16, ch1 = 440 Hz at 0.5 FS, ch2 = 1 kHz at 0.25 FS)
open-ece analyze-signal examples/signal.wav --no-record            # ch1 automatic: 1000 Hz
open-ece analyze-signal stereo.wav --no-record                     # exit 2: 2 candidates, hint --channel
open-ece analyze-signal stereo.wav --channel ch1 --no-record       # 440 Hz, ac_rms 0.3535 FS
open-ece analyze-signal stereo.wav --channel ch2 --no-record       # 1000 Hz, ac_rms 0.1768 FS
open-ece analyze-signal stereo.wav --channel ch3 --no-record       # exit 2: not found, lists ch1, ch2

# 10. CLI help and invalid commands
open-ece --help                                          # exit 0
open-ece <command> --help                                # exit 0 for all six commands
open-ece                                                 # exit 2: COMMAND required
open-ece frobnicate                                      # exit 2: invalid choice
open-ece analyze-bode                                    # exit 2: file required
open-ece analyze-bode examples/rc_sweep.csv --no-such-option   # exit 2: unrecognized arguments
python -m openece --version                              # open-ece 0.1.0

# CI: GitHub Actions run 36230630303 on fe3caa7, one job per (ubuntu-latest, windows-latest) x (3.11, 3.13)
pip install -e ".[dev]"
pytest -q
open-ece analyze-bode examples/rc_sweep.csv --spec examples/specs/rc_lowpass.yaml --runs-dir ci-runs
# all 4 jobs: every step succeeded (read from the public GitHub Actions API)
```
