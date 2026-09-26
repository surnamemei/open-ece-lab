"""Command-line front-end (``open-ece``; ``openece`` is kept as an alias).

This module only translates arguments into workflow calls and prints outcomes; it holds no
analysis or file-parsing logic.

Exit codes: 0 = completed (PASS, or no requirements given), 1 = completed with FAIL,
2 = input/usage error, 3 = unexpected internal error.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .errors import ColumnMappingError, ConfigurationError, OpenECEError
from .io import FORMATS, load_file
from .io.columns import FREQUENCY_NAMES, MAGNITUDE_NAMES, PHASE_NAMES, TIME_NAMES, ColumnRole, match_by_name
from .recipes.loader import load_spec
from .records import RunStore, load_record
from .validation import FAIL, NOT_EVALUATED, parse_requirement
from .workflows import (
    analyze_bode, analyze_signal, analyze_step, apply_requirements, run_mock_rc_sweep, save_run, spec_parameters,
)

EXIT_OK, EXIT_FAIL, EXIT_ERROR, EXIT_INTERNAL = 0, 1, 2, 3
PROG = "open-ece"

# Maps the front-end-neutral ``hint`` of an OpenECEError to the flag that fixes the problem.
_HINTS = {
    "time": "--time-column NAME", "response": "--response-column NAME", "signal": "--channel NAME",
    "frequency": "--frequency-column NAME", "magnitude": "--magnitude-column NAME", "phase": "--phase-column NAME",
    "time_unit": "--time-unit UNIT (s, ms, us, ns, min, h)",
    "frequency_unit": "--frequency-unit UNIT (Hz, kHz, MHz, GHz, mHz, rad/s)",
    "magnitude_unit": "--magnitude-unit {db,linear}", "phase_unit": "--phase-unit {deg,rad}",
    "sample_rate": "--sample-rate HZ", "sample_rate_hz": "--sample-rate HZ", "decimal": "--decimal ,",
    "delimiter": "--delimiter {comma,semicolon,tab,whitespace}", "skip_rows": "--skip-rows N",
    "encoding": "--encoding NAME (for example utf-8 or cp1252)", "format": "--format {csv,wav,npy}",
    "require": "--require NAME=MIN:MAX (for example cutoff_hz=1450:1750)", "nperseg": "--nperseg N",
    "drop_db": "--drop-db DB", "reference": "--reference VALUE", "settling_band": "--settling-band FRACTION",
    "step_time_s": "--step-time SECONDS",
}
_ROLE_NAMES = {"time": TIME_NAMES, "frequency": FREQUENCY_NAMES, "magnitude": MAGNITUDE_NAMES, "phase": PHASE_NAMES}


def main(argv: list[str] | None = None) -> int:
    _configure_output()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    args.argv = argv
    try:
        return args.handler(args)
    except OpenECEError as exc:
        _print_error(exc, getattr(args, "file", None))
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.debug:
            raise
        print(f"internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("re-run with --debug for the full traceback, and please report it", file=sys.stderr)
        return EXIT_INTERNAL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="OpenECE Lab: hardware-agnostic measurement analysis and validation. "
                    "Every analysis run is stored as a folder with record.json and report.html.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--debug", action="store_true", help="show tracebacks for unexpected errors")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    source = argparse.ArgumentParser(add_help=False)
    source.add_argument("file", type=Path, help="input file (.csv, .tsv, .txt, .dat, .wav, .npy)")
    imp = source.add_argument_group("import options")
    imp.add_argument("--format", choices=FORMATS, help="file format (default: from the extension)")
    imp.add_argument("--delimiter", help="CSV delimiter: comma, semicolon, tab, whitespace or a character (default: detect)")
    imp.add_argument("--decimal", choices=[".", ","], help="CSV decimal separator (default: '.')")
    imp.add_argument("--skip-rows", type=int, metavar="N", help="skip N lines at the top of a CSV (metadata preamble)")
    imp.add_argument("--encoding", help="CSV text encoding (default: UTF-8, falling back to Latin-1 with a warning)")

    run = argparse.ArgumentParser(add_help=False)
    val = run.add_argument_group("validation and records")
    val.add_argument("--spec", type=Path, help="YAML spec with requirements (and optional analysis parameters)")
    val.add_argument("--require", action="append", default=[], metavar="NAME=MIN:MAX",
                     help="requirement on a result, e.g. cutoff_hz=1450:1750 or overshoot_percent=:10 (repeatable; "
                          "overrides the spec for the same result)")
    val.add_argument("--runs-dir", type=Path, default=Path("runs"), help="where run folders are created (default: ./runs)")
    val.add_argument("--no-record", action="store_true", help="print results only; do not create a run folder")
    val.add_argument("--title", help="title for the report")

    bode = sub.add_parser("analyze-bode", parents=[source, run], help="frequency response: cutoff, gain, phase",
                          description="Analyse a frequency sweep (frequency, magnitude and optional phase columns).")
    cols = bode.add_argument_group("columns and units")
    cols.add_argument("--frequency-column", metavar="NAME")
    cols.add_argument("--magnitude-column", metavar="NAME")
    cols.add_argument("--phase-column", metavar="NAME")
    cols.add_argument("--frequency-unit", metavar="UNIT", help="override the header unit (Hz, kHz, MHz, rad/s, ...)")
    cols.add_argument("--magnitude-unit", type=str.lower, choices=["db", "linear"],
                      help="required when the magnitude header has no unit")
    cols.add_argument("--phase-unit", metavar="UNIT", help="deg or rad; required when the phase header has no unit")
    bode.add_argument("--drop-db", type=float, help="cutoff level below the gain maximum in dB (default: 3)")
    bode.set_defaults(handler=_cmd_bode)

    step = sub.add_parser("analyze-step", parents=[source, run], help="step response: rise, overshoot, settling",
                          description="Analyse a positive-going step response (time and response columns).")
    cols = step.add_argument_group("columns and units")
    cols.add_argument("--time-column", metavar="NAME")
    cols.add_argument("--response-column", metavar="NAME")
    cols.add_argument("--time-unit", metavar="UNIT", help="override the header unit (s, ms, us, ns, min, h)")
    cols.add_argument("--sample-rate", type=float, metavar="HZ", help="use a fixed sample rate instead of a time column")
    step.add_argument("--reference", type=float, help="commanded final value, enables steady-state error")
    step.add_argument("--settling-band", type=float, metavar="FRACTION", help="settling band as a fraction of the step (default: 0.02)")
    step.add_argument("--step-time", type=float, metavar="SECONDS", help="time of the step; times are then reported relative to it")
    step.set_defaults(handler=_cmd_step)

    sig = sub.add_parser("analyze-signal", parents=[source, run], help="waveform: levels, dominant frequency, spectrum",
                         description="Analyse one uniformly sampled channel (WAV, or CSV/NPY with a time column or --sample-rate).")
    cols = sig.add_argument_group("columns and units")
    cols.add_argument("--channel", metavar="NAME", help="signal column or WAV channel (ch1, ch2, ...) when there are several")
    cols.add_argument("--time-column", metavar="NAME")
    cols.add_argument("--time-unit", metavar="UNIT", help="override the header unit (s, ms, us, ns, min, h)")
    cols.add_argument("--sample-rate", type=float, metavar="HZ", help="sample rate; overrides the file or time column")
    sig.add_argument("--nperseg", type=int, metavar="N", help="Welch PSD segment length (default: min(1024, samples))")
    sig.set_defaults(handler=_cmd_signal)

    inspect = sub.add_parser("inspect", parents=[source], help="show how a file is imported (columns, units, roles)",
                             description="Show how a file would be imported, without analysing it.")
    inspect.set_defaults(handler=_cmd_inspect)

    demo = sub.add_parser("demo-rc", help="RC low-pass sweep on SIMULATED mock instruments (no hardware)",
                          description="Run a recipe-driven RC low-pass sweep using the mock instrument backend.")
    demo.add_argument("--recipe", type=Path, default=Path("examples/recipes/rc_lowpass.yaml"))
    demo.add_argument("--output", type=Path, help="also write a copy of the HTML report to this path")
    demo.add_argument("--require", action="append", default=[], metavar="NAME=MIN:MAX",
                      help="extra/overriding requirement (repeatable)")
    demo.add_argument("--runs-dir", type=Path, default=Path("runs"), help="where run folders are created (default: ./runs)")
    demo.add_argument("--no-record", action="store_true", help="print results only; do not create a run folder")
    demo.set_defaults(handler=_cmd_demo_rc)

    runs = sub.add_parser("runs", help="list or show stored runs", description="List or show stored run records.")
    runs_sub = runs.add_subparsers(dest="runs_command", required=True, metavar="ACTION")
    runs_list = runs_sub.add_parser("list", help="list runs, oldest first")
    runs_list.add_argument("--runs-dir", type=Path, default=Path("runs"))
    runs_list.set_defaults(handler=_cmd_runs_list)
    runs_show = runs_sub.add_parser("show", help="print a run record as JSON")
    runs_show.add_argument("run", help="run ID (or a unique prefix), run folder, or record.json path")
    runs_show.add_argument("--runs-dir", type=Path, default=Path("runs"))
    runs_show.set_defaults(handler=_cmd_runs_show)
    return parser


def _cmd_bode(args) -> int:
    spec, params, requirements = _spec_and_requirements(args, "bode")
    dataset = _load(args)
    if args.drop_db is not None:
        params["drop_db"] = args.drop_db
    outcome = analyze_bode(
        dataset,
        columns={"frequency": args.frequency_column, "magnitude": args.magnitude_column, "phase": args.phase_column},
        frequency_unit=args.frequency_unit, magnitude_unit=args.magnitude_unit, phase_unit=args.phase_unit,
        title=args.title, **params,
    )
    return _finish(args, outcome, requirements, spec)


def _cmd_step(args) -> int:
    spec, params, requirements = _spec_and_requirements(args, "step")
    dataset = _load(args)
    for key, value in (("reference", args.reference), ("settling_band", args.settling_band),
                       ("step_time_s", args.step_time)):
        if value is not None:
            params[key] = value
    outcome = analyze_step(
        dataset, columns={"time": args.time_column, "response": args.response_column},
        time_unit=args.time_unit, sample_rate_hz=args.sample_rate, title=args.title, **params,
    )
    return _finish(args, outcome, requirements, spec)


def _cmd_signal(args) -> int:
    spec, params, requirements = _spec_and_requirements(args, "signal")
    dataset = _load(args)
    if args.nperseg is not None:
        params["nperseg"] = args.nperseg
    outcome = analyze_signal(
        dataset, columns={"time": args.time_column, "signal": args.channel},
        time_unit=args.time_unit, sample_rate_hz=args.sample_rate, title=args.title, **params,
    )
    return _finish(args, outcome, requirements, spec)


def _cmd_demo_rc(args) -> int:
    if args.output is not None and args.no_record:
        raise ConfigurationError("--output needs a run record; drop --no-record")
    outcome, recipe_requirements = run_mock_rc_sweep(args.recipe)
    merged = {r.name: r for r in recipe_requirements}
    merged.update({r.name: r for r in map(parse_requirement, args.require)})
    spec = {"name": f"requirements of recipe {outcome.title!r}", "path": str(args.recipe)}
    code = _finish(args, outcome, list(merged.values()), spec)
    if args.output is not None:
        report = args.runs_dir / args.saved_run_id / "report.html"
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(report, args.output)
        except OSError as exc:
            raise ConfigurationError(f"cannot write report copy to {args.output}: {exc.strerror or exc}") from exc
        print(f"Report copy : {args.output.resolve()}")
    return code


def _cmd_inspect(args) -> int:
    dataset = _load(args)
    meta = dataset.metadata
    print(f"File     : {dataset.source.path}")
    print(f"Format   : {_format_details(meta)}")
    print(f"SHA-256  : {dataset.source.sha256}")
    for comment in meta.get("comments", []):
        print(f"Comment  : {comment}")
    print()
    rows = [("#", "column", "unit", "kind", "min", "max", "blank")]
    for i, col in enumerate(dataset.columns, start=1):
        finite = col.values[np.isfinite(col.values)]
        low, high = (f"{finite.min():.6g}", f"{finite.max():.6g}") if finite.size else ("n/a", "n/a")
        rows.append((str(i), col.name, col.unit or "-", col.kind, low, high, str(len(col.blank_rows))))
    print(f"Numeric columns ({dataset.n_rows} rows):")
    _print_table(rows)
    if dataset.skipped:
        print("\nColumns that cannot be used:")
        for skipped in dataset.skipped:
            print(f"  {skipped.name!r}: {skipped.reason}")
    print("\nColumns recognised by name (explicit --*-column options always win):")
    for role, names in _ROLE_NAMES.items():
        usable, unusable = match_by_name(dataset, ColumnRole(role, role, names))
        found = [repr(c.name) for c in usable] + [f"{s.name!r} (unusable)" for s in unusable]
        status = "none" if not found else found[0] if len(found) == 1 else "AMBIGUOUS: " + ", ".join(found)
        print(f"  {role:<10} {status}")
    _print_warnings(dataset.warnings)
    return EXIT_OK


def _cmd_runs_list(args) -> int:
    records, problems = RunStore(args.runs_dir).records()
    if not records:
        print(f"no runs in {args.runs_dir}")
    else:
        rows = [("run id", "created (UTC)", "analysis", "source", "result")]
        for record in records:
            rows.append((record.run_id, record.created_utc[:19].replace("T", " "), str(record.analysis.get("type")),
                         _source_label(record.source), record.overall))
        _print_table(rows)
    for problem in problems:
        print(f"warning: {problem}", file=sys.stderr)
    return EXIT_OK


def _cmd_runs_show(args) -> int:
    path = Path(args.run)
    record = load_record(path) if path.exists() else RunStore(args.runs_dir).load(args.run)
    print(record.to_json(), end="")
    return EXIT_OK


def _spec_and_requirements(args, analysis: str):
    spec = load_spec(args.spec) if args.spec else None
    params = spec_parameters(spec, analysis)
    merged = {r.name: r for r in (spec.requirements if spec else ())}
    merged.update({r.name: r for r in map(parse_requirement, args.require)})
    return spec, params, list(merged.values())


def _load(args):
    options = {"delimiter": args.delimiter, "decimal": args.decimal, "skip_rows": args.skip_rows,
               "encoding": args.encoding}
    return load_file(args.file, format=args.format, csv_options=options)


def _finish(args, outcome, requirements, spec) -> int:
    outcome = apply_requirements(outcome, requirements, spec=spec)
    saved = None
    if not args.no_record:
        invocation = {"program": PROG, "argv": args.argv, "cwd": os.getcwd()}
        saved = save_run(outcome, RunStore(args.runs_dir), invocation=invocation)
        args.saved_run_id = saved.record.run_id
    _print_outcome(outcome, saved)
    return EXIT_FAIL if outcome.overall == FAIL else EXIT_OK


def _print_outcome(outcome, saved) -> None:
    print(f"OpenECE Lab {__version__} - {outcome.analysis} analysis - {outcome.title}")
    if outcome.simulated:
        print("SIMULATED DATA: mock instrument backend, not a physical measurement")
    print(f"Source : {_source_label(outcome.source, detailed=True)}")
    for role, info in (outcome.parameters.get("columns") or {}).items():
        print(f"Column : {role:<9} <- {info['column']!r} ({info['selected_by']})")
    print("\nResults")
    rows = []
    for name, m in outcome.results.items():
        note = f"  [{m.note}]" if m.note else ""
        rows.append((name, _fmt(m.value), _unit(m.unit) + note))
    _print_table(rows, indent="  ")
    if outcome.checks:
        spec_name = (outcome.spec or {}).get("name")
        print(f"\nValidation{f' ({spec_name})' if spec_name else ''}")
        rows = [("PASS" if c.passed else "FAIL", c.name, f"{_fmt(c.value)} {_unit(c.unit)}",
                 _limits(c.min, c.max), c.detail) for c in outcome.checks]
        _print_table(rows, indent="  ")
    overall = outcome.overall
    if overall == NOT_EVALUATED:
        print("\nOverall: NOT EVALUATED (no requirements; use --spec or --require)")
    else:
        print(f"\nOverall: {overall}")
    _print_warnings(outcome.warnings)
    if saved is not None:
        print(f"\nRun    : {saved.record.run_id}")
        print(f"Record : {saved.record_path}")
        print(f"Report : {saved.report_path}")


def _print_error(exc: OpenECEError, file) -> None:
    print(f"error: {exc}", file=sys.stderr)
    if exc.hint in _HINTS:
        print(f"hint: use {_HINTS[exc.hint]}", file=sys.stderr)
    if isinstance(exc, ColumnMappingError) and file is not None:
        print(f"hint: run '{PROG} inspect {file}' to see the columns and units", file=sys.stderr)


def _print_warnings(warnings) -> None:
    if warnings:
        print("\nWarnings")
        for warning in warnings:
            print(f"  - {warning}")


def _print_table(rows, indent: str = "  ") -> None:
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print(indent + "  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)).rstrip())


def _source_label(source, detailed: bool = False) -> str:
    if source.get("kind") == "file":
        label = str(source.get("path"))
        if detailed:
            label += f" ({source.get('format')}, sha256 {str(source.get('sha256'))[:12]}...)"
        return label
    label = f"{source.get('backend', 'unknown')} instrument backend"
    if source.get("simulated") is True:
        label += " (SIMULATED)"
    return label


def _format_details(meta) -> str:
    fmt = meta.get("format")
    if fmt == "csv":
        return (f"csv, delimiter {meta.get('delimiter')}, decimal '{meta.get('decimal')}', encoding {meta.get('encoding')}, "
                f"header line {meta.get('header_line') or 'none'}, units line {meta.get('units_line') or 'none'}, "
                f"{meta.get('rows')} data rows")
    if fmt == "wav":
        return (f"wav, {meta.get('encoding')}, {meta.get('sample_rate_hz'):g} Hz, {meta.get('channel_layout')}, "
                f"{meta.get('frames')} frames ({meta.get('duration_s'):.6g} s), amplitude in FS (digital full scale)")
    if fmt == "npy":
        return f"npy, shape {tuple(meta.get('shape', ()))}, dtype {meta.get('dtype')}"
    return str(fmt)


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.6g}"


def _unit(unit) -> str:
    return "(unit unspecified)" if unit is None else unit


def _limits(minimum, maximum) -> str:
    if minimum is not None and maximum is not None:
        return f"[{minimum:g}, {maximum:g}]"
    return f">= {minimum:g}" if minimum is not None else f"<= {maximum:g}"


def _configure_output() -> None:
    """Avoid UnicodeEncodeError on legacy Windows consoles when names contain e.g. 'µ' or '°'."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None and encoding != "utf8":
            try:
                reconfigure(errors="backslashreplace")
            except (ValueError, OSError):
                pass


def demo_rc(recipe_path: str, output: str):
    """Gate 0 entry point, kept for compatibility: run the mock RC demo and copy the report to ``output``."""
    return main(["demo-rc", "--recipe", str(recipe_path), "--output", str(output)])


if __name__ == "__main__":
    sys.exit(main())
