"""Application workflows: import -> analysis -> validation -> record/report.

Workflows orchestrate the core layers (io, analysis, instruments, validation, records,
reporting) and are what front-ends call. They never parse command lines or print; the CLI (and
any future UI) only translates user input into these calls and presents the outcome.
"""
from __future__ import annotations

from ..errors import ConfigurationError
from . import bode, signal, step
from .bode import analyze_bode, bode_results
from .mock_rc import run_mock_rc_sweep
from .outcome import AnalysisOutcome, DataTable, Measurement, apply_requirements
from .persist import SavedRun, save_run
from .signal import analyze_signal
from .step import analyze_step

ANALYSIS_PARAMETERS = {bode.ANALYSIS: bode.PARAMETERS, step.ANALYSIS: step.PARAMETERS, signal.ANALYSIS: signal.PARAMETERS}

__all__ = [
    "ANALYSIS_PARAMETERS", "AnalysisOutcome", "DataTable", "Measurement", "SavedRun",
    "analyze_bode", "analyze_signal", "analyze_step", "apply_requirements", "bode_results",
    "run_mock_rc_sweep", "save_run", "spec_parameters",
]


def spec_parameters(spec, analysis: str) -> dict:
    """Analysis parameters from a spec, checked against what the analysis accepts."""
    if spec is None:
        return {}
    if spec.analysis and spec.analysis != analysis:
        raise ConfigurationError(f"spec {spec.name!r} is for {spec.analysis!r} analysis, not {analysis!r}", hint="spec")
    allowed = ANALYSIS_PARAMETERS[analysis]
    unknown = sorted(set(spec.parameters) - set(allowed))
    if unknown:
        raise ConfigurationError(
            f"{spec.path}: unknown parameter(s) {unknown} for {analysis} analysis (allowed: {', '.join(allowed)})",
            hint="spec",
        )
    return dict(spec.parameters)
