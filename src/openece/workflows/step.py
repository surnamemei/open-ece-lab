"""Step-response analysis of a time series from a file."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ..analysis.control import step_metrics
from ..errors import AnalysisError
from ..io import ColumnRole, Dataset, resolve_columns
from ..io.columns import TIME_NAMES
from .common import column_values, file_source, mapping_dict, number, time_seconds
from .outcome import AnalysisOutcome, Measurement

ANALYSIS = "step"
PARAMETERS = ("reference", "settling_band", "step_time_s")


def analyze_step(
    dataset: Dataset,
    *,
    columns: Mapping[str, str | None] | None = None,
    time_unit: str | None = None,
    sample_rate_hz: float | None = None,
    reference: float | None = None,
    settling_band: float = 0.02,
    step_time_s: float | None = None,
    title: str | None = None,
) -> AnalysisOutcome:
    """Rise time, overshoot, settling time and steady-state error of a positive-going step.

    Roles: ``time`` (recognised by name; optional when a sample rate is known) and ``response``
    (the only other numeric column, or explicit). Times are reported on the input time axis, or
    relative to ``step_time_s`` when it is given.
    """
    settling_band = number(settling_band, "settling_band", minimum=0.0, maximum=1.0, exclusive=True)
    if reference is not None:
        reference = number(reference, "reference")
    if step_time_s is not None:
        step_time_s = number(step_time_s, "step_time_s")
    if sample_rate_hz is not None:
        sample_rate_hz = number(sample_rate_hz, "sample_rate_hz", positive=True)
    fs_known = sample_rate_hz is not None or dataset.sample_rate_hz is not None
    roles = (
        ColumnRole("time", "time", TIME_NAMES, required=not fs_known, accepts_datetime=True),
        ColumnRole("response", "response", by_elimination=True),
    )
    mapping = resolve_columns(dataset, roles, columns)
    warnings = list(dataset.warnings)
    where = dataset.source.path
    response = column_values(dataset, mapping["response"], "response")

    if sample_rate_hz is not None or "time" not in mapping:
        fs = sample_rate_hz or dataset.sample_rate_hz
        t = np.arange(len(response)) / fs
        time_info = {"source": "explicit sample rate" if sample_rate_hz else "file sample rate", "sample_rate_hz": fs}
        if "time" in mapping:
            warnings.append(f"time column {mapping['time'].column.name!r} was ignored because a sample rate was given")
    else:
        t, time_info = time_seconds(dataset, mapping["time"], time_unit, warnings)
        steps = np.diff(t)
        bad = np.flatnonzero(steps <= 0)
        if bad.size:
            i = int(bad[0]) + 1
            raise AnalysisError(
                f"{where}: time column {mapping['time'].column.name!r} is not strictly increasing at "
                f"{dataset.locate(i)} ({t[i - 1]:.9g} s -> {t[i]:.9g} s)"
            )

    if step_time_s is not None:
        t = t - step_time_s
        origin = f"relative to the step at {step_time_s:g} s"
    else:
        origin = "on the input time axis"
        if t[0] > 0 and t[0] > t[-1] - t[0]:
            warnings.append(
                f"the time axis starts at {t[0]:.6g} s, long after t = 0; peak and settling times are reported on "
                "that axis - set the step time to measure them from the step instant"
            )
    try:
        metrics = step_metrics(t, response, reference=reference, settling_band=settling_band)
    except ValueError as exc:
        raise AnalysisError(f"{where}: step analysis failed: {exc}") from exc

    unit = mapping["response"].column.unit
    needs_reference = None if reference is not None else "not available: no reference value was given"
    results = {
        "initial": Measurement(metrics["initial"], unit, "pre-step baseline (median of the first samples)"),
        "final": Measurement(metrics["final"], unit, "final value (median of the last 10 % of samples)"),
        "rise_time_10_90_s": Measurement(metrics["rise_time_10_90_s"], "s", "10-90 % rise time"),
        "peak": Measurement(metrics["peak"], unit, "maximum response value"),
        "peak_time_s": Measurement(metrics["peak_time_s"], "s", f"time of the maximum, {origin}"),
        "overshoot_percent": Measurement(metrics["overshoot_percent"], "%", "overshoot relative to the step size"),
        "settling_time_s": Measurement(
            metrics["settling_time_s"], "s",
            f"time from which the response stays within +/-{settling_band * 100:g} % of the step size "
            f"around the final value, {origin}",
        ),
        "steady_state_error": Measurement(
            metrics.get("steady_state_error", math.nan), unit, "reference - final value", note=needs_reference),
        "steady_state_error_percent": Measurement(
            metrics.get("steady_state_error_percent", math.nan), "%", "(reference - final) / reference x 100",
            note=needs_reference),
    }
    title = title or f"Step response of {dataset.source.path}"
    parameters = {
        "reference": reference, "settling_band": settling_band, "step_time_s": step_time_s,
        "columns": mapping_dict(mapping), "time": time_info,
    }
    band_abs = settling_band * abs(metrics["final"] - metrics["initial"])
    return AnalysisOutcome(
        analysis=ANALYSIS,
        title=title,
        source=file_source(dataset),
        parameters=parameters,
        results=results,
        warnings=tuple(warnings),
        arrays={"time_s": t, "response": response},
        figures=_figures(t, response, metrics, band_abs, reference, unit, settling_band, title),
    )


def _figures(t, response, metrics, band_abs, reference, unit, settling_band, title):
    def build():
        from ..reporting.plots import step_figure

        figure = step_figure(
            t, response, final=metrics["final"], band_abs=band_abs, peak=metrics["peak"],
            peak_time_s=metrics["peak_time_s"], settling_time_s=metrics["settling_time_s"],
            reference=math.nan if reference is None else reference, response_unit=unit,
            settling_band=settling_band, title=title,
        )
        return {"step_response.png": ("step response with final value, settling band and peak", figure)}

    return build
