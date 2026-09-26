"""Recipe-driven RC low-pass frequency sweep using the mock instrument backend (SIMULATED data)."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

from ..errors import ConfigurationError
from ..instruments.mock import MockOscilloscope, MockRCPlant, MockSignalGenerator
from ..measurements.frequency_sweep import run_frequency_sweep
from ..recipes.loader import read_recipe
from ..validation import Requirement, requirements_from_mapping
from .bode import ANALYSIS, bode_figures, bode_results
from .common import integer, number
from .outcome import AnalysisOutcome, DataTable

DROP_DB = 3.0


def run_mock_rc_sweep(recipe_path) -> tuple[AnalysisOutcome, list[Requirement]]:
    """Run the recipe's sine sweep on a simulated RC plant and analyse it like any Bode data.

    Returns the outcome and the recipe's requirements (not yet applied).
    """
    recipe, raw = read_recipe(recipe_path)
    where = str(recipe_path)
    plant_cfg = _section(recipe.get("mock_plant") or {}, "mock_plant", where)
    stimulus = _section(recipe["stimulus"], "stimulus", where)
    if stimulus.get("type", "sine_sweep") != "sine_sweep":
        raise ConfigurationError(f"{where}: stimulus type {stimulus.get('type')!r} is not supported (only 'sine_sweep')")

    r_ohm = number(plant_cfg.get("r_ohm", 10_000), "mock_plant.r_ohm", positive=True)
    c_f = number(plant_cfg.get("c_f", 10e-9), "mock_plant.c_f", positive=True)
    noise_std = number(plant_cfg.get("noise_std", 0.001), "mock_plant.noise_std", minimum=0.0)
    seed = integer(plant_cfg.get("seed", 5305), "mock_plant.seed", minimum=0)
    start_hz = number(_required(stimulus, "start_hz", where), "stimulus.start_hz", positive=True)
    stop_hz = number(_required(stimulus, "stop_hz", where), "stimulus.stop_hz", positive=True)
    points = integer(_required(stimulus, "points", where), "stimulus.points", minimum=3)
    amplitude_vpk = number(stimulus.get("amplitude_vpk", 1.0), "stimulus.amplitude_vpk", positive=True)
    requirements = requirements_from_mapping(recipe["requirements"], source=where)

    plant = MockRCPlant(r_ohm=r_ohm, c_f=c_f, noise_std=noise_std, seed=seed)
    generator, scope = MockSignalGenerator(), MockOscilloscope(plant)
    try:
        sweep = run_frequency_sweep(generator, scope, start_hz, stop_hz, points, amplitude_vpk)
    except ValueError as exc:
        raise ConfigurationError(f"{where}: invalid stimulus: {exc}") from exc

    frequency, magnitude_db, phase_deg = sweep["frequency_hz"], sweep["magnitude_db"], sweep["phase_deg"]
    results, warnings = bode_results(frequency, magnitude_db, phase_deg, drop_db=DROP_DB)
    title = str(recipe["name"])
    source = {
        "kind": "instrument",
        "backend": "mock",
        "simulated": True,
        "procedure": "frequency_sweep",
        "instruments": [generator.describe().to_dict(), scope.describe().to_dict()],
        "plant": {
            "model": "MockRCPlant", "r_ohm": r_ohm, "c_f": c_f, "noise_std": noise_std, "seed": seed,
            "nominal_cutoff_hz": plant.cutoff_hz,
        },
        "recipe": {
            "path": where, "name": title, "sha256": hashlib.sha256(raw).hexdigest(),
        },
    }
    parameters = {
        "drop_db": DROP_DB,
        "stimulus": {"type": "sine_sweep", "start_hz": start_hz, "stop_hz": stop_hz, "points": points,
                     "amplitude_vpk": amplitude_vpk},
    }
    transfer = sweep["transfer"]
    raw = DataTable("simulated sweep: complex transfer ratio Vout/Vin per stimulus frequency", {
        "frequency (Hz)": frequency, "transfer real (V/V)": transfer.real, "transfer imag (V/V)": transfer.imag,
        "magnitude (dB)": magnitude_db, "phase (deg)": phase_deg,
    })
    outcome = AnalysisOutcome(
        analysis=ANALYSIS,
        title=title,
        source=source,
        parameters=parameters,
        results=results,
        warnings=tuple(warnings),
        arrays={"frequency_hz": frequency, "magnitude_db": magnitude_db, "phase_deg": phase_deg},
        tables={"sweep_raw.csv": raw},
        figures=bode_figures(frequency, magnitude_db, phase_deg, results, DROP_DB, "dB", f"{title} (SIMULATED)"),
    )
    return outcome, requirements


def _section(value, name: str, where: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{where}: '{name}' must be a mapping")
    return value


def _required(section: Mapping, key: str, where: str):
    if key not in section:
        raise ConfigurationError(f"{where}: stimulus.{key} is required")
    return section[key]
