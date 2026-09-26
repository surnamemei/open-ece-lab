import math
from pathlib import Path

import pytest

from openece.errors import ConfigurationError
from openece.recipes.loader import load_recipe, load_spec
from openece.validation import (
    FAIL, NOT_EVALUATED, PASS, Requirement, check_range, evaluate_requirement, overall_status, parse_requirement,
    requirements_from_mapping,
)
from openece.workflows import AnalysisOutcome, Measurement, apply_requirements, spec_parameters

SPECS = Path(__file__).resolve().parent.parent / "examples" / "specs"


def test_check_range_keeps_gate0_semantics():
    assert check_range("x", 5, 1, 10) == {"name": "x", "value": 5.0, "min": 1, "max": 10, "passed": True}
    assert check_range("x", 11, 1, 10)["passed"] is False
    assert check_range("x", float("nan"), 1, 10)["passed"] is False
    assert check_range("x", 3)["passed"] is True


@pytest.mark.parametrize("text, expected", [
    ("cutoff_hz=1450:1750", Requirement("cutoff_hz", 1450, 1750)),
    ("overshoot_percent=:10", Requirement("overshoot_percent", None, 10)),
    ("max_gain_db=-0.5:", Requirement("max_gain_db", -0.5, None)),
    ("steady_state_error_percent=-2:2", Requirement("steady_state_error_percent", -2, 2)),
    ("rise_time_10_90_s=:1e-3", Requirement("rise_time_10_90_s", None, 1e-3)),
])
def test_parse_requirement(text, expected):
    assert parse_requirement(text) == expected


@pytest.mark.parametrize("text", ["cutoff_hz", "cutoff_hz=1450", "cutoff_hz=a:b", "=1:2", "x=:", "x=5:1", "x=nan:1"])
def test_parse_requirement_errors(text):
    with pytest.raises(ConfigurationError):
        parse_requirement(text)


def test_evaluate_requirement_details():
    req = Requirement("cutoff_hz", 1450, 1750)
    assert evaluate_requirement(req, 1500, "Hz").detail == "within limits"
    assert evaluate_requirement(req, 1400).detail == "below minimum 1450"
    high = evaluate_requirement(req, 1800)
    assert not high.passed and high.detail == "above maximum 1750"
    missing = evaluate_requirement(req, math.nan)
    assert not missing.passed and "not determined" in missing.detail
    assert evaluate_requirement(req, 1450).passed and evaluate_requirement(req, 1750).passed  # inclusive


def test_overall_status():
    ok, bad = evaluate_requirement(Requirement("x", 0, 1), 0.5), evaluate_requirement(Requirement("x", 0, 1), 2)
    assert overall_status([]) == NOT_EVALUATED
    assert overall_status([ok]) == PASS
    assert overall_status([ok, bad]) == FAIL


def test_requirements_from_yaml_style_mapping():
    reqs = requirements_from_mapping({"a": {"min": 1, "max": 2}, "b": {"max": "1e-3"}})
    assert reqs == [Requirement("a", 1, 2), Requirement("b", None, 0.001)]
    with pytest.raises(ConfigurationError, match="unknown key"):
        requirements_from_mapping({"a": {"minimum": 1}})
    with pytest.raises(ConfigurationError, match="must be a number"):
        requirements_from_mapping({"a": {"max": True}})
    with pytest.raises(ConfigurationError):
        requirements_from_mapping(["a"])


def test_example_specs_load():
    rc = load_spec(SPECS / "rc_lowpass.yaml")
    assert rc.analysis == "bode" and {r.name for r in rc.requirements} >= {"cutoff_hz"}
    assert len(rc.sha256) == 64
    motor = load_spec(SPECS / "motor_step.yaml")
    assert motor.analysis == "step" and motor.parameters["reference"] == 1000
    assert load_spec(SPECS / "signal_1khz.yaml").analysis == "signal"


@pytest.mark.parametrize("text, message", [
    ("name: x\nrequirements: {a: {min: 1}\n", "invalid YAML at line"),
    ("name: x\nrequirement: {}\n", "unknown key"),
    ("name: x\n", "requirements"),
    ("- a\n- b\n", "mapping"),
    ("requirements: {a: {min: 1}}\nparameters: [1, 2]\n", "parameters"),
])
def test_spec_errors(tmp_path, text, message):
    path = tmp_path / "spec.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_spec(path)


def test_recipe_errors_are_value_errors(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text("name: only a name\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_recipe(path)
    with pytest.raises(ConfigurationError, match="not found"):
        load_recipe(tmp_path / "missing.yaml")


def outcome_with(**results):
    return AnalysisOutcome(
        analysis="bode", title="t", source={"kind": "file"}, parameters={},
        results={k: Measurement(v, "Hz", k) for k, v in results.items()},
    )


def test_apply_requirements():
    outcome = apply_requirements(outcome_with(cutoff_hz=1500.0), [Requirement("cutoff_hz", 1450, 1750)],
                                 spec={"name": "spec"})
    assert outcome.overall == PASS and outcome.spec == {"name": "spec"}
    assert outcome.checks[0].unit == "Hz"
    with pytest.raises(ConfigurationError, match="available results: cutoff_hz"):
        apply_requirements(outcome_with(cutoff_hz=1.0), [Requirement("cutof_hz", 1, 2)])
    with pytest.raises(ConfigurationError, match="more than once"):
        apply_requirements(outcome_with(cutoff_hz=1.0), [Requirement("cutoff_hz", 0, 2), Requirement("cutoff_hz", 0, 3)])


def test_spec_parameters_are_checked():
    motor = load_spec(SPECS / "motor_step.yaml")
    assert spec_parameters(motor, "step") == {"reference": 1000, "settling_band": 0.02}
    with pytest.raises(ConfigurationError, match="is for 'step' analysis"):
        spec_parameters(motor, "bode")
    assert spec_parameters(None, "bode") == {}
