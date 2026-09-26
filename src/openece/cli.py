from __future__ import annotations
import argparse
from pathlib import Path
from .instruments.mock import MockRCPlant, MockSignalGenerator, MockOscilloscope
from .measurements.frequency_sweep import run_frequency_sweep
from .recipes.loader import load_recipe
from .validation import check_range
from .reporting import write_html_report


def demo_rc(recipe_path: str, output: str):
    recipe = load_recipe(recipe_path)
    plant_cfg = recipe.get("mock_plant", {})
    plant = MockRCPlant(r_ohm=plant_cfg.get("r_ohm", 10_000), c_f=plant_cfg.get("c_f", 10e-9), noise_std=plant_cfg.get("noise_std", 0.001))
    gen, scope = MockSignalGenerator(), MockOscilloscope(plant)
    s = recipe["stimulus"]
    result = run_frequency_sweep(gen, scope, s["start_hz"], s["stop_hz"], s["points"], s.get("amplitude_vpk", 1.0))
    req = recipe["requirements"]["cutoff_hz"]
    checks = [check_range("cutoff_hz", result["cutoff_hz"], req.get("min"), req.get("max"))]
    write_html_report(output, recipe["name"], checks, notes=f"Mock plant nominal fc={plant.cutoff_hz:.2f} Hz")
    print(f"Measured cutoff: {result['cutoff_hz']:.2f} Hz")
    print(f"Overall: {'PASS' if all(c['passed'] for c in checks) else 'FAIL'}")
    print(f"Report: {Path(output).resolve()}")


def main():
    p = argparse.ArgumentParser(prog="openece")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo-rc", help="Run an end-to-end RC low-pass validation using mock instruments")
    d.add_argument("--recipe", default="examples/recipes/rc_lowpass.yaml")
    d.add_argument("--output", default="rc_report.html")
    args = p.parse_args()
    if args.command == "demo-rc": demo_rc(args.recipe, args.output)

if __name__ == "__main__": main()
