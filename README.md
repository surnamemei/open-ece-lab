# OpenECE Lab

A hardware-agnostic measurement, analysis and validation toolkit for electronics, control and signal-processing work.

## Gate 0 status

The first MVP is deliberately hardware-free. It already contains:

- signal-analysis primitives (FFT / dominant frequency / Welch PSD);
- control step-response metrics (10–90% rise time, overshoot, settling time, steady-state error);
- Bode conversion and -3 dB cutoff estimation;
- explicit instrument interfaces;
- mock signal-generator / oscilloscope / power-supply backends;
- an end-to-end RC low-pass frequency-sweep measurement;
- YAML measurement recipes;
- PASS/FAIL validation and an HTML report;
- deterministic tests.

The next hardware gate is **not** unlocked until this mock workflow remains useful and stable.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
openece demo-rc --recipe examples/recipes/rc_lowpass.yaml --output rc_report.html
```

Expected mock plant: 10 kOhm + 10 nF, nominal cutoff about 1591.55 Hz. The generated report should PASS the 1450–1750 Hz requirement.

## Planned gates

1. **Gate 0 (current):** mock instruments + file analysis + recipe/report engine.
2. **Gate 1:** one real programmable instrument backend (candidate: Analog Discovery 3) only after the software workflow proves useful.
3. **Gate 2:** persistent raw-measurement records, richer validation specs and a desktop/web UI.
4. **Gate 3:** WSPR/filter/FPGA DUT case studies and regression comparison across hardware revisions.

## Product principle

OpenECE Lab should make this workflow reproducible:

`stimulus -> acquisition -> analysis -> specification check -> report`

No physical measurement is implied by the mock demo.
