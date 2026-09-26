# OpenECE Lab agent rules

## Product goal
Build a hardware-agnostic engineering measurement and validation tool. The same measurement engine must work with mock instruments and future real instruments.

## Architecture rules
- Keep analysis algorithms independent of UI and instrument drivers.
- Keep instrument-specific code behind interfaces in `src/openece/instruments/`.
- Measurement recipes must be data/config driven where practical.
- A real-instrument backend must not change analysis APIs.
- Every numerical feature requires a deterministic test with synthetic data.

## Safety and integrity
- Never claim a mock/simulated result is a physical measurement.
- Default real power outputs to OFF on connection and on exceptions.
- Never silently overwrite raw measurement data.
- Report units explicitly.

## Definition of done
A feature is done when it has: implementation, tests, a documented example, input validation, and a failure mode that gives a useful error.
