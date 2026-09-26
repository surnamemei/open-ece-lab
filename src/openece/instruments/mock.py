from __future__ import annotations
import math
import numpy as np
from .base import Oscilloscope, SignalGenerator, PowerSupply

class MockRCPlant:
    def __init__(self, r_ohm=10_000.0, c_f=10e-9, noise_std=0.002, seed=5305):
        self.r_ohm = float(r_ohm)
        self.c_f = float(c_f)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.noise_std = float(noise_std)

    @property
    def cutoff_hz(self):
        return 1.0 / (2 * math.pi * self.r_ohm * self.c_f)

    def transfer(self, frequency_hz: float):
        wrc = 2 * math.pi * frequency_hz * self.r_ohm * self.c_f
        ideal = 1.0 / (1.0 + 1j * wrc)
        noise = self.rng.normal(0, self.noise_std) + 1j * self.rng.normal(0, self.noise_std)
        return ideal + noise

class MockSignalGenerator(SignalGenerator):
    backend = "mock"
    simulated = True
    def __init__(self):
        self.frequency_hz = 1000.0
        self.amplitude_vpk = 1.0
    def set_sine(self, frequency_hz: float, amplitude_vpk: float):
        if frequency_hz <= 0 or amplitude_vpk <= 0:
            raise ValueError("frequency and amplitude must be positive")
        self.frequency_hz = float(frequency_hz)
        self.amplitude_vpk = float(amplitude_vpk)

class MockOscilloscope(Oscilloscope):
    backend = "mock"
    simulated = True
    def __init__(self, plant: MockRCPlant): self.plant = plant
    def measure_transfer(self, frequency_hz: float) -> complex: return self.plant.transfer(frequency_hz)

class MockPowerSupply(PowerSupply):
    backend = "mock"
    simulated = True
    def __init__(self): self.voltage_v = 0.0
    def set_voltage(self, voltage_v: float):
        if voltage_v < 0: raise ValueError("mock supply only supports non-negative voltage")
        self.voltage_v = float(voltage_v)
