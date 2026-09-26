from __future__ import annotations
from abc import ABC, abstractmethod

class SignalGenerator(ABC):
    @abstractmethod
    def set_sine(self, frequency_hz: float, amplitude_vpk: float): ...

class Oscilloscope(ABC):
    @abstractmethod
    def measure_transfer(self, frequency_hz: float) -> complex: ...

class PowerSupply(ABC):
    @abstractmethod
    def set_voltage(self, voltage_v: float): ...
