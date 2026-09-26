from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InstrumentInfo:
    """Provenance of an instrument backend, recorded with every measurement run."""
    role: str
    backend: str
    model: str
    simulated: bool | None  # None: the backend did not declare it

    def to_dict(self):
        return asdict(self)


class Instrument(ABC):
    """Common base. Backends set ``backend`` and ``simulated`` so records never mislabel data."""
    role = "instrument"
    backend = "unknown"
    simulated: bool | None = None

    def describe(self) -> InstrumentInfo:
        return InstrumentInfo(self.role, self.backend, type(self).__name__, self.simulated)

class SignalGenerator(Instrument):
    role = "signal_generator"
    @abstractmethod
    def set_sine(self, frequency_hz: float, amplitude_vpk: float): ...

class Oscilloscope(Instrument):
    role = "oscilloscope"
    @abstractmethod
    def measure_transfer(self, frequency_hz: float) -> complex: ...

class PowerSupply(Instrument):
    role = "power_supply"
    @abstractmethod
    def set_voltage(self, voltage_v: float): ...
