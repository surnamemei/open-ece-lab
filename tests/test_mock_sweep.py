from openece.instruments.mock import MockRCPlant, MockSignalGenerator, MockOscilloscope
from openece.measurements.frequency_sweep import run_frequency_sweep

def test_mock_rc_cutoff():
    plant = MockRCPlant(noise_std=0.0)
    r = run_frequency_sweep(MockSignalGenerator(), MockOscilloscope(plant), 100, 100000, 200)
    assert abs(r["cutoff_hz"] - plant.cutoff_hz) / plant.cutoff_hz < 0.02
