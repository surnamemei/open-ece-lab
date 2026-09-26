import numpy as np
from openece.analysis.control import step_metrics

def test_first_order_step_metrics():
    t = np.linspace(0, 2, 4001)
    tau = 0.2
    y = 1 - np.exp(-t/tau)
    m = step_metrics(t, y, reference=1.0)
    assert abs(m["rise_time_10_90_s"] - 2.1972*tau) < 0.01
    assert m["overshoot_percent"] < 0.01
    assert 0.75 < m["settling_time_s"] < 0.82
