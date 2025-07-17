import numpy as np
from datetime import datetime, timedelta

from backend.app.core.coherence_calculator import CoherenceCalculator, CoherenceComponents
from backend.app.core.derivative_calculator import DerivativeCalculator


def build_series(n=10):
    calculator = CoherenceCalculator()
    start = datetime(2024, 1, 1)
    series = []
    for i in range(n):
        components = CoherenceComponents(
            psi=0.1 * i,
            rho=0.2,
            q=0.3,
            f=0.1,
            timestamp=start + timedelta(seconds=i),
            user_id="u1",
        )
        result = calculator.calculate_coherence(components)
        series.append(result)
    return series


def test_derivative_calculator_basic():
    series = build_series(8)
    calc = DerivativeCalculator(window_size=5)
    derivatives = calc.calculate_derivatives(series)
    assert len(derivatives) == len(series) - 1
    # last psi derivative should be close to 0.1 per second
    assert np.isclose(derivatives[-1].dPsi_dt, 0.1, atol=0.05)
