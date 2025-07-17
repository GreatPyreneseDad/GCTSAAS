from datetime import datetime, timedelta

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from app.core.coherence_calculator import CoherenceCalculator
from app.core.models import CoherenceComponents, BiologicalOptimizationParams


def test_basic_coherence_calculation():
    calc = CoherenceCalculator()
    components = CoherenceComponents(
        psi=0.7,
        rho=0.8,
        q=0.6,
        f=0.9,
        timestamp=datetime.utcnow(),
        user_id="user1",
    )
    result = calc.calculate_coherence(components)
    expected_q_opt = calc.biological_optimizer.calculate_q_optimal(0.6, calc.default_params)
    expected_score = 0.7 + (0.8 * 0.7) + float(expected_q_opt) + (0.9 * 0.7)
    assert abs(float(result.coherence_score) - expected_score) < 1e-6
    assert abs(float(result.q_optimal) - float(expected_q_opt)) < 1e-6


def test_edge_cases():
    calc = CoherenceCalculator()
    components = CoherenceComponents(
        psi=0.0,
        rho=0.0,
        q=0.0,
        f=0.0,
        timestamp=datetime.utcnow(),
        user_id="user1",
    )
    result = calc.calculate_coherence(components)
    assert result.coherence_score == 0



def test_parameter_personalization():
    calc = CoherenceCalculator()
    components = CoherenceComponents(
        psi=0.5,
        rho=0.5,
        q=0.5,
        f=0.5,
        timestamp=datetime.utcnow(),
        user_id="u",
    )
    params = BiologicalOptimizationParams(q_max=0.8, K_m=0.1, K_i=0.5)
    result = calc.calculate_coherence(components, params)
    assert result.optimization_params.q_max == 0.8


def test_derivative_calculations():
    calc = CoherenceCalculator()
    from app.core.derivative_calculator import DerivativeCalculator
    dcalc = DerivativeCalculator()
    now = datetime.utcnow()
    meas = [
        CoherenceComponents(0.5, 0.5, 0.5, 0.5, now, "u"),
        CoherenceComponents(0.6, 0.55, 0.55, 0.6, now + timedelta(seconds=1), "u"),
    ]
    derivative = dcalc.calculate_coherence_derivative(meas)
    assert derivative is not None
    assert derivative.user_id == "u"

