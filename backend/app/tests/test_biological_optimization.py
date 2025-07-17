from datetime import datetime

import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from app.core.biological_optimization import BiologicalOptimizer
from app.core.models import BiologicalOptimizationParams


@pytest.fixture
def optimizer():
    return BiologicalOptimizer()


def test_q_optimal_basic(optimizer):
    params = BiologicalOptimizationParams()
    expected_0_2 = (params.q_max * 0.2) / (params.K_m + 0.2 + (0.2 ** 2) / params.K_i)
    expected_0_6 = (params.q_max * 0.6) / (params.K_m + 0.6 + (0.6 ** 2) / params.K_i)
    expected_1_0 = (params.q_max * 1.0) / (params.K_m + 1.0 + (1.0 ** 2) / params.K_i)
    assert abs(float(optimizer.calculate_q_optimal(0.2, params)) - expected_0_2) < 1e-6
    assert abs(float(optimizer.calculate_q_optimal(0.6, params)) - expected_0_6) < 1e-6
    assert abs(float(optimizer.calculate_q_optimal(1.0, params)) - expected_1_0) < 1e-6


def test_validate_parameters(optimizer):
    params = BiologicalOptimizationParams(K_m=0, K_i=0.8)
    assert optimizer.validate_parameters(params) is False
    params = BiologicalOptimizationParams()
    assert optimizer.validate_parameters(params) is True
