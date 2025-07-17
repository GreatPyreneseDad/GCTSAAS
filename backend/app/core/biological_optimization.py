"""Biological optimization module implementing GCT formulas."""

from decimal import Decimal, getcontext
from typing import Dict, List, Optional

from .models import BiologicalOptimizationParams

getcontext().prec = 10


class BiologicalOptimizer:
    """Handles biological optimization of moral activation energy"""

    def calculate_q_optimal(self, q: float, params: BiologicalOptimizationParams) -> Decimal:
        """Calculate q^optimal using Liebig's Law of the Minimum with inhibition.

        Formula:
            q^optimal = (q_max × q) / (K_m + q + q²/K_i)

        Args:
            q: Raw moral activation energy (0.0-1.0)
            params: Individual optimization parameters

        Returns:
            Optimized moral activation (0.0-1.0) as Decimal
        """
        if not self.validate_parameters(params):
            params = BiologicalOptimizationParams()
        q_d = Decimal(str(q))
        q_max = Decimal(str(params.q_max))
        K_m = Decimal(str(params.K_m))
        K_i = Decimal(str(params.K_i))

        denominator = K_m + q_d + (q_d ** 2) / K_i
        if denominator == 0:
            return Decimal("0")
        return (q_max * q_d) / denominator

    def calculate_q_optimal_derivative(
        self,
        q: float,
        dq_dt: float,
        params: BiologicalOptimizationParams,
    ) -> Decimal:
        """Calculate derivative of q^optimal with respect to time.

        Using quotient rule:
            dq*/dt = [q_max * (K_m + K_i) * dq/dt] / (K_m + q + q²/K_i)²
        """
        if not self.validate_parameters(params):
            params = BiologicalOptimizationParams()
        q_d = Decimal(str(q))
        dq_dt_d = Decimal(str(dq_dt))
        q_max = Decimal(str(params.q_max))
        K_m = Decimal(str(params.K_m))
        K_i = Decimal(str(params.K_i))

        denominator = K_m + q_d + (q_d ** 2) / K_i
        if denominator == 0:
            return Decimal("0")
        numerator = q_max * (K_m + K_i) * dq_dt_d
        return numerator / (denominator ** 2)

    def get_optimization_curve_data(
        self, params: BiologicalOptimizationParams, points: int = 100
    ) -> Dict[str, List[float]]:
        """Generate q and q_optimal values for visualization."""
        if points <= 1:
            points = 2
        q_values = [i / (points - 1) for i in range(points)]
        q_optimal_values = [
            float(self.calculate_q_optimal(q, params)) for q in q_values
        ]
        return {"q": q_values, "q_optimal": q_optimal_values}

    def validate_parameters(self, params: BiologicalOptimizationParams) -> bool:
        """Validate that optimization parameters are mathematically sound."""
        return all(
            getattr(params, attr) > 0
            for attr in ("q_max", "K_m", "K_i")
        )
