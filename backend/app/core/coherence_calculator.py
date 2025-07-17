"""Core GCT coherence calculation engine."""

from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional
from datetime import datetime

from .models import (
    BiologicalOptimizationParams,
    CoherenceComponents,
    CoherenceDerivative,
    CoherenceResult,
)
from .biological_optimization import BiologicalOptimizer

getcontext().prec = 10


class CoherenceCalculator:
    """Core GCT coherence calculation engine"""

    def __init__(self) -> None:
        self.biological_optimizer = BiologicalOptimizer()
        self.default_params = BiologicalOptimizationParams()

    def calculate_coherence(
        self,
        components: CoherenceComponents,
        optimization_params: Optional[BiologicalOptimizationParams] = None,
    ) -> CoherenceResult:
        """Calculate coherence using complete GCT formula.

        C = Ψ + (ρ × Ψ) + q^optimal + (f × Ψ)
        """
        params = optimization_params or self.default_params
        issues = self.validate_measurement(components)
        if issues:
            raise ValueError(f"Invalid measurement: {issues}")

        q_opt = self.biological_optimizer.calculate_q_optimal(components.q, params)
        psi = Decimal(str(components.psi))
        rho = Decimal(str(components.rho))
        f_val = Decimal(str(components.f))
        wisdom_amplification = rho * psi
        social_boost = f_val * psi
        coherence_score = psi + wisdom_amplification + q_opt + social_boost

        breakdown = {
            "base_consistency": psi,
            "wisdom_amplification": wisdom_amplification,
            "optimized_moral_energy": q_opt,
            "social_coherence_boost": social_boost,
        }

        metadata = {
            "timestamp": components.timestamp.isoformat(),
            "user_id": components.user_id,
        }

        return CoherenceResult(
            coherence_score=coherence_score,
            q_optimal=q_opt,
            components=components,
            optimization_params=params,
            breakdown=breakdown,
            calculation_metadata=metadata,
        )

    def calculate_component_contributions(self, result: CoherenceResult) -> Dict[str, float]:
        """Calculate percentage contribution of each component."""
        total = float(result.coherence_score)
        if total == 0:
            return {k: 0.0 for k in result.breakdown}
        return {k: float(v) / total for k, v in result.breakdown.items()}

    def validate_measurement(self, components: CoherenceComponents) -> List[str]:
        """Validate measurement data and return issues."""
        issues: List[str] = []
        for name in ("psi", "rho", "q", "f"):
            val = getattr(components, name)
            if not (0.0 <= val <= 1.0):
                issues.append(f"{name} out of range")
        return issues

    def get_coherence_interpretation(self, coherence_score: float) -> Dict[str, str]:
        """Human-readable interpretation of coherence score."""
        if coherence_score >= 2.5:
            level = "High coherence"
        elif coherence_score >= 1.5:
            level = "Moderate coherence"
        else:
            level = "Development opportunity"
        return {"level": level}
