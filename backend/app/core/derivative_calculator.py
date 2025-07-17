"""Derivative calculations for GCT coherence."""

from decimal import Decimal
from typing import Dict, List, Optional

try:
    import numpy as np
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - optional deps
    np = None
    savgol_filter = None

from .biological_optimization import BiologicalOptimizer
from .models import (
    BiologicalOptimizationParams,
    CoherenceComponents,
    CoherenceDerivative,
)


class DerivativeCalculator:
    """Calculate coherence derivatives for real-time tracking"""

    def __init__(self) -> None:
        self.optimizer = BiologicalOptimizer()

    def calculate_coherence_derivative(
        self,
        measurements: List[CoherenceComponents],
        optimization_params: Optional[BiologicalOptimizationParams] = None,
        method: str = "finite_difference",
    ) -> Optional[CoherenceDerivative]:
        """Calculate derivative using selected method."""
        if len(measurements) < 2:
            return None
        if method == "savitzky_golay" and len(measurements) < 5:
            return None
        if method == "savitzky_golay":
            return self._savitzky_golay_method(measurements, optimization_params)
        return self._finite_difference_method(measurements, optimization_params)

    def _finite_difference_method(
        self,
        measurements: List[CoherenceComponents],
        optimization_params: Optional[BiologicalOptimizationParams] = None,
    ) -> CoherenceDerivative:
        m1, m2 = measurements[-2], measurements[-1]
        dt = (m2.timestamp - m1.timestamp).total_seconds() or 1.0
        params = optimization_params or BiologicalOptimizationParams()
        dq_opt_dt = self.optimizer.calculate_q_optimal_derivative(
            m2.q, (m2.q - m1.q) / dt, params
        )
        dPsi_dt = Decimal(str((m2.psi - m1.psi) / dt))
        dRho_dt = Decimal(str((m2.rho - m1.rho) / dt))
        dQ_dt = Decimal(str((m2.q - m1.q) / dt))
        dF_dt = Decimal(str((m2.f - m1.f) / dt))
        dC_dt = dPsi_dt * (1 + Decimal(str(m2.rho)) + Decimal(str(m2.f))) + (
            dRho_dt * Decimal(str(m2.psi))
        ) + dq_opt_dt + dF_dt * Decimal(str(m2.psi))

        return CoherenceDerivative(
            timestamp=m2.timestamp,
            user_id=m2.user_id,
            dC_dt=dC_dt,
            dPsi_dt=dPsi_dt,
            dRho_dt=dRho_dt,
            dQ_dt=dQ_dt,
            dF_dt=dF_dt,
            dQ_optimal_dt=dq_opt_dt,
            confidence_interval=(Decimal("0"), Decimal("0")),
        )

    def _savitzky_golay_method(
        self,
        measurements: List[CoherenceComponents],
        optimization_params: Optional[BiologicalOptimizationParams] = None,
    ) -> CoherenceDerivative:
        if np is None or savgol_filter is None:
            raise RuntimeError("NumPy and SciPy are required for savitzky_golay method")

        times = np.array([m.timestamp.timestamp() for m in measurements])
        psi = np.array([m.psi for m in measurements])
        rho = np.array([m.rho for m in measurements])
        q = np.array([m.q for m in measurements])
        f = np.array([m.f for m in measurements])
        dt = times[1] - times[0]
        psi_d = savgol_filter(psi, 5, 2, deriv=1, delta=dt)
        rho_d = savgol_filter(rho, 5, 2, deriv=1, delta=dt)
        q_d = savgol_filter(q, 5, 2, deriv=1, delta=dt)
        f_d = savgol_filter(f, 5, 2, deriv=1, delta=dt)
        params = optimization_params or BiologicalOptimizationParams()
        dq_opt_dt = self.optimizer.calculate_q_optimal_derivative(
            q[-1], q_d[-1], params
        )
        dC_dt = Decimal(str(
            psi_d[-1] * (1 + rho[-1] + f[-1]) + rho_d[-1] * psi[-1] + q_d[-1] + f_d[-1] * psi[-1]
        ))

        return CoherenceDerivative(
            timestamp=measurements[-1].timestamp,
            user_id=measurements[-1].user_id,
            dC_dt=dC_dt,
            dPsi_dt=Decimal(str(psi_d[-1])),
            dRho_dt=Decimal(str(rho_d[-1])),
            dQ_dt=Decimal(str(q_d[-1])),
            dF_dt=Decimal(str(f_d[-1])),
            dQ_optimal_dt=dq_opt_dt,
            confidence_interval=(Decimal("0"), Decimal("0")),
        )

    def detect_inflection_points(
        self, derivatives: List[CoherenceDerivative]
    ) -> List[Dict]:
        """Detect significant changes in coherence trajectory."""
        points: List[Dict] = []
        for i in range(1, len(derivatives)):
            prev, curr = derivatives[i - 1], derivatives[i]
            if prev.dC_dt * curr.dC_dt < 0:
                points.append({
                    "timestamp": curr.timestamp,
                    "user_id": curr.user_id,
                    "magnitude": float(curr.dC_dt - prev.dC_dt),
                })
        return points
