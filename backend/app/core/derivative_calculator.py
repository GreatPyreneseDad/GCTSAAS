"""Utility functions for calculating coherence derivatives.

This module exposes a small wrapper that works on :class:`CoherenceResult`
objects produced by :class:`~app.core.coherence_calculator.CoherenceCalculator`.
It mirrors the more fully fledged implementation in ``src/backend`` but keeps
the dependencies minimal for use inside the FastAPI application.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np
from scipy import signal

from .coherence_calculator import CoherenceResult


@dataclass
class CoherenceDerivative:
    """Derivative of coherence and its components at a point in time."""

    timestamp: datetime
    user_id: str
    dC_dt: float
    dPsi_dt: float
    dRho_dt: float
    dQ_dt: float
    dF_dt: float
    dQ_optimal_dt: float
    confidence_interval: Tuple[float, float]


class DerivativeCalculator:
    """Calculate coherence derivatives for a time series of results."""

    def __init__(self, window_size: int = 5, method: str = "savgol") -> None:
        self.window_size = max(3, window_size)
        self.method = method

    def calculate_derivatives(self, measurements: List[CoherenceResult]) -> List[CoherenceDerivative]:
        """Return derivatives for the provided coherence measurements.

        Parameters
        ----------
        measurements:
            List of :class:`CoherenceResult` objects sorted by timestamp.
        """

        if len(measurements) < 2:
            return []

        timestamps = np.array([m.components.timestamp.timestamp() for m in measurements])
        psi = np.array([m.components.psi for m in measurements])
        rho = np.array([m.components.rho for m in measurements])
        q = np.array([m.components.q for m in measurements])
        f = np.array([m.components.f for m in measurements])
        q_opt = np.array([m.q_optimal for m in measurements])

        dt = np.diff(timestamps)
        dt[dt == 0] = 1e-6

        if self.method == "savgol" and len(measurements) >= self.window_size:
            window = self.window_size | 1
            dPsi_dt = signal.savgol_filter(psi, window, 3, deriv=1, delta=np.mean(dt))
            dRho_dt = signal.savgol_filter(rho, window, 3, deriv=1, delta=np.mean(dt))
            dQ_dt = signal.savgol_filter(q, window, 3, deriv=1, delta=np.mean(dt))
            dF_dt = signal.savgol_filter(f, window, 3, deriv=1, delta=np.mean(dt))
            dQ_opt_dt = signal.savgol_filter(q_opt, window, 3, deriv=1, delta=np.mean(dt))
        else:
            dPsi_dt = np.concatenate(([0.0], np.diff(psi) / dt))
            dRho_dt = np.concatenate(([0.0], np.diff(rho) / dt))
            dQ_dt = np.concatenate(([0.0], np.diff(q) / dt))
            dF_dt = np.concatenate(([0.0], np.diff(f) / dt))
            dQ_opt_dt = np.concatenate(([0.0], np.diff(q_opt) / dt))

        psi_mid = psi[:-1]
        rho_mid = rho[:-1]
        f_mid = f[:-1]

        dC_dt = (
            dPsi_dt[:-1] * (1 + rho_mid + f_mid)
            + dRho_dt[:-1] * psi_mid
            + dQ_opt_dt[:-1]
            + dF_dt[:-1] * psi_mid
        )

        results: List[CoherenceDerivative] = []
        for i in range(len(dC_dt)):
            margin = abs(dC_dt[i]) * 0.1
            ci = (dC_dt[i] - margin, dC_dt[i] + margin)
            results.append(
                CoherenceDerivative(
                    timestamp=measurements[i].components.timestamp,
                    user_id=measurements[i].components.user_id,
                    dC_dt=dC_dt[i],
                    dPsi_dt=dPsi_dt[i],
                    dRho_dt=dRho_dt[i],
                    dQ_dt=dQ_dt[i],
                    dF_dt=dF_dt[i],
                    dQ_optimal_dt=dQ_opt_dt[i],
                    confidence_interval=ci,
                )
            )
        return results
