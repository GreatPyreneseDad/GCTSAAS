"""Evaluate employees for leadership qualities and intuition using GCT.

This module previously produced single point leadership scores.  The
implementation now supports analysing a series of coherence measurements
so that leadership is measured by *changes* in state over time.  This
better reflects the service mindset of leadership and the way GCT
captures emotional, memory, relational and cognitive dynamics.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import math

from app.core.coherence_calculator import CoherenceCalculator, CoherenceComponents


@dataclass
class LeadershipProfile:
    """Computed leadership and intuition scores for a single employee."""

    user_id: str
    timestamp: datetime
    leadership_score: float
    intuition_score: float
    details: Dict[str, float]


@dataclass
class LeadershipTrajectoryProfile:
    """Aggregated leadership metrics over a series of measurements."""

    user_id: str
    start_timestamp: datetime
    end_timestamp: datetime
    leadership_score: float
    intuition_score: float
    avg_dC_dt: float
    avg_dPsi_dt: float
    avg_dF_dt: float
    avg_dQ_opt_dt: float
    sample_count: int


class LeadershipEvaluator:
    """Service for evaluating leadership qualities and intuition."""

    def __init__(self, calculator: Optional[CoherenceCalculator] = None) -> None:
        self.calculator = calculator or CoherenceCalculator()

    def evaluate(self, components: CoherenceComponents) -> LeadershipProfile:
        """Return leadership and intuition scores for the given components."""
        result = self.calculator.calculate_coherence(components)

        leadership = (
            0.4 * components.psi + 0.4 * components.rho + 0.2 * components.f
        )
        intuition = 0.5 * components.psi + 0.5 * result.q_optimal

        leadership = max(0.0, min(1.0, leadership))
        intuition = max(0.0, min(1.0, intuition))

        details = {
            "psi": components.psi,
            "rho": components.rho,
            "f": components.f,
            "q_optimal": result.q_optimal,
            "coherence_score": result.coherence_score,
        }

        return LeadershipProfile(
            user_id=components.user_id,
            timestamp=components.timestamp,
            leadership_score=leadership,
            intuition_score=intuition,
            details=details,
        )

    def evaluate_series(
        self, history: List[CoherenceComponents]
    ) -> LeadershipTrajectoryProfile:
        """Evaluate leadership over a sequence of measurements."""

        if not history:
            raise ValueError("History must contain at least one measurement")

        if len(history) == 1:
            single = self.evaluate(history[0])
            return LeadershipTrajectoryProfile(
                user_id=single.user_id,
                start_timestamp=single.timestamp,
                end_timestamp=single.timestamp,
                leadership_score=single.leadership_score,
                intuition_score=single.intuition_score,
                avg_dC_dt=0.0,
                avg_dPsi_dt=0.0,
                avg_dF_dt=0.0,
                avg_dQ_opt_dt=0.0,
                sample_count=1,
            )

        results = [self.calculator.calculate_coherence(c) for c in history]

        dC = []
        dPsi = []
        dF = []
        dQopt = []
        for prev, curr in zip(results[:-1], results[1:]):
            dt = (
                curr.components.timestamp - prev.components.timestamp
            ).total_seconds() or 1e-6
            dC.append((curr.coherence_score - prev.coherence_score) / dt)
            dPsi.append((curr.components.psi - prev.components.psi) / dt)
            dF.append((curr.components.f - prev.components.f) / dt)
            dQopt.append((curr.q_optimal - prev.q_optimal) / dt)

        avg_dC_dt = sum(dC) / len(dC)
        avg_dPsi_dt = sum(dPsi) / len(dPsi)
        avg_dF_dt = sum(dF) / len(dF)
        avg_dQ_opt_dt = sum(dQopt) / len(dQopt)

        leadership_score = self._sigmoid(0.5 * avg_dC_dt + 0.5 * avg_dF_dt)
        intuition_score = self._sigmoid(0.5 * avg_dPsi_dt + 0.5 * avg_dQ_opt_dt)

        return LeadershipTrajectoryProfile(
            user_id=history[0].user_id,
            start_timestamp=history[0].timestamp,
            end_timestamp=history[-1].timestamp,
            leadership_score=leadership_score,
            intuition_score=intuition_score,
            avg_dC_dt=avg_dC_dt,
            avg_dPsi_dt=avg_dPsi_dt,
            avg_dF_dt=avg_dF_dt,
            avg_dQ_opt_dt=avg_dQ_opt_dt,
            sample_count=len(history),
        )

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Simple logistic function for normalization."""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

