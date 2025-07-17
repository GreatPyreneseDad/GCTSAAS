"""Dataclasses for persistence of coherence measurements."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class CoherenceMeasurement:
    """Single coherence measurement record."""

    user_id: str
    timestamp: datetime
    psi_score: float
    rho_score: float
    q_score: float
    f_score: float
    coherence_score: float
    q_optimal_score: float
    measurement_context: Optional[Dict[str, Any]] = None
    confidence: float = 0.95

    def to_record(self) -> Dict[str, Any]:
        """Serialize measurement for database insertion."""
        return {
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "psi_score": self.psi_score,
            "rho_score": self.rho_score,
            "q_score": self.q_score,
            "f_score": self.f_score,
            "coherence_score": self.coherence_score,
            "q_optimal_score": self.q_optimal_score,
            "measurement_context": self.measurement_context,
            "confidence": self.confidence,
        }

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "CoherenceMeasurement":
        """Create instance from database row or dictionary."""
        return cls(
            user_id=record["user_id"],
            timestamp=record["timestamp"],
            psi_score=record["psi_score"],
            rho_score=record["rho_score"],
            q_score=record["q_score"],
            f_score=record["f_score"],
            coherence_score=record["coherence_score"],
            q_optimal_score=record["q_optimal_score"],
            measurement_context=record.get("measurement_context"),
            confidence=record.get("confidence", 0.95),
        )
