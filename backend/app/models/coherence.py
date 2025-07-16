"""Coherence measurement data model placeholder"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

@dataclass
class CoherenceMeasurement:
    user_id: str
    timestamp: datetime
    psi_score: float
    rho_score: float
    q_score: float
    f_score: float
    coherence_score: float
    q_optimal_score: float
    measurement_context: Optional[Dict] = None
    confidence: float = 0.95
