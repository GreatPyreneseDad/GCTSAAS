from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CoherenceComponents:
    """Individual coherence variable measurements"""

    psi: float  # Internal consistency (0.0-1.0)
    rho: float  # Accumulated wisdom (0.0-1.0)
    q: float  # Raw moral activation (0.0-1.0)
    f: float  # Social belonging (0.0-1.0)
    timestamp: datetime
    user_id: str
    context: Optional[Dict] = None
    confidence: float = 0.95


@dataclass
class BiologicalOptimizationParams:
    """Personalized optimization parameters"""

    q_max: float = 1.0  # Maximum coherence contribution
    K_m: float = 0.2  # Half-saturation constant
    K_i: float = 0.8  # Inhibition threshold (burnout prevention)
    user_id: Optional[str] = None


@dataclass
class CoherenceResult:
    """Complete coherence calculation result"""

    coherence_score: Decimal
    q_optimal: Decimal
    components: CoherenceComponents
    optimization_params: BiologicalOptimizationParams
    breakdown: Dict[str, Decimal]
    calculation_metadata: Dict[str, Any]


@dataclass
class CoherenceDerivative:
    """Coherence velocity calculation"""

    timestamp: datetime
    user_id: str
    dC_dt: Decimal  # Total coherence velocity
    dPsi_dt: Decimal  # Internal consistency velocity
    dRho_dt: Decimal  # Wisdom accumulation rate
    dQ_dt: Decimal  # Moral activation change
    dF_dt: Decimal  # Social belonging change
    dQ_optimal_dt: Decimal  # Optimized moral activation derivative
    confidence_interval: Tuple[Decimal, Decimal]
