"""
Core GCT coherence calculation engine
Implements: C = Ψ + (ρ × Ψ) + q^optimal + (f × Ψ)
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CoherenceComponents:
    psi: float  # Internal consistency (0.0-1.0)
    rho: float  # Accumulated wisdom (0.0-1.0)
    q: float    # Raw moral activation (0.0-1.0)
    f: float    # Social belonging (0.0-1.0)
    timestamp: datetime
    user_id: str
    context: Optional[Dict] = None

@dataclass
class CoherenceResult:
    coherence_score: float
    q_optimal: float
    components: CoherenceComponents
    breakdown: Dict[str, float]

class CoherenceCalculator:
    def __init__(self):
        # Default biological optimization parameters
        self.default_params = {
            'q_max': 1.0,
            'K_m': 0.2,    # Half-saturation constant
            'K_i': 0.8     # Inhibition threshold
        }
    
    def calculate_coherence(self, components: CoherenceComponents, 
                          user_params: Optional[Dict] = None) -> CoherenceResult:
        """
        Calculate coherence using GCT formula with biological optimization
        """
        # Get user-specific or default parameters
        params = user_params or self.default_params
        
        # Calculate biological optimization of moral activation
        q_optimal = self._calculate_q_optimal(components.q, params)
        
        # GCT coherence formula: C = Ψ + (ρ × Ψ) + q^optimal + (f × Ψ)
        base_consistency = components.psi
        wisdom_amplification = components.rho * components.psi
        optimized_moral_energy = q_optimal
        social_coherence_boost = components.f * components.psi
        
        coherence_score = (
            base_consistency + 
            wisdom_amplification + 
            optimized_moral_energy + 
            social_coherence_boost
        )
        
        # Create breakdown for analysis
        breakdown = {
            'base_consistency': base_consistency,
            'wisdom_amplification': wisdom_amplification,
            'optimized_moral_energy': optimized_moral_energy,
            'social_coherence_boost': social_coherence_boost,
            'total_coherence': coherence_score
        }
        
        return CoherenceResult(
            coherence_score=coherence_score,
            q_optimal=q_optimal,
            components=components,
            breakdown=breakdown
        )
    
    def _calculate_q_optimal(self, q: float, params: Dict) -> float:
        """
        Biological optimization: q^optimal = (q_max × q) / (K_m + q + q²/K_i)
        """
        q_max = params['q_max']
        K_m = params['K_m']
        K_i = params['K_i']
        
        numerator = q_max * q
        denominator = K_m + q + (q**2 / K_i)
        
        return numerator / denominator if denominator > 0 else 0.0
