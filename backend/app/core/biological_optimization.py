"""Utility functions for biological optimization of moral activation"""

from typing import Dict


def calculate_q_optimal(q: float, params: Dict) -> float:
    """Biological optimization formula"""
    q_max = params.get('q_max', 1.0)
    K_m = params.get('K_m', 0.2)
    K_i = params.get('K_i', 0.8)

    denominator = K_m + q + (q ** 2 / K_i)
    return (q_max * q) / denominator if denominator > 0 else 0.0
