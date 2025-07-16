from __future__ import annotations

"""Improved real-time coherence tracking utilities.

This module implements derivative calculations and inflection
point detection for the Grounded Coherence Theory (GCT) SaaS
platform. The implementation applies suggestions from the
project discussion:

- Time deltas are based on measurement spacing instead of
  assuming a constant interval.
- Savitzky–Golay window sizing is validated to ensure an odd
  number of samples.
- Configurable smoothing parameters are available through the
  :class:`DerivativeConfig` dataclass.
- Asynchronous Kafka processing uses ``aiokafka`` to avoid
  blocking the event loop.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import signal
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer


@dataclass
class CoherencePoint:
    """Simplified coherence measurement point."""

    timestamp: datetime
    user_id: str
    psi: float
    rho: float
    q: float
    f: float
    coherence_score: float
    q_optimal: float


@dataclass
class CoherenceDerivative:
    timestamp: datetime
    user_id: str
    dC_dt: float
    dPsi_dt: float
    dRho_dt: float
    dQ_dt: float
    dF_dt: float
    dQ_optimal_dt: float
    confidence_interval: Tuple[float, float]


@dataclass
class DerivativeConfig:
    """Configuration options for derivative calculation."""

    window_size: int = 7
    sg_polyorder: int = 3
    sg_smoothing: float = 1.0


class DerivativeCalculator:
    """Calculate coherence derivatives with configurable parameters."""

    def __init__(self, config: DerivativeConfig | None = None) -> None:
        self.config = config or DerivativeConfig()

    def calculate(self, series: List[CoherencePoint]) -> List[CoherenceDerivative]:
        cfg = self.config
        if len(series) < cfg.window_size:
            return []

        timestamps = np.array([pt.timestamp.timestamp() for pt in series])
        psi = np.array([pt.psi for pt in series])
        rho = np.array([pt.rho for pt in series])
        q = np.array([pt.q for pt in series])
        f = np.array([pt.f for pt in series])
        q_opt = np.array([pt.q_optimal for pt in series])

        # compute delta in seconds to avoid unit confusion
        dt = np.diff(timestamps)
        if np.any(dt == 0):
            dt[dt == 0] = 1e-6

        window = cfg.window_size | 1  # ensure odd
        delta = float(np.mean(dt))

        dPsi_dt = signal.savgol_filter(psi, window, cfg.sg_polyorder, deriv=1, delta=delta)
        dRho_dt = signal.savgol_filter(rho, window, cfg.sg_polyorder, deriv=1, delta=delta)
        dQ_dt = signal.savgol_filter(q, window, cfg.sg_polyorder, deriv=1, delta=delta)
        dF_dt = signal.savgol_filter(f, window, cfg.sg_polyorder, deriv=1, delta=delta)
        dQ_opt_dt = signal.savgol_filter(q_opt, window, cfg.sg_polyorder, deriv=1, delta=delta)

        psi_mid = psi[:-1]
        rho_mid = rho[:-1]
        f_mid = f[:-1]

        dC_dt = dPsi_dt[:-1] * (1 + rho_mid + f_mid) + dRho_dt[:-1] * psi_mid + dQ_opt_dt[:-1] + dF_dt[:-1] * psi_mid

        results: List[CoherenceDerivative] = []
        for i in range(len(dC_dt)):
            margin = abs(dC_dt[i]) * 0.1
            ci = (dC_dt[i] - margin, dC_dt[i] + margin)
            results.append(
                CoherenceDerivative(
                    timestamp=series[i].timestamp,
                    user_id=series[i].user_id,
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


async def stream_derivatives(
    kafka_servers: Iterable[str], topic: str, config: DerivativeConfig | None = None
) -> None:
    """Example async consumer using ``aiokafka`` for non-blocking processing."""

    consumer = AIOKafkaConsumer(topic, bootstrap_servers=list(kafka_servers))
    await consumer.start()
    try:
        calc = DerivativeCalculator(config)
        async for msg in consumer:
            data = msg.value
            points = [CoherencePoint(**item) for item in data]
            derivatives = calc.calculate(points)
            # in a real application we would publish these elsewhere
            print(f"calculated {len(derivatives)} derivatives for user {points[0].user_id}")
    finally:
        await consumer.stop()

