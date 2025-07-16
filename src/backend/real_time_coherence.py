#!/usr/bin/env python3
"""
Real-Time Coherence Tracking System

Implements sophisticated time-series analysis for GCT coherence derivatives,
inflection point detection, and real-time streaming processing.

Key Features:
- Mathematical derivative calculations for dC/dt
- Biological optimization derivative handling
- Multi-scale inflection point detection
- Real-time streaming with Apache Kafka
- Predictive coherence modeling
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import asyncpg
import numpy as np
import pandas as pd
import redis.asyncio as redis
from kafka import KafkaConsumer, KafkaProducer
from scipy import signal


# ==================== Data Models ====================

@dataclass
class CoherencePoint:
    """Single coherence measurement point"""

    timestamp: datetime
    user_id: str
    psi: float
    rho: float
    q: float
    f: float
    coherence_score: float
    q_optimal: float
    measurement_context: Dict
    confidence: float = 0.95


@dataclass
class CoherenceDerivative:
    """Coherence velocity and component derivatives"""

    timestamp: datetime
    user_id: str
    dC_dt: float
    dPsi_dt: float
    dRho_dt: float
    dQ_dt: float
    dF_dt: float
    dQ_optimal_dt: float
    confidence_interval: Tuple[float, float]


class InflectionType(Enum):
    """Types of coherence inflection points"""

    BREAKTHROUGH = "breakthrough"
    CRISIS = "crisis"
    PLATEAU = "plateau"
    OSCILLATION = "oscillation"
    DISRUPTION = "disruption"


@dataclass
class InflectionPoint:
    """Detected inflection point in coherence trajectory"""

    timestamp: datetime
    user_id: str
    inflection_type: InflectionType
    magnitude: float
    duration_estimate: timedelta
    contributing_factors: Dict[str, float]
    confidence: float
    prediction_horizon: timedelta


# ==================== Core Derivative Calculator ====================

class CoherenceDerivativeCalculator:
    """Implements GCT derivative calculations."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_derivatives(
        self,
        time_series: List[CoherencePoint],
        window_size: int = 7,
        method: str = "savgol",
    ) -> List[CoherenceDerivative]:
        """Calculate coherence derivatives using multiple methods."""
        if len(time_series) < window_size:
            self.logger.warning(
                "Insufficient data points for derivative calculation"
            )
            return []

        timestamps = np.array([pt.timestamp.timestamp() for pt in time_series])
        psi_values = np.array([pt.psi for pt in time_series])
        rho_values = np.array([pt.rho for pt in time_series])
        q_values = np.array([pt.q for pt in time_series])
        f_values = np.array([pt.f for pt in time_series])
        coherence_values = np.array([pt.coherence_score for pt in time_series])
        q_optimal_values = np.array([pt.q_optimal for pt in time_series])

        if method == "savgol":
            derivatives = self._savgol_derivatives(
                timestamps,
                psi_values,
                rho_values,
                q_values,
                f_values,
                coherence_values,
                q_optimal_values,
                window_size,
            )
        elif method == "spline":
            derivatives = self._spline_derivatives(
                timestamps,
                psi_values,
                rho_values,
                q_values,
                f_values,
                coherence_values,
                q_optimal_values,
            )
        else:
            derivatives = self._finite_difference_derivatives(
                timestamps,
                psi_values,
                rho_values,
                q_values,
                f_values,
                coherence_values,
                q_optimal_values,
            )

        result = []
        for i, ts_point in enumerate(time_series[:-1]):
            if i < len(derivatives["dC_dt"]):
                confidence_interval = self._calculate_confidence_interval(
                    derivatives, i, method
                )

                result.append(
                    CoherenceDerivative(
                        timestamp=ts_point.timestamp,
                        user_id=ts_point.user_id,
                        dC_dt=derivatives["dC_dt"][i],
                        dPsi_dt=derivatives["dPsi_dt"][i],
                        dRho_dt=derivatives["dRho_dt"][i],
                        dQ_dt=derivatives["dQ_dt"][i],
                        dF_dt=derivatives["dF_dt"][i],
                        dQ_optimal_dt=derivatives["dQ_optimal_dt"][i],
                        confidence_interval=confidence_interval,
                    )
                )

        return result

    def _savgol_derivatives(
        self,
        timestamps,
        psi,
        rho,
        q,
        f,
        coherence,
        q_optimal,
        window,
    ):
        """Savitzky-Golay filter for smooth derivative estimation."""
        window = window if window % 2 == 1 else window + 1

        try:
            dPsi_dt = signal.savgol_filter(psi, window, 3, deriv=1, delta=1.0)
            dRho_dt = signal.savgol_filter(rho, window, 3, deriv=1, delta=1.0)
            dQ_dt = signal.savgol_filter(q, window, 3, deriv=1, delta=1.0)
            dF_dt = signal.savgol_filter(f, window, 3, deriv=1, delta=1.0)
            dQ_optimal_dt = signal.savgol_filter(
                q_optimal, window, 3, deriv=1, delta=1.0
            )

            psi_mid = psi[:-1]
            rho_mid = rho[:-1]
            f_mid = f[:-1]
            dC_dt = (
                dPsi_dt * (1 + rho_mid + f_mid)
                + dRho_dt * psi_mid
                + dQ_optimal_dt
                + dF_dt * psi_mid
            )

            return {
                "dC_dt": dC_dt,
                "dPsi_dt": dPsi_dt,
                "dRho_dt": dRho_dt,
                "dQ_dt": dQ_dt,
                "dF_dt": dF_dt,
                "dQ_optimal_dt": dQ_optimal_dt,
            }
        except Exception as e:
            self.logger.error(f"Savgol derivative calculation failed: {e}")
            return self._finite_difference_derivatives(
                timestamps, psi, rho, q, f, coherence, q_optimal
            )

    def _finite_difference_derivatives(
        self, timestamps, psi, rho, q, f, coherence, q_optimal
    ):
        """Simple finite difference derivatives."""
        dt = np.diff(timestamps)

        dPsi_dt = np.diff(psi) / dt
        dRho_dt = np.diff(rho) / dt
        dQ_dt = np.diff(q) / dt
        dF_dt = np.diff(f) / dt
        dQ_optimal_dt = np.diff(q_optimal) / dt

        psi_mid = (psi[:-1] + psi[1:]) / 2
        rho_mid = (rho[:-1] + rho[1:]) / 2
        f_mid = (f[:-1] + f[1:]) / 2

        dC_dt = (
            dPsi_dt * (1 + rho_mid + f_mid)
            + dRho_dt * psi_mid
            + dQ_optimal_dt
            + dF_dt * psi_mid
        )

        return {
            "dC_dt": dC_dt,
            "dPsi_dt": dPsi_dt,
            "dRho_dt": dRho_dt,
            "dQ_dt": dQ_dt,
            "dF_dt": dF_dt,
            "dQ_optimal_dt": dQ_optimal_dt,
        }

    def _spline_derivatives(
        self, timestamps, psi, rho, q, f, coherence, q_optimal
    ):
        """Cubic spline interpolation for smooth derivatives."""
        from scipy.interpolate import UnivariateSpline

        splines = {}
        for name, values in [
            ("psi", psi),
            ("rho", rho),
            ("q", q),
            ("f", f),
            ("q_optimal", q_optimal),
        ]:
            splines[name] = UnivariateSpline(timestamps, values, s=0.1)

        dPsi_dt = splines["psi"].derivative()(timestamps[:-1])
        dRho_dt = splines["rho"].derivative()(timestamps[:-1])
        dQ_dt = splines["q"].derivative()(timestamps[:-1])
        dF_dt = splines["f"].derivative()(timestamps[:-1])
        dQ_optimal_dt = splines["q_optimal"].derivative()(timestamps[:-1])

        psi_vals = splines["psi"](timestamps[:-1])
        rho_vals = splines["rho"](timestamps[:-1])
        f_vals = splines["f"](timestamps[:-1])

        dC_dt = (
            dPsi_dt * (1 + rho_vals + f_vals)
            + dRho_dt * psi_vals
            + dQ_optimal_dt
            + dF_dt * psi_vals
        )

        return {
            "dC_dt": dC_dt,
            "dPsi_dt": dPsi_dt,
            "dRho_dt": dRho_dt,
            "dQ_dt": dQ_dt,
            "dF_dt": dF_dt,
            "dQ_optimal_dt": dQ_optimal_dt,
        }

    def _calculate_confidence_interval(
        self, derivatives: Dict, index: int, method: str
    ) -> Tuple[float, float]:
        """Calculate confidence interval for derivative estimate."""
        dC_dt = derivatives["dC_dt"][index]

        if method == "savgol":
            margin = abs(dC_dt) * 0.1
        elif method == "spline":
            margin = abs(dC_dt) * 0.15
        else:
            margin = abs(dC_dt) * 0.2
        return (dC_dt - margin, dC_dt + margin)


# ==================== Inflection Point Detector ====================

class InflectionPointDetector:
    """Detect significant changes in coherence trajectories."""

    def __init__(self, sensitivity: float = 0.1):
        self.sensitivity = sensitivity
        self.logger = logging.getLogger(__name__)

    def detect_inflection_points(
        self,
        derivatives: List[CoherenceDerivative],
        coherence_history: List[CoherencePoint],
    ) -> List[InflectionPoint]:
        """Run multiple detection methods and merge results."""
        inflections: List[InflectionPoint] = []

        if len(derivatives) < 5:
            return inflections

        inflections.extend(self._detect_acceleration_changes(derivatives))
        inflections.extend(self._detect_velocity_changes(derivatives))
        inflections.extend(
            self._detect_pattern_disruptions(derivatives, coherence_history)
        )
        inflections.extend(self._detect_statistical_changepoints(derivatives))

        return self._merge_and_rank_inflections(inflections)

    def _detect_acceleration_changes(
        self, derivatives: List[CoherenceDerivative]
    ) -> List[InflectionPoint]:
        inflections = []
        velocities = np.array([d.dC_dt for d in derivatives])
        timestamps = [d.timestamp for d in derivatives]

        if len(velocities) < 3:
            return inflections

        accelerations = np.diff(velocities)
        zero_crossings = np.where(np.diff(np.signbit(accelerations)))[0]

        for crossing_idx in zero_crossings:
            if crossing_idx + 1 < len(derivatives):
                before_accel = accelerations[max(0, crossing_idx - 1)]
                after_accel = accelerations[
                    min(len(accelerations) - 1, crossing_idx + 1)
                ]
                magnitude = abs(after_accel - before_accel)

                if magnitude > self.sensitivity:
                    if before_accel < 0 and after_accel > 0:
                        inflection_type = InflectionType.BREAKTHROUGH
                    elif before_accel > 0 and after_accel < 0:
                        inflection_type = InflectionType.CRISIS
                    else:
                        inflection_type = InflectionType.PLATEAU

                    contributing_factors = self._analyze_component_contributions(
                        derivatives[max(0, crossing_idx - 2) : crossing_idx + 3]
                    )

                    inflections.append(
                        InflectionPoint(
                            timestamp=timestamps[crossing_idx],
                            user_id=derivatives[crossing_idx].user_id,
                            inflection_type=inflection_type,
                            magnitude=magnitude,
                            duration_estimate=self._estimate_inflection_duration(
                                derivatives, crossing_idx
                            ),
                            contributing_factors=contributing_factors,
                            confidence=min(0.95, magnitude * 2),
                            prediction_horizon=timedelta(days=14),
                        )
                    )
        return inflections

    def _detect_velocity_changes(
        self, derivatives: List[CoherenceDerivative]
    ) -> List[InflectionPoint]:
        inflections = []
        velocities = np.array([d.dC_dt for d in derivatives])
        window_size = min(7, len(velocities) // 3)
        if window_size < 3:
            return inflections

        rolling_std = pd.Series(velocities).rolling(window=window_size).std()
        velocity_magnitude = np.abs(velocities)

        for i in range(window_size, len(velocities) - 1):
            current_magnitude = velocity_magnitude[i]
            baseline_std = rolling_std.iloc[i]
            if baseline_std > 0 and current_magnitude > 3 * baseline_std:
                magnitude = current_magnitude / baseline_std
                prev_velocities = velocities[max(0, i - 3) : i]
                if len(prev_velocities) > 0:
                    trend = np.mean(prev_velocities)
                    if abs(velocities[i]) > abs(trend) * 2:
                        inflection_type = InflectionType.DISRUPTION
                    elif velocities[i] > 0 and trend <= 0:
                        inflection_type = InflectionType.BREAKTHROUGH
                    elif velocities[i] < 0 and trend >= 0:
                        inflection_type = InflectionType.CRISIS
                    else:
                        inflection_type = InflectionType.OSCILLATION

                    contributing_factors = self._analyze_component_contributions(
                        derivatives[max(0, i - 2) : i + 3]
                    )

                    inflections.append(
                        InflectionPoint(
                            timestamp=derivatives[i].timestamp,
                            user_id=derivatives[i].user_id,
                            inflection_type=inflection_type,
                            magnitude=magnitude,
                            duration_estimate=timedelta(days=7),
                            contributing_factors=contributing_factors,
                            confidence=min(0.9, magnitude / 5),
                            prediction_horizon=timedelta(days=10),
                        )
                    )
        return inflections

    def _detect_pattern_disruptions(
        self,
        derivatives: List[CoherenceDerivative],
        coherence_history: List[CoherencePoint],
    ) -> List[InflectionPoint]:
        inflections = []
        if len(coherence_history) < 14:
            return inflections

        coherence_scores = np.array([pt.coherence_score for pt in coherence_history])
        timestamps = np.array([pt.timestamp.timestamp() for pt in coherence_history])
        try:
            from sklearn.linear_model import HuberRegressor

            X = timestamps.reshape(-1, 1)
            baseline_model = HuberRegressor().fit(X, coherence_scores)
            baseline_trend = baseline_model.predict(X)
            residuals = coherence_scores - baseline_trend
            residual_std = np.std(residuals)

            for i in range(7, len(coherence_scores) - 1):
                recent_residuals = residuals[i - 7 : i]
                current_residual = residuals[i]
                if abs(current_residual) > 3 * residual_std:
                    deriv_idx = min(i, len(derivatives) - 1)
                    if current_residual > 0:
                        inflection_type = InflectionType.BREAKTHROUGH
                    else:
                        inflection_type = InflectionType.CRISIS
                    contributing_factors = self._analyze_component_contributions(
                        derivatives[max(0, deriv_idx - 2) : deriv_idx + 3]
                    )
                    inflections.append(
                        InflectionPoint(
                            timestamp=coherence_history[i].timestamp,
                            user_id=coherence_history[i].user_id,
                            inflection_type=inflection_type,
                            magnitude=abs(current_residual) / residual_std,
                            duration_estimate=timedelta(days=5),
                            contributing_factors=contributing_factors,
                            confidence=0.8,
                            prediction_horizon=timedelta(days=7),
                        )
                    )
        except Exception as e:
            self.logger.warning(f"Pattern disruption detection failed: {e}")
        return inflections

    def _detect_statistical_changepoints(
        self, derivatives: List[CoherenceDerivative]
    ) -> List[InflectionPoint]:
        inflections = []
        if len(derivatives) < 10:
            return inflections
        velocities = np.array([d.dC_dt for d in derivatives])
        cumsum = np.cumsum(velocities - np.mean(velocities))
        threshold = 2 * np.std(cumsum)
        changepoints = []
        for i in range(1, len(cumsum)):
            if abs(cumsum[i] - cumsum[i - 1]) > threshold:
                changepoints.append(i)
        for cp_idx in changepoints:
            if cp_idx < len(derivatives):
                before_window = velocities[max(0, cp_idx - 3) : cp_idx]
                after_window = velocities[cp_idx : min(len(velocities), cp_idx + 3)]
                if len(before_window) > 0 and len(after_window) > 0:
                    before_mean = np.mean(before_window)
                    after_mean = np.mean(after_window)
                    magnitude = abs(after_mean - before_mean)
                    if after_mean > before_mean:
                        inflection_type = InflectionType.BREAKTHROUGH
                    else:
                        inflection_type = InflectionType.CRISIS
                    contributing_factors = self._analyze_component_contributions(
                        derivatives[max(0, cp_idx - 2) : cp_idx + 3]
                    )
                    inflections.append(
                        InflectionPoint(
                            timestamp=derivatives[cp_idx].timestamp,
                            user_id=derivatives[cp_idx].user_id,
                            inflection_type=inflection_type,
                            magnitude=magnitude,
                            duration_estimate=timedelta(days=6),
                            contributing_factors=contributing_factors,
                            confidence=0.75,
                            prediction_horizon=timedelta(days=8),
                        )
                    )
        return inflections

    def _analyze_component_contributions(
        self, derivative_window: List[CoherenceDerivative]
    ) -> Dict[str, float]:
        if not derivative_window:
            return {}
        components = {
            "psi_velocity": [d.dPsi_dt for d in derivative_window],
            "rho_velocity": [d.dRho_dt for d in derivative_window],
            "q_velocity": [d.dQ_dt for d in derivative_window],
            "f_velocity": [d.dF_dt for d in derivative_window],
            "q_optimal_velocity": [d.dQ_optimal_dt for d in derivative_window],
        }
        contributions: Dict[str, float] = {}
        total_variance = 0.0
        for component, values in components.items():
            if len(values) > 1:
                variance = np.var(values)
                contributions[component] = variance
                total_variance += variance
        if total_variance > 0:
            for component in contributions:
                contributions[component] /= total_variance
        return contributions

    def _estimate_inflection_duration(
        self, derivatives: List[CoherenceDerivative], inflection_idx: int
    ) -> timedelta:
        if inflection_idx >= len(derivatives):
            return timedelta(days=7)
        current_velocity = abs(derivatives[inflection_idx].dC_dt)
        historical_velocities = [abs(d.dC_dt) for d in derivatives[:inflection_idx]]
        if historical_velocities:
            avg_velocity = np.mean(historical_velocities)
            velocity_ratio = current_velocity / avg_velocity if avg_velocity > 0 else 1
            base_duration = 7
            adjusted_duration = base_duration / max(1, velocity_ratio * 0.5)
            return timedelta(days=max(3, min(21, adjusted_duration)))
        return timedelta(days=7)

    def _merge_and_rank_inflections(
        self, inflections: List[InflectionPoint]
    ) -> List[InflectionPoint]:
        if not inflections:
            return []
        sorted_inflections = sorted(inflections, key=lambda x: x.timestamp)
        merged: List[InflectionPoint] = []
        current_group = [sorted_inflections[0]]
        for inflection in sorted_inflections[1:]:
            time_diff = inflection.timestamp - current_group[-1].timestamp
            if time_diff <= timedelta(hours=48):
                current_group.append(inflection)
            else:
                merged_inflection = self._merge_inflection_group(current_group)
                if merged_inflection:
                    merged.append(merged_inflection)
                current_group = [inflection]
        if current_group:
            merged_inflection = self._merge_inflection_group(current_group)
            if merged_inflection:
                merged.append(merged_inflection)
        ranked = sorted(
            merged, key=lambda x: x.confidence * x.magnitude, reverse=True
        )
        return ranked[:10]

    def _merge_inflection_group(
        self, group: List[InflectionPoint]
    ) -> Optional[InflectionPoint]:
        if not group:
            return None
        if len(group) == 1:
            return group[0]
        primary = max(group, key=lambda x: x.confidence)
        avg_magnitude = np.mean([ip.magnitude for ip in group])
        combined_factors: Dict[str, float] = {}
        for ip in group:
            for factor, value in ip.contributing_factors.items():
                combined_factors[factor] = combined_factors.get(factor, 0) + value
        total_factor_weight = sum(combined_factors.values())
        if total_factor_weight > 0:
            for factor in combined_factors:
                combined_factors[factor] /= total_factor_weight
        return InflectionPoint(
            timestamp=primary.timestamp,
            user_id=primary.user_id,
            inflection_type=primary.inflection_type,
            magnitude=avg_magnitude,
            duration_estimate=primary.duration_estimate,
            contributing_factors=combined_factors,
            confidence=primary.confidence,
            prediction_horizon=primary.prediction_horizon,
        )


# ==================== Real-Time Streaming Processor ====================

class RealTimeCoherenceProcessor:
    """Real-time streaming of coherence measurements and derivatives."""

    def __init__(self, kafka_servers: List[str], redis_url: str, db_pool):
        self.kafka_servers = kafka_servers
        self.redis_client = redis.from_url(redis_url)
        self.db_pool = db_pool
        self.derivative_calculator = CoherenceDerivativeCalculator()
        self.inflection_detector = InflectionPointDetector()
        self.logger = logging.getLogger(__name__)
        self.input_topic = "coherence_measurements"
        self.derivative_topic = "coherence_derivatives"
        self.inflection_topic = "coherence_inflections"
        self.alert_topic = "coherence_alerts"

    async def start_processing(self):
        tasks = [
            self._process_incoming_measurements(),
            self._calculate_real_time_derivatives(),
            self._detect_real_time_inflections(),
            self._generate_alerts(),
        ]
        await asyncio.gather(*tasks)

    async def _process_incoming_measurements(self):
        consumer = KafkaConsumer(
            self.input_topic,
            bootstrap_servers=self.kafka_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id="coherence_processor",
        )
        try:
            for message in consumer:
                measurement_data = message.value
                measurement = CoherencePoint(
                    timestamp=datetime.fromisoformat(measurement_data["timestamp"]),
                    user_id=measurement_data["user_id"],
                    psi=measurement_data["psi"],
                    rho=measurement_data["rho"],
                    q=measurement_data["q"],
                    f=measurement_data["f"],
                    coherence_score=measurement_data["coherence_score"],
                    q_optimal=measurement_data["q_optimal"],
                    measurement_context=measurement_data.get("context", {}),
                    confidence=measurement_data.get("confidence", 0.95),
                )
                await self._store_measurement(measurement)
                await self._update_user_buffer(measurement)
                await self._trigger_derivative_calculation(measurement.user_id)
        except Exception as e:
            self.logger.error(f"Error processing measurements: {e}")

    async def _store_measurement(self, measurement: CoherencePoint):
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
            INSERT INTO coherence_measurements
            (user_id, timestamp, psi_score, rho_score, q_score, f_score,
             coherence_score, q_optimal_score, measurement_context, confidence)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
                measurement.user_id,
                measurement.timestamp,
                measurement.psi,
                measurement.rho,
                measurement.q,
                measurement.f,
                measurement.coherence_score,
                measurement.q_optimal,
                json.dumps(measurement.measurement_context),
                measurement.confidence,
            )

    async def _update_user_buffer(self, measurement: CoherencePoint):
        buffer_key = f"user_buffer:{measurement.user_id}"
        measurement_json = {
            "timestamp": measurement.timestamp.isoformat(),
            "psi": measurement.psi,
            "rho": measurement.rho,
            "q": measurement.q,
            "f": measurement.f,
            "coherence_score": measurement.coherence_score,
            "q_optimal": measurement.q_optimal,
            "context": measurement.measurement_context,
            "confidence": measurement.confidence,
        }
        await self.redis_client.lpush(buffer_key, json.dumps(measurement_json))
        await self.redis_client.ltrim(buffer_key, 0, 29)
        await self.redis_client.expire(buffer_key, 86400)

    async def _trigger_derivative_calculation(self, user_id: str):
        buffer_key = f"user_buffer:{user_id}"
        buffer_size = await self.redis_client.llen(buffer_key)
        if buffer_size >= 7:
            producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            producer.send(
                self.derivative_topic,
                {
                    "user_id": user_id,
                    "action": "calculate_derivatives",
                    "timestamp": datetime.now().isoformat(),
                },
            )
            producer.close()

    async def _calculate_real_time_derivatives(self):
        consumer = KafkaConsumer(
            self.derivative_topic,
            bootstrap_servers=self.kafka_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id="derivative_calculator",
        )
        try:
            for message in consumer:
                trigger_data = message.value
                user_id = trigger_data["user_id"]
                measurements = await self._get_user_measurements(user_id)
                if len(measurements) >= 7:
                    derivatives = self.derivative_calculator.calculate_derivatives(
                        measurements, window_size=7, method="savgol"
                    )
                    if derivatives:
                        latest_derivative = derivatives[-1]
                        await self._store_derivative(latest_derivative)
                        producer = KafkaProducer(
                            bootstrap_servers=self.kafka_servers,
                            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                        )
                        producer.send(
                            self.inflection_topic,
                            {
                                "user_id": user_id,
                                "derivative": {
                                    "timestamp": latest_derivative.timestamp.isoformat(),
                                    "dC_dt": latest_derivative.dC_dt,
                                    "dPsi_dt": latest_derivative.dPsi_dt,
                                    "dRho_dt": latest_derivative.dRho_dt,
                                    "dQ_dt": latest_derivative.dQ_dt,
                                    "dF_dt": latest_derivative.dF_dt,
                                    "dQ_optimal_dt": latest_derivative.dQ_optimal_dt,
                                    "confidence_interval": latest_derivative.confidence_interval,
                                },
                            },
                        )
                        producer.close()
        except Exception as e:
            self.logger.error(f"Error calculating derivatives: {e}")

    async def _get_user_measurements(
        self, user_id: str, window_days: int = 30
    ) -> List[CoherencePoint]:
        buffer_key = f"user_buffer:{user_id}"
        buffer_data = await self.redis_client.lrange(buffer_key, 0, -1)
        if len(buffer_data) >= 7:
            measurements = []
            for item in reversed(buffer_data):
                data = json.loads(item)
                measurements.append(
                    CoherencePoint(
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        user_id=user_id,
                        psi=data["psi"],
                        rho=data["rho"],
                        q=data["q"],
                        f=data["f"],
                        coherence_score=data["coherence_score"],
                        q_optimal=data["q_optimal"],
                        measurement_context=data["context"],
                        confidence=data["confidence"],
                    )
                )
            return measurements
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT timestamp, psi_score, rho_score, q_score, f_score,
                       coherence_score, q_optimal_score, measurement_context, confidence
                FROM coherence_measurements
                WHERE user_id = $1
                AND timestamp >= NOW() - INTERVAL '%s days'
                ORDER BY timestamp ASC
                """,
                user_id,
                window_days,
            )
            measurements = []
            for row in rows:
                measurements.append(
                    CoherencePoint(
                        timestamp=row["timestamp"],
                        user_id=user_id,
                        psi=row["psi_score"],
                        rho=row["rho_score"],
                        q=row["q_score"],
                        f=row["f_score"],
                        coherence_score=row["coherence_score"],
                        q_optimal=row["q_optimal_score"],
                        measurement_context=json.loads(row["measurement_context"] or "{}"),
                        confidence=row["confidence"],
                    )
                )
            return measurements

    async def _store_derivative(self, derivative: CoherenceDerivative):
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO coherence_derivatives
                (user_id, timestamp, dC_dt, dPsi_dt, dRho_dt, dQ_dt, dF_dt,
                 dQ_optimal_dt, confidence_lower, confidence_upper)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                derivative.user_id,
                derivative.timestamp,
                derivative.dC_dt,
                derivative.dPsi_dt,
                derivative.dRho_dt,
                derivative.dQ_dt,
                derivative.dF_dt,
                derivative.dQ_optimal_dt,
                derivative.confidence_interval[0],
                derivative.confidence_interval[1],
            )

    async def _detect_real_time_inflections(self):
        consumer = KafkaConsumer(
            self.inflection_topic,
            bootstrap_servers=self.kafka_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id="inflection_detector",
        )
        try:
            for message in consumer:
                data = message.value
                user_id = data["user_id"]
                derivatives = await self._get_user_derivatives(user_id)
                measurements = await self._get_user_measurements(user_id)
                if len(derivatives) >= 5:
                    inflections = self.inflection_detector.detect_inflection_points(
                        derivatives, measurements
                    )
                    for inflection in inflections:
                        if inflection.confidence > 0.7:
                            await self._store_inflection(inflection)
                            if inflection.magnitude > 1.5:
                                producer = KafkaProducer(
                                    bootstrap_servers=self.kafka_servers,
                                    value_serializer=lambda v: json.dumps(v).encode(
                                        "utf-8"
                                    ),
                                )
                                producer.send(
                                    self.alert_topic,
                                    {
                                        "user_id": user_id,
                                        "inflection_type": inflection.inflection_type.value,
                                        "magnitude": inflection.magnitude,
                                        "timestamp": inflection.timestamp.isoformat(),
                                        "confidence": inflection.confidence,
                                        "contributing_factors": inflection.contributing_factors,
                                    },
                                )
                                producer.close()
        except Exception as e:
            self.logger.error(f"Error detecting inflections: {e}")

    async def _get_user_derivatives(
        self, user_id: str, window_days: int = 14
    ) -> List[CoherenceDerivative]:
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT timestamp, dC_dt, dPsi_dt, dRho_dt, dQ_dt, dF_dt,
                       dQ_optimal_dt, confidence_lower, confidence_upper
                FROM coherence_derivatives
                WHERE user_id = $1
                AND timestamp >= NOW() - INTERVAL '%s days'
                ORDER BY timestamp ASC
                """,
                user_id,
                window_days,
            )
            derivatives = []
            for row in rows:
                derivatives.append(
                    CoherenceDerivative(
                        timestamp=row["timestamp"],
                        user_id=user_id,
                        dC_dt=row["dC_dt"],
                        dPsi_dt=row["dPsi_dt"],
                        dRho_dt=row["dRho_dt"],
                        dQ_dt=row["dQ_dt"],
                        dF_dt=row["dF_dt"],
                        dQ_optimal_dt=row["dQ_optimal_dt"],
                        confidence_interval=(
                            row["confidence_lower"],
                            row["confidence_upper"],
                        ),
                    )
                )
            return derivatives

    async def _store_inflection(self, inflection: InflectionPoint):
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO coherence_inflections
                (user_id, timestamp, inflection_type, magnitude, duration_estimate_days,
                 contributing_factors, confidence, prediction_horizon_days)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                inflection.user_id,
                inflection.timestamp,
                inflection.inflection_type.value,
                inflection.magnitude,
                inflection.duration_estimate.days,
                json.dumps(inflection.contributing_factors),
                inflection.confidence,
                inflection.prediction_horizon.days,
            )

    async def _generate_alerts(self):
        consumer = KafkaConsumer(
            self.alert_topic,
            bootstrap_servers=self.kafka_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id="alert_processor",
        )
        try:
            for message in consumer:
                alert_data = message.value
                await self._process_coherence_alert(alert_data)
        except Exception as e:
            self.logger.error(f"Error processing alerts: {e}")

    async def _process_coherence_alert(self, alert_data: Dict):
        user_id = alert_data["user_id"]
        inflection_type = alert_data["inflection_type"]
        magnitude = alert_data["magnitude"]
        async with self.db_pool.acquire() as conn:
            user_data = await conn.fetchrow(
                """
                SELECT u.*, o.name as org_name, o.alert_settings
                FROM users u
                JOIN organizations o ON u.org_id = o.id
                WHERE u.id = $1
                """,
                user_id,
            )
            if user_data:
                alert_settings = json.loads(user_data["alert_settings"] or "{}")
                if inflection_type == "breakthrough" and magnitude > 2.0:
                    await self._send_breakthrough_notification(user_data, alert_data)
                elif inflection_type == "crisis" and magnitude > 2.5:
                    await self._send_crisis_alert(user_data, alert_data)
                elif inflection_type == "disruption":
                    await self._send_disruption_warning(user_data, alert_data)

    async def _send_breakthrough_notification(self, user_data, alert_data):
        self.logger.info(
            f"Breakthrough detected for user {user_data['id']}: {alert_data}"
        )

    async def _send_crisis_alert(self, user_data, alert_data):
        self.logger.warning(
            f"Crisis pattern detected for user {user_data['id']}: {alert_data}"
        )

    async def _send_disruption_warning(self, user_data, alert_data):
        self.logger.info(
            f"Disruption detected for user {user_data['id']}: {alert_data}"
        )


# ==================== Usage Example ====================

async def main():
    """Example usage of the real-time coherence tracking system."""
    db_pool = await asyncpg.create_pool(
        "postgresql://user:pass@localhost/coherence_db"
    )
    processor = RealTimeCoherenceProcessor(
        kafka_servers=["localhost:9092"],
        redis_url="redis://localhost:6379",
        db_pool=db_pool,
    )
    await processor.start_processing()


if __name__ == "__main__":
    asyncio.run(main())
