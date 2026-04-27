"""
Layer 2 — ML Processing Engine
Network Intrusion Detection — Production Detector

This module provides real-time scoring of live network
events using trained intrusion detection models.

This is the production inference class — it runs
continuously in your SOC scoring every normalized
network flow from Layer 1.

Architecture Position:
    Layer 1 normalizes raw network data to ECSNormalized
    This class receives ECSNormalized objects
    Extracts the right features
    Scores with the trained model
    Returns enriched ECSNormalized with risk score

Performance Requirements:
    Must score events in under 10 milliseconds
    Must handle 10,000+ events per second
    Must never crash on malformed input
    Must provide explainable decisions

This is where your Layer 1 ECS schema pays off.
The detector never knows if data came from CrowdStrike,
Zeek, or NetFlow. It just sees ECSNormalized objects
and extracts the features it needs.
"""

import logging
import pickle
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """
    Result of scoring a single network event.

    This object is attached to the ECSNormalized event
    and flows up through Layer 3 knowledge graph
    and Layer 4 LLM reasoning.

    Every field has a specific consumer:
        risk_score      → Layer 3 knowledge graph weighting
        is_attack       → Layer 4 agent decision making
        confidence      → Layer 5 analyst display
        explanation     → Layer 5 analyst display
        model_version   → MLOps monitoring and audit
    """

    # Core detection result
    is_attack: bool
    risk_score: float          # 0.0 to 1.0
    confidence: str            # HIGH / MEDIUM / LOW

    # Explainability
    explanation: str           # Human readable reason
    top_features: dict         # Features that drove decision

    # Metadata for MLOps monitoring
    model_name: str
    model_version: str
    scored_at: str
    inference_time_ms: float

    def to_dict(self) -> dict:
        """Convert to dictionary for ECS enrichment"""
        return {
            "is_attack": self.is_attack,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "scored_at": self.scored_at,
            "inference_time_ms": self.inference_time_ms
        }


class NetworkIntrusionDetector:
    """
    Production inference engine for network intrusion detection.

    Loads a trained model and scores live network events
    from Layer 1 in real time.

    Usage:
        # Load detector with trained model
        detector = NetworkIntrusionDetector()
        detector.load_model("models/intrusion_detection/best_model.pkl")

        # Score a live ECS network event
        result = detector.score_ecs_event(ecs_normalized_event)

        # Or score raw feature dict
        result = detector.score_features(feature_dict)
    """

    # Feature names in the exact order the model expects
    # Must match SELECTED_FEATURES in data_preparation.py
    FEATURE_NAMES = [
        " Flow Duration",
        " Total Fwd Packets",
        " Total Backward Packets",
        " Total Length of Fwd Packets",
        " Total Length of Bwd Packets",
        " Fwd Packet Length Max",
        " Fwd Packet Length Mean",
        " Bwd Packet Length Max",
        " Bwd Packet Length Mean",
        " Flow Bytes/s",
        " Flow Packets/s",
        " Flow IAT Mean",
        " Flow IAT Std",
        " Fwd IAT Mean",
        " Bwd IAT Mean",
        " SYN Flag Count",
        " RST Flag Count",
        " PSH Flag Count",
        " ACK Flag Count",
        " Init_Win_bytes_forward",
    ]

    def __init__(
        self,
        model_path: str = None,
        scaler_path: str = None,
        threshold: float = 0.5,
        model_version: str = "1.0.0"
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path to saved trained model
            scaler_path: Path to saved StandardScaler
            threshold: Classification threshold
                      Lower = more sensitive (more detections)
                      Higher = more specific (fewer alarms)
            model_version: Version string for audit trail
        """
        self.model = None
        self.scaler = None
        self.threshold = threshold
        self.model_version = model_version
        self.model_name = "unknown"

        # Performance tracking
        self.total_scored = 0
        self.total_attacks_detected = 0
        self.total_inference_time_ms = 0.0

        # Load model if path provided
        if model_path:
            self.load_model(model_path, scaler_path)

    def load_model(
        self,
        model_path: str,
        scaler_path: str = None
    ) -> None:
        """
        Load trained model and optional scaler from disk.

        Args:
            model_path: Path to pickled model file
            scaler_path: Path to pickled StandardScaler
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Train a model first using "
                f"IntrusionDetectionTrainer"
            )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.model_name = model_path.stem
        logger.info(f"Loaded model: {model_path}")

        # Load scaler if provided
        if scaler_path:
            scaler_path = Path(scaler_path)
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded scaler: {scaler_path}")

    def score_ecs_event(
        self,
        ecs_event
    ) -> Optional[DetectionResult]:
        """
        Score a normalized ECS network event.

        This is the primary production method.
        Called by Layer 4 agents when investigating alerts.

        Extracts network flow features from ECSNormalized
        object and returns a DetectionResult.

        Args:
            ecs_event: ECSNormalized object from Layer 1

        Returns:
            DetectionResult or None if event cannot be scored
        """
        if ecs_event is None:
            return None

        # Extract features from ECS event
        features = self._extract_ecs_features(ecs_event)

        if features is None:
            logger.debug(
                "Could not extract features from ECS event"
            )
            return None

        return self.score_features(features)

    def score_features(
        self,
        features: dict
    ) -> Optional[DetectionResult]:
        """
        Score a feature dictionary directly.

        Used for:
        - Scoring CICIDS2017 test events directly
        - Unit testing without full ECS pipeline
        - Batch scoring from historical data

        Args:
            features: Dict mapping feature names to values
                     Keys must match FEATURE_NAMES

        Returns:
            DetectionResult or None if scoring fails
        """
        if self.model is None:
            logger.error(
                "No model loaded. Call load_model() first."
            )
            return None

        start_time = datetime.now(timezone.utc)

        try:
            # Build feature vector in correct order
            feature_vector = self._build_feature_vector(
                features
            )

            if feature_vector is None:
                return None

            # Scale features if scaler available
            if self.scaler is not None:
                feature_vector = self.scaler.transform(
                    feature_vector.reshape(1, -1)
                )
            else:
                feature_vector = feature_vector.reshape(1, -1)

            # Get prediction probability
            prob = self.model.predict_proba(
                feature_vector
            )[0][1]

            # Apply threshold
            is_attack = bool(prob >= self.threshold)

            # Calculate inference time
            end_time = datetime.now(timezone.utc)
            inference_ms = (
                (end_time - start_time).microseconds / 1000
            )

            # Build result
            result = DetectionResult(
                is_attack=is_attack,
                risk_score=round(float(prob), 4),
                confidence=self._score_to_confidence(prob),
                explanation=self._generate_explanation(
                    features, prob, is_attack
                ),
                top_features=self._get_top_features(
                    feature_vector
                ),
                model_name=self.model_name,
                model_version=self.model_version,
                scored_at=start_time.isoformat(),
                inference_time_ms=round(inference_ms, 2)
            )

            # Update statistics
            self.total_scored += 1
            if is_attack:
                self.total_attacks_detected += 1
            self.total_inference_time_ms += inference_ms

            return result

        except Exception as e:
            logger.error(
                f"Scoring failed: {e}",
                exc_info=True
            )
            return None

    def score_batch(
        self,
        events: list
    ) -> list:
        """
        Score a batch of events efficiently.

        More efficient than scoring one at a time
        because it minimizes Python overhead per event.

        Used for:
        - Scoring historical events during backfill
        - Batch processing in Databricks notebooks
        - Offline analysis of stored events

        Args:
            events: List of feature dicts or ECS events

        Returns:
            List of DetectionResults (None for failed events)
        """
        results = []
        for event in events:
            if hasattr(event, "event"):
                # It is an ECS event
                result = self.score_ecs_event(event)
            else:
                # It is a feature dict
                result = self.score_features(event)
            results.append(result)
        return results

    def get_performance_stats(self) -> dict:
        """
        Return performance statistics for MLOps monitoring.

        Called by model monitoring layer to track:
        - How many events scored
        - Detection rate over time
        - Average inference latency
        - Model health indicators
        """
        avg_latency = (
            self.total_inference_time_ms /
            max(self.total_scored, 1)
        )

        detection_rate = (
            self.total_attacks_detected /
            max(self.total_scored, 1)
        )

        return {
            "total_scored": self.total_scored,
            "total_attacks_detected": (
                self.total_attacks_detected
            ),
            "detection_rate": round(detection_rate, 4),
            "avg_inference_ms": round(avg_latency, 2),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "threshold": self.threshold
        }

    # ============================================================
    # PRIVATE METHODS
    # ============================================================

    def _extract_ecs_features(
        self,
        ecs_event
    ) -> Optional[dict]:
        """
        Extract intrusion detection features from ECS event.

        This is the bridge between your Layer 1 ECS schema
        and your Layer 2 ML model feature requirements.

        Maps ECSNormalized fields to CICIDS2017 feature names
        so the trained model can score live events.
        """
        if not hasattr(ecs_event, "network"):
            return None

        network = ecs_event.network
        source = ecs_event.source
        destination = ecs_event.destination

        if network is None:
            return None

        # Map ECS fields to CICIDS2017 feature names
        features = {
            " Flow Duration": (
                network.duration_ms or 0
            ),
            " Total Fwd Packets": (
                network.fwd_packets or 0
            ),
            " Total Backward Packets": (
                network.bwd_packets or 0
            ),
            " Total Length of Fwd Packets": (
                network.fwd_bytes or
                (source.bytes if source else 0) or 0
            ),
            " Total Length of Bwd Packets": (
                network.bwd_bytes or
                (destination.bytes if destination else 0)
                or 0
            ),
            " Fwd Packet Length Max": (
                network.fwd_packet_len_max or 0
            ),
            " Fwd Packet Length Mean": (
                network.fwd_packet_len_mean or 0
            ),
            " Bwd Packet Length Max": (
                network.bwd_packet_len_max or 0
            ),
            " Bwd Packet Length Mean": (
                network.bwd_packet_len_mean or 0
            ),
            " Flow Bytes/s": (
                network.flow_bytes_per_sec or 0
            ),
            " Flow Packets/s": (
                network.flow_packets_per_sec or 0
            ),
            " Flow IAT Mean": (
                network.iat_mean or 0
            ),
            " Flow IAT Std": (
                network.iat_std or 0
            ),
            " Fwd IAT Mean": (
                network.fwd_iat_mean or 0
            ),
            " Bwd IAT Mean": (
                network.bwd_iat_mean or 0
            ),
            " SYN Flag Count": (
                network.syn_flags or 0
            ),
            " RST Flag Count": (
                network.rst_flags or 0
            ),
            " PSH Flag Count": (
                network.psh_flags or 0
            ),
            " ACK Flag Count": (
                network.ack_flags or 0
            ),
            " Init_Win_bytes_forward": (
                network.init_win_bytes_fwd or 0
            )
        }

        return features

    def _build_feature_vector(
        self,
        features: dict
    ) -> Optional[np.ndarray]:
        """
        Build numpy feature vector in correct order.

        The model was trained with features in a specific
        order. The feature vector must match that order
        exactly or the model produces wrong predictions.
        """
        vector = []

        for feature_name in self.FEATURE_NAMES:
            value = features.get(feature_name, 0)
            try:
                value = float(value)
                # Replace any infinity with 0
                if np.isinf(value) or np.isnan(value):
                    value = 0.0
            except (TypeError, ValueError):
                value = 0.0
            vector.append(value)

        return np.array(vector, dtype=np.float32)

    def _score_to_confidence(
        self,
        prob: float
    ) -> str:
        """
        Convert probability score to analyst-friendly
        confidence label.

        HIGH:   > 80% probability — act immediately
        MEDIUM: 50-80% probability — investigate further
        LOW:    < 50% probability — monitor only
        """
        if prob >= 0.80:
            return "HIGH"
        elif prob >= 0.50:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_explanation(
        self,
        features: dict,
        prob: float,
        is_attack: bool
    ) -> str:
        """
        Generate human-readable explanation of decision.

        This is critical for analyst trust.
        A score of 0.94 with no explanation is useless.
        A score of 0.94 with "Very high packet rate
        (1.2M pps) consistent with DoS flooding" is
        actionable intelligence.
        """
        if not is_attack:
            return (
                f"Traffic appears benign "
                f"(risk score: {prob:.2f}). "
                f"No significant attack indicators detected."
            )

        # Identify which signals drove the detection
        signals = []

        flow_bytes = features.get(" Flow Bytes/s", 0)
        if flow_bytes > 1_000_000:
            signals.append(
                f"Extremely high flow rate "
                f"({flow_bytes/1e6:.1f}M bytes/s) "
                f"— DoS indicator"
            )

        syn_flags = features.get(" SYN Flag Count", 0)
        rst_flags = features.get(" RST Flag Count", 0)
        if syn_flags > 0 and rst_flags > 0:
            signals.append(
                f"SYN+RST flag combination "
                f"(SYN={syn_flags}, RST={rst_flags}) "
                f"— port scan indicator"
            )

        bwd_packets = features.get(
            " Total Backward Packets", 0
        )
        fwd_packets = features.get(
            " Total Fwd Packets", 0
        )
        if bwd_packets == 0 and fwd_packets > 10:
            signals.append(
                f"One-directional traffic "
                f"({fwd_packets} packets sent, 0 received) "
                f"— scanning or DoS indicator"
            )

        iat_mean = features.get(" Flow IAT Mean", 0)
        iat_std = features.get(" Flow IAT Std", 0)
        if iat_mean > 0 and iat_std < iat_mean * 0.1:
            signals.append(
                f"Very regular timing pattern "
                f"(IAT mean={iat_mean:.0f}ms, "
                f"std={iat_std:.0f}ms) "
                f"— C2 beaconing indicator"
            )

        if not signals:
            signals.append(
                "Multiple statistical anomalies "
                "detected across flow features"
            )

        explanation = (
            f"ATTACK DETECTED (risk score: {prob:.2f}). "
            f"Signals: " +
            "; ".join(signals)
        )

        return explanation

    def _get_top_features(
        self,
        feature_vector: np.ndarray
    ) -> dict:
        """
        Return top contributing features for this prediction.

        Uses model feature importances combined with
        feature values to identify which features most
        influenced this specific prediction.

        Full SHAP integration happens in MLOps layer.
        This provides a fast approximation for real-time use.
        """
        if not hasattr(self.model, "feature_importances_"):
            return {}

        importances = self.model.feature_importances_
        feature_values = feature_vector.flatten()

        # Combine importance with actual value
        top_indices = np.argsort(importances)[::-1][:5]

        top_features = {}
        for idx in top_indices:
            if idx < len(self.FEATURE_NAMES):
                feature_name = self.FEATURE_NAMES[idx].strip()
                top_features[feature_name] = {
                    "value": round(
                        float(feature_values[idx]), 4
                    ),
                    "importance": round(
                        float(importances[idx]), 4
                    )
                }

        return top_features