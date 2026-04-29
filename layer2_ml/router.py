"""
Layer 2 — ML Processing Engine
Event Router

The Layer2Router is the traffic controller
between Layer 1 normalization and Layer 2
ML models.

Every normalized ECS event from Layer 1
flows through this router. The router reads
the event category and routes to the
correct ML model automatically.

This is what makes your pipeline fully automated.
No manual routing decisions. No if-else chains
scattered through your code. One central router
that knows which model handles which event type.

Routing Table:
    network        → NetworkIntrusionDetector
    dns            → NetworkIntrusionDetector
                     DNS uses same statistical approach
                     DGA detection, C2 beaconing patterns
    process        → MalwareClassifier
                     Behavioral analysis of process execution
    authentication → UEBAModel (future)
    email          → PhishingDetector (future)
    *              → GNNModel sees ALL events (future)

Architecture Note:
    The router runs synchronously for every event.
    Fast models (rule-based) run in milliseconds.
    Slower models (GNN) run asynchronously in batch.
    Layer 4 agents run only for high-priority events.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from typing import Optional

from layer2_ml.intrusion_detection.detector import (
    NetworkIntrusionDetector,
    DetectionResult
)
from layer2_ml.malware_classifier.malware_detector import (
    MalwareClassifier,
    MalwareDetectionResult
)

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """
    Complete routing result for a single ECS event.

    Contains results from ALL models that scored
    this event plus routing metadata.

    Consumed by:
        Layer 3 knowledge graph — uses all scores
        Layer 4 LLM agents — uses highest risk score
        Layer 5 dashboard — displays all results
    """

    # Event identification
    event_id: str
    event_category: str
    event_dataset: str
    host: str
    timestamp: str

    # Model results
    # None if model did not score this event
    intrusion_result: Optional[DetectionResult] = None
    malware_result: Optional[MalwareDetectionResult] = None

    # Overall risk assessment
    # Highest score across all models
    overall_risk_score: float = 0.0
    overall_verdict: str = "UNKNOWN"
    primary_model: str = "none"

    # Routing metadata
    routed_to: list = None
    routing_time_ms: float = 0.0

    def __post_init__(self):
        if self.routed_to is None:
            self.routed_to = []

    def is_threat(self) -> bool:
        """
        Returns True if any model flagged this
        event as malicious or attack.
        Used by Layer 4 agents to decide whether
        to trigger investigation workflow.
        """
        if self.intrusion_result:
            if self.intrusion_result.is_attack:
                return True
        if self.malware_result:
            if self.malware_result.is_malicious:
                return True
        return False

    def get_highest_risk(self) -> float:
        """Return highest risk score across all models"""
        scores = []
        if self.intrusion_result:
            scores.append(self.intrusion_result.risk_score)
        if self.malware_result:
            scores.append(self.malware_result.risk_score)
        return max(scores) if scores else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            "event_id": self.event_id,
            "event_category": self.event_category,
            "host": self.host,
            "timestamp": self.timestamp,
            "overall_risk_score": self.overall_risk_score,
            "overall_verdict": self.overall_verdict,
            "primary_model": self.primary_model,
            "routed_to": self.routed_to,
            "is_threat": self.is_threat(),
            "intrusion_result": (
                self.intrusion_result.to_dict()
                if self.intrusion_result else None
            ),
            "malware_result": (
                self.malware_result.to_dict()
                if self.malware_result else None
            )
        }


class Layer2Router:
    """
    Routes normalized ECS events to correct ML models.

    Central traffic controller for Layer 2.
    Receives every ECSNormalized event from Layer 1
    and routes to appropriate detection models.

    Usage:
        router = Layer2Router()
        router.load_models(
            intrusion_model_path="models/.../best_model.pkl",
            malware_model_path="models/.../best_model.pkl"
        )
        result = router.route(ecs_event)
    """

    # Event categories handled by each model
    INTRUSION_CATEGORIES = {"network", "dns"}
    MALWARE_CATEGORIES = {"process"}

    def __init__(self):
        """Initialize router with empty model slots"""

        self.intrusion_detector = NetworkIntrusionDetector(
            threshold=0.5,
            model_version="1.0.0"
        )

        self.malware_classifier = MalwareClassifier(
            threshold=0.5,
            model_version="1.0.0"
        )

        # Track routing statistics
        self.total_routed = 0
        self.routing_counts = {}
        self.threat_count = 0

        logger.info("Layer2Router initialized")

    def load_models(
        self,
        intrusion_model_path: str = None,
        intrusion_scaler_path: str = None,
        malware_model_path: str = None,
        malware_scaler_path: str = None
    ) -> dict:
        """
        Load trained models for production use.

        Args:
            intrusion_model_path: Path to intrusion
                                  detection model
            intrusion_scaler_path: Path to intrusion
                                   scaler
            malware_model_path: Path to malware
                                classifier model
            malware_scaler_path: Path to malware scaler

        Returns:
            Dictionary showing which models loaded
        """
        loaded = {}

        if intrusion_model_path:
            try:
                self.intrusion_detector.load_model(
                    intrusion_model_path,
                    intrusion_scaler_path
                )
                loaded["intrusion_detector"] = "loaded"
                logger.info(
                    "Intrusion detector loaded"
                )
            except FileNotFoundError as e:
                loaded["intrusion_detector"] = (
                    f"failed: {e}"
                )
                logger.warning(
                    f"Intrusion detector not loaded: {e}"
                )

        if malware_model_path:
            try:
                self.malware_classifier.load_model(
                    malware_model_path,
                    malware_scaler_path
                )
                loaded["malware_classifier"] = "loaded"
                logger.info(
                    "Malware classifier loaded"
                )
            except FileNotFoundError as e:
                loaded["malware_classifier"] = (
                    f"failed: {e}"
                )
                logger.warning(
                    f"Malware classifier not loaded: {e}"
                )

        return loaded

    def route(
        self,
        ecs_event
    ) -> Optional[RoutingResult]:
        """
        Route a single ECS event to correct models.

        This is the primary method called for every
        event flowing through Layer 2.

        Args:
            ecs_event: ECSNormalized from Layer 1

        Returns:
            RoutingResult with all model scores
            None if event cannot be routed
        """
        if ecs_event is None:
            return None

        start_time = datetime.now(timezone.utc)

        # Extract event metadata
        event_id = (
            ecs_event.event.id
            if hasattr(ecs_event.event, "id")
            else "unknown"
        )
        category = (
            ecs_event.event.category or "unknown"
        )
        dataset = (
            ecs_event.event.dataset or "unknown"
        )
        host = (
            ecs_event.host.hostname or "unknown"
            if ecs_event.host else "unknown"
        )
        timestamp = ecs_event.timestamp

        # Initialize result
        result = RoutingResult(
            event_id=event_id,
            event_category=category,
            event_dataset=dataset,
            host=host,
            timestamp=timestamp
        )

        # ---- ROUTE TO INTRUSION DETECTOR ----
        if category in self.INTRUSION_CATEGORIES:
            intrusion_result = (
                self.intrusion_detector.score_ecs_event(
                    ecs_event
                )
            )
            if intrusion_result:
                result.intrusion_result = intrusion_result
                result.routed_to.append(
                    "NetworkIntrusionDetector"
                )
                logger.debug(
                    f"Intrusion detector scored "
                    f"{category} event: "
                    f"{intrusion_result.risk_score}"
                )

        # ---- ROUTE TO MALWARE CLASSIFIER ----
        if category in self.MALWARE_CATEGORIES:
            malware_result = (
                self.malware_classifier.score_ecs_event(
                    ecs_event
                )
            )
            if malware_result:
                result.malware_result = malware_result
                result.routed_to.append(
                    "MalwareClassifier"
                )
                logger.debug(
                    f"Malware classifier scored "
                    f"{category} event: "
                    f"{malware_result.risk_score}"
                )

        # ---- HANDLE UNROUTED EVENTS ----
        if not result.routed_to:
            result.routed_to.append(
                f"UNROUTED:{category}"
            )
            logger.debug(
                f"No model registered for: {category}"
            )

        # ---- CALCULATE OVERALL RISK ----
        result.overall_risk_score = (
            result.get_highest_risk()
        )

        result.overall_verdict = (
            self._calculate_verdict(result)
        )

        result.primary_model = (
            self._identify_primary_model(result)
        )

        # ---- CALCULATE ROUTING TIME ----
        end_time = datetime.now(timezone.utc)
        result.routing_time_ms = round(
            (end_time - start_time).microseconds / 1000,
            2
        )

        # ---- UPDATE STATISTICS ----
        self.total_routed += 1
        self.routing_counts[category] = (
            self.routing_counts.get(category, 0) + 1
        )
        if result.is_threat():
            self.threat_count += 1

        return result

    def route_batch(
        self,
        ecs_events: list
    ) -> list:
        """
        Route a batch of events efficiently.

        Used for processing historical events
        and Databricks batch scoring jobs.

        Args:
            ecs_events: List of ECSNormalized events

        Returns:
            List of RoutingResults
        """
        results = []
        for event in ecs_events:
            result = self.route(event)
            if result:
                results.append(result)
        return results

    def get_statistics(self) -> dict:
        """
        Return routing statistics for MLOps monitoring.

        Shows which event types are most common,
        what percentage are threats, and how
        the routing load is distributed across models.
        """
        threat_rate = (
            self.threat_count /
            max(self.total_routed, 1)
        )

        return {
            "total_routed": self.total_routed,
            "threat_count": self.threat_count,
            "threat_rate": round(threat_rate, 4),
            "routing_counts": self.routing_counts,
            "intrusion_detector_stats": (
                self.intrusion_detector
                .get_performance_stats()
            ),
            "malware_classifier_stats": (
                self.malware_classifier
                .get_performance_stats()
            )
        }

    def get_model_status(self) -> dict:
        """
        Return status of each registered model.
        Used by Layer 5 dashboard health check.
        """
        return {
            "intrusion_detector": {
                "loaded": (
                    self.intrusion_detector.model
                    is not None
                ),
                "categories": list(
                    self.INTRUSION_CATEGORIES
                ),
                "model_name": (
                    self.intrusion_detector.model_name
                )
            },
            "malware_classifier": {
                "loaded": (
                    self.malware_classifier.model
                    is not None
                ),
                "categories": list(
                    self.MALWARE_CATEGORIES
                ),
                "model_name": (
                    self.malware_classifier.model_name
                )
            }
        }

    # ============================================================
    # PRIVATE METHODS
    # ============================================================

    def _calculate_verdict(
        self,
        result: RoutingResult
    ) -> str:
        """
        Calculate overall verdict across all model scores.

        ATTACK:   Any model detected an attack
        MALWARE:  Malware classifier detected malware
        BENIGN:   All models scored as benign
        UNKNOWN:  No models scored this event
        """
        if result.malware_result:
            if result.malware_result.is_malicious:
                return "MALWARE"

        if result.intrusion_result:
            if result.intrusion_result.is_attack:
                return "ATTACK"

        if result.routed_to and not any(
            "UNROUTED" in r for r in result.routed_to
        ):
            return "BENIGN"

        return "UNKNOWN"

    def _identify_primary_model(
        self,
        result: RoutingResult
    ) -> str:
        """
        Identify which model produced the
        highest risk score for this event.
        Used by Layer 4 agents to know which
        model to query for more details.
        """
        if not result.routed_to:
            return "none"

        intrusion_score = (
            result.intrusion_result.risk_score
            if result.intrusion_result else 0
        )
        malware_score = (
            result.malware_result.risk_score
            if result.malware_result else 0
        )

        if intrusion_score >= malware_score:
            return "NetworkIntrusionDetector"
        else:
            return "MalwareClassifier"