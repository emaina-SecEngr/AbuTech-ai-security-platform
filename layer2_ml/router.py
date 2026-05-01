"""
Layer 2 — ML Processing Engine
Event Router

Routes normalized ECS events to correct ML models.
"""

import logging
from dataclasses import dataclass
from dataclasses import field
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
from layer2_ml.nlp.dns_classifier import (
    DNSClassifier,
    DNSDetectionResult
)

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Complete routing result for a single ECS event"""

    event_id: str
    event_category: str
    event_dataset: str
    host: str
    timestamp: str

    intrusion_result: Optional[DetectionResult] = None
    malware_result: Optional[MalwareDetectionResult] = None
    dns_result: Optional[DNSDetectionResult] = None

    overall_risk_score: float = 0.0
    overall_verdict: str = "UNKNOWN"
    primary_model: str = "none"

    routed_to: list = field(default_factory=list)
    routing_time_ms: float = 0.0

    def is_threat(self) -> bool:
        if self.intrusion_result:
            if self.intrusion_result.is_attack:
                return True
        if self.malware_result:
            if self.malware_result.is_malicious:
                return True
        if self.dns_result:
            if self.dns_result.is_dga:
                return True
        return False

    def get_highest_risk(self) -> float:
        scores = []
        if self.intrusion_result:
            scores.append(self.intrusion_result.risk_score)
        if self.malware_result:
            scores.append(self.malware_result.risk_score)
        if self.dns_result:
            scores.append(self.dns_result.risk_score)
        return max(scores) if scores else 0.0

    def to_dict(self) -> dict:
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
            ),
            "dns_result": (
                self.dns_result.to_dict()
                if self.dns_result else None
            )
        }


class Layer2Router:
    """
    Routes normalized ECS events to correct ML models.

    Routing table:
        network → NetworkIntrusionDetector
        dns     → NetworkIntrusionDetector + DNSClassifier
        process → MalwareClassifier
    """

    INTRUSION_CATEGORIES = {"network", "dns"}
    MALWARE_CATEGORIES = {"process"}

    def __init__(self):
        self.intrusion_detector = NetworkIntrusionDetector(
            threshold=0.5,
            model_version="1.0.0"
        )
        self.malware_classifier = MalwareClassifier(
            threshold=0.5,
            model_version="1.0.0"
        )
        self.dns_classifier = DNSClassifier()
        
        self.total_routed = 0
        self.routing_counts = {}
        self.threat_count = 0

        logger.info("Layer2Router initialized")

    def load_models(
        self,
        intrusion_model_path: str = None,
        intrusion_scaler_path: str = None,
        malware_model_path: str = None,
        malware_scaler_path: str = None,
        dns_model_path: str = None,
        dns_scaler_path: str = None
    ) -> dict:
        """Load trained models for production use"""
        loaded = {}

        if intrusion_model_path:
            try:
                self.intrusion_detector.load_model(
                    intrusion_model_path,
                    intrusion_scaler_path
                )
                loaded["intrusion_detector"] = "loaded"
            except FileNotFoundError as e:
                loaded["intrusion_detector"] = f"failed: {e}"

        if malware_model_path:
            try:
                self.malware_classifier.load_model(
                    malware_model_path,
                    malware_scaler_path
                )
                loaded["malware_classifier"] = "loaded"
            except FileNotFoundError as e:
                loaded["malware_classifier"] = f"failed: {e}"

        if dns_model_path:
            try:
                self.dns_classifier.load_model(
                    dns_model_path,
                    dns_scaler_path
                )
                loaded["dns_classifier"] = "loaded"
            except FileNotFoundError as e:
                loaded["dns_classifier"] = f"failed: {e}"

        return loaded

    def route(
        self,
        ecs_event
    ) -> Optional[RoutingResult]:
        """Route a single ECS event to correct models"""

        if ecs_event is None:
            return None

        start_time = datetime.now(timezone.utc)

        event_id = (
            ecs_event.event.id
            if hasattr(ecs_event.event, "id")
            else "unknown"
        )
        category = ecs_event.event.category or "unknown"
        dataset = ecs_event.event.dataset or "unknown"
        host = (
            ecs_event.host.hostname or "unknown"
            if ecs_event.host else "unknown"
        )

        result = RoutingResult(
            event_id=event_id,
            event_category=category,
            event_dataset=dataset,
            host=host,
            timestamp=ecs_event.timestamp
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

        # ---- ROUTE TO DNS CLASSIFIER ----
        if category == "dns":
            dns_result = (
                self.dns_classifier.score_ecs_event(
                    ecs_event
                )
            )
            if dns_result:
                result.dns_result = dns_result
                result.routed_to.append("DNSClassifier")

        # ---- ROUTE TO MALWARE CLASSIFIER ----
        if category in self.MALWARE_CATEGORIES:
            malware_result = (
                self.malware_classifier.score_ecs_event(
                    ecs_event
                )
            )
            if malware_result:
                result.malware_result = malware_result
                result.routed_to.append("MalwareClassifier")

        # ---- HANDLE UNROUTED EVENTS ----
        if not result.routed_to:
            result.routed_to.append(f"UNROUTED:{category}")

        # ---- CALCULATE OVERALL RISK ----
        result.overall_risk_score = result.get_highest_risk()
        result.overall_verdict = (
            self._calculate_verdict(result)
        )
        result.primary_model = (
            self._identify_primary_model(result)
        )

        # ---- ROUTING TIME ----
        end_time = datetime.now(timezone.utc)
        result.routing_time_ms = round(
            (end_time - start_time).microseconds / 1000, 2
        )

        # ---- STATISTICS ----
        self.total_routed += 1
        self.routing_counts[category] = (
            self.routing_counts.get(category, 0) + 1
        )
        if result.is_threat():
            self.threat_count += 1

        return result

    def route_batch(self, ecs_events: list) -> list:
        """Route a batch of events efficiently"""
        return [
            r for r in
            (self.route(e) for e in ecs_events)
            if r is not None
        ]

    def get_statistics(self) -> dict:
        """Return routing statistics for MLOps monitoring"""
        threat_rate = (
            self.threat_count / max(self.total_routed, 1)
        )
        return {
            "total_routed": self.total_routed,
            "threat_count": self.threat_count,
            "threat_rate": round(threat_rate, 4),
            "routing_counts": self.routing_counts,
            "intrusion_detector_stats": (
                self.intrusion_detector.get_performance_stats()
            ),
            "malware_classifier_stats": (
                self.malware_classifier.get_performance_stats()
            ),
            "dns_classifier_stats": (
                self.dns_classifier.get_performance_stats()
            )
        }

    def get_model_status(self) -> dict:
        """Return status of each registered model"""
        return {
            "intrusion_detector": {
                "loaded": (
                    self.intrusion_detector.model is not None
                ),
                "categories": list(self.INTRUSION_CATEGORIES),
                "model_name": self.intrusion_detector.model_name
            },
            "malware_classifier": {
                "loaded": (
                    self.malware_classifier.model is not None
                ),
                "categories": list(self.MALWARE_CATEGORIES),
                "model_name": self.malware_classifier.model_name
            },
            "dns_classifier": {
                "loaded": (
                    self.dns_classifier.model is not None
                ),
                "categories": ["dns"],
                "model_name": self.dns_classifier.model_name
            }
        }

    def _calculate_verdict(self, result: RoutingResult) -> str:
        if result.dns_result and result.dns_result.is_dga:
            return "DGA_DOMAIN"
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
        self, result: RoutingResult
    ) -> str:
        if not result.routed_to:
            return "none"

        scores = {
            "NetworkIntrusionDetector": (
                result.intrusion_result.risk_score
                if result.intrusion_result else 0
            ),
            "MalwareClassifier": (
                result.malware_result.risk_score
                if result.malware_result else 0
            ),
            "DNSClassifier": (
                result.dns_result.risk_score
                if result.dns_result else 0
            )
        }

        return max(scores, key=scores.get)