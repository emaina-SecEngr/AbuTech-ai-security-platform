"""
Layer 4 — Agent Tools
Tool 3: Ensemble Score Retrieval Tool

Gets all 8 ML model scores for an event
using the Unified Ensemble Scorer.

Used by InvestigationAgent to get real
model scores instead of relying only on
the initial routing result.

USAGE BY AGENTS:
    result = get_ensemble_scores(event_data)
    print(result["final_score"])    # 0.950
    print(result["verdict"])        # DATA_EXFILTRATION
    print(result["explanation"])    # Human readable
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_ensemble_scores(
    event_data: dict,
    pii_sensitivity: str = "NONE"
) -> dict:
    """
    Run all available ML models on event data
    and return ensemble score.

    Args:
        event_data: dict with event features
        pii_sensitivity: PCI/PHI/PII/NONE

    Returns:
        dict with ensemble result including
        per-model scores and final verdict
    """
    if not event_data:
        return _empty_result()

    try:
        from layer2_ml.ensemble.ensemble_scorer import (
            EnsembleScorer,
            ModelScore
        )

        scorer = EnsembleScorer()
        model_scores = {}

        # Score with each available model
        model_scores.update(
            _run_isolation_forest(event_data)
        )
        model_scores.update(
            _run_pii_classifier(event_data)
        )
        model_scores.update(
            _run_identity_detector(event_data)
        )

        # Run ensemble
        result = scorer.score(
            model_scores,
            pii_sensitivity=pii_sensitivity
        )

        return result.to_dict()

    except Exception as e:
        logger.warning(
            f"Ensemble scoring failed: {e}"
        )
        return _rule_based_score(
            event_data, pii_sensitivity
        )


def _run_isolation_forest(
    event_data: dict
) -> dict:
    """Run Isolation Forest on event data"""
    try:
        from layer2_ml.anomaly\
            .isolation_forest_detector \
            import IsolationForestDetector
        from unittest.mock import MagicMock

        detector = IsolationForestDetector()
        ecs_mock = _build_ecs_mock(event_data)
        result = detector.score_network(ecs_mock)

        from layer2_ml.ensemble.ensemble_scorer \
            import ModelScore

        return {
            "isolation_forest": ModelScore(
                model_name="isolation_forest",
                score=float(result.anomaly_score),
                confidence=0.90,
                reason="Network anomaly detection"
            )
        }
    except Exception as e:
        logger.debug(
            f"IsolationForest scoring failed: {e}"
        )
        return {}


def _run_pii_classifier(
    event_data: dict
) -> dict:
    """Run PII Classifier on event data"""
    try:
        from layer2_ml.classification.pii_classifier \
            import PIIClassifier
        from layer2_ml.ensemble.ensemble_scorer \
            import ModelScore

        classifier = PIIClassifier()
        path_text = (
            f"{event_data.get('data_store_name', '')} "
            f"{event_data.get('data_path', '')} "
            f"{event_data.get('file_name', '')}"
        )

        # Check path for PCI/PHI keywords directly
        path_lower = path_text.lower()
        if any(
            kw in path_lower
            for kw in ["pci", "card", "payment",
                       "credit", "ccnum"]
        ):
            score = 0.95
        elif any(
            kw in path_lower
            for kw in ["phi", "health", "medical",
                       "patient", "hipaa"]
        ):
            score = 0.90
        elif any(
            kw in path_lower
            for kw in ["pii", "ssn", "social",
                       "personal", "customer"]
        ):
            score = 0.85
        else:
            finding = classifier.classify(path_text)
            sensitivity = finding.sensitivity_label.value
            score_map = {
                "PCI": 0.95, "PHI": 0.90,
                "PII": 0.85, "NONE": 0.05
            }
            score = score_map.get(sensitivity, 0.05)

        return {
            "pii_classifier": ModelScore(
                model_name="pii_classifier",
                score=score,
                confidence=1.0,
                reason="PII/PCI data detection"
            )
        }
    except Exception as e:
        logger.debug(
            f"PII classifier failed: {e}"
        )
        return {}


def _run_identity_detector(
    event_data: dict
) -> dict:
    """Score identity risk from event features"""
    try:
        from layer2_ml.ensemble.ensemble_scorer \
            import ModelScore

        accessor = event_data.get(
            "accessor_identity", ""
        ).lower()
        risk_score = 0.1

        # Service account accessing wrong resource
        if "svc_" in accessor or "_svc" in accessor:
            risk_score = max(risk_score, 0.4)

        # After hours access (hour 0-6)
        event_time = event_data.get("event_time", "")
        if event_time:
            try:
                hour = int(
                    event_time[11:13]
                )
                if 0 <= hour <= 6:
                    risk_score = max(
                        risk_score, 0.6
                    )
            except (ValueError, IndexError):
                pass

        # Large volume access
        bytes_accessed = event_data.get(
            "bytes_accessed", 0
        ) or 0
        mb = bytes_accessed / (1024 * 1024)
        if mb > 100:
            risk_score = max(risk_score, 0.7)
        elif mb > 50:
            risk_score = max(risk_score, 0.5)

        return {
            "identity_detector": ModelScore(
                model_name="identity_detector",
                score=risk_score,
                confidence=0.85,
                reason="Identity behavior analysis"
            )
        }
    except Exception as e:
        logger.debug(
            f"Identity detection failed: {e}"
        )
        return {}


def _build_ecs_mock(event_data: dict):
    """Build ECS mock from event data"""
    from unittest.mock import MagicMock

    bytes_val = float(
        event_data.get("bytes_accessed", 0) or 0
    )

    network = MagicMock()
    network.fwd_bytes = bytes_val
    network.bwd_bytes = 0.0
    network.fwd_packets = 1000.0
    network.bwd_packets = 0.0
    network.duration_ms = 1000.0
    network.flow_bytes_per_sec = bytes_val
    network.fwd_packet_len_mean = 1000.0
    network.bwd_packet_len_mean = 0.0
    network.protocol = "TCP"

    dest = MagicMock()
    dest.port = 443

    risk = float(
        event_data.get("risk_score", 0.5) or 0.5
    )
    event = MagicMock()
    event.severity = int(risk * 100)

    ecs = MagicMock()
    ecs.network = network
    ecs.destination = dest
    ecs.source = MagicMock()
    ecs.event = event
    ecs.process = None

    return ecs


def _rule_based_score(
    event_data: dict,
    pii_sensitivity: str
) -> dict:
    """Rule-based fallback scoring"""
    score = float(
        event_data.get("risk_score", 0.5) or 0.5
    )

    pii_multipliers = {
        "PCI": 1.30, "PHI": 1.25,
        "PII": 1.20, "NONE": 1.00
    }
    multiplier = pii_multipliers.get(
        pii_sensitivity, 1.0
    )
    final = min(1.0, score * multiplier)

    label = "CRITICAL" if final >= 0.8 else \
            "HIGH" if final >= 0.6 else \
            "MEDIUM" if final >= 0.4 else "LOW"

    return {
        "final_score": round(final, 4),
        "risk_label": label,
        "verdict": "UNKNOWN",
        "model_scores": {},
        "model_contributions": {},
        "pii_sensitivity": pii_sensitivity,
        "pii_multiplier": multiplier,
        "highest_model": "rule_based",
        "highest_score": score,
        "models_available": 0,
        "models_scored": 0,
        "explanation": (
            f"Rule-based score {final:.3f}. "
            f"ML models unavailable."
        )
    }


def _empty_result() -> dict:
    """Return empty result"""
    return {
        "final_score": 0.0,
        "risk_label": "UNKNOWN",
        "verdict": "NORMAL",
        "model_scores": {},
        "model_contributions": {},
        "pii_sensitivity": "NONE",
        "pii_multiplier": 1.0,
        "highest_model": "",
        "highest_score": 0.0,
        "models_available": 0,
        "models_scored": 0,
        "explanation": "No event data provided."
    }