"""
Layer 2 — ML Processing
Unified Ensemble Scorer
"""

import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Dict

logger = logging.getLogger(__name__)

MODEL_WEIGHTS = {
    "isolation_forest":   0.20,
    "autoencoder":        0.15,
    "random_forest":      0.20,
    "dns_classifier":     0.10,
    "identity_detector":  0.15,
    "pii_classifier":     0.10,
    "lstm_detector":      0.05,
    "gnn_detector":       0.05,
}

PII_MULTIPLIERS = {
    "PCI":  1.30,
    "PHI":  1.25,
    "PII":  1.20,
    "NONE": 1.00,
}

RISK_THRESHOLDS = {
    "CRITICAL": 0.80,
    "HIGH":     0.60,
    "MEDIUM":   0.40,
    "LOW":      0.20,
}


@dataclass
class ModelScore:
    model_name: str
    score: float
    confidence: float = 1.0
    available: bool = True
    reason: str = ""

    def weighted_score(self, weight: float) -> float:
        return self.score * self.confidence * weight


@dataclass
class EnsembleResult:
    final_score: float
    risk_label: str
    verdict: str
    model_scores: Dict[str, float] = field(default_factory=dict)
    model_contributions: Dict[str, float] = field(default_factory=dict)
    pii_sensitivity: str = "NONE"
    pii_multiplier: float = 1.0
    pre_pii_score: float = 0.0
    highest_model: str = ""
    highest_score: float = 0.0
    models_available: int = 0
    models_scored: int = 0
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "final_score": round(self.final_score, 4),
            "risk_label": self.risk_label,
            "verdict": self.verdict,
            "model_scores": {k: round(v, 4) for k, v in self.model_scores.items()},
            "model_contributions": {k: round(v, 4) for k, v in self.model_contributions.items()},
            "pii_sensitivity": self.pii_sensitivity,
            "pii_multiplier": self.pii_multiplier,
            "highest_model": self.highest_model,
            "highest_score": round(self.highest_score, 4),
            "models_available": self.models_available,
            "models_scored": self.models_scored,
            "explanation": self.explanation
        }


class EnsembleScorer:
    """
    Combines scores from all 8 ML models
    into a single weighted risk score.
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or MODEL_WEIGHTS.copy()
        self._validate_weights()
        logger.info(f"EnsembleScorer initialized with {len(self.weights)} models")

    def _validate_weights(self):
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Model weights must sum to 1.0. "
                f"Current sum: {total:.3f}. "
                f"SR 11-7 requires documented weights."
            )

    def score(
        self,
        model_scores: Dict[str, ModelScore],
        pii_sensitivity: str = "NONE"
    ) -> EnsembleResult:
        if not model_scores:
            return self._empty_result()

        # Step 1: Calculate weighted score
        weighted_sum = 0.0
        weight_sum = 0.0
        contributions = {}
        raw_scores = {}
        highest_model = ""
        highest_score = 0.0

        for model_name, weight in self.weights.items():
            model_score = model_scores.get(model_name)
            if model_score and model_score.available:
                contribution = (
                    model_score.score
                    * model_score.confidence
                    * weight
                )
                weighted_sum += contribution
                weight_sum += weight
                contributions[model_name] = round(contribution, 4)
                raw_scores[model_name] = round(model_score.score, 4)
                if model_score.score > highest_score:
                    highest_score = model_score.score
                    highest_model = model_name

        # Step 2: Normalize by available weights
        if weight_sum > 0:
            pre_pii_score = weighted_sum / weight_sum
        else:
            pre_pii_score = 0.0

        # Step 3: Apply PII multiplier
        pii_multiplier = PII_MULTIPLIERS.get(pii_sensitivity, 1.0)
        final_score = min(1.0, pre_pii_score * pii_multiplier)

        # Step 4: Determine risk label
        risk_label = self._score_to_label(final_score)

        # Step 5: Dominant signal rule
        # If any single model scores >= 0.85 and
        # final score >= 0.70 → elevate to CRITICAL
        # Prevents dilution of strong signals
        if highest_score >= 0.85 and final_score >= 0.70:
            final_score = max(final_score, 0.80)
            risk_label = "CRITICAL"

        # Step 6: Determine verdict
        verdict = self._determine_verdict(
            final_score, model_scores, pii_sensitivity
        )

        # Step 7: Build explanation
        explanation = self._build_explanation(
            final_score, risk_label, highest_model,
            highest_score, pii_sensitivity,
            pii_multiplier, len(contributions)
        )

        models_available = len(self.weights)
        models_scored = len(contributions)

        logger.info(
            f"Ensemble: score={final_score:.3f} "
            f"label={risk_label} "
            f"models={models_scored}/{models_available} "
            f"pii={pii_sensitivity}"
        )

        return EnsembleResult(
            final_score=round(final_score, 4),
            risk_label=risk_label,
            verdict=verdict,
            model_scores=raw_scores,
            model_contributions=contributions,
            pii_sensitivity=pii_sensitivity,
            pii_multiplier=pii_multiplier,
            pre_pii_score=round(pre_pii_score, 4),
            highest_model=highest_model,
            highest_score=round(highest_score, 4),
            models_available=models_available,
            models_scored=models_scored,
            explanation=explanation
        )

    def score_from_dict(
        self,
        scores_dict: Dict[str, float],
        pii_sensitivity: str = "NONE"
    ) -> EnsembleResult:
        model_scores = {
            name: ModelScore(
                model_name=name,
                score=float(score),
                confidence=1.0,
                available=True
            )
            for name, score in scores_dict.items()
            if score is not None
        }
        return self.score(model_scores, pii_sensitivity)

    def _score_to_label(self, score: float) -> str:
        if score >= RISK_THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        elif score >= RISK_THRESHOLDS["HIGH"]:
            return "HIGH"
        elif score >= RISK_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        elif score >= RISK_THRESHOLDS["LOW"]:
            return "LOW"
        return "UNKNOWN"

    def _determine_verdict(
        self,
        score: float,
        model_scores: Dict[str, ModelScore],
        pii_sensitivity: str
    ) -> str:
        if_score = self._get_score(model_scores, "isolation_forest")
        identity_score = self._get_score(model_scores, "identity_detector")
        rf_score = self._get_score(model_scores, "random_forest")
        lstm_score = self._get_score(model_scores, "lstm_detector")
        gnn_score = self._get_score(model_scores, "gnn_detector")
        dns_score = self._get_score(model_scores, "dns_classifier")

        if if_score >= 0.7 and pii_sensitivity in ["PCI", "PHI"] and score >= 0.8:
            return "DATA_EXFILTRATION"
        if identity_score >= 0.8 and if_score >= 0.6:
            return "IDENTITY_COMPROMISE"
        if rf_score >= 0.8 and lstm_score >= 0.7:
            return "MALWARE"
        if dns_score >= 0.8:
            return "DGA_DOMAIN"
        if gnn_score >= 0.8:
            return "GRAPH_ANOMALY"
        if score >= 0.8:
            return "ANOMALY_CRITICAL"
        elif score >= 0.6:
            return "ANOMALY_HIGH"
        elif score >= 0.4:
            return "ANOMALY_MEDIUM"
        return "NORMAL"

    def _get_score(
        self,
        model_scores: Dict[str, ModelScore],
        model_name: str
    ) -> float:
        ms = model_scores.get(model_name)
        if ms and ms.available:
            return ms.score
        return 0.0

    def _build_explanation(
        self,
        score: float,
        label: str,
        highest_model: str,
        highest_score: float,
        pii_sensitivity: str,
        pii_multiplier: float,
        models_used: int
    ) -> str:
        explanation = (
            f"{label} risk score {score:.3f} "
            f"from {models_used} models. "
        )
        if highest_model:
            explanation += (
                f"Strongest signal: "
                f"{highest_model.replace('_', ' ')} "
                f"({highest_score:.3f}). "
            )
        if pii_multiplier > 1.0:
            explanation += (
                f"Score elevated {pii_multiplier}x "
                f"due to {pii_sensitivity} data access. "
            )
        if score >= 0.8:
            explanation += "Immediate investigation required."
        elif score >= 0.6:
            explanation += "Investigate within 1 hour."
        elif score >= 0.4:
            explanation += "Monitor — escalate if score increases."
        return explanation

    def _empty_result(self) -> EnsembleResult:
        """Return empty result when no scores provided"""
        return EnsembleResult(
            final_score=0.0,
            risk_label="UNKNOWN",
            verdict="NORMAL",
            models_available=len(self.weights),
            explanation="No model scores provided."
        )

    def get_weight(self, model_name: str) -> float:
        return self.weights.get(model_name, 0.0)

    def update_weight(
        self,
        model_name: str,
        new_weight: float
    ) -> None:
        if model_name not in self.weights:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Valid models: {list(self.weights.keys())}"
            )
        old_weight = self.weights[model_name]
        delta = new_weight - old_weight
        other_models = [m for m in self.weights if m != model_name]
        adjustment = delta / len(other_models)
        self.weights[model_name] = new_weight
        for model in other_models:
            self.weights[model] -= adjustment
        self._validate_weights()
        logger.info(
            f"Weight updated: {model_name} "
            f"{old_weight:.3f} → {new_weight:.3f}. "
            f"SR 11-7: document this change."
        )