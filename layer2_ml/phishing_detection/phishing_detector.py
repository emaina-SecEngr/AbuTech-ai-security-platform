"""
Layer 2 — ML Detection
Phishing Detector

Scores a normalized email event for phishing risk.

ARCHITECTURE (same pattern as the DNS/DGA detector):
    1. PhishingFeatureExtractor turns the email into
       numerical features.
    2. This detector scores those features.

    Two scoring modes:
    - WEIGHTED SCORING (default): explainable,
      weighted combination of the phishing signals.
      Works with no trained model — useful immediately
      and fully auditable.
    - TRAINED MODEL (optional): if a trained classifier
      is loaded, the detector uses it instead. This is
      the same "feature layer + pluggable model"
      design as the DNS detector, so a model can be
      dropped in later without changing the interface.

OUTPUT:
    A PhishingDetectionResult with a 0-1 risk score,
    a verdict, the contributing reasons, and the MITRE
    technique (T1566 Phishing).
"""

import logging
import pickle

from layer2_ml.phishing_detection.phishing_features\
    import PhishingFeatureExtractor

logger = logging.getLogger(__name__)


# Weight each feature contributes to the phishing score.
# Tuned so a single strong signal (IP URL, dangerous
# attachment, lookalike brand) pushes toward high risk,
# and multiple weak signals accumulate.
FEATURE_WEIGHTS = {
    "url_is_ip":                  0.30,
    "url_is_shortener":           0.12,
    "url_has_at":                 0.20,
    "url_lookalike_brand":        0.35,
    "display_name_mismatch":      0.30,
    "reply_to_mismatch":          0.18,
    "sender_brand_impersonation": 0.30,
    "has_credential_request":     0.20,
    "dangerous_attachment":       0.40,
    "sender_free_mail":           0.08,
}

# MITRE technique for phishing
MITRE_PHISHING = "T1566"


class PhishingDetectionResult:
    """Result of scoring an email for phishing."""

    def __init__(
        self,
        risk_score: float,
        verdict: str,
        reasons: list,
        features: dict
    ):
        self.risk_score = risk_score
        self.verdict = verdict
        self.reasons = reasons
        self.features = features
        self.mitre_technique = MITRE_PHISHING

    def to_dict(self) -> dict:
        return {
            "risk_score": self.risk_score,
            "verdict": self.verdict,
            "reasons": self.reasons,
            "mitre_technique": self.mitre_technique,
            "features": self.features,
        }


class PhishingDetector:
    """
    Scores email events for phishing using weighted
    signal scoring, with an optional trained model.
    """

    def __init__(self, model=None):
        """
        Args:
            model: optional trained classifier with a
                   predict_proba(X) method. If provided,
                   it is used instead of weighted scoring.
        """
        self.extractor = PhishingFeatureExtractor()
        self.model = model

    def load_model(self, model_path: str):
        """Load a trained classifier from a pickle."""
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(
            f"Loaded phishing model from {model_path}"
        )

    def score(
        self, email_event: dict
    ) -> PhishingDetectionResult:
        """
        Score an email event for phishing risk.

        Args:
            email_event: normalized email dict

        Returns:
            PhishingDetectionResult
        """
        features = self.extractor.extract(email_event)

        if self.model is not None:
            return self._score_with_model(features)
        return self._score_weighted(features)

    def _score_weighted(
        self, features: dict
    ) -> PhishingDetectionResult:
        """Explainable weighted scoring (no model)."""
        score = 0.0
        reasons = []

        for feat, weight in FEATURE_WEIGHTS.items():
            if features.get(feat):
                score += weight
                reasons.append(feat)

        # Urgency keywords scale (capped)
        urgency = features.get(
            "urgency_keyword_count", 0
        )
        if urgency:
            bump = min(urgency * 0.06, 0.20)
            score += bump
            reasons.append(
                f"urgency_keywords:{urgency}"
            )

        # Many dots in URL (sub-domain abuse)
        if features.get("url_dot_count", 0) >= 4:
            score += 0.10
            reasons.append("url_many_subdomains")

        # High-entropy URL domain (random-looking)
        if features.get("url_entropy", 0) >= 3.5:
            score += 0.08
            reasons.append("url_high_entropy")

        score = min(round(score, 4), 1.0)
        verdict = self._verdict(score)

        return PhishingDetectionResult(
            risk_score=score,
            verdict=verdict,
            reasons=reasons,
            features=features,
        )

    def _score_with_model(
        self, features: dict
    ) -> PhishingDetectionResult:
        """Score using a trained classifier."""
        # Order features deterministically
        keys = sorted(features.keys())
        X = [[features[k] for k in keys]]
        try:
            proba = self.model.predict_proba(X)[0][1]
            score = round(float(proba), 4)
        except Exception as e:
            logger.error(
                f"Model scoring failed, falling back "
                f"to weighted: {e}"
            )
            return self._score_weighted(features)

        verdict = self._verdict(score)
        return PhishingDetectionResult(
            risk_score=score,
            verdict=verdict,
            reasons=["ml_model_score"],
            features=features,
        )

    def _verdict(self, score: float) -> str:
        if score >= 0.80:
            return "PHISHING"
        if score >= 0.50:
            return "SUSPICIOUS"
        if score >= 0.25:
            return "LOW_RISK"
        return "BENIGN"