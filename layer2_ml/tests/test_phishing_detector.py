"""
Tests for the Layer 2 Phishing Detector.
"""

import pytest

from layer2_ml.phishing_detection.phishing_detector\
    import (
        PhishingDetector,
        PhishingDetectionResult,
        MITRE_PHISHING,
    )
from layer2_ml.phishing_detection.phishing_features\
    import (
        PhishingFeatureExtractor,
        shannon_entropy,
    )


@pytest.fixture
def detector():
    return PhishingDetector()


@pytest.fixture
def extractor():
    return PhishingFeatureExtractor()


@pytest.fixture
def obvious_phish():
    return {
        "sender": "security@paypa1-verify.com",
        "display_name": "PayPal Security",
        "subject": "URGENT: Verify your account immediately or it will be suspended",
        "url": "http://192.168.99.1/paypal/login",
        "attachment": "invoice.exe",
    }


@pytest.fixture
def legit_email():
    return {
        "sender": "newsletter@github.com",
        "display_name": "GitHub",
        "subject": "Your weekly digest of repository activity",
        "url": "https://github.com/notifications",
        "attachment": "",
    }


@pytest.fixture
def subtle_phish():
    return {
        "sender": "noreply@account-update.net",
        "display_name": "Microsoft Account Team",
        "subject": "Please confirm your identity to continue",
        "url": "https://account-update.net/login",
        "attachment": "",
    }


# ============================================================
# FEATURE EXTRACTION
# ============================================================

class TestFeatureExtraction:

    def test_extract_returns_dict(
        self, extractor, obvious_phish
    ):
        features = extractor.extract(obvious_phish)
        assert isinstance(features, dict)

    def test_detects_ip_url(
        self, extractor, obvious_phish
    ):
        features = extractor.extract(obvious_phish)
        assert features["url_is_ip"] == 1

    def test_detects_dangerous_attachment(
        self, extractor, obvious_phish
    ):
        features = extractor.extract(obvious_phish)
        assert features["dangerous_attachment"] == 1

    def test_detects_lookalike_brand(self, extractor):
        # lookalike check examines the DOMAIN; use a
        # domain that impersonates a brand
        event = {
            "sender": "noreply@paypa1.com",
            "url": "http://paypa1.com/login",
        }
        features = extractor.extract(event)
        assert features["url_lookalike_brand"] == 1

    def test_detects_urgency(
        self, extractor, obvious_phish
    ):
        features = extractor.extract(obvious_phish)
        assert features["urgency_keyword_count"] >= 1

    def test_display_name_mismatch(
        self, extractor, obvious_phish
    ):
        # display "PayPal" but sender domain isn't paypal
        features = extractor.extract(obvious_phish)
        assert features["display_name_mismatch"] == 1

    def test_legit_email_clean(
        self, extractor, legit_email
    ):
        features = extractor.extract(legit_email)
        assert features["url_is_ip"] == 0
        assert features["dangerous_attachment"] == 0
        assert features["url_lookalike_brand"] == 0

    def test_empty_event(self, extractor):
        features = extractor.extract({})
        assert features["has_url"] == 0

    def test_none_event(self, extractor):
        features = extractor.extract(None)
        assert isinstance(features, dict)


# ============================================================
# SHANNON ENTROPY
# ============================================================

class TestEntropy:

    def test_entropy_empty(self):
        assert shannon_entropy("") == 0.0

    def test_entropy_uniform_low(self):
        # all same char = 0 entropy
        assert shannon_entropy("aaaa") == 0.0

    def test_entropy_random_higher(self):
        low = shannon_entropy("aaaa")
        high = shannon_entropy("a8x2k9qz")
        assert high > low


# ============================================================
# SCORING — WEIGHTED (no model)
# ============================================================

class TestWeightedScoring:

    def test_obvious_phish_high_score(
        self, detector, obvious_phish
    ):
        result = detector.score(obvious_phish)
        assert result.risk_score >= 0.80
        assert result.verdict == "PHISHING"

    def test_legit_email_low_score(
        self, detector, legit_email
    ):
        result = detector.score(legit_email)
        assert result.risk_score < 0.25
        assert result.verdict == "BENIGN"

    def test_subtle_phish_flagged(
        self, detector, subtle_phish
    ):
        result = detector.score(subtle_phish)
        # brand impersonation + credential request
        assert result.risk_score >= 0.25

    def test_result_has_reasons(
        self, detector, obvious_phish
    ):
        result = detector.score(obvious_phish)
        assert len(result.reasons) > 0

    def test_result_has_mitre(
        self, detector, obvious_phish
    ):
        result = detector.score(obvious_phish)
        assert result.mitre_technique == MITRE_PHISHING

    def test_score_never_exceeds_one(
        self, detector, obvious_phish
    ):
        result = detector.score(obvious_phish)
        assert result.risk_score <= 1.0

    def test_result_to_dict(
        self, detector, obvious_phish
    ):
        result = detector.score(obvious_phish)
        d = result.to_dict()
        assert "risk_score" in d
        assert "verdict" in d
        assert "mitre_technique" in d

    def test_empty_email_benign(self, detector):
        result = detector.score({})
        assert result.verdict == "BENIGN"


# ============================================================
# VERDICT BANDING
# ============================================================

class TestVerdicts:

    def test_verdict_bands(self, detector):
        assert detector._verdict(0.90) == "PHISHING"
        assert detector._verdict(0.60) == "SUSPICIOUS"
        assert detector._verdict(0.30) == "LOW_RISK"
        assert detector._verdict(0.10) == "BENIGN"


# ============================================================
# TRAINED MODEL PATH (mocked)
# ============================================================

class TestModelScoring:

    def test_uses_model_when_present(
        self, obvious_phish
    ):
        class FakeModel:
            def predict_proba(self, X):
                return [[0.05, 0.95]]

        detector = PhishingDetector(model=FakeModel())
        result = detector.score(obvious_phish)
        assert result.risk_score == 0.95
        assert result.reasons == ["ml_model_score"]

    def test_model_failure_falls_back(
        self, obvious_phish
    ):
        class BrokenModel:
            def predict_proba(self, X):
                raise ValueError("broken")

        detector = PhishingDetector(model=BrokenModel())
        result = detector.score(obvious_phish)
        # falls back to weighted scoring
        assert result.risk_score >= 0.80