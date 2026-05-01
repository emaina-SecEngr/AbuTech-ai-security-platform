"""
Layer 2 — ML Processing Engine
DNS Classifier Tests

Tests verify three things:

1. FEATURE EXTRACTION
   Does entropy correctly distinguish DGA from legit?
   Do structural features capture domain patterns?
   Do lexical features identify random characters?
   Does behavioral context flag system processes?

2. TRAINING DATA GENERATION
   Does synthetic data correctly separate DGA patterns?
   Do DGA samples have higher entropy than legitimate?

3. PRODUCTION CLASSIFIER
   Does rule-based scoring work without a model?
   Does the ECS bridge extract DNS events correctly?
   Does error handling work for bad input?
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from layer2_ml.nlp.dns_features import (
    DNSFeatureExtractor,
    DYNAMIC_DNS_PROVIDERS
)
from layer2_ml.nlp.dns_classifier import (
    DNSClassifier,
    DGATrainingDataGenerator,
    DNSDetectionResult
)


# ============================================================
# SAMPLE DOMAINS FOR TESTING
# ============================================================

LEGITIMATE_DOMAINS = [
    "google.com",
    "microsoft.com",
    "mail.acme.com",
    "vpn.corporate.net",
    "login.portal.org"
]

DGA_DOMAINS = [
    "xjf8k2mp.duckdns.org",      # Your test domain
    "a3f9k2xp1q.xyz",             # High entropy
    "7b4d9f2e1c3a.no-ip.com",    # Dynamic DNS
    "micros0ft-update.xyz",       # Typosquatting
    "fde3a2b1c4e5f6.com"         # Hex-like DGA
]


def make_mock_dns_ecs_event(
    domain: str,
    process_name: str = "svchost.exe"
):
    """Build mock ECSNormalized DNS event"""
    mock_dest = MagicMock()
    mock_dest.domain = domain

    mock_process = MagicMock()
    mock_process.name = process_name

    mock_event = MagicMock()
    mock_event.category = "dns"

    mock_ecs = MagicMock()
    mock_ecs.destination = mock_dest
    mock_ecs.process = mock_process
    mock_ecs.event = mock_event

    return mock_ecs


# ============================================================
# TEST CLASS — FEATURE EXTRACTION
# ============================================================

class TestDNSFeatureExtraction:
    """Tests for DNSFeatureExtractor"""

    def setup_method(self):
        self.extractor = DNSFeatureExtractor()

    def test_feature_count_correct(self):
        """Extractor produces correct number of features"""
        features = self.extractor.extract("google.com")
        assert features is not None
        assert len(features) == (
            self.extractor.get_feature_count()
        )

    def test_dga_domain_has_higher_entropy(self):
        """
        DGA domain has higher entropy than legitimate.
        xjf8k2mp has random chars — high entropy.
        google has dictionary chars — low entropy.
        """
        legit_features = self.extractor.extract(
            "google.com"
        )
        dga_features = self.extractor.extract(
            "xjf8k2mp.duckdns.org"
        )

        assert legit_features is not None
        assert dga_features is not None

        feature_names = self.extractor.get_feature_names()
        entropy_idx = feature_names.index("sld_entropy")

        assert (
            dga_features[entropy_idx] >
            legit_features[entropy_idx]
        )

    def test_dynamic_dns_correctly_flagged(self):
        """
        Dynamic DNS domains correctly identified.
        xjf8k2mp.duckdns.org should flag is_dynamic_dns.
        """
        features = self.extractor.extract(
            "xjf8k2mp.duckdns.org"
        )
        assert features is not None
        feature_names = self.extractor.get_feature_names()
        dyn_idx = feature_names.index("is_dynamic_dns")
        assert features[dyn_idx] == 1

    def test_legitimate_domain_not_dynamic_dns(self):
        """google.com is not a dynamic DNS domain"""
        features = self.extractor.extract("google.com")
        assert features is not None
        feature_names = self.extractor.get_feature_names()
        dyn_idx = feature_names.index("is_dynamic_dns")
        assert features[dyn_idx] == 0

    def test_suspicious_tld_flagged(self):
        """
        Suspicious TLD correctly identified.
        .xyz is in SUSPICIOUS_TLDS.
        """
        features = self.extractor.extract(
            "random123domain.xyz"
        )
        assert features is not None
        feature_names = self.extractor.get_feature_names()
        tld_idx = feature_names.index(
            "is_suspicious_tld"
        )
        assert features[tld_idx] == 1

    def test_known_legitimate_domain_flagged(self):
        """
        Known legitimate domains correctly identified.
        google.com should set is_known_legitimate.
        """
        features = self.extractor.extract("google.com")
        assert features is not None
        feature_names = self.extractor.get_feature_names()
        legit_idx = feature_names.index(
            "is_known_legitimate"
        )
        assert features[legit_idx] == 1

    def test_low_vowel_ratio_for_dga(self):
        """
        DGA domains have very few vowels.
        xjf8k2mp has no vowels — very low ratio.
        """
        features = self.extractor.extract(
            "xjf8k2mp.duckdns.org"
        )
        assert features is not None
        feature_names = self.extractor.get_feature_names()
        vowel_idx = feature_names.index("vowel_ratio")
        assert features[vowel_idx] < 0.2

    def test_normal_vowel_ratio_for_legitimate(self):
        """
        Legitimate domains have normal vowel ratios.
        google has vowels: o, o, e = 3/6 = 0.5
        """
        features = self.extractor.extract("google.com")
        assert features is not None
        feature_names = self.extractor.get_feature_names()
        vowel_idx = feature_names.index("vowel_ratio")
        assert features[vowel_idx] > 0.2

    def test_suspicious_process_flagged(self):
        """
        System processes making DNS requests flagged.
        svchost.exe should not query random domains.
        """
        features = self.extractor.extract(
            "xjf8k2mp.duckdns.org",
            requesting_process="svchost.exe"
        )
        assert features is not None
        feature_names = self.extractor.get_feature_names()
        proc_idx = feature_names.index(
            "requesting_process_suspicious"
        )
        assert features[proc_idx] == 1

    def test_browser_process_not_flagged(self):
        """
        Browser processes making DNS requests are normal.
        chrome.exe querying domains is expected.
        """
        features = self.extractor.extract(
            "google.com",
            requesting_process="chrome.exe"
        )
        assert features is not None
        feature_names = self.extractor.get_feature_names()
        browser_idx = feature_names.index(
            "requesting_process_browser"
        )
        assert features[browser_idx] == 1

    def test_suspicious_combination_detected(self):
        """
        System process + DGA domain = highest risk.
        svchost.exe + duckdns.org = is_suspicious_combination
        """
        features = self.extractor.extract(
            "xjf8k2mp.duckdns.org",
            requesting_process="svchost.exe"
        )
        assert features is not None
        feature_names = self.extractor.get_feature_names()
        combo_idx = feature_names.index(
            "is_suspicious_combination"
        )
        assert features[combo_idx] == 1

    def test_none_input_returns_none(self):
        """None domain handled gracefully"""
        result = self.extractor.extract(None)
        assert result is None

    def test_empty_string_returns_none(self):
        """Empty string handled gracefully"""
        result = self.extractor.extract("")
        assert result is None

    def test_high_entropy_flag_for_random_domain(self):
        """
        Random character domain flagged as high entropy.
        """
        features = self.extractor.extract(
            "xjf8k2mp.duckdns.org"
        )
        assert features is not None
        feature_names = self.extractor.get_feature_names()
        entropy_idx = feature_names.index(
            "is_high_entropy"
        )
        assert features[entropy_idx] == 1


# ============================================================
# TEST CLASS — TRAINING DATA GENERATION
# ============================================================

class TestDGATrainingData:
    """Tests for DGATrainingDataGenerator"""

    def setup_method(self):
        self.generator = DGATrainingDataGenerator()

    def test_generates_correct_counts(self):
        """Generator produces requested sample counts"""
        X, y, domains = self.generator.generate(
            n_legitimate=100,
            n_dga=50
        )
        assert len(X) == len(y) == len(domains)
        assert len(X) <= 150

    def test_class_distribution(self):
        """Labels correctly assigned"""
        X, y, domains = self.generator.generate(
            n_legitimate=100,
            n_dga=100
        )
        assert np.sum(y == 0) > 0
        assert np.sum(y == 1) > 0

    def test_dga_has_higher_entropy(self):
        """
        DGA samples have higher entropy on average
        than legitimate samples.
        Validates synthetic data quality.
        """
        X, y, domains = self.generator.generate(
            n_legitimate=500,
            n_dga=500
        )

        extractor = DNSFeatureExtractor()
        feature_names = extractor.get_feature_names()
        entropy_idx = feature_names.index("sld_entropy")

        legit_entropy = np.mean(X[y == 0, entropy_idx])
        dga_entropy = np.mean(X[y == 1, entropy_idx])

        assert dga_entropy > legit_entropy

    def test_feature_dimensions_consistent(self):
        """All samples have same feature count"""
        X, y, domains = self.generator.generate(
            n_legitimate=50,
            n_dga=50
        )
        extractor = DNSFeatureExtractor()
        assert X.shape[1] == extractor.get_feature_count()


# ============================================================
# TEST CLASS — DNS CLASSIFIER
# ============================================================

class TestDNSClassifier:
    """Tests for DNSClassifier"""

    def setup_method(self):
        self.classifier = DNSClassifier()

    def test_rule_based_scores_dga_high(self):
        """
        Rule-based scoring flags DGA domain as high risk.
        Dynamic DNS + random characters = high score.
        """
        result = self.classifier.score_domain(
            "xjf8k2mp.duckdns.org",
            requesting_process="svchost.exe"
        )
        assert result is not None
        assert result.risk_score > 0.5

    def test_rule_based_scores_legitimate_low(self):
        """
        Rule-based scoring gives legitimate domain low risk.
        """
        result = self.classifier.score_domain(
            "google.com",
            requesting_process="chrome.exe"
        )
        assert result is not None
        assert result.risk_score < 0.5

    def test_indicators_identified_for_dga(self):
        """
        DGA indicators correctly identified.
        Dynamic DNS and high entropy should appear.
        """
        result = self.classifier.score_domain(
            "xjf8k2mp.duckdns.org",
            requesting_process="svchost.exe"
        )
        assert result is not None
        assert len(result.dga_indicators) > 0

    def test_dga_family_identified(self):
        """
        DGA family correctly identified from
        domain characteristics.
        """
        result = self.classifier.score_domain(
            "xjf8k2mp.duckdns.org"
        )
        assert result is not None
        assert result.dga_family != ""
        assert result.dga_family == "dynamic_dns_abuse"

    def test_explanation_contains_domain(self):
        """Explanation includes domain name"""
        result = self.classifier.score_domain(
            "xjf8k2mp.duckdns.org"
        )
        assert result is not None
        assert "xjf8k2mp.duckdns.org" in (
            result.explanation
        )

    def test_non_dns_event_returns_none(self):
        """
        Non-DNS events rejected gracefully.
        DNS classifier only handles dns category.
        """
        mock_event = MagicMock()
        mock_event.event.category = "network"

        result = self.classifier.score_ecs_event(
            mock_event
        )
        assert result is None

    def test_none_input_returns_none(self):
        """None input handled gracefully"""
        result = self.classifier.score_domain(None)
        assert result is None

    def test_ecs_event_scoring(self):
        """
        ECS DNS event correctly scored.
        Domain extracted from destination.domain.
        """
        mock_event = make_mock_dns_ecs_event(
            "xjf8k2mp.duckdns.org",
            "svchost.exe"
        )
        result = self.classifier.score_ecs_event(
            mock_event
        )
        assert result is not None
        assert result.domain == "xjf8k2mp.duckdns.org"

    def test_confidence_mapping(self):
        """Probability maps to correct confidence label"""
        assert self.classifier._score_to_confidence(
            0.95
        ) == "HIGH"
        assert self.classifier._score_to_confidence(
            0.65
        ) == "MEDIUM"
        assert self.classifier._score_to_confidence(
            0.30
        ) == "LOW"

    def test_result_has_all_fields(self):
        """
        Result has all fields needed by
        Layer 3 and Layer 4.
        """
        result = self.classifier.score_domain(
            "xjf8k2mp.duckdns.org",
            "svchost.exe"
        )
        assert result is not None
        assert hasattr(result, "is_dga")
        assert hasattr(result, "risk_score")
        assert hasattr(result, "confidence")
        assert hasattr(result, "dga_indicators")
        assert hasattr(result, "dga_family")
        assert hasattr(result, "explanation")
        assert hasattr(result, "domain")
        assert hasattr(result, "scored_at")

    def test_performance_stats_tracked(self):
        """Performance statistics correctly tracked"""
        for _ in range(3):
            self.classifier.score_domain(
                "xjf8k2mp.duckdns.org"
            )

        stats = self.classifier.get_performance_stats()
        assert stats["total_scored"] == 3
        assert stats["total_dga_detected"] >= 0

    def test_with_mock_ml_model(self):
        """ML model scoring works when loaded"""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = (
            np.array([[0.05, 0.95]])
        )

        self.classifier.model = mock_model
        self.classifier.model_name = "test_dns_model"

        result = self.classifier.score_domain(
            "xjf8k2mp.duckdns.org"
        )
        assert result is not None
        assert result.risk_score == 0.95
        assert result.is_dga is True
        assert result.confidence == "HIGH"
        
    def test_with_mock_ml_model(self):
        """ML model scoring works when loaded"""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = (
            np.array([[0.05, 0.95]])
        )

        # Fit scaler with dummy data first
        extractor = DNSFeatureExtractor()
        dummy_X = np.random.randn(
            10, extractor.get_feature_count()
        )
        self.classifier.scaler.fit(dummy_X)

        self.classifier.model = mock_model
        self.classifier.model_name = "test_dns_model"

        result = self.classifier.score_domain(
            "xjf8k2mp.duckdns.org"
        )
        assert result is not None
        assert result.risk_score == 0.95
        assert result.is_dga is True
        assert result.confidence == "HIGH"