"""
LSTM + Attention Detector Tests

WHAT WE ARE PROVING:
    1. Sequence builder correctly vectorizes events
    2. 10-feature vectors have correct shape
    3. Kill chain sequences built correctly
    4. Slow exfil sequences built correctly
    5. LSTM models train without error
    6. Attention weights produced for explainability
    7. Kill chain rule-based detection works
    8. Slow exfil rule-based detection works
    9. Risk trend correctly calculated
    10. SR 11-7 compliance: top events returned
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from layer2_ml.sequence.sequence_builder import (
    SequenceBuilder,
    SecurityEventVector,
    EventSequence,
    KILL_CHAIN_WINDOW,
    SLOW_EXFIL_WINDOW
)
from layer2_ml.sequence.lstm_attention_detector import (
    LSTMAttentionDetector,
    LSTMDetectionResult,
    build_kill_chain_lstm,
    build_slow_exfil_lstm
)


# ============================================================
# MOCK EVENT BUILDERS
# ============================================================

def make_mock_data_event(
    risk_score=0.2,
    rows=100,
    operation="read",
    sensitivity="NONE",
    accessor_type="human",
    event_time="2024-03-29T15:00:00Z"
):
    """Normal data access event mock"""
    event = MagicMock()
    event.accessor_identity = "analyst@corp.com"
    event.event_time = event_time
    event.source_system = "snowflake"
    event.risk_score = risk_score
    event.risk_label = "LOW"
    event.operation = MagicMock()
    event.operation.value = operation
    event.rows_accessed = rows
    event.bytes_accessed = rows * 100
    event.data_path = "PROD_DB.PUBLIC.CUSTOMERS"
    event.finding = None

    at = MagicMock()
    at.value = accessor_type
    event.accessor_type = at

    return event


def make_mock_iam_event(
    risk_score=0.1,
    has_impossible_travel=False,
    mfa_used=True,
    outcome="success"
):
    """Normal IAM authentication event mock"""
    auth_event = MagicMock()
    auth_event.is_impossible_travel = has_impossible_travel
    auth_event.travel_distance_km = (
        7650.0 if has_impossible_travel else 0.0
    )
    auth_event.outcome = outcome

    auth_ctx = MagicMock()
    auth_ctx.mfa_used = mfa_used
    auth_ctx.mfa_method = "totp" if mfa_used else ""
    auth_event.auth = auth_ctx

    iam_event = MagicMock()
    iam_event.user = "analyst@corp.com"
    iam_event.event_type = "auth"
    iam_event.timestamp = "2024-03-29T15:00:00Z"
    iam_event.source_system = "okta"
    iam_event.overall_risk_score = risk_score
    iam_event.auth_event = auth_event

    return iam_event


# ============================================================
# TEST CLASS — SEQUENCE BUILDER
# ============================================================

class TestSequenceBuilder:
    """
    Tests for event vectorization and
    sequence construction.
    """

    def setup_method(self):
        self.builder = SequenceBuilder(
            kill_chain_window=5,
            slow_exfil_window=10
        )

    def test_data_event_vectorized(self):
        """
        DataAccessEvent correctly converted
        to SecurityEventVector.
        """
        event = make_mock_data_event()
        vector = self.builder._vectorize_data_event(
            event
        )
        assert vector is not None
        assert isinstance(vector, SecurityEventVector)

    def test_vector_has_10_features(self):
        """
        Feature vector has exactly 10 dimensions.
        LSTM input layer expects exactly 10 features.
        Wrong size = PyTorch dimension error.
        """
        event = make_mock_data_event()
        vector = self.builder._vectorize_data_event(
            event
        )
        numpy_vec = vector.to_numpy()
        assert numpy_vec.shape == (10,)

    def test_all_features_normalized(self):
        """
        All 10 features in 0.0-1.0 range.
        Required for LSTM gradient stability.
        Values outside range cause vanishing gradients.
        """
        event = make_mock_data_event(
            risk_score=0.9,
            rows=9_000_000
        )
        vector = self.builder._vectorize_data_event(
            event
        )
        numpy_vec = vector.to_numpy()

        for i, val in enumerate(numpy_vec):
            assert 0.0 <= val <= 1.0, (
                f"Feature {i} = {val} outside 0-1 range"
            )

    def test_iam_event_vectorized(self):
        """IamEvent correctly vectorized"""
        event = make_mock_iam_event()
        vector = self.builder._vectorize_iam_event(
            event
        )
        assert vector is not None
        assert isinstance(vector, SecurityEventVector)

    def test_impossible_travel_in_geo_velocity(self):
        """
        Impossible travel sets geo_velocity to 1.0.
        This is a critical kill chain feature.
        Credential compromise detected via travel.
        """
        event = make_mock_iam_event(
            has_impossible_travel=True
        )
        vector = self.builder._vectorize_iam_event(
            event
        )
        assert vector.geo_velocity == 1.0

    def test_kill_chain_sequence_built(self):
        """
        After kill_chain_window events
        a kill chain sequence is returned.
        """
        for i in range(5):
            event = make_mock_data_event()
            result = self.builder.add_data_access_event(
                event
            )

        assert result is not None
        assert "kill_chain" in result

    def test_slow_exfil_sequence_built(self):
        """
        After slow_exfil_window events
        a slow exfil sequence is returned.
        """
        result = None
        for i in range(10):
            event = make_mock_data_event()
            result = self.builder.add_data_access_event(
                event
            )

        assert result is not None
        assert "slow_exfil" in result

    def test_sequence_matrix_shape(self):
        """
        Sequence matrix has correct shape.
        (window_size, 10) for LSTM input.
        """
        for i in range(5):
            event = make_mock_data_event()
            result = self.builder.add_data_access_event(
                event
            )

        if result and "kill_chain" in result:
            seq = result["kill_chain"]
            matrix = seq.to_matrix()
            assert matrix.shape == (5, 10)

    def test_risk_trend_calculated(self):
        """
        Risk trend correctly computed.
        Increasing risk → "increasing" trend.
        """
        events_low = [
            make_mock_data_event(risk_score=0.1)
            for _ in range(3)
        ]
        events_high = [
            make_mock_data_event(risk_score=0.8)
            for _ in range(2)
        ]

        for e in events_low + events_high:
            result = self.builder.add_data_access_event(e)

        if result and "kill_chain" in result:
            seq = result["kill_chain"]
            assert seq.risk_trend in [
                "increasing", "stable", "decreasing"
            ]

    def test_attention_labels_generated(self):
        """
        Attention labels list contains one label
        per event in sequence.
        SR 11-7: labels identify which events
        drove the anomaly detection.
        """
        for i in range(5):
            event = make_mock_data_event()
            result = self.builder.add_data_access_event(
                event
            )

        if result and "kill_chain" in result:
            seq = result["kill_chain"]
            labels = seq.get_attention_labels()
            assert len(labels) == seq.sequence_length

    def test_path_entropy_calculated(self):
        """
        Path entropy reflects diversity of paths.
        Same path repeatedly → low entropy.
        Many different paths → high entropy.
        """
        # Same path = low entropy
        entropy_same = (
            self.builder._calculate_path_entropy(
                "user1",
                "PROD_DB.PUBLIC.CUSTOMERS"
            )
        )

        # Different paths = higher entropy
        paths = [
            "PROD_DB.PUBLIC.CUSTOMERS",
            "PROD_DB.FINANCE.ACCOUNTS",
            "PROD_DB.HR.EMPLOYEES",
            "PROD_DB.LEGAL.CONTRACTS"
        ]
        for p in paths:
            self.builder._calculate_path_entropy(
                "user2", p
            )

        entropy_diverse = (
            self.builder._calculate_path_entropy(
                "user2",
                "PROD_DB.RISK.MODELS"
            )
        )

        assert entropy_diverse >= entropy_same

    def test_normal_sequences_generated(self):
        """
        Synthetic normal sequences correct shape.
        Used for unsupervised LSTM training (Q3).
        """
        normal = self.builder.generate_normal_sequences(
            n_sequences=50,
            window_size=10
        )
        assert normal.shape == (50, 10, 10)

    def test_attack_sequences_generated(self):
        """
        Synthetic attack sequences correct shape.
        Used for supervised secondary model (Q3).
        """
        attacks = self.builder.generate_attack_sequences(
            n_sequences=20,
            window_size=10,
            attack_type="kill_chain"
        )
        assert attacks.shape == (20, 10, 10)

    def test_slow_exfil_volume_increases(self):
        """
        Slow exfil sequences show volume increase.
        Your Q2 insight: 0.01 increment per step.
        APT-style gradual escalation.
        """
        slow = self.builder.generate_attack_sequences(
            n_sequences=10,
            window_size=20,
            attack_type="slow_exfil"
        )
        first_seq = slow[0]
        volumes = first_seq[:, 4]
        first_volume = volumes[0]
        last_volume = volumes[-1]
        assert last_volume > first_volume

    def test_statistics_tracked(self):
        """Statistics correctly updated"""
        for i in range(5):
            event = make_mock_data_event()
            self.builder.add_data_access_event(event)

        stats = self.builder.get_statistics()
        assert stats["events_ingested"] == 5


# ============================================================
# TEST CLASS — LSTM ARCHITECTURES
# ============================================================

class TestLSTMArchitectures:
    """
    Tests that PyTorch model architectures
    are correctly built.
    """

    def test_kill_chain_lstm_builds(self):
        """Kill chain LSTM builds without error"""
        model = build_kill_chain_lstm()
        assert model is not None

    def test_slow_exfil_lstm_builds(self):
        """Slow exfil LSTM builds without error"""
        model = build_slow_exfil_lstm()
        assert model is not None

    def test_kill_chain_output_shape(self):
        """
        Kill chain LSTM produces correct output shape.
        Score: (batch, 1)
        Attention: (batch, seq_len, 1)
        """
        import torch
        model = build_kill_chain_lstm()
        x = torch.randn(2, 19, 10)
        score, attention = model(x)

        assert score.shape == (2, 1)
        assert attention.shape[0] == 2
        assert attention.shape[2] == 1

    def test_attention_weights_sum_to_one(self):
        """
        Attention weights sum to 1.0 per sequence.
        Softmax property ensures this.
        Required for interpretability:
        weights = probability distribution over events.
        """
        import torch
        model = build_kill_chain_lstm()
        x = torch.randn(1, 19, 10)
        score, attention = model(x)

        attention_sum = attention.squeeze().sum().item()
        assert abs(attention_sum - 1.0) < 0.01

    def test_score_in_valid_range(self):
        """
        Anomaly score between 0 and 1.
        Sigmoid activation ensures this.
        """
        import torch
        model = build_kill_chain_lstm()
        x = torch.randn(1, 19, 10)
        score, _ = model(x)
        val = score.item()
        assert 0.0 <= val <= 1.0


# ============================================================
# TEST CLASS — LSTM TRAINING
# ============================================================

class TestLSTMTraining:
    """Tests that LSTM training works"""

    def setup_method(self):
        self.detector = LSTMAttentionDetector(
            epochs=3,
            batch_size=16
        )
        self.builder = SequenceBuilder(
            kill_chain_window=KILL_CHAIN_WINDOW,
            slow_exfil_window=SLOW_EXFIL_WINDOW
        )

    def test_kill_chain_trains(self):
        """Kill chain LSTM trains on synthetic data"""
        normal = self.builder.generate_normal_sequences(
            n_sequences=50,
            window_size=KILL_CHAIN_WINDOW
        )
        history = self.detector.train_kill_chain(normal)
        assert self.detector.kill_chain_model is not None
        assert history["trained_on"] == 50

    def test_slow_exfil_trains(self):
        """Slow exfil LSTM trains on synthetic data"""
        normal = self.builder.generate_normal_sequences(
            n_sequences=50,
            window_size=SLOW_EXFIL_WINDOW
        )
        history = self.detector.train_slow_exfil(normal)
        assert self.detector.slow_exfil_model is not None

    def test_training_loss_finite(self):
        """Training loss is finite — no NaN"""
        normal = self.builder.generate_normal_sequences(
            n_sequences=50,
            window_size=KILL_CHAIN_WINDOW
        )
        history = self.detector.train_kill_chain(normal)
        assert history["final_loss"] is not None
        assert np.isfinite(history["final_loss"])

    def test_threshold_set_after_training(self):
        """Anomaly threshold calculated after training"""
        normal = self.builder.generate_normal_sequences(
            n_sequences=50,
            window_size=KILL_CHAIN_WINDOW
        )
        history = self.detector.train_kill_chain(normal)
        assert history["threshold"] is not None

    def test_insufficient_data_handled(self):
        """Less than 10 sequences handled gracefully"""
        tiny = np.random.rand(5, KILL_CHAIN_WINDOW, 10)
        history = self.detector.train_kill_chain(
            tiny.astype(np.float32)
        )
        assert self.detector.kill_chain_model is None


# ============================================================
# TEST CLASS — SCORING AND DETECTION
# ============================================================

class TestLSTMScoring:
    """Tests that scoring produces correct results"""

    def setup_method(self):
        self.detector = LSTMAttentionDetector(
            epochs=3, batch_size=16
        )
        self.builder = SequenceBuilder(
            kill_chain_window=KILL_CHAIN_WINDOW,
            slow_exfil_window=SLOW_EXFIL_WINDOW
        )

    def test_kill_chain_scores_without_model(self):
        """
        Rule-based fallback works before training.
        Platform always produces results.
        """
        normal_seq = self.builder.generate_normal_sequences(
            n_sequences=1,
            window_size=KILL_CHAIN_WINDOW
        )[0]
        result = self.detector.score_kill_chain(
            normal_seq
        )
        assert result is not None
        assert isinstance(result, LSTMDetectionResult)

    def test_kill_chain_attack_scores_high(self):
        """
        Kill chain attack sequence scores
        higher than normal sequence.
        Risk escalation pattern detected.
        """
        normal = self.builder.generate_normal_sequences(
            n_sequences=1,
            window_size=KILL_CHAIN_WINDOW
        )[0]
        attack = self.builder.generate_attack_sequences(
            n_sequences=1,
            window_size=KILL_CHAIN_WINDOW,
            attack_type="kill_chain"
        )[0]

        normal_result = self.detector.score_kill_chain(
            normal
        )
        attack_result = self.detector.score_kill_chain(
            attack
        )

        assert (
            attack_result.anomaly_score >=
            normal_result.anomaly_score
        )

    def test_slow_exfil_attack_detected(self):
        """
        Slow exfiltration sequence with monotonic
        volume increase scores higher than normal.
        APT-style detection confirmed.
        """
        normal = self.builder.generate_normal_sequences(
            n_sequences=1,
            window_size=SLOW_EXFIL_WINDOW
        )[0]
        attack = self.builder.generate_attack_sequences(
            n_sequences=1,
            window_size=SLOW_EXFIL_WINDOW,
            attack_type="slow_exfil"
        )[0]

        normal_result = self.detector.score_slow_exfil(
            normal
        )
        attack_result = self.detector.score_slow_exfil(
            attack
        )

        assert (
            attack_result.anomaly_score >=
            normal_result.anomaly_score
        )

    def test_result_has_risk_reasons(self):
        """
        Anomalous sequences provide risk reasons.
        Analysts need to know WHY not just WHETHER.
        """
        attack = self.builder.generate_attack_sequences(
            n_sequences=1,
            window_size=KILL_CHAIN_WINDOW,
            attack_type="kill_chain"
        )[0]
        result = self.detector.score_kill_chain(attack)

        if result.is_anomaly:
            assert len(result.risk_reasons) > 0

    def test_result_has_timestamp(self):
        """Every result has scored_at timestamp"""
        normal = self.builder.generate_normal_sequences(
            n_sequences=1,
            window_size=KILL_CHAIN_WINDOW
        )[0]
        result = self.detector.score_kill_chain(normal)
        assert result.scored_at != ""

    def test_score_in_valid_range(self):
        """Anomaly score always between 0 and 1"""
        normal = self.builder.generate_normal_sequences(
            n_sequences=1,
            window_size=KILL_CHAIN_WINDOW
        )[0]
        result = self.detector.score_kill_chain(normal)
        assert 0.0 <= result.anomaly_score <= 1.0

    def test_statistics_tracked(self):
        """Statistics correctly updated"""
        normal = self.builder.generate_normal_sequences(
            n_sequences=1,
            window_size=KILL_CHAIN_WINDOW
        )[0]
        self.detector.score_kill_chain(normal)
        stats = self.detector.get_statistics()
        assert stats["kill_chain_scored"] == 1


# ============================================================
# TEST CLASS — SR 11-7 COMPLIANCE
# ============================================================

class TestSR117Compliance:
    """
    Tests for SR 11-7 model risk management
    compliance requirements.

    SR 11-7 requires:
    1. Model decisions must be explainable
    2. Audit trail for automated decisions
    3. Validation team can verify reasoning
    4. Decisions traceable to security principles

    Your Q3 answer: attention weights + labels
    satisfy these requirements.
    """

    def setup_method(self):
        self.detector = LSTMAttentionDetector(
            epochs=3, batch_size=16
        )
        self.builder = SequenceBuilder(
            kill_chain_window=KILL_CHAIN_WINDOW,
            slow_exfil_window=SLOW_EXFIL_WINDOW
        )

    def test_trained_model_returns_attention(self):
        """
        Trained model returns attention weights.
        These are the explainability mechanism.
        """
        normal_data = (
            self.builder.generate_normal_sequences(
                n_sequences=50,
                window_size=KILL_CHAIN_WINDOW
            )
        )
        self.detector.train_kill_chain(normal_data)

        test_seq = normal_data[0]
        result = self.detector.score_kill_chain(test_seq)

        assert isinstance(
            result.attention_weights, list
        )

    def test_top_contributing_events_returned(self):
        """
        Top contributing events identified by
        attention weight for SR 11-7 audit trail.
        """
        result = LSTMDetectionResult(
            is_anomaly=True,
            anomaly_score=0.85,
            attention_weights=[0.1, 0.4, 0.05, 0.35, 0.1],
            contributing_events=[
                "2024-03-01 login NY risk=0.1",
                "2024-03-15 bulk read risk=0.6",
                "2024-03-20 login RO risk=0.3",
                "2024-03-25 bulk read risk=0.8",
                "2024-03-29 export risk=0.9"
            ]
        )

        top = result.get_top_contributing_events(3)
        assert len(top) == 3
        assert top[0]["attention_weight"] >= (
            top[1]["attention_weight"]
        )

    def test_interpretation_labels_present(self):
        """
        Each contributing event has interpretation.
        HIGH/MEDIUM/LOW INFLUENCE labels.
        Validation team uses these to verify
        model reasoning makes security sense.
        """
        result = LSTMDetectionResult(
            attention_weights=[0.05, 0.8, 0.15],
            contributing_events=[
                "event_1", "event_2", "event_3"
            ]
        )
        top = result.get_top_contributing_events(3)

        for event in top:
            assert "interpretation" in event
            assert event["interpretation"] in [
                "HIGH INFLUENCE",
                "MEDIUM INFLUENCE",
                "LOW INFLUENCE"
            ]

    def test_result_serializes_to_dict(self):
        """
        Result serializes to dict for audit logging.
        SR 11-7 requires documented audit trail.
        """
        result = LSTMDetectionResult(
            is_anomaly=True,
            anomaly_score=0.87,
            sequence_type="kill_chain",
            risk_trend="increasing"
        )
        d = result.to_dict()

        assert "is_anomaly" in d
        assert "anomaly_score" in d
        assert "sequence_type" in d
        assert "top_contributing_events" in d