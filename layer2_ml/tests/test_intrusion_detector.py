"""
Layer 2 — ML Processing Engine
Network Intrusion Detection Tests

These tests verify three things:

1. DATA PREPARATION
   Does CICIDS2017 loading and cleaning work correctly?
   Does SMOTE handle class imbalance properly?
   Does temporal splitting maintain data integrity?

2. FEATURE ENGINEERING
   Do engineered features produce correct values?
   Do behavioral flags correctly identify attack patterns?
   Do rate features handle edge cases like zero duration?

3. PRODUCTION DETECTOR
   Does scoring work correctly on known attack patterns?
   Does the ECS bridge extract features correctly?
   Does explanation generation produce useful output?
   Does error handling work for malformed input?

Note: These tests use synthetic data so they run
without requiring the full CICIDS2017 dataset.
This makes the CI/CD pipeline fast and reliable.

Real model performance is validated separately
using the full dataset in Databricks notebooks.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from layer2_ml.intrusion_detection.data_preparation import (
    CICIDSDataPreparation,
    SELECTED_FEATURES,
    LABEL_COLUMN
)
from layer2_ml.intrusion_detection.feature_engineering import (
    NetworkFlowFeatureEngineer
)
from layer2_ml.intrusion_detection.detector import (
    NetworkIntrusionDetector,
    DetectionResult
)


# ============================================================
# SYNTHETIC DATA GENERATORS
# Create realistic test data without needing real files
# ============================================================

def make_benign_flow(**kwargs) -> dict:
    """
    Generate a synthetic benign network flow.
    Mimics normal HTTP browsing traffic.
    """
    flow = {
        " Flow Duration": 500000,
        " Total Fwd Packets": 10,
        " Total Backward Packets": 8,
        " Total Length of Fwd Packets": 5000,
        " Total Length of Bwd Packets": 15000,
        " Fwd Packet Length Max": 1460,
        " Fwd Packet Length Mean": 500,
        " Bwd Packet Length Max": 1460,
        " Bwd Packet Length Mean": 1875,
        " Flow Bytes/s": 40000,
        " Flow Packets/s": 36,
        " Flow IAT Mean": 50000,
        " Flow IAT Std": 25000,
        " Fwd IAT Mean": 100000,
        " Bwd IAT Mean": 80000,
        " SYN Flag Count": 1,
        " RST Flag Count": 0,
        " PSH Flag Count": 2,
        " ACK Flag Count": 18,
        " Init_Win_bytes_forward": 65535,
        " Label": "BENIGN"
    }
    flow.update(kwargs)
    return flow


def make_dos_flow(**kwargs) -> dict:
    """
    Generate a synthetic DoS attack flow.
    Mimics SYN flood characteristics.
    """
    flow = {
        " Flow Duration": 100,
        " Total Fwd Packets": 10000,
        " Total Backward Packets": 0,
        " Total Length of Fwd Packets": 640000,
        " Total Length of Bwd Packets": 0,
        " Fwd Packet Length Max": 64,
        " Fwd Packet Length Mean": 64,
        " Bwd Packet Length Max": 0,
        " Bwd Packet Length Mean": 0,
        " Flow Bytes/s": 6400000,
        " Flow Packets/s": 100000,
        " Flow IAT Mean": 10,
        " Flow IAT Std": 2,
        " Fwd IAT Mean": 10,
        " Bwd IAT Mean": 0,
        " SYN Flag Count": 10000,
        " RST Flag Count": 0,
        " PSH Flag Count": 0,
        " ACK Flag Count": 0,
        " Init_Win_bytes_forward": 0,
        " Label": "DDoS"
    }
    flow.update(kwargs)
    return flow


def make_portscan_flow(**kwargs) -> dict:
    """
    Generate a synthetic port scan flow.
    Mimics TCP SYN scan characteristics.
    """
    flow = {
        " Flow Duration": 50000,
        " Total Fwd Packets": 1,
        " Total Backward Packets": 1,
        " Total Length of Fwd Packets": 40,
        " Total Length of Bwd Packets": 40,
        " Fwd Packet Length Max": 40,
        " Fwd Packet Length Mean": 40,
        " Bwd Packet Length Max": 40,
        " Bwd Packet Length Mean": 40,
        " Flow Bytes/s": 1600,
        " Flow Packets/s": 40,
        " Flow IAT Mean": 50000,
        " Flow IAT Std": 0,
        " Fwd IAT Mean": 50000,
        " Bwd IAT Mean": 50000,
        " SYN Flag Count": 1,
        " RST Flag Count": 1,
        " PSH Flag Count": 0,
        " ACK Flag Count": 0,
        " Init_Win_bytes_forward": 0,
        " Label": "PortScan"
    }
    flow.update(kwargs)
    return flow


def make_synthetic_dataset(
    n_benign: int = 1000,
    n_attack: int = 50
) -> pd.DataFrame:
    """
    Create synthetic CICIDS-like dataset for testing.

    Args:
        n_benign: Number of benign samples
        n_attack: Number of attack samples

    Returns:
        DataFrame with labeled flows
    """
    rows = []

    # Benign flows
    for _ in range(n_benign):
        rows.append(make_benign_flow())

    # DoS attack flows
    for _ in range(n_attack // 2):
        rows.append(make_dos_flow())

    # PortScan flows
    for _ in range(n_attack // 2):
        rows.append(make_portscan_flow())

    return pd.DataFrame(rows)


# ============================================================
# TEST CLASS — DATA PREPARATION
# ============================================================

class TestDataPreparation:
    """Tests for CICIDS2017 data preparation pipeline"""

    def setup_method(self):
        self.prep = CICIDSDataPreparation(
            data_dir="data/cicids2017",
            apply_smote=False,  # Disable for unit tests
            max_samples=500
        )

    def test_binary_label_creation(self):
        """
        BENIGN maps to 0, all attacks map to 1.
        This is the foundation of binary classification.
        """
        df = make_synthetic_dataset(
            n_benign=100, n_attack=20
        )
        df = self.prep._create_binary_labels(df)

        benign_labels = df[df[LABEL_COLUMN] == "BENIGN"][
            "binary_label"
        ]
        attack_labels = df[df[LABEL_COLUMN] != "BENIGN"][
            "binary_label"
        ]

        assert (benign_labels == 0).all()
        assert (attack_labels == 1).all()

    def test_cleaning_removes_infinite_values(self):
        """
        Infinite values from flow calculations are removed.
        ML models cannot handle infinity.
        """
        df = make_synthetic_dataset(n_benign=100)

        # Inject infinite values
        df.loc[0, " Flow Bytes/s"] = np.inf
        df.loc[1, " Flow Packets/s"] = -np.inf

        cleaned = self.prep._clean_data(df)

        # Rows with inf should be removed
        assert np.inf not in cleaned[" Flow Bytes/s"].values
        assert -np.inf not in cleaned[
            " Flow Packets/s"
        ].values

    def test_cleaning_removes_nan_values(self):
        """NaN values in selected features are removed"""
        df = make_synthetic_dataset(n_benign=100)

        # Inject NaN values
        df.loc[0, " Flow Duration"] = np.nan
        df.loc[1, " SYN Flag Count"] = np.nan

        initial_count = len(df)
        cleaned = self.prep._clean_data(df)

        assert len(cleaned) < initial_count
        assert cleaned[" Flow Duration"].isna().sum() == 0

    def test_class_distribution_logging(self):
        """
        Class distribution is correctly calculated.
        Used for monitoring class imbalance.
        """
        df = make_synthetic_dataset(
            n_benign=900, n_attack=100
        )
        df = self.prep._create_binary_labels(df)
        self.prep._log_class_distribution(df)

        stats = self.prep.get_statistics()
        assert stats["benign_count"] == 900
        assert stats["attack_count"] == 100
        assert stats["imbalance_ratio"] == 9.0

    def test_feature_selection_returns_correct_shape(self):
        """Feature selection returns correct dimensions"""
        df = make_synthetic_dataset(
            n_benign=100, n_attack=20
        )
        df = self.prep._create_binary_labels(df)
        X, y = self.prep._select_features(df)

        # Should have one row per sample
        assert len(X) == len(y) == 120

        # X should have features as columns
        assert X.shape[1] > 0

    def test_smote_balances_classes(self):
        """
        SMOTE increases minority class samples.
        After SMOTE classes should be more balanced.
        """
        prep = CICIDSDataPreparation(
            data_dir="data/cicids2017",
            apply_smote=True
        )

        # Create imbalanced training data
        X_train = np.random.randn(1000, 20)
        y_train = np.array([0] * 950 + [1] * 50)

        X_balanced, y_balanced = prep._apply_smote(
            X_train, y_train
        )

        # Attack class should have more samples now
        assert np.sum(y_balanced == 1) > 50

        # Benign class should be unchanged
        assert np.sum(y_balanced == 0) == 950


# ============================================================
# TEST CLASS — FEATURE ENGINEERING
# ============================================================

class TestFeatureEngineering:
    """Tests for network flow feature engineering"""

    def setup_method(self):
        self.engineer = NetworkFlowFeatureEngineer()

    def test_rate_features_calculated_correctly(self):
        """
        Packets per second correctly calculated.
        DoS attacks have very high pps.
        """
        df = pd.DataFrame([make_dos_flow()])
        result = self.engineer._engineer_rate_features(df)

        # DoS flow: 10000 packets in 100 microseconds
        # Expected pps = 10000 / (100 + epsilon) ≈ very high
        assert "engineered_fwd_pps" in result.columns
        assert result["engineered_fwd_pps"].iloc[0] > 50

    def test_rate_features_handle_zero_duration(self):
        """
        Zero duration flows do not cause division by zero.
        Epsilon prevents this edge case.
        """
        df = pd.DataFrame([
            make_benign_flow(**{" Flow Duration": 0})
        ])
        result = self.engineer._engineer_rate_features(df)

        # Should not raise exception
        assert "engineered_fwd_pps" in result.columns
        assert not np.isinf(
            result["engineered_fwd_pps"].iloc[0]
        )

    def test_scan_flag_correctly_identifies_portscan(self):
        """
        Port scan flows correctly flagged by SYN+RST pattern.
        ATT&CK T1046 Network Service Scanning.
        """
        df = pd.DataFrame([make_portscan_flow()])
        result = self.engineer._engineer_behavioral_flags(df)

        assert result["engineered_is_scan_like"].iloc[0] == 1

    def test_scan_flag_does_not_flag_benign(self):
        """
        Normal traffic not incorrectly flagged as scan.
        Benign flows have RST=0 so scan flag should be 0.
        """
        df = pd.DataFrame([make_benign_flow()])
        result = self.engineer._engineer_behavioral_flags(df)

        assert result[
            "engineered_is_scan_like"
        ].iloc[0] == 0

    def test_one_directional_flag_detects_dos(self):
        """
        DoS flows with no return traffic correctly flagged.
        ATT&CK T1498 Network Denial of Service.
        """
        df = pd.DataFrame([make_dos_flow()])
        result = self.engineer._engineer_behavioral_flags(df)

        assert result[
            "engineered_is_one_directional"
        ].iloc[0] == 1

    def test_one_directional_flag_does_not_flag_benign(self):
        """Normal bidirectional traffic not flagged"""
        df = pd.DataFrame([make_benign_flow()])
        result = self.engineer._engineer_behavioral_flags(df)

        assert result[
            "engineered_is_one_directional"
        ].iloc[0] == 0

    def test_flood_flag_detects_dos(self):
        """
        Short high-volume flows correctly flagged.
        DoS flooding characteristic.
        """
        df = pd.DataFrame([make_dos_flow()])
        result = self.engineer._engineer_behavioral_flags(df)

        assert result[
            "engineered_is_flood_like"
        ].iloc[0] == 1

    def test_packet_ratio_high_for_dos(self):
        """
        DoS has very high forward/backward packet ratio.
        Attacker sends but never receives.
        """
        df = pd.DataFrame([make_dos_flow()])
        result = self.engineer._engineer_ratio_features(df)

        # DoS: 10000 fwd packets, 0 bwd packets
        # Ratio should be very high
        assert result[
            "engineered_packet_ratio"
        ].iloc[0] > 100

    def test_engineered_feature_names_tracked(self):
        """
        Engineered feature names correctly tracked.
        Used by SHAP explainability in MLOps layer.
        """
        df = make_synthetic_dataset(n_benign=10)
        self.engineer.engineer_features(df)

        names = self.engineer.get_engineered_feature_names()
        assert len(names) > 0
        assert "engineered_fwd_pps" in names
        assert "engineered_is_scan_like" in names


# ============================================================
# TEST CLASS — PRODUCTION DETECTOR
# ============================================================

class TestNetworkIntrusionDetector:
    """Tests for production inference detector"""

    def setup_method(self):
        """Create detector with mock model for testing"""
        self.detector = NetworkIntrusionDetector(
            threshold=0.5,
            model_version="test-1.0"
        )

        # Create mock model that returns high probability
        # for DOS-like features and low for benign
        self.mock_model = MagicMock()
        self.mock_model.feature_importances_ = (
            np.ones(20) / 20
        )

    def test_score_to_confidence_mapping(self):
        """
        Probability scores map to correct confidence labels.
        Used in analyst dashboard display.
        """
        assert self.detector._score_to_confidence(
            0.95
        ) == "HIGH"
        assert self.detector._score_to_confidence(
            0.65
        ) == "MEDIUM"
        assert self.detector._score_to_confidence(
            0.30
        ) == "LOW"

    def test_explanation_for_dos_attack(self):
        """
        DoS attack features produce relevant explanation.
        Analyst sees why the flow was flagged.
        """
        dos_features = make_dos_flow()
        explanation = self.detector._generate_explanation(
            dos_features,
            prob=0.95,
            is_attack=True
        )

        assert "ATTACK DETECTED" in explanation
        assert len(explanation) > 20

    def test_explanation_for_portscan(self):
        """
        Port scan features produce scan-related explanation.
        """
        scan_features = make_portscan_flow()
        explanation = self.detector._generate_explanation(
            scan_features,
            prob=0.85,
            is_attack=True
        )

        assert "ATTACK DETECTED" in explanation

    def test_explanation_for_benign(self):
        """
        Benign traffic produces benign explanation.
        """
        benign_features = make_benign_flow()
        explanation = self.detector._generate_explanation(
            benign_features,
            prob=0.15,
            is_attack=False
        )

        assert "benign" in explanation.lower()

    def test_feature_vector_correct_length(self):
        """
        Feature vector has correct number of features.
        Must match what the model was trained on.
        """
        features = make_dos_flow()
        vector = self.detector._build_feature_vector(
            features
        )

        assert vector is not None
        assert len(vector) == len(
            NetworkIntrusionDetector.FEATURE_NAMES
        )

    def test_feature_vector_handles_missing_features(self):
        """
        Missing features default to 0.
        Prevents crashes on incomplete events.
        """
        # Only provide some features
        sparse_features = {
            " Flow Duration": 1000,
            " SYN Flag Count": 5
        }

        vector = self.detector._build_feature_vector(
            sparse_features
        )

        assert vector is not None
        assert len(vector) == len(
            NetworkIntrusionDetector.FEATURE_NAMES
        )
        # Missing features should be 0
        assert vector[1] == 0.0

    def test_feature_vector_handles_infinite_values(self):
        """
        Infinite values replaced with 0.
        Prevents model from receiving invalid input.
        """
        features = make_dos_flow()
        features[" Flow Bytes/s"] = np.inf

        vector = self.detector._build_feature_vector(
            features
        )

        assert vector is not None
        assert not np.any(np.isinf(vector))

    def test_score_features_without_model_returns_none(self):
        """
        Scoring without loaded model returns None gracefully.
        Never crashes — always returns None on failure.
        """
        detector = NetworkIntrusionDetector()
        result = detector.score_features(make_dos_flow())
        assert result is None

    def test_score_features_with_mock_model(self):
        """
        Scoring with model returns DetectionResult.
        """
        # Setup mock model
        self.mock_model.predict.return_value = np.array([1])
        self.mock_model.predict_proba.return_value = (
            np.array([[0.05, 0.95]])
        )

        self.detector.model = self.mock_model
        self.detector.model_name = "test_model"

        result = self.detector.score_features(
            make_dos_flow()
        )

        assert result is not None
        assert isinstance(result, DetectionResult)
        assert result.risk_score == 0.95
        assert result.is_attack is True
        assert result.confidence == "HIGH"

    def test_none_input_returns_none(self):
        """None input handled gracefully"""
        result = self.detector.score_ecs_event(None)
        assert result is None

    def test_performance_stats_tracked(self):
        """
        Performance statistics correctly tracked.
        Used by MLOps monitoring layer.
        """
        self.mock_model.predict.return_value = np.array([1])
        self.mock_model.predict_proba.return_value = (
            np.array([[0.05, 0.95]])
        )
        self.detector.model = self.mock_model
        self.detector.model_name = "test_model"

        # Score three events
        for _ in range(3):
            self.detector.score_features(make_dos_flow())

        stats = self.detector.get_performance_stats()

        assert stats["total_scored"] == 3
        assert stats["total_attacks_detected"] == 3
        assert stats["avg_inference_ms"] >= 0

    def test_batch_scoring(self):
        """
        Batch scoring returns correct number of results.
        """
        self.mock_model.predict.return_value = np.array([0])
        self.mock_model.predict_proba.return_value = (
            np.array([[0.9, 0.1]])
        )
        self.detector.model = self.mock_model
        self.detector.model_name = "test_model"

        events = [make_benign_flow() for _ in range(5)]
        results = self.detector.score_batch(events)

        assert len(results) == 5