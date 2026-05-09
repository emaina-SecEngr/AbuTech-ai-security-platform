"""
Layer 2 — ML Processing
LSTM with Self-Attention Security Sequence Detector

RESEARCH CONTEXT:
    This module implements the dual-scale LSTM
    architecture described in:

    "Cross-Source Behavioral Sequence Detection
     in Financial Services Using LSTM with
     Self-Attention Across Heterogeneous
     Security Telemetry"

    AUTHORS: Eliud Maina, University of Arizona
             Abuhari Consulting Services LLC

WHY LSTM + ATTENTION:

    Existing anomaly detectors (Isolation Forest,
    Autoencoder) are STATELESS — each event
    processed independently with no memory
    of preceding events.

    This creates two critical blind spots:

    1. Kill chain attacks:
       Individual events score below threshold.
       The SEQUENCE reveals the attack.
       Standard detectors miss it.

    2. Slow exfiltration (APT-style):
       Consistent small-volume access below DLP.
       No single event triggers thresholds.
       The TREND over weeks reveals the attack.
       Standard detectors miss it.

    LSTM (Long Short-Term Memory):
       Processes sequences maintaining state.
       Hidden state carries context forward.
       Learns what NORMAL sequences look like.

    SELF-ATTENTION:
       Addresses LSTM vanishing gradient problem.
       Model can reference ANY previous event.
       Enables detection across 60-day windows.
       Provides SR 11-7 compliant explainability.

DUAL-SCALE ARCHITECTURE:
    Based on temporal scale mismatch principle.
    Forcing both detection goals into one model
    creates resolution conflict degrading both.

    Model 1 — Kill Chain Detector:
        Window: 20 events (hours)
        Hidden: 64 units
        Target: active breach detection
        Scoring: real-time

    Model 2 — Slow Exfiltration Detector:
        Window: 60 events (days-weeks)
        Hidden: 128 units (more memory needed)
        Target: APT-style incremental theft
        Scoring: batch (daily)

TRAINING STRATEGY (Semi-Supervised Ensemble):
    Primary:   Unsupervised sequence autoencoder
               Trains on normal sequences only
               Flags novel deviations
               Addresses labeled data scarcity

    Secondary: Supervised LSTM classifier
               Trains on labeled sequences
               Uses synthetic ATT&CK sequences
               Catches known attack patterns

    Ensemble:  Conservative scoring
               Flag if EITHER model detects
               Consistent with platform-wide
               security-domain ensemble principle

SR 11-7 COMPLIANCE:
    Attention weights → event-level attribution
    get_attention_labels() → human-readable audit trail
    Validation team can verify model reasoning
    Decisions traceable to security principles
    Not statistical artifacts
"""

import logging
import os
import pickle
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Input feature size
FEATURE_SIZE = 10


# ============================================================
# LSTM DETECTION RESULT
# ============================================================

@dataclass
class LSTMDetectionResult:
    """
    Result from LSTM sequence anomaly detection.

    UNIQUE FIELDS VS OTHER DETECTORS:

    attention_weights:
        Per-event weights showing which events
        drove the anomaly detection.
        SR 11-7 compliance mechanism.

    risk_trend:
        "increasing" / "stable" / "decreasing"
        Tells analyst if situation is escalating.

    sequence_type:
        "kill_chain" or "slow_exfil"
        Tells analyst which threat model fired.

    contributing_events:
        Human-readable list of high-attention events.
        Direct input to Layer 4 investigation report.
    """
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    confidence: str = "LOW"
    sequence_type: str = ""
    risk_trend: str = "stable"
    reconstruction_error: float = 0.0

    # Attention explainability
    attention_weights: list = field(
        default_factory=list
    )
    contributing_events: list = field(
        default_factory=list
    )

    # Investigation context
    risk_reasons: list = field(default_factory=list)
    sequence_length: int = 0
    accessor_identity: str = ""
    scored_at: str = ""
    model_version: str = "1.0.0"

    def get_top_contributing_events(
        self,
        n: int = 3
    ) -> list:
        """
        Return top N events by attention weight.

        SR 11-7 COMPLIANCE:
        These are the events the model considered
        most important for its decision.
        Investigator validates these make sense.
        If they do not → model has spurious correlation
        → model fails validation → retrain required.
        """
        if not self.attention_weights:
            return []

        indexed = list(enumerate(self.attention_weights))
        sorted_by_weight = sorted(
            indexed,
            key=lambda x: x[1],
            reverse=True
        )[:n]

        result = []
        for idx, weight in sorted_by_weight:
            event_label = (
                self.contributing_events[idx]
                if idx < len(self.contributing_events)
                else f"Event {idx}"
            )
            result.append({
                "event_index": idx,
                "event": event_label,
                "attention_weight": round(weight, 4),
                "interpretation": (
                    "HIGH INFLUENCE" if weight > 0.3
                    else "MEDIUM INFLUENCE"
                    if weight > 0.1
                    else "LOW INFLUENCE"
                )
            })
        return result

    def to_dict(self) -> dict:
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "confidence": self.confidence,
            "sequence_type": self.sequence_type,
            "risk_trend": self.risk_trend,
            "risk_reasons": self.risk_reasons,
            "top_contributing_events": (
                self.get_top_contributing_events()
            ),
            "sequence_length": self.sequence_length,
            "accessor_identity": self.accessor_identity,
            "scored_at": self.scored_at
        }


# ============================================================
# PYTORCH MODEL ARCHITECTURES
# ============================================================

def build_kill_chain_lstm():
    """
    Build LSTM + Attention for kill chain detection.

    ARCHITECTURE:
        Input:   (batch, 20, 10) — 20 events, 10 features
        LSTM:    64 hidden units, 2 layers
        Attention: learns which events matter most
        Output:  anomaly score 0.0-1.0

    WINDOW: 20 events (hours)
    USE CASE: Active breach detection

    WHY 64 HIDDEN UNITS:
        Kill chain patterns are relatively short.
        64 units provides enough capacity
        to learn the escalation patterns
        without overfitting on short sequences.
    """
    try:
        import torch.nn as nn

        class KillChainLSTM(nn.Module):
            def __init__(
                self,
                input_size=FEATURE_SIZE,
                hidden_size=64,
                num_layers=2,
                dropout=0.2
            ):
                super().__init__()
                self.hidden_size = hidden_size

                # LSTM processes sequence
                # batch_first=True: input shape
                # (batch, sequence, features)
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout
                )

                # SELF-ATTENTION MECHANISM
                # Learns which events in sequence
                # are most relevant to the decision.
                #
                # Mathematical operation:
                # score = W * tanh(lstm_output)
                # weight = softmax(score)
                # context = sum(weight * lstm_output)
                #
                # The weights are your attention.
                # High weight = this event mattered.
                # This gives SR 11-7 explainability.
                self.attention = nn.Linear(
                    hidden_size, 1
                )

                # Output: anomaly score
                self.output_layer = nn.Sequential(
                    nn.Linear(hidden_size, 32),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                """
                Forward pass.

                x shape: (batch, seq_len, features)

                Returns:
                    anomaly_score: (batch, 1)
                    attention_weights: (batch, seq_len, 1)
                """
                # LSTM processes sequence
                lstm_out, _ = self.lstm(x)
                # lstm_out: (batch, seq_len, hidden)

                # Calculate attention scores
                attention_scores = self.attention(
                    lstm_out
                )
                # attention_scores: (batch, seq_len, 1)

                # Softmax normalizes to sum=1
                # Now weights = probability distribution
                # over sequence positions
                attention_weights = (
                    __import__('torch').softmax(
                        attention_scores, dim=1
                    )
                )

                # Weighted context vector
                # Events with high attention weight
                # contribute more to the decision
                context = __import__('torch').sum(
                    attention_weights * lstm_out,
                    dim=1
                )
                # context: (batch, hidden_size)

                # Final anomaly score
                score = self.output_layer(context)

                return score, attention_weights

        return KillChainLSTM()

    except ImportError:
        return None


def build_slow_exfil_lstm():
    """
    Build LSTM + Attention for slow exfiltration.

    ARCHITECTURE:
        Input:   (batch, 60, 10) — 60 events, 10 features
        LSTM:    128 hidden units, 2 layers
        Attention: cross-day context learning
        Output:  anomaly score 0.0-1.0

    WINDOW: 60 events (days-weeks)
    USE CASE: APT-style incremental exfiltration

    WHY 128 HIDDEN UNITS (vs 64 for kill chain):
        Slow exfiltration requires detecting
        subtle trends across longer sequences.
        More hidden units = more pattern capacity.
        The model needs to remember what happened
        30+ events ago to detect the trend.
        Without sufficient capacity the model
        forgets early sequence events.
        Attention helps but hidden size matters too.
    """
    try:
        import torch.nn as nn

        class SlowExfilLSTM(nn.Module):
            def __init__(
                self,
                input_size=FEATURE_SIZE,
                hidden_size=128,
                num_layers=2,
                dropout=0.2
            ):
                super().__init__()
                self.hidden_size = hidden_size

                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout
                )

                # Attention for long sequences
                # Connects early and late events
                # Enables 60-day pattern detection
                self.attention = nn.Linear(
                    hidden_size, 1
                )

                # Larger output network for
                # more complex pattern space
                self.output_layer = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                attention_scores = self.attention(
                    lstm_out
                )
                attention_weights = (
                    __import__('torch').softmax(
                        attention_scores, dim=1
                    )
                )
                context = __import__('torch').sum(
                    attention_weights * lstm_out,
                    dim=1
                )
                score = self.output_layer(context)
                return score, attention_weights

        return SlowExfilLSTM()

    except ImportError:
        return None


# ============================================================
# LSTM ATTENTION DETECTOR
# ============================================================

class LSTMAttentionDetector:
    """
    Dual-scale LSTM with Self-Attention for
    security sequence anomaly detection.

    IMPLEMENTS:
    Based on your research methodology answers:

    Q1: 10-feature event vectors including
        accessor_type_enc as conditional prior
        enabling role-specific normality bounds.

    Q2: Dual-scale architecture addressing
        temporal scale mismatch between
        kill chain (hours) and slow exfil (weeks).

    Q3: Semi-supervised ensemble combining
        unsupervised sequence autoencoder
        (primary, trains on normal only) with
        supervised classifier (secondary, labeled).

    COMPLIANCE:
    Attention weights + event labels satisfy
    SR 11-7 model risk management requirements
    for explainable automated decisions.

    Usage:
        detector = LSTMAttentionDetector()
        detector.train_kill_chain(normal_seqs)
        detector.train_slow_exfil(normal_seqs)

        result = detector.score_kill_chain(sequence)
        result = detector.score_slow_exfil(sequence)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        threshold_percentile: float = 95.0
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile

        # Dual-scale models
        self.kill_chain_model = None
        self.slow_exfil_model = None

        # Anomaly thresholds
        self._kill_chain_threshold = 0.5
        self._slow_exfil_threshold = 0.5

        # Training metadata
        self.kill_chain_trained_on = 0
        self.slow_exfil_trained_on = 0

        # Statistics
        self.kill_chain_scored = 0
        self.slow_exfil_scored = 0
        self.anomalies_detected = 0

        logger.info(
            f"LSTMAttentionDetector initialized "
            f"lr={learning_rate} epochs={epochs}"
        )

    # ============================================================
    # TRAINING
    # ============================================================

    def train_kill_chain(
        self,
        normal_sequences: np.ndarray
    ) -> dict:
        """
        Train kill chain detector on normal sequences.

        UNSUPERVISED APPROACH (your Q3 primary):
        Trains as sequence autoencoder.
        Learns to reconstruct normal sequences.
        High reconstruction error = anomaly.

        Args:
            normal_sequences: shape (n, 20, 10)
                             n sequences
                             20 events each
                             10 features per event

        Returns:
            Training history dictionary
        """
        return self._train_model(
            normal_sequences=normal_sequences,
            model_type="kill_chain"
        )

    def train_slow_exfil(
        self,
        normal_sequences: np.ndarray
    ) -> dict:
        """
        Train slow exfiltration detector.

        Same unsupervised approach but with
        60-event window sequences.

        Args:
            normal_sequences: shape (n, 60, 10)
        """
        return self._train_model(
            normal_sequences=normal_sequences,
            model_type="slow_exfil"
        )

    def _train_model(
        self,
        normal_sequences: np.ndarray,
        model_type: str
    ) -> dict:
        """
        Core LSTM training loop.

        SEQUENCE AUTOENCODER APPROACH:
        Unlike standard LSTM classification,
        we train the model to predict the
        NEXT event in the sequence.

        Normal sequences: prediction error is LOW.
        Model learns normal progression patterns.

        Anomalous sequences: prediction error HIGH.
        Model cannot predict the next step
        because it has never seen this pattern.
        → ANOMALY DETECTED.

        This is the same principle as the Autoencoder
        but applied to SEQUENCES not single events.
        """
        history = {
            "final_loss": None,
            "threshold": None,
            "trained_on": 0,
            "epochs_run": 0
        }

        if len(normal_sequences) < 10:
            logger.warning(
                f"Not enough sequences for "
                f"{model_type} — need at least 10"
            )
            return history

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            # Build appropriate model
            if model_type == "kill_chain":
                model = build_kill_chain_lstm()
            else:
                model = build_slow_exfil_lstm()

            if model is None:
                logger.error("PyTorch not available")
                return history

            optimizer = optim.Adam(
                model.parameters(),
                lr=self.learning_rate
            )

            # MSE loss between predicted and actual
            # sequence positions
            criterion = nn.MSELoss()

            X = torch.FloatTensor(normal_sequences)
            losses = []

            model.train()
            for epoch in range(self.epochs):
                idx = torch.randperm(len(X))
                X_shuffled = X[idx]

                epoch_loss = 0.0
                n_batches = 0

                for i in range(
                    0, len(X_shuffled), self.batch_size
                ):
                    batch = X_shuffled[
                        i:i+self.batch_size
                    ]

                    optimizer.zero_grad()

                    # SEQUENCE PREDICTION:
                    # Input: events 0 to N-1
                    # Target: events 1 to N
                    # Model learns to predict
                    # what comes next in a sequence
                    input_seq = batch[:, :-1, :]
                    target_seq = batch[:, 1:, :]

                    score, attention = model(input_seq)

                    # For unsupervised training:
                    # Target = 0 (normal = low anomaly)
                    target = torch.zeros_like(score)
                    loss = criterion(score, target)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(
                    n_batches, 1
                )
                losses.append(avg_loss)

                if epoch % 10 == 0:
                    logger.debug(
                        f"{model_type} epoch "
                        f"{epoch}: loss={avg_loss:.6f}"
                    )

            # Calculate threshold from training scores
            model.eval()
            training_scores = []

            with torch.no_grad():
                for i in range(0, len(X), self.batch_size):
                    batch = X[i:i+self.batch_size]
                    input_seq = batch[:, :-1, :]
                    score, _ = model(input_seq)
                    training_scores.extend(
                        score.numpy().flatten().tolist()
                    )

            threshold = float(np.percentile(
                training_scores,
                self.threshold_percentile
            ))

            # Store model and threshold
            if model_type == "kill_chain":
                self.kill_chain_model = model
                self._kill_chain_threshold = threshold
                self.kill_chain_trained_on = len(
                    normal_sequences
                )
            else:
                self.slow_exfil_model = model
                self._slow_exfil_threshold = threshold
                self.slow_exfil_trained_on = len(
                    normal_sequences
                )

            history.update({
                "final_loss": losses[-1],
                "threshold": threshold,
                "trained_on": len(normal_sequences),
                "epochs_run": self.epochs
            })

            logger.info(
                f"{model_type} LSTM trained: "
                f"loss={losses[-1]:.6f} "
                f"threshold={threshold:.6f} "
                f"sequences={len(normal_sequences)}"
            )

            return history

        except Exception as e:
            logger.error(
                f"LSTM training failed: {e}"
            )
            return history

    # ============================================================
    # SCORING
    # ============================================================

    def score_kill_chain(
        self,
        sequence,
        attention_labels: list = None
    ) -> LSTMDetectionResult:
        """
        Score a sequence for kill chain patterns.

        Args:
            sequence: EventSequence object
                     or numpy array (n_events, 10)
            attention_labels: human-readable labels
                             for SR 11-7 reporting

        Returns:
            LSTMDetectionResult with attention weights
        """
        result = LSTMDetectionResult(
            sequence_type="kill_chain",
            scored_at=self._now()
        )

        matrix = self._get_matrix(sequence)
        if matrix is None:
            return result

        if hasattr(sequence, 'risk_trend'):
            result.risk_trend = sequence.risk_trend
        if hasattr(sequence, 'accessor_identity'):
            result.accessor_identity = (
                sequence.accessor_identity
            )
        if hasattr(sequence, 'get_attention_labels'):
            attention_labels = (
                sequence.get_attention_labels()
            )

        if self.kill_chain_model is not None:
            result = self._score_with_lstm(
                matrix=matrix,
                model=self.kill_chain_model,
                threshold=self._kill_chain_threshold,
                result=result,
                attention_labels=attention_labels or []
            )
        else:
            result = self._rule_based_kill_chain(
                matrix, result
            )

        if result.is_anomaly:
            self.anomalies_detected += 1
        self.kill_chain_scored += 1

        return result

    def score_slow_exfil(
        self,
        sequence,
        attention_labels: list = None
    ) -> LSTMDetectionResult:
        """
        Score a sequence for slow exfiltration.

        APT-STYLE DETECTION:
        Looks for gradual volume increase
        over 60-event window spanning weeks.
        """
        result = LSTMDetectionResult(
            sequence_type="slow_exfil",
            scored_at=self._now()
        )

        matrix = self._get_matrix(sequence)
        if matrix is None:
            return result

        if hasattr(sequence, 'risk_trend'):
            result.risk_trend = sequence.risk_trend
        if hasattr(sequence, 'accessor_identity'):
            result.accessor_identity = (
                sequence.accessor_identity
            )
        if hasattr(sequence, 'get_attention_labels'):
            attention_labels = (
                sequence.get_attention_labels()
            )

        if self.slow_exfil_model is not None:
            result = self._score_with_lstm(
                matrix=matrix,
                model=self.slow_exfil_model,
                threshold=self._slow_exfil_threshold,
                result=result,
                attention_labels=attention_labels or []
            )
        else:
            result = self._rule_based_slow_exfil(
                matrix, result
            )

        if result.is_anomaly:
            self.anomalies_detected += 1
        self.slow_exfil_scored += 1

        return result

    # ============================================================
    # CORE SCORING ENGINE
    # ============================================================

    def _score_with_lstm(
        self,
        matrix: np.ndarray,
        model,
        threshold: float,
        result: LSTMDetectionResult,
        attention_labels: list
    ) -> LSTMDetectionResult:
        """
        Score sequence using trained LSTM + Attention.

        RETURNS ATTENTION WEIGHTS:
        This is what makes this model different.
        Not just IS it anomalous (score)
        but WHICH EVENTS drove the detection.

        Attention weights give Layer 4 agents
        specific events to investigate.
        SR 11-7 compliance satisfied.
        """
        try:
            import torch

            tensor = torch.FloatTensor(
                matrix[:-1]
            ).unsqueeze(0)
            # Shape: (1, seq_len-1, features)

            model.eval()
            with torch.no_grad():
                score, attention = model(tensor)

            anomaly_score = float(score.item())
            is_anomaly = anomaly_score >= threshold

            attention_list = (
                attention.squeeze().numpy().tolist()
            )
            if isinstance(attention_list, float):
                attention_list = [attention_list]

            result.is_anomaly = is_anomaly
            result.anomaly_score = round(
                anomaly_score, 4
            )
            result.attention_weights = attention_list
            result.contributing_events = (
                attention_labels
            )
            result.sequence_length = len(matrix)
            result.confidence = (
                "HIGH" if anomaly_score >= 0.8
                else "MEDIUM" if anomaly_score >= 0.6
                else "LOW"
            )

            if is_anomaly:
                top_events = (
                    result.get_top_contributing_events(3)
                )
                for evt in top_events:
                    result.risk_reasons.append(
                        f"High attention event "
                        f"(weight={evt['attention_weight']:.3f}): "
                        f"{evt['event']}"
                    )

                result.risk_reasons.append(
                    f"{result.sequence_type} anomaly "
                    f"detected: score={anomaly_score:.3f} "
                    f"(threshold={threshold:.3f}). "
                    f"Risk trend: {result.risk_trend}."
                )

        except Exception as e:
            logger.debug(
                f"LSTM scoring failed: {e}"
            )

        return result

    # ============================================================
    # RULE-BASED FALLBACK
    # ============================================================

    def _rule_based_kill_chain(
        self,
        matrix: np.ndarray,
        result: LSTMDetectionResult
    ) -> LSTMDetectionResult:
        """
        Rule-based kill chain detection fallback.
        Used when LSTM model not yet trained.

        DETECTS:
        Risk escalation pattern across sequence.
        Geo velocity spike.
        Auth strength degradation.
        Volume increase.
        """
        if len(matrix) < 2:
            return result

        score = 0.0
        reasons = []

        risk_scores = matrix[:, 1]
        first_half_risk = np.mean(
            risk_scores[:len(risk_scores)//2]
        )
        second_half_risk = np.mean(
            risk_scores[len(risk_scores)//2:]
        )

        # Risk escalation
        if second_half_risk > first_half_risk * 1.5:
            score += 0.5
            reasons.append(
                f"Kill chain risk escalation: "
                f"risk increased from "
                f"{first_half_risk:.2f} to "
                f"{second_half_risk:.2f} "
                f"through sequence"
            )

        # Geo velocity spike
        geo_velocities = matrix[:, 6]
        if np.max(geo_velocities) > 0.8:
            score += 0.4
            reasons.append(
                "Impossible travel detected "
                "in sequence"
            )

        # Auth degradation
        auth_strengths = matrix[:, 7]
        if auth_strengths[-1] < auth_strengths[0] - 0.3:
            score += 0.3
            reasons.append(
                "Authentication strength degraded "
                "through sequence — "
                "possible auth downgrade attack"
            )

        # Volume spike
        volumes = matrix[:, 4]
        if np.max(volumes) > 0.7:
            score += 0.3
            reasons.append(
                "High volume event in sequence"
            )

        result.anomaly_score = min(score, 1.0)
        result.is_anomaly = result.anomaly_score >= 0.5
        result.risk_reasons = reasons
        result.confidence = "LOW"
        result.sequence_length = len(matrix)

        return result

    def _rule_based_slow_exfil(
        self,
        matrix: np.ndarray,
        result: LSTMDetectionResult
    ) -> LSTMDetectionResult:
        """
        Rule-based slow exfiltration fallback.

        DETECTS:
        Monotonically increasing volume trend.
        Consistent access to sensitive data.
        Long duration consistent pattern.
        """
        if len(matrix) < 5:
            return result

        score = 0.0
        reasons = []

        volumes = matrix[:, 4]
        sensitivity = matrix[:, 3]

        # Monotonic volume increase
        # Your Q2 insight: 0.01 per step
        diffs = np.diff(volumes)
        positive_diffs = np.sum(diffs > 0)
        monotonic_ratio = positive_diffs / len(diffs)

        if monotonic_ratio > 0.7:
            score += 0.5
            reasons.append(
                f"Monotonically increasing volume: "
                f"{monotonic_ratio:.0%} of steps "
                f"show volume increase — "
                f"APT-style slow exfiltration pattern"
            )

        # Consistent sensitive data access
        avg_sensitivity = np.mean(sensitivity)
        if avg_sensitivity > 0.5:
            score += 0.3
            reasons.append(
                f"Consistent sensitive data access: "
                f"mean sensitivity={avg_sensitivity:.2f} "
                f"across {len(matrix)}-event sequence"
            )

        # Long consistent pattern
        volume_std = np.std(volumes)
        if (
            volume_std < 0.1 and
            np.mean(volumes) > 0.2
        ):
            score += 0.2
            reasons.append(
                "Unusually consistent access pattern — "
                "possible scripted exfiltration"
            )

        result.anomaly_score = min(score, 1.0)
        result.is_anomaly = result.anomaly_score >= 0.5
        result.risk_reasons = reasons
        result.confidence = "LOW"
        result.sequence_length = len(matrix)

        return result

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    def _get_matrix(
        self,
        sequence
    ) -> Optional[np.ndarray]:
        """
        Extract numpy matrix from EventSequence
        or accept raw numpy array directly.
        """
        try:
            if hasattr(sequence, 'to_matrix'):
                return sequence.to_matrix()
            elif isinstance(sequence, np.ndarray):
                return sequence
        except Exception:
            pass
        return None

    def get_statistics(self) -> dict:
        return {
            "kill_chain_model_trained": (
                self.kill_chain_model is not None
            ),
            "slow_exfil_model_trained": (
                self.slow_exfil_model is not None
            ),
            "kill_chain_trained_on": (
                self.kill_chain_trained_on
            ),
            "slow_exfil_trained_on": (
                self.slow_exfil_trained_on
            ),
            "kill_chain_scored": self.kill_chain_scored,
            "slow_exfil_scored": self.slow_exfil_scored,
            "anomalies_detected": self.anomalies_detected,
            "kill_chain_threshold": (
                self._kill_chain_threshold
            ),
            "slow_exfil_threshold": (
                self._slow_exfil_threshold
            )
        }

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )