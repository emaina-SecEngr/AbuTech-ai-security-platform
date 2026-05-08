"""
Layer 2 — ML Processing
Autoencoder Anomaly Detector

WHY THIS EXISTS:
    Isolation Forest measures statistical distance.
    An attacker who keeps statistics normal
    but changes behavior patterns evades it.

    Autoencoder learns the INTERNAL STRUCTURE
    of normal behavior — the relationships
    between features — not just their values.

    If reconstruction error is high →
    the event does not follow normal patterns →
    potential zero-day attack.

HOW IT WORKS:
    ENCODER: input(10) → hidden(6) → latent(3)
    DECODER: latent(3) → hidden(6) → output(10)

    Train on normal traffic only.
    Model learns to reconstruct normal perfectly.
    Novel attack patterns = high reconstruction error.

ARCHITECTURE CHOICE:
    We use PyTorch because:
    - Dynamic computation graphs (easier debugging)
    - PyTorch Geometric for future GNN work
    - Hugging Face models (SecBERT) are PyTorch
    - Security ML research is PyTorch dominant

BOTTLENECK SIZE:
    Input:   10 features (network)
    Latent:   3 features (30% compression)
    
    Too tight (1): loses signal, everything anomalous
    Too loose (8): learns nothing, misses attacks
    Sweet spot: 20-30% of input dimensions

TRAINING:
    Train ONLY on benign traffic.
    Including attack traffic teaches model
    that attacks are normal → data contamination.
    This is different from overfitting —
    it is learning the WRONG baseline.
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


# ============================================================
# AUTOENCODER RESULT
# ============================================================

@dataclass
class AutoencoderResult:
    """
    Result from Autoencoder anomaly detection.

    KEY DIFFERENCE FROM ISOLATION FOREST:
    We also return reconstruction_error
    so analysts can see HOW anomalous
    not just WHETHER anomalous.

    reconstruction_error:
        0.001 = reconstructed almost perfectly
                → very normal
        0.5+  = could not reconstruct properly
                → very anomalous
    """
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    reconstruction_error: float = 0.0
    confidence: str = "LOW"
    anomaly_type: str = "UNKNOWN"
    risk_reasons: list = field(default_factory=list)
    feature_errors: dict = field(default_factory=dict)
    scored_at: str = ""
    model_version: str = "1.0.0"

    def to_dict(self) -> dict:
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "reconstruction_error": (
                self.reconstruction_error
            ),
            "confidence": self.confidence,
            "anomaly_type": self.anomaly_type,
            "risk_reasons": self.risk_reasons,
            "feature_errors": self.feature_errors,
            "scored_at": self.scored_at
        }


# ============================================================
# PYTORCH AUTOENCODER ARCHITECTURE
#
# WHY THIS ARCHITECTURE:
# We define three separate autoencoders
# one per event type because:
# - Network features have different scales
#   than process features
# - Training one model on mixed data
#   dilutes the learning
# - Separate models = more sensitive detection
# ============================================================

def build_network_autoencoder():
    """
    Build autoencoder for network flow events.

    Architecture:
        Encoder: 10 → 6 → 3
        Decoder: 3  → 6 → 10

    WHY THESE SIZES:
        10 input features (as built in IF)
        6 hidden = capture main patterns
        3 latent = information bottleneck
                   forces model to learn
                   essential structure only
    """
    try:
        import torch
        import torch.nn as nn

        class NetworkAutoencoder(nn.Module):
            def __init__(self):
                super().__init__()

                # ENCODER
                # Compresses 10 → 6 → 3
                self.encoder = nn.Sequential(
                    nn.Linear(10, 6),
                    nn.ReLU(),
                    # ReLU: non-linear activation
                    # allows model to learn complex
                    # non-linear patterns
                    # Without it: just linear compression
                    nn.Linear(6, 3),
                    nn.ReLU()
                )

                # DECODER
                # Reconstructs 3 → 6 → 10
                self.decoder = nn.Sequential(
                    nn.Linear(3, 6),
                    nn.ReLU(),
                    nn.Linear(6, 10),
                    nn.Sigmoid()
                    # Sigmoid: output between 0 and 1
                    # matches our scaled input range
                )

            def forward(self, x):
                """
                Forward pass through encoder
                then decoder.
                Returns reconstructed input.
                """
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

            def encode(self, x):
                """Return compressed representation"""
                return self.encoder(x)

        return NetworkAutoencoder()

    except ImportError:
        return None


def build_process_autoencoder():
    """
    Build autoencoder for process events.

    Architecture:
        Encoder: 8 → 5 → 2
        Decoder: 2 → 5 → 8

    Smaller than network because
    process has fewer features.
    """
    try:
        import torch.nn as nn

        class ProcessAutoencoder(nn.Module):
            def __init__(self):
                super().__init__()

                self.encoder = nn.Sequential(
                    nn.Linear(8, 5),
                    nn.ReLU(),
                    nn.Linear(5, 2),
                    nn.ReLU()
                )

                self.decoder = nn.Sequential(
                    nn.Linear(2, 5),
                    nn.ReLU(),
                    nn.Linear(5, 8),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.decoder(
                    self.encoder(x)
                )

        return ProcessAutoencoder()

    except ImportError:
        return None


def build_iam_autoencoder():
    """
    Build autoencoder for IAM events.

    Architecture:
        Encoder: 8 → 5 → 2
        Decoder: 2 → 5 → 8
    """
    try:
        import torch.nn as nn

        class IAMAutoencoder(nn.Module):
            def __init__(self):
                super().__init__()

                self.encoder = nn.Sequential(
                    nn.Linear(8, 5),
                    nn.ReLU(),
                    nn.Linear(5, 2),
                    nn.ReLU()
                )

                self.decoder = nn.Sequential(
                    nn.Linear(2, 5),
                    nn.ReLU(),
                    nn.Linear(5, 8),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.decoder(
                    self.encoder(x)
                )

        return IAMAutoencoder()

    except ImportError:
        return None


# ============================================================
# AUTOENCODER DETECTOR
# ============================================================

class AutoencoderDetector:
    """
    Deep learning anomaly detector using Autoencoders.

    WHY BETTER THAN ISOLATION FOREST:
    Isolation Forest: measures statistical distance
    Autoencoder:      learns internal structure
                      and feature relationships

    An attacker can match statistics.
    They cannot easily fake learned patterns.

    TRAINING APPROACH:
        Train ONLY on benign events.
        Use MSE loss (mean squared error).
        Minimize reconstruction error on normal.
        High error at inference = anomaly.

    THRESHOLD SELECTION:
        After training calculate reconstruction
        error on a validation set of normal events.
        Set threshold at 95th percentile.
        Events above threshold = anomaly.
        This gives approximately 5% false positive rate.

    Usage:
        detector = AutoencoderDetector()
        detector.train_network(normal_features)
        result = detector.score_network(ecs_event)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        threshold_percentile: float = 95.0
    ):
        """
        Initialize Autoencoder detector.

        Args:
            learning_rate: How fast the model learns.
                          0.001 is a standard starting
                          point for Adam optimizer.
                          Too high = unstable training
                          Too low  = too slow to learn

            epochs:       How many times to pass
                          through training data.
                          50 is enough for our size.
                          More = better but slower.

            batch_size:   How many events per
                          gradient update.
                          32 is standard.
                          Smaller = more updates
                          Larger  = more stable

            threshold_percentile:
                          What percentile of training
                          reconstruction errors to use
                          as anomaly threshold.
                          95 = top 5% of normal
                          errors trigger anomaly.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile

        # Models
        self.network_model = None
        self.process_model = None
        self.iam_model = None

        # Scalers
        self._network_scaler = None
        self._process_scaler = None
        self._iam_scaler = None

        # Anomaly thresholds
        # Set during training from validation errors
        self._network_threshold = 0.1
        self._process_threshold = 0.1
        self._iam_threshold = 0.1

        # Training metadata
        self.network_trained_on = 0
        self.process_trained_on = 0
        self.iam_trained_on = 0

        # Statistics
        self.network_scored = 0
        self.process_scored = 0
        self.iam_scored = 0
        self.anomalies_detected = 0

        logger.info(
            f"AutoencoderDetector initialized "
            f"lr={learning_rate} "
            f"epochs={epochs}"
        )

    # ============================================================
    # TRAINING METHODS
    # ============================================================

    def train_network(
        self,
        feature_matrix: np.ndarray
    ) -> dict:
        """
        Train network anomaly autoencoder.

        TRAINING PROCESS:
        1. Scale features to 0-1 range
        2. Build PyTorch model
        3. For each epoch:
           a. Shuffle training data
           b. Process in batches
           c. Forward pass → reconstruction
           d. Calculate MSE loss
           e. Backpropagate → update weights
        4. Calculate threshold from training errors

        MSE LOSS:
        Mean Squared Error between input and output.
        Low MSE = reconstruction was accurate.
        High MSE = model could not reconstruct.

        Args:
            feature_matrix: Normal network flows only
                           Shape: (n_samples, 10)

        Returns:
            Training history dictionary
        """
        return self._train_model(
            feature_matrix=feature_matrix,
            model_type="network",
            input_size=10
        )

    def train_process(
        self,
        feature_matrix: np.ndarray
    ) -> dict:
        """Train process anomaly autoencoder"""
        return self._train_model(
            feature_matrix=feature_matrix,
            model_type="process",
            input_size=8
        )

    def train_iam(
        self,
        feature_matrix: np.ndarray
    ) -> dict:
        """Train IAM anomaly autoencoder"""
        return self._train_model(
            feature_matrix=feature_matrix,
            model_type="iam",
            input_size=8
        )

    def _train_model(
        self,
        feature_matrix: np.ndarray,
        model_type: str,
        input_size: int
    ) -> dict:
        """
        Core training loop for any model type.

        WHY WE SEPARATE TRAINING FROM SCORING:
        Training happens once (or periodically).
        Scoring happens millions of times per day.
        Keeping them separate means we can
        retrain without stopping the scoring pipeline.
        """
        history = {
            "final_loss": None,
            "threshold": None,
            "trained_on": 0,
            "epochs_run": 0
        }

        if len(feature_matrix) < 10:
            logger.warning(
                f"Not enough samples for "
                f"{model_type} — need at least 10"
            )
            return history

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from sklearn.preprocessing import (
                MinMaxScaler
            )

            # STEP 1: Scale to 0-1 range
            # Sigmoid output = 0-1
            # Input must also be 0-1 for MSE to work
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(
                feature_matrix
            ).astype(np.float32)

            # STEP 2: Build model
            if model_type == "network":
                model = build_network_autoencoder()
                self._network_scaler = scaler
            elif model_type == "process":
                model = build_process_autoencoder()
                self._process_scaler = scaler
            else:
                model = build_iam_autoencoder()
                self._iam_scaler = scaler

            if model is None:
                logger.error("PyTorch not available")
                return history

            # STEP 3: Define optimizer and loss
            # Adam optimizer: adaptive learning rate
            # Better than basic SGD for autoencoders
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.learning_rate
            )
            criterion = nn.MSELoss()
            # MSE = mean squared error
            # (prediction - actual)^2
            # Penalizes large reconstruction errors more

            # STEP 4: Convert to PyTorch tensors
            X = torch.FloatTensor(scaled)

            # STEP 5: Training loop
            model.train()
            losses = []

            for epoch in range(self.epochs):
                # Shuffle data each epoch
                # Prevents model from memorizing order
                idx = torch.randperm(len(X))
                X_shuffled = X[idx]

                epoch_loss = 0.0
                n_batches = 0

                # Process in batches
                for i in range(
                    0, len(X_shuffled), self.batch_size
                ):
                    batch = X_shuffled[
                        i:i+self.batch_size
                    ]

                    # Zero gradients from last batch
                    optimizer.zero_grad()

                    # Forward pass
                    # Model tries to reconstruct input
                    reconstructed = model(batch)

                    # Calculate how wrong it was
                    loss = criterion(
                        reconstructed, batch
                    )

                    # Backpropagation
                    # Calculate gradients
                    loss.backward()

                    # Update weights in direction
                    # that reduces loss
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

            # STEP 6: Calculate anomaly threshold
            # Threshold = 95th percentile of
            # reconstruction errors on training data
            # Events above this = anomaly
            model.eval()
            with torch.no_grad():
                reconstructed = model(X)
                errors = torch.mean(
                    (X - reconstructed) ** 2,
                    dim=1
                ).numpy()

            threshold = np.percentile(
                errors,
                self.threshold_percentile
            )

            # Store model and threshold
            if model_type == "network":
                self.network_model = model
                self._network_threshold = float(
                    threshold
                )
                self.network_trained_on = len(
                    feature_matrix
                )
            elif model_type == "process":
                self.process_model = model
                self._process_threshold = float(
                    threshold
                )
                self.process_trained_on = len(
                    feature_matrix
                )
            else:
                self.iam_model = model
                self._iam_threshold = float(threshold)
                self.iam_trained_on = len(
                    feature_matrix
                )

            history.update({
                "final_loss": losses[-1],
                "threshold": float(threshold),
                "trained_on": len(feature_matrix),
                "epochs_run": self.epochs
            })

            logger.info(
                f"{model_type} autoencoder trained: "
                f"loss={losses[-1]:.6f} "
                f"threshold={threshold:.6f} "
                f"samples={len(feature_matrix)}"
            )

            return history

        except ImportError as e:
            logger.error(
                f"PyTorch not available: {e}"
            )
            return history
        except Exception as e:
            logger.error(
                f"Training failed: {e}"
            )
            return history

    # ============================================================
    # SCORING METHODS
    # ============================================================

    def score_network(
        self,
        ecs_event
    ) -> AutoencoderResult:
        """
        Score network event using autoencoder.

        WHAT HAPPENS:
        1. Extract features from ECS event
        2. Scale using training scaler
        3. Run through encoder → decoder
        4. Calculate reconstruction error
        5. Compare to threshold
        6. Return AutoencoderResult
        """
        from layer2_ml.anomaly.isolation_forest_detector\
            import extract_network_features

        result = AutoencoderResult(
            anomaly_type="network_deep_anomaly",
            scored_at=self._now()
        )

        features = extract_network_features(ecs_event)
        if features is None:
            return result

        if self.network_model is not None:
            return self._score_with_autoencoder(
                features=features,
                model=self.network_model,
                scaler=self._network_scaler,
                threshold=self._network_threshold,
                anomaly_type="network_deep_anomaly",
                result=result
            )
        else:
            return self._rule_based_score(
                features, result, "network"
            )

    def score_process(
        self,
        ecs_event
    ) -> AutoencoderResult:
        """Score process event using autoencoder"""
        from layer2_ml.anomaly.isolation_forest_detector\
            import extract_process_features

        result = AutoencoderResult(
            anomaly_type="process_deep_anomaly",
            scored_at=self._now()
        )

        features = extract_process_features(ecs_event)
        if features is None:
            return result

        if self.process_model is not None:
            return self._score_with_autoencoder(
                features=features,
                model=self.process_model,
                scaler=self._process_scaler,
                threshold=self._process_threshold,
                anomaly_type="process_deep_anomaly",
                result=result
            )
        else:
            return self._rule_based_score(
                features, result, "process"
            )

    def score_iam(
        self,
        iam_event
    ) -> AutoencoderResult:
        """Score IAM event using autoencoder"""
        from layer2_ml.anomaly.isolation_forest_detector\
            import extract_iam_features

        result = AutoencoderResult(
            anomaly_type="iam_deep_anomaly",
            scored_at=self._now()
        )

        features = extract_iam_features(iam_event)
        if features is None:
            return result

        if self.iam_model is not None:
            return self._score_with_autoencoder(
                features=features,
                model=self.iam_model,
                scaler=self._iam_scaler,
                threshold=self._iam_threshold,
                anomaly_type="iam_deep_anomaly",
                result=result
            )
        else:
            return self._rule_based_score(
                features, result, "iam"
            )

    # ============================================================
    # CORE SCORING ENGINE
    # ============================================================

    def _score_with_autoencoder(
        self,
        features: np.ndarray,
        model,
        scaler,
        threshold: float,
        anomaly_type: str,
        result: AutoencoderResult
    ) -> AutoencoderResult:
        """
        Score features using trained autoencoder.

        RECONSTRUCTION ERROR CALCULATION:
        error = mean((input - reconstructed)^2)

        This is MSE per event.
        Same formula as training loss
        but applied to ONE event at inference.

        FEATURE-LEVEL ERROR:
        We also calculate error per feature.
        This tells the analyst WHICH features
        drove the anomaly detection.

        "Feature 0 (fwd_bytes) had 10x normal
         reconstruction error"
        → Outbound traffic volume is anomalous
        → Possible data exfiltration
        """
        try:
            import torch

            # Scale input
            scaled = scaler.transform(
                features.reshape(1, -1)
            ).astype(np.float32)
            tensor = torch.FloatTensor(scaled)

            # Reconstruct
            model.eval()
            with torch.no_grad():
                reconstructed = model(tensor)

            # Calculate overall error
            error = torch.mean(
                (tensor - reconstructed) ** 2
            ).item()

            # Calculate per-feature error
            per_feature_error = (
                (tensor - reconstructed) ** 2
            ).squeeze().numpy()

            feature_errors = {
                f"feature_{i}": float(e)
                for i, e in enumerate(per_feature_error)
            }

            # Normalize error to 0-1 score
            # Using threshold as reference point
            anomaly_score = min(
                1.0,
                error / (threshold * 2)
                if threshold > 0 else error
            )

            is_anomaly = error > threshold

            result.is_anomaly = is_anomaly
            result.reconstruction_error = round(
                error, 6
            )
            result.anomaly_score = round(
                anomaly_score, 3
            )
            result.feature_errors = feature_errors
            result.confidence = (
                "HIGH" if anomaly_score >= 0.8
                else "MEDIUM" if anomaly_score >= 0.6
                else "LOW"
            )

            if is_anomaly:
                # Find which features had highest error
                top_features = sorted(
                    feature_errors.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]

                result.risk_reasons.append(
                    f"Autoencoder reconstruction error: "
                    f"{error:.6f} "
                    f"(threshold: {threshold:.6f}) — "
                    f"event does not match learned "
                    f"normal patterns"
                )

                for feat, feat_error in top_features:
                    if feat_error > threshold * 0.1:
                        result.risk_reasons.append(
                            f"High error in {feat}: "
                            f"{feat_error:.6f}"
                        )

            self.anomalies_detected += (
                1 if is_anomaly else 0
            )

        except Exception as e:
            logger.debug(
                f"Autoencoder scoring failed: {e}"
            )

        return result

    def _rule_based_score(
        self,
        features: np.ndarray,
        result: AutoencoderResult,
        event_type: str
    ) -> AutoencoderResult:
        """
        Simple fallback when no model trained.
        Reuses Isolation Forest rule logic.
        """
        from layer2_ml.anomaly.isolation_forest_detector\
            import IsolationForestDetector

        if_detector = IsolationForestDetector()

        if event_type == "network":
            if_result = (
                if_detector._rule_based_network_score(
                    features,
                    type('R', (), {
                        'anomaly_score': 0.0,
                        'is_anomaly': False,
                        'risk_reasons': []
                    })()
                )
            )
        elif event_type == "process":
            if_result = (
                if_detector._rule_based_process_score(
                    features,
                    type('R', (), {
                        'anomaly_score': 0.0,
                        'is_anomaly': False,
                        'risk_reasons': []
                    })()
                )
            )
        else:
            if_result = (
                if_detector._rule_based_iam_score(
                    features,
                    type('R', (), {
                        'anomaly_score': 0.0,
                        'is_anomaly': False,
                        'risk_reasons': []
                    })()
                )
            )

        result.anomaly_score = if_result.anomaly_score
        result.is_anomaly = if_result.is_anomaly
        result.risk_reasons = if_result.risk_reasons
        result.confidence = "LOW"

        return result

    # ============================================================
    # MODEL PERSISTENCE
    # ============================================================

    def save_models(
        self,
        model_dir: str = "models/autoencoder"
    ) -> None:
        """Save trained models to disk"""
        try:
            import torch
            os.makedirs(model_dir, exist_ok=True)

            if self.network_model:
                torch.save({
                    "model_state": (
                        self.network_model.state_dict()
                    ),
                    "threshold": (
                        self._network_threshold
                    ),
                    "trained_on": (
                        self.network_trained_on
                    )
                }, f"{model_dir}/network_ae.pt")

                if self._network_scaler:
                    with open(
                        f"{model_dir}/network_scaler.pkl",
                        "wb"
                    ) as f:
                        pickle.dump(
                            self._network_scaler, f
                        )

            if self.process_model:
                torch.save({
                    "model_state": (
                        self.process_model.state_dict()
                    ),
                    "threshold": (
                        self._process_threshold
                    ),
                    "trained_on": (
                        self.process_trained_on
                    )
                }, f"{model_dir}/process_ae.pt")

            logger.info(
                f"Autoencoder models saved to "
                f"{model_dir}"
            )

        except Exception as e:
            logger.error(
                f"Model save failed: {e}"
            )

    def get_statistics(self) -> dict:
        return {
            "network_model_trained": (
                self.network_model is not None
            ),
            "process_model_trained": (
                self.process_model is not None
            ),
            "iam_model_trained": (
                self.iam_model is not None
            ),
            "network_trained_on": (
                self.network_trained_on
            ),
            "process_trained_on": (
                self.process_trained_on
            ),
            "anomalies_detected": (
                self.anomalies_detected
            ),
            "network_threshold": (
                self._network_threshold
            ),
            "process_threshold": (
                self._process_threshold
            )
        }

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )


# ============================================================
# SYNTHETIC DATA GENERATORS
# ============================================================

def generate_normal_network_autoencoder(
    n_samples: int = 500
) -> np.ndarray:
    """
    Generate synthetic normal network data
    for autoencoder training.

    WHY DIFFERENT FROM ISOLATION FOREST:
    Autoencoder needs MORE structure in data
    because it learns RELATIONSHIPS.
    We add correlation between features
    to simulate real traffic patterns.

    Real relationship:
    More bytes = more packets (correlated)
    Longer duration = lower flow rate (correlated)
    """
    np.random.seed(42)
    n = n_samples

    # Correlated features simulate real traffic
    # fwd_packets drives fwd_bytes
    fwd_packets = np.random.lognormal(3, 0.5, n)
    fwd_bytes = fwd_packets * np.random.normal(
        500, 50, n
    )

    bwd_packets = fwd_packets * np.random.normal(
        1.5, 0.2, n
    )
    bwd_bytes = bwd_packets * np.random.normal(
        800, 80, n
    )

    duration_ms = np.random.lognormal(7, 1, n)
    total_bytes = fwd_bytes + bwd_bytes
    flow_rate = total_bytes / (duration_ms / 1000 + 1)

    fwd_pkt_mean = fwd_bytes / (fwd_packets + 1)
    bwd_pkt_mean = bwd_bytes / (bwd_packets + 1)

    features = np.column_stack([
        np.abs(fwd_bytes),
        np.abs(bwd_bytes),
        np.abs(fwd_packets),
        np.abs(bwd_packets),
        np.abs(duration_ms),
        np.abs(flow_rate),
        np.abs(fwd_pkt_mean),
        np.abs(bwd_pkt_mean),
        np.random.choice(
            [0.007, 0.012, 0.021], n
        ),
        np.random.choice(
            [1.0, 0.5], n, p=[0.8, 0.2]
        )
    ])

    return features


def generate_normal_process_autoencoder(
    n_samples: int = 500
) -> np.ndarray:
    """
    Generate synthetic normal process data
    for autoencoder training.
    """
    np.random.seed(42)
    n = n_samples

    features = np.column_stack([
        np.abs(np.random.normal(50, 15, n)),
        np.random.beta(1, 15, n),
        np.zeros(n),
        np.zeros(n),
        np.abs(np.random.normal(12, 3, n)),
        np.abs(np.random.normal(3, 0.3, n)),
        np.zeros(n),
        np.random.beta(1, 8, n)
    ])

    return features