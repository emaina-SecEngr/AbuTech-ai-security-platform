"""
Layer 2 — ML Processing
Isolation Forest Anomaly Detector

WHY THIS FILE EXISTS:
    Your existing ML models (Random Forest,
    XGBoost, rule-based DNS) only detect
    attacks they have been trained on.

    A zero-day attack has never been seen.
    No label. No training example.
    Your current models miss it entirely.

    Isolation Forest solves this by learning
    what NORMAL looks like and flagging
    anything that deviates — regardless of
    whether it matches a known attack pattern.

HOW IT WORKS:
    Build 100 random decision trees.
    For each event count how many cuts
    are needed to isolate it.

    Short path = far from normal = ANOMALY
    Long path  = close to normal = NORMAL

    The anomaly score is based on the
    average path length across all 100 trees.
    Short average path → high anomaly score.

WHERE IT FITS IN YOUR PLATFORM:
    Runs ALONGSIDE existing models.
    Does not replace them.
    Catches what they miss.

    Event arrives → Layer2Router
        ├── MalwareClassifier    ← known malware
        ├── IntrusionDetector    ← known attacks
        ├── DNSClassifier        ← known DGA
        └── IsolationForest      ← UNKNOWN anomalies

TRAINING STRATEGY:
    Train ONLY on benign traffic.
    This is key — if you train on attacks
    the model learns attacks are normal.
    
    Your CICIDS2017 dataset has a BENIGN label.
    We extract only benign rows for training.
    Any deviation from benign = anomaly.

FEATURE GROUPS:
    Network features:
        Packet counts, byte counts, duration
        Port numbers, protocol
        Flow rates (bytes/second)

    Statistical features:
        Mean, std, min, max of packet sizes
        Inter-arrival time statistics

    Behavioral features:
        Connection patterns
        Port diversity
        Protocol diversity
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
# ANOMALY DETECTION RESULT
# ============================================================

@dataclass
class AnomalyDetectionResult:
    """
    Result from Isolation Forest anomaly detection.

    WHY SEPARATE FROM OTHER RESULTS:
    Anomaly detection is fundamentally different
    from classification.

    Classification says: "This IS malware"
    Anomaly detection says: "This is UNUSUAL"

    The distinction matters for analysts:
    Classification → high confidence, act now
    Anomaly → investigate, may be benign

    is_anomaly:     True if anomaly score > threshold
    anomaly_score:  0.0 to 1.0
                    Higher = more anomalous
    confidence:     How confident in the detection
    anomaly_type:   What kind of anomaly
    features_used:  Which features triggered it
    """
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    confidence: str = "LOW"
    anomaly_type: str = "UNKNOWN"
    features_used: list = field(default_factory=list)
    contributing_features: dict = field(
        default_factory=dict
    )
    risk_reasons: list = field(default_factory=list)
    scored_at: str = ""
    model_version: str = "1.0.0"

    def to_dict(self) -> dict:
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "confidence": self.confidence,
            "anomaly_type": self.anomaly_type,
            "risk_reasons": self.risk_reasons,
            "scored_at": self.scored_at
        }


# ============================================================
# FEATURE EXTRACTORS
#
# WHY FEATURE EXTRACTION MATTERS:
# Isolation Forest works on NUMBERS.
# Your events have a mix of strings and numbers.
# We extract the numeric signals that matter.
#
# Good features = model catches real anomalies
# Bad features  = model flags everything or nothing
# ============================================================

def extract_network_features(
    ecs_event
) -> Optional[np.ndarray]:
    """
    Extract numeric features from network flow event.

    WHY THESE FEATURES:
    These 10 features capture the statistical
    signature of a network connection.

    Normal HTTP traffic has:
        Moderate byte counts
        Regular packet sizes
        Consistent flow duration
        Standard port numbers

    Attack traffic has:
        Extreme byte counts (exfiltration)
        Very short duration (scan)
        Very high packet rate (DDoS)
        Unusual port combinations

    Args:
        ecs_event: ECSNormalized network event

    Returns:
        numpy array of 10 features or None
    """
    try:
        network = ecs_event.network
        dest = ecs_event.destination
        source = ecs_event.source

        if not network:
            return None

        features = [
            # Byte counts
            float(network.fwd_bytes or 0),
            float(network.bwd_bytes or 0),

            # Packet counts
            float(network.fwd_packets or 0),
            float(network.bwd_packets or 0),

            # Flow statistics
            float(network.duration_ms or 0),
            float(network.flow_bytes_per_sec or 0),

            # Packet size statistics
            float(
                network.fwd_packet_len_mean or 0
            ),
            float(
                network.bwd_packet_len_mean or 0
            ),

            # Port (normalized to 0-1 range)
            float(
                dest.port / 65535.0
                if dest and dest.port else 0
            ),

            # Protocol flag
            float(
                1.0 if network.protocol == "TCP"
                else 0.5 if network.protocol == "UDP"
                else 0.0
            )
        ]

        return np.array(features, dtype=np.float64)

    except Exception as e:
        logger.debug(
            f"Network feature extraction failed: {e}"
        )
        return None


def extract_process_features(
    ecs_event
) -> Optional[np.ndarray]:
    """
    Extract numeric features from process event.

    WHY THESE FEATURES:
    Process behavior has measurable patterns.
    Normal processes have consistent signatures.
    Malicious processes deviate from these patterns.

    Key signals:
    - Command line length (encoded PS = very long)
    - Special character density (obfuscation)
    - Parent-child relationship risk
    - Process name entropy (random names = malware)

    Returns:
        numpy array of 8 features or None
    """
    try:
        process = ecs_event.process
        if not process:
            return None

        cmd = process.command_line or ""
        name = process.name or ""

        # Command line features
        cmd_length = float(len(cmd))
        cmd_special_chars = float(
            sum(1 for c in cmd
                if c in "^%!@#$&*()[]{}|<>")
        )
        cmd_special_ratio = (
            cmd_special_chars / max(len(cmd), 1)
        )

        # Encoding indicators
        has_base64 = float(
            1.0 if (
                "-enc" in cmd.lower() or
                "-encodedcommand" in cmd.lower()
            ) else 0.0
        )
        has_download = float(
            1.0 if any(
                kw in cmd.lower()
                for kw in [
                    "downloadstring",
                    "webclient",
                    "invoke-webrequest",
                    "curl", "wget"
                ]
            ) else 0.0
        )

        # Process name features
        name_length = float(len(name))
        name_entropy = _calculate_entropy(name)

        # Parent process risk
        parent_risk = 0.0
        if process.parent and process.parent.name:
            parent_name = (
                process.parent.name.lower()
            )
            suspicious_parents = {
                "msbuild.exe", "wscript.exe",
                "cscript.exe", "mshta.exe",
                "regsvr32.exe", "rundll32.exe",
                "certutil.exe", "wmic.exe"
            }
            if parent_name in suspicious_parents:
                parent_risk = 1.0

        features = [
            cmd_length,
            cmd_special_ratio,
            has_base64,
            has_download,
            name_length,
            name_entropy,
            parent_risk,
            float(
                ecs_event.event.severity / 100.0
                if ecs_event.event.severity else 0
            )
        ]

        return np.array(features, dtype=np.float64)

    except Exception as e:
        logger.debug(
            f"Process feature extraction failed: {e}"
        )
        return None


def extract_iam_features(
    iam_event
) -> Optional[np.ndarray]:
    """
    Extract numeric features from IAM event.

    WHY IAM FEATURES:
    Identity behavior has measurable patterns.
    Normal users authenticate from known locations
    at regular times using known devices.

    Anomalies:
    - New country (distance from normal)
    - Impossible travel speed
    - Authentication outside normal hours
    - MFA not used when normally used
    - New device

    Returns:
        numpy array of 8 features or None
    """
    try:
        if not iam_event.auth_event:
            return None

        auth = iam_event.auth_event

        features = [
            # Risk score from normalizer
            float(auth.risk_score or 0),

            # Geographic anomaly
            float(1.0 if auth.is_new_country else 0.0),
            float(
                1.0 if auth.is_impossible_travel
                else 0.0
            ),

            # Travel distance (normalized)
            float(
                min(
                    auth.travel_distance_km / 20000.0,
                    1.0
                )
                if auth.travel_distance_km else 0.0
            ),

            # Authentication context
            float(
                0.0 if (
                    auth.auth and
                    auth.auth.mfa_used
                ) else 0.5
            ),
            float(1.0 if auth.is_new_device else 0.0),

            # Outcome
            float(
                0.0 if auth.outcome == "success"
                else 1.0
            ),

            # Time of day (normalized)
            float(
                _extract_hour(auth.event_time)
                / 24.0
            )
        ]

        return np.array(features, dtype=np.float64)

    except Exception as e:
        logger.debug(
            f"IAM feature extraction failed: {e}"
        )
        return None


# ============================================================
# ISOLATION FOREST DETECTOR
# ============================================================

class IsolationForestDetector:
    """
    Unsupervised anomaly detection using
    Isolation Forest algorithm.

    WHY UNSUPERVISED:
    Supervised models need labels.
    Zero-day attacks have no labels.
    Isolation Forest needs NO labels.
    It only needs to know what NORMAL looks like.

    THREE DETECTION MODES:
    1. Network anomaly detection
       Trained on normal network flows
       Flags unusual traffic patterns

    2. Process anomaly detection
       Trained on normal process behavior
       Flags unusual process characteristics

    3. IAM anomaly detection
       Trained on normal authentication patterns
       Flags unusual identity behavior

    TRAINING APPROACH:
    Each mode has its own model.
    Models trained ONLY on benign samples.
    Contamination parameter controls sensitivity.

    Usage:
        detector = IsolationForestDetector()

        # Train on normal data
        detector.train_network(benign_flows)
        detector.train_process(benign_processes)

        # Score new events
        result = detector.score_network(ecs_event)
        if result.is_anomaly:
            alert(result)
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest detector.

        Args:
            contamination: Expected fraction of
                          anomalies in training data.
                          0.1 = expect 10% anomalies.
                          Lower = more sensitive.
                          Higher = less sensitive.

            n_estimators:  Number of trees to build.
                          More trees = more accurate
                          but slower.
                          100 is a good balance.

            random_state:  Seed for reproducibility.
                          Same seed = same results
                          every time you train.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        # One model per event type
        self.network_model = None
        self.process_model = None
        self.iam_model = None

        # Model metadata
        self.network_trained_on = 0
        self.process_trained_on = 0
        self.iam_trained_on = 0

        # Anomaly threshold
        # Isolation Forest returns scores from
        # -1 (anomaly) to 1 (normal)
        # We convert to 0.0-1.0 scale
        self.threshold = 0.5

        # Statistics
        self.network_scored = 0
        self.process_scored = 0
        self.iam_scored = 0
        self.anomalies_detected = 0

        logger.info(
            f"IsolationForestDetector initialized "
            f"contamination={contamination} "
            f"n_estimators={n_estimators}"
        )

    # ============================================================
    # TRAINING METHODS
    # ============================================================

    def train_network(
        self,
        feature_matrix: np.ndarray
    ) -> None:
        """
        Train network anomaly model.

        WHAT THIS DOES:
        Takes a matrix of normal network flow
        features and builds an Isolation Forest
        that learns what normal traffic looks like.

        TRAINING DATA REQUIREMENT:
        ONLY benign/normal traffic.
        If you include attacks in training
        the model thinks attacks are normal.

        Args:
            feature_matrix: Shape (n_samples, 10)
                           Each row = one network flow
                           Each column = one feature
        """
        try:
            from sklearn.ensemble import (
                IsolationForest
            )
            from sklearn.preprocessing import (
                StandardScaler
            )

            if len(feature_matrix) < 10:
                logger.warning(
                    "Not enough samples to train "
                    "network model — need at least 10"
                )
                return

            # Scale features to same range
            # Important because byte counts (millions)
            # and ratios (0-1) have very different scales
            # Without scaling bytes dominate the model
            self._network_scaler = StandardScaler()
            scaled = self._network_scaler.fit_transform(
                feature_matrix
            )

            # Build the Isolation Forest
            self.network_model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state
            )
            self.network_model.fit(scaled)
            self.network_trained_on = len(
                feature_matrix
            )

            logger.info(
                f"Network model trained on "
                f"{len(feature_matrix)} samples"
            )

        except ImportError:
            logger.error(
                "sklearn not available. "
                "Run: pip install scikit-learn"
            )
        except Exception as e:
            logger.error(
                f"Network model training failed: {e}"
            )

    def train_process(
        self,
        feature_matrix: np.ndarray
    ) -> None:
        """
        Train process anomaly model.

        Same approach as network model but
        trained on process execution features.
        """
        try:
            from sklearn.ensemble import (
                IsolationForest
            )
            from sklearn.preprocessing import (
                StandardScaler
            )

            if len(feature_matrix) < 10:
                return

            self._process_scaler = StandardScaler()
            scaled = self._process_scaler.fit_transform(
                feature_matrix
            )

            self.process_model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state
            )
            self.process_model.fit(scaled)
            self.process_trained_on = len(
                feature_matrix
            )

            logger.info(
                f"Process model trained on "
                f"{len(feature_matrix)} samples"
            )

        except Exception as e:
            logger.error(
                f"Process model training failed: {e}"
            )

    def train_iam(
        self,
        feature_matrix: np.ndarray
    ) -> None:
        """
        Train IAM anomaly model.

        Trained on normal authentication patterns.
        Flags unusual identity behavior.
        """
        try:
            from sklearn.ensemble import (
                IsolationForest
            )
            from sklearn.preprocessing import (
                StandardScaler
            )

            if len(feature_matrix) < 10:
                return

            self._iam_scaler = StandardScaler()
            scaled = self._iam_scaler.fit_transform(
                feature_matrix
            )

            self.iam_model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state
            )
            self.iam_model.fit(scaled)
            self.iam_trained_on = len(feature_matrix)

            logger.info(
                f"IAM model trained on "
                f"{len(feature_matrix)} samples"
            )

        except Exception as e:
            logger.error(
                f"IAM model training failed: {e}"
            )

    # ============================================================
    # SCORING METHODS
    # ============================================================

    def score_network(
        self,
        ecs_event
    ) -> AnomalyDetectionResult:
        """
        Score a network event for anomalies.

        WHAT HAPPENS HERE:
        1. Extract 10 numeric features
        2. Scale using training scaler
        3. Run through Isolation Forest
        4. Convert score to 0.0-1.0 range
        5. Return AnomalyDetectionResult

        If model not trained:
        Returns rule-based fallback score.
        Your platform always produces a result.
        """
        result = AnomalyDetectionResult(
            anomaly_type="network_anomaly",
            scored_at=self._now()
        )

        features = extract_network_features(
            ecs_event
        )

        if features is None:
            return result

        # Use ML model if available
        if self.network_model is not None:
            result = self._score_with_model(
                features,
                self.network_model,
                self._network_scaler,
                "network_anomaly",
                result
            )
            self.network_scored += 1

        else:
            # Fallback rule-based scoring
            result = self._rule_based_network_score(
                features, result
            )

        if result.is_anomaly:
            self.anomalies_detected += 1

        return result

    def score_process(
        self,
        ecs_event
    ) -> AnomalyDetectionResult:
        """Score a process event for anomalies"""
        result = AnomalyDetectionResult(
            anomaly_type="process_anomaly",
            scored_at=self._now()
        )

        features = extract_process_features(
            ecs_event
        )

        if features is None:
            return result

        if self.process_model is not None:
            result = self._score_with_model(
                features,
                self.process_model,
                self._process_scaler,
                "process_anomaly",
                result
            )
            self.process_scored += 1
        else:
            result = self._rule_based_process_score(
                features, result
            )

        if result.is_anomaly:
            self.anomalies_detected += 1

        return result

    def score_iam(
        self,
        iam_event
    ) -> AnomalyDetectionResult:
        """Score an IAM event for anomalies"""
        result = AnomalyDetectionResult(
            anomaly_type="iam_anomaly",
            scored_at=self._now()
        )

        features = extract_iam_features(iam_event)

        if features is None:
            return result

        if self.iam_model is not None:
            result = self._score_with_model(
                features,
                self.iam_model,
                self._iam_scaler,
                "iam_anomaly",
                result
            )
            self.iam_scored += 1
        else:
            result = self._rule_based_iam_score(
                features, result
            )

        if result.is_anomaly:
            self.anomalies_detected += 1

        return result

    # ============================================================
    # CORE SCORING ENGINE
    # ============================================================

    def _score_with_model(
        self,
        features: np.ndarray,
        model,
        scaler,
        anomaly_type: str,
        result: AnomalyDetectionResult
    ) -> AnomalyDetectionResult:
        """
        Score features using trained Isolation Forest.

        SCORE CONVERSION:
        Isolation Forest returns:
        -1 = anomaly (predict)
        +1 = normal  (predict)

        score_samples() returns:
        More negative = more anomalous

        We convert to 0.0-1.0 where:
        1.0 = definitely anomalous
        0.0 = definitely normal
        """
        try:
            scaled = scaler.transform(
                features.reshape(1, -1)
            )

            # Raw anomaly score
            # More negative = more anomalous
            raw_score = model.score_samples(scaled)[0]

            # Convert to 0.0-1.0 range
            # Typical range is -0.5 to 0.5
            # We normalize and invert
            anomaly_score = max(
                0.0,
                min(1.0, (-raw_score + 0.5) * 2)
            )

            is_anomaly = anomaly_score >= self.threshold

            result.is_anomaly = is_anomaly
            result.anomaly_score = round(
                anomaly_score, 3
            )
            result.confidence = (
                "HIGH" if anomaly_score >= 0.8
                else "MEDIUM" if anomaly_score >= 0.6
                else "LOW"
            )
            result.anomaly_type = anomaly_type

            if is_anomaly:
                result.risk_reasons.append(
                    f"Isolation Forest anomaly: "
                    f"score={anomaly_score:.2f} — "
                    f"event deviates significantly "
                    f"from normal baseline"
                )

        except Exception as e:
            logger.debug(
                f"Model scoring failed: {e}"
            )

        return result

    # ============================================================
    # RULE-BASED FALLBACK SCORING
    #
    # WHY FALLBACK EXISTS:
    # When no model is trained yet the platform
    # still needs to produce a score.
    # These rules encode expert knowledge
    # until enough training data is collected.
    # ============================================================

    def _rule_based_network_score(
        self,
        features: np.ndarray,
        result: AnomalyDetectionResult
    ) -> AnomalyDetectionResult:
        """Rule-based network anomaly scoring"""
        score = 0.0
        reasons = []

        fwd_bytes = features[0]
        bwd_bytes = features[1]
        fwd_pkts = features[2]
        duration_ms = features[4]
        flow_rate = features[5]
        dest_port_norm = features[8]

        # Very high byte count = exfiltration risk
        if fwd_bytes > 10_000_000:
            score += 0.4
            reasons.append(
                f"Unusually high outbound bytes: "
                f"{fwd_bytes:.0f}"
            )

        # Very high packet rate = scan or DDoS
        if flow_rate > 100_000:
            score += 0.3
            reasons.append(
                f"Unusually high flow rate: "
                f"{flow_rate:.0f} bytes/sec"
            )

        # Very short connection = scan
        if 0 < duration_ms < 10:
            score += 0.2
            reasons.append(
                "Very short connection duration — "
                "possible port scan"
            )

        # One-directional flow = suspicious
        if fwd_pkts > 0 and features[3] == 0:
            score += 0.2
            reasons.append(
                "One-directional flow — "
                "no response packets"
            )

        result.anomaly_score = min(score, 1.0)
        result.is_anomaly = result.anomaly_score >= 0.5
        result.risk_reasons = reasons
        result.confidence = "LOW"

        return result

    def _rule_based_process_score(
        self,
        features: np.ndarray,
        result: AnomalyDetectionResult
    ) -> AnomalyDetectionResult:
        """Rule-based process anomaly scoring"""
        score = 0.0
        reasons = []

        cmd_length = features[0]
        special_ratio = features[1]
        has_base64 = features[2]
        has_download = features[3]
        name_entropy = features[5]
        parent_risk = features[6]

        if has_base64 > 0:
            score += 0.4
            reasons.append(
                "Encoded command detected — "
                "possible obfuscation"
            )

        if has_download > 0:
            score += 0.3
            reasons.append(
                "Download function in command — "
                "possible dropper"
            )

        if special_ratio > 0.2:
            score += 0.2
            reasons.append(
                f"High special character ratio: "
                f"{special_ratio:.2f} — "
                f"possible obfuscation"
            )

        if parent_risk > 0:
            score += 0.3
            reasons.append(
                "Spawned by suspicious parent process"
            )

        if cmd_length > 500:
            score += 0.2
            reasons.append(
                f"Unusually long command line: "
                f"{cmd_length:.0f} chars"
            )

        result.anomaly_score = min(score, 1.0)
        result.is_anomaly = result.anomaly_score >= 0.5
        result.risk_reasons = reasons
        result.confidence = "LOW"

        return result

    def _rule_based_iam_score(
        self,
        features: np.ndarray,
        result: AnomalyDetectionResult
    ) -> AnomalyDetectionResult:
        """Rule-based IAM anomaly scoring"""
        score = 0.0
        reasons = []

        base_risk = features[0]
        new_country = features[1]
        impossible_travel = features[2]
        travel_distance = features[3]
        no_mfa = features[4]
        new_device = features[5]

        score += base_risk * 0.5

        if impossible_travel > 0:
            score += 0.5
            reasons.append(
                "Impossible travel detected"
            )

        if new_country > 0:
            score += 0.3
            reasons.append(
                "Authentication from new country"
            )

        if no_mfa > 0:
            score += 0.2
            reasons.append(
                "Authentication without MFA"
            )

        if new_device > 0:
            score += 0.2
            reasons.append(
                "Authentication from new device"
            )

        result.anomaly_score = min(score, 1.0)
        result.is_anomaly = result.anomaly_score >= 0.5
        result.risk_reasons = reasons
        result.confidence = "LOW"

        return result

    # ============================================================
    # MODEL PERSISTENCE
    # ============================================================

    def save_models(
        self,
        model_dir: str = "models/anomaly"
    ) -> None:
        """Save trained models to disk"""
        import os
        os.makedirs(model_dir, exist_ok=True)

        if self.network_model:
            with open(
                f"{model_dir}/network_if.pkl", "wb"
            ) as f:
                pickle.dump({
                    "model": self.network_model,
                    "scaler": self._network_scaler,
                    "trained_on": self.network_trained_on
                }, f)

        if self.process_model:
            with open(
                f"{model_dir}/process_if.pkl", "wb"
            ) as f:
                pickle.dump({
                    "model": self.process_model,
                    "scaler": self._process_scaler,
                    "trained_on": self.process_trained_on
                }, f)

        logger.info(
            f"Models saved to {model_dir}"
        )

    def load_models(
        self,
        model_dir: str = "models/anomaly"
    ) -> bool:
        """Load trained models from disk"""
        try:
            network_path = (
                f"{model_dir}/network_if.pkl"
            )
            if os.path.exists(network_path):
                with open(network_path, "rb") as f:
                    data = pickle.load(f)
                    self.network_model = data["model"]
                    self._network_scaler = (
                        data["scaler"]
                    )
                    self.network_trained_on = (
                        data["trained_on"]
                    )

            process_path = (
                f"{model_dir}/process_if.pkl"
            )
            if os.path.exists(process_path):
                with open(process_path, "rb") as f:
                    data = pickle.load(f)
                    self.process_model = data["model"]
                    self._process_scaler = (
                        data["scaler"]
                    )
                    self.process_trained_on = (
                        data["trained_on"]
                    )

            return True

        except Exception as e:
            logger.error(
                f"Model loading failed: {e}"
            )
            return False

    # ============================================================
    # UTILITY METHODS
    # ============================================================

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
            "total_scored": (
                self.network_scored +
                self.process_scored +
                self.iam_scored
            )
        }

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _calculate_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of a string.

    WHY ENTROPY MATTERS:
    Normal process names have LOW entropy.
    "notepad.exe" = predictable, low entropy.

    Random malware names have HIGH entropy.
    "xjf8k2mp.exe" = random, high entropy.

    This is the same concept as your
    DNS classifier uses for DGA detection.
    """
    if not text:
        return 0.0

    import math
    freq = {}
    for c in text.lower():
        freq[c] = freq.get(c, 0) + 1

    entropy = 0.0
    length = len(text)
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def _extract_hour(timestamp: str) -> float:
    """Extract hour from timestamp string"""
    try:
        if "T" in timestamp:
            return float(
                timestamp.split("T")[1][:2]
            )
    except Exception:
        pass
    return 12.0


def generate_synthetic_normal_network(
    n_samples: int = 1000
) -> np.ndarray:
    """
    Generate synthetic normal network traffic
    for training.

    WHY SYNTHETIC DATA:
    In development we do not always have
    labeled benign network captures.
    Synthetic data encodes our domain knowledge
    of what normal traffic looks like.

    In production replace with real benign
    traffic from your CICIDS2017 BENIGN rows.

    Normal traffic characteristics:
    - Moderate byte counts (1KB to 1MB)
    - Regular packet sizes
    - Reasonable flow duration
    - Standard port distribution
    """
    np.random.seed(42)

    features = np.column_stack([
        # fwd_bytes: normal web traffic
        np.random.lognormal(10, 2, n_samples),
        # bwd_bytes
        np.random.lognormal(8, 2, n_samples),
        # fwd_packets
        np.random.lognormal(3, 1, n_samples),
        # bwd_packets
        np.random.lognormal(3, 1, n_samples),
        # duration_ms
        np.random.lognormal(7, 2, n_samples),
        # flow_bytes_per_sec
        np.random.lognormal(6, 2, n_samples),
        # fwd_packet_len_mean
        np.random.lognormal(5, 1, n_samples),
        # bwd_packet_len_mean
        np.random.lognormal(5, 1, n_samples),
        # dest_port (normalized)
        np.random.choice(
            [0.007, 0.012, 0.021],
            n_samples
        ),
        # protocol (1.0=TCP mostly)
        np.random.choice(
            [1.0, 0.5],
            n_samples,
            p=[0.8, 0.2]
        )
    ])

    return features


def generate_synthetic_normal_process(
    n_samples: int = 1000
) -> np.ndarray:
    """
    Generate synthetic normal process data
    for training.

    Normal process characteristics:
    - Short command lines
    - Low special character ratio
    - No base64 encoding
    - No download functions
    - Low entropy process names
    - Benign parent processes
    """
    np.random.seed(42)

    features = np.column_stack([
        # cmd_length: short commands
        np.random.normal(50, 20, n_samples),
        # special_char_ratio: low
        np.random.beta(1, 10, n_samples),
        # has_base64: rarely
        np.random.choice(
            [0, 1], n_samples, p=[0.98, 0.02]
        ),
        # has_download: rarely
        np.random.choice(
            [0, 1], n_samples, p=[0.99, 0.01]
        ),
        # name_length: moderate
        np.random.normal(12, 4, n_samples),
        # name_entropy: low-medium
        np.random.normal(3, 0.5, n_samples),
        # parent_risk: low
        np.random.choice(
            [0, 1], n_samples, p=[0.95, 0.05]
        ),
        # severity: low
        np.random.beta(1, 5, n_samples)
    ])

    return np.abs(features)
