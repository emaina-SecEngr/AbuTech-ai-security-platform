"""
Layer 2 — ML Processing
Security Event Sequence Builder

WHY THIS FILE EXISTS:
    LSTM requires sequences not individual events.
    This module converts the stream of individual
    DataAccessEvent and IamEvent objects into
    sequences grouped by accessor for LSTM input.

    Think of it as:
    Individual events = words
    Sequences = sentences
    LSTM = the model that understands sentences

THE 10-FEATURE VECTOR:
    Every security event reduced to 10 numbers.
    Same 10 numbers regardless of source:
    S3, RDS, Snowflake, SharePoint, Okta, Entra ID.
    This unified representation allows one LSTM
    to learn from all data sources simultaneously.

    [0] timestamp_norm      Time of day 0.0-1.0
    [1] user_risk_score     Cumulative risk 0.0-1.0
    [2] action_encoded      Operation type 0.0-1.0
    [3] sensitivity_level   Data sensitivity 0.0-1.0
    [4] volume_norm         Data volume 0.0-1.0
    [5] failure_count_norm  Recent failures 0.0-1.0
    [6] geo_velocity        Location shift 0.0-1.0
    [7] auth_strength       MFA strength 0.0-1.0
    [8] path_entropy        Resource diversity 0.0-1.0
    [9] accessor_type_enc   Who is accessing 0.0-1.0

TWO WINDOW SIZES:
    Kill chain window:        10-50 events (hours)
    Slow exfiltration window: 30-90 events (weeks)

    Based on your research answer:
    Temporal scale mismatch requires
    separate models per time scale.

RESEARCH CONTEXT:
    This implements the feature engineering
    described in the paper methodology:
    "A 10-dimensional security event embedding
     for temporal sequence anomaly detection
     in heterogeneous enterprise telemetry"
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# FEATURE ENCODING CONSTANTS
#
# WHY THESE VALUES:
# All features normalized to 0.0-1.0 range.
# Required for LSTM gradient stability.
# Values outside 0-1 cause vanishing gradients.
# ============================================================

# Action type encoding
# Higher value = higher risk operation
ACTION_ENCODING = {
    "read":    0.2,
    "list":    0.3,   # enumeration risk
    "write":   0.5,
    "update":  0.5,
    "schema":  0.6,   # structural change
    "delete":  0.8,   # destructive
    "export":  0.7,   # bulk extraction
    "unknown": 0.4
}

# Sensitivity level encoding
# Maps your SensitivityLabel enum to numeric
SENSITIVITY_ENCODING = {
    "NONE":    0.0,
    "UNKNOWN": 0.2,
    "PII":     0.6,
    "PHI":     0.7,
    "PCI":     0.8,
    "PFI":     0.7,
    "MIXED":   0.9
}

# Authentication strength encoding
# Based on NIST 800-63B assurance levels
AUTH_ENCODING = {
    "none":             0.0,  # no auth
    "password_only":    0.2,  # weakest
    "sms":              0.3,  # OTP via SMS
    "email":            0.3,  # OTP via email
    "totp":             0.6,  # authenticator app
    "push":             0.6,  # push notification
    "hardware":         0.8,  # hardware token
    "fido2":            0.9,  # FIDO2/WebAuthn
    "conditional_access": 0.7,
    "modern":           0.6,
    "legacy":           0.1,  # cannot enforce MFA
    "unknown":          0.3
}

# Accessor type encoding
# Your key insight: same event means different
# things for different accessor types
ACCESSOR_TYPE_ENCODING = {
    "human":           0.5,   # highest scrutiny
    "service_account": 0.4,
    "etl_process":     0.3,   # expected bulk access
    "application":     0.3,
    "api_client":      0.4,
    "replication":     0.2,   # expected continuous
    "backup":          0.2,   # expected bulk, scheduled
    "privileged":      0.9,   # highest risk
    "unknown":         0.6    # unknown = elevated
}

# Maximum values for normalization
MAX_ROWS_ACCESSED = 10_000_000   # 10M rows
MAX_BYTES_ACCESSED = 1_000_000_000  # 1GB
MAX_FAILURE_COUNT = 100
MAX_GEO_DISTANCE_KM = 20_000     # half earth circumference
MAX_PATH_ENTROPY = 5.0           # max Shannon entropy


# ============================================================
# SEQUENCE WINDOW CONFIGURATIONS
#
# Based on your research answer:
# Two separate window sizes for two detection goals.
# Temporal scale mismatch requires separate models.
# ============================================================

KILL_CHAIN_WINDOW = 20
# 20 events spanning minutes to hours
# Detects: active breach, lateral movement
# Model: smaller, faster, real-time scoring

SLOW_EXFIL_WINDOW = 60
# 60 events spanning days to weeks
# Detects: gradual exfiltration, privilege creep
# Model: larger, slower, batch scoring


@dataclass
class SecurityEventVector:
    """
    10-dimensional numeric representation
    of a single security event.

    WHY A DATACLASS:
    Keeps the feature names with the values.
    Makes code readable and debuggable.
    Research paper can reference feature names.

    All values normalized to 0.0-1.0 range.
    Required for LSTM gradient stability.
    """
    timestamp_norm: float = 0.0
    user_risk_score: float = 0.0
    action_encoded: float = 0.0
    sensitivity_level: float = 0.0
    volume_norm: float = 0.0
    failure_count_norm: float = 0.0
    geo_velocity: float = 0.0
    auth_strength: float = 0.0
    path_entropy: float = 0.0
    accessor_type_enc: float = 0.0

    # Metadata — not in feature vector
    # Used for investigation reports only
    accessor_identity: str = ""
    event_time: str = ""
    source_system: str = ""
    raw_risk_score: float = 0.0

    def to_numpy(self) -> np.ndarray:
        """
        Convert to numpy array for LSTM input.
        Returns exactly 10 features.
        Order matches LSTM input layer exactly.
        """
        return np.array([
            self.timestamp_norm,
            self.user_risk_score,
            self.action_encoded,
            self.sensitivity_level,
            self.volume_norm,
            self.failure_count_norm,
            self.geo_velocity,
            self.auth_strength,
            self.path_entropy,
            self.accessor_type_enc
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        """Human-readable feature representation"""
        return {
            "timestamp_norm": self.timestamp_norm,
            "user_risk_score": self.user_risk_score,
            "action": self.action_encoded,
            "sensitivity": self.sensitivity_level,
            "volume": self.volume_norm,
            "failures": self.failure_count_norm,
            "geo_velocity": self.geo_velocity,
            "auth_strength": self.auth_strength,
            "path_entropy": self.path_entropy,
            "accessor_type": self.accessor_type_enc
        }


@dataclass
class EventSequence:
    """
    A sequence of security events from
    one accessor ready for LSTM input.

    Contains both the numeric matrix
    and the metadata for investigation.
    """
    accessor_identity: str = ""
    window_type: str = ""
    # "kill_chain" or "slow_exfil"

    events: list = field(default_factory=list)
    # List of SecurityEventVector objects

    sequence_start: str = ""
    sequence_end: str = ""
    sequence_length: int = 0

    # Pre-computed risk signals
    max_risk_in_sequence: float = 0.0
    avg_risk_in_sequence: float = 0.0
    risk_trend: str = ""
    # "increasing", "stable", "decreasing"

    def to_matrix(self) -> np.ndarray:
        """
        Convert to numpy matrix for LSTM input.
        Shape: (sequence_length, 10)
        Each row = one event vector.
        Each column = one feature.
        """
        if not self.events:
            return np.zeros((1, 10), dtype=np.float32)

        return np.stack([
            e.to_numpy() for e in self.events
        ])

    def get_attention_labels(self) -> list:
        """
        Return event labels for attention visualization.
        Used in investigation reports to explain
        which events drove the detection.

        SR 11-7 COMPLIANCE:
        This is the explainability mechanism.
        Attention weights + these labels =
        human-readable model explanation.
        """
        return [
            f"{e.event_time[:16]} "
            f"{e.source_system} "
            f"risk={e.raw_risk_score:.2f}"
            for e in self.events
        ]


class SequenceBuilder:
    """
    Builds security event sequences for LSTM input.

    Receives individual events from Layer 1 normalizers.
    Groups by accessor identity.
    Maintains sliding windows of events.
    Converts to numeric vectors.
    Returns EventSequence objects ready for LSTM.

    SLIDING WINDOW CONCEPT:
    Events arrive continuously.
    Window slides forward as new events arrive.
    Each new event creates a new sequence
    containing the last N events.

    Example (window=3):
    Event 1: [e1]
    Event 2: [e1, e2]
    Event 3: [e1, e2, e3] → SCORE THIS SEQUENCE
    Event 4: [e2, e3, e4] → SCORE THIS SEQUENCE
    Event 5: [e3, e4, e5] → SCORE THIS SEQUENCE

    DUAL WINDOW:
    Kill chain window:   last 20 events (hours)
    Slow exfil window:   last 60 events (weeks)
    Both maintained simultaneously per accessor.

    Usage:
        builder = SequenceBuilder()
        builder.add_data_access_event(event)
        builder.add_iam_event(iam_event)
        sequences = builder.get_sequences("jsmith")
    """

    def __init__(
        self,
        kill_chain_window: int = KILL_CHAIN_WINDOW,
        slow_exfil_window: int = SLOW_EXFIL_WINDOW
    ):
        self.kill_chain_window = kill_chain_window
        self.slow_exfil_window = slow_exfil_window

        # Event history per accessor
        # key: accessor_identity
        # value: list of SecurityEventVector
        self._event_history = defaultdict(list)

        # Failure counts per accessor
        # For failure_count_norm feature
        self._failure_counts = defaultdict(int)

        # Path history per accessor
        # For path_entropy feature
        self._path_history = defaultdict(list)

        # Last known location per accessor
        # For geo_velocity feature
        self._last_location = {}

        # Statistics
        self.events_ingested = 0
        self.sequences_built = 0

        logger.info(
            f"SequenceBuilder initialized "
            f"kill_chain_window={kill_chain_window} "
            f"slow_exfil_window={slow_exfil_window}"
        )

    # ============================================================
    # EVENT INGESTION
    # ============================================================

    def add_data_access_event(
        self,
        event
    ) -> Optional[dict]:
        """
        Ingest a DataAccessEvent and convert
        to SecurityEventVector.

        Called after Layer 1 normalizer produces
        a DataAccessEvent from S3, RDS, Snowflake,
        or SharePoint.

        Returns sequences if window is full.
        """
        if not event:
            return None

        try:
            vector = self._vectorize_data_event(event)
            if vector:
                self._add_to_history(
                    event.accessor_identity,
                    vector
                )
                self.events_ingested += 1
                return self._get_ready_sequences(
                    event.accessor_identity
                )
        except Exception as e:
            logger.debug(
                f"Data event vectorization failed: {e}"
            )
        return None

    def add_iam_event(
        self,
        iam_event
    ) -> Optional[dict]:
        """
        Ingest an IamEvent and convert
        to SecurityEventVector.

        Called after Layer 1 normalizer produces
        an IamEvent from Okta, Entra ID, CyberArk,
        SailPoint, AWS Secrets, or Azure KV.
        """
        if not iam_event:
            return None

        try:
            vector = self._vectorize_iam_event(
                iam_event
            )
            if vector:
                self._add_to_history(
                    iam_event.user,
                    vector
                )
                self.events_ingested += 1
                return self._get_ready_sequences(
                    iam_event.user
                )
        except Exception as e:
            logger.debug(
                f"IAM event vectorization failed: {e}"
            )
        return None

    # ============================================================
    # VECTORIZATION
    # ============================================================

    def _vectorize_data_event(
        self,
        event
    ) -> Optional[SecurityEventVector]:
        """
        Convert DataAccessEvent to 10-feature vector.

        THE 10 FEATURES (your research answer):
        1. timestamp_norm
        2. user_risk_score (from event)
        3. action_encoded (operation type)
        4. sensitivity_level (from PII classifier)
        5. volume_norm (rows or bytes)
        6. failure_count_norm (recent failures)
        7. geo_velocity (location shift)
        8. auth_strength (N/A for data → 0.5)
        9. path_entropy (resource diversity)
        10. accessor_type_enc (who is accessing)
        """
        try:
            accessor = event.accessor_identity or ""

            # [0] Timestamp normalization
            timestamp_norm = self._normalize_timestamp(
                event.event_time
            )

            # [1] User risk score from normalizer
            user_risk = float(
                event.risk_score or 0.0
            )

            # [2] Action type encoding
            action = ACTION_ENCODING.get(
                event.operation.value
                if hasattr(event.operation, 'value')
                else str(event.operation),
                0.4
            )

            # [3] Sensitivity level
            sensitivity = 0.0
            if event.finding:
                label = event.finding.sensitivity_label
                label_str = (
                    label.value
                    if hasattr(label, 'value')
                    else str(label)
                )
                sensitivity = SENSITIVITY_ENCODING.get(
                    label_str, 0.2
                )
            elif hasattr(event, 'risk_label'):
                # Use risk label as proxy
                risk_label = event.risk_label or ""
                sensitivity_map = {
                    "CRITICAL": 0.9,
                    "HIGH": 0.7,
                    "MEDIUM": 0.5,
                    "LOW": 0.3,
                    "UNKNOWN": 0.2
                }
                sensitivity = sensitivity_map.get(
                    risk_label, 0.2
                )

            # [4] Volume normalization
            volume = 0.0
            if hasattr(event, 'rows_accessed') and (
                event.rows_accessed
            ):
                volume = min(
                    1.0,
                    event.rows_accessed /
                    MAX_ROWS_ACCESSED
                )
            elif hasattr(event, 'bytes_accessed') and (
                event.bytes_accessed
            ):
                volume = min(
                    1.0,
                    event.bytes_accessed /
                    MAX_BYTES_ACCESSED
                )

            # [5] Failure count normalized
            failure_norm = min(
                1.0,
                self._failure_counts[accessor] /
                MAX_FAILURE_COUNT
            )

            # [6] Geo velocity
            # Not applicable for data events
            # Data events do not have geo context
            geo_vel = 0.0

            # [7] Auth strength
            # Not applicable for data events
            # Data layer events do not have MFA context
            auth = 0.5  # neutral

            # [8] Path entropy
            # Measures diversity of resources accessed
            path_entropy = self._calculate_path_entropy(
                accessor,
                event.data_path or ""
            )

            # [9] Accessor type encoding
            accessor_type_str = (
                event.accessor_type.value
                if hasattr(event.accessor_type, 'value')
                else str(event.accessor_type)
            )
            accessor_enc = ACCESSOR_TYPE_ENCODING.get(
                accessor_type_str, 0.6
            )

            return SecurityEventVector(
                timestamp_norm=timestamp_norm,
                user_risk_score=user_risk,
                action_encoded=action,
                sensitivity_level=sensitivity,
                volume_norm=volume,
                failure_count_norm=failure_norm,
                geo_velocity=geo_vel,
                auth_strength=auth,
                path_entropy=path_entropy,
                accessor_type_enc=accessor_enc,
                accessor_identity=accessor,
                event_time=event.event_time or "",
                source_system=event.source_system or "",
                raw_risk_score=user_risk
            )

        except Exception as e:
            logger.debug(
                f"Data vectorization error: {e}"
            )
            return None

    def _vectorize_iam_event(
        self,
        iam_event
    ) -> Optional[SecurityEventVector]:
        """
        Convert IamEvent to 10-feature vector.

        IAM events add geo and auth features
        that data access events lack.
        """
        try:
            accessor = iam_event.user or ""
            user_risk = float(
                iam_event.overall_risk_score or 0.0
            )

            # Timestamp
            timestamp_norm = self._normalize_timestamp(
                iam_event.timestamp
            )

            # Action from event type
            action = ACTION_ENCODING.get(
                iam_event.event_type, 0.4
            )

            # Sensitivity — IAM events access
            # identity data — medium sensitivity
            sensitivity = 0.4

            # Volume — minimal for auth events
            volume = 0.0

            # Failure count
            failure_norm = min(
                1.0,
                self._failure_counts[accessor] /
                MAX_FAILURE_COUNT
            )

            # Geo velocity from auth event
            geo_vel = 0.0
            if iam_event.auth_event:
                auth_ev = iam_event.auth_event
                if auth_ev.is_impossible_travel:
                    geo_vel = 1.0
                elif auth_ev.travel_distance_km:
                    geo_vel = min(
                        1.0,
                        auth_ev.travel_distance_km /
                        MAX_GEO_DISTANCE_KM
                    )

                # Failure tracking
                if (
                    hasattr(auth_ev, 'outcome') and
                    auth_ev.outcome == "failure"
                ):
                    self._failure_counts[accessor] += 1

            # Auth strength from IAM event
            auth = 0.5  # default
            if iam_event.auth_event:
                auth_ev = iam_event.auth_event
                if (
                    hasattr(auth_ev, 'auth') and
                    auth_ev.auth
                ):
                    method = (
                        auth_ev.auth.mfa_method or ""
                    ).lower()
                    auth = AUTH_ENCODING.get(
                        method, 0.3
                    )
                    if auth_ev.auth.mfa_used:
                        auth = max(auth, 0.6)

            # Path entropy — IAM events access apps
            path_entropy = 0.3  # moderate diversity

            # Accessor type — humans for auth events
            accessor_enc = ACCESSOR_TYPE_ENCODING.get(
                "human", 0.5
            )

            return SecurityEventVector(
                timestamp_norm=timestamp_norm,
                user_risk_score=user_risk,
                action_encoded=action,
                sensitivity_level=sensitivity,
                volume_norm=volume,
                failure_count_norm=failure_norm,
                geo_velocity=geo_vel,
                auth_strength=auth,
                path_entropy=path_entropy,
                accessor_type_enc=accessor_enc,
                accessor_identity=accessor,
                event_time=iam_event.timestamp or "",
                source_system=(
                    iam_event.source_system or ""
                ),
                raw_risk_score=user_risk
            )

        except Exception as e:
            logger.debug(
                f"IAM vectorization error: {e}"
            )
            return None

    # ============================================================
    # SEQUENCE BUILDING
    # ============================================================

    def _add_to_history(
        self,
        accessor: str,
        vector: SecurityEventVector
    ) -> None:
        """Add event vector to accessor history"""
        self._event_history[accessor].append(vector)

        # Keep only enough for slow exfil window
        max_history = self.slow_exfil_window * 2
        if len(
            self._event_history[accessor]
        ) > max_history:
            self._event_history[accessor] = (
                self._event_history[accessor][
                    -max_history:
                ]
            )

    def _get_ready_sequences(
        self,
        accessor: str
    ) -> dict:
        """
        Get sequences when windows are full.

        Returns both kill chain and slow exfil
        sequences when enough events accumulated.
        """
        history = self._event_history[accessor]
        result = {}

        # Kill chain sequence
        if len(history) >= self.kill_chain_window:
            kc_events = history[
                -self.kill_chain_window:
            ]
            kc_seq = self._build_sequence(
                accessor, kc_events, "kill_chain"
            )
            result["kill_chain"] = kc_seq
            self.sequences_built += 1

        # Slow exfil sequence
        if len(history) >= self.slow_exfil_window:
            se_events = history[
                -self.slow_exfil_window:
            ]
            se_seq = self._build_sequence(
                accessor, se_events, "slow_exfil"
            )
            result["slow_exfil"] = se_seq
            self.sequences_built += 1

        return result if result else None

    def _build_sequence(
        self,
        accessor: str,
        events: list,
        window_type: str
    ) -> EventSequence:
        """
        Build EventSequence from event list.
        Computes risk trend for investigation reports.
        """
        risk_scores = [
            e.user_risk_score for e in events
        ]

        # Risk trend analysis
        if len(risk_scores) >= 2:
            first_half = np.mean(
                risk_scores[:len(risk_scores)//2]
            )
            second_half = np.mean(
                risk_scores[len(risk_scores)//2:]
            )

            if second_half > first_half * 1.2:
                risk_trend = "increasing"
            elif second_half < first_half * 0.8:
                risk_trend = "decreasing"
            else:
                risk_trend = "stable"
        else:
            risk_trend = "stable"

        return EventSequence(
            accessor_identity=accessor,
            window_type=window_type,
            events=events,
            sequence_start=(
                events[0].event_time
                if events else ""
            ),
            sequence_end=(
                events[-1].event_time
                if events else ""
            ),
            sequence_length=len(events),
            max_risk_in_sequence=max(
                risk_scores, default=0.0
            ),
            avg_risk_in_sequence=float(
                np.mean(risk_scores)
                if risk_scores else 0.0
            ),
            risk_trend=risk_trend
        )

    def get_sequences(
        self,
        accessor: str
    ) -> dict:
        """
        Get current sequences for an accessor.
        Returns whatever is available.
        """
        return self._get_ready_sequences(
            accessor
        ) or {}

    def get_all_accessors(self) -> list:
        """Return all accessors with history"""
        return list(self._event_history.keys())

    # ============================================================
    # FEATURE CALCULATION HELPERS
    # ============================================================

    def _normalize_timestamp(
        self,
        event_time: str
    ) -> float:
        """
        Normalize timestamp to time-of-day 0.0-1.0.
        0.0 = midnight, 0.5 = noon, 1.0 = midnight

        WHY TIME OF DAY NOT ABSOLUTE TIME:
        LSTM needs to learn PATTERNS not specific times.
        "3am access" is suspicious regardless of date.
        Normalizing to time-of-day captures this.
        """
        try:
            if "T" in str(event_time):
                parts = str(event_time).split("T")
                time_parts = parts[1][:8].split(":")
                hour = int(time_parts[0])
                minute = int(time_parts[1]) if (
                    len(time_parts) > 1
                ) else 0
                return (hour * 60 + minute) / 1440.0
        except Exception:
            pass
        return 0.5

    def _calculate_path_entropy(
        self,
        accessor: str,
        current_path: str
    ) -> float:
        """
        Calculate Shannon entropy of paths accessed.

        WHY PATH ENTROPY MATTERS:
        Normal user: accesses same 5 paths daily
                     → LOW entropy (predictable)
        Attacker:    accesses many different paths
                     → HIGH entropy (reconnaissance)

        Reconnaissance phase of kill chain:
        Attacker explores the environment.
        Path entropy spikes before exfiltration.
        This feature captures that signal.

        YOUR 10TH FEATURE (Resource Entropy):
        Exactly what you described in Q1.
        "Resource Entropy (Path Diversity)"
        """
        if current_path:
            self._path_history[accessor].append(
                current_path
            )
            self._path_history[accessor] = (
                self._path_history[accessor][-50:]
            )

        paths = self._path_history.get(accessor, [])
        if not paths:
            return 0.0

        # Count frequency of each unique path
        path_counts = {}
        for p in paths:
            path_counts[p] = path_counts.get(p, 0) + 1

        # Shannon entropy
        total = len(paths)
        entropy = 0.0
        for count in path_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return min(1.0, entropy / MAX_PATH_ENTROPY)

    # ============================================================
    # TRAINING DATA GENERATION
    # ============================================================

    def generate_normal_sequences(
        self,
        n_sequences: int = 500,
        window_size: int = KILL_CHAIN_WINDOW
    ) -> np.ndarray:
        """
        Generate synthetic normal sequences
        for unsupervised LSTM training.

        BASED ON YOUR RESEARCH ANSWER Q3:
        Primary model trains on NORMAL sequences only.
        This generates those normal sequences.

        Normal sequence characteristics:
        - Low risk scores (0.0-0.3)
        - Consistent access patterns
        - Business hours access
        - No geo velocity
        - Strong auth methods
        - Low path entropy
        - Expected accessor types
        """
        np.random.seed(42)
        sequences = []

        for _ in range(n_sequences):
            sequence = []

            # Normal sequences have consistent patterns
            base_risk = np.random.uniform(0.0, 0.25)
            base_hour = np.random.uniform(0.55, 0.95)
            # Business hours: 13:00-23:00 UTC
            # = 0.54 to 0.96 normalized

            for _ in range(window_size):
                event = [
                    # timestamp: business hours
                    base_hour + np.random.normal(
                        0, 0.05
                    ),
                    # risk score: low and stable
                    base_risk + np.random.normal(
                        0, 0.05
                    ),
                    # action: mostly reads
                    np.random.choice(
                        [0.2, 0.2, 0.2, 0.3, 0.5],
                        p=[0.5, 0.2, 0.1, 0.1, 0.1]
                    ),
                    # sensitivity: low to medium
                    np.random.choice(
                        [0.0, 0.2, 0.6],
                        p=[0.4, 0.4, 0.2]
                    ),
                    # volume: small
                    np.random.beta(1, 5),
                    # failures: rare
                    np.random.beta(1, 20),
                    # geo velocity: none
                    np.random.beta(1, 20),
                    # auth: strong
                    np.random.choice(
                        [0.6, 0.7, 0.8],
                        p=[0.5, 0.3, 0.2]
                    ),
                    # path entropy: low (familiar paths)
                    np.random.beta(1, 5),
                    # accessor type: known
                    np.random.choice(
                        [0.2, 0.3, 0.4, 0.5],
                        p=[0.3, 0.3, 0.2, 0.2]
                    )
                ]

                # Clip all values to 0-1 range
                event = [
                    max(0.0, min(1.0, v))
                    for v in event
                ]
                sequence.append(event)

            sequences.append(sequence)

        return np.array(sequences, dtype=np.float32)

    def generate_attack_sequences(
        self,
        n_sequences: int = 100,
        window_size: int = KILL_CHAIN_WINDOW,
        attack_type: str = "kill_chain"
    ) -> np.ndarray:
        """
        Generate synthetic attack sequences
        for supervised secondary LSTM.

        BASED ON YOUR RESEARCH ANSWER Q3:
        Secondary model uses labeled sequences.
        This generates attack examples.

        TWO ATTACK TYPES:

        kill_chain: events escalate rapidly
            Low risk events → high risk events
            Geo velocity spike
            Auth method weakens
            Path entropy increases

        slow_exfil: events consistent but long
            Stable moderate risk
            Consistent volume
            No single event stands out
            Duration is the signal
        """
        np.random.seed(123)
        sequences = []

        for _ in range(n_sequences):
            sequence = []

            if attack_type == "kill_chain":
                for step in range(window_size):
                    # Risk escalates through sequence
                    progress = step / window_size
                    event = [
                        # Off-hours access
                        np.random.uniform(0.0, 0.4),
                        # Risk increases
                        0.1 + (progress * 0.8),
                        # Actions escalate
                        0.2 + (progress * 0.6),
                        # Sensitivity increases
                        0.2 + (progress * 0.7),
                        # Volume spikes
                        progress * 0.9,
                        # Failures early
                        max(0, 0.5 - progress) * 0.8,
                        # Geo velocity spike
                        min(
                            1.0,
                            progress * 1.5
                        ) if step > 3 else 0.0,
                        # Auth weakens
                        max(0.1, 0.8 - progress * 0.7),
                        # Path entropy increases
                        min(1.0, progress * 1.2),
                        # Accessor type changes
                        0.5 + (progress * 0.4)
                    ]
                    event = [
                        max(0.0, min(1.0, v))
                        for v in event
                    ]
                    sequence.append(event)

            elif attack_type == "slow_exfil":
                # Consistent moderate access
                # Volume very slowly increasing
                base_volume = 0.1
                for step in range(window_size):
                    # Small volume increase per step
                    current_volume = min(
                        1.0,
                        base_volume + (
                            step * 0.01
                        )
                    )
                    event = [
                        # Normal business hours
                        np.random.uniform(0.5, 0.9),
                        # Low stable risk
                        np.random.uniform(0.1, 0.3),
                        # Reads only
                        0.2,
                        # Sensitive data
                        0.6,
                        # Slowly increasing volume
                        current_volume,
                        # No failures
                        0.0,
                        # No geo movement
                        0.0,
                        # Normal auth
                        0.6,
                        # Same paths every day
                        np.random.uniform(0.1, 0.2),
                        # Service account
                        0.4
                    ]
                    event = [
                        max(0.0, min(1.0, v))
                        for v in event
                    ]
                    sequence.append(event)

            sequences.append(sequence)

        return np.array(sequences, dtype=np.float32)

    def get_statistics(self) -> dict:
        return {
            "events_ingested": self.events_ingested,
            "sequences_built": self.sequences_built,
            "accessors_tracked": len(
                self._event_history
            ),
            "kill_chain_window": (
                self.kill_chain_window
            ),
            "slow_exfil_window": (
                self.slow_exfil_window
            )
        }