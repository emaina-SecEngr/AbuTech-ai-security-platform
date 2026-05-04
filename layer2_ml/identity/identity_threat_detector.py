"""
Layer 2 — ML Processing
Identity Threat Detector

This module detects identity-based threats using
behavioral analytics and anomaly detection.

Three Detection Engines:

1. IMPOSSIBLE TRAVEL DETECTOR
   Uses Haversine distance + time elapsed
   to identify physically impossible logins.
   Your radius concept — distance too large
   for the elapsed time = credential compromise.

2. MFA BYPASS DETECTOR
   Identifies when MFA is circumvented.
   ATT&CK T1621 MFA Request Generation
   ATT&CK T1556 Modify Authentication Process

3. BEHAVIORAL ANOMALY DETECTOR
   Scores each authentication against the
   user's historical baseline.
   Multiple small anomalies = higher risk
   than any single signal alone.

Why Not Just Rules:
   Rules miss novel attacks.
   An attacker who knows your rules can
   evade them by staying just below thresholds.
   
   Behavioral analytics catches:
   "This is unusual FOR THIS SPECIFIC USER"
   not just "this is unusual in general"

Production Upgrade Path:
   Current: Rule-based behavioral scoring
   Next:    Isolation Forest for unsupervised
            anomaly detection
   Future:  LSTM for sequence-based detection
            GNN for graph-based identity patterns
"""

import logging
import math
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Optional

from layer1_ingestion.schema.iam_schema import (
    IamEvent,
    IamAuthEvent,
    IamPrivilegedEvent,
    IamSecretEvent
)

logger = logging.getLogger(__name__)


# ============================================================
# DETECTION RESULT
# ============================================================

@dataclass
class IdentityThreatResult:
    """
    Result from identity threat detection.

    Carries all detection findings back to
    the Layer 2 router for routing to
    Layer 3 knowledge graph.
    """
    # Overall verdict
    is_threat: bool = False
    risk_score: float = 0.0
    risk_label: str = "UNKNOWN"
    confidence: str = "LOW"

    # Specific threat types detected
    impossible_travel: bool = False
    mfa_bypassed: bool = False
    new_country: bool = False
    new_device: bool = False
    credential_stuffing: bool = False
    privilege_escalation: bool = False
    secret_harvesting: bool = False

    # Evidence
    risk_reasons: list = field(default_factory=list)
    attack_techniques: list = field(
        default_factory=list
    )

    # Geographic details
    travel_distance_km: Optional[float] = None
    travel_speed_kmh: Optional[float] = None
    origin_country: str = ""

    # User context
    user_email: str = ""
    source_system: str = ""

    # Timing
    scored_at: str = ""

    def to_dict(self) -> dict:
        return {
            "is_threat": self.is_threat,
            "risk_score": self.risk_score,
            "risk_label": self.risk_label,
            "confidence": self.confidence,
            "impossible_travel": self.impossible_travel,
            "mfa_bypassed": self.mfa_bypassed,
            "new_country": self.new_country,
            "privilege_escalation": (
                self.privilege_escalation
            ),
            "secret_harvesting": self.secret_harvesting,
            "risk_reasons": self.risk_reasons,
            "attack_techniques": self.attack_techniques,
            "travel_distance_km": self.travel_distance_km,
            "origin_country": self.origin_country,
            "user_email": self.user_email,
            "scored_at": self.scored_at
        }


# ============================================================
# USER BEHAVIORAL BASELINE
# ============================================================

@dataclass
class UserBaseline:
    """
    Behavioral baseline for a single user.

    Built up over time as the user authenticates.
    Anomalies are scored against this baseline.

    In production this is stored in Redis or
    a time-series database for persistence.
    For development we use in-memory storage.
    """
    user_email: str = ""

    # Known locations
    known_countries: set = field(default_factory=set)
    known_cities: set = field(default_factory=set)
    known_ips: set = field(default_factory=set)

    # Known devices
    known_devices: set = field(default_factory=set)

    # Known applications
    known_apps: set = field(default_factory=set)

    # Authentication patterns
    typical_hours: set = field(default_factory=set)
    # Hours of day (0-23) when user normally logs in
    total_authentications: int = 0
    failed_auth_count: int = 0

    # Last known location
    last_lat: Optional[float] = None
    last_lon: Optional[float] = None
    last_auth_time: Optional[str] = None
    last_country: str = ""

    # MFA usage
    mfa_always_used: bool = False
    mfa_usage_rate: float = 0.0

    def is_established(self) -> bool:
        """
        Check if baseline has enough data.
        Need at least 5 authentications to
        establish a meaningful baseline.
        """
        return self.total_authentications >= 5

    def update_from_auth(
        self,
        auth_event: IamAuthEvent
    ) -> None:
        """Update baseline from new authentication"""
        self.total_authentications += 1

        if auth_event.geo:
            if auth_event.geo.country_code:
                self.known_countries.add(
                    auth_event.geo.country_code
                )
            if auth_event.geo.city:
                self.known_cities.add(
                    auth_event.geo.city
                )
            if auth_event.geo.ip_address:
                self.known_ips.add(
                    auth_event.geo.ip_address
                )
            if auth_event.geo.latitude:
                self.last_lat = auth_event.geo.latitude
                self.last_lon = auth_event.geo.longitude

        if auth_event.target_app:
            self.known_apps.add(auth_event.target_app)

        if auth_event.outcome == "failure":
            self.failed_auth_count += 1

        self.last_auth_time = auth_event.event_time

        if auth_event.auth and auth_event.auth.mfa_used:
            total = self.total_authentications
            current_rate = self.mfa_usage_rate
            self.mfa_usage_rate = (
                (current_rate * (total - 1) + 1) / total
            )


class IdentityThreatDetector:
    """
    Detects identity-based threats from IAM events.

    Processes IamEvent objects from any IAM source
    and produces IdentityThreatResult with risk score
    and specific threat indicators.

    Usage:
        detector = IdentityThreatDetector()

        # Score an IAM event
        result = detector.score(iam_event)

        if result.impossible_travel:
            alert("Credential compromise detected")

        if result.risk_score >= 0.7:
            force_mfa_reauthentication()
    """

    def __init__(self):
        # User behavioral baselines
        self._baselines = {}
        # key: user_email → UserBaseline

        # Failed authentication tracking
        self._failed_auths = {}
        # key: user_email → list of timestamps

        # Statistics
        self.events_scored = 0
        self.threats_detected = 0

        logger.info("IdentityThreatDetector initialized")

    def score(
        self,
        iam_event: IamEvent
    ) -> IdentityThreatResult:
        """
        Score an IAM event for identity threats.

        Routes to appropriate detector based on
        event type:
            auth       → score_auth_event
            privileged → score_privileged_event
            secret     → score_secret_event
            governance → score_governance_event

        Args:
            iam_event: Normalized IAM event

        Returns:
            IdentityThreatResult with risk score
        """
        result = IdentityThreatResult(
            user_email=iam_event.user,
            source_system=iam_event.source_system,
            scored_at=self._now()
        )

        if iam_event.auth_event:
            result = self._score_auth_event(
                iam_event.auth_event, result
            )

        elif iam_event.privileged_event:
            result = self._score_privileged_event(
                iam_event.privileged_event, result
            )

        elif iam_event.secret_event:
            result = self._score_secret_event(
                iam_event.secret_event, result
            )

        # Final verdict
        result.is_threat = result.risk_score >= 0.5
        result.risk_label = self._score_to_label(
            result.risk_score
        )
        result.confidence = self._calculate_confidence(
            result
        )

        self.events_scored += 1
        if result.is_threat:
            self.threats_detected += 1

        return result

    # ============================================================
    # AUTH EVENT SCORING
    # ============================================================

    def _score_auth_event(
        self,
        auth_event: IamAuthEvent,
        result: IdentityThreatResult
    ) -> IdentityThreatResult:
        """
        Score authentication event for threats.

        Applies three detection engines:
        1. Impossible travel
        2. MFA bypass
        3. Behavioral anomaly
        """
        user = auth_event.user_email
        baseline = self._get_or_create_baseline(user)

        # ---- ENGINE 1: IMPOSSIBLE TRAVEL ----
        if auth_event.is_impossible_travel:
            result.impossible_travel = True
            result.risk_score += 0.5
            result.travel_distance_km = (
                auth_event.travel_distance_km
            )
            result.travel_speed_kmh = (
                auth_event.travel_speed_kmh
            )
            result.risk_reasons.append(
                f"Impossible travel: "
                f"{auth_event.travel_distance_km}km "
                f"at {auth_event.travel_speed_kmh}km/h"
            )
            result.attack_techniques.append(
                "T1110 Brute Force / Credential Stuffing"
            )

        # ---- ENGINE 2: MFA BYPASS ----
        mfa_bypass_detected = (
            self._detect_mfa_bypass(
                auth_event, baseline
            )
        )

        if mfa_bypass_detected:
            result.mfa_bypassed = True
            result.risk_score += 0.4
            result.risk_reasons.append(
                "MFA bypassed or not required "
                "when historically always used"
            )
            result.attack_techniques.append(
                "T1621 Multi-Factor Authentication "
                "Request Generation"
            )

        # ---- ENGINE 3: BEHAVIORAL ANOMALY ----
        anomaly_score = self._score_behavioral_anomaly(
            auth_event, baseline
        )
        result.risk_score += anomaly_score

        # New country
        if auth_event.is_new_country:
            result.new_country = True
            result.risk_score += 0.3
            result.origin_country = (
                auth_event.geo.country_name
                if auth_event.geo else ""
            )
            result.risk_reasons.append(
                f"New country: "
                f"{auth_event.geo.country_name if auth_event.geo else 'unknown'}"
            )

        # New device
        if auth_event.is_new_device:
            result.new_device = True
            result.risk_reasons.append(
                "Authentication from new device"
            )

        # Failed authentication
        if auth_event.outcome in [
            "failure", "denied"
        ]:
            failure_risk = self._score_failed_auth(
                user, auth_event.event_time
            )
            result.risk_score += failure_risk
            if failure_risk >= 0.3:
                result.credential_stuffing = True
                result.risk_reasons.append(
                    "Multiple failed authentications "
                    "detected — possible credential stuffing"
                )
                result.attack_techniques.append(
                    "T1110.004 Credential Stuffing"
                )

        # Update baseline with this event
        if auth_event.outcome == "success":
            baseline.update_from_auth(auth_event)

        result.risk_score = min(result.risk_score, 1.0)
        return result

    def _detect_mfa_bypass(
        self,
        auth_event: IamAuthEvent,
        baseline: UserBaseline
    ) -> bool:
        """
        Detect MFA bypass.

        MFA bypass is detected when:
        1. User historically always uses MFA
        2. Current authentication succeeded without MFA

        ATT&CK T1621, T1556
        """
        if not auth_event.auth:
            return False

        if auth_event.outcome != "success":
            return False

        # If user normally uses MFA (>80% of auths)
        # and this auth did not use MFA
        if (
            baseline.is_established() and
            baseline.mfa_usage_rate >= 0.8 and
            not auth_event.auth.mfa_used
        ):
            return True

        return False

    def _score_behavioral_anomaly(
        self,
        auth_event: IamAuthEvent,
        baseline: UserBaseline
    ) -> float:
        """
        Score behavioral anomaly against baseline.

        Returns additional risk score 0.0-0.4
        based on how far this event deviates
        from the user's normal behavior.
        """
        if not baseline.is_established():
            return 0.0

        anomaly_score = 0.0

        # New country
        if (
            auth_event.geo and
            auth_event.geo.country_code and
            auth_event.geo.country_code not in
            baseline.known_countries
        ):
            anomaly_score += 0.2

        # New city
        if (
            auth_event.geo and
            auth_event.geo.city and
            auth_event.geo.city not in
            baseline.known_cities
        ):
            anomaly_score += 0.1

        # New application access
        if (
            auth_event.target_app and
            auth_event.target_app not in
            baseline.known_apps
        ):
            anomaly_score += 0.1

        return min(anomaly_score, 0.4)

    def _score_failed_auth(
        self,
        user_email: str,
        timestamp: str
    ) -> float:
        """
        Score failed authentication attempts.

        Multiple failures = credential stuffing.
        5+ failures in 10 minutes = HIGH risk.
        """
        if user_email not in self._failed_auths:
            self._failed_auths[user_email] = []

        self._failed_auths[user_email].append(
            timestamp
        )

        # Count recent failures (last 10 minutes)
        recent_failures = len(
            self._failed_auths[user_email][-10:]
        )

        if recent_failures >= 10:
            return 0.5
        elif recent_failures >= 5:
            return 0.3
        elif recent_failures >= 3:
            return 0.1

        return 0.0

    # ============================================================
    # PRIVILEGED EVENT SCORING
    # ============================================================

    def _score_privileged_event(
        self,
        priv_event: IamPrivilegedEvent,
        result: IdentityThreatResult
    ) -> IdentityThreatResult:
        """
        Score privileged access event.

        CyberArk PAM events — admin credential usage.
        """
        result.user_email = priv_event.user_name

        # After hours privileged access
        if priv_event.is_after_hours:
            result.risk_score += 0.3
            result.risk_reasons.append(
                f"After-hours privileged access: "
                f"{priv_event.target_account}"
            )

        # New privileged account
        if priv_event.is_new_account:
            result.risk_score += 0.3
            result.privilege_escalation = True
            result.risk_reasons.append(
                f"First use of privileged account: "
                f"{priv_event.target_account}"
            )
            result.attack_techniques.append(
                "T1078 Valid Accounts"
            )

        # Concurrent privileged sessions
        if priv_event.is_concurrent_session:
            result.risk_score += 0.4
            result.risk_reasons.append(
                "Concurrent privileged sessions detected"
            )
            result.attack_techniques.append(
                "T1078.002 Domain Accounts"
            )

        # Credential retrieved but not used
        if priv_event.credential_retrieved_not_used:
            result.risk_score += 0.3
            result.risk_reasons.append(
                "Credential retrieved without "
                "establishing session — possible theft"
            )
            result.attack_techniques.append(
                "T1555 Credentials from Password Stores"
            )

        result.risk_score = min(result.risk_score, 1.0)
        return result

    # ============================================================
    # SECRET EVENT SCORING
    # ============================================================

    def _score_secret_event(
        self,
        secret_event: IamSecretEvent,
        result: IdentityThreatResult
    ) -> IdentityThreatResult:
        """
        Score Vault secret access event.

        HashiCorp Vault events — secret access.
        """
        result.user_email = secret_event.accessor_name

        # Root token usage — always critical
        if secret_event.is_root_token:
            result.risk_score += 0.7
            result.risk_reasons.append(
                "Vault root token used — "
                "critical severity"
            )
            result.attack_techniques.append(
                "T1552.001 Credentials in Files"
            )

        # Bulk secret access
        if secret_event.is_bulk_access:
            result.risk_score += 0.5
            result.secret_harvesting = True
            result.risk_reasons.append(
                f"Bulk secret access: "
                f"{secret_event.secrets_accessed_count}"
                f" secrets accessed"
            )
            result.attack_techniques.append(
                "T1555 Credentials from Password Stores"
            )

        # New secret path
        if secret_event.is_new_secret_path:
            result.risk_score += 0.2
            result.risk_reasons.append(
                f"New secret path accessed: "
                f"{secret_event.secret_path}"
            )

        # Post-compromise access
        if secret_event.is_post_compromise:
            result.risk_score += 0.4
            result.risk_reasons.append(
                "Secret accessed during active "
                "compromise window — likely exfiltrated"
            )

        result.risk_score = min(result.risk_score, 1.0)
        return result

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    def _get_or_create_baseline(
        self,
        user_email: str
    ) -> UserBaseline:
        """Get existing baseline or create new one"""
        if user_email not in self._baselines:
            self._baselines[user_email] = UserBaseline(
                user_email=user_email
            )
        return self._baselines[user_email]

    def _calculate_confidence(
        self,
        result: IdentityThreatResult
    ) -> str:
        """Calculate detection confidence"""
        signals = sum([
            result.impossible_travel,
            result.mfa_bypassed,
            result.new_country,
            result.credential_stuffing,
            result.privilege_escalation,
            result.secret_harvesting
        ])

        if signals >= 3:
            return "HIGH"
        elif signals >= 2:
            return "MEDIUM"
        elif signals >= 1:
            return "LOW"
        return "LOW"

    def _score_to_label(self, score: float) -> str:
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score > 0.0:
            return "LOW"
        return "UNKNOWN"

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )

    def get_statistics(self) -> dict:
        return {
            "events_scored": self.events_scored,
            "threats_detected": self.threats_detected,
            "users_tracked": len(self._baselines),
            "threat_rate": (
                self.threats_detected /
                max(self.events_scored, 1)
            )
        }

    def get_user_baseline(
        self,
        user_email: str
    ) -> Optional[UserBaseline]:
        """Get baseline for a specific user"""
        return self._baselines.get(user_email)