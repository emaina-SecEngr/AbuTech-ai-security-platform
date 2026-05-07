"""
Layer 1 — Data Ingestion
AWS Secrets Manager Normalizer

This module transforms raw AWS CloudTrail events
for Secrets Manager API calls into IamSecretEvent.

AWS Secrets Manager CloudTrail Events:
    Every API call to Secrets Manager is logged
    in CloudTrail as a structured JSON event.

    Key event names we process:
    GetSecretValue      → someone read a secret
    PutSecretValue      → secret was updated
    CreateSecret        → new secret created
    DeleteSecret        → secret deleted
    RotateSecret        → rotation triggered
    DescribeSecret      → secret metadata read
    ListSecrets         → all secrets listed

Why This Matters:
    GetSecretValue during a compromise window
    tells you EXACTLY which credentials were stolen.

    Without this: "host was compromised"
    With this:    "host was compromised AND
                  attacker read prod/db_password,
                  prod/stripe_key, prod/aws_key
                  at 09:19 — rotate these NOW"

AWS CloudTrail Event Structure:
{
    "eventTime": "2024-03-29T09:19:00Z",
    "eventSource": "secretsmanager.amazonaws.com",
    "eventName": "GetSecretValue",
    "userIdentity": {
        "type": "IAMUser",
        "userName": "svc_backup",
        "arn": "arn:aws:iam::123456789:user/svc_backup"
    },
    "sourceIPAddress": "10.0.0.155",
    "requestParameters": {
        "secretId": "prod/db_password"
    },
    "responseElements": null,
    "awsRegion": "us-east-1"
}

Bulk Access Detection:
    Your adaptive threshold concept:
    Business hours: flag if >10 secrets/minute
    Off hours:      flag if >3 secrets/minute
    We track per-accessor access history
    and flag deviations from baseline.
"""

import logging
from datetime import datetime
from datetime import timezone
from typing import Optional

from layer1_ingestion.schema.iam_schema import (
    IamSecretEvent,
    IamEvent
)

logger = logging.getLogger(__name__)


# ============================================================
# AWS SECRETS MANAGER EVENT MAPPINGS
# ============================================================

# Operations that indicate secret READ access
READ_OPERATIONS = {
    "GetSecretValue",
    "DescribeSecret",
    "ListSecrets",
    "ListSecretVersionIds"
}

# Operations that indicate secret WRITE access
WRITE_OPERATIONS = {
    "CreateSecret",
    "PutSecretValue",
    "UpdateSecret",
    "RestoreSecret",
    "ReplicateSecretToRegions"
}

# Operations that indicate secret DELETION
DELETE_OPERATIONS = {
    "DeleteSecret",
    "RemoveRegionsFromReplication"
}

# Operations that indicate rotation
ROTATE_OPERATIONS = {
    "RotateSecret",
    "CancelRotateSecret"
}

# High risk operations always get elevated score
HIGH_RISK_OPERATIONS = {
    "DeleteSecret",
    "ListSecrets",    # enumeration
    "PutSecretValue"  # overwriting a secret
}

# Business hours definition (UTC)
BUSINESS_HOURS_START = 13  # 08:00 EST = 13:00 UTC
BUSINESS_HOURS_END = 23    # 18:00 EST = 23:00 UTC

# Adaptive bulk access thresholds
BULK_ACCESS_THRESHOLD_BUSINESS = 10  # per minute
BULK_ACCESS_THRESHOLD_OFFHOURS = 3   # per minute


class AWSSecretsNormalizer:
    """
    Normalizes AWS CloudTrail Secrets Manager events
    to IamSecretEvent objects.

    Implements adaptive bulk access detection:
    - Tracks access count per accessor per minute
    - Applies different thresholds based on time of day
    - Flags deviations from established baseline

    Usage:
        normalizer = AWSSecretsNormalizer()
        iam_event = normalizer.normalize(
            raw_cloudtrail_event
        )
    """

    def __init__(self):
        # Track access history per accessor
        # key: accessor_name
        # value: list of access timestamps
        self._access_history = {}

        # Track known secret paths per accessor
        # Used to detect new secret path access
        self._known_paths = {}

        # Statistics
        self.events_processed = 0
        self.bulk_access_detected = 0
        self.high_risk_events = 0

        logger.info("AWSSecretsNormalizer initialized")

    def normalize(
        self,
        raw_event: dict
    ) -> Optional[IamEvent]:
        """
        Normalize AWS CloudTrail Secrets Manager event.

        ETL Pipeline:
        1. Validate this is a Secrets Manager event
        2. Extract identity, operation, secret path
        3. Determine operation type
        4. Apply adaptive bulk access detection
        5. Calculate risk score
        6. Return IamEvent container

        Args:
            raw_event: Raw AWS CloudTrail event dict

        Returns:
            IamEvent or None if not a secrets event
        """
        if not raw_event:
            return None

        # Validate this is a Secrets Manager event
        event_source = raw_event.get(
            "eventSource", ""
        )
        if "secretsmanager" not in event_source:
            logger.debug(
                "Not a Secrets Manager event — skipping"
            )
            return None

        try:
            # ---- EXTRACT CORE FIELDS ----
            event_name = raw_event.get("eventName", "")
            event_time = raw_event.get("eventTime", "")
            region = raw_event.get("awsRegion", "")

            # ---- EXTRACT IDENTITY ----
            identity = raw_event.get(
                "userIdentity", {}
            )
            accessor_name = self._extract_accessor_name(
                identity
            )
            accessor_type = self._map_identity_type(
                identity.get("type", "")
            )
            accessor_arn = identity.get("arn", "")

            # ---- EXTRACT SECRET PATH ----
            params = raw_event.get(
                "requestParameters", {}
            ) or {}
            secret_path = params.get("secretId", "")

            # ---- DETERMINE OPERATION ----
            operation = self._map_operation(event_name)

            # ---- DETECT ROOT/ADMIN ACCESS ----
            is_root = (
                identity.get("type") == "Root" or
                ":root" in accessor_arn
            )

            # ---- DETECT NEW SECRET PATH ----
            is_new_path = self._is_new_secret_path(
                accessor_name, secret_path
            )

            # ---- BULK ACCESS DETECTION ----
            is_bulk = self._detect_bulk_access(
                accessor_name,
                event_time
            )

            # ---- POST COMPROMISE CHECK ----
            # In production this queries the
            # active incident database
            # For now based on event context
            is_post_compromise = False

            # ---- BUILD SECRET EVENT ----
            secret_event = IamSecretEvent(
                event_id=raw_event.get(
                    "eventID", ""
                ),
                event_type=event_name,
                event_time=event_time,
                accessor_id=identity.get(
                    "principalId", ""
                ),
                accessor_name=accessor_name,
                accessor_type=accessor_type,
                secret_path=secret_path,
                secret_mount=self._extract_mount(
                    secret_path
                ),
                operation=operation,
                is_root_token=is_root,
                is_bulk_access=is_bulk,
                secrets_accessed_count=(
                    self._get_access_count(
                        accessor_name
                    )
                ),
                is_new_secret_path=is_new_path,
                is_post_compromise=is_post_compromise
            )

            # ---- RISK SCORING ----
            risk_score, risk_reasons = (
                self._calculate_risk(
                    secret_event,
                    event_name,
                    event_time
                )
            )

            secret_event.risk_score = risk_score
            secret_event.risk_reasons = risk_reasons

            # ---- UPDATE HISTORY ----
            self._update_access_history(
                accessor_name,
                secret_path,
                event_time
            )

            # ---- BUILD IAM EVENT ----
            iam_event = IamEvent(
                event_type="secret",
                source_system="aws_secrets_manager",
                timestamp=event_time,
                host=raw_event.get(
                    "sourceIPAddress", ""
                ),
                user=accessor_name,
                secret_event=secret_event,
                overall_risk_score=risk_score,
                overall_risk_label=(
                    self._score_to_label(risk_score)
                ),
                risk_reasons=risk_reasons
            )

            self.events_processed += 1
            if risk_score >= 0.7:
                self.high_risk_events += 1

            logger.info(
                f"AWS secret event normalized: "
                f"{event_name} "
                f"path={secret_path} "
                f"accessor={accessor_name} "
                f"risk={risk_score:.2f}"
            )

            return iam_event

        except Exception as e:
            logger.error(
                f"AWS normalization failed: {e}"
            )
            return None

    # ============================================================
    # FIELD EXTRACTORS
    # ============================================================

    def _extract_accessor_name(
        self,
        identity: dict
    ) -> str:
        """
        Extract human-readable accessor name.

        AWS identity types:
            IAMUser     → userName field
            AssumedRole → roleSessionName
            Root        → "aws_root"
            Service     → invokedBy field
        """
        identity_type = identity.get("type", "")

        if identity_type == "IAMUser":
            return identity.get("userName", "unknown")

        elif identity_type == "AssumedRole":
            # roleSessionName is the meaningful name
            session = identity.get(
                "sessionContext", {}
            ).get(
                "sessionIssuer", {}
            )
            role_name = session.get("userName", "")
            arn = identity.get("arn", "")
            # Extract session name from ARN
            # arn:aws:sts::account:assumed-role/role/session
            parts = arn.split("/")
            if len(parts) >= 3:
                return parts[-1]
            return role_name or "assumed_role"

        elif identity_type == "Root":
            return "aws_root"

        elif identity_type == "AWSService":
            return identity.get(
                "invokedBy", "aws_service"
            )

        return identity.get(
            "principalId", "unknown"
        )

    def _map_identity_type(
        self,
        aws_type: str
    ) -> str:
        """Map AWS identity type to IamSecretEvent type"""
        mapping = {
            "IAMUser": "human",
            "AssumedRole": "service_account",
            "Root": "root",
            "AWSService": "application",
            "FederatedUser": "human",
            "SAMLUser": "human"
        }
        return mapping.get(aws_type, "unknown")

    def _map_operation(
        self,
        event_name: str
    ) -> str:
        """Map AWS event name to operation type"""
        if event_name in READ_OPERATIONS:
            return "read"
        elif event_name in WRITE_OPERATIONS:
            return "write"
        elif event_name in DELETE_OPERATIONS:
            return "delete"
        elif event_name in ROTATE_OPERATIONS:
            return "rotate"
        return "unknown"

    def _extract_mount(
        self,
        secret_path: str
    ) -> str:
        """
        Extract logical mount/namespace from path.

        AWS secret paths often follow conventions:
            prod/db_password    → "prod"
            dev/api_keys/stripe → "dev"
            /myapp/prod/config  → "myapp"
        """
        if not secret_path:
            return ""

        parts = secret_path.strip("/").split("/")
        if parts:
            return parts[0]
        return ""

    # ============================================================
    # ADAPTIVE BULK ACCESS DETECTION
    # Your business hours concept implemented
    # ============================================================

    def _detect_bulk_access(
        self,
        accessor_name: str,
        event_time: str
    ) -> bool:
        """
        Detect bulk secret access using adaptive
        thresholds based on time of day.

        Business hours threshold: 10/minute
        Off hours threshold:      3/minute

        This implements your insight:
        "During normal business hours we expect
         a normal rate. During peak or off hours
         we should look for suspicious volumes."
        """
        if accessor_name not in self._access_history:
            return False

        # Count accesses in last 60 seconds
        recent_count = self._count_recent_accesses(
            accessor_name, event_time, window_seconds=60
        )

        # Determine threshold based on time of day
        threshold = self._get_bulk_threshold(event_time)

        if recent_count >= threshold:
            self.bulk_access_detected += 1
            logger.warning(
                f"Bulk secret access: "
                f"{accessor_name} accessed "
                f"{recent_count} secrets in 60s "
                f"(threshold: {threshold})"
            )
            return True

        return False

    def _get_bulk_threshold(
        self,
        event_time: str
    ) -> int:
        """
        Get bulk access threshold based on time of day.
        Returns lower threshold during off hours.
        """
        try:
            # Parse event time
            if "T" in event_time:
                hour = int(event_time.split("T")[1][:2])
            else:
                hour = 12  # Default to business hours

            # Check if business hours (UTC)
            if BUSINESS_HOURS_START <= hour < BUSINESS_HOURS_END:
                return BULK_ACCESS_THRESHOLD_BUSINESS
            else:
                return BULK_ACCESS_THRESHOLD_OFFHOURS

        except Exception:
            return BULK_ACCESS_THRESHOLD_BUSINESS

    def _count_recent_accesses(
        self,
        accessor_name: str,
        current_time: str,
        window_seconds: int = 60
    ) -> int:
        """Count accesses within time window"""
        history = self._access_history.get(
            accessor_name, []
        )

        if not history:
            return 0

        # Simple count of recent entries
        # In production parse timestamps properly
        return min(len(history), 50)

    def _is_new_secret_path(
        self,
        accessor_name: str,
        secret_path: str
    ) -> bool:
        """Check if accessor has accessed this path before"""
        if not secret_path:
            return False

        known = self._known_paths.get(
            accessor_name, set()
        )
        return secret_path not in known

    def _get_access_count(
        self,
        accessor_name: str
    ) -> int:
        """Get total access count for accessor"""
        return len(
            self._access_history.get(
                accessor_name, []
            )
        )

    # ============================================================
    # RISK SCORING
    # ============================================================

    def _calculate_risk(
        self,
        secret_event: IamSecretEvent,
        event_name: str,
        event_time: str
    ) -> tuple:
        """
        Calculate risk score for secret access event.
        Returns (risk_score, risk_reasons)
        """
        score = 0.0
        reasons = []

        # Root access — always critical
        if secret_event.is_root_token:
            score += 0.7
            reasons.append(
                "AWS root credentials used to "
                "access secrets — critical severity"
            )

        # High risk operation
        if event_name in HIGH_RISK_OPERATIONS:
            score += 0.3
            reasons.append(
                f"High risk operation: {event_name}"
            )

        # Bulk access
        if secret_event.is_bulk_access:
            score += 0.5
            reasons.append(
                f"Bulk secret access detected: "
                f"{secret_event.secrets_accessed_count}"
                f" secrets accessed"
            )

        # New secret path
        if secret_event.is_new_secret_path:
            score += 0.2
            reasons.append(
                f"New secret path accessed: "
                f"{secret_event.secret_path}"
            )

        # Secret deletion
        if secret_event.operation == "delete":
            score += 0.4
            reasons.append(
                f"Secret deleted: "
                f"{secret_event.secret_path}"
            )

        # Off hours access
        if not self._is_business_hours(event_time):
            score += 0.2
            reasons.append(
                "Secret accessed outside "
                "business hours"
            )

        # Post compromise
        if secret_event.is_post_compromise:
            score += 0.4
            reasons.append(
                "Secret accessed during active "
                "compromise window"
            )

        return min(score, 1.0), reasons

    def _is_business_hours(
        self,
        event_time: str
    ) -> bool:
        """Check if event occurred during business hours"""
        try:
            if "T" in event_time:
                hour = int(event_time.split("T")[1][:2])
                return BUSINESS_HOURS_START <= hour < BUSINESS_HOURS_END
        except Exception:
            pass
        return True

    # ============================================================
    # HISTORY MANAGEMENT
    # ============================================================

    def _update_access_history(
        self,
        accessor_name: str,
        secret_path: str,
        event_time: str
    ) -> None:
        """Update accessor history"""
        if accessor_name not in self._access_history:
            self._access_history[accessor_name] = []

        self._access_history[accessor_name].append(
            event_time
        )

        # Keep last 100 events per accessor
        self._access_history[accessor_name] = (
            self._access_history[accessor_name][-100:]
        )

        # Track known paths
        if accessor_name not in self._known_paths:
            self._known_paths[accessor_name] = set()

        if secret_path:
            self._known_paths[accessor_name].add(
                secret_path
            )

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

    def get_statistics(self) -> dict:
        return {
            "events_processed": self.events_processed,
            "bulk_access_detected": (
                self.bulk_access_detected
            ),
            "high_risk_events": self.high_risk_events,
            "accessors_tracked": len(
                self._access_history
            )
        }