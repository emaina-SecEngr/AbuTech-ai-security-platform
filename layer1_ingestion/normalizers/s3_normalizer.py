"""
Layer 1 — Data Ingestion
AWS S3 Access Normalizer

WHY THIS FILE EXISTS:
    AWS S3 stores the most sensitive data
    in modern enterprises:
    Customer records, payment data,
    healthcare records, financial exports.

    Without this normalizer your platform
    is blind to S3 data access.

    With this normalizer:
    Every S3 access → DataAccessEvent
    PII classifier enriches automatically
    Knowledge graph connects to identity
    Layer 4 agents investigate automatically

TWO LOG SOURCES:
    1. CloudTrail S3 events (recommended)
       JSON format, centralized, all API calls
       We implement this one.

    2. S3 Server Access Logs
       HTTP level detail, range requests
       Reveals enumeration patterns
       Future enhancement.

KEY DETECTIONS:
    - Large object downloads (exfiltration)
    - Bucket enumeration (ListBucket)
    - Off-hours access
    - New accessor for bucket
    - Sensitive path patterns (/pii/, /customers/)
    - Public bucket access (misconfiguration)
    - Bulk operations (many objects quickly)

CLOUDTRAIL EVENT STRUCTURE:
{
    "eventTime": "2024-03-29T09:19:00Z",
    "eventName": "GetObject",
    "userIdentity": {
        "type": "IAMUser",
        "userName": "svc_backup",
        "arn": "arn:aws:iam::123:user/svc_backup",
        "principalId": "AIDABC123"
    },
    "requestParameters": {
        "bucketName": "prod-customer-data",
        "key": "customers/pii/2024-q1.csv"
    },
    "additionalEventData": {
        "bytesTransferredOut": 52428800,
        "bytesTransferredIn": 0
    },
    "sourceIPAddress": "10.0.0.155",
    "awsRegion": "us-east-1",
    "requestID": "request-uuid"
}

DRY PRINCIPLE:
    Identity extraction logic is shared
    with AWSSecretsNormalizer.
    Both normalizers extract from the same
    CloudTrail userIdentity structure.
    Change once → both benefit.
"""

import logging
from datetime import datetime
from datetime import timezone
from typing import Optional

from layer1_ingestion.schema.data_schema import (
    DataAccessEvent,
    AccessorType,
    DataStoreType,
    DataOperation,
    Environment,
    SensitivityLabel
)

logger = logging.getLogger(__name__)


# ============================================================
# S3 OPERATION MAPPINGS
#
# WHY MAP OPERATIONS:
# CloudTrail uses AWS API names.
# Your platform uses standard operation names.
# Mapping ensures consistent risk scoring
# regardless of cloud provider.
# ============================================================

READ_OPERATIONS = {
    "GetObject",
    "HeadObject",
    "GetObjectAcl",
    "GetObjectTagging",
    "GetObjectTorrent",
    "SelectObjectContent"
}

WRITE_OPERATIONS = {
    "PutObject",
    "CopyObject",
    "RestoreObject",
    "UploadPart",
    "CompleteMultipartUpload"
}

DELETE_OPERATIONS = {
    "DeleteObject",
    "DeleteObjects",
    "DeleteObjectTagging"
}

LIST_OPERATIONS = {
    "ListBucket",
    "ListObjects",
    "ListObjectsV2",
    "ListObjectVersions",
    "ListMultipartUploads"
}

HIGH_RISK_OPERATIONS = {
    "DeleteObject",
    "DeleteObjects",
    "PutBucketAcl",
    "PutBucketPolicy",
    "DeleteBucketPolicy",
    "PutBucketPublicAccessBlock"
}

RECON_OPERATIONS = {
    "GetBucketAcl",
    "GetBucketPolicy",
    "GetBucketCORS",
    "GetBucketWebsite",
    "GetBucketLogging"
}

# ============================================================
# SENSITIVE PATH PATTERNS
#
# WHY PATH PRE-SCORING:
# Before PII classifier runs we can already
# estimate sensitivity from the object key.
# Developer-named paths reveal intent.
# This allows immediate risk scoring
# even before content classification.
# ============================================================

HIGH_SENSITIVITY_PATH_KEYWORDS = {
    "pii", "phi", "pci", "ssn", "sensitive",
    "confidential", "restricted", "secret",
    "private", "protected", "classified"
}

MEDIUM_SENSITIVITY_PATH_KEYWORDS = {
    "customer", "customers", "patient", "patients",
    "employee", "employees", "member", "members",
    "account", "accounts", "payment", "payments",
    "transaction", "transactions", "financial",
    "medical", "health", "insurance", "card",
    "cardholder", "credit", "loan", "mortgage"
}

EXPORT_PATH_KEYWORDS = {
    "export", "exports", "dump", "backup",
    "archive", "extract", "full", "complete",
    "all_customers", "all_accounts"
}

# Business hours UTC
BUSINESS_HOURS_START = 13
BUSINESS_HOURS_END = 23

# Large object threshold (bytes)
# Objects above this size flagged for
# potential exfiltration review
LARGE_OBJECT_THRESHOLD = 10_000_000  # 10MB

# Very large threshold
VERY_LARGE_THRESHOLD = 100_000_000  # 100MB


class S3Normalizer:
    """
    Normalizes AWS S3 CloudTrail events
    to DataAccessEvent objects.

    THIS IS THE TEMPLATE NORMALIZER.
    Every future data source normalizer
    follows this exact pattern.

    Key detections:
    - Large object downloads (exfiltration risk)
    - Bucket listing (enumeration)
    - Off-hours access
    - New accessor patterns
    - Sensitive path pre-scoring
    - Bulk operation detection

    Usage:
        normalizer = S3Normalizer()
        event = normalizer.normalize(
            raw_cloudtrail_event
        )
        if event:
            finding = pii_classifier.classify(
                fetch_object_sample(event.data_path)
            )
            event.finding = finding
    """

    def __init__(self):
        # Track known accessors per bucket
        # key: bucket_name
        # value: set of accessor identities
        self._bucket_accessors = {}

        # Track access history per accessor
        # key: accessor_identity
        # value: list of recent access records
        self._accessor_history = {}

        # Statistics
        self.events_processed = 0
        self.high_risk_events = 0
        self.large_transfers_detected = 0
        self.new_accessors_detected = 0
        self.off_hours_detected = 0

        logger.info("S3Normalizer initialized")

    def normalize(
        self,
        raw_event: dict
    ) -> Optional[DataAccessEvent]:
        """
        Normalize AWS S3 CloudTrail event
        to DataAccessEvent.

        ETL PIPELINE:
        1. Validate this is an S3 event
        2. Extract identity (WHO)
        3. Extract bucket and object key (WHAT)
        4. Extract operation (HOW)
        5. Extract transfer size (HOW MUCH)
        6. Pre-score path sensitivity
        7. Detect behavioral signals
        8. Calculate risk score
        9. Return DataAccessEvent

        Args:
            raw_event: Raw CloudTrail event dict

        Returns:
            DataAccessEvent or None
        """
        if not raw_event:
            return None

        # Validate S3 event
        event_source = raw_event.get(
            "eventSource", ""
        )
        if "s3" not in event_source.lower():
            # Also check for direct S3 events
            event_name = raw_event.get(
                "eventName", ""
            )
            s3_events = (
                READ_OPERATIONS |
                WRITE_OPERATIONS |
                DELETE_OPERATIONS |
                LIST_OPERATIONS |
                HIGH_RISK_OPERATIONS |
                RECON_OPERATIONS
            )
            if event_name not in s3_events:
                return None

        try:
            # ---- EXTRACT CORE FIELDS ----
            event_time = raw_event.get(
                "eventTime", ""
            )
            event_name = raw_event.get(
                "eventName", ""
            )
            event_id = raw_event.get(
                "requestID", ""
            )
            region = raw_event.get(
                "awsRegion", ""
            )
            source_ip = raw_event.get(
                "sourceIPAddress", ""
            )

            # ---- EXTRACT IDENTITY ----
            # DRY PRINCIPLE:
            # Same userIdentity structure as
            # AWSSecretsNormalizer.
            # In production extract to shared utility.
            identity = raw_event.get(
                "userIdentity", {}
            )
            accessor_identity = (
                self._extract_accessor_name(identity)
            )
            accessor_type = self._map_accessor_type(
                identity.get("type", ""),
                accessor_identity
            )

            # ---- EXTRACT S3 CONTEXT ----
            params = raw_event.get(
                "requestParameters", {}
            ) or {}
            bucket_name = params.get(
                "bucketName", ""
            )
            object_key = params.get("key", "")

            # ---- EXTRACT TRANSFER SIZE ----
            additional = raw_event.get(
                "additionalEventData", {}
            ) or {}
            bytes_out = int(
                additional.get(
                    "bytesTransferredOut", 0
                ) or 0
            )
            bytes_in = int(
                additional.get(
                    "bytesTransferredIn", 0
                ) or 0
            )
            bytes_accessed = bytes_out + bytes_in

            # ---- DETERMINE OPERATION ----
            operation = self._map_operation(
                event_name
            )

            # ---- DETERMINE ENVIRONMENT ----
            environment = self._detect_environment(
                bucket_name
            )

            # ---- PATH PRE-SCORING ----
            path_sensitivity, path_confidence = (
                self._score_path_sensitivity(
                    bucket_name, object_key
                )
            )

            # ---- BEHAVIORAL SIGNALS ----
            is_off_hours = self._is_off_hours(
                event_time
            )
            if is_off_hours:
                self.off_hours_detected += 1

            is_new_accessor = self._is_new_accessor(
                accessor_identity, bucket_name
            )
            if is_new_accessor:
                self.new_accessors_detected += 1

            is_large_transfer = (
                bytes_accessed >= LARGE_OBJECT_THRESHOLD
            )
            if is_large_transfer:
                self.large_transfers_detected += 1

            is_very_large = (
                bytes_accessed >= VERY_LARGE_THRESHOLD
            )

            is_enumeration = (
                event_name in LIST_OPERATIONS or
                event_name in RECON_OPERATIONS
            )

            # ---- BUILD DATA ACCESS EVENT ----
            event = DataAccessEvent(
                event_id=event_id,
                event_time=event_time,
                source_system="aws_s3",
                accessor_identity=accessor_identity,
                accessor_type=accessor_type,
                accessor_domain=identity.get(
                    "accountId", ""
                ),
                data_store_type=DataStoreType.S3,
                data_store_name=bucket_name,
                data_path=object_key,
                operation=operation,
                bytes_accessed=bytes_accessed,
                environment=environment,
                source_ip=source_ip,
                source_region=region,
                is_off_hours=is_off_hours,
                raw_event=raw_event
            )

            # ---- RISK SCORING ----
            risk_score, risk_reasons = (
                self._calculate_risk(
                    event=event,
                    event_name=event_name,
                    path_sensitivity=path_sensitivity,
                    path_confidence=path_confidence,
                    is_new_accessor=is_new_accessor,
                    is_large_transfer=is_large_transfer,
                    is_very_large=is_very_large,
                    is_enumeration=is_enumeration,
                    bytes_accessed=bytes_accessed
                )
            )

            event.risk_score = risk_score
            event.risk_label = (
                self._score_to_label(risk_score)
            )
            event.risk_reasons = risk_reasons

            # ---- UPDATE TRACKING ----
            self._update_history(
                accessor_identity,
                bucket_name,
                event_time
            )

            self.events_processed += 1
            if risk_score >= 0.7:
                self.high_risk_events += 1

            logger.info(
                f"S3 event normalized: "
                f"{event_name} "
                f"bucket={bucket_name} "
                f"key={object_key} "
                f"accessor={accessor_identity} "
                f"bytes={bytes_accessed} "
                f"risk={risk_score:.2f}"
            )

            return event

        except Exception as e:
            logger.error(
                f"S3 normalization failed: {e}"
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
        Extract human-readable accessor name
        from CloudTrail userIdentity.

        DRY NOTE:
        This is identical to AWSSecretsNormalizer.
        In production move to:
        layer1_ingestion/utils/aws_utils.py
        """
        identity_type = identity.get("type", "")

        if identity_type == "IAMUser":
            return identity.get("userName", "unknown")

        elif identity_type == "AssumedRole":
            arn = identity.get("arn", "")
            parts = arn.split("/")
            if len(parts) >= 3:
                return parts[-1]
            return identity.get(
                "sessionContext", {}
            ).get(
                "sessionIssuer", {}
            ).get("userName", "assumed_role")

        elif identity_type == "Root":
            return "aws_root"

        elif identity_type == "AWSService":
            return identity.get(
                "invokedBy", "aws_service"
            )

        return identity.get(
            "principalId", "unknown"
        )

    def _map_accessor_type(
        self,
        aws_type: str,
        accessor_name: str
    ) -> AccessorType:
        """
        Map AWS identity type to AccessorType.

        WHY ACCESSOR_TYPE MATTERS:
        svc_backup at 2am = check schedule
        human at 2am = investigate immediately
        ETL job = check if in normal scope
        """
        accessor_lower = accessor_name.lower()

        # Service account patterns
        svc_patterns = [
            "svc_", "svc-", "service",
            "backup", "etl", "pipeline",
            "replication", "sync", "batch"
        ]

        if aws_type == "Root":
            return AccessorType.PRIVILEGED

        if aws_type == "AWSService":
            return AccessorType.APPLICATION

        if aws_type in ["IAMUser", "AssumedRole"]:
            # Check if name suggests service account
            if any(
                p in accessor_lower
                for p in svc_patterns
            ):
                # Further distinguish ETL from backup
                if any(
                    p in accessor_lower
                    for p in ["etl", "pipeline", "batch"]
                ):
                    return AccessorType.ETL_PROCESS
                if "backup" in accessor_lower:
                    return AccessorType.BACKUP
                return AccessorType.SERVICE_ACCOUNT

            # Check for application patterns
            app_patterns = ["app", "api", "lambda"]
            if any(
                p in accessor_lower
                for p in app_patterns
            ):
                return AccessorType.API_CLIENT

            return AccessorType.HUMAN

        return AccessorType.UNKNOWN

    def _map_operation(
        self,
        event_name: str
    ) -> DataOperation:
        """Map S3 event name to DataOperation"""
        if event_name in READ_OPERATIONS:
            return DataOperation.READ
        elif event_name in WRITE_OPERATIONS:
            return DataOperation.WRITE
        elif event_name in DELETE_OPERATIONS:
            return DataOperation.DELETE
        elif event_name in LIST_OPERATIONS:
            return DataOperation.LIST
        return DataOperation.UNKNOWN

    def _detect_environment(
        self,
        bucket_name: str
    ) -> Environment:
        """
        Detect environment from bucket name.

        Banks follow naming conventions:
        prod-customer-data → PRODUCTION
        staging-reports    → STAGING
        dev-test-bucket    → DEVELOPMENT

        PRODUCTION gets highest risk scores.
        Development should NEVER have real PII.
        Finding PII in dev = governance violation.
        """
        bucket_lower = bucket_name.lower()

        if any(
            p in bucket_lower
            for p in ["prod", "production", "prd"]
        ):
            return Environment.PRODUCTION

        if any(
            p in bucket_lower
            for p in ["stag", "staging", "stg", "uat"]
        ):
            return Environment.STAGING

        if any(
            p in bucket_lower
            for p in ["dev", "develop", "test", "tst"]
        ):
            return Environment.DEVELOPMENT

        if any(
            p in bucket_lower
            for p in ["dr", "disaster", "recovery"]
        ):
            return Environment.DISASTER_RECOVERY

        return Environment.UNKNOWN

    # ============================================================
    # PATH SENSITIVITY PRE-SCORING
    # ============================================================

    def _score_path_sensitivity(
        self,
        bucket_name: str,
        object_key: str
    ) -> tuple:
        """
        Pre-score sensitivity from path.

        WHY PATH PRE-SCORING:
        Before PII classifier runs we estimate
        sensitivity from naming patterns.

        Developers name folders for their content:
        /pii/        = 0.9 confidence PII
        /customers/  = 0.6 confidence PII
        /reports/    = 0.2 confidence

        Returns:
            (SensitivityLabel, confidence_float)
        """
        combined_path = (
            f"{bucket_name}/{object_key}"
        ).lower()

        # Check high sensitivity keywords
        high_hits = sum(
            1 for kw in HIGH_SENSITIVITY_PATH_KEYWORDS
            if kw in combined_path
        )

        # Check medium sensitivity keywords
        medium_hits = sum(
            1 for kw in MEDIUM_SENSITIVITY_PATH_KEYWORDS
            if kw in combined_path
        )

        # Check export keywords (bulk data risk)
        export_hits = sum(
            1 for kw in EXPORT_PATH_KEYWORDS
            if kw in combined_path
        )

        # Score based on hits
        if high_hits >= 1:
            return SensitivityLabel.PII, 0.85

        if medium_hits >= 2:
            confidence = min(
                0.75, 0.4 + (medium_hits * 0.1)
            )
            return SensitivityLabel.PII, confidence

        if medium_hits == 1:
            confidence = 0.4 + (export_hits * 0.1)
            return SensitivityLabel.PII, confidence

        return SensitivityLabel.UNKNOWN, 0.1

    # ============================================================
    # BEHAVIORAL DETECTION
    # ============================================================

    def _is_off_hours(
        self,
        event_time: str
    ) -> bool:
        """Check if access outside business hours"""
        try:
            if "T" in event_time:
                hour = int(
                    event_time.split("T")[1][:2]
                )
                return not (BUSINESS_HOURS_START <= hour < BUSINESS_HOURS_END)
                
        except Exception:
            pass
        return False

    def _is_new_accessor(
        self,
        accessor_identity: str,
        bucket_name: str
    ) -> bool:
        """
        Check if this accessor has accessed
        this bucket before.

        First-time accessor = investigation signal.
        Could be legitimate new service.
        Could be attacker using compromised account.
        Context from Layer 3 determines which.
        """
        if not bucket_name:
            return False

        known = self._bucket_accessors.get(
            bucket_name, set()
        )
        return accessor_identity not in known

    # ============================================================
    # RISK SCORING
    # ============================================================

    def _calculate_risk(
        self,
        event: DataAccessEvent,
        event_name: str,
        path_sensitivity: SensitivityLabel,
        path_confidence: float,
        is_new_accessor: bool,
        is_large_transfer: bool,
        is_very_large: bool,
        is_enumeration: bool,
        bytes_accessed: int
    ) -> tuple:
        """
        Calculate risk score for S3 access event.

        RISK FACTORS:
        1. Path sensitivity (what data likely exists)
        2. Transfer size (how much was taken)
        3. Operation type (read vs delete vs list)
        4. Timing (off hours)
        5. New accessor pattern
        6. Environment (production vs dev)
        """
        score = 0.0
        reasons = []

        # Sensitive path detected
        if path_confidence >= 0.8:
            score += 0.4
            reasons.append(
                f"High sensitivity path detected: "
                f"{event.data_path} — "
                f"likely contains "
                f"{path_sensitivity.value} data"
            )
        elif path_confidence >= 0.4:
            score += 0.2
            reasons.append(
                f"Potentially sensitive path: "
                f"{event.data_path}"
            )

        # Large transfer
        if is_very_large:
            score += 0.5
            reasons.append(
                f"Very large transfer: "
                f"{bytes_accessed:,} bytes — "
                f"possible data exfiltration"
            )
        elif is_large_transfer:
            score += 0.3
            reasons.append(
                f"Large transfer: "
                f"{bytes_accessed:,} bytes"
            )

        # High risk operation
        if event_name in HIGH_RISK_OPERATIONS:
            score += 0.4
            reasons.append(
                f"High risk operation: {event_name}"
            )

        # Enumeration / reconnaissance
        if is_enumeration:
            score += 0.3
            reasons.append(
                f"Enumeration operation: {event_name} — "
                f"possible reconnaissance"
            )

        # Off hours access
        if event.is_off_hours:
            score += 0.2
            reasons.append(
                "S3 access outside business hours"
            )

        # New accessor
        if is_new_accessor:
            score += 0.2
            reasons.append(
                f"First time "
                f"{event.accessor_identity} "
                f"accessed {event.data_store_name}"
            )

        # Production environment
        if event.environment == Environment.PRODUCTION:
            score += 0.1
            reasons.append(
                "Production environment access"
            )

        # Root access
        if event.accessor_type == AccessorType.PRIVILEGED:
            score += 0.4
            reasons.append(
                "AWS root credentials used to "
                "access S3 — critical severity"
            )

        return min(score, 1.0), reasons

    # ============================================================
    # HISTORY MANAGEMENT
    # ============================================================

    def _update_history(
        self,
        accessor_identity: str,
        bucket_name: str,
        event_time: str
    ) -> None:
        """Update accessor and bucket history"""
        # Track accessors per bucket
        if bucket_name:
            if bucket_name not in (
                self._bucket_accessors
            ):
                self._bucket_accessors[
                    bucket_name
                ] = set()
            self._bucket_accessors[
                bucket_name
            ].add(accessor_identity)

        # Track access history per accessor
        if accessor_identity not in (
            self._accessor_history
        ):
            self._accessor_history[
                accessor_identity
            ] = []

        self._accessor_history[
            accessor_identity
        ].append({
            "bucket": bucket_name,
            "time": event_time
        })

        # Keep last 100 events per accessor
        self._accessor_history[accessor_identity] = (
            self._accessor_history[
                accessor_identity
            ][-100:]
        )

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    def _score_to_label(
        self,
        score: float
    ) -> str:
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
            "high_risk_events": self.high_risk_events,
            "large_transfers_detected": (
                self.large_transfers_detected
            ),
            "new_accessors_detected": (
                self.new_accessors_detected
            ),
            "off_hours_detected": (
                self.off_hours_detected
            ),
            "buckets_tracked": len(
                self._bucket_accessors
            ),
            "accessors_tracked": len(
                self._accessor_history
            )
        }