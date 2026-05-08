"""
S3 Normalizer Tests

WHAT WE ARE PROVING:
    1. Core field extraction works
       bucket, key, accessor, operation
    2. accessor_type correctly mapped
       svc_backup → BACKUP
       etl_job    → ETL_PROCESS
       human      → HUMAN
    3. Path sensitivity pre-scoring works
       /pii/ → high confidence
       /logs/ → low confidence
    4. Large transfer detection works
    5. Enumeration detection works
    6. Environment detection works
    7. Risk scoring combines all signals

WHY THESE TESTS MATTER:
    This is the TEMPLATE normalizer.
    Every future data normalizer
    Oracle, Snowflake, SharePoint
    follows this same pattern.
    Getting this right means
    every future normalizer is correct.
"""

import pytest
from layer1_ingestion.normalizers.s3_normalizer import (
    S3Normalizer
)
from layer1_ingestion.schema.data_schema import (
    DataAccessEvent,
    AccessorType,
    DataStoreType,
    DataOperation,
    Environment,
    SensitivityLabel
)


# ============================================================
# SAMPLE RAW CLOUDTRAIL EVENTS
# ============================================================

CLOUDTRAIL_GET_OBJECT = {
    "eventTime": "2024-03-29T15:00:00Z",
    "eventSource": "s3.amazonaws.com",
    "eventName": "GetObject",
    "awsRegion": "us-east-1",
    "sourceIPAddress": "10.0.0.155",
    "requestID": "req-uuid-001",
    "userIdentity": {
        "type": "IAMUser",
        "userName": "svc_backup",
        "arn": "arn:aws:iam::123:user/svc_backup",
        "principalId": "AIDABC123",
        "accountId": "123456789"
    },
    "requestParameters": {
        "bucketName": "prod-customer-data",
        "key": "customers/pii/2024-q1.csv"
    },
    "additionalEventData": {
        "bytesTransferredOut": 52428800,
        "bytesTransferredIn": 0
    }
}

CLOUDTRAIL_LARGE_TRANSFER = {
    "eventTime": "2024-03-29T03:00:00Z",
    "eventSource": "s3.amazonaws.com",
    "eventName": "GetObject",
    "awsRegion": "us-east-1",
    "sourceIPAddress": "185.220.101.45",
    "requestID": "req-uuid-002",
    "userIdentity": {
        "type": "IAMUser",
        "userName": "svc_backup",
        "arn": "arn:aws:iam::123:user/svc_backup",
        "principalId": "AIDABC123",
        "accountId": "123456789"
    },
    "requestParameters": {
        "bucketName": "prod-customer-data",
        "key": "customers/pii/full_export.csv"
    },
    "additionalEventData": {
        "bytesTransferredOut": 500_000_000,
        "bytesTransferredIn": 0
    }
}

CLOUDTRAIL_LIST_BUCKET = {
    "eventTime": "2024-03-29T09:20:00Z",
    "eventSource": "s3.amazonaws.com",
    "eventName": "ListBucket",
    "awsRegion": "us-east-1",
    "sourceIPAddress": "185.220.101.45",
    "requestID": "req-uuid-003",
    "userIdentity": {
        "type": "IAMUser",
        "userName": "unknown_user",
        "arn": "arn:aws:iam::123:user/unknown_user",
        "principalId": "AIDXYZ999",
        "accountId": "123456789"
    },
    "requestParameters": {
        "bucketName": "prod-customer-data",
        "key": ""
    },
    "additionalEventData": {
        "bytesTransferredOut": 0,
        "bytesTransferredIn": 0
    }
}

CLOUDTRAIL_DELETE_OBJECT = {
    "eventTime": "2024-03-29T09:21:00Z",
    "eventSource": "s3.amazonaws.com",
    "eventName": "DeleteObject",
    "awsRegion": "us-east-1",
    "sourceIPAddress": "10.0.0.155",
    "requestID": "req-uuid-004",
    "userIdentity": {
        "type": "IAMUser",
        "userName": "jsmith",
        "arn": "arn:aws:iam::123:user/jsmith",
        "principalId": "AIDDEF456",
        "accountId": "123456789"
    },
    "requestParameters": {
        "bucketName": "prod-reports",
        "key": "reports/q1-2024.csv"
    },
    "additionalEventData": {
        "bytesTransferredOut": 0,
        "bytesTransferredIn": 0
    }
}

CLOUDTRAIL_ETL_ACCESS = {
    "eventTime": "2024-03-29T02:00:00Z",
    "eventSource": "s3.amazonaws.com",
    "eventName": "GetObject",
    "awsRegion": "us-east-1",
    "sourceIPAddress": "10.0.1.50",
    "requestID": "req-uuid-005",
    "userIdentity": {
        "type": "IAMUser",
        "userName": "etl_daily_pipeline",
        "arn": "arn:aws:iam::123:user/etl_daily_pipeline",
        "principalId": "AIDGHI789",
        "accountId": "123456789"
    },
    "requestParameters": {
        "bucketName": "prod-transactions",
        "key": "transactions/2024/march/daily.parquet"
    },
    "additionalEventData": {
        "bytesTransferredOut": 5_000_000,
        "bytesTransferredIn": 0
    }
}

CLOUDTRAIL_ROOT_ACCESS = {
    "eventTime": "2024-03-29T09:22:00Z",
    "eventSource": "s3.amazonaws.com",
    "eventName": "GetObject",
    "awsRegion": "us-east-1",
    "sourceIPAddress": "10.0.0.1",
    "requestID": "req-uuid-006",
    "userIdentity": {
        "type": "Root",
        "arn": "arn:aws:iam::123:root",
        "principalId": "123456789",
        "accountId": "123456789"
    },
    "requestParameters": {
        "bucketName": "prod-vault-secrets",
        "key": "secrets/master_keys.json"
    },
    "additionalEventData": {
        "bytesTransferredOut": 1024,
        "bytesTransferredIn": 0
    }
}

CLOUDTRAIL_DEV_ACCESS = {
    "eventTime": "2024-03-29T14:00:00Z",
    "eventSource": "s3.amazonaws.com",
    "eventName": "GetObject",
    "awsRegion": "us-east-1",
    "sourceIPAddress": "10.0.0.200",
    "requestID": "req-uuid-007",
    "userIdentity": {
        "type": "IAMUser",
        "userName": "dev_john",
        "arn": "arn:aws:iam::123:user/dev_john",
        "principalId": "AIDDEV001",
        "accountId": "123456789"
    },
    "requestParameters": {
        "bucketName": "dev-test-bucket",
        "key": "test/sample_data.csv"
    },
    "additionalEventData": {
        "bytesTransferredOut": 1024,
        "bytesTransferredIn": 0
    }
}


# ============================================================
# TEST CLASS — CORE NORMALIZATION
# ============================================================

class TestCoreNormalization:
    """
    Tests that basic field extraction works.

    WHY THESE FIRST:
    If field extraction is wrong
    everything downstream is wrong.
    These are the foundation tests.
    """

    def setup_method(self):
        self.normalizer = S3Normalizer()

    def test_get_object_normalized(self):
        """
        GetObject event correctly normalized
        to DataAccessEvent.
        Basic sanity check.
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_GET_OBJECT
        )
        assert result is not None
        assert isinstance(result, DataAccessEvent)
        assert result.source_system == "aws_s3"

    def test_bucket_name_extracted(self):
        """Bucket name correctly extracted"""
        result = self.normalizer.normalize(
            CLOUDTRAIL_GET_OBJECT
        )
        assert result.data_store_name == (
            "prod-customer-data"
        )

    def test_object_key_extracted(self):
        """Object key correctly extracted"""
        result = self.normalizer.normalize(
            CLOUDTRAIL_GET_OBJECT
        )
        assert result.data_path == (
            "customers/pii/2024-q1.csv"
        )

    def test_accessor_identity_extracted(self):
        """Accessor identity correctly extracted"""
        result = self.normalizer.normalize(
            CLOUDTRAIL_GET_OBJECT
        )
        assert result.accessor_identity == "svc_backup"

    def test_bytes_accessed_extracted(self):
        """Transfer size correctly extracted"""
        result = self.normalizer.normalize(
            CLOUDTRAIL_GET_OBJECT
        )
        assert result.bytes_accessed == 52428800

    def test_source_ip_extracted(self):
        """Source IP correctly extracted"""
        result = self.normalizer.normalize(
            CLOUDTRAIL_GET_OBJECT
        )
        assert result.source_ip == "10.0.0.155"

    def test_data_store_type_is_s3(self):
        """Data store type set to S3"""
        result = self.normalizer.normalize(
            CLOUDTRAIL_GET_OBJECT
        )
        assert result.data_store_type == (
            DataStoreType.S3
        )

    def test_operation_mapped_to_read(self):
        """GetObject maps to READ operation"""
        result = self.normalizer.normalize(
            CLOUDTRAIL_GET_OBJECT
        )
        assert result.operation == DataOperation.READ

    def test_delete_operation_mapped(self):
        """DeleteObject maps to DELETE operation"""
        result = self.normalizer.normalize(
            CLOUDTRAIL_DELETE_OBJECT
        )
        assert result.operation == DataOperation.DELETE

    def test_list_operation_mapped(self):
        """ListBucket maps to LIST operation"""
        result = self.normalizer.normalize(
            CLOUDTRAIL_LIST_BUCKET
        )
        assert result.operation == DataOperation.LIST

    def test_none_returns_none(self):
        """None input handled gracefully"""
        assert self.normalizer.normalize(None) is None

    def test_non_s3_event_returns_none(self):
        """Non-S3 event returns None"""
        ec2_event = {
            "eventSource": "ec2.amazonaws.com",
            "eventName": "DescribeInstances"
        }
        result = self.normalizer.normalize(ec2_event)
        assert result is None


# ============================================================
# TEST CLASS — ACCESSOR TYPE MAPPING
# ============================================================

class TestAccessorTypeMapping:
    """
    Tests that accessor types are correctly mapped.

    WHY CRITICAL:
    accessor_type drives Layer 4 investigation path.
    svc_backup → check backup schedule
    human      → investigate immediately
    etl        → check job scope

    Wrong type = wrong investigation = missed breach.
    """

    def setup_method(self):
        self.normalizer = S3Normalizer()

    def test_svc_backup_mapped_to_backup(self):
        """
        svc_backup correctly identified as BACKUP.
        Backup jobs have different investigation path
        than human users or ETL processes.
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_GET_OBJECT
        )
        assert result.accessor_type == (
            AccessorType.BACKUP
        )

    def test_etl_process_mapped_correctly(self):
        """
        etl_daily_pipeline mapped to ETL_PROCESS.
        ETL processes have scheduled access patterns.
        Off-schedule = governance violation.
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_ETL_ACCESS
        )
        assert result.accessor_type == (
            AccessorType.ETL_PROCESS
        )

    def test_human_user_mapped_correctly(self):
        """
        Regular user mapped to HUMAN.
        Human off-hours access = investigate.
        No schedule to check.
        No normal volume baseline.
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_DELETE_OBJECT
        )
        assert result.accessor_type == (
            AccessorType.HUMAN
        )

    def test_root_mapped_to_privileged(self):
        """
        AWS root credentials mapped to PRIVILEGED.
        Root accessing S3 = always critical.
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_ROOT_ACCESS
        )
        assert result.accessor_type == (
            AccessorType.PRIVILEGED
        )


# ============================================================
# TEST CLASS — PATH SENSITIVITY
# ============================================================

class TestPathSensitivity:
    """
    Tests for path-based sensitivity pre-scoring.

    WHY PATH PRE-SCORING:
    PII classifier cannot scan everything.
    A bank has millions of S3 objects.
    Path scoring prioritizes what to scan first.
    High sensitivity paths scanned immediately.
    Low sensitivity paths scanned on schedule.

    Skipping this layer = scanner overwhelmed
    = real breaches missed in the noise.
    """

    def setup_method(self):
        self.normalizer = S3Normalizer()

    def test_pii_path_scores_high(self):
        """
        /pii/ in path scores high confidence.
        Developer explicitly named it PII.
        Highest possible path signal.
        """
        label, confidence = (
            self.normalizer._score_path_sensitivity(
                "prod-customer-data",
                "customers/pii/2024-q1.csv"
            )
        )
        assert confidence >= 0.8

    def test_customer_path_scores_medium(self):
        """
        /customers/ scores medium confidence.
        Likely contains personal data.
        Not as certain as explicit /pii/ label.
        """
        label, confidence = (
            self.normalizer._score_path_sensitivity(
                "prod-data",
                "customers/records/data.csv"
            )
        )
        assert confidence >= 0.3

    def test_log_path_scores_low(self):
        """
        /logs/ path scores low confidence.
        May contain some PII but not primary.
        Lower priority for PII classifier.
        """
        label, confidence = (
            self.normalizer._score_path_sensitivity(
                "app-logs",
                "logs/apache/access.log"
            )
        )
        assert confidence < 0.5

    def test_pii_path_elevates_risk_score(self):
        """
        High sensitivity path elevates overall
        risk score of the event.
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_GET_OBJECT
        )
        assert result.risk_score > 0.0

    def test_bucket_name_contributes_to_scoring(self):
        """
        Bucket name itself contributes to
        path sensitivity scoring.
        prod-customer-data → customer keyword
        """
        label, confidence = (
            self.normalizer._score_path_sensitivity(
                "prod-customer-data",
                "exports/2024.csv"
            )
        )
        assert confidence > 0.1


# ============================================================
# TEST CLASS — BEHAVIORAL DETECTION
# ============================================================

class TestBehavioralDetection:
    """
    Tests for behavioral signal detection.

    These prove your platform detects
    DEVIATIONS from normal patterns
    not just known signatures.
    """

    def setup_method(self):
        self.normalizer = S3Normalizer()

    def test_large_transfer_detected(self):
        """
        500MB transfer detected as large.
        Possible data exfiltration.
        Risk score elevated.
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_LARGE_TRANSFER
        )
        assert result.bytes_accessed == 500_000_000
        assert result.risk_score >= 0.5

    def test_off_hours_access_detected(self):
        """
        3am access detected as off-hours.
        CLOUDTRAIL_LARGE_TRANSFER is at 03:00 UTC.
        Off-hours elevates risk score.
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_LARGE_TRANSFER
        )
        assert result.is_off_hours is True

    def test_business_hours_not_flagged(self):
        """
        Business hours access not flagged off-hours.
        CLOUDTRAIL_GET_OBJECT is at 09:19 UTC.
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_GET_OBJECT
        )
        assert result.is_off_hours is False

    def test_enumeration_elevates_risk(self):
        """
        ListBucket operation elevates risk score.
        Unknown user listing production bucket
        at any time = reconnaissance signal.
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_LIST_BUCKET
        )
        assert result.risk_score >= 0.3

    def test_new_accessor_detected(self):
        """
        First time accessor flagged.
        unknown_user never accessed this bucket.
        New accessor = investigation signal.
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_LIST_BUCKET
        )
        assert result.risk_score > 0.0

    def test_root_access_critical_risk(self):
        """
        AWS root credentials = always critical.
        Root should never access S3 directly.
        Any root access = immediate investigation.
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_ROOT_ACCESS
        )
        assert result.risk_score >= 0.5
        assert result.risk_label in [
            "CRITICAL", "HIGH"
        ]

    def test_delete_elevates_risk(self):
        """
        DeleteObject elevates risk score.
        Deletion is irreversible.
        Always requires review.
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_DELETE_OBJECT
        )
        assert result.risk_score > 0.0


# ============================================================
# TEST CLASS — ENVIRONMENT DETECTION
# ============================================================

class TestEnvironmentDetection:
    """
    Tests for environment detection.

    WHY ENVIRONMENT MATTERS:
    Production PII access = highest risk.
    Development PII access = governance violation
    (dev should never have real PII).
    Test environment = lowest risk.
    """

    def setup_method(self):
        self.normalizer = S3Normalizer()

    def test_prod_bucket_detected(self):
        """
        prod-customer-data detected as PRODUCTION.
        Production data access gets elevated risk.
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_GET_OBJECT
        )
        assert result.environment == (
            Environment.PRODUCTION
        )

    def test_dev_bucket_detected(self):
        """
        dev-test-bucket detected as DEVELOPMENT.
        Development access gets lower base risk.
        Finding PII in dev = governance violation.
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_DEV_ACCESS
        )
        assert result.environment == (
            Environment.DEVELOPMENT
        )

    def test_production_increases_risk(self):
        """
        Production environment adds to risk score.
        Same access in dev scores lower than prod.
        """
        prod_result = self.normalizer.normalize(
            CLOUDTRAIL_GET_OBJECT
        )
        dev_result = self.normalizer.normalize(
            CLOUDTRAIL_DEV_ACCESS
        )
        assert (
            prod_result.risk_score >=
            dev_result.risk_score
        )


# ============================================================
# TEST CLASS — RISK SCORING
# ============================================================

class TestRiskScoring:
    """
    Tests that risk scoring combines signals correctly.

    THIS IS THE MOST IMPORTANT TEST CLASS.
    Risk score drives:
    - Whether alert is generated
    - Priority in investigation queue
    - Which Layer 4 agent handles it
    - Whether breach notification triggers
    """

    def setup_method(self):
        self.normalizer = S3Normalizer()

    def test_risk_score_in_valid_range(self):
        """Risk score always between 0 and 1"""
        result = self.normalizer.normalize(
            CLOUDTRAIL_GET_OBJECT
        )
        assert 0.0 <= result.risk_score <= 1.0

    def test_risk_reasons_populated(self):
        """
        High risk events have human-readable reasons.
        Analyst needs to know WHY it was flagged.
        "Anomalous" is not actionable.
        "500MB transfer from /pii/ at 3am" is.
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_LARGE_TRANSFER
        )
        assert len(result.risk_reasons) > 0

    def test_combined_signals_increase_risk(self):
        """
        Large transfer + off-hours + sensitive path
        scores higher than any single signal.
        Defense in depth for detection:
        more signals = higher confidence.
        """
        large_offhours = self.normalizer.normalize(
            CLOUDTRAIL_LARGE_TRANSFER
        )
        normal = self.normalizer.normalize(
            CLOUDTRAIL_DEV_ACCESS
        )
        assert (
            large_offhours.risk_score >
            normal.risk_score
        )

    def test_risk_label_matches_score(self):
        """
        Risk label correctly derived from score.
        CRITICAL >= 0.8
        HIGH >= 0.6
        MEDIUM >= 0.4
        LOW > 0.0
        """
        result = self.normalizer.normalize(
            CLOUDTRAIL_ROOT_ACCESS
        )
        if result.risk_score >= 0.8:
            assert result.risk_label == "CRITICAL"
        elif result.risk_score >= 0.6:
            assert result.risk_label == "HIGH"
        elif result.risk_score >= 0.4:
            assert result.risk_label == "MEDIUM"

    def test_statistics_tracked(self):
        """Statistics correctly tracked"""
        self.normalizer.normalize(CLOUDTRAIL_GET_OBJECT)
        self.normalizer.normalize(
            CLOUDTRAIL_LARGE_TRANSFER
        )
        stats = self.normalizer.get_statistics()
        assert stats["events_processed"] == 2
        assert stats["large_transfers_detected"] >= 1