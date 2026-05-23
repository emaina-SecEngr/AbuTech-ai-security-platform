"""
Tests for Google Cloud Platform Normalizer
"""

import pytest
from layer1_ingestion.normalizers.gcp_normalizer import (
    GCPNormalizer,
    GCP_METHOD_MAP,
    GCP_SERVICE_MAP
)


@pytest.fixture
def normalizer():
    return GCPNormalizer()


@pytest.fixture
def bigquery_event():
    return {
        "timestamp": "2026-05-21T03:14:22Z",
        "logName": "projects/prod-bofa/logs/cloudaudit",
        "resource": {
            "type": "bigquery_dataset",
            "labels": {
                "project_id": "prod-bofa",
                "dataset_id": "customer_pci_data"
            }
        },
        "protoPayload": {
            "serviceName": "bigquery.googleapis.com",
            "methodName": (
                "google.bigquery.v2."
                "TableDataService.list"
            ),
            "authenticationInfo": {
                "principalEmail": (
                    "svc-backup@prod-bofa"
                    ".iam.gserviceaccount.com"
                )
            },
            "requestMetadata": {
                "callerIp": "185.220.101.45"
            },
            "resourceName": (
                "projects/prod-bofa/datasets/"
                "customer_pci_data/tables/cards"
            )
        }
    }


@pytest.fixture
def gcs_event():
    return {
        "timestamp": "2026-05-21T03:00:00Z",
        "resource": {
            "type": "gcs_bucket",
            "labels": {
                "project_id": "prod-bofa",
                "bucket_name": "prod-customer-data"
            }
        },
        "protoPayload": {
            "serviceName": "storage.googleapis.com",
            "methodName": "storage.objects.get",
            "authenticationInfo": {
                "principalEmail": (
                    "svc-backup@prod-bofa"
                    ".iam.gserviceaccount.com"
                )
            },
            "requestMetadata": {
                "callerIp": "185.220.101.45"
            },
            "resourceName": (
                "projects/_/buckets/"
                "prod-customer-data/objects/"
                "customers/pci/cards.csv"
            )
        }
    }


@pytest.fixture
def iam_event():
    return {
        "timestamp": "2026-05-21T02:00:00Z",
        "resource": {
            "type": "service_account",
            "labels": {
                "project_id": "prod-bofa"
            }
        },
        "protoPayload": {
            "serviceName": "iam.googleapis.com",
            "methodName": (
                "google.iam.admin.v1"
                ".CreateServiceAccountKey"
            ),
            "authenticationInfo": {
                "principalEmail": (
                    "attacker@external.com"
                )
            },
            "requestMetadata": {
                "callerIp": "185.220.101.45"
            },
            "resourceName": (
                "projects/prod-bofa/"
                "serviceAccounts/svc-backup"
            )
        }
    }


@pytest.fixture
def normal_event():
    return {
        "timestamp": "2026-05-21T09:00:00Z",
        "resource": {
            "type": "bigquery_dataset",
            "labels": {
                "project_id": "dev-project",
                "dataset_id": "analytics_logs"
            }
        },
        "protoPayload": {
            "serviceName": "bigquery.googleapis.com",
            "methodName": (
                "google.bigquery.v2.JobService.insert"
            ),
            "authenticationInfo": {
                "principalEmail": (
                    "analyst@company.com"
                )
            },
            "requestMetadata": {
                "callerIp": "10.0.1.100"
            },
            "resourceName": (
                "projects/dev-project/datasets/"
                "analytics_logs"
            )
        }
    }


# ============================================================
# INITIALIZATION TESTS
# ============================================================

class TestInitialization:

    def test_normalizer_initializes(self, normalizer):
        assert normalizer is not None

    def test_method_map_populated(self):
        assert len(GCP_METHOD_MAP) > 0
        assert (
            "storage.objects.get" in GCP_METHOD_MAP
        )

    def test_service_map_populated(self):
        assert len(GCP_SERVICE_MAP) > 0
        assert (
            "bigquery.googleapis.com"
            in GCP_SERVICE_MAP
        )


# ============================================================
# CORE NORMALIZATION TESTS
# ============================================================

class TestNormalize:

    def test_normalize_returns_dict(
        self, normalizer, bigquery_event
    ):
        result = normalizer.normalize(bigquery_event)
        assert isinstance(result, dict)

    def test_normalize_has_required_fields(
        self, normalizer, bigquery_event
    ):
        result = normalizer.normalize(bigquery_event)
        required = [
            "accessor_identity", "accessor_type",
            "data_store_name", "data_path",
            "data_classification", "bytes_accessed",
            "event_time", "source_ip",
            "risk_score", "risk_reasons",
            "source_system", "raw_event"
        ]
        for field in required:
            assert field in result

    def test_normalize_empty_returns_safe(
        self, normalizer
    ):
        result = normalizer.normalize({})
        assert result["accessor_identity"] == "unknown"
        assert result["risk_score"] == 0.0

    def test_normalize_none_returns_safe(
        self, normalizer
    ):
        result = normalizer.normalize(None)
        assert result is not None

    def test_source_system_set(
        self, normalizer, bigquery_event
    ):
        result = normalizer.normalize(bigquery_event)
        assert "gcp" in result["source_system"]


# ============================================================
# FIELD EXTRACTION TESTS
# ============================================================

class TestFieldExtraction:

    def test_extracts_service_account(
        self, normalizer, bigquery_event
    ):
        result = normalizer.normalize(bigquery_event)
        assert (
            "svc-backup" in
            result["accessor_identity"]
        )

    def test_extracts_source_ip(
        self, normalizer, bigquery_event
    ):
        result = normalizer.normalize(bigquery_event)
        assert result["source_ip"] == "185.220.101.45"

    def test_extracts_resource_name(
        self, normalizer, bigquery_event
    ):
        result = normalizer.normalize(bigquery_event)
        assert result["data_store_name"] != ""

    def test_extracts_timestamp(
        self, normalizer, bigquery_event
    ):
        result = normalizer.normalize(bigquery_event)
        assert "2026-05-21" in result["event_time"]

    def test_raw_event_preserved(
        self, normalizer, bigquery_event
    ):
        result = normalizer.normalize(bigquery_event)
        assert result["raw_event"] == bigquery_event

    def test_gcp_method_captured(
        self, normalizer, bigquery_event
    ):
        result = normalizer.normalize(bigquery_event)
        assert "gcp_method" in result
        assert "bigquery" in result["gcp_method"]

    def test_gcp_project_extracted(
        self, normalizer, bigquery_event
    ):
        result = normalizer.normalize(bigquery_event)
        assert "gcp_project" in result


# ============================================================
# ACCESSOR TYPE TESTS
# ============================================================

class TestAccessorType:

    def test_service_account_detected(
        self, normalizer, bigquery_event
    ):
        result = normalizer.normalize(bigquery_event)
        assert result["accessor_type"] == (
            "service_account"
        )

    def test_human_user_detected(
        self, normalizer, normal_event
    ):
        result = normalizer.normalize(normal_event)
        assert result["accessor_type"] == "human"

    def test_gserviceaccount_is_service(
        self, normalizer
    ):
        result = normalizer._detect_accessor_type(
            "terraform@proj.iam.gserviceaccount.com"
        )
        assert result == "service_account"


# ============================================================
# DATA CLASSIFICATION TESTS
# ============================================================

class TestDataClassification:

    def test_pci_classification(
        self, normalizer, bigquery_event
    ):
        result = normalizer.normalize(bigquery_event)
        assert result["data_classification"] == "PCI"

    def test_gcs_pci_classification(
        self, normalizer, gcs_event
    ):
        result = normalizer.normalize(gcs_event)
        assert result["data_classification"] == "PCI"

    def test_unknown_classification(
        self, normalizer, normal_event
    ):
        result = normalizer.normalize(normal_event)
        assert result["data_classification"] in [
            "UNKNOWN", "INTERNAL"
        ]


# ============================================================
# RISK SCORING TESTS
# ============================================================

class TestRiskScoring:

    def test_tor_ip_elevates_risk(
        self, normalizer, bigquery_event
    ):
        result = normalizer.normalize(bigquery_event)
        assert result["risk_score"] >= 0.4

    def test_after_hours_elevates_risk(
        self, normalizer, bigquery_event
    ):
        result = normalizer.normalize(bigquery_event)
        reasons = str(result["risk_reasons"])
        assert "after_hours" in reasons

    def test_iam_key_creation_high_risk(
        self, normalizer, iam_event
    ):
        result = normalizer.normalize(iam_event)
        assert result["risk_score"] >= 0.5

    def test_iam_key_creation_flagged(
        self, normalizer, iam_event
    ):
        result = normalizer.normalize(iam_event)
        reasons = str(result["risk_reasons"])
        assert "service_account_key" in reasons

    def test_normal_event_low_risk(
        self, normalizer, normal_event
    ):
        result = normalizer.normalize(normal_event)
        assert result["risk_score"] <= 0.35

    def test_risk_score_capped_at_one(
        self, normalizer, bigquery_event
    ):
        result = normalizer.normalize(bigquery_event)
        assert result["risk_score"] <= 1.0

    def test_risk_score_not_negative(
        self, normalizer, normal_event
    ):
        result = normalizer.normalize(normal_event)
        assert result["risk_score"] >= 0.0


# ============================================================
# BIGQUERY SPECIFIC TESTS
# ============================================================

class TestBigQueryEvents:

    def test_bigquery_normalization(
        self, normalizer, bigquery_event
    ):
        result = normalizer.normalize_bigquery_event(
            bigquery_event
        )
        assert result is not None
        assert isinstance(result, dict)

    def test_suspicious_query_detection(
        self, normalizer
    ):
        assert normalizer._is_suspicious_query(
            "SELECT * FROM customers"
        )

    def test_normal_query_not_suspicious(
        self, normalizer
    ):
        assert not normalizer._is_suspicious_query(
            "SELECT name, email FROM users LIMIT 100"
        )

    def test_union_select_suspicious(
        self, normalizer
    ):
        assert normalizer._is_suspicious_query(
            "SELECT id UNION SELECT password FROM users"
        )


# ============================================================
# GCS SPECIFIC TESTS
# ============================================================

class TestGCSEvents:

    def test_gcs_event_normalized(
        self, normalizer, gcs_event
    ):
        result = normalizer.normalize(gcs_event)
        assert result["accessor_type"] == (
            "service_account"
        )
        assert result["risk_score"] >= 0.4

    def test_gcs_source_system(
        self, normalizer, gcs_event
    ):
        result = normalizer.normalize(gcs_event)
        assert "gcp" in result["source_system"]