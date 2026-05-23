"""
Tests for CSPM and CIEM Normalizer
"""

import pytest
from layer1_ingestion.normalizers.cspm_normalizer import (
    CSPMNormalizer,
    SEVERITY_RISK_MAP,
    CSPM_FINDING_TYPES,
    CIEM_FINDING_TYPES
)


@pytest.fixture
def normalizer():
    return CSPMNormalizer()


@pytest.fixture
def wiz_finding():
    return {
        "id": "wiz-001",
        "type": "PUBLIC_BUCKET",
        "title": "Public S3 bucket with PCI data",
        "severity": "CRITICAL",
        "createdAt": "2026-05-21T03:00:00Z",
        "hasExternalExposure": True,
        "isToxicCombination": False,
        "resource": {
            "name": "prod-customer-pci-data",
            "type": "BUCKET"
        },
        "entity": {
            "name": "prod-customer-pci-data",
            "type": "BUCKET"
        }
    }


@pytest.fixture
def wiz_toxic_finding():
    return {
        "id": "wiz-002",
        "type": "TOXIC_COMBINATION",
        "title": "Shadow admin with public exposure",
        "severity": "CRITICAL",
        "createdAt": "2026-05-21T02:00:00Z",
        "hasExternalExposure": True,
        "isToxicCombination": True,
        "resource": {
            "name": "svc-backup-role",
            "type": "IAM_ROLE"
        },
        "entity": {
            "name": "svc-backup-role",
            "type": "IAM_ROLE"
        }
    }


@pytest.fixture
def defender_finding():
    return {
        "name": "alert-001",
        "startTimeUtc": "2026-05-21T03:00:00Z",
        "severity": "High",
        "properties": {
            "alertDisplayName": (
                "Suspicious PowerShell activity"
            ),
            "severity": "High",
            "resourceType": "VirtualMachine",
            "resourceDetails": {
                "ResourceName": "prod-vm-01",
                "ResourceType": "VirtualMachine"
            },
            "isIncident": True,
            "compromisedEntity": "10.0.1.105"
        }
    }


@pytest.fixture
def aws_hub_finding():
    return {
        "Id": "arn:aws:securityhub:us-east-1::finding/001",
        "Title": "S3 bucket should not be publicly accessible",
        "AwsAccountId": "123456789012",
        "ProductArn": "arn:aws:securityhub:::product/aws/securityhub",
        "GeneratorId": "arn:aws:securityhub:::ruleset/cis-aws-foundations-benchmark",
        "Severity": {"Label": "HIGH"},
        "CreatedAt": "2026-05-21T03:00:00Z",
        "UpdatedAt": "2026-05-21T03:00:00Z",
        "Resources": [{
            "Type": "AwsS3Bucket",
            "Id": "arn:aws:s3:::prod-customer-data"
        }],
        "Compliance": {"Status": "FAILED"},
        "Workflow": {"Status": "NEW"}
    }


@pytest.fixture
def iam_analyzer_finding():
    return {
        "id": "finding-001",
        "findingType": "ExternalAccess",
        "status": "ACTIVE",
        "analyzedAt": "2026-05-21T03:00:00Z",
        "isPublic": True,
        "resource": (
            "arn:aws:s3:::prod-customer-data"
        ),
        "resourceType": "AWS::S3::Bucket",
        "principal": {
            "AWS": "*"
        },
        "condition": {}
    }


@pytest.fixture
def ciem_finding():
    return {
        "findingType": "shadow_admin",
        "severity": "CRITICAL",
        "createdAt": "2026-05-21T02:00:00Z",
        "identity": {
            "name": "svc_backup",
            "arn": "arn:aws:iam::123:role/svc_backup"
        },
        "entitlement": {
            "permissionsCount": 847,
            "lastUsed": "2026-05-20T03:00:00Z"
        },
        "riskReason": (
            "Service account has AdministratorAccess"
        ),
        "affectedResources": [
            "arn:aws:s3:::prod-customer-data",
            "arn:aws:rds:::prod-db-01"
        ]
    }


# ============================================================
# INITIALIZATION TESTS
# ============================================================

class TestInitialization:

    def test_normalizer_initializes(self, normalizer):
        assert normalizer is not None

    def test_severity_map_populated(self):
        assert len(SEVERITY_RISK_MAP) > 0
        assert "CRITICAL" in SEVERITY_RISK_MAP

    def test_finding_types_populated(self):
        assert len(CSPM_FINDING_TYPES) > 0
        assert len(CIEM_FINDING_TYPES) > 0


# ============================================================
# SOURCE DETECTION TESTS
# ============================================================

class TestSourceDetection:

    def test_detects_wiz(
        self, normalizer, wiz_finding
    ):
        source = normalizer._detect_source(wiz_finding)
        assert source == "wiz"

    def test_detects_defender(
        self, normalizer, defender_finding
    ):
        source = normalizer._detect_source(
            defender_finding
        )
        assert source == "defender"

    def test_detects_aws_hub(
        self, normalizer, aws_hub_finding
    ):
        source = normalizer._detect_source(
            aws_hub_finding
        )
        assert source == "aws_security_hub"

    def test_detects_iam_analyzer(
        self, normalizer, iam_analyzer_finding
    ):
        source = normalizer._detect_source(
            iam_analyzer_finding
        )
        assert source == "iam_access_analyzer"

    def test_unknown_source_generic(self, normalizer):
        source = normalizer._detect_source({
            "someField": "someValue"
        })
        assert source == "generic"


# ============================================================
# WIZ NORMALIZATION TESTS
# ============================================================

class TestWizNormalization:

    def test_wiz_returns_dict(
        self, normalizer, wiz_finding
    ):
        result = normalizer.normalize(wiz_finding)
        assert isinstance(result, dict)

    def test_wiz_critical_high_risk(
        self, normalizer, wiz_finding
    ):
        result = normalizer.normalize(wiz_finding)
        assert result["risk_score"] >= 0.85

    def test_wiz_external_exposure_elevated(
        self, normalizer, wiz_finding
    ):
        result = normalizer.normalize(wiz_finding)
        reasons = str(result["risk_reasons"])
        assert "external_exposure" in reasons

    def test_wiz_toxic_combo_elevated(
        self, normalizer, wiz_toxic_finding
    ):
        result = normalizer.normalize(wiz_toxic_finding)
        assert result["risk_score"] >= 0.90

    def test_wiz_source_system(
        self, normalizer, wiz_finding
    ):
        result = normalizer.normalize(wiz_finding)
        assert "wiz" in result["source_system"]

    def test_wiz_cspm_tool_set(
        self, normalizer, wiz_finding
    ):
        result = normalizer.normalize(wiz_finding)
        assert result["cspm_tool"] == "wiz"

    def test_wiz_pci_classification(
        self, normalizer, wiz_finding
    ):
        result = normalizer.normalize(wiz_finding)
        assert result["data_classification"] == "PCI"


# ============================================================
# DEFENDER FOR CLOUD TESTS
# ============================================================

class TestDefenderNormalization:

    def test_defender_returns_dict(
        self, normalizer, defender_finding
    ):
        result = normalizer.normalize(defender_finding)
        assert isinstance(result, dict)

    def test_defender_high_risk(
        self, normalizer, defender_finding
    ):
        result = normalizer.normalize(defender_finding)
        assert result["risk_score"] >= 0.70

    def test_defender_incident_elevated(
        self, normalizer, defender_finding
    ):
        result = normalizer.normalize(defender_finding)
        assert result["risk_score"] >= 0.75

    def test_defender_source_system(
        self, normalizer, defender_finding
    ):
        result = normalizer.normalize(defender_finding)
        assert "defender" in result["source_system"]


# ============================================================
# AWS SECURITY HUB TESTS
# ============================================================

class TestAWSSecurityHubNormalization:

    def test_aws_hub_returns_dict(
        self, normalizer, aws_hub_finding
    ):
        result = normalizer.normalize(aws_hub_finding)
        assert isinstance(result, dict)

    def test_aws_hub_high_risk(
        self, normalizer, aws_hub_finding
    ):
        result = normalizer.normalize(aws_hub_finding)
        assert result["risk_score"] >= 0.65

    def test_aws_hub_compliance_failed_elevated(
        self, normalizer, aws_hub_finding
    ):
        result = normalizer.normalize(aws_hub_finding)
        reasons = str(result["risk_reasons"])
        assert "compliance_check_failed" in reasons

    def test_aws_hub_source_system(
        self, normalizer, aws_hub_finding
    ):
        result = normalizer.normalize(aws_hub_finding)
        assert "aws_hub" in result["source_system"]


# ============================================================
# IAM ACCESS ANALYZER TESTS
# ============================================================

class TestIAMAnalyzerNormalization:

    def test_iam_analyzer_returns_dict(
        self, normalizer, iam_analyzer_finding
    ):
        result = normalizer.normalize(
            iam_analyzer_finding
        )
        assert isinstance(result, dict)

    def test_public_resource_high_risk(
        self, normalizer, iam_analyzer_finding
    ):
        result = normalizer.normalize(
            iam_analyzer_finding
        )
        assert result["risk_score"] >= 0.80

    def test_public_flag_in_reasons(
        self, normalizer, iam_analyzer_finding
    ):
        result = normalizer.normalize(
            iam_analyzer_finding
        )
        reasons = str(result["risk_reasons"])
        assert "publicly_accessible" in reasons

    def test_iam_analyzer_source(
        self, normalizer, iam_analyzer_finding
    ):
        result = normalizer.normalize(
            iam_analyzer_finding
        )
        assert "iam_analyzer" in (
            result["source_system"]
        )


# ============================================================
# CIEM NORMALIZATION TESTS
# ============================================================

class TestCIEMNormalization:

    def test_ciem_returns_dict(
        self, normalizer, ciem_finding
    ):
        result = normalizer.normalize_ciem_event(
            ciem_finding
        )
        assert isinstance(result, dict)

    def test_ciem_event_type_set(
        self, normalizer, ciem_finding
    ):
        result = normalizer.normalize_ciem_event(
            ciem_finding
        )
        assert result["event_type"] == "CIEM_FINDING"

    def test_ciem_source_system(
        self, normalizer, ciem_finding
    ):
        result = normalizer.normalize_ciem_event(
            ciem_finding
        )
        assert result["source_system"] == "ciem"

    def test_ciem_shadow_admin_high_risk(
        self, normalizer, ciem_finding
    ):
        result = normalizer.normalize_ciem_event(
            ciem_finding
        )
        assert result["risk_score"] >= 0.80

    def test_ciem_identity_captured(
        self, normalizer, ciem_finding
    ):
        result = normalizer.normalize_ciem_event(
            ciem_finding
        )
        assert "svc_backup" in (
            result["ciem_identity"]
        )

    def test_ciem_permissions_count(
        self, normalizer, ciem_finding
    ):
        result = normalizer.normalize_ciem_event(
            ciem_finding
        )
        assert result["ciem_permissions_count"] == 847

    def test_ciem_finding_type_set(
        self, normalizer, ciem_finding
    ):
        result = normalizer.normalize_ciem_event(
            ciem_finding
        )
        assert "ciem_finding_type" in result

    def test_ciem_affected_resources(
        self, normalizer, ciem_finding
    ):
        result = normalizer.normalize_ciem_event(
            ciem_finding
        )
        assert len(
            result["ciem_affected_resources"]
        ) == 2


# ============================================================
# GENERIC AND EDGE CASE TESTS
# ============================================================

class TestEdgeCases:

    def test_empty_event_safe(self, normalizer):
        result = normalizer.normalize({})
        assert result["risk_score"] == 0.0

    def test_none_event_safe(self, normalizer):
        result = normalizer.normalize(None)
        assert result is not None

    def test_risk_score_capped(
        self, normalizer, wiz_toxic_finding
    ):
        result = normalizer.normalize(wiz_toxic_finding)
        assert result["risk_score"] <= 1.0

    def test_required_fields_present(
        self, normalizer, wiz_finding
    ):
        result = normalizer.normalize(wiz_finding)
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

    def test_raw_event_preserved(
        self, normalizer, wiz_finding
    ):
        result = normalizer.normalize(wiz_finding)
        assert result["raw_event"] == wiz_finding