"""
Tests for AWS Security Hub (ASFF) Normalizer
"""

import pytest
from layer1_ingestion.normalizers.security_hub_normalizer\
    import (
        SecurityHubNormalizer,
        ASFF_SEVERITY_RISK,
        PRODUCT_KEYWORDS,
        PRODUCT_TO_TECHNIQUE
    )


@pytest.fixture
def normalizer():
    return SecurityHubNormalizer()


@pytest.fixture
def inspector_cve():
    return {
        "ProductArn": "arn:aws:securityhub:us-east-1::product/aws/inspector",
        "Title": "CVE-2024-1234 - Critical RCE with known exploit",
        "Description": "A critical remote code execution vulnerability with a public exploit available",
        "Severity": {"Label": "CRITICAL"},
        "Resources": [
            {"Id": "arn:aws:ec2:us-east-1:111:instance/i-0payment", "Type": "AwsEc2Instance"}
        ],
        "Types": ["Software and Configuration Checks/Vulnerabilities/CVE"],
        "UpdatedAt": "2026-06-01T10:00:00Z"
    }


@pytest.fixture
def macie_sensitive():
    return {
        "ProductArn": "arn:aws:securityhub:us-east-1::product/aws/macie",
        "Title": "Sensitive data discovered: credit card numbers in S3",
        "Description": "Macie found PCI cardholder data in a publicly accessible bucket",
        "Severity": {"Label": "HIGH"},
        "Resources": [
            {"Id": "arn:aws:s3:::prod-pci-bucket", "Type": "AwsS3Bucket"}
        ],
        "Types": ["Sensitive Data Identifications/PII"]
    }


@pytest.fixture
def config_misconfig():
    return {
        "ProductArn": "arn:aws:securityhub:us-east-1::product/aws/config",
        "Title": "S3 bucket allows public access",
        "Description": "An S3 bucket is publicly accessible, violating policy",
        "Severity": {"Label": "HIGH"},
        "Resources": [
            {"Id": "arn:aws:s3:::exposed-bucket", "Type": "AwsS3Bucket"}
        ],
        "Types": ["Software and Configuration Checks"],
        "Compliance": {
            "Status": "FAILED",
            "RelatedRequirements": ["PCI DSS 1.2.1", "CIS AWS 2.1.5"]
        }
    }


@pytest.fixture
def access_analyzer():
    return {
        "ProductArn": "arn:aws:securityhub:us-east-1::product/aws/access-analyzer",
        "Title": "IAM role allows external access",
        "Description": "An IAM role grants external access to an unknown principal",
        "Severity": {"Label": "MEDIUM"},
        "Resources": [
            {"Id": "arn:aws:iam::111:role/cross-account-role", "Type": "AwsIamRole"}
        ],
        "Types": ["Effects/Data Exposure"]
    }


@pytest.fixture
def normalized_severity_finding():
    return {
        "ProductArn": "arn:aws:securityhub:us-east-1::product/aws/inspector",
        "Title": "Vulnerability finding",
        "Severity": {"Normalized": 95},
        "Resources": [{"Id": "i-test", "Type": "AwsEc2Instance"}]
    }


@pytest.fixture
def wrapped_findings():
    return {
        "Findings": [
            {
                "ProductArn": "arn:aws:securityhub:us-east-1::product/aws/macie",
                "Title": "PII discovered in S3",
                "Severity": {"Label": "HIGH"},
                "Resources": [{"Id": "arn:aws:s3:::pii-bucket"}],
                "Types": ["Sensitive Data Identifications/PII"]
            }
        ]
    }


# ============================================================
# INITIALIZATION
# ============================================================

class TestInitialization:

    def test_normalizer_initializes(self, normalizer):
        assert normalizer is not None
        assert normalizer.source_system == (
            "aws_security_hub"
        )

    def test_severity_map(self):
        assert ASFF_SEVERITY_RISK["CRITICAL"] > (
            ASFF_SEVERITY_RISK["LOW"]
        )

    def test_product_keywords(self):
        assert "inspector" in PRODUCT_KEYWORDS
        assert "macie" in PRODUCT_KEYWORDS

    def test_product_technique_map(self):
        assert PRODUCT_TO_TECHNIQUE[
            "macie"
        ] == "T1530"


# ============================================================
# INSPECTOR (vulnerabilities)
# ============================================================

class TestInspector:

    def test_returns_dict(
        self, normalizer, inspector_cve
    ):
        result = normalizer.normalize(inspector_cve)
        assert isinstance(result, dict)

    def test_product_identified(
        self, normalizer, inspector_cve
    ):
        result = normalizer.normalize(inspector_cve)
        assert result["securityhub_product"] == (
            "inspector"
        )

    def test_critical_severity(
        self, normalizer, inspector_cve
    ):
        result = normalizer.normalize(inspector_cve)
        assert result["securityhub_severity"] == (
            "CRITICAL"
        )

    def test_cve_technique(
        self, normalizer, inspector_cve
    ):
        result = normalizer.normalize(inspector_cve)
        assert result["mitre_technique"] == "T1190"

    def test_exploitable_escalation(
        self, normalizer, inspector_cve
    ):
        result = normalizer.normalize(inspector_cve)
        assert result["risk_score"] >= 0.85
        assert "exploitable_vulnerability" in (
            result["risk_reasons"]
        )

    def test_resource_shortened(
        self, normalizer, inspector_cve
    ):
        result = normalizer.normalize(inspector_cve)
        assert result["data_store_name"] == (
            "i-0payment"
        )


# ============================================================
# MACIE (sensitive data)
# ============================================================

class TestMacie:

    def test_product_identified(
        self, normalizer, macie_sensitive
    ):
        result = normalizer.normalize(
            macie_sensitive
        )
        assert result["securityhub_product"] == (
            "macie"
        )

    def test_pci_classification(
        self, normalizer, macie_sensitive
    ):
        result = normalizer.normalize(
            macie_sensitive
        )
        assert result["data_classification"] == (
            "PCI"
        )

    def test_sensitive_data_escalation(
        self, normalizer, macie_sensitive
    ):
        result = normalizer.normalize(
            macie_sensitive
        )
        assert result["risk_score"] >= 0.78
        assert "sensitive_data_exposed" in (
            result["risk_reasons"]
        )

    def test_data_technique(
        self, normalizer, macie_sensitive
    ):
        result = normalizer.normalize(
            macie_sensitive
        )
        assert result["mitre_technique"] == "T1530"


# ============================================================
# CONFIG (compliance)
# ============================================================

class TestConfig:

    def test_product_identified(
        self, normalizer, config_misconfig
    ):
        result = normalizer.normalize(
            config_misconfig
        )
        assert result["securityhub_product"] == (
            "config"
        )

    def test_compliance_requirements_captured(
        self, normalizer, config_misconfig
    ):
        result = normalizer.normalize(
            config_misconfig
        )
        assert len(
            result["compliance_requirements"]
        ) == 2

    def test_compliance_failed_status(
        self, normalizer, config_misconfig
    ):
        result = normalizer.normalize(
            config_misconfig
        )
        assert result[
            "securityhub_compliance_status"
        ] == "FAILED"

    def test_compliance_failed_reason(
        self, normalizer, config_misconfig
    ):
        result = normalizer.normalize(
            config_misconfig
        )
        assert "compliance_failed" in (
            result["risk_reasons"]
        )

    def test_public_exposure_escalation(
        self, normalizer, config_misconfig
    ):
        result = normalizer.normalize(
            config_misconfig
        )
        assert "public_exposure" in (
            result["risk_reasons"]
        )


# ============================================================
# ACCESS ANALYZER
# ============================================================

class TestAccessAnalyzer:

    def test_product_identified(
        self, normalizer, access_analyzer
    ):
        result = normalizer.normalize(
            access_analyzer
        )
        assert result["securityhub_product"] == (
            "access_analyzer"
        )

    def test_external_access_escalation(
        self, normalizer, access_analyzer
    ):
        result = normalizer.normalize(
            access_analyzer
        )
        assert "external_access_granted" in (
            result["risk_reasons"]
        )

    def test_access_technique(
        self, normalizer, access_analyzer
    ):
        result = normalizer.normalize(
            access_analyzer
        )
        assert result["mitre_technique"] == "T1098"


# ============================================================
# SEVERITY HANDLING
# ============================================================

class TestSeverityHandling:

    def test_normalized_severity_mapped(
        self, normalizer, normalized_severity_finding
    ):
        result = normalizer.normalize(
            normalized_severity_finding
        )
        # Normalized 95 -> CRITICAL
        assert result["securityhub_severity"] == (
            "CRITICAL"
        )


# ============================================================
# WRAPPED FINDINGS
# ============================================================

class TestWrappedFindings:

    def test_findings_list_handled(
        self, normalizer, wrapped_findings
    ):
        result = normalizer.normalize(
            wrapped_findings
        )
        assert result["securityhub_product"] == (
            "macie"
        )

    def test_empty_findings_list(self, normalizer):
        result = normalizer.normalize(
            {"Findings": []}
        )
        assert result["risk_score"] == 0.0


# ============================================================
# EDGE CASES
# ============================================================

class TestEdgeCases:

    def test_empty_event(self, normalizer):
        result = normalizer.normalize({})
        assert result["accessor_identity"] == (
            "unknown"
        )
        assert result["risk_score"] == 0.0

    def test_none_event(self, normalizer):
        result = normalizer.normalize(None)
        assert result["source_system"] == (
            "aws_security_hub"
        )

    def test_risk_never_exceeds_one(
        self, normalizer, inspector_cve
    ):
        result = normalizer.normalize(inspector_cve)
        assert result["risk_score"] <= 1.0

    def test_raw_event_preserved(
        self, normalizer, macie_sensitive
    ):
        result = normalizer.normalize(
            macie_sensitive
        )
        assert result["raw_event"] == macie_sensitive

    def test_event_time_present(
        self, normalizer, inspector_cve
    ):
        result = normalizer.normalize(inspector_cve)
        assert result["event_time"] != ""

    def test_unknown_product_defaults(
        self, normalizer
    ):
        result = normalizer.normalize({
            "ProductArn": "arn:aws:securityhub:us-east-1::product/aws/unknown-thing",
            "Title": "Some finding",
            "Severity": {"Label": "LOW"},
            "Resources": [{"Id": "x"}]
        })
        assert result["securityhub_product"] == (
            "security_hub"
        )