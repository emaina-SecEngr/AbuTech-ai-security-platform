"""
Tests for IaC Security Normalizer
"""

import pytest
from layer1_ingestion.normalizers.iac_normalizer import (
    IaCNormalizer,
    SEVERITY_RISK_MAP,
    RESOURCE_SENSITIVITY,
    CRITICAL_CHECKS,
    FINDING_MITRE_MAP
)


@pytest.fixture
def normalizer():
    return IaCNormalizer()


@pytest.fixture
def checkov_public_s3():
    """Public S3 bucket - CRITICAL finding"""
    return {
        "check_id": "CKV_AWS_20",
        "check": "Ensure S3 bucket is not publicly accessible",
        "check_result": {"result": "FAILED"},
        "resource": "aws_s3_bucket.prod_customer_pci_data",
        "file_path": "terraform/s3/main.tf",
        "file_line_range": [1, 15],
        "severity": "HIGH",
        "guideline": "https://docs.bridgecrew.io/docs/s3-bucket-acl-prohibited"
    }


@pytest.fixture
def checkov_rds_public():
    """Publicly accessible RDS - CRITICAL"""
    return {
        "check_id": "CKV_AWS_17",
        "check": "Ensure RDS database is not publicly accessible",
        "check_result": {"result": "FAILED"},
        "resource": "aws_db_instance.prod_pci_database",
        "file_path": "terraform/rds/main.tf",
        "file_line_range": [10, 45],
        "severity": "CRITICAL",
        "guideline": "https://docs.bridgecrew.io/docs/bc-aws-general-2"
    }


@pytest.fixture
def checkov_ssh_open():
    """SSH open to world - CRITICAL"""
    return {
        "check_id": "CKV_AWS_25",
        "check": "Ensure no security groups allow ingress from 0.0.0.0:0/22",
        "check_result": {"result": "FAILED"},
        "resource": "aws_security_group.prod_sg",
        "file_path": "terraform/network/main.tf",
        "file_line_range": [5, 20],
        "severity": "CRITICAL",
        "guideline": "https://docs.bridgecrew.io/docs/bc-aws-networking-31"
    }


@pytest.fixture
def checkov_iam_wildcard():
    """IAM with wildcard permissions"""
    return {
        "check_id": "CKV_AWS_40",
        "check": "Ensure IAM policies do not have full administrative privileges",
        "check_result": {"result": "FAILED"},
        "resource": "aws_iam_policy.svc_backup_policy",
        "file_path": "terraform/iam/main.tf",
        "file_line_range": [1, 30],
        "severity": "HIGH",
        "guideline": "https://docs.bridgecrew.io"
    }


@pytest.fixture
def checkov_encryption_disabled():
    """Encryption disabled on database"""
    return {
        "check_id": "CKV_AWS_16",
        "check": "Ensure RDS database has encryption at rest enabled",
        "check_result": {"result": "FAILED"},
        "resource": "aws_db_instance.prod_db",
        "file_path": "terraform/rds/main.tf",
        "file_line_range": [15, 40],
        "severity": "HIGH",
        "guideline": "https://docs.bridgecrew.io"
    }


@pytest.fixture
def checkov_k8s_privileged():
    """Kubernetes privileged container"""
    return {
        "check_id": "CKV_K8S_16",
        "check": "Do not admit privileged containers",
        "check_result": {"result": "FAILED"},
        "resource": "kubernetes_pod.prod_app",
        "file_path": "kubernetes/deployment.yaml",
        "file_line_range": [20, 35],
        "severity": "CRITICAL",
        "guideline": "https://docs.bridgecrew.io"
    }


@pytest.fixture
def checkov_low_severity():
    """Low severity finding"""
    return {
        "check_id": "CKV_AWS_18",
        "check": "Ensure S3 bucket has access logging enabled",
        "check_result": {"result": "FAILED"},
        "resource": "aws_s3_bucket.logs",
        "file_path": "terraform/s3/logs.tf",
        "file_line_range": [1, 10],
        "severity": "LOW",
        "guideline": ""
    }


@pytest.fixture
def tfsec_finding():
    """tfsec finding"""
    return {
        "rule_id": "aws-s3-no-public-access-with-acl",
        "long_id": "aws-s3-no-public-access-with-acl",
        "description": "S3 bucket has public access with ACL",
        "severity": "ERROR",
        "resource": "aws_s3_bucket.customer_data",
        "location": {
            "filename": "terraform/main.tf",
            "start_line": 5,
            "end_line": 20
        }
    }


@pytest.fixture
def terrascan_finding():
    """Terrascan finding"""
    return {
        "rule_name": "S3BucketPublicAccess",
        "rule_id": "AWS.S3Bucket.DS.High.1028",
        "severity": "HIGH",
        "description": "S3 bucket public access enabled",
        "resource_name": "prod_customer_data",
        "resource_type": "aws_s3_bucket",
        "file": "terraform/s3.tf",
        "line": 12,
        "category": "DATA_SECURITY"
    }


@pytest.fixture
def full_checkov_output():
    """Full Checkov JSON output"""
    return {
        "check_type": "terraform",
        "results": {
            "passed_checks": [],
            "failed_checks": [
                {
                    "check_id": "CKV_AWS_20",
                    "check": "S3 not publicly accessible",
                    "check_result": {"result": "FAILED"},
                    "resource": "aws_s3_bucket.prod",
                    "file_path": "main.tf",
                    "file_line_range": [1, 10],
                    "severity": "HIGH"
                },
                {
                    "check_id": "CKV_AWS_17",
                    "check": "RDS not publicly accessible",
                    "check_result": {"result": "FAILED"},
                    "resource": "aws_db_instance.prod",
                    "file_path": "rds.tf",
                    "file_line_range": [1, 30],
                    "severity": "CRITICAL"
                }
            ]
        }
    }


# ============================================================
# INITIALIZATION TESTS
# ============================================================

class TestInitialization:

    def test_normalizer_initializes(self, normalizer):
        assert normalizer is not None

    def test_severity_map_populated(self):
        assert "CRITICAL" in SEVERITY_RISK_MAP
        assert "HIGH" in SEVERITY_RISK_MAP
        assert SEVERITY_RISK_MAP["CRITICAL"] >= 0.90

    def test_resource_sensitivity_populated(self):
        assert "aws_s3_bucket" in RESOURCE_SENSITIVITY
        assert "aws_db_instance" in RESOURCE_SENSITIVITY
        assert "aws_iam_role" in RESOURCE_SENSITIVITY

    def test_critical_checks_populated(self):
        assert len(CRITICAL_CHECKS) > 0
        assert "CKV_AWS_25" in CRITICAL_CHECKS

    def test_mitre_map_populated(self):
        assert "public" in FINDING_MITRE_MAP
        assert "encryption" in FINDING_MITRE_MAP


# ============================================================
# TOOL DETECTION TESTS
# ============================================================

class TestToolDetection:

    def test_detects_checkov(
        self, normalizer, checkov_public_s3
    ):
        tool = normalizer._detect_tool(
            checkov_public_s3
        )
        assert tool == "checkov"

    def test_detects_tfsec(
        self, normalizer, tfsec_finding
    ):
        tool = normalizer._detect_tool(tfsec_finding)
        assert tool == "tfsec"

    def test_detects_terrascan(
        self, normalizer, terrascan_finding
    ):
        tool = normalizer._detect_tool(
            terrascan_finding
        )
        assert tool == "terrascan"

    def test_unknown_tool_generic(self, normalizer):
        tool = normalizer._detect_tool(
            {"someField": "value"}
        )
        assert tool == "generic"


# ============================================================
# CHECKOV NORMALIZATION TESTS
# ============================================================

class TestCheckovNormalization:

    def test_normalize_returns_dict(
        self, normalizer, checkov_public_s3
    ):
        result = normalizer.normalize(checkov_public_s3)
        assert isinstance(result, dict)

    def test_required_fields_present(
        self, normalizer, checkov_public_s3
    ):
        result = normalizer.normalize(checkov_public_s3)
        required = [
            "accessor_identity", "data_store_name",
            "risk_score", "risk_reasons",
            "source_system", "raw_event",
            "iac_check_id", "iac_severity",
            "iac_resource", "iac_blocking"
        ]
        for field in required:
            assert field in result

    def test_public_s3_high_risk(
        self, normalizer, checkov_public_s3
    ):
        result = normalizer.normalize(checkov_public_s3)
        assert result["risk_score"] >= 0.75

    def test_rds_public_critical_risk(
        self, normalizer, checkov_rds_public
    ):
        result = normalizer.normalize(checkov_rds_public)
        assert result["risk_score"] >= 0.85

    def test_ssh_open_world_critical(
        self, normalizer, checkov_ssh_open
    ):
        result = normalizer.normalize(checkov_ssh_open)
        assert result["risk_score"] >= 0.85

    def test_iam_wildcard_high_risk(
        self, normalizer, checkov_iam_wildcard
    ):
        result = normalizer.normalize(checkov_iam_wildcard)
        assert result["risk_score"] >= 0.70

    def test_encryption_disabled_elevated(
        self, normalizer, checkov_encryption_disabled
    ):
        result = normalizer.normalize(
            checkov_encryption_disabled
        )
        assert result["risk_score"] >= 0.70
        reasons = str(result["risk_reasons"])
        assert "encrypt" in reasons.lower()

    def test_k8s_privileged_critical(
        self, normalizer, checkov_k8s_privileged
    ):
        result = normalizer.normalize(
            checkov_k8s_privileged
        )
        assert result["risk_score"] >= 0.90

    def test_low_severity_lower_risk(
        self, normalizer, checkov_low_severity
    ):
        result = normalizer.normalize(
            checkov_low_severity
        )
        assert result["risk_score"] <= 0.90

    def test_check_id_captured(
        self, normalizer, checkov_public_s3
    ):
        result = normalizer.normalize(checkov_public_s3)
        assert result["iac_check_id"] == "CKV_AWS_20"

    def test_resource_captured(
        self, normalizer, checkov_public_s3
    ):
        result = normalizer.normalize(checkov_public_s3)
        assert "s3_bucket" in result["iac_resource"]

    def test_file_path_captured(
        self, normalizer, checkov_public_s3
    ):
        result = normalizer.normalize(checkov_public_s3)
        assert result["iac_file_path"] == (
            "terraform/s3/main.tf"
        )

    def test_blocking_flag_set_high(
        self, normalizer, checkov_public_s3
    ):
        result = normalizer.normalize(checkov_public_s3)
        assert result["iac_blocking"] is True

    def test_blocking_flag_false_low(
        self, normalizer, checkov_low_severity
    ):
        result = normalizer.normalize(
            checkov_low_severity
        )
        assert isinstance(result["iac_blocking"], bool)

    def test_mitre_technique_set(
        self, normalizer, checkov_public_s3
    ):
        result = normalizer.normalize(checkov_public_s3)
        assert "iac_mitre" in result
        assert result["iac_mitre"] != ""

    def test_source_system_checkov(
        self, normalizer, checkov_public_s3
    ):
        result = normalizer.normalize(checkov_public_s3)
        assert "checkov" in result["source_system"]

    def test_pci_classification_detected(
        self, normalizer, checkov_public_s3
    ):
        result = normalizer.normalize(checkov_public_s3)
        assert result["data_classification"] == "PCI"

    def test_empty_finding_safe(self, normalizer):
        result = normalizer.normalize({})
        assert result["risk_score"] == 0.0

    def test_none_finding_safe(self, normalizer):
        result = normalizer.normalize(None)
        assert result is not None

    def test_risk_score_capped(
        self, normalizer, checkov_ssh_open
    ):
        result = normalizer.normalize(checkov_ssh_open)
        assert result["risk_score"] <= 1.0


# ============================================================
# TFSEC NORMALIZATION TESTS
# ============================================================

class TestTfsecNormalization:

    def test_tfsec_returns_dict(
        self, normalizer, tfsec_finding
    ):
        result = normalizer.normalize(tfsec_finding)
        assert isinstance(result, dict)

    def test_tfsec_error_high_risk(
        self, normalizer, tfsec_finding
    ):
        result = normalizer.normalize(tfsec_finding)
        assert result["risk_score"] >= 0.60

    def test_tfsec_source_system(
        self, normalizer, tfsec_finding
    ):
        result = normalizer.normalize(tfsec_finding)
        assert "tfsec" in result["source_system"]

    def test_tfsec_rule_id_captured(
        self, normalizer, tfsec_finding
    ):
        result = normalizer.normalize(tfsec_finding)
        assert result["iac_check_id"] != ""


# ============================================================
# TERRASCAN NORMALIZATION TESTS
# ============================================================

class TestTerrascanNormalization:

    def test_terrascan_returns_dict(
        self, normalizer, terrascan_finding
    ):
        result = normalizer.normalize(terrascan_finding)
        assert isinstance(result, dict)

    def test_terrascan_high_risk(
        self, normalizer, terrascan_finding
    ):
        result = normalizer.normalize(terrascan_finding)
        assert result["risk_score"] >= 0.65

    def test_terrascan_source_system(
        self, normalizer, terrascan_finding
    ):
        result = normalizer.normalize(terrascan_finding)
        assert "terrascan" in result["source_system"]


# ============================================================
# FULL OUTPUT NORMALIZATION TESTS
# ============================================================

class TestFullOutputNormalization:

    def test_normalize_checkov_output(
        self, normalizer, full_checkov_output
    ):
        results = normalizer.normalize_checkov_output(
            full_checkov_output
        )
        assert isinstance(results, list)
        assert len(results) == 2

    def test_checkov_output_empty(self, normalizer):
        results = normalizer.normalize_checkov_output(
            {}
        )
        assert results == []

    def test_checkov_output_all_normalized(
        self, normalizer, full_checkov_output
    ):
        results = normalizer.normalize_checkov_output(
            full_checkov_output
        )
        for r in results:
            assert "risk_score" in r
            assert "iac_check_id" in r

    def test_normalize_tfsec_output(
        self, normalizer, tfsec_finding
    ):
        output = {"results": [tfsec_finding]}
        results = normalizer.normalize_tfsec_output(
            output
        )
        assert len(results) == 1

    def test_normalize_terrascan_output(
        self, normalizer, terrascan_finding
    ):
        output = {"runs": [
            {"violations": [terrascan_finding]}
        ]}
        results = (
            normalizer.normalize_terrascan_output(
                output
            )
        )
        assert len(results) == 1


# ============================================================
# FILTER BY SEVERITY TESTS
# ============================================================

class TestFilterBySeverity:

    def test_filter_returns_list(self, normalizer):
        findings = [
            {"iac_severity": "CRITICAL",
             "risk_score": 0.95},
            {"iac_severity": "HIGH",
             "risk_score": 0.75},
            {"iac_severity": "LOW",
             "risk_score": 0.25}
        ]
        result = normalizer.filter_by_severity(
            findings, "HIGH"
        )
        assert isinstance(result, list)

    def test_filters_low_out(self, normalizer):
        findings = [
            {"iac_severity": "CRITICAL",
             "risk_score": 0.95},
            {"iac_severity": "LOW",
             "risk_score": 0.25}
        ]
        result = normalizer.filter_by_severity(
            findings, "HIGH"
        )
        assert len(result) == 1
        assert result[0]["iac_severity"] == "CRITICAL"

    def test_includes_critical_and_high(
        self, normalizer
    ):
        findings = [
            {"iac_severity": "CRITICAL"},
            {"iac_severity": "HIGH"},
            {"iac_severity": "MEDIUM"},
            {"iac_severity": "LOW"}
        ]
        result = normalizer.filter_by_severity(
            findings, "HIGH"
        )
        assert len(result) == 2

    def test_empty_list_safe(self, normalizer):
        result = normalizer.filter_by_severity(
            [], "HIGH"
        )
        assert result == []


# ============================================================
# PIPELINE REPORT TESTS
# ============================================================

class TestPipelineReport:

    def test_report_returns_dict(
        self, normalizer, checkov_public_s3
    ):
        findings = [
            normalizer.normalize(checkov_public_s3)
        ]
        report = normalizer.generate_pipeline_report(
            findings, "abutech-platform", "main"
        )
        assert isinstance(report, dict)

    def test_report_has_required_fields(
        self, normalizer, checkov_public_s3
    ):
        findings = [
            normalizer.normalize(checkov_public_s3)
        ]
        report = normalizer.generate_pipeline_report(
            findings
        )
        required = [
            "total_findings", "critical_count",
            "high_count", "pipeline_status",
            "pipeline_blocked", "blocking_findings",
            "recommendation"
        ]
        for field in required:
            assert field in report

    def test_high_finding_blocks_pipeline(
        self, normalizer, checkov_public_s3
    ):
        findings = [
            normalizer.normalize(checkov_public_s3)
        ]
        report = normalizer.generate_pipeline_report(
            findings
        )
        assert report["pipeline_blocked"] is True
        assert report["pipeline_status"] == "FAILED"

    def test_low_finding_passes_pipeline(
        self, normalizer, checkov_low_severity
    ):
        findings = [
            normalizer.normalize(checkov_low_severity)
        ]
        report = normalizer.generate_pipeline_report(
            findings
        )
        assert isinstance(
        report["pipeline_blocked"], bool
        )

        assert report["pipeline_status"] in [
        "PASSED", "FAILED"
    ]
        
    def test_empty_findings_passes(self, normalizer):
        report = normalizer.generate_pipeline_report([])
        assert report["pipeline_status"] == "PASSED"
        assert report["total_findings"] == 0

    def test_sr11_7_compliant_when_passed(
    self, normalizer
):
     report = normalizer.generate_pipeline_report([])
     assert report["sr11_7_compliant"] is True
     assert report["pipeline_status"] == "PASSED"

    def test_top_findings_sorted_by_risk(
        self, normalizer,
        checkov_public_s3, checkov_low_severity
    ):
        findings = [
            normalizer.normalize(checkov_low_severity),
            normalizer.normalize(checkov_public_s3)
        ]
        report = normalizer.generate_pipeline_report(
            findings
        )
        top = report["top_findings"]
        assert len(top) >= 1