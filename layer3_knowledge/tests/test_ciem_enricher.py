"""
Tests for Deep CIEM Enricher
"""

import pytest
from layer3_knowledge.enrichment.ciem_enricher import (
    CIEMEnricher,
    AWS_TOXIC_COMBINATIONS,
    AZURE_TOXIC_COMBINATIONS,
    GCP_TOXIC_COMBINATIONS,
    HIGH_RISK_PERMISSIONS
)


@pytest.fixture
def enricher():
    return CIEMEnricher()


@pytest.fixture
def svc_backup_permissions():
    """svc_backup — massively over-privileged"""
    return [
        "s3:GetObject",
        "s3:ListBucket",
        "s3:DeleteObject",
        "kms:Decrypt",
        "iam:PassRole",
        "iam:AttachUserPolicy",
        "iam:CreateUser",
        "ec2:TerminateInstances",
        "cloudtrail:StopLogging",
        "cloudtrail:DeleteTrail",
        "secretsmanager:GetSecretValue",
        "ssm:GetParameter",
        "rds:DeleteDBInstance",
        "lambda:CreateFunction",
        "lambda:InvokeFunction"
    ]


@pytest.fixture
def readonly_permissions():
    """Minimal read-only permissions"""
    return [
        "s3:GetObject",
        "s3:ListBucket"
    ]


@pytest.fixture
def sample_data_event():
    return {
        "accessor_identity": "svc_backup",
        "accessor_type": "service_account",
        "data_store_name": "prod-customer-data",
        "data_path": "customers/pci/cards.csv",
        "risk_score": 0.45,
        "risk_reasons": ["after_hours", "tor_ip"]
    }


@pytest.fixture
def permission_history():
    return [
        {"date": "2025-01-01", "permission_count": 12},
        {"date": "2025-06-01", "permission_count": 145},
        {"date": "2026-01-01", "permission_count": 847}
    ]


@pytest.fixture
def accessible_resources():
    return [
        "s3://prod-customer-pci-data",
        "s3://prod-customer-data",
        "s3://prod-backup-data",
        "rds://prod-db-01",
        "rds://prod-db-02",
        "secretsmanager://prod-db-credentials",
        "kms://prod-encryption-key",
        "s3://pci-card-data-archive"
    ]


# ============================================================
# INITIALIZATION TESTS
# ============================================================

class TestInitialization:

    def test_enricher_initializes(self, enricher):
        assert enricher is not None

    def test_toxic_combos_loaded(self, enricher):
        assert len(enricher.toxic_combos) == 3
        assert "aws" in enricher.toxic_combos
        assert "azure" in enricher.toxic_combos
        assert "gcp" in enricher.toxic_combos

    def test_aws_toxic_combos_populated(self):
        assert len(AWS_TOXIC_COMBINATIONS) >= 5

    def test_high_risk_permissions_populated(self):
        assert "data_access" in HIGH_RISK_PERMISSIONS
        assert "identity_manipulation" in (
            HIGH_RISK_PERMISSIONS
        )
        assert "audit_evasion" in HIGH_RISK_PERMISSIONS


# ============================================================
# ANALYZE IDENTITY TESTS
# ============================================================

class TestAnalyzeIdentity:

    def test_analyze_returns_dict(
        self, enricher, svc_backup_permissions
    ):
        result = enricher.analyze_identity(
            identity="svc_backup",
            permissions=svc_backup_permissions
        )
        assert isinstance(result, dict)

    def test_analyze_has_required_fields(
        self, enricher, svc_backup_permissions
    ):
        result = enricher.analyze_identity(
            identity="svc_backup",
            permissions=svc_backup_permissions
        )
        required = [
            "identity", "cloud", "permission_count",
            "risk_score", "risk_reasons",
            "toxic_combinations", "escalation_paths",
            "unused_permissions", "peer_group_analysis",
            "permission_categories", "recommendation"
        ]
        for field in required:
            assert field in result

    def test_over_privileged_high_risk(
        self, enricher, svc_backup_permissions
    ):
        result = enricher.analyze_identity(
            identity="svc_backup",
            permissions=svc_backup_permissions,
            peer_group_avg=8
        )
        assert result["risk_score"] >= 0.70

    def test_minimal_permissions_low_risk(
        self, enricher, readonly_permissions
    ):
        result = enricher.analyze_identity(
            identity="readonly_user",
            permissions=readonly_permissions,
            peer_group_avg=5
        )
        assert result["risk_score"] <= 0.40

    def test_permission_count_correct(
        self, enricher, svc_backup_permissions
    ):
        result = enricher.analyze_identity(
            identity="svc_backup",
            permissions=svc_backup_permissions
        )
        assert result["permission_count"] == len(
            svc_backup_permissions
        )


# ============================================================
# TOXIC COMBINATION TESTS
# ============================================================

class TestToxicCombinations:

    def test_detect_returns_list(
        self, enricher, svc_backup_permissions
    ):
        result = enricher.detect_toxic_combinations(
            svc_backup_permissions, "aws"
        )
        assert isinstance(result, list)

    def test_detects_data_exfiltration_combo(
        self, enricher
    ):
        perms = ["s3:GetObject", "kms:Decrypt"]
        result = enricher.detect_toxic_combinations(
            perms, "aws"
        )
        assert len(result) >= 1
        names = [r["name"] for r in result]
        assert "Full Data Exfiltration" in names

    def test_detects_shadow_admin_combo(
        self, enricher
    ):
        perms = [
            "iam:CreateUser",
            "iam:AttachUserPolicy"
        ]
        result = enricher.detect_toxic_combinations(
            perms, "aws"
        )
        assert len(result) >= 1

    def test_detects_audit_evasion(self, enricher):
        perms = [
            "cloudtrail:StopLogging",
            "cloudtrail:DeleteTrail"
        ]
        result = enricher.detect_toxic_combinations(
            perms, "aws"
        )
        assert len(result) >= 1
        names = [r["name"] for r in result]
        assert "CloudTrail Blind Spot" in names

    def test_no_toxics_for_readonly(
        self, enricher, readonly_permissions
    ):
        result = enricher.detect_toxic_combinations(
            readonly_permissions, "aws"
        )
        assert len(result) == 0

    def test_toxic_has_mitre_technique(
        self, enricher
    ):
        perms = ["s3:GetObject", "kms:Decrypt"]
        result = enricher.detect_toxic_combinations(
            perms, "aws"
        )
        if result:
            assert "mitre_technique" in result[0]
            assert result[0]["mitre_technique"] != ""

    def test_toxic_has_attack_description(
        self, enricher
    ):
        perms = ["s3:GetObject", "kms:Decrypt"]
        result = enricher.detect_toxic_combinations(
            perms, "aws"
        )
        if result:
            assert "attack_description" in result[0]

    def test_multiple_toxics_detected(
        self, enricher, svc_backup_permissions
    ):
        result = enricher.detect_toxic_combinations(
            svc_backup_permissions, "aws"
        )
        assert len(result) >= 2


# ============================================================
# PRIVILEGE ESCALATION TESTS
# ============================================================

class TestPrivilegeEscalation:

    def test_find_paths_returns_dict(
        self, enricher, svc_backup_permissions
    ):
        result = enricher.find_escalation_paths(
            "svc_backup",
            svc_backup_permissions,
            "aws"
        )
        assert isinstance(result, dict)

    def test_escalation_path_detected(
        self, enricher
    ):
        perms = ["iam:PassRole", "ec2:RunInstances"]
        result = enricher.find_escalation_paths(
            "svc_backup", perms, "aws"
        )
        assert result["has_escalation_path"] is True

    def test_no_escalation_for_readonly(
        self, enricher, readonly_permissions
    ):
        result = enricher.find_escalation_paths(
            "readonly", readonly_permissions, "aws"
        )
        assert result["has_escalation_path"] is False

    def test_escalation_result_has_paths(
        self, enricher, svc_backup_permissions
    ):
        result = enricher.find_escalation_paths(
            "svc_backup",
            svc_backup_permissions,
            "aws"
        )
        if result["has_escalation_path"]:
            assert len(result["paths"]) > 0
            assert "permission" in result["paths"][0]
            assert "enables" in result["paths"][0]

    def test_escalation_max_risk_score(
        self, enricher, svc_backup_permissions
    ):
        result = enricher.find_escalation_paths(
            "svc_backup",
            svc_backup_permissions,
            "aws"
        )
        assert 0.0 <= result["max_risk"] <= 1.0

    def test_gcp_escalation_handled(
        self, enricher, svc_backup_permissions
    ):
        result = enricher.find_escalation_paths(
            "svc-account",
            svc_backup_permissions,
            "gcp"
        )
        assert isinstance(result, dict)
        assert "has_escalation_path" in result


# ============================================================
# BLAST RADIUS TESTS
# ============================================================

class TestBlastRadius:

    def test_blast_radius_returns_dict(
        self, enricher, accessible_resources
    ):
        result = enricher.calculate_blast_radius(
            "svc_backup", accessible_resources
        )
        assert isinstance(result, dict)

    def test_blast_radius_has_required_fields(
        self, enricher, accessible_resources
    ):
        result = enricher.calculate_blast_radius(
            "svc_backup", accessible_resources
        )
        required = [
            "identity",
            "total_accessible_resources",
            "s3_buckets_at_risk",
            "databases_at_risk",
            "secrets_at_risk",
            "estimated_records_at_risk",
            "estimated_breach_cost_usd",
            "blast_radius_risk_score",
            "severity"
        ]
        for field in required:
            assert field in result

    def test_pci_resources_counted(
        self, enricher, accessible_resources
    ):
        result = enricher.calculate_blast_radius(
            "svc_backup", accessible_resources
        )
        assert result["pci_resources_at_risk"] >= 1

    def test_breach_cost_estimated(
        self, enricher, accessible_resources
    ):
        result = enricher.calculate_blast_radius(
            "svc_backup", accessible_resources
        )
        assert result[
            "estimated_breach_cost_usd"
        ] > 0

    def test_empty_resources_low_risk(
        self, enricher
    ):
        result = enricher.calculate_blast_radius(
            "svc_backup", []
        )
        assert result[
            "blast_radius_risk_score"
        ] <= 0.30

    def test_many_resources_high_risk(
        self, enricher, accessible_resources
    ):
        result = enricher.calculate_blast_radius(
            "svc_backup", accessible_resources * 10
        )
        assert result["blast_radius_risk_score"] >= 0.5


# ============================================================
# PEER GROUP ANALYSIS TESTS
# ============================================================

class TestPeerGroupAnalysis:

    def test_analyze_returns_dict(self, enricher):
        result = enricher.analyze_peer_group(
            847, 8
        )
        assert isinstance(result, dict)

    def test_extreme_anomaly_detected(
        self, enricher
    ):
        result = enricher.analyze_peer_group(
            847, 8
        )
        assert result["is_anomalous"] is True
        assert result["anomaly_label"] == "EXTREME"
        assert result["anomaly_score"] >= 0.85

    def test_normal_not_anomalous(self, enricher):
        result = enricher.analyze_peer_group(10, 8)
        assert result["is_anomalous"] is False
        assert result["anomaly_label"] == "NORMAL"

    def test_ratio_calculated_correctly(
        self, enricher
    ):
        result = enricher.analyze_peer_group(80, 8)
        assert result["ratio"] == 10.0

    def test_high_percentile_for_anomaly(
        self, enricher
    ):
        result = enricher.analyze_peer_group(
            847, 8
        )
        assert result["estimated_percentile"] >= 95.0


# ============================================================
# UNUSED PERMISSIONS TESTS
# ============================================================

class TestUnusedPermissions:

    def test_unused_returns_list(self, enricher):
        result = enricher.identify_unused_permissions(
            ["s3:GetObject", "ec2:TerminateInstances"],
            {"s3:GetObject": 5, "ec2:TerminateInstances": -1}
        )
        assert isinstance(result, list)

    def test_never_used_detected(self, enricher):
        result = enricher.identify_unused_permissions(
            ["ec2:TerminateInstances"],
            {"ec2:TerminateInstances": -1}
        )
        assert len(result) == 1
        assert result[0][
            "days_since_used"
        ] == "Never"

    def test_high_risk_unused_flagged(
        self, enricher
    ):
        result = enricher.identify_unused_permissions(
            ["cloudtrail:StopLogging"],
            {"cloudtrail:StopLogging": -1}
        )
        assert len(result) >= 1
        assert result[0]["is_high_risk"] is True

    def test_recently_used_not_flagged(
        self, enricher
    ):
        result = enricher.identify_unused_permissions(
            ["s3:GetObject"],
            {"s3:GetObject": 5}
        )
        assert len(result) == 0

    def test_old_permission_flagged(self, enricher):
        result = enricher.identify_unused_permissions(
            ["s3:ListBucket"],
            {"s3:ListBucket": 120}
        )
        assert len(result) == 1


# ============================================================
# PERMISSION CREEP TESTS
# ============================================================

class TestPermissionCreep:

    def test_creep_returns_dict(
        self, enricher, permission_history
    ):
        result = enricher.detect_permission_creep(
            permission_history
        )
        assert isinstance(result, dict)

    def test_creep_detected(
        self, enricher, permission_history
    ):
        result = enricher.detect_permission_creep(
            permission_history
        )
        assert result["creep_detected"] is True

    def test_creep_growth_calculated(
        self, enricher, permission_history
    ):
        result = enricher.detect_permission_creep(
            permission_history
        )
        assert result["total_growth"] == 835
        assert result["initial_count"] == 12
        assert result["final_count"] == 847

    def test_high_growth_critical_severity(
        self, enricher, permission_history
    ):
        result = enricher.detect_permission_creep(
            permission_history
        )
        assert result["severity"] in [
            "CRITICAL", "HIGH"
        ]

    def test_stable_permissions_no_creep(
        self, enricher
    ):
        stable = [
            {"date": "2025-01-01", "permission_count": 10},
            {"date": "2025-06-01", "permission_count": 11},
            {"date": "2026-01-01", "permission_count": 12}
        ]
        result = enricher.detect_permission_creep(
            stable
        )
        assert result["creep_detected"] is False

    def test_single_entry_no_creep(self, enricher):
        result = enricher.detect_permission_creep(
            [{"date": "2026-01-01",
              "permission_count": 10}]
        )
        assert result["creep_detected"] is False


# ============================================================
# ENRICH EVENT TESTS
# ============================================================

class TestEnrichEvent:

    def test_enrich_returns_dict(
        self, enricher, sample_data_event
    ):
        result = enricher.enrich_event(
            sample_data_event
        )
        assert isinstance(result, dict)

    def test_enrich_elevates_risk(
        self, enricher, sample_data_event
    ):
        result = enricher.enrich_event(
            sample_data_event,
            identity_permissions=[
                "s3:GetObject",
                "kms:Decrypt",
                "iam:PassRole"
            ]
        )
        original = sample_data_event["risk_score"]
        assert result["risk_score"] >= original

    def test_enrich_adds_ciem_analysis(
        self, enricher, sample_data_event
    ):
        result = enricher.enrich_event(
            sample_data_event
        )
        assert "ciem_analysis" in result

    def test_enrich_score_capped(
        self, enricher, sample_data_event
    ):
        result = enricher.enrich_event(
            sample_data_event
        )
        assert result["risk_score"] <= 1.0

    def test_enrich_preserves_original_fields(
        self, enricher, sample_data_event
    ):
        result = enricher.enrich_event(
            sample_data_event
        )
        assert result["accessor_identity"] == (
            "svc_backup"
        )
        assert result["data_store_name"] == (
            "prod-customer-data"
        )