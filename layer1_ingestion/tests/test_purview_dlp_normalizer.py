"""
Tests for Microsoft Purview DLP Normalizer
"""

import pytest
from layer1_ingestion.normalizers.purview_dlp_normalizer\
    import (
        PurviewDLPNormalizer,
        SENSITIVE_INFO_TYPES,
        DLP_ACTION_RISK,
        DLP_WORKLOAD_TECHNIQUE
    )


@pytest.fixture
def normalizer():
    return PurviewDLPNormalizer()


@pytest.fixture
def email_pci_external_allowed():
    """Credit card data emailed externally, allowed"""
    return {
        "UserId": "john.smith@company.com",
        "Workload": "Exchange",
        "PolicyName": "PCI Data Protection",
        "DLPAction": "Allow",
        "Subject": "customer card export.xlsx",
        "SensitiveInfoTypeData": [
            {"SensitiveType": "Credit Card Number"}
        ],
        "Recipients": ["attacker@external-evil.com"],
        "ClientIP": "10.0.0.45",
        "CreationTime": "2026-06-01T14:00:00Z"
    }


@pytest.fixture
def email_pci_blocked():
    """Credit card data emailed externally, BLOCKED"""
    return {
        "UserId": "jane.doe@company.com",
        "Workload": "Exchange",
        "PolicyName": "PCI Data Protection",
        "DLPAction": "Block",
        "Subject": "card numbers.xlsx",
        "SensitiveInfoTypeData": [
            {"SensitiveType": "Credit Card Number"}
        ],
        "Recipients": ["someone@external-evil.com"],
        "ClientIP": "10.0.0.50"
    }


@pytest.fixture
def sharepoint_phi_share():
    """PHI shared via SharePoint"""
    return {
        "UserId": "bob@company.com",
        "Workload": "SharePoint",
        "PolicyName": "HIPAA PHI Protection",
        "DLPAction": "BlockOverride",
        "ObjectId": "/sites/medical/patient_records.xlsx",
        "SensitiveInfoTypeData": [
            {"SensitiveType": "Medical Record Number"}
        ],
        "ClientIP": "10.0.1.20"
    }


@pytest.fixture
def endpoint_usb_pii():
    """PII copied to USB from endpoint"""
    return {
        "UserId": "carol@company.com",
        "Workload": "Endpoint",
        "PolicyName": "PII Endpoint Protection",
        "DLPAction": "Audit",
        "ObjectId": "ssn_list.csv",
        "SensitiveInfoTypeData": [
            {"SensitiveType": "U.S. Social Security Number"}
        ],
        "ClientIP": "10.0.2.30"
    }


@pytest.fixture
def teams_pci_tor():
    """PCI posted in Teams from Tor IP"""
    return {
        "UserId": "dave@company.com",
        "Workload": "Teams",
        "PolicyName": "PCI Protection",
        "DLPAction": "Allow",
        "ObjectId": "message-12345",
        "SensitiveInfoTypeData": [
            {"SensitiveType": "Credit Card Number"}
        ],
        "ClientIP": "185.220.101.45"
    }


# ============================================================
# INITIALIZATION
# ============================================================

class TestInitialization:

    def test_normalizer_initializes(self, normalizer):
        assert normalizer is not None
        assert normalizer.source_system == (
            "purview_dlp"
        )

    def test_sensitive_types_populated(self):
        assert "credit card number" in (
            SENSITIVE_INFO_TYPES
        )
        assert SENSITIVE_INFO_TYPES[
            "credit card number"
        ][0] == "PCI"

    def test_action_risk_populated(self):
        assert "block" in DLP_ACTION_RISK
        assert DLP_ACTION_RISK["block"] == 0.0
        assert DLP_ACTION_RISK["blockoverride"] > 0

    def test_workload_technique_populated(self):
        assert "exchange" in DLP_WORKLOAD_TECHNIQUE


# ============================================================
# PCI EMAIL SCENARIOS
# ============================================================

class TestPCIEmail:

    def test_returns_dict(
        self, normalizer, email_pci_external_allowed
    ):
        result = normalizer.normalize(
            email_pci_external_allowed
        )
        assert isinstance(result, dict)

    def test_pci_classification(
        self, normalizer, email_pci_external_allowed
    ):
        result = normalizer.normalize(
            email_pci_external_allowed
        )
        assert result["data_classification"] == (
            "PCI"
        )

    def test_external_recipient_detected(
        self, normalizer, email_pci_external_allowed
    ):
        result = normalizer.normalize(
            email_pci_external_allowed
        )
        assert result["dlp_external_recipient"] is (
            True
        )

    def test_allowed_pci_external_high_risk(
        self, normalizer, email_pci_external_allowed
    ):
        result = normalizer.normalize(
            email_pci_external_allowed
        )
        # PCI left org externally, not blocked
        assert result["risk_score"] >= 0.85
        assert "pci_exfiltrated" in (
            result["risk_reasons"]
        )

    def test_exchange_technique(
        self, normalizer, email_pci_external_allowed
    ):
        result = normalizer.normalize(
            email_pci_external_allowed
        )
        assert result["mitre_technique"] == "T1048"

    def test_blocked_pci_lower_risk(
        self, normalizer, email_pci_blocked
    ):
        result = normalizer.normalize(
            email_pci_blocked
        )
        # DLP blocked the data loss
        assert result["risk_score"] <= 0.45
        assert "dlp_blocked_data_loss" in (
            result["risk_reasons"]
        )

    def test_blocked_vs_allowed_risk_difference(
        self, normalizer,
        email_pci_external_allowed,
        email_pci_blocked
    ):
        allowed = normalizer.normalize(
            email_pci_external_allowed
        )
        blocked = normalizer.normalize(
            email_pci_blocked
        )
        # Allowed exfil must score higher than blocked
        assert allowed["risk_score"] > (
            blocked["risk_score"]
        )


# ============================================================
# SHAREPOINT PHI
# ============================================================

class TestSharePointPHI:

    def test_phi_classification(
        self, normalizer, sharepoint_phi_share
    ):
        result = normalizer.normalize(
            sharepoint_phi_share
        )
        assert result["data_classification"] == (
            "PHI"
        )

    def test_sharepoint_technique(
        self, normalizer, sharepoint_phi_share
    ):
        result = normalizer.normalize(
            sharepoint_phi_share
        )
        assert result["mitre_technique"] == "T1530"

    def test_override_action(
        self, normalizer, sharepoint_phi_share
    ):
        result = normalizer.normalize(
            sharepoint_phi_share
        )
        assert result["dlp_action"] == (
            "blockoverride"
        )

    def test_workload_label(
        self, normalizer, sharepoint_phi_share
    ):
        result = normalizer.normalize(
            sharepoint_phi_share
        )
        assert result["data_store_name"] == (
            "SharePoint Online"
        )


# ============================================================
# ENDPOINT DLP
# ============================================================

class TestEndpointDLP:

    def test_pii_classification(
        self, normalizer, endpoint_usb_pii
    ):
        result = normalizer.normalize(
            endpoint_usb_pii
        )
        assert result["data_classification"] == (
            "PII"
        )

    def test_endpoint_technique(
        self, normalizer, endpoint_usb_pii
    ):
        result = normalizer.normalize(
            endpoint_usb_pii
        )
        assert result["mitre_technique"] == "T1052"

    def test_endpoint_label(
        self, normalizer, endpoint_usb_pii
    ):
        result = normalizer.normalize(
            endpoint_usb_pii
        )
        assert result["data_store_name"] == (
            "Endpoint DLP"
        )

    def test_audit_action_data_left(
        self, normalizer, endpoint_usb_pii
    ):
        result = normalizer.normalize(
            endpoint_usb_pii
        )
        assert result["dlp_action"] == "audit"


# ============================================================
# TEAMS + TOR
# ============================================================

class TestTeamsTor:

    def test_tor_ip_flagged(
        self, normalizer, teams_pci_tor
    ):
        result = normalizer.normalize(
            teams_pci_tor
        )
        assert "tor_exit_node_src" in (
            result["risk_reasons"]
        )

    def test_tor_high_risk(
        self, normalizer, teams_pci_tor
    ):
        result = normalizer.normalize(
            teams_pci_tor
        )
        assert result["risk_score"] >= 0.90

    def test_teams_technique(
        self, normalizer, teams_pci_tor
    ):
        result = normalizer.normalize(
            teams_pci_tor
        )
        assert result["mitre_technique"] == (
            "T1567.002"
        )


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
            "purview_dlp"
        )

    def test_risk_never_exceeds_one(
        self, normalizer, teams_pci_tor
    ):
        result = normalizer.normalize(
            teams_pci_tor
        )
        assert result["risk_score"] <= 1.0

    def test_risk_never_negative(
        self, normalizer, email_pci_blocked
    ):
        result = normalizer.normalize(
            email_pci_blocked
        )
        assert result["risk_score"] >= 0.0

    def test_raw_event_preserved(
        self, normalizer, endpoint_usb_pii
    ):
        result = normalizer.normalize(
            endpoint_usb_pii
        )
        assert result["raw_event"] == (
            endpoint_usb_pii
        )

    def test_event_time_present(
        self, normalizer, email_pci_external_allowed
    ):
        result = normalizer.normalize(
            email_pci_external_allowed
        )
        assert result["event_time"] != ""

    def test_accessor_type_human(
        self, normalizer, endpoint_usb_pii
    ):
        result = normalizer.normalize(
            endpoint_usb_pii
        )
        assert result["accessor_type"] == "human"

    def test_sensitive_types_captured(
        self, normalizer, email_pci_external_allowed
    ):
        result = normalizer.normalize(
            email_pci_external_allowed
        )
        assert len(
            result["dlp_sensitive_types"]
        ) > 0


# ============================================================
# RECIPIENT DETECTION
# ============================================================

class TestRecipientDetection:

    def test_external_recipient_true(
        self, normalizer
    ):
        result = normalizer._has_external_recipient(
            ["attacker@external-evil.com"]
        )
        assert result is True

    def test_internal_recipient_false(
        self, normalizer
    ):
        result = normalizer._has_external_recipient(
            ["colleague@company.com"]
        )
        assert result is False

    def test_no_recipients_false(self, normalizer):
        result = normalizer._has_external_recipient(
            []
        )
        assert result is False