"""
Tests for Microsoft Sentinel Normalizer
"""

import pytest
from layer1_ingestion.normalizers.sentinel_normalizer import (
    SentinelNormalizer,
    SENTINEL_TABLE_MAP,
    ASIM_FIELD_MAP
)


@pytest.fixture
def normalizer():
    return SentinelNormalizer()


@pytest.fixture
def signin_event():
    """Entra ID sign-in log from Sentinel"""
    return {
        "Type": "SigninLogs",
        "TimeGenerated": "2026-05-21T03:14:22Z",
        "UserPrincipalName": "jsmith@company.com",
        "IPAddress": "185.220.101.45",
        "Location": "Amsterdam, Netherlands",
        "ResultType": "0",
        "AppDisplayName": "Microsoft SharePoint",
        "ResourceDisplayName": "prod-sharepoint",
        "RiskLevel": "High",
        "RiskState": "atRisk",
        "ConditionalAccessStatus": "notApplied",
        "AuthenticationRequirement": "singleFactorAuthentication"
    }


@pytest.fixture
def security_alert():
    """Sentinel SecurityAlert event"""
    return {
        "Type": "SecurityAlert",
        "TimeGenerated": "2026-05-21T03:00:00Z",
        "AlertName": "Suspicious PowerShell Activity",
        "AlertSeverity": "High",
        "Description": "Encoded PowerShell detected spawned by winword.exe",
        "AccountName": "jsmith",
        "SourceIP": "192.168.1.105",
        "RemediationSteps": "Isolate host and investigate",
        "Techniques": ["T1059.001", "T1566.001"],
        "Entities": [
            {"Type": "host", "HostName": "workstation-jsmith"},
            {"Type": "account", "Name": "jsmith"}
        ],
        "SystemAlertId": "alert-abc-123"
    }


@pytest.fixture
def firewall_event():
    """Palo Alto firewall log via CommonSecurityLog"""
    return {
        "Type": "CommonSecurityLog",
        "TimeGenerated": "2026-05-21T03:14:22Z",
        "SourceIP": "10.0.1.105",
        "DestinationHostName": "198.51.100.42",
        "SentBytes": 15234567,
        "DestinationPort": 443,
        "DeviceAction": "allow",
        "Protocol": "TCP",
        "DeviceVendor": "Palo Alto Networks",
        "Activity": "TRAFFIC"
    }


@pytest.fixture
def endpoint_event():
    """CrowdStrike/Defender endpoint event"""
    return {
        "Type": "DeviceProcessEvents",
        "TimeGenerated": "2026-05-21T03:14:00Z",
        "AccountName": "DOMAIN\\jsmith",
        "FileName": "powershell.exe",
        "FolderPath": "C:\\Windows\\System32",
        "ProcessCommandLine": "powershell.exe -enc JABzAHQAcgA",
        "InitiatingProcessFileName": "winword.exe",
        "InitiatingProcessCommandLine": "winword.exe /w",
        "DeviceName": "workstation-jsmith",
        "RemoteIP": "192.168.1.1"
    }


@pytest.fixture
def dns_event():
    """DNS query log from Zeek/Sentinel"""
    return {
        "Type": "DnsEvents",
        "TimeGenerated": "2026-05-21T03:14:00Z",
        "ClientIP": "10.0.1.105",
        "Name": "xkj92mslp.duckdns.org",
        "QueryType": "A",
        "IPAddresses": "198.51.100.42",
        "SubType": "LookupQuery",
        "Computer": "workstation-jsmith"
    }


@pytest.fixture
def cloud_event():
    """AWS CloudTrail event via Sentinel"""
    return {
        "Type": "AWSCloudTrail",
        "TimeGenerated": "2026-05-21T03:00:00Z",
        "UserIdentityUserName": "svc_backup",
        "EventName": "GetObject",
        "SourceIPAddress": "185.220.101.45",
        "RequestParameters": "prod-customer-data/customers/pci/cards.csv",
        "AdditionalEventData": "bytes=524288000"
    }


# ============================================================
# INITIALIZATION TESTS
# ============================================================

class TestInitialization:

    def test_normalizer_initializes(self, normalizer):
        assert normalizer is not None

    def test_table_map_has_entries(self):
        assert len(SENTINEL_TABLE_MAP) > 0
        assert "SigninLogs" in SENTINEL_TABLE_MAP
        assert "SecurityAlert" in SENTINEL_TABLE_MAP
        assert "CommonSecurityLog" in SENTINEL_TABLE_MAP

    def test_field_map_has_entries(self):
        assert len(ASIM_FIELD_MAP) > 0
        assert "UserPrincipalName" in ASIM_FIELD_MAP
        assert "SourceIP" in ASIM_FIELD_MAP
        assert "TimeGenerated" in ASIM_FIELD_MAP


# ============================================================
# CORE NORMALIZATION TESTS
# ============================================================

class TestNormalize:

    def test_normalize_returns_dict(
        self, normalizer, signin_event
    ):
        result = normalizer.normalize(signin_event)
        assert isinstance(result, dict)

    def test_normalize_has_required_fields(
        self, normalizer, signin_event
    ):
        result = normalizer.normalize(signin_event)
        required = [
            "accessor_identity",
            "accessor_type",
            "data_store_name",
            "data_path",
            "data_classification",
            "bytes_accessed",
            "event_time",
            "source_ip",
            "risk_score",
            "risk_reasons",
            "source_system",
            "raw_event"
        ]
        for field in required:
            assert field in result

    def test_normalize_empty_event(self, normalizer):
        result = normalizer.normalize({})
        assert result["accessor_identity"] == "unknown"
        assert result["risk_score"] == 0.0

    def test_normalize_none_event(self, normalizer):
        result = normalizer.normalize(None)
        assert result is not None
        assert result["accessor_identity"] == "unknown"

    def test_source_system_set_correctly(
        self, normalizer, signin_event
    ):
        result = normalizer.normalize(signin_event)
        assert "sentinel" in result["source_system"]
        assert "entraid" in result["source_system"]


# ============================================================
# FIELD EXTRACTION TESTS
# ============================================================

class TestFieldExtraction:

    def test_extracts_upn_as_accessor(
        self, normalizer, signin_event
    ):
        result = normalizer.normalize(signin_event)
        assert result["accessor_identity"] == (
            "jsmith@company.com"
        )

    def test_extracts_domain_username(
        self, normalizer, endpoint_event
    ):
        result = normalizer.normalize(endpoint_event)
        assert result["accessor_identity"] == "jsmith"

    def test_extracts_source_ip(
        self, normalizer, signin_event
    ):
        result = normalizer.normalize(signin_event)
        assert result["source_ip"] == "185.220.101.45"

    def test_extracts_firewall_bytes(
        self, normalizer, firewall_event
    ):
        result = normalizer.normalize(firewall_event)
        assert result["bytes_accessed"] == 15234567

    def test_extracts_event_time(
        self, normalizer, signin_event
    ):
        result = normalizer.normalize(signin_event)
        assert "2026-05-21" in result["event_time"]

    def test_raw_event_preserved(
        self, normalizer, signin_event
    ):
        result = normalizer.normalize(signin_event)
        assert result["raw_event"] == signin_event


# ============================================================
# RISK SCORING TESTS
# ============================================================

class TestRiskScoring:

    def test_tor_ip_elevates_risk(
        self, normalizer, signin_event
    ):
        result = normalizer.normalize(signin_event)
        assert result["risk_score"] >= 0.4
        reasons_str = str(result["risk_reasons"])
        assert "tor" in reasons_str.lower()

    def test_after_hours_elevates_risk(
        self, normalizer, signin_event
    ):
        result = normalizer.normalize(signin_event)
        assert result["risk_score"] > 0.0
        reasons_str = str(result["risk_reasons"])
        assert "after_hours" in reasons_str

    def test_high_severity_elevates_risk(
        self, normalizer, security_alert
    ):
        result = normalizer.normalize(security_alert)
        assert result["risk_score"] >= 0.3

    def test_large_volume_elevates_risk(
        self, normalizer, firewall_event
    ):
        result = normalizer.normalize(firewall_event)
        assert result["risk_score"] >= 0.1

    def test_pci_data_elevates_risk(
        self, normalizer
    ):
        event = {
            "Type": "StorageBlobLogs",
            "TimeGenerated": "2026-05-21T03:00:00Z",
            "AccountName": "svc_backup",
            "SourceIP": "10.0.0.1",
            "ResourceId": "prod-pci-data",
            "OperationName": "GetBlob"
        }
        result = normalizer.normalize(event)
        assert result["data_classification"] == "PCI"
        assert result["risk_score"] >= 0.1

    def test_risk_score_capped_at_one(
        self, normalizer, signin_event
    ):
        result = normalizer.normalize(signin_event)
        assert result["risk_score"] <= 1.0

    def test_risk_score_not_negative(
        self, normalizer, signin_event
    ):
        result = normalizer.normalize(signin_event)
        assert result["risk_score"] >= 0.0


# ============================================================
# ACCESSOR TYPE DETECTION TESTS
# ============================================================

class TestAccessorTypeDetection:

    def test_service_account_detected(
        self, normalizer
    ):
        event = {
            "Type": "AWSCloudTrail",
            "TimeGenerated": "2026-05-21T02:00:00Z",
            "AccountName": "svc_backup",
            "SourceIP": "10.0.0.1"
        }
        result = normalizer.normalize(event)
        assert result["accessor_type"] == (
            "service_account"
        )

    def test_human_user_detected(
        self, normalizer, signin_event
    ):
        result = normalizer.normalize(signin_event)
        assert result["accessor_type"] == "human"

    def test_machine_account_detected(
        self, normalizer
    ):
        event = {
            "Type": "SecurityEvent",
            "TimeGenerated": "2026-05-21T02:00:00Z",
            "AccountName": "WORKSTATION-01$",
            "SourceIP": "10.0.0.1"
        }
        result = normalizer.normalize(event)
        assert result["accessor_type"] == (
            "service_account"
        )


# ============================================================
# DATA CLASSIFICATION TESTS
# ============================================================

class TestDataClassification:

    def test_pci_classification(self, normalizer):
        cls = normalizer._detect_classification(
            "prod-pci-data",
            "customers/cards.csv"
        )
        assert cls == "PCI"

    def test_phi_classification(self, normalizer):
        cls = normalizer._detect_classification(
            "patient-health-records",
            "ehr/records.db"
        )
        assert cls == "PHI"

    def test_pii_classification(self, normalizer):
        cls = normalizer._detect_classification(
            "customer-data",
            "personal/ssn.csv"
        )
        assert cls == "PII"

    def test_unknown_classification(self, normalizer):
        cls = normalizer._detect_classification(
            "app-logs",
            "system/debug.log"
        )
        assert cls == "UNKNOWN"


# ============================================================
# SECURITY ALERT TESTS
# ============================================================

class TestSecurityAlert:

    def test_normalize_security_alert(
        self, normalizer, security_alert
    ):
        result = normalizer.normalize_security_alert(
            security_alert
        )
        assert result is not None
        assert result["risk_score"] >= 0.3

    def test_security_alert_has_name(
        self, normalizer, security_alert
    ):
        result = normalizer.normalize_security_alert(
            security_alert
        )
        assert "sentinel_alert_name" in result
        assert result["sentinel_alert_name"] == (
            "Suspicious PowerShell Activity"
        )

    def test_high_severity_alert_score(
        self, normalizer, security_alert
    ):
        result = normalizer.normalize_security_alert(
            security_alert
        )
        assert result["risk_score"] >= 0.8

    def test_mitre_techniques_preserved(
        self, normalizer, security_alert
    ):
        result = normalizer.normalize_security_alert(
            security_alert
        )
        assert "T1059.001" in str(
            result.get("sentinel_mitre_techniques", [])
        )


# ============================================================
# ENRICHMENT WRITE-BACK TESTS
# ============================================================

class TestEnrichmentWriteBack:

    def test_write_enrichment_returns_dict(
        self, normalizer
    ):
        result = normalizer.write_enrichment(
            incident_id="abc-123",
            risk_score=0.974,
            verdict="DATA_EXFILTRATION",
            agent_summary="svc_backup compromised"
        )
        assert isinstance(result, dict)

    def test_enrichment_has_risk_score(
        self, normalizer
    ):
        result = normalizer.write_enrichment(
            incident_id="abc-123",
            risk_score=0.974,
            verdict="DATA_EXFILTRATION",
            agent_summary="test"
        )
        assert result["abutech_risk_score"] == 0.974

    def test_enrichment_has_verdict(
        self, normalizer
    ):
        result = normalizer.write_enrichment(
            incident_id="abc-123",
            risk_score=0.974,
            verdict="DATA_EXFILTRATION",
            agent_summary="test"
        )
        assert result["abutech_verdict"] == (
            "DATA_EXFILTRATION"
        )

    def test_enrichment_severity_high(
        self, normalizer
    ):
        result = normalizer.write_enrichment(
            incident_id="abc-123",
            risk_score=0.974,
            verdict="DATA_EXFILTRATION",
            agent_summary="test"
        )
        assert result["abutech_severity"] == "High"

    def test_enrichment_has_comment(
        self, normalizer
    ):
        result = normalizer.write_enrichment(
            incident_id="abc-123",
            risk_score=0.974,
            verdict="DATA_EXFILTRATION",
            agent_summary="Compromised account detected",
            mitre_techniques=["T1530", "T1048"],
            policy_violations=["no_pci_access"]
        )
        comment = result["sentinel_comment"]
        assert "AbuTech" in comment
        assert "0.974" in comment
        assert "DATA_EXFILTRATION" in comment
        assert "T1530" in comment

    def test_enrichment_hitl_status(
        self, normalizer
    ):
        result = normalizer.write_enrichment(
            incident_id="abc-123",
            risk_score=0.974,
            verdict="DATA_EXFILTRATION",
            agent_summary="test",
            hitl_status="APPROVED"
        )
        assert result["abutech_hitl_status"] == (
            "APPROVED"
        )


# ============================================================
# BATCH NORMALIZATION TESTS
# ============================================================

class TestBatchNormalization:

    def test_normalize_batch_returns_list(
        self, normalizer, signin_event, firewall_event
    ):
        results = normalizer.normalize_batch(
            [signin_event, firewall_event]
        )
        assert isinstance(results, list)
        assert len(results) == 2

    def test_normalize_batch_empty_list(
        self, normalizer
    ):
        results = normalizer.normalize_batch([])
        assert results == []

    def test_normalize_batch_handles_errors(
        self, normalizer, signin_event
    ):
        events = [
            signin_event,
            None,
            signin_event
        ]
        results = normalizer.normalize_batch(events)
        assert len(results) >= 2


# ============================================================
# TIMESTAMP NORMALIZATION TESTS
# ============================================================

class TestTimestampNormalization:

    def test_iso_timestamp_preserved(
        self, normalizer
    ):
        ts = normalizer._normalize_timestamp(
            "2026-05-21T03:14:22Z"
        )
        assert "2026-05-21" in ts

    def test_space_timestamp_converted(
        self, normalizer
    ):
        ts = normalizer._normalize_timestamp(
            "2026-05-21 03:14:22"
        )
        assert "2026-05-21" in ts

    def test_empty_timestamp_returns_now(
        self, normalizer
    ):
        ts = normalizer._normalize_timestamp("")
        assert "202" in ts