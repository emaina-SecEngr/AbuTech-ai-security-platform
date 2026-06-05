"""
Tests for Microsoft Defender for Cloud Normalizer
"""

import pytest
from layer1_ingestion.normalizers.defender_cloud_normalizer\
    import (
        DefenderForCloudNormalizer,
        DEFENDER_SEVERITY_RISK,
        DEFENDER_INTENT_TO_TACTIC,
        DEFENDER_KEYWORD_TECHNIQUE
    )


@pytest.fixture
def normalizer():
    return DefenderForCloudNormalizer()


@pytest.fixture
def vm_cryptomining():
    return {
        "properties": {
            "alertDisplayName": "Digital currency mining related behavior detected",
            "description": "Analysis of host data detected crypto mining activity",
            "severity": "High",
            "intent": "Execution",
            "compromisedEntity": "prod-vm-payment-01",
            "productName": "Azure Security Center",
            "entities": [
                {"type": "host", "hostName": "prod-vm-payment-01"},
                {"type": "ip", "address": "45.142.100.10"}
            ],
            "startTimeUtc": "2026-06-01T03:00:00Z"
        }
    }


@pytest.fixture
def container_escape():
    return {
        "properties": {
            "alertDisplayName": "Container escape attempt detected in AKS cluster",
            "description": "A container escape was attempted on the Kubernetes node",
            "severity": "High",
            "intent": "PrivilegeEscalation",
            "compromisedEntity": "aks-prod-cluster",
            "productName": "Microsoft Defender for Containers",
            "entities": [
                {"type": "host", "hostName": "aks-node-03"}
            ]
        }
    }


@pytest.fixture
def storage_exfil():
    return {
        "properties": {
            "alertDisplayName": "Anomalous data extraction from storage account",
            "description": "Unusual volume of blob downloads detected indicating data exfiltration of PCI data",
            "severity": "High",
            "intent": "Exfiltration",
            "compromisedEntity": "prodpcistorageacct",
            "productName": "Microsoft Defender for Storage",
            "entities": [
                {"type": "ip", "address": "185.220.101.45"}
            ]
        }
    }


@pytest.fixture
def keyvault_access():
    return {
        "properties": {
            "alertDisplayName": "Unusual secret access pattern in Key Vault",
            "description": "Anomalous enumeration of secrets and credentials detected",
            "severity": "Medium",
            "intent": "CredentialAccess",
            "compromisedEntity": "prod-keyvault-01",
            "productName": "Microsoft Defender for Key Vault"
        }
    }


@pytest.fixture
def sql_injection():
    return {
        "properties": {
            "alertDisplayName": "Potential SQL injection against Azure SQL Database",
            "description": "A potential SQL injection was detected against the database",
            "severity": "High",
            "intent": "InitialAccess",
            "compromisedEntity": "prod-sql-db",
            "productName": "Microsoft Defender for SQL"
        }
    }


@pytest.fixture
def flat_alert():
    """Alert without properties nesting"""
    return {
        "AlertDisplayName": "Suspicious process executed on VM",
        "Severity": "Medium",
        "Intent": "Execution",
        "CompromisedEntity": "test-vm-01",
        "ProductName": "Azure Security Center"
    }


# ============================================================
# INITIALIZATION
# ============================================================

class TestInitialization:

    def test_normalizer_initializes(self, normalizer):
        assert normalizer is not None
        assert normalizer.source_system == (
            "defender_for_cloud"
        )

    def test_severity_map(self):
        assert DEFENDER_SEVERITY_RISK["HIGH"] > (
            DEFENDER_SEVERITY_RISK["LOW"]
        )

    def test_intent_map_populated(self):
        assert "Exfiltration" in (
            DEFENDER_INTENT_TO_TACTIC
        )
        assert DEFENDER_INTENT_TO_TACTIC[
            "Exfiltration"
        ][0] == "TA0010"

    def test_keyword_map_populated(self):
        assert "cryptomining" in (
            DEFENDER_KEYWORD_TECHNIQUE
        )


# ============================================================
# VM / SERVERS
# ============================================================

class TestServers:

    def test_returns_dict(
        self, normalizer, vm_cryptomining
    ):
        result = normalizer.normalize(
            vm_cryptomining
        )
        assert isinstance(result, dict)

    def test_plan_identified_servers(
        self, normalizer, vm_cryptomining
    ):
        result = normalizer.normalize(
            vm_cryptomining
        )
        assert result["defender_plan"] == "servers"

    def test_cryptomining_technique(
        self, normalizer, vm_cryptomining
    ):
        result = normalizer.normalize(
            vm_cryptomining
        )
        assert result["mitre_technique"] == "T1496"

    def test_cryptomining_risk(
        self, normalizer, vm_cryptomining
    ):
        result = normalizer.normalize(
            vm_cryptomining
        )
        assert result["risk_score"] >= 0.80
        assert "cryptomining_detected" in (
            result["risk_reasons"]
        )

    def test_source_ip_extracted(
        self, normalizer, vm_cryptomining
    ):
        result = normalizer.normalize(
            vm_cryptomining
        )
        assert result["source_ip"] == (
            "45.142.100.10"
        )

    def test_resource_captured(
        self, normalizer, vm_cryptomining
    ):
        result = normalizer.normalize(
            vm_cryptomining
        )
        assert result["data_store_name"] == (
            "prod-vm-payment-01"
        )


# ============================================================
# CONTAINERS
# ============================================================

class TestContainers:

    def test_plan_identified(
        self, normalizer, container_escape
    ):
        result = normalizer.normalize(
            container_escape
        )
        assert result["defender_plan"] == (
            "containers"
        )

    def test_escape_technique(
        self, normalizer, container_escape
    ):
        result = normalizer.normalize(
            container_escape
        )
        assert result["mitre_technique"] == "T1611"

    def test_escape_risk(
        self, normalizer, container_escape
    ):
        result = normalizer.normalize(
            container_escape
        )
        assert result["risk_score"] >= 0.93
        assert "container_escape_attempt" in (
            result["risk_reasons"]
        )

    def test_privesc_tactic(
        self, normalizer, container_escape
    ):
        result = normalizer.normalize(
            container_escape
        )
        assert result["mitre_tactic"] == (
            "Privilege Escalation"
        )


# ============================================================
# STORAGE
# ============================================================

class TestStorage:

    def test_plan_identified(
        self, normalizer, storage_exfil
    ):
        result = normalizer.normalize(
            storage_exfil
        )
        assert result["defender_plan"] == "storage"

    def test_exfil_risk(
        self, normalizer, storage_exfil
    ):
        result = normalizer.normalize(
            storage_exfil
        )
        assert result["risk_score"] >= 0.80

    def test_exfil_tactic(
        self, normalizer, storage_exfil
    ):
        result = normalizer.normalize(
            storage_exfil
        )
        assert result["mitre_tactic"] == (
            "Exfiltration"
        )

    def test_tor_ip_flagged(
        self, normalizer, storage_exfil
    ):
        result = normalizer.normalize(
            storage_exfil
        )
        assert "tor_exit_node_src" in (
            result["risk_reasons"]
        )

    def test_pci_classification(
        self, normalizer, storage_exfil
    ):
        result = normalizer.normalize(
            storage_exfil
        )
        assert result["data_classification"] == (
            "PCI"
        )


# ============================================================
# KEY VAULT
# ============================================================

class TestKeyVault:

    def test_plan_identified(
        self, normalizer, keyvault_access
    ):
        result = normalizer.normalize(
            keyvault_access
        )
        assert result["defender_plan"] == "keyvault"

    def test_credential_technique(
        self, normalizer, keyvault_access
    ):
        result = normalizer.normalize(
            keyvault_access
        )
        assert result["mitre_technique"] == "T1552"

    def test_credential_access_tactic(
        self, normalizer, keyvault_access
    ):
        result = normalizer.normalize(
            keyvault_access
        )
        assert result["mitre_tactic"] == (
            "Credential Access"
        )


# ============================================================
# SQL
# ============================================================

class TestSQL:

    def test_plan_identified(
        self, normalizer, sql_injection
    ):
        result = normalizer.normalize(
            sql_injection
        )
        assert result["defender_plan"] == "sql"

    def test_sqli_technique(
        self, normalizer, sql_injection
    ):
        result = normalizer.normalize(
            sql_injection
        )
        assert result["mitre_technique"] == "T1190"

    def test_sqli_risk(
        self, normalizer, sql_injection
    ):
        result = normalizer.normalize(
            sql_injection
        )
        assert result["risk_score"] >= 0.82
        assert "sql_injection_detected" in (
            result["risk_reasons"]
        )


# ============================================================
# FORMAT HANDLING
# ============================================================

class TestFormatHandling:

    def test_flat_alert_handled(
        self, normalizer, flat_alert
    ):
        result = normalizer.normalize(flat_alert)
        assert isinstance(result, dict)
        assert result["defender_severity"] == (
            "MEDIUM"
        )

    def test_flat_alert_resource(
        self, normalizer, flat_alert
    ):
        result = normalizer.normalize(flat_alert)
        assert result["data_store_name"] == (
            "test-vm-01"
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
            "defender_for_cloud"
        )

    def test_risk_never_exceeds_one(
        self, normalizer, storage_exfil
    ):
        result = normalizer.normalize(
            storage_exfil
        )
        assert result["risk_score"] <= 1.0

    def test_event_time_present(
        self, normalizer, vm_cryptomining
    ):
        result = normalizer.normalize(
            vm_cryptomining
        )
        assert result["event_time"] != ""

    def test_raw_event_preserved(
        self, normalizer, sql_injection
    ):
        result = normalizer.normalize(
            sql_injection
        )
        assert result["raw_event"] == sql_injection

    def test_accessor_type(
        self, normalizer, vm_cryptomining
    ):
        result = normalizer.normalize(
            vm_cryptomining
        )
        assert result["accessor_type"] == (
            "cloud_resource"
        )


# ============================================================
# INTENT MAPPING
# ============================================================

class TestIntentMapping:

    def test_exfiltration_intent(self, normalizer):
        tactic_id, name = normalizer._map_intent(
            "Exfiltration"
        )
        assert tactic_id == "TA0010"
        assert name == "Exfiltration"

    def test_lateral_movement_intent(
        self, normalizer
    ):
        tactic_id, name = normalizer._map_intent(
            "LateralMovement"
        )
        assert tactic_id == "TA0008"

    def test_empty_intent(self, normalizer):
        tactic_id, name = normalizer._map_intent("")
        assert tactic_id == ""

    def test_comma_separated_intent(
        self, normalizer
    ):
        tactic_id, name = normalizer._map_intent(
            "Exfiltration, Collection"
        )
        assert tactic_id == "TA0010"