"""
Tests for Firewall Normalizer
"""
import pytest
from layer1_ingestion.normalizers.firewall_normalizer import (
    FirewallNormalizer,
    FIREWALL_ACTION_RISK,
    HIGH_RISK_PORTS,
    HIGH_RISK_COUNTRIES
)


@pytest.fixture
def normalizer():
    return FirewallNormalizer()


@pytest.fixture
def palo_alto_traffic_exfil():
    return {
        "log_type": "TRAFFIC",
        "src": "10.0.1.105",
        "dst": "198.51.100.42",
        "spt": "54321",
        "dpt": "443",
        "proto": "TCP",
        "act": "allow",
        "out": "524288000",
        "in": "1024",
        "app": "ssl",
        "suser": "svc_backup",
        "cs1": "trust",
        "cs3": "untrust",
        "src_country": "US",
        "timestamp": "2026-05-21T03:00:00Z"
    }


@pytest.fixture
def palo_alto_threat():
    return {
        "log_type": "THREAT",
        "src": "185.220.101.45",
        "dst": "10.0.1.105",
        "dpt": "445",
        "proto": "TCP",
        "act": "sinkhole",
        "threatid": "Emotet.Gen",
        "severity": "critical",
        "timestamp": "2026-05-21T03:00:00Z"
    }


@pytest.fixture
def fortinet_blocked():
    return {
        "type": "traffic",
        "srcip": "185.220.101.45",
        "dstip": "10.0.0.1",
        "srcport": "54321",
        "dstport": "22",
        "proto": "TCP",
        "action": "deny",
        "sentbyte": "0",
        "rcvdbyte": "0",
        "attack": "SSH Brute Force",
        "srccountry": "RU",
        "policyid": "100",
        "timestamp": "2026-05-21T03:00:00Z"
    }


@pytest.fixture
def cisco_asa_deny():
    return {
        "message_id": "ASA-4-106023",
        "action": "deny",
        "src_ip": "185.220.101.45",
        "dst_ip": "10.0.0.1",
        "src_port": "12345",
        "dst_port": "3389",
        "protocol": "TCP",
        "bytes": "0",
        "interface": "outside",
        "reason": "ACL block",
        "timestamp": "2026-05-21T03:00:00Z"
    }


@pytest.fixture
def checkpoint_threat():
    return {
        "action": "Drop",
        "src": "45.142.100.10",
        "dst": "10.0.0.5",
        "dpt": "1433",
        "proto": "TCP",
        "blade": "IPS",
        "malware_name": "SQL Slammer",
        "rule_name": "IPS_Block_SQLi",
        "src_country": "CN",
        "bytes": "4096",
        "timestamp": "2026-05-21T02:00:00Z"
    }


@pytest.fixture
def normal_allowed():
    return {
        "log_type": "TRAFFIC",
        "src": "10.0.1.100",
        "dst": "8.8.8.8",
        "dpt": "443",
        "proto": "TCP",
        "act": "allow",
        "out": "10240",
        "in": "51200",
        "app": "ssl",
        "suser": "john.smith",
        "timestamp": "2026-05-21T09:00:00Z"
    }


# ============================================================
# INITIALIZATION TESTS
# ============================================================

class TestInitialization:

    def test_normalizer_initializes(self, normalizer):
        assert normalizer is not None

    def test_action_risk_map_populated(self):
        assert "allow" in FIREWALL_ACTION_RISK
        assert "deny" in FIREWALL_ACTION_RISK
        assert "threat" in FIREWALL_ACTION_RISK

    def test_high_risk_ports_populated(self):
        assert 22 in HIGH_RISK_PORTS
        assert 3389 in HIGH_RISK_PORTS
        assert 445 in HIGH_RISK_PORTS

    def test_high_risk_countries_populated(self):
        assert "RU" in HIGH_RISK_COUNTRIES
        assert "CN" in HIGH_RISK_COUNTRIES


# ============================================================
# VENDOR DETECTION TESTS
# ============================================================

class TestVendorDetection:

    def test_detects_palo_alto(
        self, normalizer, palo_alto_traffic_exfil
    ):
        vendor = normalizer._detect_vendor(
            palo_alto_traffic_exfil
        )
        assert vendor in ["palo_alto", "generic"]

    def test_detects_fortinet(
        self, normalizer, fortinet_blocked
    ):
        vendor = normalizer._detect_vendor(
            fortinet_blocked
        )
        assert vendor == "fortinet"

    def test_detects_cisco_asa(
        self, normalizer, cisco_asa_deny
    ):
        vendor = normalizer._detect_vendor(
            cisco_asa_deny
        )
        assert vendor == "cisco_asa"

    def test_detects_checkpoint(
        self, normalizer, checkpoint_threat
    ):
        vendor = normalizer._detect_vendor(
            checkpoint_threat
        )
        assert vendor == "checkpoint"

    def test_unknown_vendor_generic(self, normalizer):
        vendor = normalizer._detect_vendor(
            {"someField": "value"}
        )
        assert vendor == "generic"


# ============================================================
# PALO ALTO TESTS
# ============================================================

class TestPaloAlto:

    def test_normalize_returns_dict(
        self, normalizer, palo_alto_traffic_exfil
    ):
        result = normalizer.normalize(
            palo_alto_traffic_exfil
        )
        assert isinstance(result, dict)

    def test_required_fields_present(
        self, normalizer, palo_alto_traffic_exfil
    ):
        result = normalizer.normalize(
            palo_alto_traffic_exfil
        )
        required = [
            "accessor_identity", "source_ip",
            "risk_score", "risk_reasons",
            "source_system", "raw_event",
            "fw_vendor", "fw_action"
        ]
        for field in required:
            assert field in result

    def test_large_transfer_elevated(
        self, normalizer, palo_alto_traffic_exfil
    ):
        result = normalizer.normalize(
            palo_alto_traffic_exfil
        )
        assert result["risk_score"] >= 0.25
        reasons = str(result["risk_reasons"])
        assert "large_transfer" in reasons

    def test_threat_log_high_risk(
        self, normalizer, palo_alto_threat
    ):
        result = normalizer.normalize(
            palo_alto_threat
        )
        assert result["risk_score"] >= 0.80

    def test_threat_name_captured(
        self, normalizer, palo_alto_threat
    ):
        result = normalizer.normalize(
            palo_alto_threat
        )
        assert result["fw_threat_name"] == "Emotet.Gen"

    def test_source_ip_extracted(
        self, normalizer, palo_alto_traffic_exfil
    ):
        result = normalizer.normalize(
            palo_alto_traffic_exfil
        )
        assert result["source_ip"] == "10.0.1.105"

    def test_bytes_captured(
        self, normalizer, palo_alto_traffic_exfil
    ):
        result = normalizer.normalize(
            palo_alto_traffic_exfil
        )
        assert result["bytes_accessed"] >= 524288000

    def test_after_hours_elevated(
        self, normalizer, palo_alto_traffic_exfil
    ):
        result = normalizer.normalize(
            palo_alto_traffic_exfil
        )
        reasons = str(result["risk_reasons"])
        assert "after_hours" in reasons

    def test_source_system_palo_alto(
        self, normalizer, palo_alto_traffic_exfil
    ):
        result = normalizer.normalize(
            palo_alto_traffic_exfil
        )
        assert "firewall" in result["source_system"]

    def test_normal_allowed_low_risk(
        self, normalizer, normal_allowed
    ):
        result = normalizer.normalize(normal_allowed)
        assert result["risk_score"] <= 0.35

    def test_empty_event_safe(self, normalizer):
        result = normalizer.normalize({})
        assert result["risk_score"] == 0.0

    def test_none_event_safe(self, normalizer):
        result = normalizer.normalize(None)
        assert result is not None

    def test_risk_score_capped(
        self, normalizer, palo_alto_threat
    ):
        result = normalizer.normalize(palo_alto_threat)
        assert result["risk_score"] <= 1.0


# ============================================================
# FORTINET TESTS
# ============================================================

class TestFortinet:

    def test_fortinet_blocked_elevated(
        self, normalizer, fortinet_blocked
    ):
        result = normalizer.normalize(fortinet_blocked)
        assert result["risk_score"] >= 0.50

    def test_fortinet_ssh_port_flagged(
        self, normalizer, fortinet_blocked
    ):
        result = normalizer.normalize(fortinet_blocked)
        reasons = str(result["risk_reasons"])
        assert "high_risk_port" in reasons

    def test_fortinet_ru_country_flagged(
        self, normalizer, fortinet_blocked
    ):
        result = normalizer.normalize(fortinet_blocked)
        reasons = str(result["risk_reasons"])
        assert "high_risk_country" in reasons

    def test_fortinet_source_system(
        self, normalizer, fortinet_blocked
    ):
        result = normalizer.normalize(fortinet_blocked)
        assert "fortinet" in result["source_system"]

    def test_fortinet_threat_captured(
        self, normalizer, fortinet_blocked
    ):
        result = normalizer.normalize(fortinet_blocked)
        assert "SSH Brute Force" in str(
            result.get("fw_threat_name", "")
        )


# ============================================================
# CISCO ASA TESTS
# ============================================================

class TestCiscoASA:

    def test_cisco_asa_deny_elevated(
        self, normalizer, cisco_asa_deny
    ):
        result = normalizer.normalize(cisco_asa_deny)
        assert result["risk_score"] >= 0.55

    def test_cisco_asa_rdp_port_flagged(
        self, normalizer, cisco_asa_deny
    ):
        result = normalizer.normalize(cisco_asa_deny)
        reasons = str(result["risk_reasons"])
        assert "high_risk_port" in reasons

    def test_cisco_asa_source_system(
        self, normalizer, cisco_asa_deny
    ):
        result = normalizer.normalize(cisco_asa_deny)
        assert "cisco" in result["source_system"]

    def test_cisco_asa_message_id_captured(
        self, normalizer, cisco_asa_deny
    ):
        result = normalizer.normalize(cisco_asa_deny)
        assert result.get("fw_message_id") == (
            "ASA-4-106023"
        )


# ============================================================
# CHECK POINT TESTS
# ============================================================

class TestCheckPoint:

    def test_checkpoint_threat_high_risk(
        self, normalizer, checkpoint_threat
    ):
        result = normalizer.normalize(checkpoint_threat)
        assert result["risk_score"] >= 0.70

    def test_checkpoint_mssql_port_flagged(
        self, normalizer, checkpoint_threat
    ):
        result = normalizer.normalize(checkpoint_threat)
        reasons = str(result["risk_reasons"])
        assert "high_risk_port" in reasons

    def test_checkpoint_cn_country_flagged(
        self, normalizer, checkpoint_threat
    ):
        result = normalizer.normalize(checkpoint_threat)
        reasons = str(result["risk_reasons"])
        assert "high_risk_country" in reasons

    def test_checkpoint_source_system(
        self, normalizer, checkpoint_threat
    ):
        result = normalizer.normalize(checkpoint_threat)
        assert "checkpoint" in result[
            "source_system"
        ]


# ============================================================
# RISK SCORING TESTS
# ============================================================

class TestRiskScoring:

    def test_tor_src_elevated(self, normalizer):
        event = {
            "log_type": "TRAFFIC",
            "src": "185.220.101.45",
            "dst": "10.0.0.1",
            "dpt": "443",
            "act": "allow",
            "out": "1024",
            "in": "512",
            "timestamp": "2026-05-21T03:00:00Z"
        }
        result = normalizer.normalize(event)
        reasons = str(result["risk_reasons"])
        assert "tor" in reasons.lower()

    def test_high_risk_port_22_flagged(
        self, normalizer
    ):
        event = {
            "src": "10.0.0.1",
            "dst": "192.168.1.100",
            "dpt": "22",
            "act": "allow",
            "out": "1024",
            "timestamp": "2026-05-21T09:00:00Z"
        }
        result = normalizer.normalize(event)
        reasons = str(result["risk_reasons"])
        assert "high_risk_port" in reasons

    def test_large_transfer_500mb_flagged(
        self, normalizer
    ):
        event = {
            "src": "10.0.0.1",
            "dst": "8.8.8.8",
            "dpt": "443",
            "act": "allow",
            "out": "600000000",
            "in": "1024",
            "timestamp": "2026-05-21T09:00:00Z"
        }
        result = normalizer.normalize(event)
        reasons = str(result["risk_reasons"])
        assert "500mb" in reasons.lower()

    def test_risk_score_not_negative(
        self, normalizer, normal_allowed
    ):
        result = normalizer.normalize(normal_allowed)
        assert result["risk_score"] >= 0.0

    def test_risk_score_max_one(
        self, normalizer, palo_alto_threat
    ):
        result = normalizer.normalize(palo_alto_threat)
        assert result["risk_score"] <= 1.0


# ============================================================
# CEF PARSING TESTS
# ============================================================

class TestCEFParsing:

    def test_parse_cef_returns_dict(self, normalizer):
        cef = (
            "CEF:0|Palo Alto Networks|PAN-OS|10.2|"
            "TRAFFIC|Traffic Log|3|"
            "src=10.0.1.105 dst=198.51.100.42 "
            "dpt=443 act=allow out=1024"
        )
        result = normalizer._parse_cef(cef)
        assert isinstance(result, dict)

    def test_parse_cef_extracts_vendor(
        self, normalizer
    ):
        cef = (
            "CEF:0|Palo Alto Networks|PAN-OS|10.2|"
            "TRAFFIC|Traffic Log|3|"
            "src=10.0.1.105 dst=8.8.8.8"
        )
        result = normalizer._parse_cef(cef)
        assert result.get("device_vendor") == (
            "Palo Alto Networks"
        )

    def test_parse_cef_extracts_ext_fields(
        self, normalizer
    ):
        cef = (
            "CEF:0|Vendor|Product|1.0|ID|Name|3|"
            "src=10.0.0.1 dst=8.8.8.8 dpt=443"
        )
        result = normalizer._parse_cef(cef)
        assert result.get("src") == "10.0.0.1"
        assert result.get("dpt") == "443"

    def test_normalize_cef_string(self, normalizer):
        cef = (
            "CEF:0|Palo Alto Networks|PAN-OS|10.2|"
            "TRAFFIC|Traffic Log|3|"
            "src=10.0.1.105 dst=198.51.100.42 "
            "dpt=443 act=allow out=1024 in=512"
        )
        result = normalizer.normalize_cef(cef)
        assert isinstance(result, dict)
        assert "risk_score" in result

    def test_normalize_empty_cef_safe(
        self, normalizer
    ):
        result = normalizer.normalize_cef("")
        assert result["risk_score"] == 0.0