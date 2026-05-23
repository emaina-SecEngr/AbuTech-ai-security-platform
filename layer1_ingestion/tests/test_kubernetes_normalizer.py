"""
Tests for Kubernetes and Container Security Normalizer
"""

import pytest
from layer1_ingestion.normalizers.kubernetes_normalizer import (
    KubernetesNormalizer,
    K8S_VERB_RISK,
    FALCO_PRIORITY_RISK,
    CRITICAL_FALCO_RULES
)


@pytest.fixture
def normalizer():
    return KubernetesNormalizer()


@pytest.fixture
def k8s_audit_exec():
    """Pod exec command - highest risk K8s event"""
    return {
        "requestReceivedTimestamp": "2026-05-21T03:00:00Z",
        "verb": "exec",
        "user": {
            "username": "system:serviceaccount:default:svc-backup",
            "groups": ["system:serviceaccounts"]
        },
        "objectRef": {
            "resource": "pods",
            "subresource": "exec",
            "namespace": "production",
            "name": "prod-app-pod-abc123"
        },
        "sourceIPs": ["185.220.101.45"],
        "responseStatus": {"code": 200},
        "userAgent": "kubectl/v1.28"
    }


@pytest.fixture
def k8s_audit_secret():
    """Secret access event"""
    return {
        "requestReceivedTimestamp": "2026-05-21T03:00:00Z",
        "verb": "get",
        "user": {
            "username": "system:serviceaccount:default:svc-backup",
            "groups": ["system:serviceaccounts"]
        },
        "objectRef": {
            "resource": "secrets",
            "namespace": "kube-system",
            "name": "admin-token"
        },
        "sourceIPs": ["10.0.0.1"],
        "responseStatus": {"code": 200}
    }


@pytest.fixture
def k8s_audit_rbac():
    """ClusterRoleBinding creation"""
    return {
        "requestReceivedTimestamp": "2026-05-21T02:00:00Z",
        "verb": "create",
        "user": {
            "username": "attacker@external.com",
            "groups": []
        },
        "objectRef": {
            "resource": "clusterrolebindings",
            "name": "evil-admin-binding"
        },
        "sourceIPs": ["185.220.101.45"],
        "responseStatus": {"code": 201}
    }


@pytest.fixture
def k8s_audit_normal():
    """Normal K8s API call"""
    return {
        "requestReceivedTimestamp": "2026-05-21T09:00:00Z",
        "verb": "get",
        "user": {
            "username": "developer@company.com",
            "groups": ["developers"]
        },
        "objectRef": {
            "resource": "pods",
            "namespace": "development",
            "name": "dev-app-pod"
        },
        "sourceIPs": ["10.0.0.50"],
        "responseStatus": {"code": 200}
    }


@pytest.fixture
def falco_privileged():
    """Falco privileged container alert"""
    return {
        "rule": "Launch Privileged Container",
        "priority": "CRITICAL",
        "time": "2026-05-21T03:00:00Z",
        "output": "Privileged container started",
        "output_fields": {
            "container.id": "abc123",
            "container.name": "malicious-container",
            "container.image.repository": "evil/image",
            "proc.name": "sh",
            "proc.cmdline": "sh -i",
            "k8s.pod.name": "evil-pod",
            "k8s.ns.name": "production",
            "user.name": "root",
            "fd.rip": "10.0.0.1"
        }
    }


@pytest.fixture
def falco_escape():
    """Falco container escape attempt"""
    return {
        "rule": "Container Escape via runc",
        "priority": "EMERGENCY",
        "time": "2026-05-21T03:00:00Z",
        "output": "Container escape detected",
        "output_fields": {
            "container.id": "xyz789",
            "container.name": "compromised",
            "proc.name": "runc",
            "proc.cmdline": "runc --root /var/run",
            "k8s.pod.name": "compromised-pod",
            "k8s.ns.name": "default",
            "fd.rip": "185.220.101.45"
        }
    }


@pytest.fixture
def falco_crypto():
    """Falco cryptomining detection"""
    return {
        "rule": "Crypto Mining Activity",
        "priority": "ALERT",
        "time": "2026-05-21T03:00:00Z",
        "output": "Cryptomining process detected",
        "output_fields": {
            "container.id": "mine123",
            "container.name": "miner",
            "proc.name": "xmrig",
            "proc.cmdline": "xmrig --pool stratum+tcp://pool.minexmr.com",
            "k8s.pod.name": "miner-pod",
            "k8s.ns.name": "default"
        }
    }


@pytest.fixture
def vulnerability_critical():
    """Critical CVE with exploit"""
    return {
        "VulnerabilityID": "CVE-2024-12345",
        "PkgName": "openssl",
        "Severity": "CRITICAL",
        "Target": "myregistry.azurecr.io/prod-app:latest",
        "FixedVersion": "3.0.12",
        "ExploitAvailable": True,
        "CVSS": {
            "nvd": {"V3Score": 9.8}
        }
    }


@pytest.fixture
def vulnerability_low():
    """Low severity CVE"""
    return {
        "VulnerabilityID": "CVE-2024-99999",
        "PkgName": "libzip",
        "Severity": "LOW",
        "Target": "dev-app:latest",
        "FixedVersion": "",
        "ExploitAvailable": False
    }


# ============================================================
# INITIALIZATION TESTS
# ============================================================

class TestInitialization:

    def test_normalizer_initializes(self, normalizer):
        assert normalizer is not None

    def test_verb_risk_map_populated(self):
        assert len(K8S_VERB_RISK) > 0
        assert "exec" in K8S_VERB_RISK
        assert K8S_VERB_RISK["exec"] >= 0.35

    def test_falco_priority_map_populated(self):
        assert "CRITICAL" in FALCO_PRIORITY_RISK
        assert "EMERGENCY" in FALCO_PRIORITY_RISK

    def test_critical_rules_list_populated(self):
        assert len(CRITICAL_FALCO_RULES) > 0


# ============================================================
# K8S AUDIT LOG TESTS
# ============================================================

class TestK8sAuditLog:

    def test_normalize_returns_dict(
        self, normalizer, k8s_audit_exec
    ):
        result = normalizer.normalize(k8s_audit_exec)
        assert isinstance(result, dict)

    def test_required_fields_present(
        self, normalizer, k8s_audit_exec
    ):
        result = normalizer.normalize(k8s_audit_exec)
        required = [
            "accessor_identity", "accessor_type",
            "data_store_name", "data_path",
            "event_time", "source_ip",
            "risk_score", "risk_reasons",
            "source_system", "raw_event"
        ]
        for field in required:
            assert field in result

    def test_pod_exec_high_risk(
        self, normalizer, k8s_audit_exec
    ):
        result = normalizer.normalize(k8s_audit_exec)
        assert result["risk_score"] >= 0.65

    def test_secret_access_flagged(
        self, normalizer, k8s_audit_secret
    ):
        result = normalizer.normalize(k8s_audit_secret)
        assert result["risk_score"] >= 0.3
        reasons = str(result["risk_reasons"])
        assert "secret" in reasons.lower()

    def test_kube_system_elevated(
        self, normalizer, k8s_audit_secret
    ):
        result = normalizer.normalize(k8s_audit_secret)
        reasons = str(result["risk_reasons"])
        assert "kube_system" in reasons

    def test_rbac_creation_flagged(
        self, normalizer, k8s_audit_rbac
    ):
        result = normalizer.normalize(k8s_audit_rbac)
        assert result["risk_score"] >= 0.35
        reasons = str(result["risk_reasons"])
        assert "clusterrole" in reasons.lower()

    def test_normal_event_low_risk(
        self, normalizer, k8s_audit_normal
    ):
        result = normalizer.normalize(k8s_audit_normal)
        assert result["risk_score"] <= 0.25

    def test_service_account_detected(
        self, normalizer, k8s_audit_exec
    ):
        result = normalizer.normalize(k8s_audit_exec)
        assert result["accessor_type"] == (
            "service_account"
        )

    def test_source_ip_extracted(
        self, normalizer, k8s_audit_exec
    ):
        result = normalizer.normalize(k8s_audit_exec)
        assert result["source_ip"] == "185.220.101.45"

    def test_k8s_verb_captured(
        self, normalizer, k8s_audit_exec
    ):
        result = normalizer.normalize(k8s_audit_exec)
        assert result["k8s_verb"] == "exec"

    def test_k8s_namespace_captured(
        self, normalizer, k8s_audit_exec
    ):
        result = normalizer.normalize(k8s_audit_exec)
        assert result["k8s_namespace"] == "production"

    def test_empty_event_safe(self, normalizer):
        result = normalizer.normalize({})
        assert result["risk_score"] == 0.0

    def test_none_event_safe(self, normalizer):
        result = normalizer.normalize(None)
        assert result is not None

    def test_risk_score_capped(
        self, normalizer, k8s_audit_exec
    ):
        result = normalizer.normalize(k8s_audit_exec)
        assert result["risk_score"] <= 1.0


# ============================================================
# FALCO ALERT TESTS
# ============================================================

class TestFalcoAlerts:

    def test_falco_returns_dict(
        self, normalizer, falco_privileged
    ):
        result = normalizer.normalize_falco_alert(
            falco_privileged
        )
        assert isinstance(result, dict)

    def test_privileged_container_critical(
        self, normalizer, falco_privileged
    ):
        result = normalizer.normalize_falco_alert(
            falco_privileged
        )
        assert result["risk_score"] >= 0.80

    def test_container_escape_highest_risk(
        self, normalizer, falco_escape
    ):
        result = normalizer.normalize_falco_alert(
            falco_escape
        )
        assert result["risk_score"] >= 0.90

    def test_container_escape_flagged(
        self, normalizer, falco_escape
    ):
        result = normalizer.normalize_falco_alert(
            falco_escape
        )
        reasons = str(result["risk_reasons"])
        assert "escape" in reasons.lower()

    def test_cryptomining_detected(
        self, normalizer, falco_crypto
    ):
        result = normalizer.normalize_falco_alert(
            falco_crypto
        )
        assert result["risk_score"] >= 0.75
        reasons = str(result["risk_reasons"])
        assert "crypto" in reasons.lower()

    def test_falco_rule_captured(
        self, normalizer, falco_privileged
    ):
        result = normalizer.normalize_falco_alert(
            falco_privileged
        )
        assert result["falco_rule"] == (
            "Launch Privileged Container"
        )

    def test_falco_priority_captured(
        self, normalizer, falco_privileged
    ):
        result = normalizer.normalize_falco_alert(
            falco_privileged
        )
        assert result["falco_priority"] == "CRITICAL"

    def test_source_system_falco(
        self, normalizer, falco_privileged
    ):
        result = normalizer.normalize_falco_alert(
            falco_privileged
        )
        assert result["source_system"] == (
            "falco_runtime"
        )

    def test_tor_in_falco_elevated(
        self, normalizer, falco_escape
    ):
        result = normalizer.normalize_falco_alert(
            falco_escape
        )
        assert result["risk_score"] >= 0.90

    def test_container_image_captured(
        self, normalizer, falco_privileged
    ):
        result = normalizer.normalize_falco_alert(
            falco_privileged
        )
        assert "container_image" in result

    def test_empty_falco_safe(self, normalizer):
        result = normalizer.normalize_falco_alert({})
        assert result["risk_score"] == 0.0


# ============================================================
# VULNERABILITY TESTS
# ============================================================

class TestVulnerabilityNormalization:

    def test_vulnerability_returns_dict(
        self, normalizer, vulnerability_critical
    ):
        result = normalizer.normalize_vulnerability(
            vulnerability_critical
        )
        assert isinstance(result, dict)

    def test_critical_cve_high_risk(
        self, normalizer, vulnerability_critical
    ):
        result = normalizer.normalize_vulnerability(
            vulnerability_critical
        )
        assert result["risk_score"] >= 0.85

    def test_exploit_available_elevated(
        self, normalizer, vulnerability_critical
    ):
        result = normalizer.normalize_vulnerability(
            vulnerability_critical
        )
        reasons = str(result["risk_reasons"])
        assert "exploit" in reasons

    def test_patch_available_flagged(
        self, normalizer, vulnerability_critical
    ):
        result = normalizer.normalize_vulnerability(
            vulnerability_critical
        )
        reasons = str(result["risk_reasons"])
        assert "patch_available" in reasons

    def test_low_cve_low_risk(
        self, normalizer, vulnerability_low
    ):
        result = normalizer.normalize_vulnerability(
            vulnerability_low
        )
        assert result["risk_score"] <= 0.30

    def test_cve_id_captured(
        self, normalizer, vulnerability_critical
    ):
        result = normalizer.normalize_vulnerability(
            vulnerability_critical
        )
        assert result["cve_id"] == "CVE-2024-12345"

    def test_source_system_vuln(
        self, normalizer, vulnerability_critical
    ):
        result = normalizer.normalize_vulnerability(
            vulnerability_critical
        )
        assert result["source_system"] == (
            "container_vulnerability"
        )

    def test_empty_vulnerability_safe(
        self, normalizer
    ):
        result = normalizer.normalize_vulnerability({})
        assert result["risk_score"] == 0.0