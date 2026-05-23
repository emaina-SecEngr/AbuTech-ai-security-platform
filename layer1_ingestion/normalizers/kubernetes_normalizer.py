"""
Layer 1 — Data Ingestion
Kubernetes and Container Security Normalizer

Handles container and orchestration security events.

SOURCES HANDLED:
    Kubernetes Audit Logs:
        API server audit trail
        Every kubectl command executed
        RBAC authorization decisions
        Resource creation/deletion
    
    Falco Runtime Security:
        Container process anomalies
        File system violations
        Network connection alerts
        Privilege escalation attempts
        Container escape attempts
    
    Container Image Vulnerabilities:
        Trivy scan results
        Snyk container findings
        ECR/ACR/GCR scan results
    
    Kubernetes RBAC Violations:
        ClusterRole binding changes
        ServiceAccount permission grants
        Privileged pod creation

WHY KUBERNETES MATTERS FOR BANKS:
    Banks are moving core workloads to containers.
    Kubernetes is the standard orchestrator.
    Container escape = full node compromise.
    
    ATTACK SCENARIOS YOUR PLATFORM CATCHES:
    
    1. Privileged container escape:
       Attacker runs privileged pod.
       Mounts host filesystem.
       Reads /etc/shadow or cloud credentials.
       YOUR PLATFORM: Falco event + risk = CRITICAL
    
    2. Lateral movement via service accounts:
       Compromised pod uses K8s service account.
       Service account has cluster-admin binding.
       Attacker reads all secrets in cluster.
       YOUR PLATFORM: RBAC violation + secret access
    
    3. Cryptomining via compromised image:
       Malicious image deployed to cluster.
       High CPU detected + unusual network.
       YOUR PLATFORM: IsolationForest flags anomaly
    
    4. Supply chain attack:
       Malicious dependency in container image.
       Image has known CVE with exploit available.
       YOUR PLATFORM: Vulnerability + behavior = risk

USAGE:
    normalizer = KubernetesNormalizer()
    
    # K8s audit log
    event = normalizer.normalize(k8s_audit_log)
    
    # Falco alert
    event = normalizer.normalize_falco_alert(falco)
    
    # Vulnerability finding
    event = normalizer.normalize_vulnerability(vuln)
"""

import logging
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)

# Kubernetes audit verb risk levels
K8S_VERB_RISK = {
    "create":           0.10,
    "update":           0.10,
    "patch":            0.10,
    "delete":           0.20,
    "deletecollection": 0.25,
    "get":              0.05,
    "list":             0.05,
    "watch":            0.05,
    "proxy":            0.20,
    "redirect":         0.20,
    "exec":             0.40,
    "portforward":      0.35,
    "attach":           0.30,
    "bind":             0.25,
    "escalate":         0.50,
    "impersonate":      0.45,
}

# High risk Kubernetes resources
HIGH_RISK_RESOURCES = {
    "secrets":                  0.30,
    "clusterrolebindings":      0.35,
    "rolebindings":             0.25,
    "clusterroles":             0.30,
    "serviceaccounts":          0.20,
    "nodes":                    0.25,
    "persistentvolumes":        0.20,
    "validatingwebhookconfigurations": 0.30,
    "mutatingwebhookconfigurations":   0.30,
    "customresourcedefinitions":       0.20,
    "namespaces":               0.20,
    "pods/exec":                0.45,
    "pods/attach":              0.35,
    "pods/portforward":         0.35,
}

# Falco priority to risk score
FALCO_PRIORITY_RISK = {
    "EMERGENCY": 0.98,
    "ALERT":     0.90,
    "CRITICAL":  0.85,
    "ERROR":     0.70,
    "WARNING":   0.55,
    "NOTICE":    0.35,
    "INFO":      0.15,
    "DEBUG":     0.05,
}

# Critical Falco rules
CRITICAL_FALCO_RULES = [
    "Terminal Shell in Container",
    "Container Escape via runc",
    "Privileged Container Started",
    "Write below etc",
    "Read sensitive file trusted after startup",
    "Launch Privileged Container",
    "Container Drift Detected",
    "Crypto Mining Activity",
    "Launch Package Management Process in Container",
    "Detect outbound connections to common miner pool ports",
    "K8s Secret Get",
    "Attach/Exec Pod",
    "ClusterRole With Write Privileges Created",
    "Service Account Created in Kube System Namespace",
]


class KubernetesNormalizer:
    """
    Normalizes Kubernetes and container security
    events into DataAccessEvent format.
    """

    def __init__(self):
        self.source_system = "kubernetes"

    def normalize(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize K8s audit log event.
        K8s API server logs every action.

        Args:
            raw_event: K8s audit log dict

        Returns:
            DataAccessEvent compatible dict
        """
        if not raw_event:
            return self._empty_event()

        user = raw_event.get("user", {})
        accessor = (
            user.get("username", "")
            or user.get("name", "unknown")
        )

        source_ip = self._extract_source_ip(raw_event)
        verb = raw_event.get("verb", "")
        resource = raw_event.get(
            "objectRef", {}
        ).get("resource", "")
        subresource = raw_event.get(
            "objectRef", {}
        ).get("subresource", "")
        namespace = raw_event.get(
            "objectRef", {}
        ).get("namespace", "")
        resource_name = raw_event.get(
            "objectRef", {}
        ).get("name", "")

        full_resource = resource
        if subresource:
            full_resource = f"{resource}/{subresource}"

        data_store = (
            f"k8s/{namespace}/{resource}"
            if namespace
            else f"k8s/{resource}"
        )
        data_path = (
            f"{verb}/{full_resource}/{resource_name}"
        )

        accessor_type = self._detect_accessor_type(
            accessor, user
        )
        classification = self._detect_classification(
            resource, resource_name, namespace
        )
        risk_score, risk_reasons = (
            self._calculate_k8s_risk(
                verb, full_resource, accessor,
                raw_event, namespace
            )
        )

        return {
            "accessor_identity": accessor,
            "accessor_type": accessor_type,
            "data_store_name": data_store,
            "data_path": data_path,
            "data_classification": classification,
            "bytes_accessed": 0,
            "event_time": raw_event.get(
                "requestReceivedTimestamp",
                raw_event.get(
                    "stageTimestamp", _now()
                )
            ),
            "source_ip": source_ip,
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "kubernetes_audit",
            "raw_event": raw_event,
            "k8s_verb": verb,
            "k8s_resource": full_resource,
            "k8s_namespace": namespace,
            "k8s_resource_name": resource_name,
            "k8s_response_code": raw_event.get(
                "responseStatus", {}
            ).get("code", 0)
        }

    def normalize_falco_alert(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize Falco runtime security alert.
        Falco monitors running containers for
        suspicious behavior.

        Args:
            raw_event: Falco alert dict

        Returns:
            DataAccessEvent with container context
        """
        if not raw_event:
            return self._empty_event()

        priority = raw_event.get(
            "priority", "WARNING"
        ).upper()
        rule = raw_event.get("rule", "")
        output = raw_event.get("output", "")

        fields = raw_event.get(
            "output_fields",
            raw_event.get("fields", {})
        )

        container_id = fields.get(
            "container.id", ""
        )
        container_name = fields.get(
            "container.name", ""
        )
        image = fields.get(
            "container.image.repository",
            fields.get("container.image", "")
        )
        process_name = fields.get(
            "proc.name", ""
        )
        process_cmd = fields.get(
            "proc.cmdline", ""
        )
        k8s_pod = fields.get("k8s.pod.name", "")
        k8s_ns = fields.get(
            "k8s.ns.name", "default"
        )
        source_ip = fields.get(
            "fd.rip",
            fields.get("fd.sip", "")
        )

        accessor = (
            fields.get("user.name", "")
            or process_name
            or container_name
            or "unknown"
        )

        base_risk = FALCO_PRIORITY_RISK.get(
            priority, 0.55
        )

        risk_reasons = [
            f"falco_priority:{priority}",
            f"falco_rule:{rule}"
        ]

        if rule in CRITICAL_FALCO_RULES:
            base_risk = max(base_risk, 0.85)
            risk_reasons.append(
                "critical_falco_rule_matched"
            )

        if "privilege" in rule.lower():
            base_risk = max(base_risk, 0.80)
            risk_reasons.append(
                "privilege_escalation_detected"
            )

        if "escape" in rule.lower():
            base_risk = max(base_risk, 0.95)
            risk_reasons.append(
                "container_escape_attempt"
            )

        if "crypto" in rule.lower() or (
            "miner" in rule.lower()
        ):
            base_risk = max(base_risk, 0.80)
            risk_reasons.append("cryptomining_detected")

        if source_ip.startswith("185.220"):
            base_risk = max(base_risk, 0.90)
            risk_reasons.append("tor_exit_node_c2")

        return {
            "accessor_identity": accessor,
            "accessor_type": "service_account",
            "data_store_name": (
                f"k8s/{k8s_ns}/{k8s_pod}"
            ),
            "data_path": (
                f"{process_name}: {process_cmd}"
                [:500]
            ),
            "data_classification": "UNKNOWN",
            "bytes_accessed": 0,
            "event_time": raw_event.get(
                "time",
                raw_event.get("timestamp", _now())
            ),
            "source_ip": source_ip,
            "risk_score": min(base_risk, 1.0),
            "risk_reasons": risk_reasons,
            "source_system": "falco_runtime",
            "raw_event": raw_event,
            "falco_rule": rule,
            "falco_priority": priority,
            "falco_output": output[:500],
            "container_id": container_id,
            "container_image": image,
            "k8s_namespace": k8s_ns,
            "k8s_pod": k8s_pod
        }

    def normalize_vulnerability(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize container image vulnerability.
        From: Trivy, Snyk, ECR, ACR, GCR scans.

        High severity CVE with exploit available
        = CRITICAL risk even before exploitation.

        Args:
            raw_event: Vulnerability scan result

        Returns:
            DataAccessEvent with vuln context
        """
        if not raw_event:
            return self._empty_event()

        severity = raw_event.get(
            "Severity",
            raw_event.get("severity", "MEDIUM")
        ).upper()

        cve_id = raw_event.get(
            "VulnerabilityID",
            raw_event.get("cve_id", "UNKNOWN")
        )

        pkg_name = raw_event.get(
            "PkgName",
            raw_event.get("package", "unknown")
        )
        image = raw_event.get(
            "Target",
            raw_event.get("image", "unknown")
        )

        severity_risk = {
            "CRITICAL": 0.85,
            "HIGH":     0.65,
            "MEDIUM":   0.40,
            "LOW":      0.15,
            "UNKNOWN":  0.30
        }
        base_risk = severity_risk.get(
            severity, 0.30
        )

        risk_reasons = [
            f"cve_severity:{severity}",
            f"cve_id:{cve_id}"
        ]

        if raw_event.get(
            "FixedVersion",
            raw_event.get("fixed_version", "")
        ):
            risk_reasons.append(
                "patch_available_not_applied"
            )

        cvss_score = raw_event.get(
            "CVSS",
            raw_event.get("cvss_score", {})
        )
        if isinstance(cvss_score, dict):
            nvd_score = (
                cvss_score.get("nvd", {})
                .get("V3Score", 0)
            )
            if nvd_score >= 9.0:
                base_risk = max(base_risk, 0.90)
                risk_reasons.append(
                    f"cvss_score_critical:{nvd_score}"
                )

        if raw_event.get(
            "ExploitAvailable",
            raw_event.get("exploit_available", False)
        ):
            base_risk = max(base_risk + 0.15, 1.0)
            risk_reasons.append(
                "exploit_publicly_available"
            )

        return {
            "accessor_identity": image,
            "accessor_type": "service_account",
            "data_store_name": image,
            "data_path": (
                f"{pkg_name}/{cve_id}"
            ),
            "data_classification": "UNKNOWN",
            "bytes_accessed": 0,
            "event_time": raw_event.get(
                "LastModifiedDate",
                raw_event.get("scanned_at", _now())
            ),
            "source_ip": "",
            "risk_score": min(base_risk, 1.0),
            "risk_reasons": risk_reasons,
            "source_system": "container_vulnerability",
            "raw_event": raw_event,
            "cve_id": cve_id,
            "cve_severity": severity,
            "affected_package": pkg_name,
            "container_image": image
        }

    def _extract_source_ip(
        self, raw_event: dict
    ) -> str:
        """Extract source IP from K8s audit log"""
        source_ips = raw_event.get("sourceIPs", [])
        if source_ips:
            ip = source_ips[0]
            if ip not in [
                "::1", "127.0.0.1",
                "localhost", ""
            ]:
                return ip

        annotations = raw_event.get(
            "annotations", {}
        )
        return annotations.get("sourceIP", "")

    def _detect_accessor_type(
        self,
        accessor: str,
        user: dict
    ) -> str:
        """Detect if K8s principal is service account"""
        if "system:serviceaccount" in accessor:
            return "service_account"

        groups = user.get("groups", [])
        if "system:serviceaccounts" in groups:
            return "service_account"

        if accessor.startswith("system:"):
            return "system"

        if any(
            kw in accessor.lower()
            for kw in [
                "serviceaccount", "sa-",
                "-sa", "svc"
            ]
        ):
            return "service_account"

        return "human"

    def _detect_classification(
        self,
        resource: str,
        resource_name: str,
        namespace: str
    ) -> str:
        """Detect data classification"""
        combined = (
            f"{resource} {resource_name} {namespace}"
        ).lower()

        if "secret" in combined:
            return "PII"

        if any(
            kw in combined
            for kw in [
                "pci", "payment", "card"
            ]
        ):
            return "PCI"

        if any(
            kw in combined
            for kw in [
                "health", "medical", "patient"
            ]
        ):
            return "PHI"

        if namespace in [
            "kube-system", "kube-public",
            "cert-manager", "monitoring"
        ]:
            return "INTERNAL"

        return "UNKNOWN"

    def _calculate_k8s_risk(
        self,
        verb: str,
        resource: str,
        accessor: str,
        raw_event: dict,
        namespace: str
    ) -> tuple:
        """Calculate K8s audit event risk"""
        risk = 0.0
        reasons = []

        verb_risk = K8S_VERB_RISK.get(verb, 0.05)
        risk += verb_risk
        if verb_risk >= 0.30:
            reasons.append(f"high_risk_verb:{verb}")

        resource_risk = HIGH_RISK_RESOURCES.get(
            resource, 0.0
        )
        risk += resource_risk
        if resource_risk >= 0.25:
            reasons.append(
                f"high_risk_resource:{resource}"
            )

        if namespace == "kube-system":
            risk += 0.20
            reasons.append("kube_system_access")

        if "system:serviceaccount" in accessor:
            if any(
                hr in resource
                for hr in [
                    "secrets", "clusterrole",
                    "rolebinding"
                ]
            ):
                risk += 0.15
                reasons.append(
                    "service_account_privileged_op"
                )

        response_code = (
            raw_event.get(
                "responseStatus", {}
            ).get("code", 200)
        )
        if response_code in [403, 401]:
            risk += 0.10
            reasons.append(
                f"auth_failure:{response_code}"
            )

        if verb == "exec" and resource == "pods":
            risk = max(risk, 0.70)
            reasons.append(
                "pod_exec_command_execution"
            )

        user_agent = raw_event.get(
            "userAgent", ""
        ).lower()
        if any(
            kw in user_agent
            for kw in ["curl", "python", "go-http"]
        ):
            risk += 0.10
            reasons.append(
                "non_standard_client_detected"
            )

        return min(risk, 1.0), reasons

    def _empty_event(self) -> dict:
        """Empty event for invalid input"""
        return {
            "accessor_identity": "unknown",
            "accessor_type": "unknown",
            "data_store_name": "unknown",
            "data_path": "",
            "data_classification": "UNKNOWN",
            "bytes_accessed": 0,
            "event_time": _now(),
            "source_ip": "",
            "risk_score": 0.0,
            "risk_reasons": [],
            "source_system": "kubernetes",
            "raw_event": {}
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")