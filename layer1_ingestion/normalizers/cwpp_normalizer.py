"""
Layer 1 — Data Ingestion
Cloud Workload Protection Platform (CWPP) Normalizer

Handles RUNTIME workload protection events — the
behavioral half of cloud security that complements
CSPM's configuration scanning.

THE CSPM vs CWPP DISTINCTION:
    CSPM asks: "Is this workload misconfigured?"
               (static — checks settings before runtime)
    CWPP asks: "Is this workload behaving maliciously
               RIGHT NOW while it runs?"
               (dynamic — watches live behavior)

    Together with CIEM (identity entitlements) these
    three form CNAPP — Cloud Native Application
    Protection Platform.

SOURCES HANDLED:
    Prisma Cloud Compute (Twistlock):
        Runtime defense, container forensics,
        host/container/serverless protection
    Aqua Security:
        Runtime policy violations, drift prevention
    CrowdStrike Falcon Cloud Workload:
        Cloud workload runtime detections
    Sysdig Secure:
        Falco-based runtime threat detection
    Generic CWPP runtime alerts:
        Any agent reporting workload behavior

WHY CWPP MATTERS FOR BANKS:
    Banks run core workloads in containers and VMs.
    A misconfiguration (CSPM) is a door left open.
    A runtime threat (CWPP) is the burglar already
    inside the workload — escaping the container,
    mining crypto, spawning a reverse shell, or
    moving laterally to the host.

    ATTACK SCENARIOS THIS NORMALIZER CATCHES:

    1. Container escape to host:
       Workload breaks container isolation,
       mounts host filesystem, steals cloud creds.
       → T1611 Escape to Host, risk CRITICAL

    2. Cryptomining hijack:
       Compromised workload mines cryptocurrency,
       high CPU + miner pool connections.
       → T1496 Resource Hijacking

    3. Reverse shell from workload:
       Workload spawns shell connecting outbound
       to attacker C2.
       → T1059 Command and Scripting Interpreter

    4. Runtime drift / unexpected binary:
       Process running that was not in the image.
       → T1525 Implant Internal Image / drift

    5. Workload lateral movement:
       One workload connecting to many internal
       hosts it never normally touches.
       → T1021 Remote Services

NOTE ON SCOPE vs KubernetesNormalizer:
    KubernetesNormalizer focuses on K8s audit logs,
    Falco K8s events, and container image vulns.
    CWPPNormalizer focuses on RUNTIME WORKLOAD
    PROTECTION across containers AND VMs AND
    serverless — the dedicated CWPP product layer
    (Prisma Compute, Aqua, Falcon Cloud Workload).
    They are complementary, not duplicate.

USAGE:
    normalizer = CWPPNormalizer()
    event = normalizer.normalize(prisma_runtime_alert)
    event = normalizer.normalize_aqua(aqua_alert)
    event = normalizer.normalize_falcon_cwp(cs_alert)
"""

import logging
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)


# CWPP runtime event type to base risk
CWPP_EVENT_RISK = {
    "container_escape":        0.95,
    "host_breakout":           0.95,
    "reverse_shell":           0.90,
    "cryptomining":            0.82,
    "privilege_escalation":    0.85,
    "lateral_movement":        0.80,
    "runtime_drift":           0.70,
    "malicious_binary":        0.85,
    "file_integrity":          0.65,
    "suspicious_process":      0.70,
    "fileless_execution":      0.85,
    "data_exfiltration":       0.88,
    "policy_violation":        0.55,
    "anomalous_network":       0.65,
    "unknown":                 0.50,
}

# Vendor severity to base risk
CWPP_SEVERITY_RISK = {
    "CRITICAL": 0.88,
    "HIGH":     0.70,
    "MEDIUM":   0.45,
    "LOW":      0.20,
    "INFO":     0.10,
    "UNKNOWN":  0.40,
}

# Keyword to canonical event type
# Maps vendor rule/alert text to a normalized type
CWPP_KEYWORD_MAP = {
    "escape":           "container_escape",
    "breakout":         "host_breakout",
    "break out":        "host_breakout",
    "reverse shell":    "reverse_shell",
    "reverse_shell":    "reverse_shell",
    "shell":            "suspicious_process",
    "crypto":           "cryptomining",
    "miner":            "cryptomining",
    "mining":           "cryptomining",
    "privilege":        "privilege_escalation",
    "privesc":          "privilege_escalation",
    "lateral":          "lateral_movement",
    "drift":            "runtime_drift",
    "unexpected binary":"malicious_binary",
    "malicious":        "malicious_binary",
    "file integrity":   "file_integrity",
    "fim":              "file_integrity",
    "tampering":        "file_integrity",
    "fileless":         "fileless_execution",
    "exfil":            "data_exfiltration",
    "exfiltration":     "data_exfiltration",
    "outbound":         "anomalous_network",
    "c2":               "anomalous_network",
    "command and control": "anomalous_network",
}

# Canonical event type to MITRE technique
CWPP_EVENT_TO_MITRE = {
    "container_escape":     "T1611",
    "host_breakout":        "T1611",
    "reverse_shell":        "T1059",
    "cryptomining":         "T1496",
    "privilege_escalation": "T1068",
    "lateral_movement":     "T1021",
    "runtime_drift":        "T1525",
    "malicious_binary":     "T1204",
    "file_integrity":       "T1565",
    "suspicious_process":   "T1059",
    "fileless_execution":   "T1620",
    "data_exfiltration":    "T1048",
    "policy_violation":     "T1610",
    "anomalous_network":    "T1071",
}


class CWPPNormalizer:
    """
    Normalizes Cloud Workload Protection Platform
    runtime events into DataAccessEvent format.

    Covers Prisma Cloud Compute, Aqua Security,
    CrowdStrike Falcon Cloud Workload, and Sysdig.
    """

    def __init__(self):
        self.source_system = "cwpp"

    def normalize(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize a generic / Prisma Cloud Compute
        runtime workload protection alert.

        Args:
            raw_event: CWPP runtime alert dict

        Returns:
            DataAccessEvent compatible dict
        """
        if not raw_event:
            return self._empty_event()

        # Prisma uses 'type', '_id', 'msg', 'severity'
        severity = str(
            raw_event.get(
                "severity",
                raw_event.get("Severity", "MEDIUM")
            )
        ).upper()

        rule_text = (
            raw_event.get("rule", "")
            or raw_event.get("ruleName", "")
            or raw_event.get("msg", "")
            or raw_event.get("type", "")
        )

        workload = (
            raw_event.get("hostname", "")
            or raw_event.get("host", "")
            or raw_event.get("containerName", "")
            or raw_event.get("imageName", "")
            or "unknown_workload"
        )

        process_name = (
            raw_event.get("processName", "")
            or raw_event.get("process", "")
        )
        process_cmd = (
            raw_event.get("command", "")
            or raw_event.get("cmdline", "")
        )
        source_ip = (
            raw_event.get("ip", "")
            or raw_event.get("sourceIP", "")
            or raw_event.get("remoteIP", "")
        )

        return self._build_event(
            vendor="prisma_cloud_compute",
            severity=severity,
            rule_text=rule_text,
            workload=workload,
            process_name=process_name,
            process_cmd=process_cmd,
            source_ip=source_ip,
            raw_event=raw_event
        )

    def normalize_aqua(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize Aqua Security runtime alert.

        Args:
            raw_event: Aqua alert dict

        Returns:
            DataAccessEvent compatible dict
        """
        if not raw_event:
            return self._empty_event()

        severity = str(
            raw_event.get("severity", "MEDIUM")
        ).upper()

        rule_text = (
            raw_event.get("control", "")
            or raw_event.get("rule", "")
            or raw_event.get("description", "")
        )

        workload = (
            raw_event.get("container", "")
            or raw_event.get("image", "")
            or raw_event.get("host", "")
            or "unknown_workload"
        )

        process_name = raw_event.get("process", "")
        process_cmd = raw_event.get("cmd", "")
        source_ip = raw_event.get("source_ip", "")

        return self._build_event(
            vendor="aqua_security",
            severity=severity,
            rule_text=rule_text,
            workload=workload,
            process_name=process_name,
            process_cmd=process_cmd,
            source_ip=source_ip,
            raw_event=raw_event
        )

    def normalize_falcon_cwp(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize CrowdStrike Falcon Cloud Workload
        Protection detection.

        Args:
            raw_event: Falcon CWP detection dict

        Returns:
            DataAccessEvent compatible dict
        """
        if not raw_event:
            return self._empty_event()

        severity_num = raw_event.get(
            "severity", raw_event.get("Severity", 3)
        )
        # Falcon uses numeric severity 1-5 sometimes
        severity = self._falcon_severity(severity_num)

        rule_text = (
            raw_event.get("detect_name", "")
            or raw_event.get("technique", "")
            or raw_event.get("description", "")
        )

        workload = (
            raw_event.get("hostname", "")
            or raw_event.get("instance_id", "")
            or raw_event.get("container_id", "")
            or "unknown_workload"
        )

        process_name = (
            raw_event.get("filename", "")
            or raw_event.get("process_name", "")
        )
        process_cmd = raw_event.get(
            "command_line", ""
        )
        source_ip = (
            raw_event.get("local_ip", "")
            or raw_event.get("external_ip", "")
        )

        return self._build_event(
            vendor="crowdstrike_falcon_cwp",
            severity=severity,
            rule_text=rule_text,
            workload=workload,
            process_name=process_name,
            process_cmd=process_cmd,
            source_ip=source_ip,
            raw_event=raw_event
        )

    def _build_event(
        self,
        vendor: str,
        severity: str,
        rule_text: str,
        workload: str,
        process_name: str,
        process_cmd: str,
        source_ip: str,
        raw_event: dict
    ) -> dict:
        """Shared event construction logic"""

        event_type = self._classify_event(rule_text)

        # Base risk: take the higher of severity
        # and event-type risk so a CRITICAL escape
        # never gets under-scored
        sev_risk = CWPP_SEVERITY_RISK.get(
            severity, 0.40
        )
        type_risk = CWPP_EVENT_RISK.get(
            event_type, 0.50
        )
        risk = max(sev_risk, type_risk)

        risk_reasons = [
            f"cwpp_vendor:{vendor}",
            f"cwpp_severity:{severity}",
            f"cwpp_event_type:{event_type}",
        ]

        # Escalations
        if event_type in (
            "container_escape", "host_breakout"
        ):
            risk = max(risk, 0.95)
            risk_reasons.append(
                "workload_escape_attempt"
            )

        if event_type == "cryptomining":
            risk = max(risk, 0.82)
            risk_reasons.append(
                "cryptomining_detected"
            )

        if event_type == "reverse_shell":
            risk = max(risk, 0.90)
            risk_reasons.append(
                "reverse_shell_spawned"
            )

        if event_type == "privilege_escalation":
            risk = max(risk, 0.85)
            risk_reasons.append(
                "workload_privilege_escalation"
            )

        if event_type == "lateral_movement":
            risk = max(risk, 0.80)
            risk_reasons.append(
                "workload_lateral_movement"
            )

        # Tor / known-bad source IP
        if source_ip.startswith("185.220"):
            risk = max(risk, 0.90)
            risk_reasons.append(
                "tor_exit_node_src"
            )

        # MITRE technique tag
        mitre = CWPP_EVENT_TO_MITRE.get(event_type)
        if mitre:
            risk_reasons.append(
                f"mitre_technique:{mitre}"
            )

        data_path = process_name
        if process_cmd:
            data_path = (
                f"{process_name}: {process_cmd}"
            )[:500]

        return {
            "accessor_identity": (
                process_name or workload
            ),
            "accessor_type": "workload",
            "data_store_name": workload,
            "data_path": data_path,
            "data_classification": "UNKNOWN",
            "bytes_accessed": 0,
            "event_time": (
                raw_event.get("time")
                or raw_event.get("timestamp")
                or raw_event.get("eventTime")
                or _now()
            ),
            "source_ip": source_ip,
            "risk_score": min(round(risk, 4), 1.0),
            "risk_reasons": risk_reasons,
            "source_system": f"cwpp_{vendor}",
            "raw_event": raw_event,
            "cwpp_vendor": vendor,
            "cwpp_event_type": event_type,
            "cwpp_severity": severity,
            "cwpp_rule": rule_text[:300],
            "workload_name": workload,
            "mitre_technique": mitre or "",
        }

    def _classify_event(
        self, rule_text: str
    ) -> str:
        """
        Classify vendor rule text into a canonical
        CWPP event type using keyword matching.
        """
        if not rule_text:
            return "unknown"

        text = rule_text.lower()

        for keyword, event_type in (
            CWPP_KEYWORD_MAP.items()
        ):
            if keyword in text:
                return event_type

        return "policy_violation"

    def _falcon_severity(
        self, severity_num
    ) -> str:
        """Map Falcon numeric severity to label"""
        try:
            n = int(severity_num)
        except (ValueError, TypeError):
            return str(severity_num).upper()

        if n >= 5:
            return "CRITICAL"
        if n == 4:
            return "HIGH"
        if n == 3:
            return "MEDIUM"
        if n == 2:
            return "LOW"
        return "INFO"

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
            "source_system": "cwpp",
            "raw_event": {}
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")