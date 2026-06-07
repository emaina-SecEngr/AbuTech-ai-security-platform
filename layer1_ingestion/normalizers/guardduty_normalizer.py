"""
Layer 1 — Data Ingestion
AWS GuardDuty Normalizer

Handles AWS GuardDuty findings — Amazon's ML-powered
threat detection service.

WHY GUARDDUTY MATTERS:
    GuardDuty is AWS's native threat detection.
    It uses machine learning and threat intelligence
    to spot crypto mining, compromised credentials,
    reconnaissance, and command-and-control activity
    across CloudTrail, VPC flow logs, and DNS logs.

HOW ABUTECH ADDS VALUE (the pitch):
    GuardDuty's ML only sees AWS, cannot correlate
    with non-AWS sources, and cannot explain its
    decisions for SR 11-7 governance. We ingest
    GuardDuty findings, correlate them with every
    other source, score them with explainable ML,
    map them to MITRE, and unify them across the
    whole estate. GuardDuty finds the AWS threat;
    we tell you how it connects to everything else.

GUARDDUTY FINDING ANATOMY:
    type: "ThreatPurpose:ResourceType/ThreatName"
          e.g. "CryptoCurrency:EC2/BitcoinTool.B!DNS"
    severity: numeric 1.0-8.9
          Low 1.0-3.9, Medium 4.0-6.9, High 7.0-8.9
    resource: the affected AWS resource
    service: action, actor, evidence, count

THREAT PURPOSE TO MITRE TACTIC:
    The first segment of the finding type tells you
    the adversary's goal. We map it to a MITRE tactic
    and, where possible, a specific technique.

USAGE:
    normalizer = GuardDutyNormalizer()
    event = normalizer.normalize(guardduty_finding)
"""

import logging
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)


# GuardDuty numeric severity bands to risk + label
def _severity_band(severity: float) -> tuple:
    """Return (label, base_risk) for a numeric severity"""
    try:
        s = float(severity)
    except (ValueError, TypeError):
        return "UNKNOWN", 0.40

    if s >= 7.0:
        return "HIGH", 0.80
    if s >= 4.0:
        return "MEDIUM", 0.55
    if s >= 1.0:
        return "LOW", 0.30
    return "INFO", 0.12


# GuardDuty threat purpose (first segment of type)
# to MITRE tactic (id, name)
THREAT_PURPOSE_TO_TACTIC = {
    "Backdoor":            ("TA0011", "Command and Control"),
    "CommandAndControl":   ("TA0011", "Command and Control"),
    "CryptoCurrency":      ("TA0040", "Impact"),
    "Discovery":           ("TA0007", "Discovery"),
    "Recon":               ("TA0007", "Discovery"),
    "Exfiltration":        ("TA0010", "Exfiltration"),
    "Impact":              ("TA0040", "Impact"),
    "InitialAccess":       ("TA0001", "Initial Access"),
    "Persistence":         ("TA0003", "Persistence"),
    "PrivilegeEscalation": ("TA0004", "Privilege Escalation"),
    "Trojan":              ("TA0002", "Execution"),
    "UnauthorizedAccess":  ("TA0001", "Initial Access"),
    "CredentialAccess":    ("TA0006", "Credential Access"),
    "DefenseEvasion":      ("TA0005", "Defense Evasion"),
    "Execution":           ("TA0002", "Execution"),
    "Policy":              ("", ""),
    "Stealth":             ("TA0005", "Defense Evasion"),
}

# Threat purpose to a representative MITRE technique
THREAT_PURPOSE_TO_TECHNIQUE = {
    "CryptoCurrency":      "T1496",
    "Backdoor":            "T1071",
    "CommandAndControl":   "T1071",
    "Recon":               "T1046",
    "Discovery":           "T1046",
    "Exfiltration":        "T1048",
    "UnauthorizedAccess":  "T1078",
    "CredentialAccess":    "T1552",
    "PrivilegeEscalation": "T1068",
    "Persistence":         "T1098",
    "Trojan":              "T1204",
    "Impact":              "T1485",
    "DefenseEvasion":      "T1562",
}

# Keyword escalations for specific high-signal threats
GUARDDUTY_KEYWORD_RISK = {
    "bitcointool":      0.82,
    "cryptocurrency":   0.82,
    "tor":              0.85,
    "maliciousipcaller":0.80,
    "credentialexfiltration": 0.90,
    "instancecredentialexfiltration": 0.92,
    "anomalousbehavior":0.70,
    "passwordpolicy":   0.40,
    "datacompromise":   0.85,
}


class GuardDutyNormalizer:
    """
    Normalizes AWS GuardDuty findings into
    DataAccessEvent format.
    """

    def __init__(self):
        self.source_system = "aws_guardduty"

    def normalize(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize an AWS GuardDuty finding.

        Args:
            raw_event: GuardDuty finding dict

        Returns:
            DataAccessEvent compatible dict
        """
        if not raw_event:
            return self._empty_event()

        finding_type = (
            raw_event.get("type", "")
            or raw_event.get("Type", "")
        )

        severity_num = raw_event.get(
            "severity", raw_event.get("Severity", 4.0)
        )
        severity_label, base_risk = _severity_band(
            severity_num
        )

        title = (
            raw_event.get("title", "")
            or raw_event.get("Title", "")
        )
        description = (
            raw_event.get("description", "")
            or raw_event.get("Description", "")
        )

        # Parse the finding type into segments
        threat_purpose, resource_type, threat_name = (
            self._parse_finding_type(finding_type)
        )

        # Extract resource and actor
        resource = self._extract_resource(raw_event)
        source_ip = self._extract_source_ip(raw_event)
        accessor = self._extract_accessor(
            raw_event, resource
        )

        # MITRE mapping
        tactic_id, tactic_name = (
            THREAT_PURPOSE_TO_TACTIC.get(
                threat_purpose, ("", "")
            )
        )
        technique = THREAT_PURPOSE_TO_TECHNIQUE.get(
            threat_purpose, ""
        )

        # Risk scoring
        risk, reasons = self._calculate_risk(
            base_risk, severity_label,
            threat_purpose, finding_type,
            source_ip, raw_event
        )

        return {
            "accessor_identity": accessor,
            "accessor_type": self._accessor_type(
                resource_type
            ),
            "data_store_name": resource,
            "data_path": (title or finding_type)[:300],
            "data_classification": "UNKNOWN",
            "bytes_accessed": 0,
            "event_time": (
                raw_event.get("updatedAt")
                or raw_event.get("createdAt")
                or raw_event.get("UpdatedAt")
                or _now()
            ),
            "source_ip": source_ip,
            "risk_score": min(round(risk, 4), 1.0),
            "risk_reasons": reasons,
            "source_system": "aws_guardduty",
            "raw_event": raw_event,
            "guardduty_type": finding_type,
            "guardduty_threat_purpose": threat_purpose,
            "guardduty_resource_type": resource_type,
            "guardduty_threat_name": threat_name,
            "guardduty_severity": severity_label,
            "guardduty_severity_score": severity_num,
            "mitre_technique": technique,
            "mitre_tactic_id": tactic_id,
            "mitre_tactic": tactic_name,
        }

    def _parse_finding_type(
        self, finding_type: str
    ) -> tuple:
        """
        Parse "ThreatPurpose:ResourceType/ThreatName"
        into its three components.

        Example:
            "CryptoCurrency:EC2/BitcoinTool.B!DNS"
            → ("CryptoCurrency", "EC2",
               "BitcoinTool.B!DNS")
        """
        if not finding_type:
            return "", "", ""

        threat_purpose = ""
        resource_type = ""
        threat_name = ""

        # Split on first colon
        if ":" in finding_type:
            threat_purpose, rest = finding_type.split(
                ":", 1
            )
            # Split rest on first slash
            if "/" in rest:
                resource_type, threat_name = (
                    rest.split("/", 1)
                )
            else:
                resource_type = rest
        else:
            threat_purpose = finding_type

        return (
            threat_purpose.strip(),
            resource_type.strip(),
            threat_name.strip()
        )

    def _extract_resource(
        self, raw_event: dict
    ) -> str:
        """Extract the affected AWS resource"""
        resource = raw_event.get(
            "resource", raw_event.get("Resource", {})
        )
        if not isinstance(resource, dict):
            return "unknown_resource"

        # EC2 instance
        instance = resource.get(
            "instanceDetails",
            resource.get("InstanceDetails", {})
        )
        if isinstance(instance, dict) and instance:
            iid = (
                instance.get("instanceId")
                or instance.get("InstanceId")
            )
            if iid:
                return str(iid)

        # IAM / access key
        access_key = resource.get(
            "accessKeyDetails",
            resource.get("AccessKeyDetails", {})
        )
        if isinstance(access_key, dict) and access_key:
            user = (
                access_key.get("userName")
                or access_key.get("UserName")
            )
            if user:
                return str(user)

        # S3 bucket
        s3 = resource.get(
            "s3BucketDetails",
            resource.get("S3BucketDetails", [])
        )
        if isinstance(s3, list) and s3:
            name = s3[0].get("name") if isinstance(
                s3[0], dict
            ) else None
            if name:
                return str(name)

        # Resource type fallback
        rtype = (
            resource.get("resourceType")
            or resource.get("ResourceType")
        )
        return str(rtype) if rtype else (
            "unknown_resource"
        )

    def _extract_source_ip(
        self, raw_event: dict
    ) -> str:
        """Extract attacker source IP from service action"""
        service = raw_event.get(
            "service", raw_event.get("Service", {})
        )
        if not isinstance(service, dict):
            return ""

        action = service.get(
            "action", service.get("Action", {})
        )
        if not isinstance(action, dict):
            return ""

        # Network connection action
        for key in [
            "networkConnectionAction",
            "awsApiCallAction",
            "NetworkConnectionAction",
            "AwsApiCallAction"
        ]:
            act = action.get(key, {})
            if isinstance(act, dict):
                remote = act.get(
                    "remoteIpDetails",
                    act.get("RemoteIpDetails", {})
                )
                if isinstance(remote, dict):
                    ip = (
                        remote.get("ipAddressV4")
                        or remote.get("IpAddressV4")
                    )
                    if ip:
                        return str(ip)

        return ""

    def _extract_accessor(
        self, raw_event: dict, resource: str
    ) -> str:
        """Determine the accessor identity"""
        service = raw_event.get(
            "service", raw_event.get("Service", {})
        )
        if isinstance(service, dict):
            action = service.get(
                "action", service.get("Action", {})
            )
            if isinstance(action, dict):
                api = action.get(
                    "awsApiCallAction",
                    action.get("AwsApiCallAction", {})
                )
                if isinstance(api, dict):
                    caller = api.get(
                        "callerType", ""
                    )
                    if caller:
                        return resource
        return resource

    def _accessor_type(
        self, resource_type: str
    ) -> str:
        """Map AWS resource type to accessor type"""
        rt = resource_type.lower()
        if "iam" in rt or "accesskey" in rt:
            return "iam_principal"
        if "ec2" in rt:
            return "compute_instance"
        if "s3" in rt:
            return "data_store"
        if "eks" in rt or "kubernetes" in rt:
            return "workload"
        if "lambda" in rt:
            return "serverless"
        return "cloud_resource"

    def _calculate_risk(
        self,
        base_risk: float,
        severity_label: str,
        threat_purpose: str,
        finding_type: str,
        source_ip: str,
        raw_event: dict
    ) -> tuple:
        """Calculate GuardDuty finding risk"""
        risk = base_risk
        reasons = [
            f"guardduty_severity:{severity_label}",
            f"threat_purpose:{threat_purpose}",
        ]

        type_lower = finding_type.lower()

        # Keyword escalations
        for keyword, kw_risk in (
            GUARDDUTY_KEYWORD_RISK.items()
        ):
            if keyword in type_lower:
                risk = max(risk, kw_risk)
                reasons.append(
                    f"guardduty_signal:{keyword}"
                )
                break

        # High-impact threat purposes escalate
        if threat_purpose in (
            "Exfiltration", "Impact",
            "Backdoor", "CommandAndControl"
        ):
            risk = max(risk, 0.78)
            reasons.append(
                f"high_impact_purpose:{threat_purpose}"
            )

        if threat_purpose == "CryptoCurrency":
            risk = max(risk, 0.80)
            reasons.append("cryptomining_detected")

        if threat_purpose == "UnauthorizedAccess":
            risk = max(risk, 0.75)
            reasons.append("unauthorized_access")

        # Credential exfiltration is critical
        if "credentialexfiltration" in type_lower:
            risk = max(risk, 0.92)
            reasons.append(
                "iam_credential_exfiltration"
            )

        # Tor / known-bad source
        if source_ip.startswith("185.220"):
            risk = max(risk, 0.90)
            reasons.append("tor_exit_node_src")

        # Repeated finding (count) escalates
        service = raw_event.get(
            "service", raw_event.get("Service", {})
        )
        if isinstance(service, dict):
            count = service.get(
                "count", service.get("Count", 0)
            )
            try:
                if int(count) >= 10:
                    risk = min(risk + 0.05, 1.0)
                    reasons.append(
                        f"repeated_finding:{count}"
                    )
            except (ValueError, TypeError):
                pass

        return risk, reasons

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
            "source_system": "aws_guardduty",
            "raw_event": {}
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")