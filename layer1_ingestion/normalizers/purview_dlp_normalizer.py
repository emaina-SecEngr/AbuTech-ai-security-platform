"""
Layer 1 — Data Ingestion
Microsoft Purview DLP Normalizer

Handles Data Loss Prevention events from Microsoft
Purview — detecting sensitive data leaving the
organization across Microsoft 365 and endpoints.

WHY THIS MATTERS:
    DLP is core to financial-sector compliance.
    PCI-DSS requires DLP for cardholder data.
    GDPR requires DLP for personal data.
    HIPAA requires DLP for health records.

    This normalizer turns raw Purview DLP events
    into scored, MITRE-mapped exfiltration signals.

WHAT PURVIEW DLP MONITORS:
    Exchange Online:
        Emails containing sensitive data sent
        externally
    SharePoint / OneDrive:
        Sensitive files shared externally or
        uploaded to personal storage
    Microsoft Teams:
        Sensitive info posted in chats/channels
    Endpoint DLP:
        USB copy, clipboard, print, cloud upload
        of sensitive files from managed devices

THE DLP EVENT ANATOMY:
    Policy matched: which DLP rule fired
    Sensitive info type: SSN, credit card, PHI
    Action: Block / BlockOverride / Audit / Allow
    Location/workload: Exchange/SharePoint/Endpoint
    User, recipient, file name

DLP TO MITRE MAPPING:
    Email exfil       → T1048 / T1567.001
    Cloud upload      → T1567.002 (web service)
    SharePoint share  → T1530
    USB copy          → T1052.001
    Clipboard/print   → T1052

THE RISK LOGIC:
    A BLOCKED action = DLP worked, lower urgency.
    An ALLOWED or OVERRIDDEN action with PCI/PHI
    = data actually left = HIGH risk.
    External recipient + sensitive type = elevated.

USAGE:
    normalizer = PurviewDLPNormalizer()
    event = normalizer.normalize(dlp_event)
"""

import logging
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)


# Sensitive information type to base risk + classification
SENSITIVE_INFO_TYPES = {
    "credit card number":      ("PCI", 0.70),
    "credit card":             ("PCI", 0.70),
    "creditcardnumber":        ("PCI", 0.70),
    "bank account":            ("PCI", 0.60),
    "abaroutingnumber":        ("PCI", 0.55),
    "ssn":                     ("PII", 0.65),
    "social security":         ("PII", 0.65),
    "u.s. social security number": ("PII", 0.65),
    "passport":                ("PII", 0.55),
    "drivers license":         ("PII", 0.50),
    "medical":                 ("PHI", 0.65),
    "health":                  ("PHI", 0.65),
    "diagnosis":               ("PHI", 0.65),
    "patient":                 ("PHI", 0.65),
    "phi":                     ("PHI", 0.65),
    "pii":                     ("PII", 0.55),
    "ip address":              ("INTERNAL", 0.30),
}

# DLP action to risk modifier
# A blocked action means DLP stopped the leak.
# An allowed/overridden action means data left.
DLP_ACTION_RISK = {
    "block":          0.0,    # DLP worked
    "blocked":        0.0,
    "blockoverride":  0.30,   # user overrode the block
    "override":       0.30,
    "audit":          0.20,   # logged but allowed
    "allow":          0.25,
    "allowed":        0.25,
    "notify":         0.15,
    "warn":           0.15,
    "remove":         0.0,
}

# Workload/location to MITRE technique
DLP_WORKLOAD_TECHNIQUE = {
    "exchange":     "T1048",       # email exfil
    "email":        "T1048",
    "sharepoint":   "T1530",       # cloud storage
    "onedrive":     "T1567.002",   # web service
    "teams":        "T1567.002",
    "endpoint":     "T1052",       # physical/local
    "usb":          "T1052.001",   # USB
    "device":       "T1052",
    "cloud":        "T1567.002",
}


class PurviewDLPNormalizer:
    """
    Normalizes Microsoft Purview DLP events into
    DataAccessEvent format.

    Detects sensitive-data exfiltration across
    Exchange, SharePoint, OneDrive, Teams, and
    endpoint DLP.
    """

    def __init__(self):
        self.source_system = "purview_dlp"

    def normalize(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize a Microsoft Purview DLP event.

        Args:
            raw_event: Purview DLP event dict
                       (Office 365 Management API or
                       Compliance audit format)

        Returns:
            DataAccessEvent compatible dict
        """
        if not raw_event:
            return self._empty_event()

        # DLP events may nest data differently.
        # Handle Office 365 Management API format.
        user = (
            raw_event.get("UserId", "")
            or raw_event.get("userId", "")
            or raw_event.get("user", "")
            or raw_event.get("Actor", "")
            or "unknown"
        )

        workload = str(
            raw_event.get("Workload", "")
            or raw_event.get("workload", "")
            or raw_event.get("location", "")
        ).lower()

        # Policy and rule
        policy_name = (
            raw_event.get("PolicyName", "")
            or raw_event.get("policyName", "")
            or self._extract_policy(raw_event)
        )

        # Sensitive info types matched
        sit_list = self._extract_sensitive_types(
            raw_event
        )

        # Action taken
        action = str(
            raw_event.get("DLPAction", "")
            or raw_event.get("action", "")
            or raw_event.get("Operation", "")
            or self._extract_action(raw_event)
        ).lower()

        # File / object
        object_id = (
            raw_event.get("ObjectId", "")
            or raw_event.get("objectId", "")
            or raw_event.get("fileName", "")
            or raw_event.get("Subject", "")
            or "unknown_object"
        )

        # Recipients (for email)
        recipients = self._extract_recipients(
            raw_event
        )
        external_recipient = (
            self._has_external_recipient(recipients)
        )

        source_ip = (
            raw_event.get("ClientIP", "")
            or raw_event.get("clientIp", "")
            or raw_event.get("source_ip", "")
        )

        # Classify + score
        classification, base_risk = (
            self._classify_sensitivity(sit_list)
        )
        technique = self._map_technique(workload)
        risk, reasons = self._calculate_risk(
            base_risk, action, classification,
            external_recipient, sit_list,
            workload, source_ip, technique
        )

        return {
            "accessor_identity": user,
            "accessor_type": "human",
            "data_store_name": (
                self._workload_label(workload)
            ),
            "data_path": object_id[:300],
            "data_classification": classification,
            "bytes_accessed": 0,
            "event_time": (
                raw_event.get("CreationTime")
                or raw_event.get("creationTime")
                or raw_event.get("timestamp")
                or _now()
            ),
            "source_ip": source_ip,
            "risk_score": min(round(risk, 4), 1.0),
            "risk_reasons": reasons,
            "source_system": "purview_dlp",
            "raw_event": raw_event,
            "dlp_policy": policy_name[:200],
            "dlp_action": action,
            "dlp_workload": workload,
            "dlp_sensitive_types": sit_list,
            "dlp_external_recipient": (
                external_recipient
            ),
            "dlp_recipients": recipients[:5],
            "mitre_technique": technique,
        }

    def _extract_sensitive_types(
        self, raw_event: dict
    ) -> list:
        """Extract sensitive info types matched"""
        types = []

        # Direct field
        sit = (
            raw_event.get("SensitiveInfoTypeData")
            or raw_event.get("sensitiveInfoTypes")
            or raw_event.get("SensitiveInformationType")
        )

        if isinstance(sit, list):
            for item in sit:
                if isinstance(item, dict):
                    name = (
                        item.get("SensitiveType")
                        or item.get("name")
                        or item.get("Name")
                    )
                    if name:
                        types.append(str(name))
                elif isinstance(item, str):
                    types.append(item)
        elif isinstance(sit, str):
            types.append(sit)

        # Try nested policy details
        if not types:
            details = raw_event.get(
                "PolicyDetails", []
            )
            if isinstance(details, list):
                for d in details:
                    rules = d.get("Rules", []) \
                        if isinstance(d, dict) else []
                    for r in rules:
                        cond = r.get(
                            "ConditionsMatched", {}
                        ) if isinstance(r, dict) else {}
                        sits = cond.get(
                            "SensitiveInformation", []
                        )
                        for s in sits:
                            name = s.get(
                                "SensitiveType"
                            ) if isinstance(
                                s, dict
                            ) else None
                            if name:
                                types.append(str(name))

        return types

    def _extract_recipients(
        self, raw_event: dict
    ) -> list:
        """Extract email recipients"""
        recipients = (
            raw_event.get("Recipients")
            or raw_event.get("recipients")
            or raw_event.get("To")
            or []
        )
        if isinstance(recipients, str):
            return [recipients]
        if isinstance(recipients, list):
            return [str(r) for r in recipients]
        return []

    def _extract_policy(
        self, raw_event: dict
    ) -> str:
        """Extract policy name from nested details"""
        details = raw_event.get("PolicyDetails", [])
        if isinstance(details, list) and details:
            first = details[0]
            if isinstance(first, dict):
                return first.get("PolicyName", "")
        return "DLP Policy"

    def _extract_action(
        self, raw_event: dict
    ) -> str:
        """Extract action from nested policy details"""
        details = raw_event.get("PolicyDetails", [])
        if isinstance(details, list):
            for d in details:
                if not isinstance(d, dict):
                    continue
                for r in d.get("Rules", []):
                    if not isinstance(r, dict):
                        continue
                    actions = r.get("Actions", [])
                    if actions:
                        return str(actions[0])
        return "audit"

    def _has_external_recipient(
        self, recipients: list
    ) -> bool:
        """Detect if any recipient is external"""
        internal_domains = [
            "company.com", "bank.com",
            "internal", "corp"
        ]
        for r in recipients:
            r_lower = r.lower()
            if "@" in r_lower:
                domain = r_lower.split("@")[-1]
                if not any(
                    d in domain
                    for d in internal_domains
                ):
                    return True
        return False

    def _classify_sensitivity(
        self, sit_list: list
    ) -> tuple:
        """Classify data and get base risk"""
        classification = "UNKNOWN"
        base_risk = 0.30
        highest = 0.0

        for sit in sit_list:
            sit_lower = sit.lower()
            for key, (cls, risk) in (
                SENSITIVE_INFO_TYPES.items()
            ):
                if key in sit_lower:
                    if risk > highest:
                        highest = risk
                        classification = cls
                        base_risk = risk

        return classification, base_risk

    def _map_technique(
        self, workload: str
    ) -> str:
        """Map workload to MITRE exfil technique"""
        for key, technique in (
            DLP_WORKLOAD_TECHNIQUE.items()
        ):
            if key in workload:
                return technique
        return "T1530"

    def _calculate_risk(
        self,
        base_risk: float,
        action: str,
        classification: str,
        external_recipient: bool,
        sit_list: list,
        workload: str,
        source_ip: str,
        technique: str
    ) -> tuple:
        """Calculate DLP event risk"""
        risk = base_risk
        reasons = [
            f"dlp_classification:{classification}",
            f"dlp_workload:{workload}",
        ]

        if sit_list:
            reasons.append(
                f"sensitive_types:{len(sit_list)}"
            )

        # Action modifier
        action_mod = 0.0
        for key, mod in DLP_ACTION_RISK.items():
            if key in action:
                action_mod = mod
                break

        is_clean_block = (
            "block" in action
            and "override" not in action
        )

        if is_clean_block:
            # DLP successfully blocked the data loss.
            # Data never left, so this is the final
            # word - external recipient is irrelevant
            # because nothing was exfiltrated.
            reasons.append("dlp_blocked_data_loss")
            if technique:
                reasons.append(
                    f"mitre_technique:{technique}"
                )
            return min(risk, 0.45), reasons

        # Data was allowed, audited, or overridden -
        # meaning it actually left the organization
        risk += action_mod
        if action_mod >= 0.25:
            reasons.append(
                f"data_left_org:{action}"
            )

        # External recipient is a major escalator
        if external_recipient:
            risk += 0.20
            reasons.append("external_recipient")

        # PCI/PHI that actually left = critical
        if classification in ("PCI", "PHI"):
            if external_recipient:
                risk = max(risk, 0.85)
                reasons.append(
                    f"{classification.lower()}_exfiltrated"
                )

        # Tor / bad source
        if source_ip.startswith("185.220"):
            risk = max(risk, 0.90)
            reasons.append("tor_exit_node_src")

        if technique:
            reasons.append(
                f"mitre_technique:{technique}"
            )

        return max(risk, 0.0), reasons

    def _workload_label(
        self, workload: str
    ) -> str:
        """Human-readable workload label"""
        labels = {
            "exchange":   "Exchange Online",
            "email":      "Exchange Online",
            "sharepoint": "SharePoint Online",
            "onedrive":   "OneDrive for Business",
            "teams":      "Microsoft Teams",
            "endpoint":   "Endpoint DLP",
            "device":     "Endpoint DLP",
        }
        for key, label in labels.items():
            if key in workload:
                return label
        return "Microsoft Purview DLP"

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
            "source_system": "purview_dlp",
            "raw_event": {}
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")