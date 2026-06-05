"""
Layer 1 — Data Ingestion
Microsoft Defender for Cloud Normalizer

Handles Microsoft's Azure-native CNAPP alerts —
the Azure equivalent of AWS GuardDuty + Security Hub.

WHY THIS MATTERS:
    Microsoft Defender for Cloud is THE Azure CNAPP.
    It feeds Microsoft Defender XDR and Sentinel.
    Financial-sector banks running Azure rely on it.
    This normalizer makes the platform fluent in the
    Sentinel + Defender XDR stack used by IBM SIOC.

THE DEFENDER FOR CLOUD PLANS:
    Defender for Servers:
        VM threat detection (fileless attacks,
        crypto mining, suspicious processes)
    Defender for Containers:
        AKS runtime threats + image scanning
        (the Azure CWPP layer)
    Defender for Storage:
        Malicious blob upload, anomalous access,
        sensitive data exposure
    Defender for Key Vault:
        Anomalous secret access patterns
    Defender for SQL:
        SQL injection, anomalous DB access
    Defender for App Service:
        Web shell, suspicious app behavior
    Defender for Resource Manager:
        Suspicious control-plane operations
    Defender CSPM:
        Azure configuration posture findings

ALERT FORMAT (Microsoft Security Graph):
    Defender alerts use a consistent schema:
        AlertType / alertDisplayName
        Severity (High/Medium/Low/Informational)
        CompromisedEntity
        ProductName ("Azure Security Center")
        ResourceIdentifiers
        Entities (account, host, ip, file, process)
        Intent (MITRE-aligned kill chain stage)

WHY THE 'Intent' FIELD IS GOLD:
    Defender already maps alerts to a kill-chain
    intent (Exfiltration, LateralMovement, etc).
    We translate that directly to MITRE tactics —
    so Defender does half the MITRE work for us.

ATTACK SCENARIOS THIS CATCHES:
    1. Crypto mining on an Azure VM
       → Defender for Servers → T1496
    2. Malicious container in AKS
       → Defender for Containers → T1610
    3. Anomalous blob download (data theft)
       → Defender for Storage → T1530
    4. Suspicious Key Vault secret enumeration
       → Defender for Key Vault → T1552
    5. SQL injection against Azure SQL
       → Defender for SQL → T1190

USAGE:
    normalizer = DefenderForCloudNormalizer()
    event = normalizer.normalize(defender_alert)
"""

import logging
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)


# Defender severity to base risk
DEFENDER_SEVERITY_RISK = {
    "HIGH":          0.80,
    "MEDIUM":        0.55,
    "LOW":           0.30,
    "INFORMATIONAL": 0.12,
    "INFO":          0.12,
    "UNKNOWN":       0.40,
}

# Defender 'Intent' (kill chain stage) to MITRE tactic
# Microsoft pre-maps alerts to a kill-chain intent.
# We translate that to a MITRE tactic ID + name.
DEFENDER_INTENT_TO_TACTIC = {
    "PreAttack":          ("TA0043", "Reconnaissance"),
    "InitialAccess":      ("TA0001", "Initial Access"),
    "Persistence":        ("TA0003", "Persistence"),
    "PrivilegeEscalation":("TA0004", "Privilege Escalation"),
    "DefenseEvasion":     ("TA0005", "Defense Evasion"),
    "CredentialAccess":   ("TA0006", "Credential Access"),
    "Discovery":          ("TA0007", "Discovery"),
    "LateralMovement":    ("TA0008", "Lateral Movement"),
    "Execution":          ("TA0002", "Execution"),
    "Collection":         ("TA0009", "Collection"),
    "Exfiltration":       ("TA0010", "Exfiltration"),
    "CommandAndControl":  ("TA0011", "Command and Control"),
    "Impact":             ("TA0040", "Impact"),
    "Probing":            ("TA0043", "Reconnaissance"),
}

# Defender plan (product/resource) to MITRE technique
# Maps the source plan + keywords to a likely technique
DEFENDER_PLAN_TECHNIQUE = {
    "servers_cryptomining":   "T1496",
    "servers_fileless":       "T1620",
    "servers_suspicious":     "T1059",
    "containers_runtime":     "T1610",
    "containers_escape":      "T1611",
    "storage_exfil":          "T1530",
    "storage_malware":        "T1204",
    "keyvault_access":        "T1552",
    "sql_injection":          "T1190",
    "appservice_webshell":    "T1505.003",
    "rm_suspicious":          "T1078.004",
}

# Keyword classification for the alert text
DEFENDER_KEYWORD_TECHNIQUE = {
    "cryptomining":       "T1496",
    "crypto mining":      "T1496",
    "coin miner":         "T1496",
    "miner":              "T1496",
    "fileless":           "T1620",
    "reverse shell":      "T1059",
    "web shell":          "T1505.003",
    "webshell":           "T1505.003",
    "sql injection":      "T1190",
    "container escape":   "T1611",
    "privilege escalation": "T1068",
    "lateral movement":   "T1021",
    "data exfiltration":  "T1530",
    "exfiltration":       "T1048",
    "anomalous access":   "T1530",
    "brute force":        "T1110",
    "malware":            "T1204",
    "ransomware":         "T1486",
    "secret":             "T1552",
    "credential":         "T1552",
}


class DefenderForCloudNormalizer:
    """
    Normalizes Microsoft Defender for Cloud alerts
    into DataAccessEvent format.

    Handles all Defender plans: Servers, Containers,
    Storage, Key Vault, SQL, App Service, Resource
    Manager, and CSPM.
    """

    def __init__(self):
        self.source_system = "defender_for_cloud"

    def normalize(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize a Microsoft Defender for Cloud alert.

        Args:
            raw_event: Defender alert dict (Security
                       Graph or Security Center format)

        Returns:
            DataAccessEvent compatible dict
        """
        if not raw_event:
            return self._empty_event()

        # Defender alerts may nest under 'properties'
        # (Azure Resource Graph) or be flat. Handle both.
        props = raw_event.get("properties", raw_event)

        severity = str(
            props.get("severity",
                      props.get("Severity", "MEDIUM"))
        ).upper()

        alert_name = (
            props.get("alertDisplayName", "")
            or props.get("AlertDisplayName", "")
            or props.get("alertType", "")
            or props.get("AlertType", "")
            or props.get("displayName", "")
        )

        description = (
            props.get("description", "")
            or props.get("Description", "")
        )

        intent = (
            props.get("intent", "")
            or props.get("Intent", "")
        )

        compromised_entity = (
            props.get("compromisedEntity", "")
            or props.get("CompromisedEntity", "")
            or props.get("entity", "")
        )

        product = (
            props.get("productName", "")
            or props.get("ProductName", "")
            or "Microsoft Defender for Cloud"
        )

        # Identify which Defender plan this came from
        plan = self._identify_plan(
            alert_name, description, product,
            props
        )

        # Extract source IP from entities if present
        source_ip = self._extract_source_ip(props)

        # Extract the affected resource
        resource = (
            compromised_entity
            or self._extract_resource(props)
            or "unknown_resource"
        )

        # Classify and map to MITRE
        text = f"{alert_name} {description}".lower()
        technique = self._classify_technique(
            text, plan
        )
        tactic_id, tactic_name = (
            self._map_intent(intent)
        )

        # Risk scoring
        risk, reasons = self._calculate_risk(
            severity, intent, technique,
            text, source_ip, plan
        )

        return {
            "accessor_identity": resource,
            "accessor_type": "cloud_resource",
            "data_store_name": resource,
            "data_path": alert_name[:300],
            "data_classification": (
                self._detect_classification(text)
            ),
            "bytes_accessed": 0,
            "event_time": (
                props.get("timeGeneratedUtc")
                or props.get("startTimeUtc")
                or props.get("detectedTimeUtc")
                or raw_event.get("time")
                or _now()
            ),
            "source_ip": source_ip,
            "risk_score": min(round(risk, 4), 1.0),
            "risk_reasons": reasons,
            "source_system": "defender_for_cloud",
            "raw_event": raw_event,
            "defender_plan": plan,
            "defender_severity": severity,
            "defender_intent": intent,
            "defender_alert_name": alert_name[:300],
            "defender_product": product,
            "mitre_technique": technique or "",
            "mitre_tactic_id": tactic_id,
            "mitre_tactic": tactic_name,
        }

    def _identify_plan(
        self,
        alert_name: str,
        description: str,
        product: str,
        props: dict
    ) -> str:
        """Identify which Defender plan generated this"""
        text = (
            f"{alert_name} {description} {product}"
        ).lower()

        # Check resource identifiers too
        resource_id = str(
            props.get("resourceIdentifiers", "")
        ).lower()
        combined = f"{text} {resource_id}"

        if any(k in combined for k in [
            "container", "kubernetes", "aks", "pod"
        ]):
            return "containers"
        if any(k in combined for k in [
            "storage", "blob", "/storageaccounts/"
        ]):
            return "storage"
        if any(k in combined for k in [
            "keyvault", "key vault", "/vaults/"
        ]):
            return "keyvault"
        if any(k in combined for k in [
            "sql", "database", "/databases/"
        ]):
            return "sql"
        if any(k in combined for k in [
            "app service", "appservice", "web app",
            "/sites/"
        ]):
            return "appservice"
        if any(k in combined for k in [
            "resource manager", "arm",
            "control plane", "subscription"
        ]):
            return "resource_manager"
        if any(k in combined for k in [
            "vm", "virtual machine", "server",
            "/virtualmachines/", "host"
        ]):
            return "servers"

        return "general"

    def _classify_technique(
        self,
        text: str,
        plan: str
    ) -> str:
        """Map alert text to a MITRE technique"""
        for keyword, technique in (
            DEFENDER_KEYWORD_TECHNIQUE.items()
        ):
            if keyword in text:
                return technique

        # Plan-based fallback
        plan_defaults = {
            "containers":       "T1610",
            "storage":          "T1530",
            "keyvault":         "T1552",
            "sql":              "T1190",
            "appservice":       "T1505.003",
            "resource_manager": "T1078.004",
            "servers":          "T1059",
        }
        return plan_defaults.get(plan, "")

    def _map_intent(
        self, intent: str
    ) -> tuple:
        """Map Defender intent to MITRE tactic"""
        if not intent:
            return "", ""

        # Intent may be comma-separated; take first
        first = intent.split(",")[0].strip()
        return DEFENDER_INTENT_TO_TACTIC.get(
            first, ("", "")
        )

    def _calculate_risk(
        self,
        severity: str,
        intent: str,
        technique: str,
        text: str,
        source_ip: str,
        plan: str
    ) -> tuple:
        """Calculate Defender alert risk"""
        risk = DEFENDER_SEVERITY_RISK.get(
            severity, 0.40
        )
        reasons = [
            f"defender_severity:{severity}",
            f"defender_plan:{plan}",
        ]

        if intent:
            reasons.append(
                f"defender_intent:{intent[:40]}"
            )

        # High-impact intents escalate risk
        if intent in (
            "Exfiltration", "Impact",
            "LateralMovement"
        ):
            risk = max(risk, 0.78)
            reasons.append(
                f"high_impact_intent:{intent}"
            )

        # Keyword-based escalations
        if any(k in text for k in [
            "cryptomining", "crypto mining",
            "coin miner", "miner", "mining"
        ]):
            risk = max(risk, 0.80)
            reasons.append("cryptomining_detected")

        if "ransomware" in text:
            risk = max(risk, 0.92)
            reasons.append("ransomware_detected")

        if "container escape" in text:
            risk = max(risk, 0.93)
            reasons.append("container_escape_attempt")

        if "web shell" in text or "webshell" in text:
            risk = max(risk, 0.85)
            reasons.append("web_shell_detected")

        if "sql injection" in text:
            risk = max(risk, 0.82)
            reasons.append("sql_injection_detected")

        if "exfiltration" in text or (
            "data exfiltration" in text
        ):
            risk = max(risk, 0.80)
            reasons.append("data_exfiltration")

        # Tor / known-bad source
        if source_ip.startswith("185.220"):
            risk = max(risk, 0.90)
            reasons.append("tor_exit_node_src")

        if technique:
            reasons.append(
                f"mitre_technique:{technique}"
            )

        return risk, reasons

    def _extract_source_ip(
        self, props: dict
    ) -> str:
        """Extract attacker source IP from entities"""
        entities = props.get(
            "entities",
            props.get("Entities", [])
        )

        if isinstance(entities, list):
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                etype = str(
                    entity.get("type",
                               entity.get("Type", ""))
                ).lower()
                if etype in ("ip", "ipaddress"):
                    ip = (
                        entity.get("address")
                        or entity.get("Address")
                        or entity.get("ipAddress")
                    )
                    if ip:
                        return str(ip)

        # Flat fields fallback
        return (
            props.get("sourceIp", "")
            or props.get("attackerIp", "")
            or props.get("remoteIpAddress", "")
        )

    def _extract_resource(
        self, props: dict
    ) -> str:
        """Extract the affected Azure resource"""
        entities = props.get(
            "entities",
            props.get("Entities", [])
        )

        if isinstance(entities, list):
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                etype = str(
                    entity.get("type",
                               entity.get("Type", ""))
                ).lower()
                if etype in ("host", "azureresource"):
                    name = (
                        entity.get("hostName")
                        or entity.get("HostName")
                        or entity.get("resourceId")
                        or entity.get("name")
                    )
                    if name:
                        return str(name)

        return props.get("resourceId", "")

    def _detect_classification(
        self, text: str
    ) -> str:
        """Detect data classification from alert text"""
        if any(k in text for k in [
            "pci", "payment", "card", "cardholder"
        ]):
            return "PCI"
        if any(k in text for k in [
            "health", "medical", "patient", "phi"
        ]):
            return "PHI"
        if any(k in text for k in [
            "personal", "pii", "ssn", "social security"
        ]):
            return "PII"
        return "UNKNOWN"

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
            "source_system": "defender_for_cloud",
            "raw_event": {}
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")