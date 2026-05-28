"""
Layer 4 — Reasoning
Threat Hunting Hypothesis Engine

Automatically generates threat hunting
hypotheses from threat intelligence.

WHY THIS EXISTS:
    Traditional threat hunting:
    Analyst reads threat intel (30 mins)
    Analyst forms hypothesis (30 mins)
    Analyst writes KQL (60 mins)
    Analyst runs hunt (30 mins)
    Total: 2.5 hours per hunt

    With hypothesis engine:
    Threat intel feeds in automatically
    Hypothesis generated in seconds
    KQL retrieved from repository
    Hunt ready to run immediately
    Total: 15 minutes per hunt

    95% time reduction.
    Same quality.
    IBM charges premium for this service.

HOW IT WORKS:
    1. Reads threat intelligence input
       (advisory text, IOCs, actor profile)
    2. Extracts key signals:
       Threat actor, technique, target,
       IOCs, affected platforms
    3. Maps to MITRE ATT&CK technique
    4. Retrieves matching KQL from repository
    5. Generates structured hypothesis
    6. Outputs hunt-ready package

HYPOTHESIS STRUCTURE:
    hypothesis_id:      Unique identifier
    title:              Human readable title
    threat_actor:       Who is behind this
    mitre_technique:    ATT&CK technique
    mitre_tactic:       ATT&CK tactic
    confidence:         How confident we are
    hypothesis_statement: Plain English
    kql_query:          Ready to run KQL
    affected_systems:   What to check
    iocs:               Known bad indicators
    hunt_status:        PENDING/ACTIVE/CLOSED
    priority:           HIGH/MEDIUM/LOW

USAGE:
    engine = HypothesisEngine()

    # From threat intel text
    hypothesis = engine.generate_from_text(
        intel_text="FS-ISAC advisory about
        Scattered Spider MFA fatigue..."
    )

    # From structured intel
    hypothesis = engine.generate_from_intel({
        "threat_actor": "Scattered Spider",
        "technique": "MFA Fatigue",
        "target": "Financial sector Okta"
    })

    # Get all pending hunts
    pending = engine.get_pending_hypotheses()
"""

import logging
import uuid
from datetime import datetime
from datetime import timezone


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# Keyword to technique mapping

from layer5_interface.analytics.kql_repository\
    import KQLRepository

logger = logging.getLogger(__name__)

# Keyword to technique mapping
# Maps threat intel keywords to ATT&CK techniques
KEYWORD_TECHNIQUE_MAP = {
    # MFA and authentication
    "mfa fatigue":          "T1621",
    "mfa bombing":          "T1621",
    "push notification":    "T1621",
    "authentication fatigue": "T1621",
    "brute force":          "T1110",
    "password spray":       "T1110",
    "credential stuffing":  "T1110.004",
    "impossible travel":    "T1078",
    "account takeover":     "T1078",
    "valid accounts":       "T1078",
    "stolen credentials":   "T1078",

    # Exfiltration
    "data exfiltration":    "T1530",
    "cloud storage":        "T1530",
    "s3 bucket":            "T1530",
    "bulk download":        "T1530",
    "dns tunneling":        "T1071.004",
    "dns exfiltration":     "T1071.004",
    "exfiltration":         "T1048",

    # C2
    "command and control":  "T1071",
    "c2 beacon":            "T1071",
    "beaconing":            "T1071",
    "cobalt strike":        "T1071",

    # Execution
    "powershell":           "T1059.001",
    "macro":                "T1204",
    "phishing":             "T1566",
    "spear phishing":       "T1566.001",
    "bec":                  "T1566.002",
    "business email":       "T1566.002",

    # Privilege escalation
    "privilege escalation": "T1078.004",
    "iam manipulation":     "T1078.004",
    "role assumption":      "T1078.004",
    "pass the hash":        "T1550.002",

    # Defense evasion
    "disable logging":      "T1562.008",
    "cloudtrail":           "T1562.008",
    "audit log":            "T1562.008",
    "log tampering":        "T1562.008",

    # Lateral movement
    "lateral movement":     "T1021",
    "service account":      "T1021",

    # Impact
    "ransomware":           "T1486",
    "data destruction":     "T1485",
    "deletion":             "T1485",
    "encryption":           "T1486",

    # Container
    "container escape":     "T1611",
    "kubernetes":           "T1610",
    "pod execution":        "T1609",
}

# Threat actor to technique mapping
ACTOR_TECHNIQUE_MAP = {
    "scattered spider":     ["T1621", "T1566"],
    "unc3944":              ["T1621", "T1566"],
    "lazarus":              ["T1059", "T1566"],
    "lazarus group":        ["T1059", "T1566"],
    "fin7":                 ["T1566", "T1059"],
    "carbanak":             ["T1566", "T1059"],
    "lapsus$":              ["T1078", "T1621"],
    "apt29":                ["T1078", "T1071"],
    "cozy bear":            ["T1078", "T1071"],
    "apt28":                ["T1566", "T1110"],
    "fancy bear":           ["T1566", "T1110"],
    "lockbit":              ["T1486", "T1485"],
    "blackcat":             ["T1486", "T1078"],
    "alphv":                ["T1486", "T1078"],
}

# Priority scoring
PRIORITY_SCORES = {
    "financial_sector":     0.30,
    "active_campaign":      0.25,
    "known_actor":          0.20,
    "critical_technique":   0.15,
    "has_iocs":             0.10,
}

# Critical techniques that auto-elevate priority
CRITICAL_TECHNIQUES = [
    "T1485",    # Data Destruction
    "T1486",    # Ransomware
    "T1562.008", # Disable Logging
    "T1078.004", # Cloud Account Escalation
    "T1530",    # Data from Cloud Storage
]


class HypothesisEngine:
    """
    Generates threat hunting hypotheses
    from threat intelligence.

    Connects to KQL repository to provide
    ready-to-run hunt queries.
    """

    def __init__(self):
        self.kql_repo = KQLRepository()
        self._hypotheses = {}
        logger.info(
            "Hypothesis Engine initialized"
        )

    def generate_from_text(
        self,
        intel_text: str,
        source: str = "manual",
        analyst: str = "system"
    ) -> dict:
        """
        Generate hunting hypothesis from
        free-form threat intelligence text.

        Args:
            intel_text: Advisory or report text
            source: Intel source (FS-ISAC, X-Force)
            analyst: Analyst creating the hunt

        Returns:
            Complete hypothesis dict
        """
        if not intel_text:
            return {}

        text_lower = intel_text.lower()

        # Extract technique
        technique_id = self._extract_technique(
            text_lower
        )

        # Extract threat actor
        threat_actor = self._extract_threat_actor(
            text_lower
        )

        # Extract IOCs
        iocs = self._extract_iocs(intel_text)

        # Extract affected systems
        affected_systems = (
            self._extract_affected_systems(
                text_lower
            )
        )

        # Build hypothesis
        return self._build_hypothesis(
            technique_id=technique_id,
            threat_actor=threat_actor,
            iocs=iocs,
            affected_systems=affected_systems,
            intel_text=intel_text,
            source=source,
            analyst=analyst
        )

    def generate_from_intel(
        self,
        intel: dict,
        analyst: str = "system"
    ) -> dict:
        """
        Generate hypothesis from structured
        intelligence dict.

        Args:
            intel: Dict with keys:
                threat_actor: Actor name
                technique: Technique name or ID
                target: Target description
                iocs: List of IOC strings
                source: Intel source
            analyst: Analyst name

        Returns:
            Complete hypothesis dict
        """
        if not intel:
            return {}

        technique_input = intel.get(
            "technique", ""
        ).lower()

        # Check if it is already an ID
        if technique_input.startswith("t1"):
            technique_id = technique_input.upper()
        else:
            technique_id = KEYWORD_TECHNIQUE_MAP.get(
                technique_input,
                self._extract_technique(
                    technique_input
                )
            )

        threat_actor = intel.get(
            "threat_actor", "Unknown"
        )
        iocs = intel.get("iocs", [])
        target = intel.get("target", "")
        source = intel.get("source", "manual")

        affected_systems = (
            self._extract_affected_systems(
                target.lower()
            )
        )

        return self._build_hypothesis(
            technique_id=technique_id,
            threat_actor=threat_actor,
            iocs=iocs,
            affected_systems=affected_systems,
            intel_text=target,
            source=source,
            analyst=analyst
        )

    def generate_batch(
        self,
        intel_items: list
    ) -> list:
        """
        Generate multiple hypotheses from
        a list of intelligence items.

        Args:
            intel_items: List of intel dicts

        Returns:
            List of hypothesis dicts
        """
        hypotheses = []
        for item in intel_items:
            if isinstance(item, str):
                h = self.generate_from_text(item)
            elif isinstance(item, dict):
                h = self.generate_from_intel(item)
            else:
                continue

            if h:
                hypotheses.append(h)

        logger.info(
            f"Generated {len(hypotheses)} "
            f"hypotheses from {len(intel_items)} "
            f"intel items"
        )
        return hypotheses

    def get_hypothesis(
        self, hypothesis_id: str
    ) -> dict:
        """Get hypothesis by ID"""
        return self._hypotheses.get(
            hypothesis_id, {}
        )

    def get_pending_hypotheses(self) -> list:
        """Get all hypotheses pending execution"""
        return [
            h for h in self._hypotheses.values()
            if h.get("hunt_status") == "PENDING"
        ]

    def get_hypotheses_by_technique(
        self, technique_id: str
    ) -> list:
        """Get hypotheses for a technique"""
        return [
            h for h in self._hypotheses.values()
            if h.get("mitre_technique") == (
                technique_id
            )
        ]

    def update_hunt_status(
        self,
        hypothesis_id: str,
        status: str,
        findings: str = ""
    ) -> dict:
        """
        Update hunt status after execution.

        Args:
            hypothesis_id: Hypothesis to update
            status: PENDING/ACTIVE/CONFIRMED/
                    FALSE_POSITIVE/CLOSED
            findings: What the hunt found

        Returns:
            Updated hypothesis
        """
        if hypothesis_id in self._hypotheses:
            self._hypotheses[hypothesis_id][
                "hunt_status"
            ] = status
            self._hypotheses[hypothesis_id][
                "findings"
            ] = findings
            self._hypotheses[hypothesis_id][
                "updated_at"
            ] = datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

            if status == "CONFIRMED":
                self._hypotheses[hypothesis_id][
                    "convert_to_rule"
                ] = True

        return self._hypotheses.get(
            hypothesis_id, {}
        )

    def get_statistics(self) -> dict:
        """Get engine statistics"""
        hypotheses = list(
            self._hypotheses.values()
        )

        status_counts = {}
        for h in hypotheses:
            status = h.get("hunt_status", "")
            status_counts[status] = (
                status_counts.get(status, 0) + 1
            )

        return {
            "total_hypotheses": len(hypotheses),
            "by_status": status_counts,
            "kql_queries_available": (
                len(
                    self.kql_repo.get_all_queries()
                )
            ),
            "techniques_covered": len(
                set(
                    h.get("mitre_technique", "")
                    for h in hypotheses
                )
            )
        }

    # ============================================================
    # PRIVATE METHODS
    # ============================================================

    def _build_hypothesis(
        self,
        technique_id: str,
        threat_actor: str,
        iocs: list,
        affected_systems: list,
        intel_text: str,
        source: str,
        analyst: str
    ) -> dict:
        """Build complete hypothesis"""

        hypothesis_id = f"HYP-{str(uuid.uuid4())[:8].upper()}"

        # Get KQL queries for technique
        kql_queries = []
        if technique_id:
            kql_queries = (
                self.kql_repo.get_by_technique(
                    technique_id
                )
            )

        primary_kql = (
            kql_queries[0].get("kql", "")
            if kql_queries else ""
        )
        query_title = (
            kql_queries[0].get("title", "")
            if kql_queries else ""
        )

        # Get tactic from technique
        tactic = self._get_tactic(technique_id)

        # Calculate priority
        priority = self._calculate_priority(
            technique_id,
            threat_actor,
            iocs,
            intel_text
        )

        # Generate hypothesis statement
        statement = self._generate_statement(
            technique_id,
            threat_actor,
            affected_systems,
            source
        )

        hypothesis = {
            "hypothesis_id": hypothesis_id,
            "title": self._generate_title(
                technique_id, threat_actor
            ),
            "threat_actor": threat_actor,
            "mitre_technique": technique_id,
            "mitre_tactic": tactic,
            "confidence": self._calculate_confidence(
                technique_id, threat_actor, iocs
            ),
            "priority": priority,
            "hypothesis_statement": statement,
            "kql_query": primary_kql,
            "kql_query_title": query_title,
            "additional_queries": [
                q.get("query_id", "")
                for q in kql_queries[1:3]
            ],
            "affected_systems": affected_systems,
            "iocs": iocs,
            "intel_source": source,
            "intel_summary": intel_text[:300],
            "hunt_status": "PENDING",
            "created_by": analyst,
            "created_at": _now(),
            "updated_at": _now(),
            "findings": "",
            "convert_to_rule": False
        }

        self._hypotheses[hypothesis_id] = (
            hypothesis
        )

        logger.info(
            f"Hypothesis generated: "
            f"{hypothesis_id} "
            f"technique={technique_id} "
            f"actor={threat_actor} "
            f"priority={priority}"
        )

        return hypothesis

    def _extract_technique(
        self, text: str
    ) -> str:
        """Extract MITRE technique from text"""
        for keyword, technique in (
            KEYWORD_TECHNIQUE_MAP.items()
        ):
            if keyword in text:
                return technique
        return "T1078"

    def _extract_threat_actor(
        self, text: str
    ) -> str:
        """Extract threat actor from text"""
        for actor in ACTOR_TECHNIQUE_MAP.keys():
            if actor in text:
                return actor.title()
        return "Unknown"

    def _extract_iocs(self, text: str) -> list:
        """Extract IOCs from text"""
        import re
        iocs = []

        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        ips = re.findall(ip_pattern, text)
        iocs.extend(ips[:5])

        domain_pattern = (
            r'\b[a-zA-Z0-9][a-zA-Z0-9-]{1,61}'
            r'[a-zA-Z0-9]\.[a-zA-Z]{2,}\b'
        )
        domains = re.findall(domain_pattern, text)
        for d in domains[:3]:
            if d not in iocs:
                iocs.append(d)

        hash_pattern = r'\b[a-fA-F0-9]{32,64}\b'
        hashes = re.findall(hash_pattern, text)
        iocs.extend(hashes[:3])

        return iocs[:10]

    def _extract_affected_systems(
        self, text: str
    ) -> list:
        """Extract affected systems from text"""
        systems = []
        system_keywords = {
            "okta":         "Okta Identity Engine",
            "entra":        "Microsoft Entra ID",
            "azure ad":     "Microsoft Entra ID",
            "azure":        "Microsoft Azure",
            "aws":          "Amazon Web Services",
            "s3":           "AWS S3",
            "sentinel":     "Microsoft Sentinel",
            "defender":     "Microsoft Defender XDR",
            "cyberark":     "CyberArk PAM",
            "sailpoint":    "SailPoint IGA",
            "kubernetes":   "Kubernetes",
            "windows":      "Windows",
            "linux":        "Linux",
            "exchange":     "Microsoft Exchange",
            "office 365":   "Microsoft 365",
            "sharepoint":   "SharePoint Online",
            "cloudtrail":   "AWS CloudTrail",
            "qradar":       "IBM QRadar",
            "splunk":       "Splunk SIEM",
        }

        for keyword, system in (
            system_keywords.items()
        ):
            if keyword in text:
                if system not in systems:
                    systems.append(system)

        if not systems:
            systems = ["Microsoft Sentinel"]

        return systems[:5]

    def _get_tactic(
        self, technique_id: str
    ) -> str:
        """Get tactic name from technique ID"""
        tactic_map = {
            "T1566": "Initial Access",
            "T1078": "Initial Access",
            "T1190": "Initial Access",
            "T1059": "Execution",
            "T1204": "Execution",
            "T1136": "Persistence",
            "T1548": "Privilege Escalation",
            "T1078.004": "Privilege Escalation",
            "T1562": "Defense Evasion",
            "T1562.008": "Defense Evasion",
            "T1110": "Credential Access",
            "T1110.004": "Credential Access",
            "T1621": "Credential Access",
            "T1539": "Credential Access",
            "T1021": "Lateral Movement",
            "T1550": "Lateral Movement",
            "T1114": "Collection",
            "T1530": "Collection",
            "T1213": "Collection",
            "T1071": "Command and Control",
            "T1071.004": "Command and Control",
            "T1568": "Command and Control",
            "T1048": "Exfiltration",
            "T1029": "Exfiltration",
            "T1485": "Impact",
            "T1486": "Impact",
            "T1496": "Impact",
            "T1610": "Defense Evasion",
            "T1609": "Execution",
            "T1611": "Privilege Escalation",
        }
        return tactic_map.get(
            technique_id, "Unknown"
        )

    def _calculate_priority(
        self,
        technique_id: str,
        threat_actor: str,
        iocs: list,
        intel_text: str
    ) -> str:
        """Calculate hunt priority"""
        score = 0.0

        text_lower = intel_text.lower()
        if any(
            kw in text_lower
            for kw in [
                "financial", "banking", "payment",
                "bank", "insurance"
            ]
        ):
            score += PRIORITY_SCORES[
                "financial_sector"
            ]

        if any(
            kw in text_lower
            for kw in [
                "active", "ongoing", "current",
                "today", "this week"
            ]
        ):
            score += PRIORITY_SCORES[
                "active_campaign"
            ]

        if threat_actor != "Unknown":
            score += PRIORITY_SCORES["known_actor"]

        if technique_id in CRITICAL_TECHNIQUES:
            score += PRIORITY_SCORES[
                "critical_technique"
            ]

        if iocs:
            score += PRIORITY_SCORES["has_iocs"]

        if score >= 0.60:
            return "CRITICAL"
        elif score >= 0.40:
            return "HIGH"
        elif score >= 0.20:
            return "MEDIUM"
        return "LOW"

    def _calculate_confidence(
        self,
        technique_id: str,
        threat_actor: str,
        iocs: list
    ) -> float:
        """Calculate hypothesis confidence"""
        confidence = 0.40

        if technique_id:
            confidence += 0.20

        if threat_actor != "Unknown":
            confidence += 0.20

        if iocs:
            confidence += min(
                len(iocs) * 0.05, 0.20
            )

        return round(min(confidence, 1.0), 2)

    def _generate_title(
        self,
        technique_id: str,
        threat_actor: str
    ) -> str:
        """Generate hypothesis title"""
        tech_names = {
            "T1621": "MFA Fatigue Attack",
            "T1110": "Brute Force Attack",
            "T1110.004": "Credential Stuffing",
            "T1078": "Valid Account Abuse",
            "T1078.004": "Cloud Account Escalation",
            "T1530": "Cloud Storage Exfiltration",
            "T1566": "Phishing Campaign",
            "T1566.002": "BEC Attack",
            "T1071": "C2 Communication",
            "T1071.004": "DNS Tunneling",
            "T1562.008": "Audit Log Tampering",
            "T1485": "Data Destruction",
            "T1486": "Ransomware Activity",
            "T1059": "Script Execution",
            "T1021": "Lateral Movement",
        }

        tech_name = tech_names.get(
            technique_id, f"Attack ({technique_id})"
        )

        if threat_actor != "Unknown":
            return (
                f"{threat_actor} — {tech_name}"
            )
        return tech_name

    def _generate_statement(
        self,
        technique_id: str,
        threat_actor: str,
        affected_systems: list,
        source: str
    ) -> str:
        """Generate plain English hypothesis"""
        tech_descriptions = {
            "T1621": (
                "conducting MFA fatigue attacks "
                "generating rapid push notifications"
            ),
            "T1110": (
                "conducting brute force attacks "
                "against user credentials"
            ),
            "T1078": (
                "using stolen valid account "
                "credentials for unauthorized access"
            ),
            "T1530": (
                "exfiltrating data from cloud "
                "storage resources"
            ),
            "T1566": (
                "delivering phishing emails to "
                "target users"
            ),
            "T1071": (
                "maintaining C2 communication "
                "via application layer protocols"
            ),
            "T1071.004": (
                "using DNS tunneling for "
                "C2 communication or exfiltration"
            ),
            "T1562.008": (
                "disabling audit logging to "
                "blind defenders before attack"
            ),
        }

        action = tech_descriptions.get(
            technique_id,
            f"exploiting technique {technique_id}"
        )

        actor = (
            threat_actor
            if threat_actor != "Unknown"
            else "An unknown threat actor"
        )

        systems = (
            ", ".join(affected_systems[:2])
            if affected_systems
            else "our environment"
        )

        return (
            f"I believe {actor} may be {action} "
            f"against {systems} based on "
            f"intelligence from {source}."
        )
    