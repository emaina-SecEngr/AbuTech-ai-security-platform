"""
Layer 3 — Knowledge Graph
MITRE ATT&CK Enricher

Maps security events to MITRE ATT&CK
tactics, techniques, and mitigations.

WHY THIS IS CRITICAL:
    Without MITRE mapping:
    "Risk score: 0.974 CRITICAL"
    SOC analyst thinks: "OK but what IS this?"

    With MITRE mapping:
    "Risk score: 0.974 CRITICAL
     Tactic: Exfiltration (TA0010)
     Technique: T1530 Data from Cloud Storage
     Mitigation: Restrict S3 bucket permissions"

    SOC analyst thinks:
    "I know exactly what this is.
     I know exactly how to contain it.
     I know exactly how to prevent recurrence."

WHAT THIS ENRICHER DOES:
    1. Loads local MITRE ATT&CK database
    2. Maps event signals to techniques
    3. Returns full ATT&CK context per event
    4. Provides mitigation recommendations
    5. Calculates ATT&CK coverage metrics

HOW TECHNIQUE MAPPING WORKS:
    Each ML model detects specific patterns.
    Each pattern maps to ATT&CK techniques.

    ISOLATION FOREST (anomalous access):
        Large volume transfer → T1530
        After hours access → T1078
        Unusual protocol → T1048

    LSTM (time-based patterns):
        Regular beaconing → T1071
        Slow exfiltration → T1048
        Periodic access → T1029

    DNS CLASSIFIER (malicious DNS):
        DGA domain → T1568
        DNS tunneling → T1071.004
        C2 over DNS → T1071.004

    IDENTITY DETECTOR (account abuse):
        MFA fatigue → T1621
        Brute force → T1110
        Impossible travel → T1078
        Credential stuffing → T1110.003

    GNN (relationship anomalies):
        Fraud ring → T1078
        Lateral movement → T1021
        Account compromise → T1550

    WAF EVENTS (web attacks):
        SQL injection → T1190
        XSS → T1059.007
        Path traversal → T1083

    IaC FINDINGS (misconfigurations):
        Public S3 bucket → T1530
        Open SSH port → T1190
        IAM wildcard → T1078

USAGE:
    enricher = MITREEnricher()

    # Map event to technique
    result = enricher.map_event(data_event)

    # Get specific technique
    technique = enricher.get_technique("T1530")

    # Get mitigations for technique
    mitigations = enricher.get_mitigations("T1530")

    # Enrich a DataAccessEvent
    enriched = enricher.enrich_event(event)

    # Get coverage report
    coverage = enricher.get_coverage_report(events)
"""

import json
import logging
import os
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)

# Path to local MITRE ATT&CK database
MITRE_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )),
    "data", "mitre_attack_db.json"
)

# Signal to technique mapping
# Maps risk_reasons and source_system
# to MITRE ATT&CK technique IDs
SIGNAL_TO_TECHNIQUE = {

    # DATA EXFILTRATION SIGNALS
    "large_transfer_500mb+":    ["T1530", "T1048"],
    "large_transfer_100mb+":    ["T1530", "T1048"],
    "large_api_response":       ["T1530"],
    "bulk_download_detected":   ["T1530"],
    "data_exfil":               ["T1530", "T1048"],

    # IDENTITY AND AUTH SIGNALS
    "after_hours":              ["T1078"],
    "after_hours_traffic":      ["T1078"],
    "impossible_travel":        ["T1078"],
    "new_device":               ["T1078"],
    "authentication_failure":   ["T1110"],
    "authorization_failure":    ["T1078"],
    "mfa_fatigue":              ["T1621"],
    "user_clicked_malicious_url": ["T1566"],
    "credential_stuffing":      ["T1110.003"],
    "brute_force":              ["T1110"],
    "tor_exit_node":            ["T1090"],
    "tor_exit_node_src":        ["T1090"],
    "tor_ip":                   ["T1090"],
    "rate_limit_exceeded":      ["T1498"],

    # NETWORK SIGNALS
    "c2_communication":         ["T1071"],
    "dns_tunneling":            ["T1071.004"],
    "dga_domain":               ["T1568"],
    "beaconing":                ["T1071"],
    "high_risk_port:22_SSH":    ["T1021.004"],
    "high_risk_port:3389_RDP":  ["T1021.001"],
    "high_risk_port:445_SMB":   ["T1021.002"],
    "high_risk_country":        ["T1078"],

    # WEB ATTACK SIGNALS
    "sql_injection_pattern_detected": ["T1190"],
    "sql_injection_pattern":    ["T1190"],
    "sqli_payload_in_data":     ["T1190"],
    "xss_pattern_detected":     ["T1059.007"],
    "path_traversal_detected":  ["T1083"],
    "remote_code_execution":    ["T1059"],
    "waf_action:BLOCK":         ["T1190"],

    # IaC SIGNALS
    "public_exposure_risk":     ["T1530"],
    "encryption_missing":       ["T1485"],
    "remote_access_open_to_world": ["T1190"],
    "database_resource":        ["T1530"],
    "secrets_resource":         ["T1552"],
    "identity_resource":        ["T1078"],
    "audit_logging_disabled":   ["T1562.008"],

    # CLOUD SIGNALS
    "kube_system_access":       ["T1610"],
    "pod_exec_command_execution": ["T1609"],
    "container_escape_attempt": ["T1611"],
    "cryptomining_detected":    ["T1496"],
    "privilege_escalation":     ["T1078.004"],
    "shadow_admin":             ["T1136"],
    "service_account_privileged_op": ["T1078.004"],

    # THREAT INTELLIGENCE SIGNALS
    "threat_detected":          ["T1190"],
    "threat_log_type":          ["T1190"],
    "virus_detected":           ["T1204"],
    "malware_detected":         ["T1204"],
    "phishing_detected":        ["T1566"],
    "bec_detected":             ["T1566.002"],
    "executive_impersonation":  ["T1566.002"],
    "malicious_url_blocked":    ["T1566.002"],
    "lookalike_domain_detected": ["T1566.002"],

    # IAM SIGNALS
    "toxic_combinations":       ["T1078"],
    "peer_group_anomaly":       ["T1078"],
    "unused_permissions":       ["T1078.004"],
    "permission_creep":         ["T1078.004"],
    "escalation_paths":         ["T1548"],
}

# Source system to technique mapping
# When specific signals not available
SOURCE_TO_TECHNIQUE = {
    "waf_aws":              ["T1190"],
    "waf_azure":            ["T1190"],
    "waf_cloudflare":       ["T1190"],
    "waf_modsecurity":      ["T1190"],
    "firewall_palo_alto":   ["T1190"],
    "firewall_fortinet":    ["T1190"],
    "firewall_cisco_asa":   ["T1190"],
    "email_proofpoint":     ["T1566"],
    "email_mimecast":       ["T1566"],
    "email_defender_o365":  ["T1566"],
    "api_gateway_aws":      ["T1190"],
    "api_gateway_azure":    ["T1190"],
    "api_gateway_kong":     ["T1190"],
    "iac_checkov":          ["T1190"],
    "iac_tfsec":            ["T1190"],
    "iac_terrascan":        ["T1190"],
    "kubernetes_audit":     ["T1610"],
    "falco_runtime":        ["T1610"],
    "container_vulnerability": ["T1525"],
    "iac_security":         ["T1190"],
    "s3_normalizer":        ["T1530"],
    "rds_normalizer":       ["T1530"],
    "snowflake_normalizer":  ["T1530"],
    "okta_normalizer":      ["T1078"],
    "entraid_normalizer":   ["T1078"],
    "cyberark_normalizer":  ["T1078"],
    "sailpoint_normalizer":  ["T1078"],
    "sentinel_normalizer":  ["T1078"],
    "crowdstrike_normalizer": ["T1059"],
    "network_flow":         ["T1071"],
    "dns_classifier":       ["T1071.004"],
    "ciem_enricher":        ["T1078.004"],
}


class MITREEnricher:
    """
    Maps security events to MITRE ATT&CK
    tactics, techniques, and mitigations.

    Provides full ATT&CK context for every
    security event processed by AbuTech.
    """

    def __init__(self):
        self._db = None
        self._load_database()

    def _load_database(self):
        """Load local MITRE ATT&CK database"""
        try:
            if os.path.exists(MITRE_DB_PATH):
                with open(MITRE_DB_PATH, "r") as f:
                    self._db = json.load(f)
                logger.info(
                    f"MITRE ATT&CK database loaded: "
                    f"{len(self._db.get('techniques', {}))} "
                    f"techniques"
                )
            else:
                logger.warning(
                    "MITRE database not found. "
                    "Run: python scripts/"
                    "download_mitre_attack.py"
                )
                self._db = {
                    "tactics": {},
                    "techniques": {},
                    "mitigations": {}
                }
        except Exception as e:
            logger.error(
                f"Failed to load MITRE database: {e}"
            )
            self._db = {
                "tactics": {},
                "techniques": {},
                "mitigations": {}
            }

    def enrich_event(
        self, data_event: dict
    ) -> dict:
        """
        Enrich a DataAccessEvent with full
        MITRE ATT&CK context.

        This is the main method called by
        Layer 2 ML pipeline after scoring.

        Args:
            data_event: DataAccessEvent dict

        Returns:
            Event enriched with mitre_context
        """
        if not data_event:
            return data_event

        mitre_context = self.map_event(data_event)

        return {
            **data_event,
            "mitre_context": mitre_context,
            "mitre_techniques": [
                t["id"]
                for t in mitre_context.get(
                    "techniques", []
                )
            ],
            "mitre_tactic": mitre_context.get(
                "primary_tactic", {}).get("name", ""),
            "mitre_tactic_id": mitre_context.get(
                "primary_tactic", {}).get("id", "")
        }

    def map_event(
        self, data_event: dict
    ) -> dict:
        """
        Map security event to MITRE ATT&CK.

        Analyzes risk_reasons and source_system
        to identify relevant techniques.

        Args:
            data_event: DataAccessEvent dict

        Returns:
            dict with full ATT&CK context
        """
        risk_reasons = data_event.get(
            "risk_reasons", []
        )
        source_system = data_event.get(
            "source_system", ""
        )
        risk_score = float(
            data_event.get("risk_score", 0.0) or 0.0
        )

        # Collect technique IDs from signals
        technique_ids = set()

        # Map from risk reasons
        for reason in risk_reasons:
            reason_lower = reason.lower()
            for signal, techs in (
                SIGNAL_TO_TECHNIQUE.items()
            ):
                if signal.lower() in reason_lower:
                    technique_ids.update(techs)

        # Map from source system
        for source, techs in (
            SOURCE_TO_TECHNIQUE.items()
        ):
            if source in source_system.lower():
                technique_ids.update(techs)

        # Get full technique details
        techniques = []
        for tech_id in technique_ids:
            tech = self.get_technique(tech_id)
            if tech:
                techniques.append(tech)

        # Sort by relevance
        techniques.sort(
            key=lambda x: (
                len(x.get("mitigations", [])),
                x.get("id", "")
            ),
            reverse=True
        )

        # Get primary tactic
        primary_tactic = {}
        if techniques:
            tactic_ref = (
                techniques[0].get(
                    "tactic_refs", []
                ) or []
            )
            if tactic_ref:
                primary_tactic = (
                    self._get_tactic_by_shortname(
                        tactic_ref[0]
                    )
                )

        # Get all mitigations
        all_mitigations = []
        seen_mit_ids = set()
        for tech in techniques:
            for mit in tech.get("mitigations", []):
                if mit["id"] not in seen_mit_ids:
                    all_mitigations.append(mit)
                    seen_mit_ids.add(mit["id"])

        # Build procedures from risk reasons
        procedures = self._build_procedures(
            risk_reasons, data_event
        )

        return {
            "techniques": techniques[:5],
            "technique_ids": [
                t["id"] for t in techniques[:5]
            ],
            "primary_tactic": primary_tactic,
            "all_mitigations": all_mitigations[:5],
            "procedures": procedures,
            "confidence": self._calculate_confidence(
                technique_ids, risk_score
            ),
            "coverage": len(technique_ids) > 0,
            "mapped_at": _now()
        }

    def get_technique(
        self, technique_id: str
    ) -> dict:
        """
        Get full technique details by ID.

        Args:
            technique_id: ATT&CK technique ID
                          e.g. "T1530"

        Returns:
            Technique dict or empty dict
        """
        techniques = self._db.get(
            "techniques", {}
        )
        return techniques.get(
            technique_id, {}
        )

    def get_mitigations(
        self, technique_id: str
    ) -> list:
        """
        Get mitigations for a technique.

        Args:
            technique_id: ATT&CK technique ID

        Returns:
            List of mitigation dicts
        """
        technique = self.get_technique(
            technique_id
        )
        return technique.get("mitigations", [])

    def get_tactic(
        self, tactic_id: str
    ) -> dict:
        """
        Get tactic details by ID.

        Args:
            tactic_id: ATT&CK tactic ID
                       e.g. "TA0010"

        Returns:
            Tactic dict or empty dict
        """
        tactics = self._db.get("tactics", {})
        return tactics.get(tactic_id, {})

    def get_coverage_report(
        self, events: list
    ) -> dict:
        """
        Generate ATT&CK coverage report
        for a set of events.

        Shows which tactics and techniques
        were detected in this event set.

        Useful for:
            Monthly SOC reports
            Executive briefings
            IBM client presentations

        Args:
            events: List of enriched events

        Returns:
            Coverage report dict
        """
        if not events:
            return {
                "total_events": 0,
                "techniques_detected": [],
                "tactics_covered": [],
                "coverage_percentage": 0.0
            }

        techniques_detected = {}
        tactics_covered = set()

        for event in events:
            mitre = event.get(
                "mitre_context", {}
            )
            for tech in mitre.get(
                "techniques", []
            ):
                tech_id = tech.get("id", "")
                if tech_id:
                    if tech_id not in (
                        techniques_detected
                    ):
                        techniques_detected[
                            tech_id
                        ] = {
                            "technique": tech,
                            "event_count": 0
                        }
                    techniques_detected[
                        tech_id
                    ]["event_count"] += 1

                for tactic_ref in tech.get(
                    "tactic_refs", []
                ):
                    tactics_covered.add(tactic_ref)

        total_techniques = len(
            self._db.get("techniques", {})
        )
        detected_count = len(techniques_detected)
        coverage_pct = (
            detected_count / total_techniques * 100
            if total_techniques > 0
            else 0.0
        )

        top_techniques = sorted(
            techniques_detected.values(),
            key=lambda x: x["event_count"],
            reverse=True
        )[:10]

        return {
            "total_events": len(events),
            "techniques_detected": detected_count,
            "total_techniques_in_db": total_techniques,
            "coverage_percentage": round(
                coverage_pct, 2
            ),
            "tactics_covered": list(tactics_covered),
            "tactics_covered_count": len(
                tactics_covered
            ),
            "top_techniques": top_techniques,
            "report_generated_at": _now()
        }

    def search_techniques(
        self,
        keyword: str
    ) -> list:
        """
        Search techniques by keyword.

        Useful for threat hunting:
        "What techniques involve DNS?"
        "What techniques involve S3?"

        Args:
            keyword: Search term

        Returns:
            List of matching technique dicts
        """
        keyword_lower = keyword.lower()
        results = []

        for tech_id, tech in (
            self._db.get("techniques", {}).items()
        ):
            if (
                keyword_lower in
                tech.get("name", "").lower() or
                keyword_lower in
                tech.get("description", "").lower()
            ):
                results.append(tech)

        return results[:20]

    def get_financial_sector_techniques(
        self
    ) -> list:
        """
        Return techniques most relevant
        to financial services.

        Used for:
            IBM client onboarding reports
            Financial sector threat briefings
            SOC detection priority setting

        Returns:
            List of high priority techniques
            for financial sector
        """
        financial_technique_ids = [
            "T1530",    # Data from Cloud Storage
            "T1078",    # Valid Accounts
            "T1110",    # Brute Force
            "T1621",    # MFA Request Generation
            "T1566",    # Phishing
            "T1566.002", # Spearphishing Link (BEC)
            "T1190",    # Exploit Public-Facing App
            "T1048",    # Exfiltration Alt Protocol
            "T1071",    # Application Layer Protocol
            "T1071.004", # DNS C2
            "T1562",    # Impair Defenses
            "T1562.008", # Disable Cloud Logs
            "T1485",    # Data Destruction
            "T1136",    # Create Account
            "T1552",    # Unsecured Credentials
            "T1548",    # Abuse Elevation Control
            "T1021",    # Remote Services
            "T1610",    # Deploy Container
            "T1611",    # Escape to Host
            "T1496",    # Resource Hijacking
        ]

        techniques = []
        for tech_id in financial_technique_ids:
            tech = self.get_technique(tech_id)
            if tech:
                techniques.append(tech)

        return techniques

    def _build_procedures(
        self,
        risk_reasons: list,
        event: dict
    ) -> list:
        """Build human readable procedures"""
        procedures = []

        accessor = event.get(
            "accessor_identity", "Unknown identity"
        )
        data_store = event.get(
            "data_store_name", "unknown resource"
        )
        source_ip = event.get("source_ip", "")
        classification = event.get(
            "data_classification", ""
        )

        for reason in risk_reasons:
            reason_lower = reason.lower()

            if "large_transfer" in reason_lower:
                procedures.append(
                    f"{accessor} transferred large "
                    f"volume of data from "
                    f"{data_store}"
                )
            elif "after_hours" in reason_lower:
                procedures.append(
                    f"{accessor} accessed "
                    f"{data_store} outside "
                    f"business hours"
                )
            elif "tor" in reason_lower:
                procedures.append(
                    f"Access originated from "
                    f"Tor exit node: {source_ip}"
                )
            elif "mfa" in reason_lower:
                procedures.append(
                    f"{accessor} subjected to "
                    f"MFA fatigue attack"
                )
            elif "sql_injection" in reason_lower:
                procedures.append(
                    f"SQL injection attempted "
                    f"against {data_store}"
                )
            elif "brute_force" in reason_lower:
                procedures.append(
                    f"Brute force attack against "
                    f"{accessor} credentials"
                )
            elif "public_exposure" in reason_lower:
                procedures.append(
                    f"{data_store} exposed to "
                    f"public internet"
                )
            elif "encryption" in reason_lower:
                procedures.append(
                    f"{data_store} missing "
                    f"encryption at rest"
                )

        if classification in ["PCI", "PHI", "PII"]:
            procedures.append(
                f"{classification} sensitive data "
                f"involved in this event"
            )

        return procedures[:5]

    def _get_tactic_by_shortname(
        self, shortname: str
    ) -> dict:
        """Get tactic by shortname"""
        for tactic in self._db.get(
            "tactics", {}
        ).values():
            if tactic.get(
                "shortname", ""
            ) == shortname:
                return tactic
        return {}

    def _calculate_confidence(
        self,
        technique_ids: set,
        risk_score: float
    ) -> float:
        """Calculate mapping confidence"""
        if not technique_ids:
            return 0.0

        base = min(
            len(technique_ids) * 0.15, 0.60
        )
        score_boost = risk_score * 0.40

        return round(
            min(base + score_boost, 1.0), 2
        )

    def get_database_stats(self) -> dict:
        """Get database statistics"""
        return {
            "version": self._db.get(
                "version", "unknown"
            ),
            "tactics_count": len(
                self._db.get("tactics", {})
            ),
            "techniques_count": len(
                self._db.get("techniques", {})
            ),
            "mitigations_count": len(
                self._db.get("mitigations", {})
            ),
            "db_path": MITRE_DB_PATH,
            "db_exists": os.path.exists(
                MITRE_DB_PATH
            )
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")