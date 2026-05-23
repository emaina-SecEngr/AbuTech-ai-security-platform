"""
Layer 1 — Data Ingestion
Microsoft Sentinel Normalizer

Converts Microsoft Sentinel / ASIM format events
into DataAccessEvent for ML processing.

WHY THIS IS THE MOST POWERFUL NORMALIZER:
    Every other normalizer handles ONE source.
    S3Normalizer: AWS CloudTrail only.
    OktaNormalizer: Okta only.
    
    This normalizer handles EVERYTHING
    that Sentinel already collected:
    
    Firewall logs (Palo Alto, Fortinet, Cisco)
    Endpoint logs (Defender, CrowdStrike, Carbon Black)
    Identity logs (Entra ID, Okta, CyberArk)
    Cloud logs (AWS CloudTrail, Azure Activity)
    Network logs (Zeek, Suricata, NetFlow)
    Email logs (Defender O365, Proofpoint)
    Vulnerability data (Qualys, Tenable)
    Threat intelligence (MSFT TI, MITRE ATT&CK)
    
    ONE normalizer → ALL data sources.
    This is the power of Sentinel integration.

MICROSOFT ASIM:
    Advanced Security Information Model.
    Microsoft's normalized schema for Sentinel.
    All data sources mapped to consistent fields.
    Very similar to our DataAccessEvent design.

BIDIRECTIONAL INTEGRATION:
    INBOUND:  Sentinel → Your Platform
              Read ASIM events from Sentinel
              Run through all 8 ML models
              Run through 5 LLM agents
              
    OUTBOUND: Your Platform → Sentinel
              Write enriched results back
              Analyst sees AbuTech scores
              in their existing Sentinel UI
              No new tool to learn

USAGE:
    normalizer = SentinelNormalizer()
    
    # From Sentinel alert webhook
    data_event = normalizer.normalize(sentinel_alert)
    
    # From Sentinel query results
    events = normalizer.normalize_batch(query_results)
    
    # Write enrichment back to Sentinel
    normalizer.write_enrichment(
        incident_id="abc123",
        risk_score=0.974,
        verdict="DATA_EXFILTRATION",
        agent_summary="svc_backup compromised..."
    )
"""

import logging
import os
from datetime import datetime
from datetime import timezone
from typing import List
from typing import Optional

logger = logging.getLogger(__name__)

# Sentinel event table names mapped to our source systems
SENTINEL_TABLE_MAP = {
    # Network
    "CommonSecurityLog":        "firewall",
    "AzureNetworkAnalytics_CL": "network_flow",
    "DnsEvents":                "dns",
    "VMConnection":             "network_flow",
    "W3CIISLog":                "web_app",

    # Endpoint
    "SecurityEvent":            "windows_endpoint",
    "DeviceProcessEvents":      "endpoint",
    "DeviceNetworkEvents":      "endpoint",
    "DeviceFileEvents":         "endpoint",
    "DeviceLogonEvents":        "endpoint",
    "DeviceRegistryEvents":     "endpoint",

    # Identity
    "SigninLogs":               "entraid",
    "AuditLogs":                "entraid",
    "AADNonInteractiveUserSignInLogs": "entraid",
    "IdentityLogonEvents":      "identity",
    "BehaviorAnalytics":        "ueba",

    # Cloud
    "AWSCloudTrail":            "aws_cloudtrail",
    "AzureActivity":            "azure_activity",
    "StorageBlobLogs":          "azure_storage",

    # Email
    "EmailEvents":              "email",
    "EmailAttachmentInfo":      "email",
    "EmailUrlInfo":             "email",

    # Threat Intelligence
    "ThreatIntelligenceIndicator": "threat_intel",

    # Alerts
    "SecurityAlert":            "sentinel_alert",
    "SecurityIncident":         "sentinel_incident",
}

# ASIM field mappings to DataAccessEvent fields
# Format: asim_field → data_access_event_field
ASIM_FIELD_MAP = {
    # WHO
    "AccountName":              "accessor_identity",
    "SubjectUserName":          "accessor_identity",
    "TargetUserName":           "accessor_identity",
    "UserPrincipalName":        "accessor_identity",
    "InitiatedBy":              "accessor_identity",
    "SrcUserName":              "accessor_identity",
    "ActorUsername":            "accessor_identity",
    "UserId":                   "accessor_identity",

    # WHERE FROM
    "SourceIP":                 "source_ip",
    "SrcIpAddr":                "source_ip",
    "ClientIP":                 "source_ip",
    "IPAddress":                "source_ip",
    "CallerIpAddress":          "source_ip",
    "RemoteIP":                 "source_ip",

    # WHAT
    "DestinationHostName":      "data_store_name",
    "DstHostname":              "data_store_name",
    "ResourceId":               "data_store_name",
    "OperationName":            "data_path",
    "ObjectName":               "data_path",
    "DestinationFileName":      "data_path",
    "FilePath":                 "data_path",
    "RequestURL":               "data_path",
    "Url":                      "data_path",

    # HOW MUCH
    "SentBytes":                "bytes_accessed",
    "BytesSent":                "bytes_accessed",
    "ReceivedBytes":            "bytes_accessed",
    "FileSize":                 "bytes_accessed",
    "NetworkBytes":             "bytes_accessed",

    # WHEN
    "TimeGenerated":            "event_time",
    "StartTime":                "event_time",
    "CreatedTimeUTC":           "event_time",
}


class SentinelNormalizer:
    """
    Converts Microsoft Sentinel ASIM events
    into DataAccessEvent format.

    Handles all Sentinel data source types
    through the unified ASIM schema.
    """

    def __init__(
        self,
        workspace_id: str = None,
        workspace_key: str = None
    ):
        self.workspace_id = (
            workspace_id or
            os.getenv("SENTINEL_WORKSPACE_ID", "")
        )
        self.workspace_key = (
            workspace_key or
            os.getenv("SENTINEL_WORKSPACE_KEY", "")
        )

    def normalize(self, raw_event: dict) -> dict:
        """
        Convert a Sentinel ASIM event to
        DataAccessEvent format.

        Args:
            raw_event: Sentinel event dict
                       in ASIM format

        Returns:
            DataAccessEvent compatible dict
        """
        if not raw_event:
            return self._empty_event()

        # Detect the table/source type
        table = raw_event.get(
            "Type", raw_event.get("TableName", "")
        )
        source_system = SENTINEL_TABLE_MAP.get(
            table, "sentinel_unknown"
        )

        # Extract core fields via ASIM mapping
        accessor = self._extract_accessor(raw_event)
        source_ip = self._extract_source_ip(raw_event)
        data_store = self._extract_data_store(raw_event)
        data_path = self._extract_data_path(raw_event)
        bytes_accessed = self._extract_bytes(raw_event)
        event_time = self._extract_time(raw_event)
        accessor_type = self._detect_accessor_type(
            accessor, raw_event
        )

        # Calculate initial risk score
        risk_score, risk_reasons = (
            self._calculate_risk(raw_event, source_system)
        )

        return {
            "accessor_identity": accessor,
            "accessor_type": accessor_type,
            "data_store_name": data_store,
            "data_path": data_path,
            "data_classification": (
                self._detect_classification(
                    data_store, data_path
                )
            ),
            "bytes_accessed": bytes_accessed,
            "event_time": event_time,
            "source_ip": source_ip,
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": f"sentinel_{source_system}",
            "raw_event": raw_event,
            # Sentinel-specific enrichment
            "sentinel_table": table,
            "sentinel_alert_severity": raw_event.get(
                "AlertSeverity", ""
            ),
            "sentinel_incident_id": raw_event.get(
                "SystemAlertId", ""
            ),
            "sentinel_mitre_techniques": raw_event.get(
                "Techniques", []
            ),
            "sentinel_entities": raw_event.get(
                "Entities", []
            ),
        }

    def normalize_batch(
        self,
        events: list
    ) -> list:
        """
        Normalize a batch of Sentinel events.

        Args:
            events: List of ASIM event dicts

        Returns:
            List of DataAccessEvent dicts
        """
        normalized = []
        for event in events:
            try:
                normalized.append(
                    self.normalize(event)
                )
            except Exception as e:
                logger.warning(
                    f"Failed to normalize event: {e}"
                )
        return normalized

    def normalize_security_alert(
        self,
        alert: dict
    ) -> dict:
        """
        Normalize a Sentinel SecurityAlert
        specifically.

        SecurityAlerts are high-value events:
        Microsoft's own detection rules fired.
        Our platform adds ML intelligence on top.

        Args:
            alert: Sentinel SecurityAlert dict

        Returns:
            DataAccessEvent with alert context
        """
        base = self.normalize(alert)

        # Enrich with alert-specific fields
        alert_name = alert.get("AlertName", "")
        alert_severity = alert.get(
            "AlertSeverity", "Informational"
        )
        description = alert.get(
            "Description", ""
        )
        remediation = alert.get(
            "RemediationSteps", ""
        )
        extended = alert.get(
            "ExtendedProperties", {}
        )

        # Map Microsoft severity to risk score
        severity_map = {
            "High":          0.85,
            "Medium":        0.65,
            "Low":           0.35,
            "Informational": 0.10
        }
        ms_risk = severity_map.get(
            alert_severity, 0.30
        )

        # Take higher of our score or Microsoft's
        final_risk = max(
            base.get("risk_score", 0.0),
            ms_risk
        )

        base["risk_score"] = final_risk
        base["sentinel_alert_name"] = alert_name
        base["sentinel_description"] = description
        base["sentinel_remediation"] = remediation
        base["sentinel_extended"] = extended

        if ms_risk >= 0.7:
            base["risk_reasons"].append(
                f"sentinel_high_severity_alert:"
                f"{alert_name}"
            )

        logger.info(
            f"Normalized SecurityAlert: {alert_name} "
            f"severity={alert_severity} "
            f"risk={final_risk:.2f}"
        )

        return base

    def write_enrichment(
        self,
        incident_id: str,
        risk_score: float,
        verdict: str,
        agent_summary: str,
        mitre_techniques: list = None,
        policy_violations: list = None,
        hitl_status: str = "PENDING",
        workspace_id: str = None
    ) -> dict:
        """
        Write AbuTech investigation results
        BACK to Sentinel incident.

        This closes the bidirectional loop:
        Sentinel → AbuTech → Sentinel

        Analyst opens Sentinel and sees
        AbuTech enrichment inline.
        No new tool to learn.

        Args:
            incident_id: Sentinel incident ID
            risk_score: AbuTech ensemble score
            verdict: ML verdict
            agent_summary: Claude investigation
            mitre_techniques: ATT&CK techniques
            policy_violations: IAM violations
            hitl_status: Approval status
            workspace_id: Override workspace

        Returns:
            Enrichment payload dict
        """
        enrichment = {
            "abutech_risk_score": risk_score,
            "abutech_verdict": verdict,
            "abutech_severity": (
                self._score_to_severity(risk_score)
            ),
            "abutech_agent_summary": agent_summary,
            "abutech_mitre_techniques": (
                mitre_techniques or []
            ),
            "abutech_policy_violations": (
                policy_violations or []
            ),
            "abutech_hitl_status": hitl_status,
            "abutech_platform_version": "1.0.0",
            "abutech_investigation_timestamp": _now()
        }

        # Build comment for Sentinel incident
        comment = self._build_sentinel_comment(
            risk_score, verdict, agent_summary,
            mitre_techniques, policy_violations,
            hitl_status
        )
        enrichment["sentinel_comment"] = comment

        # In production this calls Sentinel REST API:
        # PATCH /incidents/{incidentId}
        # Authorization: Bearer {token}
        # {
        #   "properties": {
        #     "labels": [{"labelName": "AbuTech-CRITICAL"}],
        #     "comments": [{"message": comment}]
        #   }
        # }

        logger.info(
            f"Enrichment prepared for incident "
            f"{incident_id}: "
            f"score={risk_score:.3f} "
            f"verdict={verdict}"
        )

        return enrichment

    # ============================================================
    # FIELD EXTRACTORS
    # ============================================================

    def _extract_accessor(self, event: dict) -> str:
        """Extract user/accessor identity from ASIM"""
        for field in [
            "AccountName", "UserPrincipalName",
            "SubjectUserName", "TargetUserName",
            "SrcUserName", "ActorUsername",
            "UserId", "Account"
        ]:
            val = event.get(field, "")
            if val:
                # Clean domain prefix if present
                if "\\" in val:
                    val = val.split("\\")[-1]
                return str(val)
        return "unknown"

    def _extract_source_ip(self, event: dict) -> str:
        """Extract source IP from ASIM"""
        for field in [
            "SourceIP", "SrcIpAddr", "ClientIP",
            "IPAddress", "CallerIpAddress",
            "RemoteIP", "SourceAddress"
        ]:
            val = event.get(field, "")
            if val and val not in [
                "::1", "127.0.0.1", ""
            ]:
                return str(val)
        return ""

    def _extract_data_store(
        self, event: dict
    ) -> str:
        """Extract destination/resource name"""
        for field in [
            "DestinationHostName", "ResourceId",
            "DstHostname", "Computer",
            "TargetResource", "ResourceGroup",
            "WorkspaceName", "StorageAccount"
        ]:
            val = event.get(field, "")
            if val:
                # Extract last segment of resource ID
                if "/" in val:
                    val = val.split("/")[-1]
                return str(val)
        return "unknown-resource"

    def _extract_data_path(
        self, event: dict
    ) -> str:
        """Extract specific file/path/operation"""
        for field in [
            "OperationName", "ObjectName",
            "FilePath", "FileName",
            "RequestURL", "Url",
            "DestinationFileName", "CommandLine"
        ]:
            val = event.get(field, "")
            if val:
                return str(val)[:500]
        return ""

    def _extract_bytes(self, event: dict) -> int:
        """Extract bytes accessed/transferred"""
        for field in [
            "SentBytes", "BytesSent",
            "ReceivedBytes", "NetworkBytes",
            "FileSize", "RequestBodySize",
            "ResponseBodySize"
        ]:
            val = event.get(field)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
        return 0

    def _extract_time(self, event: dict) -> str:
        """Extract and normalize event time to ISO 8601"""
        for field in [
            "TimeGenerated", "StartTime",
            "CreatedTimeUTC", "EventTime",
            "Timestamp", "timestamp"
        ]:
            val = event.get(field, "")
            if val:
                return self._normalize_timestamp(
                    str(val)
                )
        return _now()

    def _normalize_timestamp(self, ts: str) -> str:
        """Normalize various timestamp formats to ISO 8601"""
        if not ts:
            return _now()

        # Already ISO 8601
        if "T" in ts and "Z" in ts:
            return ts
        if "T" in ts:
            return ts + "Z"

        # Try common formats
        formats = [
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(ts[:19], fmt[:len(fmt)])
                return dt.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            except ValueError:
                continue

        return ts

    def _detect_accessor_type(
        self,
        accessor: str,
        event: dict
    ) -> str:
        """Detect if accessor is human, service, or API"""
        accessor_lower = accessor.lower()

        if any(
            p in accessor_lower
            for p in [
                "svc_", "_svc", "service_",
                "func_", "_func", "app_",
                "system", "automation", "pipeline"
            ]
        ):
            return "service_account"

        account_type = event.get(
            "AccountType", ""
        ).lower()
        if "machine" in account_type:
            return "service_account"
        if "service" in account_type:
            return "service_account"

        if "$" in accessor:
            return "service_account"

        if event.get("ServicePrincipalId"):
            return "api_key"

        if event.get("AppId"):
            return "api_key"

        return "human"

    def _detect_classification(
        self,
        data_store: str,
        data_path: str
    ) -> str:
        """Detect data sensitivity classification"""
        combined = (
            data_store.lower() + " " +
            data_path.lower()
        )

        if any(
            kw in combined
            for kw in [
                "pci", "card", "payment",
                "credit", "cardholder"
            ]
        ):
            return "PCI"

        if any(
            kw in combined
            for kw in [
                "phi", "health", "medical",
                "patient", "hipaa", "ehr"
            ]
        ):
            return "PHI"

        if any(
            kw in combined
            for kw in [
                "pii", "personal", "ssn",
                "social_security", "passport",
                "customer", "member"
            ]
        ):
            return "PII"

        if any(
            kw in combined
            for kw in [
                "confidential", "internal",
                "restricted", "sensitive"
            ]
        ):
            return "INTERNAL"

        return "UNKNOWN"

    def _calculate_risk(
        self,
        event: dict,
        source_system: str
    ) -> tuple:
        """Calculate initial risk score from ASIM fields"""
        risk = 0.0
        reasons = []

        # After hours
        event_time = self._extract_time(event)
        if event_time:
            try:
                hour = int(event_time[11:13])
                if hour < 6 or hour > 22:
                    risk += 0.15
                    reasons.append("after_hours_access")
            except (ValueError, IndexError):
                pass

        # Known bad IP
        source_ip = self._extract_source_ip(event)
        if source_ip:
            if source_ip.startswith("185.220"):
                risk += 0.40
                reasons.append("tor_exit_node")
            elif source_ip.startswith("45.142"):
                risk += 0.35
                reasons.append("known_attacker_range")

        # Large volume
        bytes_val = self._extract_bytes(event)
        bytes_mb = bytes_val / (1024 * 1024)
        if bytes_mb > 500:
            risk += 0.30
            reasons.append("large_volume_500mb+")
        elif bytes_mb > 100:
            risk += 0.20
            reasons.append("large_volume_100mb+")

        # Microsoft already flagged this
        severity = event.get(
            "AlertSeverity", ""
        )
        if severity == "High":
            risk += 0.30
            reasons.append(
                "microsoft_high_severity"
            )
        elif severity == "Medium":
            risk += 0.15
            reasons.append(
                "microsoft_medium_severity"
            )

        # Sentinel risk level
        risk_level = event.get("RiskLevel", "")
        if risk_level in ["High", "high"]:
            risk += 0.20
            reasons.append("sentinel_high_risk")

        # Sensitive data path
        data_path = self._extract_data_path(event)
        data_store = self._extract_data_store(event)
        classification = self._detect_classification(
            data_store, data_path
        )
        if classification == "PCI":
            risk += 0.20
            reasons.append("pci_data_access")
        elif classification == "PHI":
            risk += 0.18
            reasons.append("phi_data_access")
        elif classification == "PII":
            risk += 0.15
            reasons.append("pii_data_access")

        # MITRE techniques detected
        techniques = event.get("Techniques", [])
        if techniques:
            risk += 0.10
            reasons.append(
                f"mitre_techniques_detected:"
                f"{','.join(techniques[:3])}"
            )

        return min(risk, 1.0), reasons

    def _score_to_severity(
        self, score: float
    ) -> str:
        """Convert score to Sentinel severity label"""
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Medium"
        elif score >= 0.3:
            return "Low"
        return "Informational"

    def _build_sentinel_comment(
        self,
        risk_score: float,
        verdict: str,
        agent_summary: str,
        mitre_techniques: list,
        policy_violations: list,
        hitl_status: str
    ) -> str:
        """Build formatted comment for Sentinel incident"""
        severity = self._score_to_severity(risk_score)
        lines = [
            "## AbuTech AI Security Platform",
            f"**Risk Score:** {risk_score:.3f} "
            f"({severity})",
            f"**ML Verdict:** {verdict}",
            "",
            "### Investigation Summary",
            agent_summary,
            "",
        ]

        if mitre_techniques:
            lines.append("### MITRE ATT&CK")
            lines.append(
                " | ".join(mitre_techniques)
            )
            lines.append("")

        if policy_violations:
            lines.append("### Policy Violations")
            for v in policy_violations:
                lines.append(f"- {v}")
            lines.append("")

        lines += [
            "### HITL Status",
            f"**Approval required:** {hitl_status}",
            "",
            f"*Investigated by AbuTech Platform "
            f"v1.0.0 at {_now()}*"
        ]

        return "\n".join(lines)

    def _empty_event(self) -> dict:
        """Return empty event for invalid input"""
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
            "source_system": "sentinel_unknown",
            "raw_event": {}
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")