"""
Layer 1 — Data Ingestion
CSPM and CIEM Normalizer

Handles Cloud Security Posture Management
and Cloud Identity Entitlement Management events.

CSPM SOURCES HANDLED:
    Wiz:
        Misconfiguration findings
        Vulnerability alerts
        Toxic combination detection
    
    Microsoft Defender for Cloud:
        Security recommendations
        Security alerts
        Compliance violations
    
    AWS Security Hub:
        CIS benchmark findings
        PCI-DSS findings
        IAM Access Analyzer findings
    
    Prisma Cloud:
        Policy violations
        Anomaly alerts
        Compliance reports

CIEM SOURCES HANDLED:
    AWS IAM Access Analyzer:
        Over-privileged roles
        Cross-account access
        Public resource exposure
    
    Azure Entra ID Access Reviews:
        Unused permissions
        Orphaned accounts
        Guest user access
    
    GCP IAM Recommender:
        Excess permission alerts
        Unused role bindings
    
    Wiz CIEM:
        Privilege escalation paths
        Toxic permission combinations
        Shadow admin detection

WHY CSPM + CIEM TOGETHER:
    CSPM: "The S3 bucket is public."
    CIEM: "svc_backup can access all S3 buckets."
    
    TOGETHER:
    "The public S3 bucket contains PCI data
     AND svc_backup has already accessed it
     47 times this week."
    
    COMBINED RISK > SUM OF PARTS.
    Your ensemble scores them together.
    Agents investigate the combined picture.

USAGE:
    normalizer = CSPMNormalizer()
    data_event = normalizer.normalize(wiz_finding)
    data_event = normalizer.normalize(aws_hub_finding)
    data_event = normalizer.normalize_ciem_event(
        iam_finding
    )
"""

import logging
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)

# Severity mapping from various CSPM tools
SEVERITY_RISK_MAP = {
    # Wiz severities
    "CRITICAL": 0.90,
    "HIGH":     0.70,
    "MEDIUM":   0.45,
    "LOW":      0.20,
    "INFO":     0.05,

    # AWS Security Hub
    "CRITICAL": 0.90,
    "HIGH":     0.70,
    "MEDIUM":   0.45,
    "LOW":      0.20,
    "INFORMATIONAL": 0.05,

    # Defender for Cloud
    "High":   0.75,
    "Medium": 0.45,
    "Low":    0.20,
}

# CSPM finding types
CSPM_FINDING_TYPES = {
    "public_storage":       "PUBLIC_STORAGE_EXPOSURE",
    "unencrypted_storage":  "UNENCRYPTED_DATA",
    "open_security_group":  "NETWORK_EXPOSURE",
    "no_mfa":               "MFA_NOT_ENFORCED",
    "public_rds":           "DATABASE_EXPOSURE",
    "weak_password_policy": "WEAK_AUTHENTICATION",
    "logging_disabled":     "AUDIT_GAP",
    "public_ami":           "IMAGE_EXPOSURE",
    "root_access_key":      "ROOT_CREDENTIAL_RISK",
    "excessive_permissions": "OVER_PRIVILEGED_IDENTITY",
    "unused_credentials":   "STALE_CREDENTIALS",
    "privilege_escalation": "PRIVILEGE_ESCALATION_PATH",
}

# CIEM specific finding types
CIEM_FINDING_TYPES = {
    "over_privileged_role":     "CIEM_OVER_PRIVILEGED",
    "unused_permissions":       "CIEM_UNUSED_PERMISSIONS",
    "cross_account_access":     "CIEM_CROSS_ACCOUNT",
    "privilege_escalation_path": "CIEM_PRIV_ESC_PATH",
    "shadow_admin":             "CIEM_SHADOW_ADMIN",
    "toxic_combination":        "CIEM_TOXIC_COMBO",
    "orphaned_account":         "CIEM_ORPHANED_ACCOUNT",
    "excessive_trust_policy":   "CIEM_EXCESSIVE_TRUST",
}


class CSPMNormalizer:
    """
    Normalizes CSPM and CIEM findings from
    multiple cloud security tools into
    DataAccessEvent format.

    Handles: Wiz, Defender for Cloud,
    AWS Security Hub, Prisma Cloud,
    AWS IAM Access Analyzer.
    """

    def __init__(self):
        self.source_system = "cspm"

    def normalize(self, raw_event: dict) -> dict:
        """
        Normalize CSPM or CIEM finding.
        Auto-detects source tool and event type.

        Args:
            raw_event: CSPM/CIEM finding dict

        Returns:
            DataAccessEvent compatible dict
        """
        if not raw_event:
            return self._empty_event()

        source_tool = self._detect_source(raw_event)

        if source_tool == "wiz":
            return self._normalize_wiz(raw_event)
        elif source_tool == "defender":
            return self._normalize_defender(raw_event)
        elif source_tool == "aws_security_hub":
            return self._normalize_aws_hub(raw_event)
        elif source_tool == "iam_access_analyzer":
            return self._normalize_iam_analyzer(
                raw_event
            )
        else:
            return self._normalize_generic(raw_event)

    def normalize_ciem_event(
        self, raw_event: dict
    ) -> dict:
        """
        Specialized normalization for CIEM findings.
        Adds identity entitlement specific fields.

        CIEM findings are IAM-focused:
        Who has too much access?
        What can they do?
        Have they used it?
        """
        base = self.normalize(raw_event)

        base["event_type"] = "CIEM_FINDING"
        base["source_system"] = "ciem"

        identity = raw_event.get("identity", {})
        entitlement = raw_event.get(
            "entitlement", {}
        )
        finding_type = raw_event.get(
            "findingType",
            raw_event.get("finding_type", "")
        )

        base["ciem_finding_type"] = CIEM_FINDING_TYPES.get(
            finding_type.lower().replace(" ", "_"),
            finding_type
        )
        base["ciem_identity"] = (
            identity.get("name", "")
            or identity.get("arn", "")
            or raw_event.get("principal", "")
        )
        base["ciem_permissions_count"] = (
            entitlement.get(
                "permissionsCount",
                entitlement.get(
                    "permissions_count", 0
                )
            )
        )
        base["ciem_last_used"] = (
            entitlement.get(
                "lastUsed",
                entitlement.get("last_used", "")
            )
        )
        base["ciem_risk_reason"] = raw_event.get(
            "riskReason",
            raw_event.get("risk_reason", "")
        )
        base["ciem_affected_resources"] = (
            raw_event.get("affectedResources", [])
        )

        # Elevate risk for high-value CIEM findings
        ciem_risk_elevation = {
            "shadow_admin": 0.30,
            "privilege_escalation_path": 0.25,
            "toxic_combination": 0.25,
            "cross_account_access": 0.20,
            "over_privileged_role": 0.15,
            "unused_permissions": 0.05
        }

        finding_lower = finding_type.lower().replace(
            " ", "_"
        )
        elevation = ciem_risk_elevation.get(
            finding_lower, 0.10
        )
        base["risk_score"] = min(
            base["risk_score"] + elevation, 1.0
        )
        base["risk_reasons"].append(
            f"ciem_finding:{finding_type}"
        )

        return base

    def _normalize_wiz(
        self, raw_event: dict
    ) -> dict:
        """Normalize Wiz security finding"""
        severity = raw_event.get("severity", "LOW")
        risk_score = SEVERITY_RISK_MAP.get(
            severity.upper(), 0.20
        )

        resource = raw_event.get("resource", {})
        entity = raw_event.get(
            "entity",
            raw_event.get("entitySnapshot", {})
        )

        accessor = resource.get(
            "name",
            entity.get("name", "unknown-resource")
        )
        data_store = resource.get(
            "type",
            entity.get("type", "unknown")
        )
        data_path = raw_event.get(
            "title",
            raw_event.get("name", "")
        )

        risk_reasons = [
            f"wiz_severity:{severity}",
            f"wiz_finding:{raw_event.get('type', '')}"
        ]

        if raw_event.get("hasExternalExposure"):
            risk_score = min(risk_score + 0.20, 1.0)
            risk_reasons.append(
                "external_exposure_confirmed"
            )

        if raw_event.get(
            "isToxicCombination",
            raw_event.get("toxic_combination", False)
        ):
            risk_score = min(risk_score + 0.25, 1.0)
            risk_reasons.append(
                "toxic_permission_combination"
            )

        subscriptions = raw_event.get(
            "subscriptions", []
        )
        affected_ip = (
            subscriptions[0].get("externalIp", "")
            if subscriptions else ""
        )

        return {
            "accessor_identity": accessor,
            "accessor_type": self._detect_accessor_type(
                accessor
            ),
            "data_store_name": data_store,
            "data_path": data_path,
            "data_classification": (
                self._detect_classification(
                    data_store, data_path
                )
            ),
            "bytes_accessed": 0,
            "event_time": raw_event.get(
                "createdAt",
                raw_event.get("created_at", _now())
            ),
            "source_ip": affected_ip,
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "cspm_wiz",
            "raw_event": raw_event,
            "cspm_tool": "wiz",
            "cspm_severity": severity,
            "cspm_finding_id": raw_event.get(
                "id", ""
            )
        }

    def _normalize_defender(
        self, raw_event: dict
    ) -> dict:
        """Normalize Microsoft Defender for Cloud"""
        severity = raw_event.get(
            "severity",
            raw_event.get("properties", {})
            .get("severity", "Medium")
        )
        risk_score = SEVERITY_RISK_MAP.get(
            severity, 0.45
        )

        properties = raw_event.get("properties", {})
        resource_details = properties.get(
            "resourceDetails", {}
        )

        accessor = resource_details.get(
            "ResourceName",
            raw_event.get("name", "unknown")
        )
        data_store = resource_details.get(
            "ResourceType",
            properties.get(
                "resourceType", "unknown"
            )
        )

        risk_reasons = [
            f"defender_severity:{severity}",
            f"defender_alert:"
            f"{properties.get('alertDisplayName', '')}"
        ]

        if properties.get("isIncident"):
            risk_score = min(risk_score + 0.15, 1.0)
            risk_reasons.append(
                "defender_confirmed_incident"
            )

        return {
            "accessor_identity": accessor,
            "accessor_type": self._detect_accessor_type(
                accessor
            ),
            "data_store_name": data_store,
            "data_path": properties.get(
                "alertDisplayName", ""
            ),
            "data_classification": (
                self._detect_classification(
                    data_store,
                    properties.get(
                        "alertDisplayName", ""
                    )
                )
            ),
            "bytes_accessed": 0,
            "event_time": raw_event.get(
                "startTimeUtc",
                properties.get(
                    "startTimeUtc", _now()
                )
            ),
            "source_ip": properties.get(
                "compromisedEntity", ""
            ),
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "cspm_defender",
            "raw_event": raw_event,
            "cspm_tool": "defender_for_cloud",
            "cspm_severity": severity
        }

    def _normalize_aws_hub(
        self, raw_event: dict
    ) -> dict:
        """Normalize AWS Security Hub finding"""
        severity_label = (
            raw_event.get("Severity", {})
            .get("Label", "LOW")
        )
        risk_score = SEVERITY_RISK_MAP.get(
            severity_label, 0.20
        )

        resources = raw_event.get("Resources", [{}])
        resource = resources[0] if resources else {}

        accessor = resource.get("Id", "unknown")
        data_store = resource.get("Type", "unknown")
        data_path = raw_event.get("Title", "")

        compliance = raw_event.get("Compliance", {})
        status = compliance.get("Status", "")

        risk_reasons = [
            f"aws_hub_severity:{severity_label}",
            f"aws_hub_finding:{raw_event.get('Id', '')}"
        ]

        if status == "FAILED":
            risk_score = min(risk_score + 0.10, 1.0)
            risk_reasons.append(
                "compliance_check_failed"
            )

        workflow = raw_event.get("Workflow", {})
        if workflow.get("Status") == "NEW":
            risk_reasons.append(
                "new_unreviewed_finding"
            )

        return {
            "accessor_identity": accessor,
            "accessor_type": self._detect_accessor_type(
                accessor
            ),
            "data_store_name": data_store,
            "data_path": data_path,
            "data_classification": (
                self._detect_classification(
                    data_store, data_path
                )
            ),
            "bytes_accessed": 0,
            "event_time": raw_event.get(
                "UpdatedAt",
                raw_event.get("CreatedAt", _now())
            ),
            "source_ip": "",
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "cspm_aws_hub",
            "raw_event": raw_event,
            "cspm_tool": "aws_security_hub",
            "cspm_severity": severity_label,
            "cspm_compliance_status": status
        }

    def _normalize_iam_analyzer(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize AWS IAM Access Analyzer finding.
        IAM Access Analyzer detects:
        - Resources shared externally
        - Cross-account access
        - Public S3 buckets
        - Over-permissive policies
        """
        finding_type = raw_event.get(
            "findingType",
            raw_event.get("type", "")
        )
        status = raw_event.get("status", "ACTIVE")

        base_risk = 0.45
        if finding_type in [
            "ExternalAccess",
            "UnusedAccess",
            "PublicAccess"
        ]:
            base_risk = 0.70

        if raw_event.get("isPublic", False):
            base_risk = max(base_risk, 0.85)

        resource = raw_event.get("resource", "")
        resource_type = raw_event.get(
            "resourceType", ""
        )
        principal = (
            raw_event.get("principal", {})
        )
        principal_str = (
            principal.get("AWS", "")
            if isinstance(principal, dict)
            else str(principal)
        )

        risk_reasons = [
            f"iam_analyzer_type:{finding_type}",
            f"iam_analyzer_status:{status}"
        ]

        if raw_event.get("isPublic"):
            risk_reasons.append(
                "resource_publicly_accessible"
            )

        condition = raw_event.get("condition", {})
        if not condition:
            risk_reasons.append(
                "no_condition_on_policy"
            )

        return {
            "accessor_identity": (
                principal_str or resource
            ),
            "accessor_type": (
                "service_account"
                if "arn:aws:iam" in principal_str
                else "unknown"
            ),
            "data_store_name": resource,
            "data_path": finding_type,
            "data_classification": (
                self._detect_classification(
                    resource, resource_type
                )
            ),
            "bytes_accessed": 0,
            "event_time": raw_event.get(
                "analyzedAt",
                raw_event.get("createdAt", _now())
            ),
            "source_ip": "",
            "risk_score": base_risk,
            "risk_reasons": risk_reasons,
            "source_system": "ciem_iam_analyzer",
            "raw_event": raw_event,
            "cspm_tool": "aws_iam_access_analyzer",
            "ciem_finding_type": finding_type
        }

    def _normalize_generic(
        self, raw_event: dict
    ) -> dict:
        """Generic normalization for unknown tools"""
        severity = (
            raw_event.get("severity", "")
            or raw_event.get("Severity", "")
            or "MEDIUM"
        )

        if isinstance(severity, dict):
            severity = severity.get("Label", "MEDIUM")

        risk_score = SEVERITY_RISK_MAP.get(
            str(severity).upper(), 0.45
        )

        return {
            "accessor_identity": (
                raw_event.get("resource", "")
                or raw_event.get("ResourceName", "")
                or "unknown"
            ),
            "accessor_type": "unknown",
            "data_store_name": (
                raw_event.get("resourceType", "")
                or raw_event.get("type", "unknown")
            ),
            "data_path": (
                raw_event.get("title", "")
                or raw_event.get("Title", "")
                or raw_event.get("description", "")
            ),
            "data_classification": "UNKNOWN",
            "bytes_accessed": 0,
            "event_time": (
                raw_event.get("createdAt", "")
                or raw_event.get("CreatedAt", "")
                or _now()
            ),
            "source_ip": "",
            "risk_score": risk_score,
            "risk_reasons": [
                f"cspm_severity:{severity}"
            ],
            "source_system": "cspm_generic",
            "raw_event": raw_event,
            "cspm_tool": "unknown"
        }

    def _detect_source(
        self, raw_event: dict
    ) -> str:
        """Auto-detect which CSPM tool sent this"""
        if "entitySnapshot" in raw_event:
            return "wiz"
        if raw_event.get("cspm_tool") == "wiz":
            return "wiz"
        if "hasExternalExposure" in raw_event:
            return "wiz"
        if "isToxicCombination" in raw_event:
            return "wiz"

        if "startTimeUtc" in raw_event:
            return "defender"
        if "alertDisplayName" in str(
            raw_event.get("properties", {})
        ):
            return "defender"

        if "AwsAccountId" in raw_event:
            return "aws_security_hub"
        if "ProductArn" in raw_event:
            return "aws_security_hub"
        if "GeneratorId" in raw_event:
            return "aws_security_hub"

        if "analyzedAt" in raw_event:
            return "iam_access_analyzer"
        if "findingType" in raw_event and (
            "ExternalAccess" in str(
                raw_event.get("findingType", "")
            )
        ):
            return "iam_access_analyzer"

        return "generic"

    def _detect_accessor_type(
        self, accessor: str
    ) -> str:
        """Detect accessor type"""
        if not accessor:
            return "unknown"

        accessor_lower = accessor.lower()

        if any(
            kw in accessor_lower
            for kw in [
                "arn:aws:iam", "serviceaccount",
                "svc", "service", "func", "lambda",
                "role", "gserviceaccount"
            ]
        ):
            return "service_account"

        return "unknown"

    def _detect_classification(
        self,
        data_store: str,
        data_path: str
    ) -> str:
        """Detect data sensitivity"""
        combined = (
            str(data_store).lower() + " " +
            str(data_path).lower()
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
                "patient", "hipaa"
            ]
        ):
            return "PHI"

        if any(
            kw in combined
            for kw in [
                "pii", "personal", "ssn",
                "customer", "member"
            ]
        ):
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
            "source_system": "cspm",
            "raw_event": {}
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")