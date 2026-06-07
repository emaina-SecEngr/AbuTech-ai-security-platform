"""
Layer 1 — Data Ingestion
Source Detector

Identifies which security data source produced a
raw event, so the ingestion router can dispatch it
to the correct normalizer.

TWO DETECTION MODES:

1. EXPLICIT (preferred):
   The event arrived via a known channel — a
   webhook path like /api/ingest/crowdstrike, or
   a Kafka topic name. The source is already known
   and passed in directly. No detection needed.

2. INFERRED (fallback):
   A raw event arrives with no label (e.g. mixed
   syslog stream). We inspect the event's structure
   and fields to identify the source by its
   fingerprint.

WHY FINGERPRINTING WORKS:
   Each vendor's events have telltale fields.
   A Kubernetes audit log has "objectRef" and "verb".
   A Defender alert has "AlertDisplayName".
   An S3 event has "eventName" and "bucketName".
   We match on these signatures.

DESIGN NOTE:
   Detection is ordered most-specific first.
   More distinctive signatures are checked before
   generic ones to avoid misclassification.

USAGE:
    detector = SourceDetector()
    source = detector.detect(raw_event)
    # returns e.g. "kubernetes", "defender_cloud",
    # or "unknown"
"""

import logging

logger = logging.getLogger(__name__)


# Known source names — the canonical identifiers
# used by the router to look up normalizers.
KNOWN_SOURCES = {
    "s3", "rds", "snowflake", "sharepoint", "oracle",
    "aws_secrets", "azure_keyvault",
    "okta", "entraid", "cyberark", "sailpoint",
    "sentinel",
    "firewall", "waf", "api_gateway", "network_flow",
    "email_gateway", "email",
    "cspm", "iac", "gcp",
    "crowdstrike", "kubernetes",
    "cwpp", "defender_cloud", "purview_dlp", "guardduty",
}


class SourceDetector:
    """
    Detects the source system of a raw security
    event through field-signature fingerprinting.
    """

    def __init__(self):
        self.detections = 0
        self.unknown = 0

    def detect(
        self,
        raw_event: dict,
        hint: str = None
    ) -> str:
        """
        Identify the source of a raw event.

        Args:
            raw_event: The raw event dict
            hint: Optional explicit source name
                  (from webhook path, topic, etc.)

        Returns:
            Canonical source name, or "unknown"
        """
        # Mode 1: explicit hint wins
        if hint:
            normalized_hint = hint.lower().strip()
            if normalized_hint in KNOWN_SOURCES:
                self.detections += 1
                return normalized_hint
            # hint given but unrecognized - still
            # try to honor it if close, else infer
            logger.warning(
                f"Unrecognized source hint: {hint}"
            )

        # Mode 2: infer from event structure
        if not isinstance(raw_event, dict) or (
            not raw_event
        ):
            self.unknown += 1
            return "unknown"

        source = self._fingerprint(raw_event)

        if source == "unknown":
            self.unknown += 1
        else:
            self.detections += 1

        return source

    def _fingerprint(self, e: dict) -> str:
        """
        Inspect event fields to identify the source.
        Ordered most-specific first.
        """

        # --- Kubernetes audit log ---
        # Distinctive: objectRef + verb, or
        # requestReceivedTimestamp
        if "objectRef" in e and "verb" in e:
            return "kubernetes"
        if "requestReceivedTimestamp" in e:
            return "kubernetes"

        # --- Falco runtime (part of k8s normalizer) ---
        if "rule" in e and "priority" in e and (
            "output" in e
        ):
            return "kubernetes"

        # --- Microsoft Defender for Cloud ---
        # Distinctive: AlertDisplayName / alertType
        # under properties or flat
        # --- AWS GuardDuty ---
        # Distinctive: type like "X:Y/Z" + severity
        # + service.action structure
        ftype = e.get("type", e.get("Type", ""))
        if (
            isinstance(ftype, str)
            and ":" in ftype
            and "/" in ftype
            and ("service" in e or "Service" in e
                 or "resource" in e or "Resource" in e)
        ):
            return "guardduty"
        props = e.get("properties", e)
        if isinstance(props, dict):
            if any(k in props for k in [
                "alertDisplayName", "AlertDisplayName",
                "alertType", "AlertType"
            ]):
                return "defender_cloud"
            if "intent" in props and (
                "compromisedEntity" in props
            ):
                return "defender_cloud"

        # --- Microsoft Purview DLP ---
        # Distinctive: Workload + DLP fields
        if "Workload" in e or "workload" in e:
            if any(k in e for k in [
                "DLPAction", "SensitiveInfoTypeData",
                "PolicyDetails", "SensitiveInformationType"
            ]):
                return "purview_dlp"

        # --- AWS S3 (CloudTrail) ---
        if "eventName" in e:
            req = e.get("requestParameters", {})
            if isinstance(req, dict) and (
                "bucketName" in req
            ):
                return "s3"
            # general CloudTrail S3 source
            event_source = e.get("eventSource", "")
            if "s3" in str(event_source):
                return "s3"
            if "rds" in str(event_source):
                return "rds"

        # --- Okta system log ---
        if "eventType" in e and "actor" in e:
            actor = e.get("actor", {})
            if isinstance(actor, dict) and (
                "alternateId" in actor
            ):
                return "okta"

        # --- Microsoft Entra ID ---
        if "userPrincipalName" in e or (
            "appDisplayName" in e and "ipAddress" in e
        ):
            return "entraid"

        # --- CrowdStrike EDR ---
        if "DetectName" in e or "detect_name" in e:
            if "behaviors" in e or "device" in e:
                return "crowdstrike"
        if "event_simpleName" in e:
            return "crowdstrike"

        # --- CWPP runtime (Prisma/Aqua/Falcon) ---
        if "containerName" in e or (
            "imageName" in e
        ):
            return "cwpp"
        if "control" in e and "container" in e:
            return "cwpp"

        # --- Firewall (Palo Alto / generic) ---
        if "src" in e and "dst" in e and (
            "action" in e
        ):
            return "firewall"

        # --- WAF ---
        if "httpRequest" in e or (
            "waf_rule" in e
        ):
            return "waf"

        # --- Network flow ---
        if "flow_duration" in e or (
            "fwd_packets" in e
        ):
            return "network_flow"

        # --- Email gateway ---
        if "sender" in e and "recipient" in e and (
            "subject" in e
        ):
            return "email_gateway"

        # --- CSPM ---
        if "finding_type" in e or (
            "compliance_standard" in e
        ):
            return "cspm"

        # --- IaC scan ---
        if "check_id" in e or "resource_type" in e:
            return "iac"

        # --- GCP ---
        if "protoPayload" in e or (
            "resource" in e and "logName" in e
        ):
            return "gcp"

        # --- SailPoint ---
        if "identityId" in e or (
            "accessProfile" in e
        ):
            return "sailpoint"

        # --- CyberArk ---
        if "SafeName" in e or "AccountId" in e:
            return "cyberark"

        return "unknown"

    def get_statistics(self) -> dict:
        """Return detection statistics"""
        total = self.detections + self.unknown
        rate = (
            self.detections / total * 100
            if total > 0 else 0
        )
        return {
            "total": total,
            "detected": self.detections,
            "unknown": self.unknown,
            "detection_rate_pct": round(rate, 2)
        }