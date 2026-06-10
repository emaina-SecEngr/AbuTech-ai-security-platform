"""
Layer 1 — Data Ingestion
AWS Security Hub Normalizer

Handles AWS Security Hub findings in the AWS Security
Finding Format (ASFF) — the aggregation point for
most AWS-native security services.

WHY SECURITY HUB MATTERS:
    Security Hub is AWS's central findings aggregator.
    Instead of integrating five services separately,
    one ASFF normalizer ingests findings from:
        Inspector  - vulnerability findings (CVEs)
        Macie      - sensitive data discovery (PII/PCI)
        Config     - configuration compliance
        IAM Access Analyzer - excess/external access
        GuardDuty  - threats (also direct)
        Firewall Manager, partner tools

    All of them conform to ASFF, so we identify the
    originating product from the ProductArn and
    enrich accordingly.

HOW ABUTECH ADDS VALUE (the pitch):
    Security Hub aggregates AWS findings but only
    within AWS, generates thousands of findings with
    flat severities, and cannot correlate with your
    non-AWS estate. We ingest Security Hub findings,
    ML-score them, map them to MITRE, attach the
    compliance requirement they touch, and correlate
    them with every other source - so your analysts
    see which AWS finding is actually being exploited
    right now, not just a wall of alerts.

THE COMPLIANCE ANGLE (financial sector):
    ASFF findings carry Compliance.RelatedRequirements
    listing the PCI-DSS, CIS, and NIST control IDs a
    finding touches. We surface these directly, which
    is essential for banks proving control coverage.

ASFF FINDING ANATOMY:
    ProductArn  - identifies the originating service
    Title, Description
    Severity: { Label, Normalized }
    Resources: [ affected AWS resources ]
    Types: [ classification strings ]
    Compliance: { Status, RelatedRequirements }

USAGE:
    normalizer = SecurityHubNormalizer()
    event = normalizer.normalize(asff_finding)
"""

import logging
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)


# ASFF severity label to base risk
ASFF_SEVERITY_RISK = {
    "CRITICAL":      0.90,
    "HIGH":          0.75,
    "MEDIUM":        0.50,
    "LOW":           0.25,
    "INFORMATIONAL": 0.10,
    "INFO":          0.10,
    "UNKNOWN":       0.40,
}

# Identify the originating product from the ProductArn.
# ProductArn looks like:
# arn:aws:securityhub:us-east-1::product/aws/inspector
PRODUCT_KEYWORDS = {
    "inspector":        "inspector",
    "macie":            "macie",
    "config":           "config",
    "access-analyzer":  "access_analyzer",
    "accessanalyzer":   "access_analyzer",
    "guardduty":        "guardduty",
    "firewall":         "firewall_manager",
    "securityhub":      "security_hub",
    "detective":        "detective",
}

# Originating product to representative MITRE technique
PRODUCT_TO_TECHNIQUE = {
    "inspector":       "T1190",   # exploit public vuln
    "macie":           "T1530",   # data from cloud storage
    "config":          "T1578",   # modify cloud infra
    "access_analyzer": "T1098",   # account manipulation
    "guardduty":       "T1078",   # valid accounts
    "firewall_manager":"T1562",   # impair defenses
}

# Keyword classification within Title/Types
ASFF_KEYWORD_TECHNIQUE = {
    "cve":                 "T1190",
    "vulnerability":       "T1190",
    "sensitive data":      "T1530",
    "pii":                 "T1530",
    "credit card":         "T1530",
    "public access":       "T1530",
    "publicly accessible": "T1530",
    "external access":     "T1098",
    "unrestricted":        "T1190",
    "unencrypted":         "T1565",
    "encryption":          "T1565",
    "root":                "T1078.004",
    "mfa":                 "T1078",
    "privilege":           "T1068",
    "exposed":             "T1530",
}

# Data classification keywords
CLASSIFICATION_KEYWORDS = {
    "PCI": ["pci", "payment", "card", "cardholder"],
    "PHI": ["phi", "health", "medical", "patient"],
    "PII": ["pii", "personal", "ssn", "social security"],
}


class SecurityHubNormalizer:
    """
    Normalizes AWS Security Hub (ASFF) findings into
    DataAccessEvent format, aggregating Inspector,
    Macie, Config, IAM Access Analyzer, and more.
    """

    def __init__(self):
        self.source_system = "aws_security_hub"

    def normalize(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize an AWS Security Hub ASFF finding.

        Accepts either a single finding dict, or a
        wrapper dict with a "Findings" list (takes
        the first finding).

        Args:
            raw_event: ASFF finding dict

        Returns:
            DataAccessEvent compatible dict
        """
        if not raw_event:
            return self._empty_event()

        # Some payloads wrap findings in a list
        finding = raw_event
        if "Findings" in raw_event:
            findings = raw_event.get("Findings", [])
            if isinstance(findings, list) and findings:
                finding = findings[0]
            else:
                return self._empty_event()

        product_arn = (
            finding.get("ProductArn", "")
            or finding.get("productArn", "")
        )
        product = self._identify_product(product_arn)

        title = (
            finding.get("Title", "")
            or finding.get("title", "")
        )
        description = (
            finding.get("Description", "")
            or finding.get("description", "")
        )

        severity_label = self._extract_severity(
            finding
        )
        base_risk = ASFF_SEVERITY_RISK.get(
            severity_label, 0.40
        )

        # Types classification
        types = finding.get(
            "Types", finding.get("types", [])
        )
        types_text = " ".join(
            t for t in types
            if isinstance(t, str)
        ) if isinstance(types, list) else ""

        # Resource
        resource = self._extract_resource(finding)

        # Compliance requirements (the bank-grade gold)
        compliance_reqs = self._extract_compliance(
            finding
        )
        compliance_status = self._compliance_status(
            finding
        )

        # Classification + MITRE
        text = (
            f"{title} {description} {types_text}"
        ).lower()
        classification = self._detect_classification(
            text
        )
        technique = self._classify_technique(
            text, product
        )

        # Risk
        risk, reasons = self._calculate_risk(
            base_risk, severity_label, product,
            text, compliance_reqs, finding
        )

        return {
            "accessor_identity": resource,
            "accessor_type": "cloud_resource",
            "data_store_name": resource,
            "data_path": (title or product)[:300],
            "data_classification": classification,
            "bytes_accessed": 0,
            "event_time": (
                finding.get("UpdatedAt")
                or finding.get("CreatedAt")
                or finding.get("updatedAt")
                or _now()
            ),
            "source_ip": "",
            "risk_score": min(round(risk, 4), 1.0),
            "risk_reasons": reasons,
            "source_system": "aws_security_hub",
            "raw_event": raw_event,
            "securityhub_product": product,
            "securityhub_severity": severity_label,
            "securityhub_title": title[:300],
            "securityhub_compliance_status": (
                compliance_status
            ),
            "compliance_requirements": compliance_reqs,
            "mitre_technique": technique,
        }

    def _identify_product(
        self, product_arn: str
    ) -> str:
        """Identify originating product from ProductArn"""
        arn_lower = product_arn.lower()
        for keyword, product in (
            PRODUCT_KEYWORDS.items()
        ):
            if keyword in arn_lower:
                return product
        return "security_hub"

    def _extract_severity(
        self, finding: dict
    ) -> str:
        """Extract ASFF severity label"""
        severity = finding.get(
            "Severity", finding.get("severity", {})
        )
        if isinstance(severity, dict):
            label = (
                severity.get("Label")
                or severity.get("label")
            )
            if label:
                return str(label).upper()
            # Fall back to normalized 0-100
            normalized = (
                severity.get("Normalized")
                or severity.get("normalized")
            )
            if normalized is not None:
                try:
                    n = int(normalized)
                    if n >= 90:
                        return "CRITICAL"
                    if n >= 70:
                        return "HIGH"
                    if n >= 40:
                        return "MEDIUM"
                    if n >= 1:
                        return "LOW"
                except (ValueError, TypeError):
                    pass
        return "UNKNOWN"

    def _extract_resource(
        self, finding: dict
    ) -> str:
        """Extract the first affected resource"""
        resources = finding.get(
            "Resources", finding.get("resources", [])
        )
        if isinstance(resources, list) and resources:
            first = resources[0]
            if isinstance(first, dict):
                # Prefer a readable Id
                rid = (
                    first.get("Id")
                    or first.get("id")
                    or first.get("Type")
                )
                if rid:
                    # Shorten full ARNs to the tail
                    rid = str(rid)
                    if "/" in rid:
                        return rid.split("/")[-1]
                    if ":" in rid:
                        return rid.split(":")[-1]
                    return rid
        return "unknown_resource"

    def _extract_compliance(
        self, finding: dict
    ) -> list:
        """Extract related compliance requirements"""
        compliance = finding.get(
            "Compliance", finding.get("compliance", {})
        )
        if isinstance(compliance, dict):
            reqs = (
                compliance.get("RelatedRequirements")
                or compliance.get("relatedRequirements")
                or []
            )
            if isinstance(reqs, list):
                return [str(r) for r in reqs][:10]
        return []

    def _compliance_status(
        self, finding: dict
    ) -> str:
        """Extract compliance status"""
        compliance = finding.get(
            "Compliance", finding.get("compliance", {})
        )
        if isinstance(compliance, dict):
            status = (
                compliance.get("Status")
                or compliance.get("status")
            )
            if status:
                return str(status).upper()
        return "UNKNOWN"

    def _detect_classification(
        self, text: str
    ) -> str:
        """Detect data classification from text"""
        for cls, keywords in (
            CLASSIFICATION_KEYWORDS.items()
        ):
            if any(k in text for k in keywords):
                return cls
        return "UNKNOWN"

    def _classify_technique(
        self, text: str, product: str
    ) -> str:
        """Map finding to a MITRE technique"""
        for keyword, technique in (
            ASFF_KEYWORD_TECHNIQUE.items()
        ):
            if keyword in text:
                return technique
        # Product-based fallback
        return PRODUCT_TO_TECHNIQUE.get(product, "")

    def _calculate_risk(
        self,
        base_risk: float,
        severity_label: str,
        product: str,
        text: str,
        compliance_reqs: list,
        finding: dict
    ) -> tuple:
        """Calculate Security Hub finding risk"""
        risk = base_risk
        reasons = [
            f"securityhub_severity:{severity_label}",
            f"securityhub_product:{product}",
        ]

        # Macie sensitive data exposure
        if product == "macie":
            if any(k in text for k in [
                "pci", "pii", "credit card",
                "sensitive"
            ]):
                risk = max(risk, 0.78)
                reasons.append(
                    "sensitive_data_exposed"
                )

        # Inspector exploitable vulnerability
        if product == "inspector":
            if "exploit" in text:
                risk = max(risk, 0.85)
                reasons.append(
                    "exploitable_vulnerability"
                )

        # Public / external exposure
        if any(k in text for k in [
            "public access", "publicly accessible",
            "exposed", "unrestricted"
        ]):
            risk = max(risk, 0.72)
            reasons.append("public_exposure")

        # External access (Access Analyzer)
        if "external access" in text:
            risk = max(risk, 0.70)
            reasons.append("external_access_granted")

        # Failed compliance escalates for regulated data
        compliance_status = self._compliance_status(
            finding
        )
        if compliance_status == "FAILED":
            reasons.append("compliance_failed")
            if compliance_reqs:
                risk = max(risk, base_risk + 0.05)

        # Surface the specific compliance controls
        if compliance_reqs:
            reasons.append(
                f"compliance_reqs:{len(compliance_reqs)}"
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
            "source_system": "aws_security_hub",
            "raw_event": {}
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")