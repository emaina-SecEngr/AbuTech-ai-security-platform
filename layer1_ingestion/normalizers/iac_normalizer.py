"""
Layer 1 — Data Ingestion
Infrastructure as Code (IaC) Security Normalizer

Converts IaC security scan findings into
DataAccessEvent format for ML scoring.

WHAT IaC SECURITY IS:
    Traditional security: deploy first, scan later.
    IaC security: scan BEFORE deployment.
    
    Developer writes Terraform:
    resource "aws_s3_bucket" "customer_data" {
      bucket = "prod-customer-pci-data"
      acl    = "public-read"  ← CRITICAL FINDING
    }
    
    YOUR PIPELINE catches this BEFORE terraform apply.
    Never reaches production.
    Zero breach risk.
    Developer fixes in 5 minutes.
    
    THIS IS SHIFT LEFT SECURITY.
    The most cost-effective security control.
    IBM sells this as a premium service.

TOOLS SUPPORTED:
    Checkov:
        Most popular IaC scanner.
        3000+ checks built-in.
        Supports: Terraform, CloudFormation,
        Bicep, Kubernetes, Helm, Dockerfile.
        Output: JSON format.
        Cost: FREE.
    
    tfsec:
        Terraform-focused scanner.
        Fast and lightweight.
        Good for pre-commit hooks.
        Output: JSON format.
    
    Terrascan:
        Multi-IaC support.
        OPA policy engine.
        Output: JSON format.
    
    Semgrep IaC:
        Pattern-based scanning.
        Custom rules support.
        Output: SARIF format.

WHY IaC FINDINGS FEED YOUR ML MODELS:
    A Terraform file with public S3 bucket
    AND your S3 access logs showing that bucket
    was accessed by a suspicious IP
    = confirmed misconfiguration exploitation.
    
    Your knowledge graph connects:
    IaC finding (bucket was misconfigured)
    + S3Normalizer event (bucket was accessed)
    + DarkWebEnricher (attacker IP known)
    = CRITICAL: confirmed breach via misconfiguration

IaC FINDINGS IN YOUR PLATFORM:
    Layer 1: IaCNormalizer converts finding
    Layer 2: PIIClassifier checks if PCI resource
             IsolationForest checks severity anomaly
    Layer 3: KG links finding to affected resource
    Layer 4: Agents investigate if resource was hit
    Layer 5: HITL: block deployment until fixed

USAGE:
    normalizer = IaCNormalizer()
    
    # From Checkov JSON output
    events = normalizer.normalize_checkov_output(
        checkov_json
    )
    
    # Single finding
    event = normalizer.normalize(finding_dict)
    
    # From tfsec output
    events = normalizer.normalize_tfsec_output(
        tfsec_json
    )
    
    # Get high risk findings only
    critical = normalizer.filter_by_severity(
        findings, min_severity="HIGH"
    )
"""

import logging
import os
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)

# Severity mapping from various tools
SEVERITY_RISK_MAP = {
    # Checkov severities
    "CRITICAL": 0.95,
    "HIGH":     0.75,
    "MEDIUM":   0.50,
    "LOW":      0.25,
    "INFO":     0.10,

    # tfsec severities
    "error":    0.80,
    "warning":  0.50,
    "info":     0.20,

    # Terrascan severities
    "CRITICAL": 0.95,
    "HIGH":     0.75,
    "MEDIUM":   0.50,
    "LOW":      0.25,
}

# IaC resource types mapped to data sensitivity
RESOURCE_SENSITIVITY = {
    # Storage
    "aws_s3_bucket":                "STORAGE",
    "aws_s3_bucket_object":         "STORAGE",
    "azurerm_storage_account":      "STORAGE",
    "google_storage_bucket":        "STORAGE",

    # Database
    "aws_db_instance":              "DATABASE",
    "aws_rds_cluster":              "DATABASE",
    "azurerm_sql_database":         "DATABASE",
    "google_sql_database_instance": "DATABASE",
    "aws_dynamodb_table":           "DATABASE",

    # Identity
    "aws_iam_role":                 "IDENTITY",
    "aws_iam_policy":               "IDENTITY",
    "aws_iam_user":                 "IDENTITY",
    "azurerm_role_assignment":      "IDENTITY",
    "google_service_account":       "IDENTITY",

    # Network
    "aws_security_group":           "NETWORK",
    "aws_vpc":                      "NETWORK",
    "azurerm_network_security_group": "NETWORK",
    "google_compute_firewall":      "NETWORK",

    # Secrets
    "aws_secretsmanager_secret":    "SECRETS",
    "aws_ssm_parameter":            "SECRETS",
    "azurerm_key_vault":            "SECRETS",
    "google_secret_manager_secret": "SECRETS",

    # Compute
    "aws_instance":                 "COMPUTE",
    "aws_lambda_function":          "COMPUTE",
    "azurerm_virtual_machine":      "COMPUTE",
    "google_compute_instance":      "COMPUTE",

    # Container
    "aws_eks_cluster":              "CONTAINER",
    "azurerm_kubernetes_cluster":   "CONTAINER",
    "google_container_cluster":     "CONTAINER",
    "kubernetes_pod":               "CONTAINER",

    # Encryption
    "aws_kms_key":                  "ENCRYPTION",
    "azurerm_key_vault_key":        "ENCRYPTION",
    "google_kms_key_ring":          "ENCRYPTION",
}

# High risk check IDs from Checkov
CRITICAL_CHECKS = {
    # Public access
    "CKV_AWS_20":   "S3 bucket publicly accessible",
    "CKV_AWS_57":   "S3 bucket public access block",
    "CKV_AWS_19":   "S3 SSE encryption disabled",
    "CKV_AWS_18":   "S3 access logging disabled",

    # Database
    "CKV_AWS_17":   "RDS publicly accessible",
    "CKV_AWS_16":   "RDS encryption disabled",
    "CKV_AWS_133":  "RDS deletion protection off",
    "CKV_AWS_129":  "RDS logging disabled",

    # Network
    "CKV_AWS_25":   "Security group SSH open to world",
    "CKV_AWS_24":   "Security group RDP open to world",
    "CKV_AWS_23":   "Security group all traffic open",

    # IAM
    "CKV_AWS_40":   "IAM policy with wildcard actions",
    "CKV_AWS_1":    "IAM policy admin privileges",
    "CKV_AWS_2":    "Lambda not in VPC",

    # Encryption
    "CKV_AWS_7":    "KMS key rotation disabled",
    "CKV_AWS_119":  "DynamoDB not encrypted",
    "CKV_AWS_86":   "CloudFront not HTTPS only",

    # Container
    "CKV_K8S_16":   "Container privileged mode",
    "CKV_K8S_28":   "Container shares host PID",
    "CKV_K8S_30":   "Container shares host network",
    "CKV_K8S_6":    "Root containers allowed",

    # Azure
    "CKV_AZURE_3":  "Azure storage public access",
    "CKV_AZURE_22": "Azure SQL no audit",
    "CKV_AZURE_35": "Key Vault soft delete off",

    # GCP
    "CKV_GCP_28":   "GCS bucket public",
    "CKV_GCP_29":   "GCS no uniform access",
    "CKV_GCP_14":   "GCP SQL no SSL",
}

# MITRE ATT&CK mapping for IaC findings
FINDING_MITRE_MAP = {
    "public":       "T1190",
    "encryption":   "T1485",
    "iam":          "T1078",
    "logging":      "T1562.008",
    "network":      "T1190",
    "container":    "T1610",
    "secret":       "T1552",
    "backup":       "T1490",
}


class IaCNormalizer:
    """
    Normalizes Infrastructure as Code security
    findings from multiple scanning tools into
    DataAccessEvent format.

    Supports: Checkov, tfsec, Terrascan, Semgrep.
    """

    def __init__(self):
        self.source_system = "iac_security"

    def normalize_checkov_output(
        self,
        checkov_output: dict
    ) -> list:
        """
        Normalize full Checkov JSON output.
        Checkov returns results per check.

        Args:
            checkov_output: Full Checkov JSON dict
                            (from checkov -o json)

        Returns:
            List of DataAccessEvent dicts
        """
        if not checkov_output:
            return []

        results = checkov_output.get("results", {})
        failed_checks = results.get(
            "failed_checks", []
        )

        normalized = []
        for check in failed_checks:
            try:
                event = self.normalize_checkov_finding(
                    check
                )
                normalized.append(event)
            except Exception as e:
                logger.warning(
                    f"Failed to normalize check: {e}"
                )

        logger.info(
            f"Normalized {len(normalized)} "
            f"Checkov findings"
        )
        return normalized

    def normalize_checkov_finding(
        self,
        finding: dict
    ) -> dict:
        """
        Normalize single Checkov finding.

        Args:
            finding: Single failed_checks entry

        Returns:
            DataAccessEvent compatible dict
        """
        if not finding:
            return self._empty_event()

        check_id = finding.get("check_id", "")
        check_name = finding.get(
            "check", finding.get("check_name", "")
        )
        check_result = finding.get("check_result", {})
        resource = finding.get("resource", "")
        file_path = finding.get(
            "file_path",
            finding.get("repo_file_path", "")
        )
        file_line = finding.get(
            "file_line_range", [0, 0]
        )
        severity = finding.get(
            "severity",
            self._infer_severity(check_id)
        )
        guideline = finding.get("guideline", "")
        code_block = finding.get("code_block", [])

        resource_type = resource.split(".")[0] if (
            "." in resource
        ) else resource

        resource_sensitivity = RESOURCE_SENSITIVITY.get(
            resource_type, "UNKNOWN"
        )

        risk_score, risk_reasons = (
            self._calculate_risk(
                check_id, check_name,
                severity, resource_type,
                resource_sensitivity
            )
        )

        data_classification = (
            self._detect_classification(
                resource, check_name, file_path
            )
        )

        mitre = self._get_mitre_technique(
            check_name, check_id
        )

        return {
            "accessor_identity": resource,
            "accessor_type": "service_account",
            "data_store_name": file_path,
            "data_path": (
                f"{resource}:{file_line[0]}"
                if file_line else resource
            ),
            "data_classification": data_classification,
            "bytes_accessed": 0,
            "event_time": _now(),
            "source_ip": "",
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "iac_checkov",
            "raw_event": finding,
            "iac_tool": "checkov",
            "iac_check_id": check_id,
            "iac_check_name": check_name,
            "iac_severity": severity,
            "iac_resource": resource,
            "iac_resource_type": resource_type,
            "iac_resource_sensitivity": (
                resource_sensitivity
            ),
            "iac_file_path": file_path,
            "iac_file_line": file_line,
            "iac_guideline": guideline,
            "iac_mitre": mitre,
            "iac_fix_available": True,
            "iac_blocking": risk_score >= 0.70
        }

    def normalize_tfsec_output(
        self,
        tfsec_output: dict
    ) -> list:
        """
        Normalize tfsec JSON output.
        tfsec focuses on Terraform.

        Args:
            tfsec_output: Full tfsec JSON output

        Returns:
            List of DataAccessEvent dicts
        """
        if not tfsec_output:
            return []

        results = tfsec_output.get("results", [])
        if not results:
            return []

        normalized = []
        for result in results:
            try:
                event = self._normalize_tfsec_finding(
                    result
                )
                normalized.append(event)
            except Exception as e:
                logger.warning(
                    f"Failed to normalize tfsec: {e}"
                )

        return normalized

    def normalize_terrascan_output(
        self,
        terrascan_output: dict
    ) -> list:
        """
        Normalize Terrascan JSON output.

        Args:
            terrascan_output: Terrascan JSON output

        Returns:
            List of DataAccessEvent dicts
        """
        if not terrascan_output:
            return []

        runs = terrascan_output.get("runs", [])
        if not runs:
            return []

        normalized = []
        for run in runs:
            violations = run.get(
                "violations", []
            )
            for v in violations:
                try:
                    event = (
                        self._normalize_terrascan_finding(v)
                    )
                    normalized.append(event)
                except Exception as e:
                    logger.warning(
                        f"Terrascan normalize failed: {e}"
                    )

        return normalized

    def normalize(
        self, raw_finding: dict
    ) -> dict:
        """
        Auto-detect tool and normalize finding.

        Args:
            raw_finding: Finding dict from any tool

        Returns:
            DataAccessEvent compatible dict
        """
        if not raw_finding:
            return self._empty_event()

        tool = self._detect_tool(raw_finding)

        if tool == "checkov":
            return self.normalize_checkov_finding(
                raw_finding
            )
        elif tool == "tfsec":
            return self._normalize_tfsec_finding(
                raw_finding
            )
        elif tool == "terrascan":
            return self._normalize_terrascan_finding(
                raw_finding
            )
        else:
            return self._normalize_generic(raw_finding)

    def filter_by_severity(
        self,
        findings: list,
        min_severity: str = "HIGH"
    ) -> list:
        """
        Filter findings by minimum severity.
        Used by CI/CD gate to block on HIGH+.

        Args:
            findings: List of normalized findings
            min_severity: Minimum severity to include

        Returns:
            Filtered list of findings
        """
        severity_order = {
            "CRITICAL": 4,
            "HIGH":     3,
            "MEDIUM":   2,
            "LOW":      1,
            "INFO":     0
        }

        min_level = severity_order.get(
            min_severity.upper(), 3
        )

        return [
            f for f in findings
            if severity_order.get(
                f.get("iac_severity", "LOW").upper(),
                0
            ) >= min_level
        ]

    def generate_pipeline_report(
        self,
        findings: list,
        repo_name: str = "unknown",
        branch: str = "unknown"
    ) -> dict:
        """
        Generate CI/CD pipeline security report.
        Sent to platform after IaC scan completes.

        Args:
            findings: All normalized findings
            repo_name: GitLab/GitHub repo name
            branch: Branch being scanned

        Returns:
            Pipeline security report dict
        """
        critical = [
            f for f in findings
            if f.get("iac_severity") == "CRITICAL"
        ]
        high = [
            f for f in findings
            if f.get("iac_severity") == "HIGH"
        ]
        medium = [
            f for f in findings
            if f.get("iac_severity") == "MEDIUM"
        ]
        low = [
            f for f in findings
            if f.get(
                "iac_severity"
            ) in ["LOW", "INFO"]
        ]

        blocking = [
            f for f in findings
            if f.get("iac_blocking", False)
        ]

        max_risk = max(
            [f.get("risk_score", 0) for f in findings],
            default=0.0
        )

        pipeline_status = (
            "FAILED" if blocking
            else "PASSED"
        )

        return {
            "repo_name": repo_name,
            "branch": branch,
            "scan_time": _now(),
            "pipeline_status": pipeline_status,
            "total_findings": len(findings),
            "critical_count": len(critical),
            "high_count": len(high),
            "medium_count": len(medium),
            "low_count": len(low),
            "blocking_findings": len(blocking),
            "max_risk_score": max_risk,
            "pipeline_blocked": bool(blocking),
            "blocking_resources": [
                f.get("iac_resource", "")
                for f in blocking
            ],
            "top_findings": [
                {
                    "check_id": f.get(
                        "iac_check_id", ""
                    ),
                    "resource": f.get(
                        "iac_resource", ""
                    ),
                    "severity": f.get(
                        "iac_severity", ""
                    ),
                    "file": f.get(
                        "iac_file_path", ""
                    )
                }
                for f in sorted(
                    findings,
                    key=lambda x: x.get(
                        "risk_score", 0
                    ),
                    reverse=True
                )[:10]
            ],
            "sr11_7_compliant": (
                pipeline_status == "PASSED"
            ),
            "recommendation": (
                f"BLOCK deployment: "
                f"{len(blocking)} critical/high "
                f"findings must be fixed."
                if blocking
                else "APPROVE deployment: "
                     "No blocking findings."
            )
        }

    # ============================================================
    # PRIVATE METHODS
    # ============================================================

    def _normalize_tfsec_finding(
        self, result: dict
    ) -> dict:
        """Normalize tfsec finding"""
        severity = result.get(
            "severity", "WARNING"
        ).upper()
        description = result.get(
            "description",
            result.get("rule_description", "")
        )
        rule_id = result.get(
            "rule_id",
            result.get("long_id", "")
        )
        resource = result.get(
            "resource",
            result.get("location", {})
            .get("filename", "")
        )
        location = result.get("location", {})
        file_path = location.get("filename", "")
        start_line = location.get("start_line", 0)

        tfsec_severity_map = {
            "ERROR":   "HIGH",
            "WARNING": "MEDIUM",
            "INFO":    "LOW"
        }
        normalized_severity = tfsec_severity_map.get(
            severity, severity
        )

        risk_score = SEVERITY_RISK_MAP.get(
            normalized_severity, 0.40
        )
        risk_reasons = [
            f"tfsec_severity:{severity}",
            f"tfsec_rule:{rule_id}"
        ]

        return {
            "accessor_identity": resource,
            "accessor_type": "service_account",
            "data_store_name": file_path,
            "data_path": (
                f"{resource}:{start_line}"
            ),
            "data_classification": (
                self._detect_classification(
                    resource, description, file_path
                )
            ),
            "bytes_accessed": 0,
            "event_time": _now(),
            "source_ip": "",
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "iac_tfsec",
            "raw_event": result,
            "iac_tool": "tfsec",
            "iac_check_id": rule_id,
            "iac_check_name": description,
            "iac_severity": normalized_severity,
            "iac_resource": resource,
            "iac_file_path": file_path,
            "iac_mitre": self._get_mitre_technique(
                description, rule_id
            ),
            "iac_blocking": risk_score >= 0.70
        }

    def _normalize_terrascan_finding(
        self, violation: dict
    ) -> dict:
        """Normalize Terrascan violation"""
        severity = violation.get(
            "severity", "MEDIUM"
        ).upper()
        rule_name = violation.get("rule_name", "")
        rule_id = violation.get("rule_id", "")
        resource_name = violation.get(
            "resource_name", ""
        )
        resource_type = violation.get(
            "resource_type", ""
        )
        file_path = violation.get("file", "")
        line = violation.get("line", 0)
        category = violation.get("category", "")
        description = violation.get(
            "description", ""
        )

        risk_score = SEVERITY_RISK_MAP.get(
            severity, 0.50
        )
        risk_reasons = [
            f"terrascan_severity:{severity}",
            f"terrascan_rule:{rule_id}",
            f"terrascan_category:{category}"
        ]

        return {
            "accessor_identity": resource_name,
            "accessor_type": "service_account",
            "data_store_name": file_path,
            "data_path": (
                f"{resource_type}.{resource_name}"
                f":{line}"
            ),
            "data_classification": (
                self._detect_classification(
                    resource_type,
                    description,
                    file_path
                )
            ),
            "bytes_accessed": 0,
            "event_time": _now(),
            "source_ip": "",
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "iac_terrascan",
            "raw_event": violation,
            "iac_tool": "terrascan",
            "iac_check_id": rule_id,
            "iac_check_name": rule_name,
            "iac_severity": severity,
            "iac_resource": resource_name,
            "iac_resource_type": resource_type,
            "iac_file_path": file_path,
            "iac_mitre": self._get_mitre_technique(
                description, rule_id
            ),
            "iac_blocking": risk_score >= 0.70
        }

    def _normalize_generic(
        self, finding: dict
    ) -> dict:
        """Generic normalization fallback"""
        severity = (
            finding.get("severity", "")
            or finding.get("Severity", "MEDIUM")
        ).upper()

        risk_score = SEVERITY_RISK_MAP.get(
            severity, 0.40
        )

        return {
            "accessor_identity": finding.get(
                "resource", "unknown"
            ),
            "accessor_type": "service_account",
            "data_store_name": finding.get(
                "file", "unknown"
            ),
            "data_path": finding.get(
                "check_name",
                finding.get("message", "")
            ),
            "data_classification": "UNKNOWN",
            "bytes_accessed": 0,
            "event_time": _now(),
            "source_ip": "",
            "risk_score": risk_score,
            "risk_reasons": [
                f"iac_severity:{severity}"
            ],
            "source_system": "iac_generic",
            "raw_event": finding,
            "iac_tool": "unknown",
            "iac_severity": severity,
            "iac_blocking": risk_score >= 0.70
        }

    def _detect_tool(
        self, finding: dict
    ) -> str:
        """Auto-detect IaC scanning tool"""
        if "check_id" in finding:
            if finding["check_id"].startswith("CKV"):
                return "checkov"

        if "rule_id" in finding:
            rule_id = str(
                finding.get("rule_id", "")
            )
            if rule_id.startswith("aws") or (
                rule_id.startswith("GEN")
            ):
                return "tfsec"

        if "long_id" in finding:
            return "tfsec"

        if "rule_name" in finding and (
            "resource_type" in finding
        ):
            return "terrascan"

        return "generic"

    def _calculate_risk(
        self,
        check_id: str,
        check_name: str,
        severity: str,
        resource_type: str,
        resource_sensitivity: str
    ) -> tuple:
        """Calculate risk score for IaC finding"""
        base_risk = SEVERITY_RISK_MAP.get(
            severity.upper() if severity else "MEDIUM",
            0.50
        )

        risk = base_risk
        reasons = [f"iac_severity:{severity}"]

        if check_id in CRITICAL_CHECKS:
            risk = max(risk, 0.80)
            reasons.append(
                f"critical_check:{check_id}"
            )

        name_lower = check_name.lower()
        if "public" in name_lower:
            risk = max(risk, 0.85)
            reasons.append("public_exposure_risk")

        if "encrypt" in name_lower:
            risk = max(risk, 0.70)
            reasons.append("encryption_missing")

        if resource_sensitivity == "DATABASE":
            risk = min(risk + 0.10, 1.0)
            reasons.append("database_resource")

        if resource_sensitivity == "SECRETS":
            risk = min(risk + 0.15, 1.0)
            reasons.append("secrets_resource")

        if resource_sensitivity == "IDENTITY":
            risk = min(risk + 0.10, 1.0)
            reasons.append("identity_resource")

        if "ssh" in name_lower or (
            "rdp" in name_lower
        ):
            risk = max(risk, 0.90)
            reasons.append(
                "remote_access_open_to_world"
            )

        if "logging" in name_lower or (
            "audit" in name_lower
        ):
            risk = max(risk, 0.60)
            reasons.append("audit_logging_disabled")

        return min(risk, 1.0), reasons

    def _detect_classification(
        self,
        resource: str,
        check_name: str,
        file_path: str
    ) -> str:
        """Detect data classification"""
        combined = (
            resource + " " +
            check_name + " " +
            file_path
        ).lower()

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
                "pii", "personal", "customer",
                "member", "user"
            ]
        ):
            return "PII"

        if any(
            kw in combined
            for kw in [
                "database", "db", "sql",
                "rds", "dynamo"
            ]
        ):
            return "INTERNAL"

        return "UNKNOWN"

    def _get_mitre_technique(
        self,
        check_name: str,
        check_id: str
    ) -> str:
        """Map finding to MITRE ATT&CK technique"""
        combined = (
            check_name + " " + check_id
        ).lower()

        for keyword, technique in (
            FINDING_MITRE_MAP.items()
        ):
            if keyword in combined:
                return technique

        return "T1190"

    def _infer_severity(
        self, check_id: str
    ) -> str:
        """Infer severity from check ID"""
        if check_id in CRITICAL_CHECKS:
            desc = CRITICAL_CHECKS[check_id].lower()
            if any(
                kw in desc
                for kw in [
                    "public", "open", "ssh",
                    "rdp", "wildcard"
                ]
            ):
                return "HIGH"
        return "MEDIUM"

    def _empty_event(self) -> dict:
        """Empty event for invalid input"""
        return {
            "accessor_identity": "unknown",
            "accessor_type": "service_account",
            "data_store_name": "unknown",
            "data_path": "",
            "data_classification": "UNKNOWN",
            "bytes_accessed": 0,
            "event_time": _now(),
            "source_ip": "",
            "risk_score": 0.0,
            "risk_reasons": [],
            "source_system": "iac_security",
            "raw_event": {}
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")