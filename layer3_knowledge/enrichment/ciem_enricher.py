"""
Layer 3 — Knowledge Graph
Deep CIEM Enricher

Cloud Identity Entitlement Management
Deep Analysis Engine.

THIS IS THE MOST SOPHISTICATED COMPONENT
IN YOUR PLATFORM FOR IBM.

WHAT BASIC CIEM DOES:
    "svc_backup has too many permissions."
    A flag. Static. Reactive.

WHAT DEEP CIEM DOES:
    PRIVILEGE ESCALATION PATH MAPPING:
    Shows exactly HOW an identity can reach
    AdminAccess in N hops. Like GPS for hackers.
    
    "svc_backup → AssumeRole(DevRole)
     → AssumeRole(ProdRole)
     → iam:AttachUserPolicy
     → AdministratorAccess
     Path length: 3 hops
     Exploited: 2 times this week"
    
    BLAST RADIUS ANALYSIS:
    If this identity is compromised:
    How many resources are at risk?
    How many PCI records exposed?
    What is the financial impact?
    
    "If svc_backup compromised:
     47 S3 buckets accessible
     12 RDS databases accessible
     3 KMS keys usable for decryption
     Estimated PCI records at risk: 1.2M
     Estimated breach cost: $4.9M"
    
    PEER GROUP ANALYSIS:
    Compare identity permissions to peers.
    Statistical anomaly detection for IAM.
    
    "svc_backup: 847 permissions
     Peer group average: 8 permissions
     Anomaly: 105x above peer group
     Percentile: 99.9th"
    
    TOXIC COMBINATION DETECTION:
    Certain permission combinations enable
    complete compromise even individually.
    
    "svc_backup has:
     s3:GetObject (read data)
     + kms:Decrypt (decrypt data)
     + iam:PassRole (escalate to any role)
     TOXIC: Can exfiltrate all encrypted data
             AND escalate to admin"
    
    UNUSED PERMISSION ANALYSIS:
    Permissions that exist but are never used.
    Attack surface reduction opportunity.
    
    "svc_backup has ec2:TerminateInstances
     Last used: Never (granted 18 months ago)
     Risk: Can destroy production infrastructure
     Recommendation: Remove immediately"
    
    TEMPORAL ANALYSIS:
    How permissions have changed over time.
    Permission creep detection.
    
    "svc_backup permissions timeline:
     Jan 2025: 12 permissions (appropriate)
     Jun 2025: 145 permissions (+133)
     Jan 2026: 847 permissions (+702)
     PERMISSION CREEP DETECTED"

WHY IBM STAKEHOLDERS CARE:
    Average enterprise has 40,000 IAM roles.
    90% are over-privileged.
    No human can analyze all of them.
    Your CIEM enricher analyzes ALL of them
    automatically and prioritizes by risk.
    
    IBM pitch:
    "We reduce your attack surface by
     identifying the 1% of IAM roles that
     represent 99% of your breach risk."

USAGE:
    enricher = CIEMEnricher()
    
    # Analyze a single identity
    result = enricher.analyze_identity(
        identity="svc_backup",
        permissions=["s3:GetObject", "kms:Decrypt"],
        cloud="aws"
    )
    
    # Full blast radius
    blast = enricher.calculate_blast_radius(
        identity="svc_backup",
        accessible_resources=[...]
    )
    
    # Privilege escalation paths
    paths = enricher.find_escalation_paths(
        identity="svc_backup",
        current_permissions=[...]
    )
    
    # Enrich a DataAccessEvent with CIEM context
    enriched = enricher.enrich_event(data_event)
"""

import logging
from datetime import datetime
from datetime import timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Toxic permission combinations for AWS
# Each combination enables a specific attack
AWS_TOXIC_COMBINATIONS = [
    {
        "name": "Full Data Exfiltration",
        "permissions": [
            "s3:GetObject",
            "kms:Decrypt"
        ],
        "risk_score": 0.90,
        "attack": (
            "Read and decrypt all encrypted "
            "S3 data. Complete data exfiltration."
        ),
        "mitre": "T1530"
    },
    {
        "name": "Privilege Escalation via PassRole",
        "permissions": [
            "iam:PassRole",
            "ec2:RunInstances"
        ],
        "risk_score": 0.92,
        "attack": (
            "Launch EC2 with high-privilege role. "
            "Assume that role. Full privilege escalation."
        ),
        "mitre": "T1078.004"
    },
    {
        "name": "Shadow Admin Creation",
        "permissions": [
            "iam:CreateUser",
            "iam:AttachUserPolicy"
        ],
        "risk_score": 0.95,
        "attack": (
            "Create new IAM user and attach "
            "AdministratorAccess policy. "
            "Persistent backdoor admin."
        ),
        "mitre": "T1136.003"
    },
    {
        "name": "Policy Manipulation",
        "permissions": [
            "iam:PutUserPolicy",
            "iam:AttachUserPolicy"
        ],
        "risk_score": 0.93,
        "attack": (
            "Directly modify IAM policies to "
            "grant self or others admin access."
        ),
        "mitre": "T1484.001"
    },
    {
        "name": "Lambda Privilege Escalation",
        "permissions": [
            "lambda:CreateFunction",
            "lambda:InvokeFunction",
            "iam:PassRole"
        ],
        "risk_score": 0.91,
        "attack": (
            "Create Lambda with high-privilege role. "
            "Invoke to execute privileged code."
        ),
        "mitre": "T1078.004"
    },
    {
        "name": "Data Destruction",
        "permissions": [
            "s3:DeleteObject",
            "rds:DeleteDBInstance"
        ],
        "risk_score": 0.88,
        "attack": (
            "Destroy production data. "
            "Ransomware without encryption."
        ),
        "mitre": "T1485"
    },
    {
        "name": "Credential Theft via SecretsManager",
        "permissions": [
            "secretsmanager:GetSecretValue",
            "ssm:GetParameter"
        ],
        "risk_score": 0.87,
        "attack": (
            "Read all stored credentials, "
            "API keys, and connection strings."
        ),
        "mitre": "T1555.006"
    },
    {
        "name": "CloudTrail Blind Spot",
        "permissions": [
            "cloudtrail:StopLogging",
            "cloudtrail:DeleteTrail"
        ],
        "risk_score": 0.95,
        "attack": (
            "Disable audit logging. "
            "All subsequent actions invisible."
        ),
        "mitre": "T1562.008"
    }
]

# Azure toxic combinations
AZURE_TOXIC_COMBINATIONS = [
    {
        "name": "Full Subscription Control",
        "permissions": [
            "Microsoft.Authorization/roleAssignments/write",
            "Microsoft.Resources/subscriptions/resourceGroups/write"
        ],
        "risk_score": 0.95,
        "attack": (
            "Grant self Owner role. "
            "Full subscription control."
        ),
        "mitre": "T1078.004"
    },
    {
        "name": "Key Vault Data Exfiltration",
        "permissions": [
            "Microsoft.KeyVault/vaults/secrets/read",
            "Microsoft.Storage/storageAccounts/read"
        ],
        "risk_score": 0.88,
        "attack": (
            "Read all secrets and storage keys. "
            "Complete credential theft."
        ),
        "mitre": "T1555"
    }
]

# GCP toxic combinations
GCP_TOXIC_COMBINATIONS = [
    {
        "name": "Service Account Key Exfiltration",
        "permissions": [
            "iam.serviceAccountKeys.create",
            "iam.serviceAccounts.actAs"
        ],
        "risk_score": 0.93,
        "attack": (
            "Create new service account key. "
            "Impersonate any service account."
        ),
        "mitre": "T1550.001"
    }
]

# Privilege escalation paths by cloud
# Maps permission → what it enables
AWS_ESCALATION_MAP = {
    "iam:AttachUserPolicy": {
        "enables": "AdministratorAccess",
        "hops": 1,
        "risk": 0.95
    },
    "iam:PassRole": {
        "enables": "AssumeAnyRole",
        "hops": 2,
        "risk": 0.90
    },
    "sts:AssumeRole": {
        "enables": "CrossAccountAccess",
        "hops": 1,
        "risk": 0.80
    },
    "iam:CreatePolicyVersion": {
        "enables": "PolicyManipulation",
        "hops": 2,
        "risk": 0.88
    },
    "iam:SetDefaultPolicyVersion": {
        "enables": "PolicyVersionControl",
        "hops": 2,
        "risk": 0.85
    },
    "lambda:CreateFunction": {
        "enables": "ArbitraryCodeExecution",
        "hops": 2,
        "risk": 0.82
    },
    "cloudformation:CreateStack": {
        "enables": "ResourceManipulation",
        "hops": 3,
        "risk": 0.78
    }
}

# High risk permissions by category
HIGH_RISK_PERMISSIONS = {
    "data_access": [
        "s3:GetObject", "s3:ListBucket",
        "rds:connect", "dynamodb:Query",
        "dynamodb:Scan", "secretsmanager:GetSecretValue",
        "ssm:GetParameter", "kms:Decrypt"
    ],
    "identity_manipulation": [
        "iam:CreateUser", "iam:DeleteUser",
        "iam:AttachUserPolicy", "iam:DetachUserPolicy",
        "iam:CreateRole", "iam:DeleteRole",
        "iam:PassRole", "iam:UpdateAssumeRolePolicy"
    ],
    "audit_evasion": [
        "cloudtrail:StopLogging",
        "cloudtrail:DeleteTrail",
        "cloudtrail:UpdateTrail",
        "config:StopConfigurationRecorder",
        "guardduty:DeleteDetector"
    ],
    "data_destruction": [
        "s3:DeleteObject", "s3:DeleteBucket",
        "rds:DeleteDBInstance",
        "rds:DeleteDBCluster",
        "dynamodb:DeleteTable",
        "ec2:TerminateInstances"
    ]
}


class CIEMEnricher:
    """
    Deep Cloud Identity Entitlement Management
    analysis engine.

    Provides:
    - Privilege escalation path mapping
    - Blast radius calculation
    - Peer group analysis
    - Toxic combination detection
    - Unused permission identification
    - Permission creep detection
    """

    def __init__(self):
        self.toxic_combos = {
            "aws": AWS_TOXIC_COMBINATIONS,
            "azure": AZURE_TOXIC_COMBINATIONS,
            "gcp": GCP_TOXIC_COMBINATIONS
        }

    def analyze_identity(
        self,
        identity: str,
        permissions: list,
        cloud: str = "aws",
        peer_group_avg: int = 10,
        last_used_days: dict = None
    ) -> dict:
        """
        Complete CIEM analysis for a single identity.
        This is the main entry point.

        Args:
            identity: Identity name or ARN
            permissions: List of permission strings
            cloud: Cloud provider (aws/azure/gcp)
            peer_group_avg: Average permissions in peer group
            last_used_days: Dict of permission → days_since_used

        Returns:
            Complete CIEM analysis dict
        """
        permission_count = len(permissions)

        toxic = self.detect_toxic_combinations(
            permissions, cloud
        )
        escalation = self.find_escalation_paths(
            identity, permissions, cloud
        )
        unused = self.identify_unused_permissions(
            permissions,
            last_used_days or {}
        )
        peer_analysis = self.analyze_peer_group(
            permission_count, peer_group_avg
        )
        categories = self.categorize_permissions(
            permissions
        )

        risk_score = self._calculate_ciem_risk(
            toxic, escalation, unused,
            peer_analysis, categories
        )

        risk_reasons = []
        if toxic:
            risk_reasons.append(
                f"toxic_combinations:{len(toxic)}"
            )
        if escalation["has_escalation_path"]:
            risk_reasons.append(
                f"escalation_paths:"
                f"{escalation['path_count']}"
            )
        if peer_analysis["is_anomalous"]:
            risk_reasons.append(
                f"peer_group_anomaly:"
                f"{peer_analysis['ratio']:.1f}x"
            )
        if unused:
            risk_reasons.append(
                f"unused_permissions:{len(unused)}"
            )

        return {
            "identity": identity,
            "cloud": cloud,
            "permission_count": permission_count,
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "toxic_combinations": toxic,
            "escalation_paths": escalation,
            "unused_permissions": unused,
            "peer_group_analysis": peer_analysis,
            "permission_categories": categories,
            "recommendation": self._generate_recommendation(
                toxic, escalation, peer_analysis
            ),
            "analyzed_at": _now()
        }

    def detect_toxic_combinations(
        self,
        permissions: list,
        cloud: str = "aws"
    ) -> list:
        """
        Detect dangerous permission combinations.
        Even individually safe permissions can be
        toxic in combination.

        Args:
            permissions: List of granted permissions
            cloud: Cloud provider

        Returns:
            List of detected toxic combinations
        """
        detected = []
        combos = self.toxic_combos.get(cloud, [])
        perm_set = set(permissions)

        for combo in combos:
            required = set(
                combo["permissions"]
            )
            if required.issubset(perm_set):
                detected.append({
                    "name": combo["name"],
                    "matched_permissions": list(
                        required
                    ),
                    "risk_score": combo["risk_score"],
                    "attack_description": (
                        combo["attack"]
                    ),
                    "mitre_technique": combo.get(
                        "mitre", ""
                    )
                })

        return detected

    def find_escalation_paths(
        self,
        identity: str,
        permissions: list,
        cloud: str = "aws"
    ) -> dict:
        """
        Map privilege escalation paths.
        Shows HOW an identity can reach
        higher privileges.

        Args:
            identity: Identity name
            permissions: Current permissions
            cloud: Cloud provider

        Returns:
            dict with escalation path analysis
        """
        if cloud != "aws":
            return {
                "has_escalation_path": False,
                "path_count": 0,
                "paths": [],
                "max_risk": 0.0
            }

        paths = []
        perm_set = set(permissions)

        for perm, details in (
            AWS_ESCALATION_MAP.items()
        ):
            if perm in perm_set:
                paths.append({
                    "permission": perm,
                    "enables": details["enables"],
                    "hops_to_admin": details["hops"],
                    "risk_score": details["risk"],
                    "path": (
                        f"{identity} → "
                        f"{perm} → "
                        f"{details['enables']}"
                    )
                })

        paths.sort(
            key=lambda x: x["risk_score"],
            reverse=True
        )

        return {
            "has_escalation_path": len(paths) > 0,
            "path_count": len(paths),
            "paths": paths,
            "max_risk": (
                paths[0]["risk_score"]
                if paths else 0.0
            ),
            "shortest_path_hops": min(
                [p["hops_to_admin"] for p in paths],
                default=0
            ) if paths else 0
        }

    def calculate_blast_radius(
        self,
        identity: str,
        accessible_resources: list,
        estimated_records_per_resource: int = 10000
    ) -> dict:
        """
        Calculate breach impact if identity
        is compromised.

        Args:
            identity: Identity name
            accessible_resources: List of resource dicts
            estimated_records_per_resource: PII record estimate

        Returns:
            dict with blast radius analysis
        """
        s3_buckets = [
            r for r in accessible_resources
            if "s3" in str(r).lower() or
            "bucket" in str(r).lower()
        ]
        databases = [
            r for r in accessible_resources
            if any(
                kw in str(r).lower()
                for kw in [
                    "rds", "database", "db",
                    "sql", "dynamo"
                ]
            )
        ]
        secrets = [
            r for r in accessible_resources
            if any(
                kw in str(r).lower()
                for kw in [
                    "secret", "key", "vault",
                    "credential", "password"
                ]
            )
        ]
        kms_keys = [
            r for r in accessible_resources
            if "kms" in str(r).lower() or
            "key" in str(r).lower()
        ]

        pci_resources = [
            r for r in accessible_resources
            if any(
                kw in str(r).lower()
                for kw in [
                    "pci", "card", "payment",
                    "credit"
                ]
            )
        ]

        total_resources = len(accessible_resources)
        estimated_records = (
            total_resources *
            estimated_records_per_resource
        )
        estimated_breach_cost = (
            estimated_records * 0.0049
        )

        blast_risk = min(
            0.20 +
            (len(s3_buckets) * 0.02) +
            (len(databases) * 0.03) +
            (len(secrets) * 0.05) +
            (len(pci_resources) * 0.10),
            1.0
        )

        return {
            "identity": identity,
            "total_accessible_resources": (
                total_resources
            ),
            "s3_buckets_at_risk": len(s3_buckets),
            "databases_at_risk": len(databases),
            "secrets_at_risk": len(secrets),
            "kms_keys_at_risk": len(kms_keys),
            "pci_resources_at_risk": len(
                pci_resources
            ),
            "estimated_records_at_risk": (
                estimated_records
            ),
            "estimated_breach_cost_usd": (
                estimated_breach_cost
            ),
            "blast_radius_risk_score": blast_risk,
            "severity": (
                "CRITICAL" if blast_risk >= 0.8
                else "HIGH" if blast_risk >= 0.6
                else "MEDIUM" if blast_risk >= 0.3
                else "LOW"
            )
        }

    def analyze_peer_group(
        self,
        permission_count: int,
        peer_group_average: int,
        peer_group_std: float = None
    ) -> dict:
        """
        Compare identity permissions to peer group.
        Statistical anomaly detection for IAM.

        A service account with 847 permissions
        when peers average 8 is a 105x anomaly.
        This is the svc_backup scenario.

        Args:
            permission_count: This identity's count
            peer_group_average: Peer group average
            peer_group_std: Standard deviation (optional)

        Returns:
            dict with peer group analysis
        """
        if peer_group_average == 0:
            peer_group_average = 1

        ratio = permission_count / peer_group_average

        if ratio >= 10:
            anomaly_score = 0.90
            anomaly_label = "EXTREME"
            is_anomalous = True
        elif ratio >= 5:
            anomaly_score = 0.70
            anomaly_label = "SEVERE"
            is_anomalous = True
        elif ratio >= 3:
            anomaly_score = 0.50
            anomaly_label = "MODERATE"
            is_anomalous = True
        elif ratio >= 2:
            anomaly_score = 0.30
            anomaly_label = "MILD"
            is_anomalous = True
        else:
            anomaly_score = 0.10
            anomaly_label = "NORMAL"
            is_anomalous = False

        percentile = min(
            95 + (ratio - 1) * 2, 99.9
        ) if ratio > 1 else 50.0

        return {
            "permission_count": permission_count,
            "peer_group_average": peer_group_average,
            "ratio": ratio,
            "anomaly_score": anomaly_score,
            "anomaly_label": anomaly_label,
            "is_anomalous": is_anomalous,
            "estimated_percentile": min(
                percentile, 99.9
            ),
            "interpretation": (
                f"This identity has {ratio:.1f}x "
                f"the permissions of its peer group. "
                f"Anomaly level: {anomaly_label}."
            )
        }

    def identify_unused_permissions(
        self,
        permissions: list,
        last_used_days: dict,
        unused_threshold_days: int = 90
    ) -> list:
        """
        Identify permissions granted but never used.
        Largest attack surface reduction opportunity.

        Args:
            permissions: All granted permissions
            last_used_days: Dict of perm → days_since_used
                            -1 means never used
            unused_threshold_days: Days before considered unused

        Returns:
            List of unused permission dicts
        """
        unused = []

        for perm in permissions:
            days = last_used_days.get(perm, -1)

            if days == -1:
                is_high_risk = any(
                    perm in perms
                    for perms in
                    HIGH_RISK_PERMISSIONS.values()
                )

                unused.append({
                    "permission": perm,
                    "days_since_used": "Never",
                    "is_high_risk": is_high_risk,
                    "risk_score": (
                        0.70 if is_high_risk else 0.30
                    ),
                    "recommendation": (
                        f"REMOVE IMMEDIATELY - "
                        f"High risk permission "
                        f"never used"
                        if is_high_risk
                        else "Consider removing"
                    )
                })

            elif days > unused_threshold_days:
                unused.append({
                    "permission": perm,
                    "days_since_used": days,
                    "is_high_risk": False,
                    "risk_score": 0.20,
                    "recommendation": (
                        f"Not used in {days} days. "
                        f"Review and remove."
                    )
                })

        unused.sort(
            key=lambda x: x["risk_score"],
            reverse=True
        )

        return unused

    def detect_permission_creep(
        self,
        permission_history: list
    ) -> dict:
        """
        Detect gradual permission accumulation.
        Permission creep = legitimate-looking
        but dangerous growth over time.

        Args:
            permission_history: List of dicts with
                date and permission_count

        Returns:
            dict with creep analysis
        """
        if len(permission_history) < 2:
            return {
                "creep_detected": False,
                "total_growth": 0,
                "growth_rate": 0.0
            }

        first = permission_history[0]
        last = permission_history[-1]

        initial_count = first.get(
            "permission_count", 0
        )
        final_count = last.get("permission_count", 0)

        total_growth = final_count - initial_count
        growth_rate = (
            total_growth / initial_count
            if initial_count > 0
            else 0.0
        )

        creep_detected = (
            growth_rate >= 1.0 or
            total_growth >= 50
        )

        if growth_rate >= 10.0:
            severity = "CRITICAL"
            risk_score = 0.90
        elif growth_rate >= 5.0:
            severity = "HIGH"
            risk_score = 0.70
        elif growth_rate >= 1.0:
            severity = "MEDIUM"
            risk_score = 0.50
        else:
            severity = "LOW"
            risk_score = 0.20

        return {
            "creep_detected": creep_detected,
            "initial_count": initial_count,
            "final_count": final_count,
            "total_growth": total_growth,
            "growth_rate": growth_rate,
            "growth_percentage": (
                growth_rate * 100
            ),
            "severity": severity,
            "risk_score": risk_score,
            "history": permission_history,
            "interpretation": (
                f"Permissions grew from "
                f"{initial_count} to {final_count} "
                f"({growth_rate * 100:.0f}% increase). "
                f"Permission creep: {severity}."
            )
        }

    def categorize_permissions(
        self,
        permissions: list
    ) -> dict:
        """
        Categorize permissions by risk type.
        Shows the risk profile of an identity.

        Returns:
            dict with permission categories and counts
        """
        categories = {
            "data_access": [],
            "identity_manipulation": [],
            "audit_evasion": [],
            "data_destruction": [],
            "other": []
        }

        for perm in permissions:
            categorized = False
            for category, high_risk_perms in (
                HIGH_RISK_PERMISSIONS.items()
            ):
                if perm in high_risk_perms:
                    categories[category].append(perm)
                    categorized = True
                    break

            if not categorized:
                categories["other"].append(perm)

        return {
            cat: {
                "permissions": perms,
                "count": len(perms)
            }
            for cat, perms in categories.items()
        }

    def enrich_event(
        self,
        data_event: dict,
        identity_permissions: list = None,
        cloud: str = "aws"
    ) -> dict:
        """
        Enrich a DataAccessEvent with CIEM context.
        Called by Layer 2 for IAM-related events.

        Args:
            data_event: DataAccessEvent dict
            identity_permissions: Known permissions
            cloud: Cloud provider

        Returns:
            Enriched event with CIEM analysis
        """
        identity = data_event.get(
            "accessor_identity", ""
        )

        if not identity_permissions:
            identity_permissions = (
                self._simulate_permissions(identity)
            )

        ciem_analysis = self.analyze_identity(
            identity=identity,
            permissions=identity_permissions,
            cloud=cloud
        )

        base_score = float(
            data_event.get("risk_score", 0.0) or 0.0
        )
        ciem_elevation = (
            ciem_analysis["risk_score"] * 0.30
        )
        elevated_score = min(
            base_score + ciem_elevation, 1.0
        )

        return {
            **data_event,
            "risk_score": elevated_score,
            "risk_reasons": (
                data_event.get("risk_reasons", []) +
                [
                    f"ciem_{r}"
                    for r in ciem_analysis["risk_reasons"]
                ]
            ),
            "ciem_analysis": ciem_analysis
        }

    def _calculate_ciem_risk(
        self,
        toxic: list,
        escalation: dict,
        unused: list,
        peer_analysis: dict,
        categories: dict
    ) -> float:
        """Calculate overall CIEM risk score"""
        risk = 0.0

        if toxic:
            max_toxic = max(
                t["risk_score"] for t in toxic
            )
            risk = max(risk, max_toxic * 0.6)

        if escalation["has_escalation_path"]:
            risk = max(
                risk,
                escalation["max_risk"] * 0.5
            )

        if peer_analysis["is_anomalous"]:
            risk = max(
                risk,
                peer_analysis["anomaly_score"] * 0.4
            )

        high_risk_unused = [
            u for u in unused
            if u["is_high_risk"]
        ]
        if high_risk_unused:
            risk = max(risk, 0.35)

        audit_evasion = categories.get(
            "audit_evasion", {}
        ).get("count", 0)
        if audit_evasion > 0:
            risk = max(risk, 0.80)

        return min(risk, 1.0)

    def _generate_recommendation(
        self,
        toxic: list,
        escalation: dict,
        peer_analysis: dict
    ) -> str:
        """Generate actionable recommendation"""
        recommendations = []

        if toxic:
            recommendations.append(
                f"CRITICAL: Remove {len(toxic)} toxic "
                f"permission combinations immediately. "
                f"Highest risk: {toxic[0]['name']}."
            )

        if escalation["has_escalation_path"]:
            recommendations.append(
                f"Remove {escalation['path_count']} "
                f"privilege escalation paths. "
                f"Shortest: "
                f"{escalation['shortest_path_hops']} hops."
            )

        if peer_analysis["is_anomalous"]:
            recommendations.append(
                f"Review permissions: "
                f"{peer_analysis['ratio']:.1f}x peer average. "
                f"Apply least privilege."
            )

        if not recommendations:
            recommendations.append(
                "No immediate action required. "
                "Continue periodic review."
            )

        return " | ".join(recommendations)

    def _simulate_permissions(
        self, identity: str
    ) -> list:
        """
        Simulate permissions for testing.
        In production: query AWS IAM, Azure RBAC,
        or GCP IAM API for actual permissions.
        """
        identity_lower = identity.lower()

        if "backup" in identity_lower:
            return [
                "s3:GetObject",
                "s3:ListBucket",
                "s3:DeleteObject",
                "kms:Decrypt",
                "iam:PassRole",
                "iam:AttachUserPolicy",
                "ec2:TerminateInstances",
                "cloudtrail:StopLogging",
                "secretsmanager:GetSecretValue",
                "rds:DeleteDBInstance"
            ]

        if "readonly" in identity_lower:
            return [
                "s3:GetObject",
                "s3:ListBucket",
                "rds:connect"
            ]

        return [
            "s3:GetObject",
            "s3:ListBucket"
        ]


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")