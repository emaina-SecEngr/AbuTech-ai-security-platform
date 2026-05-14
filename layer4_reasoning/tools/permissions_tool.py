"""
Layer 4 — Agent Tools
Tool 5: User Permissions Lookup

Gets IAM permissions for a user or
service account.

In production: queries SailPoint ISC or Okta.
In development: uses rule-based heuristics.

Used by InvestigationAgent to understand
blast radius and permission violations.

USAGE BY AGENTS:
    result = get_user_permissions("svc_backup")
    print(result["permissions"])   # ["s3:GetObject"]
    print(result["violations"])    # ["accessing wrong bucket"]
    print(result["risk_level"])    # "HIGH"
"""

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# Known service account permission profiles
# In production: query SailPoint ISC API
SERVICE_ACCOUNT_PROFILES = {
    "svc_backup": {
        "type": "service_account",
        "expected_permissions": [
            "s3:GetObject",
            "s3:PutObject",
            "s3:ListBucket"
        ],
        "allowed_resources": [
            "prod-backup-data",
            "staging-backup-data",
            "dev-backup-data"
        ],
        "risk_level": "MEDIUM",
        "notes": (
            "Backup service account. Should only "
            "access backup buckets. Any access to "
            "customer or PCI data is a violation."
        )
    },
    "svc_etl": {
        "type": "service_account",
        "expected_permissions": [
            "s3:GetObject",
            "s3:ListBucket",
            "glue:StartJobRun"
        ],
        "allowed_resources": [
            "data-lake-raw",
            "data-lake-processed"
        ],
        "risk_level": "MEDIUM",
        "notes": (
            "ETL pipeline service account. "
            "Read access to data lake only."
        )
    },
    "admin": {
        "type": "privileged_user",
        "expected_permissions": ["*"],
        "allowed_resources": ["*"],
        "risk_level": "CRITICAL",
        "notes": (
            "Full administrator access. "
            "Maximum blast radius. "
            "All actions should be reviewed."
        )
    }
}


def get_user_permissions(
    user: str,
    accessed_resource: str = None,
    api_key: str = None
) -> dict:
    """
    Get IAM permissions for a user or
    service account.

    Args:
        user: Username or service account name
        accessed_resource: Resource being accessed
                           (for violation checking)
        api_key: SailPoint/Okta API key (optional)

    Returns:
        dict with:
            user: str
            type: str (user/service_account)
            permissions: list of granted permissions
            allowed_resources: list
            risk_level: str
            violations: list of policy violations
            is_privileged: bool
            summary: str
    """
    if not user:
        return _empty_result(user)

    # Try real IAM API first
    if api_key or os.getenv("SAILPOINT_API_KEY"):
        result = _query_sailpoint(user, api_key)
        if result:
            if accessed_resource:
                result["violations"] = (
                    _check_violations(
                        result, accessed_resource
                    )
                )
            return result

    # Fall back to rule-based
    return _rule_based_lookup(
        user, accessed_resource
    )


def _query_sailpoint(
    user: str,
    api_key: str = None
) -> dict:
    """
    Query SailPoint ISC for user permissions.
    Returns None if unavailable.
    """
    # In production: real SailPoint API call
    # For now return None to use rule-based
    return None


def _rule_based_lookup(
    user: str,
    accessed_resource: str = None
) -> dict:
    """
    Rule-based permissions lookup.
    Uses known profiles and heuristics.
    """
    user_lower = user.lower()

    # Check known profiles
    for profile_name, profile in (
        SERVICE_ACCOUNT_PROFILES.items()
    ):
        if profile_name in user_lower:
            result = {
                "user": user,
                "type": profile["type"],
                "permissions": profile[
                    "expected_permissions"
                ],
                "allowed_resources": profile[
                    "allowed_resources"
                ],
                "risk_level": profile["risk_level"],
                "is_privileged": (
                    profile["risk_level"] == "CRITICAL"
                ),
                "violations": [],
                "notes": profile["notes"],
                "source": "rule_based",
                "summary": ""
            }

            if accessed_resource:
                result["violations"] = (
                    _check_violations(
                        result, accessed_resource
                    )
                )

            result["summary"] = _build_summary(
                result
            )
            return result

    # Unknown user — generic profile
    is_privileged = any(
        p in user_lower
        for p in ["admin", "root", "sudo",
                  "superuser", "dba"]
    )

    risk_level = "HIGH" if is_privileged else "LOW"
    permissions = (
        ["*"] if is_privileged
        else ["limited_access"]
    )

    result = {
        "user": user,
        "type": (
            "privileged_user"
            if is_privileged
            else "standard_user"
        ),
        "permissions": permissions,
        "allowed_resources": (
            ["*"] if is_privileged else []
        ),
        "risk_level": risk_level,
        "is_privileged": is_privileged,
        "violations": [],
        "notes": (
            "Profile not found in IAM system. "
            "Manual verification recommended."
        ),
        "source": "rule_based",
        "summary": ""
    }

    if accessed_resource:
        result["violations"] = _check_violations(
            result, accessed_resource
        )

    result["summary"] = _build_summary(result)
    return result


def _check_violations(
    permission_data: dict,
    accessed_resource: str
) -> List[str]:
    """
    Check if resource access violates
    permission policy.
    """
    violations = []
    allowed = permission_data.get(
        "allowed_resources", []
    )

    # Wildcard = no violations
    if "*" in allowed:
        return violations

    # Check if resource is in allowed list
    resource_allowed = any(
        allowed_res.lower() in accessed_resource.lower()
        or accessed_resource.lower() in allowed_res.lower()
        for allowed_res in allowed
    )

    if not resource_allowed and allowed:
        violations.append(
            f"POLICY VIOLATION: {permission_data['user']} "
            f"accessed '{accessed_resource}' but is only "
            f"authorized for: {', '.join(allowed)}"
        )

    # Check for sensitive data access
    resource_lower = accessed_resource.lower()
    if any(
        kw in resource_lower
        for kw in ["pci", "card", "payment",
                   "customer", "phi", "health"]
    ):
        if permission_data["type"] == "service_account":
            violations.append(
                f"SENSITIVE DATA VIOLATION: "
                f"Service account {permission_data['user']} "
                f"accessed sensitive resource "
                f"'{accessed_resource}'. "
                f"Service accounts should not access "
                f"customer or regulated data."
            )

    return violations


def _build_summary(permission_data: dict) -> str:
    """Build human readable permissions summary"""
    user = permission_data["user"]
    user_type = permission_data["type"]
    risk = permission_data["risk_level"]
    violations = permission_data.get("violations", [])

    summary = (
        f"{user} ({user_type}): "
        f"Risk level {risk}. "
        f"Permissions: "
        f"{', '.join(permission_data['permissions'][:3])}. "
    )

    if permission_data.get("allowed_resources"):
        allowed = permission_data["allowed_resources"]
        if "*" not in allowed:
            summary += (
                f"Authorized resources: "
                f"{', '.join(allowed[:3])}. "
            )

    if violations:
        summary += (
            f"⚠️  {len(violations)} POLICY VIOLATION(S) DETECTED. "
        )
        summary += violations[0]

    return summary


def _empty_result(user: str) -> dict:
    """Return empty result for invalid input"""
    return {
        "user": user or "unknown",
        "type": "unknown",
        "permissions": [],
        "allowed_resources": [],
        "risk_level": "UNKNOWN",
        "is_privileged": False,
        "violations": [],
        "notes": "No user provided.",
        "source": "rule_based",
        "summary": "No user provided for lookup."
    }