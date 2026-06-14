"""
Cloud Security Scripts
AWS Revoke Stolen Temporary Tokens (STS Session Abuse)

A standalone cloud-security automation script that
neutralizes stolen EC2 instance-role credentials —
temporary STS tokens an attacker exfiltrated and is
using from outside AWS.

THE ATTACK:
    An EC2 instance has an IAM role (instance profile).
    AWS auto-issues temporary STS credentials for that
    role at the instance metadata endpoint. An attacker
    who compromises the instance steals those temporary
    tokens and uses them from outside AWS. GuardDuty
    flags this as InstanceCredentialExfiltration.

WHY YOU CAN'T JUST DELETE THE TOKEN:
    STS tokens are temporary and self-expiring - there
    is no "delete token" API, and you cannot deactivate
    them like a long-term access key. So how do you
    kill a credential you cannot delete?

THE TECHNIQUE:
    Attach an inline policy to the ROLE that denies all
    actions where aws:TokenIssueTime is earlier than
    the moment of revocation. Every EXISTING token
    (issued in the past) is instantly rejected, while
    NEW tokens issued after the host is cleaned still
    work. This is AWS's AWSRevokeOlderSessions pattern.

HOW THIS DIFFERS FROM revoke_iam_credentials:
    That script targets a USER with a long-term ACCESS
    KEY - you deactivate the key. This script targets a
    ROLE with TEMPORARY tokens - there is no key to
    deactivate, so you invalidate sessions by time.
    Key vs token. User vs role. Deactivate vs
    time-based revoke.

THE ELEGANCE:
    It kills only the stolen sessions without breaking
    the role. Once the compromised host is cleaned and
    receives fresh tokens, those work fine because they
    are issued after the revocation timestamp - so the
    legitimate workload is never taken offline.

REAL EXECUTION vs SAFE TESTING:
    Makes REAL boto3 IAM calls. dry_run=True (default)
    reports only. Tests use the boto3 Stubber - no AWS
    account needed.

STANDALONE USE:
    python -m cloud_security_scripts.revoke_sts_sessions \\
        --role compromised-ec2-role --execute

PROGRAMMATIC USE:
    r = RevokeSTSSessions()
    result = r.revoke_role_sessions(
        "compromised-ec2-role", dry_run=True)
"""

import argparse
import json
import logging
from datetime import datetime
from datetime import timezone

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# Inline policy name attached to the role to revoke
# sessions (matches AWS's own naming convention).
REVOKE_POLICY_NAME = "AWSRevokeOlderSessions"


def _revoke_older_sessions_policy(
    revocation_time: str
) -> dict:
    """
    Build the policy that denies any action whose
    credentials were issued before the revocation
    timestamp. Existing (stolen) tokens are rejected;
    tokens issued later still work.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AbuTechRevokeOlderSessions",
                "Effect": "Deny",
                "Action": "*",
                "Resource": "*",
                "Condition": {
                    "DateLessThan": {
                        "aws:TokenIssueTime": (
                            revocation_time
                        )
                    }
                },
            }
        ],
    }


class RevokeSTSSessions:
    """
    Neutralizes stolen EC2 instance-role STS tokens by
    attaching a time-based deny policy to the role.
    """

    def __init__(self, iam_client=None):
        """
        Args:
            iam_client: optional boto3 IAM client (tests)
        """
        # IAM is a global service; no region needed.
        self.iam = iam_client or boto3.client("iam")

    # --------------------------------------------------------
    # Extract the role from a GuardDuty finding
    # --------------------------------------------------------

    def extract_compromised_role(
        self, guardduty_finding: dict
    ) -> str:
        """
        Pull the IAM role name from a GuardDuty
        InstanceCredentialExfiltration finding.

        The role appears in the access key details
        (the instance profile's role) or in the
        resource's IAM instance profile.

        Returns the role name, or "" if not found.
        """
        if not guardduty_finding:
            return ""

        resource = guardduty_finding.get(
            "resource",
            guardduty_finding.get("Resource", {})
        )
        if not isinstance(resource, dict):
            return ""

        # Path 1: accessKeyDetails.userName carries the
        # role's session name for assumed-role creds.
        access_key = resource.get(
            "accessKeyDetails",
            resource.get("AccessKeyDetails", {})
        )
        if isinstance(access_key, dict):
            principal = (
                access_key.get("userName")
                or access_key.get("UserName")
                or ""
            )
            # Assumed-role identities look like
            # "role-name" or contain a ':' session.
            if principal:
                # Strip any session suffix
                return str(principal).split(":")[0]

        # Path 2: instance details -> IAM instance profile
        instance = resource.get(
            "instanceDetails",
            resource.get("InstanceDetails", {})
        )
        if isinstance(instance, dict):
            profile = instance.get(
                "iamInstanceProfile",
                instance.get("IamInstanceProfile", {})
            )
            if isinstance(profile, dict):
                arn = (
                    profile.get("arn")
                    or profile.get("Arn")
                    or ""
                )
                # arn:aws:iam::acct:instance-profile/NAME
                if "/" in arn:
                    return arn.split("/")[-1]

        return ""

    # --------------------------------------------------------
    # Revoke sessions
    # --------------------------------------------------------

    def revoke_role_sessions(
        self,
        role_name: str,
        dry_run: bool = True
    ) -> dict:
        """
        Attach the time-based deny policy to the role,
        invalidating all currently-issued STS tokens.

        Args:
            role_name: the compromised IAM role
            dry_run: if True, report only

        Returns:
            Result dict with rollback data and an event.
        """
        if not role_name:
            return self._fail(
                role_name, "No role name provided"
            )

        revocation_time = _now()

        try:
            if dry_run:
                logger.info(
                    f"DRY RUN: would attach revoke-older-"
                    f"sessions policy to role "
                    f"{role_name} (cutoff "
                    f"{revocation_time})"
                )
            else:
                self.iam.put_role_policy(
                    RoleName=role_name,
                    PolicyName=REVOKE_POLICY_NAME,
                    PolicyDocument=json.dumps(
                        _revoke_older_sessions_policy(
                            revocation_time
                        )
                    ),
                )
                logger.info(
                    f"Revoked older sessions for role "
                    f"{role_name} (cutoff "
                    f"{revocation_time})"
                )

            return {
                "success": True,
                "action": "revoke_sts_sessions",
                "role": role_name,
                "revocation_time": revocation_time,
                "dry_run": dry_run,
                "message": (
                    f"{'Would revoke' if dry_run else 'Revoked'} "
                    f"stolen sessions for role "
                    f"{role_name}; tokens issued before "
                    f"{revocation_time} are now denied"
                ),
                "rollback": {
                    "action": "remove_revoke_policy",
                    "role": role_name,
                    "policy_name": REVOKE_POLICY_NAME,
                },
                "event": self._build_event(
                    role_name, revocation_time, dry_run
                ),
                "executed_at": revocation_time,
            }

        except ClientError as e:
            return self._fail(
                role_name,
                f"AWS error: {e.response['Error']['Code']}"
            )
        except Exception as e:
            return self._fail(
                role_name, f"Error: {str(e)}"
            )

    def restore(
        self,
        role_name: str,
        dry_run: bool = True
    ) -> dict:
        """
        Remove the revoke-sessions policy from the role
        (rollback). Use only after confirming a false
        positive - existing legitimate sessions will
        have expired naturally by then anyway.
        """
        try:
            if not dry_run:
                try:
                    self.iam.delete_role_policy(
                        RoleName=role_name,
                        PolicyName=REVOKE_POLICY_NAME,
                    )
                except ClientError:
                    pass
                logger.info(
                    f"Removed revoke policy from role "
                    f"{role_name}"
                )
            return {
                "success": True,
                "action": "restore_sts_sessions",
                "role": role_name,
                "dry_run": dry_run,
                "message": (
                    f"{'Would remove' if dry_run else 'Removed'} "
                    f"revoke policy from role {role_name}"
                ),
                "executed_at": _now(),
            }
        except ClientError as e:
            return self._fail(
                role_name,
                f"AWS error: {e.response['Error']['Code']}"
            )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _build_event(
        self,
        role_name: str,
        revocation_time: str,
        dry_run: bool
    ) -> dict:
        """Build a standard event record"""
        return {
            "accessor_identity": (
                "AbuTech-RevokeSTSSessions"
            ),
            "accessor_type": "automation",
            "data_store_name": role_name,
            "data_path": "iam:sts_session_revocation",
            "data_classification": "UNKNOWN",
            "event_time": revocation_time,
            "source_ip": "",
            "risk_score": 0.0,
            "risk_reasons": [
                "automated_remediation",
                "stolen_sts_token_revoked",
                f"cutoff:{revocation_time}",
                f"dry_run:{dry_run}",
            ],
            "source_system": "revoke_sts_sessions",
            "remediation_action": (
                "sts_session_revocation"
            ),
        }

    def _fail(self, role_name: str, message: str) -> dict:
        """Build a failure result"""
        logger.error(
            f"STS session revoke failed for "
            f"{role_name}: {message}"
        )
        return {
            "success": False,
            "action": "revoke_sts_sessions",
            "role": role_name,
            "message": message,
            "executed_at": _now(),
        }


def main():
    """Command-line entry point for standalone use."""
    parser = argparse.ArgumentParser(
        description=(
            "Revoke stolen EC2 instance-role STS tokens "
            "via a time-based deny policy."
        )
    )
    parser.add_argument(
        "--role", required=True,
        help="Compromised IAM role name"
    )
    parser.add_argument(
        "--restore", action="store_true",
        help="Remove the revoke policy instead"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually apply (default is dry-run)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    r = RevokeSTSSessions()
    dry = not args.execute

    if args.restore:
        result = r.restore(args.role, dry_run=dry)
    else:
        result = r.revoke_role_sessions(
            args.role, dry_run=dry
        )

    print(result["message"])
    return result


if __name__ == "__main__":
    main()