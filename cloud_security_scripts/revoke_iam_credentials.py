"""
Cloud Security Scripts
AWS Revoke Leaked IAM Credentials

A standalone cloud-security automation script that
contains a compromised IAM identity — fast — when an
access key leaks or is used maliciously.

THE PROBLEM:
    An access key leaks (committed to GitHub, caught
    by Trufflehog/git-secrets) or GuardDuty sees it
    used from a Tor/malicious IP
    (UnauthorizedAccess:IAMUser/...). An attacker now
    holds valid AWS credentials. Every minute they
    can exfiltrate data or deploy rogue resources
    (e.g. crypto mining). You must kill the access
    immediately.

THE THREE ACTIONS (defense in depth):
    1. Deactivate the leaked access key
       (iam.update_access_key Status=Inactive).
    2. Attach a DenyAll inline policy to the identity
       so EVERY action is blocked - covers other keys,
       assumed roles, anything that identity can do.
    3. Revoke active sessions by denying any request
       whose token was issued before "now", which
       invalidates existing console/STS sessions.

    Why all three: deactivating one key is not enough.
    A real attacker may have created extra keys,
    assumed roles, or have a live console session. We
    kill the known key, deny-all the identity to stop
    everything else, and revoke existing sessions to
    kick out anyone currently logged in.

THE SAFETY NUANCE:
    This is HIGH impact - it locks out a real user.
    dry_run=True (default) reports the plan so a human
    confirms the identity before lockout. Rollback
    data lets a false positive be undone.

REAL EXECUTION vs SAFE TESTING:
    Makes REAL boto3 IAM calls. Tests use the boto3
    Stubber - no AWS account needed.

STANDALONE USE:
    python -m cloud_security_scripts.revoke_iam_credentials \\
        --user compromised-user --key AKIA... \\
        --execute

PROGRAMMATIC USE:
    r = RevokeIAMCredentials()
    result = r.revoke("compromised-user",
                      access_key_id="AKIA...",
                      dry_run=True)
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


# Inline policy name used for the deny-all lockout
DENY_ALL_POLICY_NAME = "AbuTech-IncidentDenyAll"

# A deny-all policy document
DENY_ALL_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Deny",
            "Action": "*",
            "Resource": "*",
        }
    ],
}


def _revoke_sessions_policy() -> dict:
    """
    Build a policy that denies any action whose
    credentials were issued before now. This
    invalidates existing sessions (the same approach
    the IAM console 'Revoke active sessions' uses).
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Deny",
                "Action": "*",
                "Resource": "*",
                "Condition": {
                    "DateLessThan": {
                        "aws:TokenIssueTime": _now()
                    }
                },
            }
        ],
    }


class RevokeIAMCredentials:
    """
    Contains a compromised IAM user by deactivating
    the leaked key, attaching a deny-all policy, and
    revoking active sessions.
    """

    def __init__(
        self,
        iam_client=None
    ):
        """
        Args:
            iam_client: optional boto3 IAM client (tests)
        """
        # IAM is a global service; region is not needed
        self.iam = iam_client or boto3.client("iam")

    # --------------------------------------------------------
    # Extract the user from a GuardDuty finding
    # --------------------------------------------------------

    def extract_compromised_user(
        self, guardduty_finding: dict
    ) -> str:
        """
        Pull the IAM user name from a GuardDuty
        UnauthorizedAccess:IAMUser finding.

        Returns the user name, or "" if not found.
        """
        if not guardduty_finding:
            return ""

        resource = guardduty_finding.get(
            "resource",
            guardduty_finding.get("Resource", {})
        )
        if not isinstance(resource, dict):
            return ""

        access_key = resource.get(
            "accessKeyDetails",
            resource.get("AccessKeyDetails", {})
        )
        if isinstance(access_key, dict):
            user = (
                access_key.get("userName")
                or access_key.get("UserName")
            )
            if user:
                return str(user)
        return ""

    # --------------------------------------------------------
    # The three containment actions
    # --------------------------------------------------------

    def deactivate_access_key(
        self,
        user_name: str,
        access_key_id: str,
        dry_run: bool = True
    ) -> dict:
        """Action 1: deactivate the leaked access key."""
        if dry_run:
            logger.info(
                f"DRY RUN: would deactivate key "
                f"{access_key_id} for {user_name}"
            )
            return {
                "step": "deactivate_access_key",
                "done": False, "dry_run": True
            }
        self.iam.update_access_key(
            UserName=user_name,
            AccessKeyId=access_key_id,
            Status="Inactive",
        )
        logger.info(
            f"Deactivated key {access_key_id} for "
            f"{user_name}"
        )
        return {
            "step": "deactivate_access_key",
            "done": True, "dry_run": False
        }

    def attach_deny_all(
        self,
        user_name: str,
        dry_run: bool = True
    ) -> dict:
        """Action 2: attach a deny-all inline policy."""
        if dry_run:
            logger.info(
                f"DRY RUN: would attach deny-all policy "
                f"to {user_name}"
            )
            return {
                "step": "attach_deny_all",
                "done": False, "dry_run": True
            }
        self.iam.put_user_policy(
            UserName=user_name,
            PolicyName=DENY_ALL_POLICY_NAME,
            PolicyDocument=json.dumps(DENY_ALL_POLICY),
        )
        logger.info(
            f"Attached deny-all policy to {user_name}"
        )
        return {
            "step": "attach_deny_all",
            "done": True, "dry_run": False
        }

    def revoke_sessions(
        self,
        user_name: str,
        dry_run: bool = True
    ) -> dict:
        """Action 3: revoke active sessions."""
        if dry_run:
            logger.info(
                f"DRY RUN: would revoke active sessions "
                f"for {user_name}"
            )
            return {
                "step": "revoke_sessions",
                "done": False, "dry_run": True
            }
        self.iam.put_user_policy(
            UserName=user_name,
            PolicyName="AbuTech-RevokeOlderSessions",
            PolicyDocument=json.dumps(
                _revoke_sessions_policy()
            ),
        )
        logger.info(
            f"Revoked active sessions for {user_name}"
        )
        return {
            "step": "revoke_sessions",
            "done": True, "dry_run": False
        }

    # --------------------------------------------------------
    # Orchestrated full revoke
    # --------------------------------------------------------

    def revoke(
        self,
        user_name: str,
        access_key_id: str = None,
        dry_run: bool = True
    ) -> dict:
        """
        Run all containment actions for a compromised
        IAM identity.

        Args:
            user_name: the compromised IAM user
            access_key_id: the leaked key (optional; if
                           omitted, key deactivation is
                           skipped and only policy
                           lockout + session revoke run)
            dry_run: if True, report the plan only

        Returns:
            Result dict with each step and rollback data.
        """
        if not user_name:
            return self._fail(
                user_name, "No user name provided"
            )

        try:
            steps = []

            if access_key_id:
                steps.append(
                    self.deactivate_access_key(
                        user_name, access_key_id,
                        dry_run=dry_run
                    )
                )

            steps.append(
                self.attach_deny_all(
                    user_name, dry_run=dry_run
                )
            )
            steps.append(
                self.revoke_sessions(
                    user_name, dry_run=dry_run
                )
            )

            return {
                "success": True,
                "action": "revoke_iam_credentials",
                "user": user_name,
                "access_key_id": access_key_id,
                "dry_run": dry_run,
                "steps": steps,
                "message": (
                    f"{'Would contain' if dry_run else 'Contained'} "
                    f"compromised identity {user_name} "
                    f"({len(steps)} action(s))"
                ),
                "rollback": {
                    "action": "restore_identity",
                    "user": user_name,
                    "reactivate_key": access_key_id,
                    "remove_policies": [
                        DENY_ALL_POLICY_NAME,
                        "AbuTech-RevokeOlderSessions",
                    ],
                },
                "event": self._build_event(
                    user_name, access_key_id, dry_run
                ),
                "executed_at": _now(),
            }

        except ClientError as e:
            return self._fail(
                user_name,
                f"AWS error: {e.response['Error']['Code']}"
            )
        except Exception as e:
            return self._fail(
                user_name, f"Error: {str(e)}"
            )

    def restore(
        self,
        user_name: str,
        access_key_id: str = None,
        dry_run: bool = True
    ) -> dict:
        """
        Roll back containment: reactivate the key and
        remove the incident policies. Use only after
        confirming a false positive.
        """
        try:
            if not dry_run:
                if access_key_id:
                    self.iam.update_access_key(
                        UserName=user_name,
                        AccessKeyId=access_key_id,
                        Status="Active",
                    )
                for policy in [
                    DENY_ALL_POLICY_NAME,
                    "AbuTech-RevokeOlderSessions",
                ]:
                    try:
                        self.iam.delete_user_policy(
                            UserName=user_name,
                            PolicyName=policy,
                        )
                    except ClientError:
                        pass
                logger.info(
                    f"Restored identity {user_name}"
                )

            return {
                "success": True,
                "action": "restore_iam_identity",
                "user": user_name,
                "dry_run": dry_run,
                "message": (
                    f"{'Would restore' if dry_run else 'Restored'} "
                    f"identity {user_name}"
                ),
                "executed_at": _now(),
            }
        except ClientError as e:
            return self._fail(
                user_name,
                f"AWS error: {e.response['Error']['Code']}"
            )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _build_event(
        self,
        user_name: str,
        access_key_id: str,
        dry_run: bool
    ) -> dict:
        """Build a standard event record"""
        return {
            "accessor_identity": (
                "AbuTech-RevokeIAMCredentials"
            ),
            "accessor_type": "automation",
            "data_store_name": user_name,
            "data_path": "iam:identity_containment",
            "data_classification": "UNKNOWN",
            "event_time": _now(),
            "source_ip": "",
            "risk_score": 0.0,
            "risk_reasons": [
                "automated_remediation",
                "compromised_iam_identity",
                f"key:{access_key_id or 'none'}",
                f"dry_run:{dry_run}",
            ],
            "source_system": "revoke_iam_credentials",
            "remediation_action": "iam_containment",
        }

    def _fail(self, user_name: str, message: str) -> dict:
        """Build a failure result"""
        logger.error(
            f"IAM revoke failed for {user_name}: "
            f"{message}"
        )
        return {
            "success": False,
            "action": "revoke_iam_credentials",
            "user": user_name,
            "message": message,
            "executed_at": _now(),
        }


def main():
    """Command-line entry point for standalone use."""
    parser = argparse.ArgumentParser(
        description=(
            "Contain a compromised IAM identity: "
            "deactivate key, deny-all, revoke sessions."
        )
    )
    parser.add_argument(
        "--user", required=True,
        help="Compromised IAM user name"
    )
    parser.add_argument(
        "--key",
        help="Leaked access key id (optional)"
    )
    parser.add_argument(
        "--restore", action="store_true",
        help="Restore instead of revoke"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually perform actions (default dry-run)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    r = RevokeIAMCredentials()
    dry = not args.execute

    if args.restore:
        result = r.restore(
            args.user, access_key_id=args.key,
            dry_run=dry
        )
    else:
        result = r.revoke(
            args.user, access_key_id=args.key,
            dry_run=dry
        )

    print(result["message"])
    return result


if __name__ == "__main__":
    main()