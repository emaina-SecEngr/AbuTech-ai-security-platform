"""
Layer 5 — Interface
SOAR Playbook Actions

Individual containment actions that
SOAR playbooks execute.

SAFETY NOTE:
    All actions are SIMULATED by default.
    Real API calls are stubbed.
    In production these connect to:
        AWS IAM, STS, S3 APIs
        Palo Alto firewall API
        AWS WAF API
        Okta Management API
        Microsoft Entra ID Graph API
        CrowdStrike API
        PagerDuty API

    dry_run=True means simulate only.
    dry_run=False means execute for real.
    Default is ALWAYS dry_run=True for safety.

WHY SIMULATED:
    Testing must never disable real accounts.
    Testing must never block real IPs.
    Each action returns a result object
    showing what WOULD happen.
    Production deployment flips dry_run to False
    only after thorough validation.

ACTION CATEGORIES:
    Identity: disable account, revoke sessions
    Network: block IP, isolate endpoint
    Data: restrict bucket, snapshot
    Notification: page, email, ticket
    Evidence: preserve logs, snapshot
"""

import logging
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class ActionResult:
    """Result of a single playbook action"""

    def __init__(
        self,
        action_name: str,
        success: bool,
        message: str,
        target: str = "",
        dry_run: bool = True,
        rollback_data: dict = None
    ):
        self.action_name = action_name
        self.success = success
        self.message = message
        self.target = target
        self.dry_run = dry_run
        self.rollback_data = rollback_data or {}
        self.executed_at = _now()

    def to_dict(self) -> dict:
        return {
            "action_name": self.action_name,
            "success": self.success,
            "message": self.message,
            "target": self.target,
            "dry_run": self.dry_run,
            "rollback_data": self.rollback_data,
            "executed_at": self.executed_at
        }


class PlaybookActions:
    """
    Library of containment actions.

    Each action returns an ActionResult.
    All actions default to dry_run mode.
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        logger.info(
            f"PlaybookActions initialized "
            f"(dry_run={dry_run})"
        )

    # ============================================================
    # IDENTITY ACTIONS
    # ============================================================

    def disable_identity(
        self,
        identity: str,
        platform: str = "aws_iam"
    ) -> ActionResult:
        """
        Disable a user or service account.

        Args:
            identity: Account to disable
            platform: aws_iam/okta/entra_id

        Returns:
            ActionResult with rollback data
        """
        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would disable "
                f"{identity} on {platform}"
            )
            return ActionResult(
                action_name="disable_identity",
                success=True,
                message=(
                    f"[SIMULATED] Identity "
                    f"{identity} would be disabled "
                    f"on {platform}"
                ),
                target=identity,
                dry_run=True,
                rollback_data={
                    "identity": identity,
                    "platform": platform,
                    "action": "re_enable"
                }
            )

        # Production code would call real API here
        # e.g. boto3 iam.update_access_key
        return ActionResult(
            action_name="disable_identity",
            success=True,
            message=(
                f"Identity {identity} disabled "
                f"on {platform}"
            ),
            target=identity,
            dry_run=False,
            rollback_data={
                "identity": identity,
                "platform": platform,
                "action": "re_enable"
            }
        )

    def revoke_sessions(
        self,
        identity: str,
        platform: str = "aws_sts"
    ) -> ActionResult:
        """Revoke all active sessions for identity"""
        msg_prefix = (
            "[SIMULATED] " if self.dry_run else ""
        )
        return ActionResult(
            action_name="revoke_sessions",
            success=True,
            message=(
                f"{msg_prefix}All sessions for "
                f"{identity} revoked on {platform}"
            ),
            target=identity,
            dry_run=self.dry_run,
            rollback_data={
                "note": "Sessions cannot be "
                        "un-revoked. User must "
                        "re-authenticate."
            }
        )

    def force_mfa_reregistration(
        self,
        identity: str
    ) -> ActionResult:
        """Force user to re-register MFA"""
        msg_prefix = (
            "[SIMULATED] " if self.dry_run else ""
        )
        return ActionResult(
            action_name="force_mfa_reregistration",
            success=True,
            message=(
                f"{msg_prefix}MFA re-registration "
                f"required for {identity}"
            ),
            target=identity,
            dry_run=self.dry_run,
            rollback_data={
                "identity": identity,
                "action": "clear_mfa_requirement"
            }
        )

    def rotate_credentials(
        self,
        identity: str,
        vault: str = "cyberark"
    ) -> ActionResult:
        """Rotate credentials in PAM vault"""
        msg_prefix = (
            "[SIMULATED] " if self.dry_run else ""
        )
        return ActionResult(
            action_name="rotate_credentials",
            success=True,
            message=(
                f"{msg_prefix}Credentials for "
                f"{identity} rotated in {vault}"
            ),
            target=identity,
            dry_run=self.dry_run,
            rollback_data={
                "note": "New credentials issued. "
                        "Old credentials invalid."
            }
        )

    # ============================================================
    # NETWORK ACTIONS
    # ============================================================

    def block_ip(
        self,
        ip_address: str,
        platform: str = "palo_alto"
    ) -> ActionResult:
        """
        Block an IP at firewall or WAF.

        Args:
            ip_address: IP to block
            platform: palo_alto/aws_waf/cloudflare

        Returns:
            ActionResult with rollback data
        """
        msg_prefix = (
            "[SIMULATED] " if self.dry_run else ""
        )
        return ActionResult(
            action_name="block_ip",
            success=True,
            message=(
                f"{msg_prefix}IP {ip_address} "
                f"blocked on {platform}"
            ),
            target=ip_address,
            dry_run=self.dry_run,
            rollback_data={
                "ip_address": ip_address,
                "platform": platform,
                "action": "unblock_ip"
            }
        )

    def isolate_endpoint(
        self,
        device_id: str,
        platform: str = "crowdstrike"
    ) -> ActionResult:
        """Isolate an endpoint from network"""
        msg_prefix = (
            "[SIMULATED] " if self.dry_run else ""
        )
        return ActionResult(
            action_name="isolate_endpoint",
            success=True,
            message=(
                f"{msg_prefix}Endpoint {device_id} "
                f"isolated via {platform}"
            ),
            target=device_id,
            dry_run=self.dry_run,
            rollback_data={
                "device_id": device_id,
                "platform": platform,
                "action": "release_endpoint"
            }
        )

    def block_domain(
        self,
        domain: str,
        platform: str = "dns_firewall"
    ) -> ActionResult:
        """Block a malicious domain"""
        msg_prefix = (
            "[SIMULATED] " if self.dry_run else ""
        )
        return ActionResult(
            action_name="block_domain",
            success=True,
            message=(
                f"{msg_prefix}Domain {domain} "
                f"blocked on {platform}"
            ),
            target=domain,
            dry_run=self.dry_run,
            rollback_data={
                "domain": domain,
                "action": "unblock_domain"
            }
        )

    # ============================================================
    # DATA ACTIONS
    # ============================================================

    def restrict_data_store(
        self,
        data_store: str,
        platform: str = "aws_s3"
    ) -> ActionResult:
        """
        Apply restrictive policy to data store.

        Args:
            data_store: Bucket or database name
            platform: aws_s3/azure_blob/rds

        Returns:
            ActionResult with rollback data
        """
        msg_prefix = (
            "[SIMULATED] " if self.dry_run else ""
        )
        return ActionResult(
            action_name="restrict_data_store",
            success=True,
            message=(
                f"{msg_prefix}Restrictive policy "
                f"applied to {data_store} "
                f"on {platform}"
            ),
            target=data_store,
            dry_run=self.dry_run,
            rollback_data={
                "data_store": data_store,
                "platform": platform,
                "action": "restore_policy"
            }
        )

    def snapshot_for_forensics(
        self,
        resource: str,
        resource_type: str = "rds"
    ) -> ActionResult:
        """Create forensic snapshot of resource"""
        msg_prefix = (
            "[SIMULATED] " if self.dry_run else ""
        )
        snapshot_id = (
            f"forensic-{resource}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        )
        return ActionResult(
            action_name="snapshot_for_forensics",
            success=True,
            message=(
                f"{msg_prefix}Forensic snapshot "
                f"{snapshot_id} created for "
                f"{resource}"
            ),
            target=resource,
            dry_run=self.dry_run,
            rollback_data={
                "snapshot_id": snapshot_id,
                "note": "Snapshot preserved "
                        "for investigation"
            }
        )

    # ============================================================
    # EVIDENCE ACTIONS
    # ============================================================

    def preserve_logs(
        self,
        log_source: str,
        time_range_hours: int = 24
    ) -> ActionResult:
        """Preserve logs for investigation"""
        msg_prefix = (
            "[SIMULATED] " if self.dry_run else ""
        )
        return ActionResult(
            action_name="preserve_logs",
            success=True,
            message=(
                f"{msg_prefix}Logs from "
                f"{log_source} preserved "
                f"({time_range_hours}h window)"
            ),
            target=log_source,
            dry_run=self.dry_run,
            rollback_data={
                "log_source": log_source,
                "note": "Logs locked from deletion"
            }
        )

    # ============================================================
    # NOTIFICATION ACTIONS
    # ============================================================

    def page_oncall(
        self,
        message: str,
        severity: str = "HIGH"
    ) -> ActionResult:
        """Page the on-call engineer"""
        msg_prefix = (
            "[SIMULATED] " if self.dry_run else ""
        )
        return ActionResult(
            action_name="page_oncall",
            success=True,
            message=(
                f"{msg_prefix}On-call paged "
                f"({severity}): {message[:100]}"
            ),
            target="oncall_engineer",
            dry_run=self.dry_run
        )

    def notify_team(
        self,
        team: str,
        message: str
    ) -> ActionResult:
        """Notify a specific team"""
        msg_prefix = (
            "[SIMULATED] " if self.dry_run else ""
        )
        return ActionResult(
            action_name="notify_team",
            success=True,
            message=(
                f"{msg_prefix}{team} notified: "
                f"{message[:100]}"
            ),
            target=team,
            dry_run=self.dry_run
        )

    def create_incident(
        self,
        title: str,
        platform: str = "sentinel"
    ) -> ActionResult:
        """Create a security incident"""
        msg_prefix = (
            "[SIMULATED] " if self.dry_run else ""
        )
        incident_id = (
            f"INC-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        )
        return ActionResult(
            action_name="create_incident",
            success=True,
            message=(
                f"{msg_prefix}Incident "
                f"{incident_id} created in "
                f"{platform}: {title[:80]}"
            ),
            target=incident_id,
            dry_run=self.dry_run,
            rollback_data={
                "incident_id": incident_id
            }
        )

    def start_compliance_timer(
        self,
        regulation: str = "PCI-DSS"
    ) -> ActionResult:
        """Start regulatory notification timer"""
        msg_prefix = (
            "[SIMULATED] " if self.dry_run else ""
        )
        return ActionResult(
            action_name="start_compliance_timer",
            success=True,
            message=(
                f"{msg_prefix}{regulation} "
                f"notification timer started"
            ),
            target=regulation,
            dry_run=self.dry_run
        )