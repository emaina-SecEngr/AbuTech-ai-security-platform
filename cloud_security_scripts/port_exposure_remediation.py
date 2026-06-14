"""
Cloud Security Scripts
AWS Port Exposure Remediation

A standalone cloud-security automation script that
closes dangerously exposed ports — security group
ingress rules that allow 0.0.0.0/0 (the whole
internet) on sensitive ports like 22 (SSH) or
3389 (RDP).

THE PROBLEM:
    An SG rule allowing 0.0.0.0/0 on port 22 or 3389
    is an open door. Internet-wide scanners find it
    within minutes and begin brute-forcing. AWS
    Config and Security Hub flag these constantly.

THE FIX (surgical, not destructive):
    1. Inspect the security group's ingress rules.
    2. Find rules that allow 0.0.0.0/0 on a sensitive
       port.
    3. Revoke ONLY those offending rules - other
       legitimate rules stay intact.
    4. Capture each removed rule as rollback data.
    5. Emit an owner-alert event.

THE SAFETY NUANCE:
    We never delete the whole security group or all
    its rules. We surgically remove only the
    offending ingress rule, and we capture the exact
    rule so it can be restored if it turns out to be
    load-bearing (e.g. a misconfigured but needed
    bastion). Dry-run shows exactly what WOULD be
    removed so a human can confirm.

REAL EXECUTION vs SAFE TESTING:
    Makes REAL boto3 calls. dry_run=True (default)
    reports only. Tests use the boto3 Stubber - no
    AWS account needed.

STANDALONE USE:
    python -m cloud_security_scripts.port_exposure_remediation \\
        --sg sg-0abc123 --region us-east-1 --execute

PROGRAMMATIC USE:
    r = PortExposureRemediation(region="us-east-1")
    result = r.remediate("sg-0abc123", dry_run=True)
"""

import argparse
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


# Ports considered dangerous when open to the world
DEFAULT_SENSITIVE_PORTS = {
    22: "SSH",
    3389: "RDP",
    5432: "PostgreSQL",
    3306: "MySQL",
    1433: "MSSQL",
    27017: "MongoDB",
    6379: "Redis",
    9200: "Elasticsearch",
}

# CIDRs that mean "the whole internet"
WORLD_CIDRS = {"0.0.0.0/0", "::/0"}


class PortExposureRemediation:
    """
    Finds and removes security-group ingress rules
    that expose sensitive ports to the entire internet.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        ec2_client=None,
        sensitive_ports: dict = None
    ):
        """
        Args:
            region: AWS region
            ec2_client: optional boto3 client (tests)
            sensitive_ports: override the default
                             {port: name} map
        """
        self.region = region
        self.ec2 = ec2_client or boto3.client(
            "ec2", region_name=region
        )
        self.sensitive_ports = (
            sensitive_ports or DEFAULT_SENSITIVE_PORTS
        )

    def find_exposed_rules(
        self, sg_id: str
    ) -> list:
        """
        Return the list of offending ingress rules for
        a security group: those allowing a world CIDR
        on a sensitive port.

        Each item is the IpPermission dict plus the
        matched port/cidr for clarity.
        """
        response = self.ec2.describe_security_groups(
            GroupIds=[sg_id]
        )
        groups = response.get("SecurityGroups", [])
        if not groups:
            return []

        offending = []
        for perm in groups[0].get("IpPermissions", []):
            from_port = perm.get("FromPort")
            to_port = perm.get("ToPort")

            # which world CIDRs are in this rule
            world_ranges = [
                r for r in perm.get("IpRanges", [])
                if r.get("CidrIp") in WORLD_CIDRS
            ]
            world_v6 = [
                r for r in perm.get("Ipv6Ranges", [])
                if r.get("CidrIpv6") in WORLD_CIDRS
            ]
            if not world_ranges and not world_v6:
                continue

            # which sensitive ports this rule covers
            matched_ports = self._matched_sensitive_ports(
                from_port, to_port
            )
            if not matched_ports:
                continue

            offending.append({
                "permission": perm,
                "from_port": from_port,
                "to_port": to_port,
                "matched_ports": matched_ports,
                "world_cidrs": (
                    [r["CidrIp"] for r in world_ranges]
                    + [r["CidrIpv6"] for r in world_v6]
                ),
            })

        return offending

    def remediate(
        self,
        sg_id: str,
        dry_run: bool = True
    ) -> dict:
        """
        Remove all offending ingress rules from a
        security group.

        Args:
            sg_id: security group id
            dry_run: if True, report only

        Returns:
            Result dict with the removed rules as
            rollback data and an owner-alert event.
        """
        try:
            offending = self.find_exposed_rules(sg_id)

            if not offending:
                return {
                    "success": True,
                    "action": "port_exposure_remediation",
                    "security_group": sg_id,
                    "dry_run": dry_run,
                    "exposed_rules_found": 0,
                    "message": (
                        f"No internet-exposed sensitive "
                        f"ports found on {sg_id}"
                    ),
                    "executed_at": _now(),
                }

            removed = []
            for item in offending:
                perm = item["permission"]
                ports = item["matched_ports"]

                if dry_run:
                    logger.info(
                        f"DRY RUN: would revoke ingress "
                        f"on {sg_id} for ports {ports} "
                        f"from {item['world_cidrs']}"
                    )
                else:
                    self.ec2.revoke_security_group_ingress(
                        GroupId=sg_id,
                        IpPermissions=[
                            self._clean_permission(perm)
                        ]
                    )
                    logger.info(
                        f"REMOVED exposed ingress on "
                        f"{sg_id} for ports {ports}"
                    )

                removed.append({
                    "ports": ports,
                    "world_cidrs": item["world_cidrs"],
                    "rule": self._clean_permission(perm),
                })

            return {
                "success": True,
                "action": "port_exposure_remediation",
                "security_group": sg_id,
                "dry_run": dry_run,
                "exposed_rules_found": len(offending),
                "rules_removed": len(removed),
                "message": (
                    f"{'Would remove' if dry_run else 'Removed'} "
                    f"{len(removed)} internet-exposed "
                    f"rule(s) from {sg_id}"
                ),
                "rollback": {
                    "action": "restore_ingress_rules",
                    "security_group": sg_id,
                    "rules": [r["rule"] for r in removed],
                },
                "owner_alert": self._build_owner_alert(
                    sg_id, removed
                ),
                "event": self._build_event(
                    sg_id, removed, dry_run
                ),
                "executed_at": _now(),
            }

        except ClientError as e:
            return self._fail(
                sg_id,
                f"AWS error: {e.response['Error']['Code']}"
            )
        except Exception as e:
            return self._fail(sg_id, f"Error: {str(e)}")

    def restore(
        self,
        sg_id: str,
        rule: dict,
        dry_run: bool = True
    ) -> dict:
        """
        Restore a previously-removed ingress rule
        (rollback).
        """
        try:
            if not dry_run:
                self.ec2.authorize_security_group_ingress(
                    GroupId=sg_id,
                    IpPermissions=[rule]
                )
                logger.info(
                    f"RESTORED ingress rule on {sg_id}"
                )
            return {
                "success": True,
                "action": "port_exposure_restore",
                "security_group": sg_id,
                "dry_run": dry_run,
                "message": (
                    f"{'Would restore' if dry_run else 'Restored'} "
                    f"ingress rule on {sg_id}"
                ),
                "executed_at": _now(),
            }
        except ClientError as e:
            return self._fail(
                sg_id,
                f"AWS error: {e.response['Error']['Code']}"
            )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _matched_sensitive_ports(
        self, from_port, to_port
    ) -> list:
        """
        Return the sensitive ports covered by a rule's
        port range. A rule with from/to None (all
        traffic) matches every sensitive port.
        """
        # All-traffic rule (no port range) exposes all
        if from_port is None and to_port is None:
            return sorted(self.sensitive_ports.keys())

        if from_port is None or to_port is None:
            return []

        matched = []
        for port in self.sensitive_ports:
            if from_port <= port <= to_port:
                matched.append(port)
        return sorted(matched)

    def _clean_permission(self, perm: dict) -> dict:
        """
        Build a clean IpPermission for revoke/authorize,
        keeping only the world CIDR ranges so we remove
        exactly the dangerous part.
        """
        clean = {
            "IpProtocol": perm.get("IpProtocol", "tcp"),
        }
        if perm.get("FromPort") is not None:
            clean["FromPort"] = perm["FromPort"]
        if perm.get("ToPort") is not None:
            clean["ToPort"] = perm["ToPort"]

        world_v4 = [
            {"CidrIp": r["CidrIp"]}
            for r in perm.get("IpRanges", [])
            if r.get("CidrIp") in WORLD_CIDRS
        ]
        if world_v4:
            clean["IpRanges"] = world_v4

        world_v6 = [
            {"CidrIpv6": r["CidrIpv6"]}
            for r in perm.get("Ipv6Ranges", [])
            if r.get("CidrIpv6") in WORLD_CIDRS
        ]
        if world_v6:
            clean["Ipv6Ranges"] = world_v6

        return clean

    def _build_owner_alert(
        self, sg_id: str, removed: list
    ) -> dict:
        """Build an alert payload for the resource owner"""
        port_list = sorted({
            p for r in removed for p in r["ports"]
        })
        return {
            "to": "resource_owner",
            "subject": (
                f"Security remediation on {sg_id}"
            ),
            "summary": (
                f"Removed internet-exposed ingress on "
                f"ports {port_list} from security group "
                f"{sg_id}. If this access was required, "
                f"reconfigure it scoped to specific IPs."
            ),
            "ports_closed": port_list,
        }

    def _build_event(
        self, sg_id: str, removed: list, dry_run: bool
    ) -> dict:
        """Build a standard event record"""
        port_list = sorted({
            p for r in removed for p in r["ports"]
        })
        return {
            "accessor_identity": (
                "AbuTech-PortExposureRemediation"
            ),
            "accessor_type": "automation",
            "data_store_name": sg_id,
            "data_path": "ec2:security_group_ingress",
            "data_classification": "UNKNOWN",
            "event_time": _now(),
            "source_ip": "",
            "risk_score": 0.0,
            "risk_reasons": [
                "automated_remediation",
                "internet_exposed_port_closed",
                f"ports:{port_list}",
                f"dry_run:{dry_run}",
            ],
            "source_system": "port_exposure_remediation",
            "remediation_action": "close_exposed_port",
        }

    def _fail(self, sg_id: str, message: str) -> dict:
        """Build a failure result"""
        logger.error(
            f"Port remediation failed for {sg_id}: "
            f"{message}"
        )
        return {
            "success": False,
            "action": "port_exposure_remediation",
            "security_group": sg_id,
            "message": message,
            "executed_at": _now(),
        }


def main():
    """Command-line entry point for standalone use."""
    parser = argparse.ArgumentParser(
        description=(
            "Remove internet-exposed sensitive-port "
            "rules from a security group."
        )
    )
    parser.add_argument(
        "--sg", required=True,
        help="Security group id"
    )
    parser.add_argument(
        "--region", default="us-east-1",
        help="AWS region"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually remove rules (default dry-run)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    r = PortExposureRemediation(region=args.region)
    result = r.remediate(args.sg, dry_run=not args.execute)
    print(result["message"])
    return result


if __name__ == "__main__":
    main()