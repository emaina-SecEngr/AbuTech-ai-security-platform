"""
Cloud Security Scripts
AWS Perimeter Block (Malicious IP Blocking)

A standalone cloud-security automation script that
blocks a malicious source IP at the cloud edge —
before it reaches your resources.

THE TRIGGER:
    A GuardDuty finding identifies a malicious IP:
        UnauthorizedAccess:EC2/SSHBruteForce
        Recon:EC2/Portscan
    This script extracts that IP and blocks it.

TWO ENFORCEMENT POINTS (the key design decision):

    NETWORK ACL (NACL) - Layer 3/4:
        Adds a DENY rule to a VPC subnet's NACL.
        Blocks ALL traffic - SSH, RDP, any port.
        The RIGHT tool for SSH brute-force and port
        scans, which are network-layer attacks.
        boto3: ec2.create_network_acl_entry()

    WAF IP SET - Layer 7:
        Adds the IP to a WAF IP Set referenced by a
        block rule. Blocks HTTP/HTTPS only.
        The RIGHT tool for web-layer attacks against
        an ALB, CloudFront, or API Gateway.
        boto3: wafv2.update_ip_set()

    WHY BOTH: match the enforcement point to the
    attack layer. Blocking SSH brute-force at a WAF
    would do nothing - WAF only inspects HTTP. A
    senior engineer picks the right layer.

REAL EXECUTION vs SAFE TESTING:
    Makes REAL boto3 calls. dry_run=True (default)
    reports what WOULD be blocked. Tests use the
    boto3 Stubber - no AWS account needed.

STANDALONE USE:
    python -m cloud_security_scripts.perimeter_block \\
        --ip 185.220.101.45 --nacl acl-0abc \\
        --region us-east-1 --execute

PROGRAMMATIC USE:
    pb = PerimeterBlock(region="us-east-1")
    ip = pb.extract_malicious_ip(guardduty_finding)
    pb.block_ip_nacl(ip, "acl-0abc", dry_run=True)
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


class PerimeterBlock:
    """
    Blocks malicious source IPs at the cloud edge via
    VPC Network ACL deny rules (L3/4) or WAF IP Sets
    (L7).
    """

    def __init__(
        self,
        region: str = "us-east-1",
        ec2_client=None,
        wafv2_client=None
    ):
        """
        Args:
            region: AWS region
            ec2_client: optional boto3 EC2 client (tests)
            wafv2_client: optional boto3 WAFv2 client (tests)
        """
        self.region = region
        self.ec2 = ec2_client or boto3.client(
            "ec2", region_name=region
        )
        # WAF client is created lazily only if needed,
        # so EC2-only use does not require it.
        self._wafv2 = wafv2_client

    @property
    def wafv2(self):
        if self._wafv2 is None:
            self._wafv2 = boto3.client(
                "wafv2", region_name=self.region
            )
        return self._wafv2

    # --------------------------------------------------------
    # IP extraction from a GuardDuty finding
    # --------------------------------------------------------

    def extract_malicious_ip(
        self, guardduty_finding: dict
    ) -> str:
        """
        Extract the attacker's source IP from a
        GuardDuty finding's service.action block.

        Returns the IP string, or "" if none found.
        """
        if not guardduty_finding:
            return ""

        service = guardduty_finding.get(
            "service",
            guardduty_finding.get("Service", {})
        )
        if not isinstance(service, dict):
            return ""

        action = service.get(
            "action", service.get("Action", {})
        )
        if not isinstance(action, dict):
            return ""

        for key in [
            "networkConnectionAction",
            "awsApiCallAction",
            "portProbeAction",
            "NetworkConnectionAction",
            "AwsApiCallAction",
            "PortProbeAction",
        ]:
            act = action.get(key, {})
            if isinstance(act, dict):
                # portProbeAction nests a details list
                if "portProbeDetails" in act:
                    details = act.get(
                        "portProbeDetails", []
                    )
                    if details and isinstance(
                        details, list
                    ):
                        remote = details[0].get(
                            "remoteIpDetails", {}
                        )
                        ip = remote.get("ipAddressV4")
                        if ip:
                            return str(ip)
                remote = act.get(
                    "remoteIpDetails",
                    act.get("RemoteIpDetails", {})
                )
                if isinstance(remote, dict):
                    ip = (
                        remote.get("ipAddressV4")
                        or remote.get("IpAddressV4")
                    )
                    if ip:
                        return str(ip)

        return ""

    # --------------------------------------------------------
    # NACL block (Layer 3/4)
    # --------------------------------------------------------

    def block_ip_nacl(
        self,
        ip: str,
        nacl_id: str,
        rule_number: int = 100,
        dry_run: bool = True
    ) -> dict:
        """
        Block an IP at a VPC Network ACL with a DENY
        rule covering all traffic.

        Args:
            ip: the malicious IP (a /32 is applied)
            nacl_id: the network ACL id
            rule_number: NACL rule number (lower =
                         higher priority). NACLs allow
                         a limited number of rules.
            dry_run: if True, report only

        Returns:
            Result dict with rollback data.
        """
        if not self._valid_ip(ip):
            return self._fail(
                ip, "nacl", "Invalid IP address"
            )

        cidr = f"{ip}/32"

        try:
            if dry_run:
                logger.info(
                    f"DRY RUN: would add DENY rule "
                    f"#{rule_number} for {cidr} to "
                    f"NACL {nacl_id}"
                )
            else:
                self.ec2.create_network_acl_entry(
                    NetworkAclId=nacl_id,
                    RuleNumber=rule_number,
                    Protocol="-1",          # all protocols
                    RuleAction="deny",
                    Egress=False,           # inbound
                    CidrBlock=cidr,
                )
                logger.info(
                    f"BLOCKED {cidr} at NACL "
                    f"{nacl_id} rule #{rule_number}"
                )

            return {
                "success": True,
                "action": "perimeter_block_nacl",
                "ip": ip,
                "cidr": cidr,
                "nacl_id": nacl_id,
                "rule_number": rule_number,
                "layer": "L3/4 (network)",
                "dry_run": dry_run,
                "message": (
                    f"{'Would block' if dry_run else 'Blocked'} "
                    f"{cidr} at NACL {nacl_id}"
                ),
                "rollback": {
                    "action": "delete_nacl_entry",
                    "nacl_id": nacl_id,
                    "rule_number": rule_number,
                    "egress": False,
                },
                "event": self._build_event(
                    ip, "nacl", dry_run
                ),
                "executed_at": _now(),
            }

        except ClientError as e:
            return self._fail(
                ip, "nacl",
                f"AWS error: {e.response['Error']['Code']}"
            )

    def unblock_ip_nacl(
        self,
        nacl_id: str,
        rule_number: int,
        dry_run: bool = True
    ) -> dict:
        """Remove a NACL deny rule (rollback)."""
        try:
            if not dry_run:
                self.ec2.delete_network_acl_entry(
                    NetworkAclId=nacl_id,
                    RuleNumber=rule_number,
                    Egress=False,
                )
                logger.info(
                    f"UNBLOCKED NACL {nacl_id} rule "
                    f"#{rule_number}"
                )
            return {
                "success": True,
                "action": "perimeter_unblock_nacl",
                "nacl_id": nacl_id,
                "rule_number": rule_number,
                "dry_run": dry_run,
                "message": (
                    f"{'Would remove' if dry_run else 'Removed'} "
                    f"NACL rule #{rule_number}"
                ),
                "executed_at": _now(),
            }
        except ClientError as e:
            return self._fail(
                "", "nacl",
                f"AWS error: {e.response['Error']['Code']}"
            )

    # --------------------------------------------------------
    # WAF IP Set block (Layer 7)
    # --------------------------------------------------------

    def block_ip_waf(
        self,
        ip: str,
        ip_set_id: str,
        ip_set_name: str,
        scope: str = "REGIONAL",
        dry_run: bool = True
    ) -> dict:
        """
        Block an IP by adding it to a WAF IP Set.

        WAF IP Sets require the current lock token and
        the full address list, so we read, append, and
        update.

        Args:
            ip: the malicious IP
            ip_set_id: the WAF IP Set id
            ip_set_name: the WAF IP Set name
            scope: REGIONAL (ALB/API GW) or CLOUDFRONT
            dry_run: if True, report only

        Returns:
            Result dict.
        """
        if not self._valid_ip(ip):
            return self._fail(
                ip, "waf", "Invalid IP address"
            )

        cidr = f"{ip}/32"

        try:
            # Read current IP set (need addresses + token)
            current = self.wafv2.get_ip_set(
                Name=ip_set_name,
                Scope=scope,
                Id=ip_set_id,
            )
            addresses = list(
                current["IPSet"].get("Addresses", [])
            )
            lock_token = current["LockToken"]

            if cidr in addresses:
                return {
                    "success": True,
                    "action": "perimeter_block_waf",
                    "ip": ip,
                    "dry_run": dry_run,
                    "already_blocked": True,
                    "message": f"{cidr} already in IP set",
                    "executed_at": _now(),
                }

            if dry_run:
                logger.info(
                    f"DRY RUN: would add {cidr} to WAF "
                    f"IP set {ip_set_name}"
                )
            else:
                addresses.append(cidr)
                self.wafv2.update_ip_set(
                    Name=ip_set_name,
                    Scope=scope,
                    Id=ip_set_id,
                    Addresses=addresses,
                    LockToken=lock_token,
                )
                logger.info(
                    f"BLOCKED {cidr} in WAF IP set "
                    f"{ip_set_name}"
                )

            return {
                "success": True,
                "action": "perimeter_block_waf",
                "ip": ip,
                "cidr": cidr,
                "ip_set_id": ip_set_id,
                "ip_set_name": ip_set_name,
                "layer": "L7 (application/HTTP)",
                "dry_run": dry_run,
                "message": (
                    f"{'Would block' if dry_run else 'Blocked'} "
                    f"{cidr} in WAF IP set {ip_set_name}"
                ),
                "rollback": {
                    "action": "remove_from_ip_set",
                    "ip_set_id": ip_set_id,
                    "cidr": cidr,
                },
                "event": self._build_event(
                    ip, "waf", dry_run
                ),
                "executed_at": _now(),
            }

        except ClientError as e:
            return self._fail(
                ip, "waf",
                f"AWS error: {e.response['Error']['Code']}"
            )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _valid_ip(self, ip: str) -> bool:
        """Basic IPv4 validation"""
        if not ip or not isinstance(ip, str):
            return False
        parts = ip.split(".")
        if len(parts) != 4:
            return False
        try:
            return all(
                0 <= int(p) <= 255 for p in parts
            )
        except ValueError:
            return False

    def _build_event(
        self, ip: str, method: str, dry_run: bool
    ) -> dict:
        """Build a standard event record"""
        return {
            "accessor_identity": "AbuTech-PerimeterBlock",
            "accessor_type": "automation",
            "data_store_name": f"perimeter:{method}",
            "data_path": "network:ip_block",
            "data_classification": "UNKNOWN",
            "event_time": _now(),
            "source_ip": ip,
            "risk_score": 0.0,
            "risk_reasons": [
                "automated_remediation",
                f"block_method:{method}",
                f"dry_run:{dry_run}",
            ],
            "source_system": "perimeter_block",
            "remediation_action": "ip_block",
        }

    def _fail(
        self, ip: str, method: str, message: str
    ) -> dict:
        """Build a failure result"""
        logger.error(
            f"Perimeter block failed ({method}) for "
            f"{ip}: {message}"
        )
        return {
            "success": False,
            "action": f"perimeter_block_{method}",
            "ip": ip,
            "message": message,
            "executed_at": _now(),
        }


def main():
    """Command-line entry point for standalone use."""
    parser = argparse.ArgumentParser(
        description=(
            "Block a malicious IP at the cloud edge "
            "(NACL or WAF)."
        )
    )
    parser.add_argument(
        "--ip", required=True,
        help="Malicious IP to block"
    )
    parser.add_argument(
        "--nacl",
        help="Network ACL id (network-layer block)"
    )
    parser.add_argument(
        "--rule-number", type=int, default=100,
        help="NACL rule number"
    )
    parser.add_argument(
        "--region", default="us-east-1",
        help="AWS region"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually block (default is dry-run)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    pb = PerimeterBlock(region=args.region)

    if args.nacl:
        result = pb.block_ip_nacl(
            args.ip, args.nacl,
            rule_number=args.rule_number,
            dry_run=not args.execute
        )
    else:
        print("Specify --nacl for a network block")
        return None

    print(result["message"])
    return result


if __name__ == "__main__":
    main()