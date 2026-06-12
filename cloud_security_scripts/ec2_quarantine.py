"""
Cloud Security Scripts
AWS EC2 Quarantine (Network Isolation)

A standalone cloud-security automation script that
network-isolates a compromised EC2 instance for
forensic investigation — without terminating it.

THE SECURITY PATTERN:
    When an instance is compromised, you must cut
    the attacker off WITHOUT destroying evidence.
    The technique: replace the instance's security
    groups with a single deny-all "quarantine"
    security group.

    Result:
        - Network access severed (attacker locked out)
        - Instance stays running (memory + disk intact
          for forensics)
        - Fully reversible (original SGs restored after
          investigation)
        - Auditable (instance tagged, action logged)

    This mirrors CrowdStrike Falcon network
    containment: isolate the host, keep it alive
    for investigation.

REAL EXECUTION vs SAFE TESTING:
    This makes REAL boto3 calls to AWS. To stay safe:
        - dry_run=True (default) shows what WOULD
          happen without changing anything.
        - dry_run=False performs the real isolation.
    Tests use the boto3 Stubber, so no AWS account
    or credentials are needed to run them.

STANDALONE USE (command line):
    python -m cloud_security_scripts.ec2_quarantine \\
        --instance i-0abc123 --region us-east-1 --dry-run

PROGRAMMATIC USE:
    q = EC2Quarantine(region="us-east-1")
    result = q.quarantine("i-0abc123", dry_run=True)
    # later, after forensics:
    q.restore("i-0abc123", result["rollback"]["original_sgs"])
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


# Name of the deny-all quarantine security group.
# Created on demand if it does not exist.
QUARANTINE_SG_NAME = "abutech-quarantine-deny-all"
QUARANTINE_SG_DESC = (
    "AbuTech forensic quarantine - deny all traffic"
)


class EC2Quarantine:
    """
    Network-isolates an EC2 instance by replacing its
    security groups with a deny-all quarantine group.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        ec2_client=None
    ):
        """
        Args:
            region: AWS region
            ec2_client: optional pre-built boto3 client
                        (used by tests with a Stubber)
        """
        self.region = region
        self.ec2 = ec2_client or boto3.client(
            "ec2", region_name=region
        )

    def quarantine(
        self,
        instance_id: str,
        dry_run: bool = True
    ) -> dict:
        """
        Network-isolate an EC2 instance.

        Steps:
            1. Look up the instance + its current SGs
               (this is the rollback data).
            2. Find or create the deny-all quarantine SG.
            3. Swap the instance's SGs to quarantine.
            4. Tag the instance as quarantined.

        Args:
            instance_id: EC2 instance to isolate
            dry_run: if True, only report what would
                     happen; make no changes

        Returns:
            Result dict with success, action, rollback
            data, and an event record for the pipeline.
        """
        try:
            # 1. Find the instance and its current SGs
            current_sgs, vpc_id = (
                self._get_instance_sgs(instance_id)
            )

            if current_sgs is None:
                return self._fail(
                    instance_id,
                    f"Instance {instance_id} not found"
                )

            # 2. Find or create the quarantine SG
            if dry_run:
                quarantine_sg_id = "(dry-run-sg-id)"
            else:
                quarantine_sg_id = (
                    self._ensure_quarantine_sg(vpc_id)
                )

            # 3. Swap security groups
            if dry_run:
                logger.info(
                    f"DRY RUN: would isolate "
                    f"{instance_id} - swap SGs "
                    f"{current_sgs} -> "
                    f"[{QUARANTINE_SG_NAME}]"
                )
            else:
                self.ec2.modify_instance_attribute(
                    InstanceId=instance_id,
                    Groups=[quarantine_sg_id]
                )
                self._tag_quarantined(instance_id)
                logger.info(
                    f"ISOLATED {instance_id}: SGs "
                    f"replaced with quarantine group"
                )

            return {
                "success": True,
                "action": "ec2_quarantine",
                "instance_id": instance_id,
                "dry_run": dry_run,
                "quarantine_sg": (
                    QUARANTINE_SG_NAME
                ),
                "message": (
                    f"{'Would isolate' if dry_run else 'Isolated'} "
                    f"{instance_id} for forensics"
                ),
                "rollback": {
                    "action": "restore_sgs",
                    "instance_id": instance_id,
                    "original_sgs": current_sgs,
                },
                "event": self._build_event(
                    instance_id, current_sgs, dry_run
                ),
                "executed_at": _now(),
            }

        except ClientError as e:
            return self._fail(
                instance_id,
                f"AWS error: {e.response['Error']['Code']}"
            )
        except Exception as e:
            return self._fail(
                instance_id, f"Error: {str(e)}"
            )

    def restore(
        self,
        instance_id: str,
        original_sgs: list,
        dry_run: bool = True
    ) -> dict:
        """
        Restore an instance's original security groups
        after forensic investigation is complete.

        Args:
            instance_id: instance to restore
            original_sgs: the SG ids captured at
                          quarantine time (rollback data)
            dry_run: if True, report only

        Returns:
            Result dict
        """
        if not original_sgs:
            return self._fail(
                instance_id,
                "No original security groups provided"
            )

        try:
            if dry_run:
                logger.info(
                    f"DRY RUN: would restore "
                    f"{instance_id} SGs -> {original_sgs}"
                )
            else:
                self.ec2.modify_instance_attribute(
                    InstanceId=instance_id,
                    Groups=original_sgs
                )
                self._tag_restored(instance_id)
                logger.info(
                    f"RESTORED {instance_id}: original "
                    f"SGs reapplied"
                )

            return {
                "success": True,
                "action": "ec2_restore",
                "instance_id": instance_id,
                "dry_run": dry_run,
                "restored_sgs": original_sgs,
                "message": (
                    f"{'Would restore' if dry_run else 'Restored'} "
                    f"{instance_id}"
                ),
                "executed_at": _now(),
            }

        except ClientError as e:
            return self._fail(
                instance_id,
                f"AWS error: {e.response['Error']['Code']}"
            )

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _get_instance_sgs(
        self, instance_id: str
    ) -> tuple:
        """
        Return (list_of_sg_ids, vpc_id) for an instance,
        or (None, None) if not found.
        """
        response = self.ec2.describe_instances(
            InstanceIds=[instance_id]
        )

        reservations = response.get(
            "Reservations", []
        )
        if not reservations:
            return None, None

        instances = reservations[0].get(
            "Instances", []
        )
        if not instances:
            return None, None

        instance = instances[0]
        sgs = [
            g["GroupId"]
            for g in instance.get("SecurityGroups", [])
        ]
        vpc_id = instance.get("VpcId", "")
        return sgs, vpc_id

    def _ensure_quarantine_sg(
        self, vpc_id: str
    ) -> str:
        """
        Find the quarantine SG in this VPC, or create it.
        A deny-all SG has no inbound rules and we strip
        its default outbound rule.
        """
        # Look for an existing quarantine SG
        response = self.ec2.describe_security_groups(
            Filters=[
                {
                    "Name": "group-name",
                    "Values": [QUARANTINE_SG_NAME]
                },
                {
                    "Name": "vpc-id",
                    "Values": [vpc_id]
                },
            ]
        )
        groups = response.get("SecurityGroups", [])
        if groups:
            return groups[0]["GroupId"]

        # Create it
        created = self.ec2.create_security_group(
            GroupName=QUARANTINE_SG_NAME,
            Description=QUARANTINE_SG_DESC,
            VpcId=vpc_id
        )
        sg_id = created["GroupId"]

        # Strip the default allow-all egress rule so
        # the group truly denies all traffic
        try:
            self.ec2.revoke_security_group_egress(
                GroupId=sg_id,
                IpPermissions=[{
                    "IpProtocol": "-1",
                    "IpRanges": [
                        {"CidrIp": "0.0.0.0/0"}
                    ]
                }]
            )
        except ClientError:
            # If the default rule is already gone,
            # that is fine
            pass

        return sg_id

    def _tag_quarantined(self, instance_id: str):
        """Tag the instance as quarantined for audit"""
        self.ec2.create_tags(
            Resources=[instance_id],
            Tags=[
                {
                    "Key": "Status",
                    "Value": "QUARANTINED"
                },
                {
                    "Key": "QuarantinedAt",
                    "Value": _now()
                },
                {
                    "Key": "QuarantinedBy",
                    "Value": "AbuTech-EC2Quarantine"
                },
            ]
        )

    def _tag_restored(self, instance_id: str):
        """Tag the instance as restored"""
        self.ec2.create_tags(
            Resources=[instance_id],
            Tags=[
                {
                    "Key": "Status",
                    "Value": "RESTORED"
                },
                {
                    "Key": "RestoredAt",
                    "Value": _now()
                },
            ]
        )

    def _build_event(
        self,
        instance_id: str,
        original_sgs: list,
        dry_run: bool
    ) -> dict:
        """
        Build an event record in the platform's
        standard shape so a quarantine action can flow
        into the pipeline / audit store.
        """
        return {
            "accessor_identity": "AbuTech-EC2Quarantine",
            "accessor_type": "automation",
            "data_store_name": instance_id,
            "data_path": "ec2:network_isolation",
            "data_classification": "UNKNOWN",
            "event_time": _now(),
            "source_ip": "",
            "risk_score": 0.0,
            "risk_reasons": [
                "automated_remediation",
                f"dry_run:{dry_run}",
            ],
            "source_system": "ec2_quarantine",
            "remediation_action": "network_isolation",
            "original_security_groups": original_sgs,
        }

    def _fail(
        self, instance_id: str, message: str
    ) -> dict:
        """Build a failure result"""
        logger.error(
            f"Quarantine failed for {instance_id}: "
            f"{message}"
        )
        return {
            "success": False,
            "action": "ec2_quarantine",
            "instance_id": instance_id,
            "message": message,
            "executed_at": _now(),
        }


def main():
    """Command-line entry point for standalone use."""
    parser = argparse.ArgumentParser(
        description=(
            "Network-isolate a compromised EC2 "
            "instance for forensics."
        )
    )
    parser.add_argument(
        "--instance", required=True,
        help="EC2 instance id to quarantine"
    )
    parser.add_argument(
        "--region", default="us-east-1",
        help="AWS region"
    )
    parser.add_argument(
        "--restore", action="store_true",
        help="Restore instead of quarantine"
    )
    parser.add_argument(
        "--sgs", nargs="*", default=[],
        help="Original SG ids (for --restore)"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually perform the action "
             "(default is dry-run)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    q = EC2Quarantine(region=args.region)
    dry = not args.execute

    if args.restore:
        result = q.restore(
            args.instance, args.sgs, dry_run=dry
        )
    else:
        result = q.quarantine(
            args.instance, dry_run=dry
        )

    print(result["message"])
    return result


if __name__ == "__main__":
    main()