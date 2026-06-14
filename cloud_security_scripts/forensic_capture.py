"""
Cloud Security Scripts
AWS Forensic Evidence Capture (EBS Snapshots)

A standalone cloud-security automation script that
preserves the disk state of a compromised EC2 instance
by snapshotting all of its attached EBS volumes — so
investigators can analyze the evidence offline even if
the instance is later terminated or tampered with.

THE INCIDENT-RESPONSE PATTERN:
    This is the evidence-preservation half of incident
    response. It pairs with EC2 quarantine:
        Quarantine       = stop the bleeding (cut network)
        Forensic capture = preserve the evidence (snapshot)
    Real IR runbooks do both.

THE AWS MECHANISM:
    1. Find all EBS volumes attached to the instance.
    2. Create a point-in-time snapshot of each volume.
    3. Tag every snapshot with chain-of-custody data:
       IncidentId, SecurityIncident=True, SourceInstance,
       CapturedAt.
    Snapshots persist independently of the instance, so
    the evidence survives termination. Investigators can
    later create a volume from a snapshot, attach it to a
    forensic workstation, and analyze it offline.

WHY SNAPSHOTS:
    - Immutable point-in-time copy (evidence integrity)
    - Survives instance termination (persistence)
    - Can be shared to a separate forensics account
    - Tags provide chain-of-custody metadata
    - Read-only; does not disturb the running instance

TRIGGERS (in a full deployment):
    A GuardDuty critical finding, or the EC2 quarantine
    action, would trigger this capture automatically.

REAL EXECUTION vs SAFE TESTING:
    Makes REAL boto3 calls. dry_run=True (default)
    reports what WOULD be snapshotted without creating
    anything. Tests use the boto3 Stubber — no AWS
    account needed.

STANDALONE USE:
    python -m cloud_security_scripts.forensic_capture \\
        --instance i-0abc123 --incident INC-2026-001 \\
        --region us-east-1 --execute

PROGRAMMATIC USE:
    fc = ForensicCapture(region="us-east-1")
    result = fc.capture("i-0abc123", "INC-2026-001",
                        dry_run=True)
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


class ForensicCapture:
    """
    Captures forensic EBS snapshots of all volumes
    attached to a compromised EC2 instance.
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

    def capture(
        self,
        instance_id: str,
        incident_id: str,
        dry_run: bool = True
    ) -> dict:
        """
        Snapshot all EBS volumes attached to an instance.

        Steps:
            1. Describe the instance, list attached
               EBS volumes.
            2. For each volume, create a snapshot.
            3. Tag each snapshot with incident metadata.

        Args:
            instance_id: the compromised instance
            incident_id: incident identifier for tagging
            dry_run: if True, report only; create nothing

        Returns:
            Result dict with success, snapshot ids,
            rollback/cleanup data, and an event record.
        """
        try:
            volumes = self._get_attached_volumes(
                instance_id
            )

            if volumes is None:
                return self._fail(
                    instance_id,
                    f"Instance {instance_id} not found"
                )

            if not volumes:
                return self._fail(
                    instance_id,
                    f"No EBS volumes attached to "
                    f"{instance_id}"
                )

            snapshots = []

            for vol in volumes:
                vol_id = vol["volume_id"]
                device = vol["device"]

                if dry_run:
                    logger.info(
                        f"DRY RUN: would snapshot volume "
                        f"{vol_id} ({device}) of "
                        f"{instance_id}"
                    )
                    snapshots.append({
                        "volume_id": vol_id,
                        "device": device,
                        "snapshot_id": "(dry-run)",
                    })
                else:
                    snap = self.ec2.create_snapshot(
                        VolumeId=vol_id,
                        Description=(
                            f"Forensic capture of "
                            f"{instance_id} {device} "
                            f"for {incident_id}"
                        ),
                        TagSpecifications=[{
                            "ResourceType": "snapshot",
                            "Tags": self._forensic_tags(
                                instance_id,
                                incident_id,
                                vol_id
                            )
                        }]
                    )
                    snap_id = snap["SnapshotId"]
                    snapshots.append({
                        "volume_id": vol_id,
                        "device": device,
                        "snapshot_id": snap_id,
                    })
                    logger.info(
                        f"CAPTURED snapshot {snap_id} "
                        f"of volume {vol_id} for "
                        f"incident {incident_id}"
                    )

            return {
                "success": True,
                "action": "forensic_capture",
                "instance_id": instance_id,
                "incident_id": incident_id,
                "dry_run": dry_run,
                "volumes_captured": len(snapshots),
                "snapshots": snapshots,
                "message": (
                    f"{'Would capture' if dry_run else 'Captured'} "
                    f"{len(snapshots)} volume snapshot(s) "
                    f"for {instance_id}"
                ),
                "cleanup": {
                    "action": "delete_snapshots",
                    "snapshot_ids": [
                        s["snapshot_id"]
                        for s in snapshots
                        if s["snapshot_id"] != "(dry-run)"
                    ],
                },
                "event": self._build_event(
                    instance_id, incident_id,
                    snapshots, dry_run
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

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _get_attached_volumes(
        self, instance_id: str
    ):
        """
        Return a list of {volume_id, device} for all EBS
        volumes attached to the instance, or None if the
        instance is not found.
        """
        response = self.ec2.describe_instances(
            InstanceIds=[instance_id]
        )

        reservations = response.get("Reservations", [])
        if not reservations:
            return None

        instances = reservations[0].get("Instances", [])
        if not instances:
            return None

        instance = instances[0]
        volumes = []
        for bdm in instance.get(
            "BlockDeviceMappings", []
        ):
            ebs = bdm.get("Ebs", {})
            vol_id = ebs.get("VolumeId")
            if vol_id:
                volumes.append({
                    "volume_id": vol_id,
                    "device": bdm.get(
                        "DeviceName", "unknown"
                    ),
                })
        return volumes

    def _forensic_tags(
        self,
        instance_id: str,
        incident_id: str,
        volume_id: str
    ) -> list:
        """Build chain-of-custody tags for a snapshot"""
        return [
            {"Key": "SecurityIncident", "Value": "True"},
            {"Key": "IncidentId", "Value": incident_id},
            {"Key": "SourceInstance", "Value": instance_id},
            {"Key": "SourceVolume", "Value": volume_id},
            {"Key": "CapturedAt", "Value": _now()},
            {"Key": "CapturedBy",
             "Value": "AbuTech-ForensicCapture"},
            {"Key": "Purpose", "Value": "ForensicEvidence"},
        ]

    def _build_event(
        self,
        instance_id: str,
        incident_id: str,
        snapshots: list,
        dry_run: bool
    ) -> dict:
        """Build a standard event record for the pipeline"""
        return {
            "accessor_identity": (
                "AbuTech-ForensicCapture"
            ),
            "accessor_type": "automation",
            "data_store_name": instance_id,
            "data_path": "ec2:forensic_snapshot",
            "data_classification": "UNKNOWN",
            "event_time": _now(),
            "source_ip": "",
            "risk_score": 0.0,
            "risk_reasons": [
                "forensic_evidence_capture",
                f"incident:{incident_id}",
                f"dry_run:{dry_run}",
            ],
            "source_system": "forensic_capture",
            "remediation_action": "evidence_preservation",
            "snapshot_count": len(snapshots),
        }

    def _fail(
        self, instance_id: str, message: str
    ) -> dict:
        """Build a failure result"""
        logger.error(
            f"Forensic capture failed for "
            f"{instance_id}: {message}"
        )
        return {
            "success": False,
            "action": "forensic_capture",
            "instance_id": instance_id,
            "message": message,
            "executed_at": _now(),
        }


def main():
    """Command-line entry point for standalone use."""
    parser = argparse.ArgumentParser(
        description=(
            "Capture forensic EBS snapshots of a "
            "compromised EC2 instance."
        )
    )
    parser.add_argument(
        "--instance", required=True,
        help="EC2 instance id to capture"
    )
    parser.add_argument(
        "--incident", required=True,
        help="Incident id for snapshot tagging"
    )
    parser.add_argument(
        "--region", default="us-east-1",
        help="AWS region"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually create snapshots "
             "(default is dry-run)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    fc = ForensicCapture(region=args.region)
    result = fc.capture(
        args.instance,
        args.incident,
        dry_run=not args.execute
    )
    print(result["message"])
    return result


if __name__ == "__main__":
    main() 