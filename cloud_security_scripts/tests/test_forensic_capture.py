"""
Tests for the AWS Forensic Evidence Capture script.

Uses the boto3 Stubber so tests run with NO AWS account
and NO credentials.
"""

import pytest
import boto3
from botocore.stub import Stubber

from cloud_security_scripts.forensic_capture import (
    ForensicCapture,
)


@pytest.fixture
def ec2_client():
    return boto3.client(
        "ec2",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


@pytest.fixture
def capture(ec2_client):
    return ForensicCapture(
        region="us-east-1", ec2_client=ec2_client
    )


def _describe_with_volumes(
    instance_id="i-0abc123",
    volumes=(("vol-001", "/dev/xvda"),)
):
    """describe_instances response with EBS volumes"""
    return {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": instance_id,
                        "BlockDeviceMappings": [
                            {
                                "DeviceName": dev,
                                "Ebs": {"VolumeId": vid},
                            }
                            for vid, dev in volumes
                        ],
                    }
                ]
            }
        ]
    }


# ============================================================
# DRY RUN
# ============================================================

class TestDryRun:

    def test_dry_run_success(
        self, capture, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_instances",
            _describe_with_volumes(),
            {"InstanceIds": ["i-0abc123"]}
        )
        with stubber:
            result = capture.capture(
                "i-0abc123", "INC-001", dry_run=True
            )
        assert result["success"] is True
        assert result["dry_run"] is True

    def test_dry_run_counts_volumes(
        self, capture, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_instances",
            _describe_with_volumes(
                volumes=(
                    ("vol-001", "/dev/xvda"),
                    ("vol-002", "/dev/xvdb"),
                )
            ),
            {"InstanceIds": ["i-0abc123"]}
        )
        with stubber:
            result = capture.capture(
                "i-0abc123", "INC-001", dry_run=True
            )
        assert result["volumes_captured"] == 2

    def test_dry_run_makes_no_snapshots(
        self, capture, ec2_client
    ):
        # Only describe_instances stubbed. If the script
        # tried create_snapshot, the Stubber would raise.
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_instances",
            _describe_with_volumes(),
            {"InstanceIds": ["i-0abc123"]}
        )
        with stubber:
            result = capture.capture(
                "i-0abc123", "INC-001", dry_run=True
            )
        assert result["success"] is True
        assert result["snapshots"][0][
            "snapshot_id"
        ] == "(dry-run)"

    def test_dry_run_builds_event(
        self, capture, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_instances",
            _describe_with_volumes(),
            {"InstanceIds": ["i-0abc123"]}
        )
        with stubber:
            result = capture.capture(
                "i-0abc123", "INC-001", dry_run=True
            )
        event = result["event"]
        assert event["source_system"] == (
            "forensic_capture"
        )
        assert "incident:INC-001" in (
            event["risk_reasons"]
        )


# ============================================================
# REAL EXECUTION (stubbed)
# ============================================================

class TestExecute:

    def test_capture_single_volume(
        self, capture, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_instances",
            _describe_with_volumes(),
            {"InstanceIds": ["i-0abc123"]}
        )
        # create_snapshot for the one volume.
        # We do not assert exact params (tags include a
        # timestamp), so no expected_params here.
        stubber.add_response(
            "create_snapshot",
            {"SnapshotId": "snap-001"},
        )
        with stubber:
            result = capture.capture(
                "i-0abc123", "INC-001", dry_run=False
            )
        assert result["success"] is True
        assert result["volumes_captured"] == 1
        assert result["snapshots"][0][
            "snapshot_id"
        ] == "snap-001"

    def test_capture_multiple_volumes(
        self, capture, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_instances",
            _describe_with_volumes(
                volumes=(
                    ("vol-001", "/dev/xvda"),
                    ("vol-002", "/dev/xvdb"),
                )
            ),
            {"InstanceIds": ["i-0abc123"]}
        )
        stubber.add_response(
            "create_snapshot",
            {"SnapshotId": "snap-001"},
        )
        stubber.add_response(
            "create_snapshot",
            {"SnapshotId": "snap-002"},
        )
        with stubber:
            result = capture.capture(
                "i-0abc123", "INC-001", dry_run=False
            )
        assert result["volumes_captured"] == 2
        ids = [
            s["snapshot_id"]
            for s in result["snapshots"]
        ]
        assert "snap-001" in ids
        assert "snap-002" in ids

    def test_capture_cleanup_data(
        self, capture, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_instances",
            _describe_with_volumes(),
            {"InstanceIds": ["i-0abc123"]}
        )
        stubber.add_response(
            "create_snapshot",
            {"SnapshotId": "snap-001"},
        )
        with stubber:
            result = capture.capture(
                "i-0abc123", "INC-001", dry_run=False
            )
        assert result["cleanup"][
            "snapshot_ids"
        ] == ["snap-001"]


# ============================================================
# EDGE CASES
# ============================================================

class TestEdgeCases:

    def test_instance_not_found(
        self, capture, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_instances",
            {"Reservations": []},
            {"InstanceIds": ["i-missing"]}
        )
        with stubber:
            result = capture.capture(
                "i-missing", "INC-001", dry_run=True
            )
        assert result["success"] is False
        assert "not found" in result["message"].lower()

    def test_no_volumes_attached(
        self, capture, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_instances",
            {
                "Reservations": [
                    {"Instances": [
                        {
                            "InstanceId": "i-0abc123",
                            "BlockDeviceMappings": []
                        }
                    ]}
                ]
            },
            {"InstanceIds": ["i-0abc123"]}
        )
        with stubber:
            result = capture.capture(
                "i-0abc123", "INC-001", dry_run=True
            )
        assert result["success"] is False
        assert "no ebs volumes" in (
            result["message"].lower()
        )

    def test_aws_client_error(
        self, capture, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_client_error(
            "describe_instances",
            service_error_code="UnauthorizedOperation"
        )
        with stubber:
            result = capture.capture(
                "i-0abc123", "INC-001", dry_run=True
            )
        assert result["success"] is False
        assert "AWS error" in result["message"]