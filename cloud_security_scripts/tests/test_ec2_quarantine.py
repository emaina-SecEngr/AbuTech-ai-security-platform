"""
Tests for the AWS EC2 Quarantine script.

Uses the boto3 Stubber so the tests run with NO AWS
account and NO credentials — the Stubber feeds the
client canned responses, and asserts the script makes
the expected API calls.
"""

import pytest
import boto3
from botocore.stub import Stubber

from cloud_security_scripts.ec2_quarantine import (
    EC2Quarantine,
    QUARANTINE_SG_NAME,
)


@pytest.fixture
def ec2_client():
    """A real boto3 client with fake credentials,
    safe because the Stubber intercepts all calls."""
    return boto3.client(
        "ec2",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


@pytest.fixture
def quarantine(ec2_client):
    return EC2Quarantine(
        region="us-east-1", ec2_client=ec2_client
    )


# A reusable describe_instances response
def _describe_response(
    instance_id="i-0abc123",
    sgs=("sg-original1",),
    vpc="vpc-123"
):
    return {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": instance_id,
                        "VpcId": vpc,
                        "SecurityGroups": [
                            {"GroupId": g}
                            for g in sgs
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

    def test_quarantine_dry_run_success(
        self, quarantine, ec2_client
    ):
        stubber = Stubber(ec2_client)
        # Dry run only reads the instance
        stubber.add_response(
            "describe_instances",
            _describe_response(),
            {"InstanceIds": ["i-0abc123"]}
        )
        with stubber:
            result = quarantine.quarantine(
                "i-0abc123", dry_run=True
            )

        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["action"] == "ec2_quarantine"

    def test_dry_run_captures_rollback(
        self, quarantine, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_instances",
            _describe_response(sgs=("sg-a", "sg-b")),
            {"InstanceIds": ["i-0abc123"]}
        )
        with stubber:
            result = quarantine.quarantine(
                "i-0abc123", dry_run=True
            )

        assert result["rollback"][
            "original_sgs"
        ] == ["sg-a", "sg-b"]

    def test_dry_run_makes_no_changes(
        self, quarantine, ec2_client
    ):
        # Only describe_instances is stubbed.
        # If the script tried to modify anything,
        # the Stubber would raise (no response queued).
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_instances",
            _describe_response(),
            {"InstanceIds": ["i-0abc123"]}
        )
        with stubber:
            result = quarantine.quarantine(
                "i-0abc123", dry_run=True
            )
        assert result["success"] is True
        # No assert_no_pending_responses error means
        # exactly one call was made

    def test_dry_run_builds_event(
        self, quarantine, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_instances",
            _describe_response(),
            {"InstanceIds": ["i-0abc123"]}
        )
        with stubber:
            result = quarantine.quarantine(
                "i-0abc123", dry_run=True
            )
        event = result["event"]
        assert event["source_system"] == (
            "ec2_quarantine"
        )
        assert event["data_store_name"] == (
            "i-0abc123"
        )


# ============================================================
# REAL EXECUTION (stubbed)
# ============================================================

class TestExecute:

    def test_quarantine_creates_sg_and_isolates(
        self, quarantine, ec2_client
    ):
        stubber = Stubber(ec2_client)

        # 1. describe the instance
        stubber.add_response(
            "describe_instances",
            _describe_response(),
            {"InstanceIds": ["i-0abc123"]}
        )
        # 2. look for existing quarantine SG (none)
        stubber.add_response(
            "describe_security_groups",
            {"SecurityGroups": []},
            {"Filters": [
                {"Name": "group-name",
                 "Values": [QUARANTINE_SG_NAME]},
                {"Name": "vpc-id",
                 "Values": ["vpc-123"]},
            ]}
        )
        # 3. create the quarantine SG
        stubber.add_response(
            "create_security_group",
            {"GroupId": "sg-quarantine"},
            {"GroupName": QUARANTINE_SG_NAME,
             "Description": (
                 "AbuTech forensic quarantine - "
                 "deny all traffic"
             ),
             "VpcId": "vpc-123"}
        )
        # 4. revoke egress (deny-all)
        stubber.add_response(
            "revoke_security_group_egress",
            {},
            {"GroupId": "sg-quarantine",
             "IpPermissions": [{
                 "IpProtocol": "-1",
                 "IpRanges": [{"CidrIp": "0.0.0.0/0"}]
             }]}
        )
        # 5. swap the instance SGs
        stubber.add_response(
            "modify_instance_attribute",
            {},
            {"InstanceId": "i-0abc123",
             "Groups": ["sg-quarantine"]}
        )
        # 6. tag the instance
        stubber.add_response(
            "create_tags",
            {},
        )

        with stubber:
            result = quarantine.quarantine(
                "i-0abc123", dry_run=False
            )

        assert result["success"] is True
        assert result["dry_run"] is False
        assert result["rollback"][
            "original_sgs"
        ] == ["sg-original1"]

    def test_quarantine_reuses_existing_sg(
        self, quarantine, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_instances",
            _describe_response(),
            {"InstanceIds": ["i-0abc123"]}
        )
        # existing quarantine SG found - no create
        stubber.add_response(
            "describe_security_groups",
            {"SecurityGroups": [
                {"GroupId": "sg-existing-q"}
            ]},
            {"Filters": [
                {"Name": "group-name",
                 "Values": [QUARANTINE_SG_NAME]},
                {"Name": "vpc-id",
                 "Values": ["vpc-123"]},
            ]}
        )
        stubber.add_response(
            "modify_instance_attribute",
            {},
            {"InstanceId": "i-0abc123",
             "Groups": ["sg-existing-q"]}
        )
        stubber.add_response("create_tags", {})

        with stubber:
            result = quarantine.quarantine(
                "i-0abc123", dry_run=False
            )
        assert result["success"] is True


# ============================================================
# RESTORE
# ============================================================

class TestRestore:

    def test_restore_dry_run(self, quarantine):
        result = quarantine.restore(
            "i-0abc123",
            ["sg-original1"],
            dry_run=True
        )
        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["restored_sgs"] == (
            ["sg-original1"]
        )

    def test_restore_execute(
        self, quarantine, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "modify_instance_attribute",
            {},
            {"InstanceId": "i-0abc123",
             "Groups": ["sg-original1"]}
        )
        stubber.add_response("create_tags", {})

        with stubber:
            result = quarantine.restore(
                "i-0abc123",
                ["sg-original1"],
                dry_run=False
            )
        assert result["success"] is True

    def test_restore_no_sgs_fails(self, quarantine):
        result = quarantine.restore(
            "i-0abc123", [], dry_run=True
        )
        assert result["success"] is False


# ============================================================
# ERROR HANDLING
# ============================================================

class TestErrorHandling:

    def test_instance_not_found(
        self, quarantine, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_instances",
            {"Reservations": []},
            {"InstanceIds": ["i-missing"]}
        )
        with stubber:
            result = quarantine.quarantine(
                "i-missing", dry_run=True
            )
        assert result["success"] is False
        assert "not found" in result["message"].lower()

    def test_aws_client_error(
        self, quarantine, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_client_error(
            "describe_instances",
            service_error_code="UnauthorizedOperation"
        )
        with stubber:
            result = quarantine.quarantine(
                "i-0abc123", dry_run=True
            )
        assert result["success"] is False
        assert "AWS error" in result["message"]