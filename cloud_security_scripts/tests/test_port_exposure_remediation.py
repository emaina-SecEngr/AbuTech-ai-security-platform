"""
Tests for the AWS Port Exposure Remediation script.

Uses the boto3 Stubber so tests run with NO AWS account.
"""

import pytest
import boto3
from botocore.stub import Stubber

from cloud_security_scripts.port_exposure_remediation\
    import (
        PortExposureRemediation,
        DEFAULT_SENSITIVE_PORTS,
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
def remediator(ec2_client):
    return PortExposureRemediation(
        region="us-east-1", ec2_client=ec2_client
    )


def _sg_response(permissions, sg_id="sg-0abc123"):
    """Build a describe_security_groups response"""
    return {
        "SecurityGroups": [
            {
                "GroupId": sg_id,
                "GroupName": "test-sg",
                "IpPermissions": permissions,
            }
        ]
    }


# Common permission fixtures
SSH_OPEN = {
    "IpProtocol": "tcp",
    "FromPort": 22,
    "ToPort": 22,
    "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
}

RDP_OPEN = {
    "IpProtocol": "tcp",
    "FromPort": 3389,
    "ToPort": 3389,
    "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
}

SSH_RESTRICTED = {
    "IpProtocol": "tcp",
    "FromPort": 22,
    "ToPort": 22,
    "IpRanges": [{"CidrIp": "10.0.0.0/8"}],
}

HTTP_OPEN = {
    "IpProtocol": "tcp",
    "FromPort": 80,
    "ToPort": 80,
    "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
}

ALL_TRAFFIC_OPEN = {
    "IpProtocol": "-1",
    "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
}


# ============================================================
# DETECTION
# ============================================================

class TestFindExposedRules:

    def test_finds_open_ssh(
        self, remediator, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_security_groups",
            _sg_response([SSH_OPEN]),
            {"GroupIds": ["sg-0abc123"]}
        )
        with stubber:
            rules = remediator.find_exposed_rules(
                "sg-0abc123"
            )
        assert len(rules) == 1
        assert 22 in rules[0]["matched_ports"]

    def test_finds_open_rdp(
        self, remediator, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_security_groups",
            _sg_response([RDP_OPEN]),
            {"GroupIds": ["sg-0abc123"]}
        )
        with stubber:
            rules = remediator.find_exposed_rules(
                "sg-0abc123"
            )
        assert 3389 in rules[0]["matched_ports"]

    def test_ignores_restricted_ssh(
        self, remediator, ec2_client
    ):
        # SSH open only to 10.0.0.0/8 is NOT exposed
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_security_groups",
            _sg_response([SSH_RESTRICTED]),
            {"GroupIds": ["sg-0abc123"]}
        )
        with stubber:
            rules = remediator.find_exposed_rules(
                "sg-0abc123"
            )
        assert len(rules) == 0

    def test_ignores_open_http(
        self, remediator, ec2_client
    ):
        # Port 80 open to world is normal, not sensitive
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_security_groups",
            _sg_response([HTTP_OPEN]),
            {"GroupIds": ["sg-0abc123"]}
        )
        with stubber:
            rules = remediator.find_exposed_rules(
                "sg-0abc123"
            )
        assert len(rules) == 0

    def test_all_traffic_open_matches(
        self, remediator, ec2_client
    ):
        # -1 (all traffic) to world exposes everything
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_security_groups",
            _sg_response([ALL_TRAFFIC_OPEN]),
            {"GroupIds": ["sg-0abc123"]}
        )
        with stubber:
            rules = remediator.find_exposed_rules(
                "sg-0abc123"
            )
        assert len(rules) == 1
        assert 22 in rules[0]["matched_ports"]
        assert 3389 in rules[0]["matched_ports"]

    def test_no_such_sg(
        self, remediator, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_security_groups",
            {"SecurityGroups": []},
            {"GroupIds": ["sg-missing"]}
        )
        with stubber:
            rules = remediator.find_exposed_rules(
                "sg-missing"
            )
        assert rules == []


# ============================================================
# REMEDIATION — DRY RUN
# ============================================================

class TestRemediateDryRun:

    def test_dry_run_reports_only(
        self, remediator, ec2_client
    ):
        stubber = Stubber(ec2_client)
        # find_exposed_rules calls describe once
        stubber.add_response(
            "describe_security_groups",
            _sg_response([SSH_OPEN]),
            {"GroupIds": ["sg-0abc123"]}
        )
        with stubber:
            result = remediator.remediate(
                "sg-0abc123", dry_run=True
            )
        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["rules_removed"] == 1

    def test_dry_run_makes_no_revoke(
        self, remediator, ec2_client
    ):
        # Only describe is stubbed. A revoke call would
        # raise (no response queued).
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_security_groups",
            _sg_response([SSH_OPEN, RDP_OPEN]),
            {"GroupIds": ["sg-0abc123"]}
        )
        with stubber:
            result = remediator.remediate(
                "sg-0abc123", dry_run=True
            )
        assert result["rules_removed"] == 2

    def test_no_exposure_clean_result(
        self, remediator, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_security_groups",
            _sg_response([SSH_RESTRICTED]),
            {"GroupIds": ["sg-0abc123"]}
        )
        with stubber:
            result = remediator.remediate(
                "sg-0abc123", dry_run=True
            )
        assert result["success"] is True
        assert result["exposed_rules_found"] == 0


# ============================================================
# REMEDIATION — EXECUTE
# ============================================================

class TestRemediateExecute:

    def test_execute_revokes_ssh(
        self, remediator, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_security_groups",
            _sg_response([SSH_OPEN]),
            {"GroupIds": ["sg-0abc123"]}
        )
        stubber.add_response(
            "revoke_security_group_ingress",
            {},
            {
                "GroupId": "sg-0abc123",
                "IpPermissions": [{
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [
                        {"CidrIp": "0.0.0.0/0"}
                    ],
                }]
            }
        )
        with stubber:
            result = remediator.remediate(
                "sg-0abc123", dry_run=False
            )
        assert result["success"] is True
        assert result["rules_removed"] == 1
        assert result["dry_run"] is False

    def test_execute_provides_rollback(
        self, remediator, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_security_groups",
            _sg_response([SSH_OPEN]),
            {"GroupIds": ["sg-0abc123"]}
        )
        stubber.add_response(
            "revoke_security_group_ingress",
            {},
            {
                "GroupId": "sg-0abc123",
                "IpPermissions": [{
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [
                        {"CidrIp": "0.0.0.0/0"}
                    ],
                }]
            }
        )
        with stubber:
            result = remediator.remediate(
                "sg-0abc123", dry_run=False
            )
        assert len(result["rollback"]["rules"]) == 1

    def test_owner_alert_built(
        self, remediator, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_security_groups",
            _sg_response([SSH_OPEN]),
            {"GroupIds": ["sg-0abc123"]}
        )
        with stubber:
            result = remediator.remediate(
                "sg-0abc123", dry_run=True
            )
        assert 22 in result["owner_alert"][
            "ports_closed"
        ]


# ============================================================
# RESTORE
# ============================================================

class TestRestore:

    def test_restore_dry_run(self, remediator):
        result = remediator.restore(
            "sg-0abc123",
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
            },
            dry_run=True
        )
        assert result["success"] is True
        assert result["dry_run"] is True

    def test_restore_execute(
        self, remediator, ec2_client
    ):
        rule = {
            "IpProtocol": "tcp",
            "FromPort": 22,
            "ToPort": 22,
            "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
        }
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "authorize_security_group_ingress",
            {},
            {
                "GroupId": "sg-0abc123",
                "IpPermissions": [rule]
            }
        )
        with stubber:
            result = remediator.restore(
                "sg-0abc123", rule, dry_run=False
            )
        assert result["success"] is True


# ============================================================
# CONFIG
# ============================================================

class TestConfig:

    def test_default_sensitive_ports(self):
        assert 22 in DEFAULT_SENSITIVE_PORTS
        assert 3389 in DEFAULT_SENSITIVE_PORTS

    def test_custom_sensitive_ports(self, ec2_client):
        r = PortExposureRemediation(
            ec2_client=ec2_client,
            sensitive_ports={8080: "Custom"}
        )
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "describe_security_groups",
            _sg_response([SSH_OPEN]),
            {"GroupIds": ["sg-0abc123"]}
        )
        with stubber:
            rules = r.find_exposed_rules("sg-0abc123")
        # SSH (22) is not in the custom port list
        assert len(rules) == 0