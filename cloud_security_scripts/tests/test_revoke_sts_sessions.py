"""
Tests for the AWS Revoke Stolen Temporary Tokens
(STS Session Abuse) script.

Uses the boto3 Stubber so tests run with NO AWS account.
"""

import pytest
import boto3
from botocore.stub import Stubber

from cloud_security_scripts.revoke_sts_sessions import (
    RevokeSTSSessions,
    REVOKE_POLICY_NAME,
)


@pytest.fixture
def iam_client():
    return boto3.client(
        "iam",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


@pytest.fixture
def revoker(iam_client):
    return RevokeSTSSessions(iam_client=iam_client)


@pytest.fixture
def exfil_finding_accesskey():
    """Finding where role is in accessKeyDetails"""
    return {
        "type": "UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration.OutsideAWS",
        "severity": 8.5,
        "resource": {
            "resourceType": "AccessKey",
            "accessKeyDetails": {
                "userName": "ec2-payment-role",
                "accessKeyId": "ASIAIOSFODNN7EXAMPLE"
            }
        }
    }


@pytest.fixture
def exfil_finding_instance():
    """Finding where role is in the instance profile"""
    return {
        "type": "UnauthorizedAccess:EC2/InstanceCredentialExfiltration.B",
        "severity": 8.0,
        "resource": {
            "resourceType": "Instance",
            "instanceDetails": {
                "instanceId": "i-0compromised",
                "iamInstanceProfile": {
                    "arn": "arn:aws:iam::111122223333:instance-profile/web-server-role"
                }
            }
        }
    }


# ============================================================
# EXTRACT ROLE FROM FINDING
# ============================================================

class TestExtractRole:

    def test_extract_from_accesskey(
        self, revoker, exfil_finding_accesskey
    ):
        role = revoker.extract_compromised_role(
            exfil_finding_accesskey
        )
        assert role == "ec2-payment-role"

    def test_extract_from_instance_profile(
        self, revoker, exfil_finding_instance
    ):
        role = revoker.extract_compromised_role(
            exfil_finding_instance
        )
        assert role == "web-server-role"

    def test_extract_strips_session_suffix(
        self, revoker
    ):
        finding = {
            "resource": {
                "accessKeyDetails": {
                    "userName": "my-role:session-name"
                }
            }
        }
        role = revoker.extract_compromised_role(
            finding
        )
        assert role == "my-role"

    def test_extract_empty(self, revoker):
        assert revoker.extract_compromised_role(
            {}
        ) == ""

    def test_extract_no_role(self, revoker):
        finding = {
            "resource": {"resourceType": "Instance"}
        }
        assert revoker.extract_compromised_role(
            finding
        ) == ""


# ============================================================
# REVOKE — DRY RUN
# ============================================================

class TestRevokeDryRun:

    def test_dry_run_success(self, revoker):
        # Dry run makes NO AWS calls
        result = revoker.revoke_role_sessions(
            "ec2-payment-role", dry_run=True
        )
        assert result["success"] is True
        assert result["dry_run"] is True

    def test_dry_run_has_revocation_time(
        self, revoker
    ):
        result = revoker.revoke_role_sessions(
            "ec2-payment-role", dry_run=True
        )
        assert result["revocation_time"] != ""

    def test_dry_run_rollback_data(self, revoker):
        result = revoker.revoke_role_sessions(
            "ec2-payment-role", dry_run=True
        )
        assert result["rollback"][
            "policy_name"
        ] == REVOKE_POLICY_NAME

    def test_dry_run_builds_event(self, revoker):
        result = revoker.revoke_role_sessions(
            "ec2-payment-role", dry_run=True
        )
        event = result["event"]
        assert event["source_system"] == (
            "revoke_sts_sessions"
        )
        assert "stolen_sts_token_revoked" in (
            event["risk_reasons"]
        )


# ============================================================
# REVOKE — EXECUTE
# ============================================================

class TestRevokeExecute:

    def test_execute_attaches_policy(
        self, revoker, iam_client
    ):
        stubber = Stubber(iam_client)
        # PolicyDocument contains a timestamp, so we
        # do not assert exact params.
        stubber.add_response("put_role_policy", {})
        with stubber:
            result = revoker.revoke_role_sessions(
                "ec2-payment-role", dry_run=False
            )
        assert result["success"] is True
        assert result["dry_run"] is False

    def test_execute_no_role_fails(self, revoker):
        result = revoker.revoke_role_sessions(
            "", dry_run=False
        )
        assert result["success"] is False

    def test_execute_aws_error(
        self, revoker, iam_client
    ):
        stubber = Stubber(iam_client)
        stubber.add_client_error(
            "put_role_policy",
            service_error_code="NoSuchEntity"
        )
        with stubber:
            result = revoker.revoke_role_sessions(
                "ec2-payment-role", dry_run=False
            )
        assert result["success"] is False
        assert "AWS error" in result["message"]


# ============================================================
# RESTORE
# ============================================================

class TestRestore:

    def test_restore_dry_run(self, revoker):
        result = revoker.restore(
            "ec2-payment-role", dry_run=True
        )
        assert result["success"] is True
        assert result["dry_run"] is True

    def test_restore_execute(
        self, revoker, iam_client
    ):
        stubber = Stubber(iam_client)
        stubber.add_response(
            "delete_role_policy",
            {},
            {
                "RoleName": "ec2-payment-role",
                "PolicyName": REVOKE_POLICY_NAME,
            }
        )
        with stubber:
            result = revoker.restore(
                "ec2-payment-role", dry_run=False
            )
        assert result["success"] is True


# ============================================================
# END TO END (finding -> revoke)
# ============================================================

class TestEndToEnd:

    def test_extract_then_revoke(
        self, revoker, exfil_finding_instance
    ):
        role = revoker.extract_compromised_role(
            exfil_finding_instance
        )
        result = revoker.revoke_role_sessions(
            role, dry_run=True
        )
        assert result["success"] is True
        assert result["role"] == "web-server-role"


# ============================================================
# CONFIG
# ============================================================

class TestConfig:

    def test_policy_name_matches_aws_convention(self):
        assert REVOKE_POLICY_NAME == (
            "AWSRevokeOlderSessions"
        )