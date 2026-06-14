"""
Tests for the AWS Revoke Leaked IAM Credentials script.

Uses the boto3 Stubber so tests run with NO AWS account.
"""

import pytest
import boto3
from botocore.stub import Stubber

from cloud_security_scripts.revoke_iam_credentials import (
    RevokeIAMCredentials,
    DENY_ALL_POLICY_NAME,
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
    return RevokeIAMCredentials(iam_client=iam_client)


@pytest.fixture
def iam_finding():
    return {
        "type": "UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration.OutsideAWS",
        "severity": 8.5,
        "resource": {
            "resourceType": "AccessKey",
            "accessKeyDetails": {
                "userName": "compromised-user",
                "accessKeyId": "AKIAIOSFODNN7EXAMPLE"
            }
        }
    }


# ============================================================
# EXTRACT USER FROM FINDING
# ============================================================

class TestExtractUser:

    def test_extract_from_finding(
        self, revoker, iam_finding
    ):
        user = revoker.extract_compromised_user(
            iam_finding
        )
        assert user == "compromised-user"

    def test_extract_empty_finding(self, revoker):
        assert revoker.extract_compromised_user(
            {}
        ) == ""

    def test_extract_no_user(self, revoker):
        finding = {
            "resource": {"resourceType": "Instance"}
        }
        assert revoker.extract_compromised_user(
            finding
        ) == ""


# ============================================================
# DRY RUN
# ============================================================

class TestDryRun:

    def test_revoke_dry_run(self, revoker):
        # Dry run makes NO AWS calls at all
        result = revoker.revoke(
            "compromised-user",
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            dry_run=True
        )
        assert result["success"] is True
        assert result["dry_run"] is True

    def test_dry_run_plans_three_steps(self, revoker):
        result = revoker.revoke(
            "compromised-user",
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            dry_run=True
        )
        # key deactivate + deny-all + revoke sessions
        assert len(result["steps"]) == 3

    def test_dry_run_two_steps_without_key(
        self, revoker
    ):
        result = revoker.revoke(
            "compromised-user", dry_run=True
        )
        # no key -> deny-all + revoke sessions only
        assert len(result["steps"]) == 2

    def test_dry_run_rollback_data(self, revoker):
        result = revoker.revoke(
            "compromised-user",
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            dry_run=True
        )
        assert result["rollback"][
            "reactivate_key"
        ] == "AKIAIOSFODNN7EXAMPLE"


# ============================================================
# EXECUTE (stubbed)
# ============================================================

class TestExecute:

    def test_full_revoke(
        self, revoker, iam_client
    ):
        stubber = Stubber(iam_client)
        # 1. deactivate key
        stubber.add_response(
            "update_access_key",
            {},
            {
                "UserName": "compromised-user",
                "AccessKeyId": "AKIAIOSFODNN7EXAMPLE",
                "Status": "Inactive",
            }
        )
        # 2. attach deny-all (PolicyDocument varies,
        #    so do not assert exact params)
        stubber.add_response("put_user_policy", {})
        # 3. revoke sessions
        stubber.add_response("put_user_policy", {})

        with stubber:
            result = revoker.revoke(
                "compromised-user",
                access_key_id="AKIAIOSFODNN7EXAMPLE",
                dry_run=False
            )
        assert result["success"] is True
        assert result["dry_run"] is False
        assert all(
            s["done"] for s in result["steps"]
        )

    def test_revoke_without_key(
        self, revoker, iam_client
    ):
        stubber = Stubber(iam_client)
        # only deny-all + revoke sessions
        stubber.add_response("put_user_policy", {})
        stubber.add_response("put_user_policy", {})

        with stubber:
            result = revoker.revoke(
                "compromised-user", dry_run=False
            )
        assert result["success"] is True
        assert len(result["steps"]) == 2


# ============================================================
# INDIVIDUAL ACTIONS
# ============================================================

class TestIndividualActions:

    def test_deactivate_key(
        self, revoker, iam_client
    ):
        stubber = Stubber(iam_client)
        stubber.add_response(
            "update_access_key",
            {},
            {
                "UserName": "u1",
                "AccessKeyId": "AKIAIOSFODNN7EXAMPLE",
                "Status": "Inactive",
            }
        )
        with stubber:
            step = revoker.deactivate_access_key(
                "u1", "AKIAIOSFODNN7EXAMPLE",
                dry_run=False
            )
            
        assert step["done"] is True

    def test_attach_deny_all(
        self, revoker, iam_client
    ):
        stubber = Stubber(iam_client)
        stubber.add_response("put_user_policy", {})
        with stubber:
            step = revoker.attach_deny_all(
                "u1", dry_run=False
            )
        assert step["done"] is True

    def test_revoke_sessions(
        self, revoker, iam_client
    ):
        stubber = Stubber(iam_client)
        stubber.add_response("put_user_policy", {})
        with stubber:
            step = revoker.revoke_sessions(
                "u1", dry_run=False
            )
        assert step["done"] is True


# ============================================================
# RESTORE
# ============================================================

class TestRestore:

    def test_restore_dry_run(self, revoker):
        result = revoker.restore(
            "compromised-user",
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            dry_run=True
        )
        assert result["success"] is True
        assert result["dry_run"] is True

    def test_restore_execute(
        self, revoker, iam_client
    ):
        stubber = Stubber(iam_client)
        # reactivate key
        stubber.add_response(
            "update_access_key",
            {},
            {
                "UserName": "compromised-user",
                "AccessKeyId": "AKIAIOSFODNN7EXAMPLE",
                "Status": "Active",
            }
        )
        # delete the two incident policies
        stubber.add_response("delete_user_policy", {})
        stubber.add_response("delete_user_policy", {})

        with stubber:
            result = revoker.restore(
                "compromised-user",
                access_key_id="AKIAIOSFODNN7EXAMPLE",
                dry_run=False
            )
        assert result["success"] is True


# ============================================================
# EDGE CASES
# ============================================================

class TestEdgeCases:

    def test_no_user_fails(self, revoker):
        result = revoker.revoke("", dry_run=True)
        assert result["success"] is False

    def test_aws_error(
        self, revoker, iam_client
    ):
        stubber = Stubber(iam_client)
        stubber.add_client_error(
            "update_access_key",
            service_error_code="NoSuchEntity"
        )
        with stubber:
            result = revoker.revoke(
                "compromised-user",
                access_key_id="AKIAIOSFODNN7EXAMPLE",
                dry_run=False
            )
        assert result["success"] is False
        assert "AWS error" in result["message"]

    def test_event_built(self, revoker):
        result = revoker.revoke(
            "compromised-user",
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            dry_run=True
        )
        event = result["event"]
        assert event["source_system"] == (
            "revoke_iam_credentials"
        )
        assert "compromised_iam_identity" in (
            event["risk_reasons"]
        )

    def test_deny_policy_name_constant(self):
        assert DENY_ALL_POLICY_NAME == (
            "AbuTech-IncidentDenyAll"
        )
