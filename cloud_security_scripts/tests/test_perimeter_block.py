"""
Tests for the AWS Perimeter Block script.

Uses the boto3 Stubber so tests run with NO AWS account.
Covers IP extraction, NACL blocking (L3/4), and WAF
IP Set blocking (L7).
"""

import pytest
import boto3
from botocore.stub import Stubber

from cloud_security_scripts.perimeter_block import (
    PerimeterBlock,
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
def wafv2_client():
    return boto3.client(
        "wafv2",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


@pytest.fixture
def blocker(ec2_client, wafv2_client):
    return PerimeterBlock(
        region="us-east-1",
        ec2_client=ec2_client,
        wafv2_client=wafv2_client,
    )


@pytest.fixture
def ssh_bruteforce_finding():
    return {
        "type": "UnauthorizedAccess:EC2/SSHBruteForce",
        "severity": 5.0,
        "service": {
            "action": {
                "networkConnectionAction": {
                    "remoteIpDetails": {
                        "ipAddressV4": "185.220.101.45"
                    }
                }
            }
        }
    }


@pytest.fixture
def portscan_finding():
    return {
        "type": "Recon:EC2/Portscan",
        "severity": 5.0,
        "service": {
            "action": {
                "portProbeAction": {
                    "portProbeDetails": [
                        {
                            "remoteIpDetails": {
                                "ipAddressV4": "92.63.197.10"
                            }
                        }
                    ]
                }
            }
        }
    }


# ============================================================
# IP EXTRACTION
# ============================================================

class TestIpExtraction:

    def test_extract_from_ssh_bruteforce(
        self, blocker, ssh_bruteforce_finding
    ):
        ip = blocker.extract_malicious_ip(
            ssh_bruteforce_finding
        )
        assert ip == "185.220.101.45"

    def test_extract_from_portscan(
        self, blocker, portscan_finding
    ):
        ip = blocker.extract_malicious_ip(
            portscan_finding
        )
        assert ip == "92.63.197.10"

    def test_extract_empty_finding(self, blocker):
        assert blocker.extract_malicious_ip({}) == ""

    def test_extract_no_ip(self, blocker):
        finding = {
            "type": "Recon:EC2/Portscan",
            "service": {"action": {}}
        }
        assert blocker.extract_malicious_ip(
            finding
        ) == ""


# ============================================================
# NACL BLOCK (L3/4)
# ============================================================

class TestNaclBlock:

    def test_nacl_dry_run(self, blocker):
        result = blocker.block_ip_nacl(
            "185.220.101.45", "acl-0abc",
            dry_run=True
        )
        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["layer"] == "L3/4 (network)"

    def test_nacl_dry_run_no_calls(
        self, blocker
    ):
        # Dry run must not call AWS. No stub queued;
        # if it called create_network_acl_entry it
        # would raise.
        result = blocker.block_ip_nacl(
            "185.220.101.45", "acl-0abc",
            dry_run=True
        )
        assert result["cidr"] == "185.220.101.45/32"

    def test_nacl_execute(
        self, blocker, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "create_network_acl_entry",
            {},
            {
                "NetworkAclId": "acl-0abc",
                "RuleNumber": 100,
                "Protocol": "-1",
                "RuleAction": "deny",
                "Egress": False,
                "CidrBlock": "185.220.101.45/32",
            }
        )
        with stubber:
            result = blocker.block_ip_nacl(
                "185.220.101.45", "acl-0abc",
                rule_number=100, dry_run=False
            )
        assert result["success"] is True
        assert result["rollback"][
            "rule_number"
        ] == 100

    def test_nacl_invalid_ip(self, blocker):
        result = blocker.block_ip_nacl(
            "not-an-ip", "acl-0abc", dry_run=True
        )
        assert result["success"] is False

    def test_nacl_unblock(
        self, blocker, ec2_client
    ):
        stubber = Stubber(ec2_client)
        stubber.add_response(
            "delete_network_acl_entry",
            {},
            {
                "NetworkAclId": "acl-0abc",
                "RuleNumber": 100,
                "Egress": False,
            }
        )
        with stubber:
            result = blocker.unblock_ip_nacl(
                "acl-0abc", 100, dry_run=False
            )
        assert result["success"] is True


# ============================================================
# WAF BLOCK (L7)
# ============================================================

class TestWafBlock:

    def test_waf_dry_run(
        self, blocker, wafv2_client
    ):
        stubber = Stubber(wafv2_client)
        stubber.add_response(
            "get_ip_set",
            {
                "IPSet": {
                    "Name": "blocklist",
                    "Id": "ipset-1",
                    "ARN": "arn:aws:wafv2:us-east-1:111122223333:regional/ipset/blocklist/ipset-1",
                    "Addresses": ["10.0.0.1/32"],
                    "IPAddressVersion": "IPV4",
                },
                "LockToken": "token-123",
            },
            {
                "Name": "blocklist",
                "Scope": "REGIONAL",
                "Id": "ipset-1",
            }
        )
        with stubber:
            result = blocker.block_ip_waf(
                "185.220.101.45", "ipset-1",
                "blocklist", dry_run=True
            )
        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["layer"] == (
            "L7 (application/HTTP)"
        )

    def test_waf_execute(
        self, blocker, wafv2_client
    ):
        stubber = Stubber(wafv2_client)
        stubber.add_response(
            "get_ip_set",
            {
                "IPSet": {
                    "Name": "blocklist",
                    "Id": "ipset-1",
                    "ARN": "arn:aws:wafv2:us-east-1:111122223333:regional/ipset/blocklist/ipset-1",
                    "Addresses": ["10.0.0.1/32"],
                    "IPAddressVersion": "IPV4",
                },
                "LockToken": "token-123",
            },
            {
                "Name": "blocklist",
                "Scope": "REGIONAL",
                "Id": "ipset-1",
            }
        )
        stubber.add_response(
            "update_ip_set",
            {"NextLockToken": "token-456"},
            {
                "Name": "blocklist",
                "Scope": "REGIONAL",
                "Id": "ipset-1",
                "Addresses": [
                    "10.0.0.1/32",
                    "185.220.101.45/32"
                ],
                "LockToken": "token-123",
            }
        )
        with stubber:
            result = blocker.block_ip_waf(
                "185.220.101.45", "ipset-1",
                "blocklist", dry_run=False
            )
        assert result["success"] is True

    def test_waf_already_blocked(
        self, blocker, wafv2_client
    ):
        stubber = Stubber(wafv2_client)
        stubber.add_response(
            "get_ip_set",
            {
                "IPSet": {
                    "Name": "blocklist",
                    "Id": "ipset-1",
                    "ARN": "arn:aws:wafv2:us-east-1:111122223333:regional/ipset/blocklist/ipset-1",
                    "Addresses": ["185.220.101.45/32"],
                    "IPAddressVersion": "IPV4",
                },
                "LockToken": "token-123",
            },
            {
                "Name": "blocklist",
                "Scope": "REGIONAL",
                "Id": "ipset-1",
            }
        )
        with stubber:
            result = blocker.block_ip_waf(
                "185.220.101.45", "ipset-1",
                "blocklist", dry_run=False
            )
        assert result["already_blocked"] is True


# ============================================================
# END TO END (finding -> block)
# ============================================================

class TestEndToEnd:

    def test_extract_then_block(
        self, blocker, ssh_bruteforce_finding
    ):
        ip = blocker.extract_malicious_ip(
            ssh_bruteforce_finding
        )
        result = blocker.block_ip_nacl(
            ip, "acl-0abc", dry_run=True
        )
        assert result["success"] is True
        assert result["ip"] == "185.220.101.45"