"""
Tests for AWS GuardDuty Normalizer
"""

import pytest
from layer1_ingestion.normalizers.guardduty_normalizer\
    import (
        GuardDutyNormalizer,
        THREAT_PURPOSE_TO_TACTIC,
        THREAT_PURPOSE_TO_TECHNIQUE,
        GUARDDUTY_KEYWORD_RISK
    )


@pytest.fixture
def normalizer():
    return GuardDutyNormalizer()


@pytest.fixture
def cryptomining_finding():
    return {
        "type": "CryptoCurrency:EC2/BitcoinTool.B!DNS",
        "severity": 8.0,
        "title": "EC2 instance querying crypto mining domain",
        "description": "EC2 instance is querying a domain associated with crypto mining",
        "resource": {
            "resourceType": "Instance",
            "instanceDetails": {
                "instanceId": "i-0abc123payment"
            }
        },
        "service": {
            "action": {
                "networkConnectionAction": {
                    "remoteIpDetails": {
                        "ipAddressV4": "45.142.100.10"
                    }
                }
            },
            "count": 12
        },
        "updatedAt": "2026-06-01T03:00:00Z"
    }


@pytest.fixture
def credential_exfil_finding():
    return {
        "type": "UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration.OutsideAWS",
        "severity": 8.5,
        "title": "Credentials exfiltrated from EC2",
        "description": "IAM credentials created for an EC2 instance are being used from an external IP",
        "resource": {
            "resourceType": "AccessKey",
            "accessKeyDetails": {
                "userName": "ec2-payment-role"
            }
        },
        "service": {
            "action": {
                "awsApiCallAction": {
                    "remoteIpDetails": {
                        "ipAddressV4": "185.220.101.45"
                    }
                }
            },
            "count": 3
        }
    }


@pytest.fixture
def recon_finding():
    return {
        "type": "Recon:EC2/PortProbeUnprotectedPort",
        "severity": 5.0,
        "title": "Unprotected port probed",
        "description": "EC2 instance has an unprotected port being probed",
        "resource": {
            "resourceType": "Instance",
            "instanceDetails": {
                "instanceId": "i-0web999"
            }
        },
        "service": {
            "action": {
                "networkConnectionAction": {
                    "remoteIpDetails": {
                        "ipAddressV4": "92.63.197.10"
                    }
                }
            },
            "count": 1
        }
    }


@pytest.fixture
def backdoor_finding():
    return {
        "type": "Backdoor:EC2/C&CActivity.B!DNS",
        "severity": 8.0,
        "title": "C2 activity detected",
        "description": "EC2 instance is communicating with a known command and control server",
        "resource": {
            "resourceType": "Instance",
            "instanceDetails": {
                "instanceId": "i-0compromised"
            }
        },
        "service": {"count": 50}
    }


@pytest.fixture
def s3_finding():
    return {
        "type": "Policy:S3/BucketPublicAccessGranted",
        "severity": 4.0,
        "title": "S3 bucket public access granted",
        "description": "Public access was granted to an S3 bucket",
        "resource": {
            "resourceType": "S3Bucket",
            "s3BucketDetails": [
                {"name": "prod-pci-bucket"}
            ]
        },
        "service": {"count": 1}
    }


# ============================================================
# INITIALIZATION
# ============================================================

class TestInitialization:

    def test_normalizer_initializes(self, normalizer):
        assert normalizer is not None
        assert normalizer.source_system == (
            "aws_guardduty"
        )

    def test_tactic_map_populated(self):
        assert "CryptoCurrency" in (
            THREAT_PURPOSE_TO_TACTIC
        )
        assert "Backdoor" in (
            THREAT_PURPOSE_TO_TACTIC
        )

    def test_technique_map_populated(self):
        assert THREAT_PURPOSE_TO_TECHNIQUE[
            "CryptoCurrency"
        ] == "T1496"

    def test_keyword_risk_populated(self):
        assert "bitcointool" in (
            GUARDDUTY_KEYWORD_RISK
        )


# ============================================================
# FINDING TYPE PARSING
# ============================================================

class TestFindingTypeParsing:

    def test_parse_full_type(self, normalizer):
        tp, rt, tn = normalizer._parse_finding_type(
            "CryptoCurrency:EC2/BitcoinTool.B!DNS"
        )
        assert tp == "CryptoCurrency"
        assert rt == "EC2"
        assert tn == "BitcoinTool.B!DNS"

    def test_parse_iam_type(self, normalizer):
        tp, rt, tn = normalizer._parse_finding_type(
            "UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration.OutsideAWS"
        )
        assert tp == "UnauthorizedAccess"
        assert rt == "IAMUser"

    def test_parse_empty(self, normalizer):
        tp, rt, tn = normalizer._parse_finding_type("")
        assert tp == ""

    def test_parse_no_slash(self, normalizer):
        tp, rt, tn = normalizer._parse_finding_type(
            "Policy:IAMUser"
        )
        assert tp == "Policy"
        assert rt == "IAMUser"


# ============================================================
# CRYPTOMINING
# ============================================================

class TestCryptomining:

    def test_returns_dict(
        self, normalizer, cryptomining_finding
    ):
        result = normalizer.normalize(
            cryptomining_finding
        )
        assert isinstance(result, dict)

    def test_threat_purpose(
        self, normalizer, cryptomining_finding
    ):
        result = normalizer.normalize(
            cryptomining_finding
        )
        assert result[
            "guardduty_threat_purpose"
        ] == "CryptoCurrency"

    def test_mitre_technique(
        self, normalizer, cryptomining_finding
    ):
        result = normalizer.normalize(
            cryptomining_finding
        )
        assert result["mitre_technique"] == "T1496"

    def test_high_risk(
        self, normalizer, cryptomining_finding
    ):
        result = normalizer.normalize(
            cryptomining_finding
        )
        assert result["risk_score"] >= 0.80
        assert "cryptomining_detected" in (
            result["risk_reasons"]
        )

    def test_resource_extracted(
        self, normalizer, cryptomining_finding
    ):
        result = normalizer.normalize(
            cryptomining_finding
        )
        assert result["data_store_name"] == (
            "i-0abc123payment"
        )

    def test_source_ip_extracted(
        self, normalizer, cryptomining_finding
    ):
        result = normalizer.normalize(
            cryptomining_finding
        )
        assert result["source_ip"] == (
            "45.142.100.10"
        )

    def test_repeated_finding_escalates(
        self, normalizer, cryptomining_finding
    ):
        result = normalizer.normalize(
            cryptomining_finding
        )
        # count=12 >= 10
        assert any(
            "repeated_finding" in r
            for r in result["risk_reasons"]
        )

    def test_high_severity_label(
        self, normalizer, cryptomining_finding
    ):
        result = normalizer.normalize(
            cryptomining_finding
        )
        assert result["guardduty_severity"] == "HIGH"


# ============================================================
# CREDENTIAL EXFILTRATION
# ============================================================

class TestCredentialExfil:

    def test_threat_purpose(
        self, normalizer, credential_exfil_finding
    ):
        result = normalizer.normalize(
            credential_exfil_finding
        )
        assert result[
            "guardduty_threat_purpose"
        ] == "UnauthorizedAccess"

    def test_critical_risk(
        self, normalizer, credential_exfil_finding
    ):
        result = normalizer.normalize(
            credential_exfil_finding
        )
        assert result["risk_score"] >= 0.92

    def test_credential_exfil_reason(
        self, normalizer, credential_exfil_finding
    ):
        result = normalizer.normalize(
            credential_exfil_finding
        )
        assert "iam_credential_exfiltration" in (
            result["risk_reasons"]
        )

    def test_tor_ip_flagged(
        self, normalizer, credential_exfil_finding
    ):
        result = normalizer.normalize(
            credential_exfil_finding
        )
        assert "tor_exit_node_src" in (
            result["risk_reasons"]
        )

    def test_iam_accessor_type(
        self, normalizer, credential_exfil_finding
    ):
        result = normalizer.normalize(
            credential_exfil_finding
        )
        assert result["accessor_type"] == (
            "iam_principal"
        )

    def test_iam_user_extracted(
        self, normalizer, credential_exfil_finding
    ):
        result = normalizer.normalize(
            credential_exfil_finding
        )
        assert result["data_store_name"] == (
            "ec2-payment-role"
        )


# ============================================================
# RECON
# ============================================================

class TestRecon:

    def test_threat_purpose(
        self, normalizer, recon_finding
    ):
        result = normalizer.normalize(recon_finding)
        assert result[
            "guardduty_threat_purpose"
        ] == "Recon"

    def test_discovery_tactic(
        self, normalizer, recon_finding
    ):
        result = normalizer.normalize(recon_finding)
        assert result["mitre_tactic"] == "Discovery"

    def test_medium_severity(
        self, normalizer, recon_finding
    ):
        result = normalizer.normalize(recon_finding)
        assert result["guardduty_severity"] == (
            "MEDIUM"
        )


# ============================================================
# BACKDOOR / C2
# ============================================================

class TestBackdoor:

    def test_threat_purpose(
        self, normalizer, backdoor_finding
    ):
        result = normalizer.normalize(
            backdoor_finding
        )
        assert result[
            "guardduty_threat_purpose"
        ] == "Backdoor"

    def test_c2_tactic(
        self, normalizer, backdoor_finding
    ):
        result = normalizer.normalize(
            backdoor_finding
        )
        assert result["mitre_tactic"] == (
            "Command and Control"
        )

    def test_high_impact_escalation(
        self, normalizer, backdoor_finding
    ):
        result = normalizer.normalize(
            backdoor_finding
        )
        assert result["risk_score"] >= 0.78


# ============================================================
# S3 / POLICY
# ============================================================

class TestS3Policy:

    def test_s3_bucket_extracted(
        self, normalizer, s3_finding
    ):
        result = normalizer.normalize(s3_finding)
        assert result["data_store_name"] == (
            "prod-pci-bucket"
        )

    def test_policy_purpose(
        self, normalizer, s3_finding
    ):
        result = normalizer.normalize(s3_finding)
        assert result[
            "guardduty_threat_purpose"
        ] == "Policy"


# ============================================================
# EDGE CASES
# ============================================================

class TestEdgeCases:

    def test_empty_event(self, normalizer):
        result = normalizer.normalize({})
        assert result["accessor_identity"] == (
            "unknown"
        )
        assert result["risk_score"] == 0.0

    def test_none_event(self, normalizer):
        result = normalizer.normalize(None)
        assert result["source_system"] == (
            "aws_guardduty"
        )

    def test_risk_never_exceeds_one(
        self, normalizer, credential_exfil_finding
    ):
        result = normalizer.normalize(
            credential_exfil_finding
        )
        assert result["risk_score"] <= 1.0

    def test_raw_event_preserved(
        self, normalizer, cryptomining_finding
    ):
        result = normalizer.normalize(
            cryptomining_finding
        )
        assert result["raw_event"] == (
            cryptomining_finding
        )

    def test_event_time_present(
        self, normalizer, cryptomining_finding
    ):
        result = normalizer.normalize(
            cryptomining_finding
        )
        assert result["event_time"] != ""

    def test_severity_bands(self, normalizer):
        # Test the severity banding via findings
        high = normalizer.normalize({
            "type": "Recon:EC2/X",
            "severity": 7.5,
            "resource": {},
            "service": {}
        })
        assert high["guardduty_severity"] == "HIGH"

        low = normalizer.normalize({
            "type": "Recon:EC2/X",
            "severity": 2.0,
            "resource": {},
            "service": {}
        })
        assert low["guardduty_severity"] == "LOW"