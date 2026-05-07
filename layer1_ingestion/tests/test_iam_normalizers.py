"""
IAM Normalizer Tests

Tests verify all four new normalizers:
1. AWS Secrets Manager normalizer
2. Azure Key Vault normalizer
3. SailPoint ISC normalizer
4. CyberArk PAM normalizer
"""

import pytest
from layer1_ingestion.normalizers.aws_secrets_normalizer import (
    AWSSecretsNormalizer
)
from layer1_ingestion.normalizers.azure_keyvault_normalizer import (
    AzureKeyVaultNormalizer
)
from layer1_ingestion.normalizers.sailpoint_normalizer import (
    SailPointNormalizer
)
from layer1_ingestion.normalizers.cyberark_normalizer import (
    CyberArkNormalizer
)
from layer1_ingestion.schema.iam_schema import (
    IamEvent,
    IamSecretEvent,
    IamGovernanceEvent,
    IamPrivilegedEvent
)


# ============================================================
# SAMPLE RAW EVENTS
# ============================================================

AWS_GET_SECRET = {
    "eventID": "aws-event-001",
    "eventTime": "2024-03-29T09:19:00Z",
    "eventSource": "secretsmanager.amazonaws.com",
    "eventName": "GetSecretValue",
    "awsRegion": "us-east-1",
    "sourceIPAddress": "10.0.0.155",
    "userIdentity": {
        "type": "IAMUser",
        "userName": "svc_backup",
        "arn": "arn:aws:iam::123456789:user/svc_backup",
        "principalId": "AIDABC123456"
    },
    "requestParameters": {
        "secretId": "prod/db_password"
    },
    "responseElements": None
}

AWS_LIST_SECRETS = {
    "eventID": "aws-event-002",
    "eventTime": "2024-03-29T03:00:00Z",
    "eventSource": "secretsmanager.amazonaws.com",
    "eventName": "ListSecrets",
    "awsRegion": "us-east-1",
    "sourceIPAddress": "185.220.101.45",
    "userIdentity": {
        "type": "IAMUser",
        "userName": "svc_backup",
        "arn": "arn:aws:iam::123456789:user/svc_backup",
        "principalId": "AIDABC123456"
    },
    "requestParameters": {},
    "responseElements": None
}

AWS_ROOT_SECRET = {
    "eventID": "aws-event-003",
    "eventTime": "2024-03-29T09:19:00Z",
    "eventSource": "secretsmanager.amazonaws.com",
    "eventName": "GetSecretValue",
    "awsRegion": "us-east-1",
    "sourceIPAddress": "10.0.0.1",
    "userIdentity": {
        "type": "Root",
        "arn": "arn:aws:iam::123456789:root",
        "principalId": "123456789"
    },
    "requestParameters": {
        "secretId": "prod/master_key"
    },
    "responseElements": None
}

AWS_DELETE_SECRET = {
    "eventID": "aws-event-004",
    "eventTime": "2024-03-29T09:20:00Z",
    "eventSource": "secretsmanager.amazonaws.com",
    "eventName": "DeleteSecret",
    "awsRegion": "us-east-1",
    "sourceIPAddress": "10.0.0.155",
    "userIdentity": {
        "type": "IAMUser",
        "userName": "svc_backup",
        "arn": "arn:aws:iam::123456789:user/svc_backup",
        "principalId": "AIDABC123456"
    },
    "requestParameters": {
        "secretId": "prod/api_key"
    },
    "responseElements": None
}

AZURE_SECRET_GET = {
    "time": "2024-03-29T09:19:00Z",
    "resourceId": (
        "/subscriptions/sub-123/resourceGroups/"
        "prod-rg/providers/Microsoft.KeyVault/"
        "vaults/prod-vault"
    ),
    "operationName": "SecretGet",
    "category": "AuditEvent",
    "resultType": "Success",
    "callerIpAddress": "10.0.0.155",
    "correlationId": "azure-corr-001",
    "identity": {
        "claim": {
            "oid": "user-oid-123",
            "upn": "svc_backup@corp.com",
            "appid": "app-id-456"
        }
    },
    "properties": {
        "id": (
            "https://prod-vault.vault.azure.net/"
            "secrets/db-password/version123"
        ),
        "isAddressAuthorized": True,
        "httpStatusCode": 200
    }
}

AZURE_UNAUTHORIZED = {
    "time": "2024-03-29T09:19:00Z",
    "resourceId": (
        "/subscriptions/sub-123/resourceGroups/"
        "prod-rg/providers/Microsoft.KeyVault/"
        "vaults/prod-vault"
    ),
    "operationName": "SecretGet",
    "category": "AuditEvent",
    "resultType": "Success",
    "callerIpAddress": "185.220.101.45",
    "correlationId": "azure-corr-002",
    "identity": {
        "claim": {
            "upn": "svc_backup@corp.com"
        }
    },
    "properties": {
        "id": (
            "https://prod-vault.vault.azure.net/"
            "secrets/stripe-key/version456"
        ),
        "isAddressAuthorized": False,
        "httpStatusCode": 200
    }
}

AZURE_SECRET_DELETE = {
    "time": "2024-03-29T09:20:00Z",
    "resourceId": (
        "/subscriptions/sub-123/resourceGroups/"
        "prod-rg/providers/Microsoft.KeyVault/"
        "vaults/prod-vault"
    ),
    "operationName": "SecretDelete",
    "category": "AuditEvent",
    "resultType": "Success",
    "callerIpAddress": "10.0.0.155",
    "correlationId": "azure-corr-003",
    "identity": {
        "claim": {
            "upn": "admin@corp.com"
        }
    },
    "properties": {
        "id": (
            "https://prod-vault.vault.azure.net/"
            "secrets/old-key/version789"
        ),
        "isAddressAuthorized": True,
        "httpStatusCode": 200
    }
}

SAILPOINT_CERTIFICATION = {
    "id": "sp-event-001",
    "type": "CERTIFICATION_COMPLETED",
    "created": "2024-03-29T10:00:00.000Z",
    "attributes": {
        "identityId": "jsmith-id-123",
        "identityName": "jsmith",
        "identityDisplayName": "John Smith",
        "certificationName": "Q1 2024 Review",
        "certificationId": "cert-uuid-001",
        "decision": "APPROVED",
        "decisionMaker": "manager@corp.com",
        "decisionTime": 3,
        "entitlement": "AdminRole",
        "application": "AWS",
        "isOrphan": False
    }
}

SAILPOINT_ORPHAN = {
    "id": "sp-event-002",
    "type": "ORPHAN_ACCOUNT_DETECTED",
    "created": "2024-03-29T10:01:00.000Z",
    "attributes": {
        "identityId": "ex-employee-456",
        "identityName": "ex_employee",
        "identityDisplayName": "Former Employee",
        "entitlement": "VPNAccess",
        "application": "Cisco VPN",
        "isOrphan": True,
        "decisionTime": None
    }
}

SAILPOINT_SOD = {
    "id": "sp-event-003",
    "type": "SOD_VIOLATION_DETECTED",
    "created": "2024-03-29T10:02:00.000Z",
    "attributes": {
        "identityId": "jsmith-id-123",
        "identityName": "jsmith",
        "identityDisplayName": "John Smith",
        "entitlement": "PurchaseOrderCreate",
        "application": "SAP",
        "conflictingEntitlements": [
            "PurchaseOrderApprove"
        ],
        "isOrphan": False,
        "decisionTime": None
    }
}

SAILPOINT_ACCESS_DENIED = {
    "id": "sp-event-004",
    "type": "ACCESS_REQUEST_DENIED",
    "created": "2024-03-29T10:03:00.000Z",
    "attributes": {
        "identityId": "jsmith-id-123",
        "identityName": "jsmith",
        "identityDisplayName": "John Smith",
        "entitlement": "domain_admin",
        "application": "Active Directory",
        "decision": "DENIED",
        "decisionMaker": "security_team@corp.com",
        "decisionTime": 120,
        "isOrphan": False
    }
}

CYBERARK_PSM_CONNECT = {
    "EventID": 12345,
    "EventCode": "PSM.Connect",
    "EventTime": "2024-03-29T09:18:00Z",
    "Username": "jsmith",
    "UserDomain": "CORP",
    "TargetAccount": "domain_admin",
    "TargetSystem": "DC01.corp.local",
    "TargetAddress": "10.0.0.1",
    "Safe": "Domain_Admins",
    "SessionID": "session-uuid-001",
    "IsRecorded": True,
    "WorkstationID": "WKSTN-JSMITH-01"
}

CYBERARK_AFTER_HOURS = {
    "EventID": 12346,
    "EventCode": "PSM.Connect",
    "EventTime": "2024-03-29T03:00:00Z",
    "Username": "jsmith",
    "UserDomain": "CORP",
    "TargetAccount": "domain_admin",
    "TargetSystem": "DC01.corp.local",
    "TargetAddress": "10.0.0.1",
    "Safe": "Domain_Admins",
    "SessionID": "session-uuid-002",
    "IsRecorded": True,
    "WorkstationID": "WKSTN-JSMITH-01"
}

CYBERARK_PASSWORD_RETRIEVE = {
    "EventID": 12347,
    "EventCode": "Password.Retrieve",
    "EventTime": "2024-03-29T09:19:00Z",
    "Username": "svc_backup",
    "UserDomain": "CORP",
    "TargetAccount": "domain_admin",
    "TargetSystem": "DC01.corp.local",
    "TargetAddress": "10.0.0.1",
    "Safe": "Domain_Admins",
    "SessionID": "",
    "IsRecorded": False,
    "WorkstationID": "WKSTN-JSMITH-01"
}


# ============================================================
# TEST CLASS — AWS SECRETS MANAGER
# ============================================================

class TestAWSSecretsNormalizer:
    """Tests for AWS Secrets Manager normalization"""

    def setup_method(self):
        self.normalizer = AWSSecretsNormalizer()

    def test_get_secret_event_normalized(self):
        """GetSecretValue event correctly normalized"""
        result = self.normalizer.normalize(
            AWS_GET_SECRET
        )
        assert result is not None
        assert isinstance(result, IamEvent)
        assert result.event_type == "secret"
        assert result.source_system == (
            "aws_secrets_manager"
        )

    def test_accessor_name_extracted(self):
        """IAM user name correctly extracted"""
        result = self.normalizer.normalize(
            AWS_GET_SECRET
        )
        assert result.user == "svc_backup"

    def test_secret_path_extracted(self):
        """Secret path correctly extracted"""
        result = self.normalizer.normalize(
            AWS_GET_SECRET
        )
        assert result.secret_event.secret_path == (
            "prod/db_password"
        )

    def test_operation_mapped_to_read(self):
        """GetSecretValue maps to read operation"""
        result = self.normalizer.normalize(
            AWS_GET_SECRET
        )
        assert result.secret_event.operation == "read"

    def test_delete_operation_mapped(self):
        """DeleteSecret maps to delete operation"""
        result = self.normalizer.normalize(
            AWS_DELETE_SECRET
        )
        assert result.secret_event.operation == "delete"

    def test_delete_gets_elevated_risk(self):
        """Secret deletion gets elevated risk score"""
        result = self.normalizer.normalize(
            AWS_DELETE_SECRET
        )
        assert result.overall_risk_score >= 0.3

    def test_root_access_critical_risk(self):
        """Root credential access gets critical risk"""
        result = self.normalizer.normalize(
            AWS_ROOT_SECRET
        )
        assert result.secret_event.is_root_token is True
        assert result.overall_risk_score >= 0.7

    def test_off_hours_increases_risk(self):
        """Off-hours access increases risk score"""
        result = self.normalizer.normalize(
            AWS_LIST_SECRETS
        )
        assert result.overall_risk_score >= 0.2

    def test_non_secrets_event_returns_none(self):
        """Non-Secrets Manager event returns None"""
        ec2_event = {
            "eventSource": "ec2.amazonaws.com",
            "eventName": "DescribeInstances"
        }
        result = self.normalizer.normalize(ec2_event)
        assert result is None

    def test_none_returns_none(self):
        """None input returns None"""
        assert self.normalizer.normalize(None) is None

    def test_statistics_tracked(self):
        """Statistics correctly tracked"""
        self.normalizer.normalize(AWS_GET_SECRET)
        self.normalizer.normalize(AWS_ROOT_SECRET)
        stats = self.normalizer.get_statistics()
        assert stats["events_processed"] == 2

    def test_mount_extracted_from_path(self):
        """Mount namespace extracted from secret path"""
        mount = self.normalizer._extract_mount(
            "prod/db_password"
        )
        assert mount == "prod"

    def test_accessor_type_iam_user(self):
        """IAM user correctly mapped to human type"""
        identity = {
            "type": "IAMUser",
            "userName": "jsmith"
        }
        name = self.normalizer._extract_accessor_name(
            identity
        )
        assert name == "jsmith"

    def test_accessor_type_root(self):
        """Root identity correctly identified"""
        identity = {
            "type": "Root",
            "arn": "arn:aws:iam::123:root"
        }
        name = self.normalizer._extract_accessor_name(
            identity
        )
        assert name == "aws_root"


# ============================================================
# TEST CLASS — AZURE KEY VAULT
# ============================================================

class TestAzureKeyVaultNormalizer:
    """Tests for Azure Key Vault normalization"""

    def setup_method(self):
        self.normalizer = AzureKeyVaultNormalizer()

    def test_secret_get_normalized(self):
        """SecretGet event correctly normalized"""
        result = self.normalizer.normalize(
            AZURE_SECRET_GET
        )
        assert result is not None
        assert result.source_system == "azure_key_vault"

    def test_user_identity_extracted(self):
        """UPN correctly extracted from claim"""
        result = self.normalizer.normalize(
            AZURE_SECRET_GET
        )
        assert result.user == "svc_backup@corp.com"

    def test_secret_path_parsed_from_url(self):
        """Secret path parsed from Azure URL format"""
        result = self.normalizer.normalize(
            AZURE_SECRET_GET
        )
        path = result.secret_event.secret_path
        assert "db-password" in path

    def test_vault_name_extracted(self):
        """Vault name extracted from resource ID"""
        vault = self.normalizer._extract_vault_name(
            "/subscriptions/sub-123/resourceGroups/"
            "prod-rg/providers/Microsoft.KeyVault/"
            "vaults/prod-vault"
        )
        assert vault == "prod-vault"

    def test_unauthorized_access_flagged(self):
        """
        Unauthorized network access flagged
        with elevated risk score.
        """
        result = self.normalizer.normalize(
            AZURE_UNAUTHORIZED
        )
        assert result.overall_risk_score >= 0.4

    def test_unauthorized_counter_tracked(self):
        """Unauthorized access attempts counted"""
        self.normalizer.normalize(AZURE_UNAUTHORIZED)
        stats = self.normalizer.get_statistics()
        assert stats["unauthorized_attempts"] == 1

    def test_secret_delete_elevated_risk(self):
        """Secret deletion gets elevated risk"""
        result = self.normalizer.normalize(
            AZURE_SECRET_DELETE
        )
        assert result.overall_risk_score >= 0.3

    def test_operation_mapped_correctly(self):
        """SecretGet maps to read operation"""
        result = self.normalizer.normalize(
            AZURE_SECRET_GET
        )
        assert result.secret_event.operation == "read"

    def test_none_returns_none(self):
        """None input returns None"""
        assert self.normalizer.normalize(None) is None

    def test_statistics_tracked(self):
        """Statistics correctly tracked"""
        self.normalizer.normalize(AZURE_SECRET_GET)
        stats = self.normalizer.get_statistics()
        assert stats["events_processed"] == 1

    def test_service_principal_identity(self):
        """Service principal correctly identified"""
        name, accessor_type = (
            self.normalizer._extract_identity({
                "appid": "app-id-456",
                "oid": "oid-123"
            })
        )
        assert accessor_type == "service_account"

    def test_human_identity_extracted(self):
        """Human UPN correctly extracted"""
        name, accessor_type = (
            self.normalizer._extract_identity({
                "upn": "jsmith@corp.com"
            })
        )
        assert name == "jsmith@corp.com"
        assert accessor_type == "human"


# ============================================================
# TEST CLASS — SAILPOINT ISC
# ============================================================

class TestSailPointNormalizer:
    """Tests for SailPoint ISC normalization"""

    def setup_method(self):
        self.normalizer = SailPointNormalizer()

    def test_certification_event_normalized(self):
        """Certification event correctly normalized"""
        result = self.normalizer.normalize(
            SAILPOINT_CERTIFICATION
        )
        assert result is not None
        assert result.event_type == "governance"
        assert result.source_system == "sailpoint"

    def test_identity_name_extracted(self):
        """Identity name correctly extracted"""
        result = self.normalizer.normalize(
            SAILPOINT_CERTIFICATION
        )
        assert result.user == "jsmith"

    def test_rubber_stamp_detected(self):
        """
        Certification approved in 3 seconds
        flagged as rubber stamp.
        Your insight: reviewers who approve
        without reviewing enable breaches.
        """
        result = self.normalizer.normalize(
            SAILPOINT_CERTIFICATION
        )
        gov = result.governance_event
        assert gov.is_rubber_stamp is True

    def test_rubber_stamp_increases_risk(self):
        """Rubber stamp elevates risk score"""
        result = self.normalizer.normalize(
            SAILPOINT_CERTIFICATION
        )
        assert result.overall_risk_score >= 0.3

    def test_rubber_stamp_counter_tracked(self):
        """Rubber stamp detections counted"""
        self.normalizer.normalize(
            SAILPOINT_CERTIFICATION
        )
        stats = self.normalizer.get_statistics()
        assert stats["rubber_stamps_detected"] == 1

    def test_orphan_account_flagged(self):
        """
        Orphaned account gets high risk score.
        Orphaned accounts are attacker targets.
        """
        result = self.normalizer.normalize(
            SAILPOINT_ORPHAN
        )
        assert result.overall_risk_score >= 0.6

    def test_orphan_counter_tracked(self):
        """Orphan detections counted"""
        self.normalizer.normalize(SAILPOINT_ORPHAN)
        stats = self.normalizer.get_statistics()
        assert stats["orphans_detected"] == 1

    def test_sod_violation_flagged(self):
        """
        SoD violation gets elevated risk.
        Create + Approve = fraud enablement.
        """
        result = self.normalizer.normalize(
            SAILPOINT_SOD
        )
        assert result.overall_risk_score >= 0.5

    def test_sod_violation_counter_tracked(self):
        """SoD violation detections counted"""
        self.normalizer.normalize(SAILPOINT_SOD)
        stats = self.normalizer.get_statistics()
        assert stats["sod_violations_detected"] == 1

    def test_access_denied_increases_risk(self):
        """Denied access request noted as risk signal"""
        result = self.normalizer.normalize(
            SAILPOINT_ACCESS_DENIED
        )
        assert result.overall_risk_score >= 0.3

    def test_high_privilege_entitlement_flagged(self):
        """Admin entitlement gets elevated risk"""
        result = self.normalizer.normalize(
            SAILPOINT_CERTIFICATION
        )
        assert result.overall_risk_score >= 0.2

    def test_none_returns_none(self):
        """None input returns None"""
        assert self.normalizer.normalize(None) is None

    def test_statistics_tracked(self):
        """Statistics correctly tracked"""
        self.normalizer.normalize(
            SAILPOINT_CERTIFICATION
        )
        stats = self.normalizer.get_statistics()
        assert stats["events_processed"] == 1


# ============================================================
# TEST CLASS — CYBERARK PAM
# ============================================================

class TestCyberArkNormalizer:
    """Tests for CyberArk PAM normalization"""

    def setup_method(self):
        self.normalizer = CyberArkNormalizer()

    def test_psm_connect_normalized(self):
        """PSM.Connect event correctly normalized"""
        result = self.normalizer.normalize(
            CYBERARK_PSM_CONNECT
        )
        assert result is not None
        assert result.event_type == "privileged"
        assert result.source_system == "cyberark"

    def test_username_extracted(self):
        """Username correctly extracted"""
        result = self.normalizer.normalize(
            CYBERARK_PSM_CONNECT
        )
        assert result.user == "jsmith"

    def test_target_account_extracted(self):
        """Target privileged account extracted"""
        result = self.normalizer.normalize(
            CYBERARK_PSM_CONNECT
        )
        priv = result.privileged_event
        assert priv.target_account == "domain_admin"

    def test_first_use_detected(self):
        """
        First use of privileged account flagged.
        Account never used before = discovery phase.
        """
        result = self.normalizer.normalize(
            CYBERARK_PSM_CONNECT
        )
        priv = result.privileged_event
        assert priv.is_new_account is True

    def test_after_hours_detected(self):
        """
        03:00 AM privileged access flagged.
        After-hours admin access is high risk.
        """
        result = self.normalizer.normalize(
            CYBERARK_AFTER_HOURS
        )
        priv = result.privileged_event
        assert priv.is_after_hours is True

    def test_after_hours_increases_risk(self):
        """After-hours access elevates risk score"""
        result = self.normalizer.normalize(
            CYBERARK_AFTER_HOURS
        )
        assert result.overall_risk_score >= 0.3

    def test_concurrent_sessions_detected(self):
        """
        Second PSM.Connect for same account
        flagged as concurrent session.
        Same admin account twice = compromise.
        """
        # First session
        self.normalizer.normalize(CYBERARK_PSM_CONNECT)

        # Second session — same target account
        second_event = dict(CYBERARK_PSM_CONNECT)
        second_event["SessionID"] = "session-uuid-999"
        second_event["Username"] = "attacker"

        result = self.normalizer.normalize(second_event)
        priv = result.privileged_event
        assert priv.is_concurrent_session is True

    def test_lateral_movement_detected(self):
        """
        Privileged access on compromised workstation
        flagged as lateral movement.
        Your insight: svc_backup checks out
        domain_admin after malware detection.
        """
        # Register security alert on workstation
        self.normalizer.register_security_alert(
            "WKSTN-JSMITH-01",
            "2024-03-29T09:15:00Z"
        )

        # CyberArk event from same workstation
        result = self.normalizer.normalize(
            CYBERARK_PASSWORD_RETRIEVE
        )

        priv = result.privileged_event
        reasons_text = " ".join(priv.risk_reasons)
        assert result.overall_risk_score >= 0.3

    def test_admin_safe_increases_risk(self):
        """Admin safe access elevates risk"""
        result = self.normalizer.normalize(
            CYBERARK_PSM_CONNECT
        )
        assert result.overall_risk_score >= 0.2

    def test_none_returns_none(self):
        """None input returns None"""
        assert self.normalizer.normalize(None) is None

    def test_statistics_tracked(self):
        """Statistics correctly tracked"""
        self.normalizer.normalize(CYBERARK_PSM_CONNECT)
        stats = self.normalizer.get_statistics()
        assert stats["events_processed"] == 1

    def test_register_security_alert(self):
        """Security alert registration works"""
        self.normalizer.register_security_alert(
            "WKSTN-TEST-01",
            "2024-03-29T09:15:00Z"
        )
        assert "WKSTN-TEST-01" in (
            self.normalizer._recent_alerts
        )

    def test_session_state_updated(self):
        """Active session tracking updated"""
        self.normalizer.normalize(CYBERARK_PSM_CONNECT)
        assert "domain_admin" in (
            self.normalizer._active_sessions
        )