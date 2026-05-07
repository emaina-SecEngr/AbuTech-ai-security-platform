"""
Entra ID Normalizer Tests

Each test proves one specific concept:

TestEntraIDSignIn:
    - Basic field extraction
    - Microsoft risk score integration
    - Impossible travel detection
    - Password spray detection
    - Legacy auth detection
    - New device flagging

TestEntraIDAudit:
    - Admin role assignment detection
    - App registration detection
    - After-hours change detection
    - High risk operation flagging
"""

import pytest
from layer1_ingestion.normalizers.entraid_normalizer import (
    EntraIDSignInNormalizer,
    EntraIDAuditNormalizer
)
from layer1_ingestion.schema.iam_schema import (
    IamEvent,
    IamAuthEvent,
    IamGovernanceEvent
)


# ============================================================
# SAMPLE ENTRA ID EVENTS
# ============================================================

SIGNIN_SUCCESS_NY = {
    "id": "signin-001",
    "createdDateTime": "2024-03-29T08:42:00Z",
    "userPrincipalName": "jsmith@corp.com",
    "userDisplayName": "John Smith",
    "userId": "user-id-123",
    "ipAddress": "10.0.0.155",
    "location": {
        "city": "New York",
        "countryOrRegion": "US",
        "geoCoordinates": {
            "latitude": 40.7128,
            "longitude": -74.0060
        }
    },
    "status": {"errorCode": 0},
    "conditionalAccessStatus": "success",
    "authenticationRequirement": (
        "multiFactorAuthentication"
    ),
    "riskLevelAggregated": "none",
    "riskState": "none",
    "clientAppUsed": "Browser",
    "appDisplayName": "Microsoft 365",
    "deviceDetail": {
        "deviceId": "device-uuid-001",
        "displayName": "WKSTN-JSMITH-01",
        "operatingSystem": "Windows10"
    }
}

SIGNIN_SUCCESS_ROMANIA = {
    "id": "signin-002",
    "createdDateTime": "2024-03-29T09:14:00Z",
    "userPrincipalName": "jsmith@corp.com",
    "userDisplayName": "John Smith",
    "userId": "user-id-123",
    "ipAddress": "185.220.101.45",
    "location": {
        "city": "Bucharest",
        "countryOrRegion": "RO",
        "geoCoordinates": {
            "latitude": 44.4268,
            "longitude": 26.1025
        }
    },
    "status": {"errorCode": 0},
    "conditionalAccessStatus": "notApplied",
    "authenticationRequirement": "singleFactorAuthentication",
    "riskLevelAggregated": "high",
    "riskState": "atRisk",
    "clientAppUsed": "Browser",
    "appDisplayName": "AdminPanel",
    "deviceDetail": {
        "deviceId": "unknown-device-999",
        "displayName": "Unknown Device",
        "operatingSystem": "Unknown"
    }
}

SIGNIN_FAILURE = {
    "id": "signin-003",
    "createdDateTime": "2024-03-29T09:15:00Z",
    "userPrincipalName": "jsmith@corp.com",
    "userDisplayName": "John Smith",
    "userId": "user-id-123",
    "ipAddress": "185.220.101.45",
    "location": {
        "city": "Bucharest",
        "countryOrRegion": "RO",
        "geoCoordinates": {
            "latitude": 44.4268,
            "longitude": 26.1025
        }
    },
    "status": {
        "errorCode": 50126,
        "failureReason": "Invalid credentials"
    },
    "conditionalAccessStatus": "notApplied",
    "authenticationRequirement": "singleFactorAuthentication",
    "riskLevelAggregated": "medium",
    "riskState": "atRisk",
    "clientAppUsed": "Browser",
    "appDisplayName": "Microsoft 365",
    "deviceDetail": {
        "deviceId": "",
        "displayName": "",
        "operatingSystem": ""
    }
}

SIGNIN_LEGACY_AUTH = {
    "id": "signin-004",
    "createdDateTime": "2024-03-29T09:16:00Z",
    "userPrincipalName": "jsmith@corp.com",
    "userDisplayName": "John Smith",
    "userId": "user-id-123",
    "ipAddress": "10.0.0.155",
    "location": {
        "city": "New York",
        "countryOrRegion": "US",
        "geoCoordinates": {
            "latitude": 40.7128,
            "longitude": -74.0060
        }
    },
    "status": {"errorCode": 0},
    "conditionalAccessStatus": "notApplied",
    "authenticationRequirement": "singleFactorAuthentication",
    "riskLevelAggregated": "none",
    "riskState": "none",
    "clientAppUsed": "Exchange ActiveSync",
    "appDisplayName": "Exchange",
    "deviceDetail": {
        "deviceId": "mobile-001",
        "displayName": "iPhone",
        "operatingSystem": "iOS"
    }
}

AUDIT_ADMIN_ROLE = {
    "id": "audit-001",
    "activityDateTime": "2024-03-29T09:21:00Z",
    "activityDisplayName": "Add member to role",
    "category": "RoleManagement",
    "operationType": "Assign",
    "result": "success",
    "initiatedBy": {
        "user": {
            "userPrincipalName": "jsmith@corp.com",
            "id": "user-id-123"
        }
    },
    "targetResources": [
        {
            "displayName": "Global Administrator",
            "type": "Role"
        }
    ]
}

AUDIT_APP_REGISTRATION = {
    "id": "audit-002",
    "activityDateTime": "2024-03-29T09:22:00Z",
    "activityDisplayName": "Add service principal credentials",
    "category": "ApplicationManagement",
    "operationType": "Add",
    "result": "success",
    "initiatedBy": {
        "user": {
            "userPrincipalName": "jsmith@corp.com"
        }
    },
    "targetResources": [
        {
            "displayName": "HR Portal App",
            "type": "ServicePrincipal"
        }
    ]
}

AUDIT_AFTER_HOURS = {
    "id": "audit-003",
    "activityDateTime": "2024-03-29T03:00:00Z",
    "activityDisplayName": "Update application",
    "category": "ApplicationManagement",
    "operationType": "Update",
    "result": "success",
    "initiatedBy": {
        "user": {
            "userPrincipalName": "admin@corp.com"
        }
    },
    "targetResources": [
        {
            "displayName": "Production App",
            "type": "Application"
        }
    ]
}


# ============================================================
# TEST CLASS — ENTRA ID SIGN-IN
# ============================================================

class TestEntraIDSignInNormalizer:
    """
    Tests for Entra ID Sign-in normalization.

    Each test proves one detection capability.
    """

    def setup_method(self):
        self.normalizer = EntraIDSignInNormalizer()

    def test_successful_signin_normalized(self):
        """
        Basic normalization works.
        This is the foundation test.
        If this fails nothing else matters.
        """
        result = self.normalizer.normalize(
            SIGNIN_SUCCESS_NY
        )
        assert result is not None
        assert isinstance(result, IamEvent)
        assert result.source_system == "entra_id"

    def test_user_email_extracted(self):
        """
        userPrincipalName correctly maps to user.
        This is the ETL mapping test.
        Microsoft calls it UPN.
        We call it user.
        Same data — different name.
        """
        result = self.normalizer.normalize(
            SIGNIN_SUCCESS_NY
        )
        assert result.user == "jsmith@corp.com"

    def test_geo_location_extracted(self):
        """
        Geographic coordinates correctly extracted.
        These are what the Haversine formula uses
        for impossible travel detection.
        Without correct coordinates
        the distance calculation is wrong.
        """
        result = self.normalizer.normalize(
            SIGNIN_SUCCESS_NY
        )
        geo = result.auth_event.geo
        assert geo.latitude == 40.7128
        assert geo.longitude == -74.0060
        assert geo.country_code == "US"

    def test_mfa_usage_detected(self):
        """
        MFA correctly detected from
        authenticationRequirement field.
        Microsoft uses a string value
        we convert to boolean.
        """
        result = self.normalizer.normalize(
            SIGNIN_SUCCESS_NY
        )
        assert result.auth_event.auth.mfa_used is True

    def test_failure_outcome_extracted(self):
        """
        Authentication failure correctly identified.
        errorCode 0 = success
        Any other code = failure
        """
        result = self.normalizer.normalize(
            SIGNIN_FAILURE
        )
        assert result.auth_event.outcome == "failure"

    def test_microsoft_risk_score_applied(self):
        """
        Microsoft's own risk assessment incorporated.
        HIGH risk level from Microsoft
        contributes to our combined score.

        This is ensemble scoring —
        two independent systems combined.
        """
        result = self.normalizer.normalize(
            SIGNIN_SUCCESS_ROMANIA
        )
        assert result.overall_risk_score > 0.0

    def test_impossible_travel_detected(self):
        """
        New York to Romania in 32 minutes
        is physically impossible.

        This tests the core of our
        Haversine implementation.
        If this fails our distance
        calculation is broken.
        """
        self.normalizer.normalize(SIGNIN_SUCCESS_NY)
        result = self.normalizer.normalize(
            SIGNIN_SUCCESS_ROMANIA
        )
        assert (
            result.auth_event.is_impossible_travel
            is True
        )

    def test_impossible_travel_elevates_risk(self):
        """
        Impossible travel produces high risk score.
        The combination of:
        - Microsoft says HIGH risk
        - We detect impossible travel
        - New country
        should produce CRITICAL score.
        """
        self.normalizer.normalize(SIGNIN_SUCCESS_NY)
        result = self.normalizer.normalize(
            SIGNIN_SUCCESS_ROMANIA
        )
        assert result.overall_risk_score >= 0.6

    def test_new_country_flagged(self):
        """
        Romania is a new country for jsmith.
        After establishing US as known country
        Romania login should be flagged.
        """
        self.normalizer.normalize(SIGNIN_SUCCESS_NY)
        result = self.normalizer.normalize(
            SIGNIN_SUCCESS_ROMANIA
        )
        assert (
            result.auth_event.is_new_country is True
        )

    def test_legacy_auth_detected(self):
        """
        Exchange ActiveSync is a legacy protocol.
        Legacy auth cannot enforce MFA.
        Attackers deliberately use it to
        bypass MFA requirements.
        ATT&CK T1078 Valid Accounts.
        """
        result = self.normalizer.normalize(
            SIGNIN_LEGACY_AUTH
        )
        assert result.overall_risk_score >= 0.2

    def test_legacy_auth_counter_tracked(self):
        """Statistics track legacy auth events"""
        self.normalizer.normalize(SIGNIN_LEGACY_AUTH)
        stats = self.normalizer.get_statistics()
        assert stats["legacy_auth_detected"] == 1

    def test_password_spray_detected(self):
        """
        Multiple failures from multiple IPs
        detected as password spray.

        This tests our multi-IP tracking logic.
        Each failure from a different IP
        contributes to the spray detection.
        ATT&CK T1110.003 Password Spraying.
        """
        ips = [
            "1.1.1.1", "2.2.2.2", "3.3.3.3",
            "4.4.4.4", "5.5.5.5"
        ]
        for i, ip in enumerate(ips):
            failure = dict(SIGNIN_FAILURE)
            failure["id"] = f"fail-{i}"
            failure["ipAddress"] = ip
            self.normalizer.normalize(failure)

        stats = self.normalizer.get_statistics()
        assert stats["password_spray_detected"] >= 1

    def test_new_device_flagged(self):
        """
        Unknown device flagged on sign-in.
        New device = potential account takeover.
        """
        self.normalizer.normalize(SIGNIN_SUCCESS_NY)
        result = self.normalizer.normalize(
            SIGNIN_SUCCESS_ROMANIA
        )
        assert (
            result.auth_event.is_new_device is True
        )

    def test_none_returns_none(self):
        """None input handled gracefully"""
        assert self.normalizer.normalize(None) is None

    def test_statistics_tracked(self):
        """Processing statistics correctly counted"""
        self.normalizer.normalize(SIGNIN_SUCCESS_NY)
        self.normalizer.normalize(SIGNIN_FAILURE)
        stats = self.normalizer.get_statistics()
        assert stats["events_processed"] == 2


# ============================================================
# TEST CLASS — ENTRA ID AUDIT
# ============================================================

class TestEntraIDAuditNormalizer:
    """
    Tests for Entra ID Audit log normalization.

    Audit logs catch what happens AFTER
    an attacker gets in.
    These tests prove we detect persistence
    and privilege escalation.
    """

    def setup_method(self):
        self.normalizer = EntraIDAuditNormalizer()

    def test_audit_event_normalized(self):
        """Basic audit event normalization works"""
        result = self.normalizer.normalize(
            AUDIT_ADMIN_ROLE
        )
        assert result is not None
        assert result.event_type == "governance"
        assert result.source_system == (
            "entra_id_audit"
        )

    def test_actor_extracted(self):
        """
        Who made the change correctly extracted.
        initiatedBy.user.userPrincipalName
        is the actor in Entra ID audit logs.
        """
        result = self.normalizer.normalize(
            AUDIT_ADMIN_ROLE
        )
        assert result.user == "jsmith@corp.com"

    def test_admin_role_assignment_flagged(self):
        """
        Global Administrator role assignment
        gets high risk score.

        This is privilege escalation.
        Attacker granting themselves admin
        after initial compromise.
        ATT&CK T1078 Valid Accounts.
        """
        result = self.normalizer.normalize(
            AUDIT_ADMIN_ROLE
        )
        assert result.overall_risk_score >= 0.6

    def test_admin_role_counter_tracked(self):
        """Admin role assignments counted"""
        self.normalizer.normalize(AUDIT_ADMIN_ROLE)
        stats = self.normalizer.get_statistics()
        assert stats["admin_role_assignments"] == 1

    def test_app_registration_flagged(self):
        """
        Service principal credential addition flagged.

        This is OAuth persistence.
        Attacker adds credentials to an app
        so they maintain access even after
        password reset.
        ATT&CK T1528 Steal Application Access Token.

        As you correctly identified:
        This is more dangerous than password theft
        because it survives password resets.
        """
        result = self.normalizer.normalize(
            AUDIT_APP_REGISTRATION
        )
        assert result.overall_risk_score >= 0.3

    def test_app_registration_counter_tracked(self):
        """App registrations counted"""
        self.normalizer.normalize(
            AUDIT_APP_REGISTRATION
        )
        stats = self.normalizer.get_statistics()
        assert stats["app_registrations"] == 1

    def test_after_hours_change_flagged(self):
        """
        Administrative change at 03:00 AM flagged.
        Legitimate admins do not change production
        app configurations at 3am.
        Attackers do.
        """
        result = self.normalizer.normalize(
            AUDIT_AFTER_HOURS
        )
        assert result.overall_risk_score >= 0.2

    def test_target_resource_extracted(self):
        """Target of the change correctly extracted"""
        result = self.normalizer.normalize(
            AUDIT_ADMIN_ROLE
        )
        gov = result.governance_event
        assert "Global Administrator" in (
            gov.entitlement
        )

    def test_none_returns_none(self):
        """None input handled gracefully"""
        assert self.normalizer.normalize(None) is None

    def test_statistics_tracked(self):
        """Statistics correctly tracked"""
        self.normalizer.normalize(AUDIT_ADMIN_ROLE)
        self.normalizer.normalize(
            AUDIT_APP_REGISTRATION
        )
        stats = self.normalizer.get_statistics()
        assert stats["events_processed"] == 2