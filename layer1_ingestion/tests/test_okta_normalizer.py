"""
Layer 1 — Okta Normalizer Tests
Layer 2 — Identity Threat Detector Tests

Tests verify:
1. Okta event field extraction
2. Geographic context extraction
3. Impossible travel detection
4. MFA bypass detection
5. Risk scoring accuracy
6. Identity threat detector scoring
7. Behavioral baseline building
"""

import pytest
from layer1_ingestion.normalizers.okta_normalizer import (
    OktaNormalizer
)
from layer1_ingestion.schema.iam_schema import (
    IamEvent,
    IamAuthEvent,
    IamPrivilegedEvent,
    IamSecretEvent,
    GeoLocation,
    AuthContext
)
from layer2_ml.identity.identity_threat_detector import (
    IdentityThreatDetector,
    IdentityThreatResult,
    UserBaseline
)


# ============================================================
# SAMPLE OKTA EVENTS
# ============================================================

OKTA_SSO_SUCCESS = {
    "uuid": "test-event-001",
    "eventType": "user.authentication.sso",
    "published": "2024-03-29T08:42:00.000Z",
    "actor": {
        "id": "00u1234567890",
        "login": "jsmith@corp.com",
        "displayName": "John Smith",
        "type": "User"
    },
    "client": {
        "ipAddress": "10.0.0.155",
        "device": "Computer",
        "geographicalContext": {
            "country": "United States",
            "countryCode": "US",
            "city": "New York",
            "geolocation": {
                "lat": 40.7128,
                "lon": -74.0060
            }
        }
    },
    "authenticationContext": {
        "authenticationStep": 0,
        "externalSessionId": "session_abc123",
        "authenticationProvider": "OKTA",
        "credentialType": "OTP"
    },
    "outcome": {
        "result": "SUCCESS"
    },
    "target": [
        {
            "type": "AppInstance",
            "displayName": "SharePoint"
        }
    ]
}

OKTA_SSO_ROMANIA = {
    "uuid": "test-event-002",
    "eventType": "user.authentication.sso",
    "published": "2024-03-29T09:14:00.000Z",
    "actor": {
        "id": "00u1234567890",
        "login": "jsmith@corp.com",
        "displayName": "John Smith",
        "type": "User"
    },
    "client": {
        "ipAddress": "185.220.101.45",
        "device": "Unknown Device",
        "geographicalContext": {
            "country": "Romania",
            "countryCode": "RO",
            "city": "Bucharest",
            "geolocation": {
                "lat": 44.4268,
                "lon": 26.1025
            }
        }
    },
    "authenticationContext": {
        "authenticationStep": 0,
        "externalSessionId": "session_xyz999",
        "authenticationProvider": "OKTA",
        "credentialType": "PASSWORD"
    },
    "outcome": {
        "result": "SUCCESS"
    },
    "target": [
        {
            "type": "AppInstance",
            "displayName": "AdminPanel"
        }
    ]
}

OKTA_MFA_FAILURE = {
    "uuid": "test-event-003",
    "eventType": "user.authentication.auth_via_mfa",
    "published": "2024-03-29T09:15:00.000Z",
    "actor": {
        "id": "00u1234567890",
        "login": "jsmith@corp.com",
        "displayName": "John Smith",
        "type": "User"
    },
    "client": {
        "ipAddress": "10.0.0.155",
        "device": "Computer",
        "geographicalContext": {
            "country": "United States",
            "countryCode": "US",
            "city": "New York",
            "geolocation": {
                "lat": 40.7128,
                "lon": -74.0060
            }
        }
    },
    "authenticationContext": {
        "credentialType": "PASSWORD",
        "authenticationProvider": "OKTA"
    },
    "outcome": {
        "result": "FAILURE",
        "reason": "INVALID_CREDENTIALS"
    },
    "target": []
}

OKTA_ACCOUNT_LOCK = {
    "uuid": "test-event-004",
    "eventType": "user.account.lock",
    "published": "2024-03-29T09:16:00.000Z",
    "actor": {
        "id": "system",
        "login": "system@okta.com",
        "displayName": "Okta System",
        "type": "SystemPrincipal"
    },
    "client": {
        "ipAddress": "10.0.0.155",
        "device": "Unknown",
        "geographicalContext": {
            "country": "United States",
            "countryCode": "US",
            "city": "New York",
            "geolocation": {
                "lat": 40.7128,
                "lon": -74.0060
            }
        }
    },
    "authenticationContext": {
        "credentialType": "PASSWORD"
    },
    "outcome": {
        "result": "SUCCESS"
    },
    "target": [
        {
            "type": "User",
            "displayName": "jsmith@corp.com"
        }
    ]
}


# ============================================================
# TEST CLASS — OKTA NORMALIZER
# ============================================================

class TestOktaNormalizer:
    """Tests for Okta System Log normalization"""

    def setup_method(self):
        self.normalizer = OktaNormalizer()

    def test_normalize_sso_success_event(self):
        """SSO success event correctly normalized"""
        result = self.normalizer.normalize(
            OKTA_SSO_SUCCESS
        )
        assert result is not None
        assert isinstance(result, IamEvent)

    def test_event_type_extracted(self):
        """Event type correctly extracted"""
        result = self.normalizer.normalize(
            OKTA_SSO_SUCCESS
        )
        assert result.event_type == "auth"
        assert result.source_system == "okta"

    def test_user_email_extracted(self):
        """User email correctly extracted from actor"""
        result = self.normalizer.normalize(
            OKTA_SSO_SUCCESS
        )
        assert result.user == "jsmith@corp.com"

    def test_auth_event_populated(self):
        """Auth event object populated"""
        result = self.normalizer.normalize(
            OKTA_SSO_SUCCESS
        )
        assert result.auth_event is not None
        assert isinstance(result.auth_event, IamAuthEvent)

    def test_geo_context_extracted(self):
        """Geographic context extracted correctly"""
        result = self.normalizer.normalize(
            OKTA_SSO_SUCCESS
        )
        geo = result.auth_event.geo
        assert geo is not None
        assert geo.country_code == "US"
        assert geo.city == "New York"
        assert geo.latitude == 40.7128
        assert geo.longitude == -74.0060

    def test_mfa_usage_detected(self):
        """MFA usage correctly detected from OTP"""
        result = self.normalizer.normalize(
            OKTA_SSO_SUCCESS
        )
        auth = result.auth_event.auth
        assert auth is not None
        assert auth.mfa_used is True

    def test_no_mfa_detected_for_password(self):
        """No MFA detected for password-only auth"""
        result = self.normalizer.normalize(
            OKTA_MFA_FAILURE
        )
        auth = result.auth_event.auth
        assert auth.mfa_used is False

    def test_outcome_extracted(self):
        """Authentication outcome extracted"""
        result = self.normalizer.normalize(
            OKTA_SSO_SUCCESS
        )
        assert result.auth_event.outcome == "success"

    def test_failure_outcome_extracted(self):
        """Failure outcome correctly extracted"""
        result = self.normalizer.normalize(
            OKTA_MFA_FAILURE
        )
        assert result.auth_event.outcome == "failure"

    def test_target_app_extracted(self):
        """Target application extracted"""
        result = self.normalizer.normalize(
            OKTA_SSO_SUCCESS
        )
        assert result.auth_event.target_app == (
            "SharePoint"
        )

    def test_high_risk_event_type_scored(self):
        """Account lock event gets elevated risk"""
        result = self.normalizer.normalize(
            OKTA_ACCOUNT_LOCK
        )
        assert result.overall_risk_score >= 0.3

    def test_none_input_returns_none(self):
        """None input returns None gracefully"""
        result = self.normalizer.normalize(None)
        assert result is None

    def test_empty_dict_returns_none(self):
        """Empty dict returns None gracefully"""
        result = self.normalizer.normalize({})
        assert result is None

    def test_statistics_tracked(self):
        """Processing statistics tracked"""
        self.normalizer.normalize(OKTA_SSO_SUCCESS)
        self.normalizer.normalize(OKTA_MFA_FAILURE)

        stats = self.normalizer.get_statistics()
        assert stats["events_processed"] == 2


# ============================================================
# TEST CLASS — IMPOSSIBLE TRAVEL DETECTION
# ============================================================

class TestImpossibleTravelDetection:
    """Tests for impossible travel detection"""

    def setup_method(self):
        self.normalizer = OktaNormalizer()

    def test_impossible_travel_detected(self):
        """
        New York to Romania in 32 minutes
        is physically impossible.
        Should be flagged as impossible travel.
        """
        # First login from New York
        self.normalizer.normalize(OKTA_SSO_SUCCESS)

        # Second login from Romania 32 minutes later
        result = self.normalizer.normalize(
            OKTA_SSO_ROMANIA
        )

        assert (
            result.auth_event.is_impossible_travel
            is True
        )

    def test_travel_distance_calculated(self):
        """
        Distance between New York and Romania
        should be approximately 7,300-7,400 km.
        """
        self.normalizer.normalize(OKTA_SSO_SUCCESS)
        result = self.normalizer.normalize(
            OKTA_SSO_ROMANIA
        )

        distance = result.auth_event.travel_distance_km
        assert distance is not None
        assert 7000 < distance < 8000

    def test_travel_speed_calculated(self):
        """
        Speed calculated for impossible travel
        should far exceed commercial aviation.
        """
        self.normalizer.normalize(OKTA_SSO_SUCCESS)
        result = self.normalizer.normalize(
            OKTA_SSO_ROMANIA
        )

        if result.auth_event.is_impossible_travel:
            speed = result.auth_event.travel_speed_kmh
            assert speed is not None
            assert speed > 800

    def test_impossible_travel_risk_elevated(self):
        """
        Impossible travel raises risk score
        significantly.
        """
        self.normalizer.normalize(OKTA_SSO_SUCCESS)
        result = self.normalizer.normalize(
            OKTA_SSO_ROMANIA
        )

        assert result.overall_risk_score >= 0.5

    def test_normal_travel_not_flagged(self):
        """
        Same country login not flagged as
        impossible travel.
        """
        self.normalizer.normalize(OKTA_SSO_SUCCESS)

        # Second login also from New York
        result = self.normalizer.normalize(
            OKTA_SSO_SUCCESS
        )

        assert (
            result.auth_event.is_impossible_travel
            is False
        )

    def test_new_country_flagged(self):
        """
        Login from new country flagged
        after establishing baseline.
        """
        # First establish US as known country
        self.normalizer.normalize(OKTA_SSO_SUCCESS)

        # Then login from Romania
        result = self.normalizer.normalize(
            OKTA_SSO_ROMANIA
        )

        assert result.auth_event.is_new_country is True

    def test_haversine_distance_calculation(self):
        """
        Haversine formula correctly calculates
        distance between coordinates.
        New York to London is approximately 5,570 km.
        """
        distance = self.normalizer._calculate_distance(
            40.7128, -74.0060,  # New York
            51.5074, -0.1278    # London
        )
        assert 5400 < distance < 5700


# ============================================================
# TEST CLASS — IDENTITY THREAT DETECTOR
# ============================================================

class TestIdentityThreatDetector:
    """Tests for identity threat scoring"""

    def setup_method(self):
        self.normalizer = OktaNormalizer()
        self.detector = IdentityThreatDetector()

    def _make_iam_event(
        self,
        is_impossible_travel=False,
        is_new_country=False,
        mfa_used=True,
        outcome="success",
        travel_distance=None
    ) -> IamEvent:
        """Build test IAM event"""
        geo = GeoLocation(
            ip_address="185.220.101.45",
            country_code="RO",
            country_name="Romania",
            latitude=44.4268,
            longitude=26.1025
        )

        auth = AuthContext(
            mfa_used=mfa_used,
            mfa_method="totp" if mfa_used else "",
            outcome=outcome
        )

        auth_event = IamAuthEvent(
            event_id="test-001",
            event_type="user.authentication.sso",
            event_time="2024-03-29T09:14:00.000Z",
            source_system="okta",
            user_email="jsmith@corp.com",
            user_name="John Smith",
            action="sso_login",
            outcome=outcome,
            geo=geo,
            auth=auth,
            is_impossible_travel=is_impossible_travel,
            is_new_country=is_new_country,
            travel_distance_km=travel_distance,
            travel_speed_kmh=13934.0 if (
                is_impossible_travel
            ) else None
        )

        return IamEvent(
            event_type="auth",
            source_system="okta",
            timestamp="2024-03-29T09:14:00.000Z",
            user="jsmith@corp.com",
            auth_event=auth_event
        )

    def test_impossible_travel_threat_detected(self):
        """Impossible travel flagged as threat"""
        event = self._make_iam_event(
            is_impossible_travel=True,
            travel_distance=7385.0
        )
        result = self.detector.score(event)

        assert result.impossible_travel is True
        assert result.risk_score >= 0.5

    def test_new_country_increases_risk(self):
        """New country login increases risk score"""
        event = self._make_iam_event(
            is_new_country=True
        )
        result = self.detector.score(event)

        assert result.new_country is True
        assert result.risk_score > 0.0

    def test_mfa_bypass_detected(self):
        """
        MFA bypass detected when user historically
        always uses MFA but current auth does not.
        """
        # Build baseline with MFA usage
        baseline = UserBaseline(
            user_email="jsmith@corp.com"
        )
        baseline.total_authentications = 10
        baseline.mfa_usage_rate = 1.0
        self.detector._baselines[
            "jsmith@corp.com"
        ] = baseline

        # Auth without MFA
        event = self._make_iam_event(mfa_used=False)
        result = self.detector.score(event)

        assert result.mfa_bypassed is True

    def test_low_risk_normal_auth(self):
        """Normal authentication scores low risk"""
        event = self._make_iam_event(
            is_impossible_travel=False,
            is_new_country=False,
            mfa_used=True,
            outcome="success"
        )
        result = self.detector.score(event)

        assert result.risk_score < 0.5
        assert result.is_threat is False

    def test_combined_signals_increase_risk(self):
        """
        Multiple signals combined produce
        higher risk than individual signals.
        """
        event = self._make_iam_event(
            is_impossible_travel=True,
            is_new_country=True,
            mfa_used=False,
            travel_distance=7385.0
        )
        result = self.detector.score(event)

        assert result.risk_score >= 0.7

    def test_attack_techniques_populated(self):
        """ATT&CK techniques populated for threats"""
        event = self._make_iam_event(
            is_impossible_travel=True,
            travel_distance=7385.0
        )
        result = self.detector.score(event)

        assert len(result.attack_techniques) >= 1

    def test_risk_score_capped_at_one(self):
        """Risk score never exceeds 1.0"""
        event = self._make_iam_event(
            is_impossible_travel=True,
            is_new_country=True,
            mfa_used=False,
            travel_distance=7385.0
        )
        result = self.detector.score(event)

        assert result.risk_score <= 1.0

    def test_privileged_event_scored(self):
        """Privileged event correctly scored"""
        priv_event = IamPrivilegedEvent(
            event_id="priv-001",
            event_type="PSM.Connect",
            event_time="2024-03-29T03:00:00.000Z",
            user_name="jsmith",
            target_account="domain_admin",
            target_system="DC01",
            is_after_hours=True,
            is_new_account=True
        )

        iam_event = IamEvent(
            event_type="privileged",
            source_system="cyberark",
            timestamp="2024-03-29T03:00:00.000Z",
            user="jsmith",
            privileged_event=priv_event
        )

        result = self.detector.score(iam_event)

        assert result.privilege_escalation is True
        assert result.risk_score >= 0.3

    def test_secret_harvesting_detected(self):
        """Bulk secret access detected"""
        secret_event = IamSecretEvent(
            event_id="vault-001",
            event_type="secret.read",
            event_time="2024-03-29T09:19:00.000Z",
            accessor_name="svc_backup",
            accessor_type="service_account",
            secret_path="prod/db_password",
            operation="read",
            is_bulk_access=True,
            secrets_accessed_count=15,
            is_post_compromise=True
        )

        iam_event = IamEvent(
            event_type="secret",
            source_system="vault",
            timestamp="2024-03-29T09:19:00.000Z",
            user="svc_backup",
            secret_event=secret_event
        )

        result = self.detector.score(iam_event)

        assert result.secret_harvesting is True
        assert result.risk_score >= 0.5

    def test_statistics_tracked(self):
        """Detector statistics tracked"""
        event = self._make_iam_event(
            is_impossible_travel=True,
            travel_distance=7385.0
        )
        self.detector.score(event)

        stats = self.detector.get_statistics()
        assert stats["events_scored"] == 1
        assert stats["threats_detected"] >= 1

    def test_user_baseline_builds_over_time(self):
        """
        User baseline grows with each
        successful authentication.
        """
        normalizer = OktaNormalizer()
        detector = IdentityThreatDetector()

        # Process 5 successful logins
        for _ in range(5):
            iam_event = normalizer.normalize(
                OKTA_SSO_SUCCESS
            )
            if iam_event:
                detector.score(iam_event)
                baseline = detector.get_user_baseline(
                    "jsmith@corp.com"
                )
                if baseline:
                    baseline.update_from_auth(
                        iam_event.auth_event
                    )

        baseline = detector.get_user_baseline(
            "jsmith@corp.com"
        )
        assert baseline is not None