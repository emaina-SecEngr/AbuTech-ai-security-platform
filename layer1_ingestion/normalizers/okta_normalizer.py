"""
Layer 1 — Data Ingestion
Okta System Log Normalizer

This module transforms raw Okta System Log events
into normalized IamAuthEvent objects.

Okta System Log Structure:
    Every Okta event has this structure:
    {
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
            "geographicalContext": {
                "country": "United States",
                "city": "New York",
                "geolocation": {
                    "lat": 40.7128,
                    "lon": -74.0060
                }
            },
            "device": "Computer",
            "userAgent": {
                "browser": "Chrome",
                "os": "Windows 10"
            }
        },
        "authenticationContext": {
            "authenticationStep": 0,
            "externalSessionId": "session_abc123",
            "authenticationProvider": "OKTA",
            "credentialType": "PASSWORD"
        },
        "outcome": {
            "result": "SUCCESS",
            "reason": null
        },
        "target": [
            {
                "type": "AppInstance",
                "displayName": "SharePoint"
            }
        ]
    }

ETL Mapping:
    Okta "actor.login"           → user_email
    Okta "client.ipAddress"      → geo.ip_address
    Okta "client.geographicalContext.country"
                                 → geo.country_name
    Okta "outcome.result"        → outcome
    Okta "authenticationContext" → auth context

Impossible Travel Detection:
    This normalizer tracks the last known location
    per user and calculates travel distance and
    speed for each new authentication event.
    Impossible travel is flagged here in Layer 1
    before the event reaches Layer 2 ML models.
"""

import logging
import math
from datetime import datetime
from datetime import timezone
from typing import Optional

from layer1_ingestion.schema.iam_schema import (
    IamAuthEvent,
    IamEvent,
    GeoLocation,
    AuthContext
)

logger = logging.getLogger(__name__)


# ============================================================
# OKTA EVENT TYPE MAPPINGS
# Maps Okta eventType strings to readable actions
# ============================================================

OKTA_EVENT_ACTIONS = {
    "user.authentication.sso": "sso_login",
    "user.authentication.auth_via_mfa": "mfa_verify",
    "user.authentication.sso_error": "sso_failure",
    "user.mfa.factor.activate": "mfa_activate",
    "user.mfa.factor.deactivate": "mfa_deactivate",
    "user.mfa.factor.update": "mfa_update",
    "user.session.start": "session_start",
    "user.session.end": "session_end",
    "user.account.lock": "account_locked",
    "user.account.unlock": "account_unlocked",
    "user.account.reset_password": "password_reset",
    "policy.evaluate_sign_on": "policy_eval",
    "application.user_membership.add": "app_access_grant",
    "user.authentication.auth_via_radius": "radius_auth"
}

# High risk Okta event types
HIGH_RISK_EVENTS = {
    "user.account.lock",
    "user.mfa.factor.deactivate",
    "user.authentication.sso_error",
    "security.threat.detected",
    "user.account.reset_password_auth_via_social"
}

# MFA method mappings
MFA_METHODS = {
    "OKTA": "okta_verify",
    "GOOGLE": "google_authenticator",
    "SMS": "sms",
    "CALL": "phone_call",
    "EMAIL": "email",
    "FIDO2": "hardware_token",
    "YUBIKEY": "hardware_token",
    "WINDOWS_HELLO": "biometric"
}


class OktaNormalizer:
    """
    Normalizes Okta System Log events to IamAuthEvent.

    Implements:
    - Field mapping from Okta schema to IAM schema
    - Impossible travel detection
    - MFA bypass detection
    - New device and location flagging
    - Risk scoring based on behavioral signals

    Usage:
        normalizer = OktaNormalizer()
        iam_event = normalizer.normalize(raw_okta_event)

        if iam_event.overall_risk_score >= 0.7:
            # High risk authentication
            alert(iam_event)
    """

    def __init__(self):
        # Track last known location per user
        # Used for impossible travel detection
        self._user_locations = {}
        # key: user_email
        # value: {lat, lon, timestamp, country}

        # Track known devices per user
        self._user_devices = {}
        # key: user_email
        # value: set of device_ids

        # Track known countries per user
        self._user_countries = {}
        # key: user_email
        # value: set of country_codes

        # Statistics
        self.events_processed = 0
        self.high_risk_events = 0
        self.impossible_travel_detected = 0

        logger.info("OktaNormalizer initialized")

    def normalize(
        self,
        raw_event: dict
    ) -> Optional[IamEvent]:
        """
        Normalize a raw Okta System Log event.

        Full ETL pipeline:
        1. Extract fields from Okta JSON
        2. Map to IamAuthEvent schema
        3. Calculate behavioral signals
        4. Detect impossible travel
        5. Score overall risk
        6. Return IamEvent container

        Args:
            raw_event: Raw Okta System Log dict

        Returns:
            IamEvent or None if extraction fails
        """
        if not raw_event:
            return None

        try:
            # ---- EXTRACT CORE FIELDS ----
            event_type = raw_event.get(
                "eventType", ""
            )
            published = raw_event.get("published", "")

            # ---- EXTRACT ACTOR ----
            actor = raw_event.get("actor", {})
            user_id = actor.get("id", "")
            user_email = actor.get("login", "")
            user_name = actor.get(
                "displayName", user_email
            )

            # ---- EXTRACT CLIENT/GEO ----
            client = raw_event.get("client", {})
            ip_address = client.get("ipAddress", "")
            device_type = client.get("device", "")

            geo_context = client.get(
                "geographicalContext", {}
            )
            geo = self._extract_geo(
                ip_address, geo_context
            )

            # ---- EXTRACT AUTH CONTEXT ----
            auth_context_raw = raw_event.get(
                "authenticationContext", {}
            )
            outcome_raw = raw_event.get("outcome", {})
            outcome = outcome_raw.get(
                "result", "UNKNOWN"
            ).lower()

            auth = self._extract_auth_context(
                auth_context_raw, outcome
            )

            # ---- EXTRACT TARGET ----
            targets = raw_event.get("target", [])
            target_app = ""
            if targets:
                target_app = targets[0].get(
                    "displayName", ""
                )

            # ---- BUILD AUTH EVENT ----
            auth_event = IamAuthEvent(
                event_id=raw_event.get("uuid", ""),
                event_type=event_type,
                event_time=published,
                source_system="okta",
                user_id=user_id,
                user_email=user_email,
                user_name=user_name,
                user_display_name=user_name,
                action=OKTA_EVENT_ACTIONS.get(
                    event_type, event_type
                ),
                outcome=outcome,
                target_app=target_app,
                geo=geo,
                auth=auth,
                raw_event=raw_event
            )

            # ---- BEHAVIORAL SIGNALS ----
            self._calculate_behavioral_signals(
                auth_event, user_email, geo
            )

            # ---- RISK SCORING ----
            risk_score, risk_reasons = (
                self._calculate_risk_score(
                    auth_event, event_type
                )
            )

            auth_event.risk_score = risk_score
            auth_event.risk_label = (
                self._score_to_label(risk_score)
            )
            auth_event.risk_reasons = risk_reasons

            # ---- UPDATE USER HISTORY ----
            self._update_user_history(
                user_email, geo, published
            )

            # ---- BUILD IAM EVENT CONTAINER ----
            iam_event = IamEvent(
                event_type="auth",
                source_system="okta",
                timestamp=published,
                host=client.get("device", ""),
                user=user_email,
                auth_event=auth_event,
                overall_risk_score=risk_score,
                overall_risk_label=(
                    self._score_to_label(risk_score)
                ),
                risk_reasons=risk_reasons
            )

            self.events_processed += 1
            if risk_score >= 0.7:
                self.high_risk_events += 1

            logger.info(
                f"Okta event normalized: "
                f"{event_type} "
                f"user={user_email} "
                f"risk={risk_score:.2f}"
            )

            return iam_event

        except Exception as e:
            logger.error(
                f"Okta normalization failed: {e}"
            )
            return None

    # ============================================================
    # FIELD EXTRACTORS
    # ============================================================

    def _extract_geo(
        self,
        ip_address: str,
        geo_context: dict
    ) -> GeoLocation:
        """Extract geographic location from Okta event"""
        geolocation = geo_context.get(
            "geolocation", {}
        )

        return GeoLocation(
            ip_address=ip_address,
            country_code=geo_context.get(
                "countryCode", ""
            ),
            country_name=geo_context.get(
                "country", ""
            ),
            city=geo_context.get("city", ""),
            latitude=geolocation.get("lat"),
            longitude=geolocation.get("lon")
        )

    def _extract_auth_context(
        self,
        auth_context: dict,
        outcome: str
    ) -> AuthContext:
        """Extract authentication context"""
        credential_type = auth_context.get(
            "credentialType", ""
        )
        auth_provider = auth_context.get(
            "authenticationProvider", ""
        )

        mfa_used = credential_type in [
            "OTP", "FIDO2", "SMS", "CALL",
            "EMAIL", "YUBIKEY"
        ]

        mfa_method = MFA_METHODS.get(
            credential_type, credential_type.lower()
        )

        return AuthContext(
            auth_method=credential_type.lower(),
            mfa_used=mfa_used,
            mfa_method=mfa_method if mfa_used else "",
            session_id=auth_context.get(
                "externalSessionId", ""
            ),
            outcome=outcome
        )

    # ============================================================
    # BEHAVIORAL SIGNAL CALCULATION
    # ============================================================

    def _calculate_behavioral_signals(
        self,
        auth_event: IamAuthEvent,
        user_email: str,
        geo: GeoLocation
    ) -> None:
        """
        Calculate behavioral anomaly signals.

        Compares current event against user's
        historical behavior to identify anomalies.
        """
        # ---- NEW COUNTRY DETECTION ----
        if geo.country_code:
            known_countries = (
                self._user_countries.get(
                    user_email, set()
                )
            )
            if (
                known_countries and
                geo.country_code not in known_countries
            ):
                auth_event.is_new_country = True
                logger.warning(
                    f"New country detected: "
                    f"{user_email} from "
                    f"{geo.country_name}"
                )

        # ---- IMPOSSIBLE TRAVEL DETECTION ----
        last_location = self._user_locations.get(
            user_email
        )

        if (
            last_location and
            geo.latitude and
            geo.longitude
        ):
            distance_km = self._calculate_distance(
                last_location["lat"],
                last_location["lon"],
                geo.latitude,
                geo.longitude
            )

            time_elapsed_hours = (
                self._calculate_time_elapsed(
                    last_location["timestamp"],
                    auth_event.event_time
                )
            )

            if time_elapsed_hours > 0:
                speed_kmh = (
                    distance_km / time_elapsed_hours
                )

                auth_event.travel_distance_km = (
                    round(distance_km, 1)
                )

                # Fastest commercial plane: ~900 km/h
                # We use 800 km/h as threshold
                # to account for direct flights
                if (
                    distance_km > 100 and
                    speed_kmh > 800
                ):
                    auth_event.is_impossible_travel = True
                    auth_event.travel_speed_kmh = (
                        round(speed_kmh, 1)
                    )
                    self.impossible_travel_detected += 1

                    logger.warning(
                        f"IMPOSSIBLE TRAVEL: "
                        f"{user_email} "
                        f"distance={distance_km:.0f}km "
                        f"speed={speed_kmh:.0f}km/h "
                        f"in {time_elapsed_hours:.2f}h"
                    )

    def _calculate_risk_score(
        self,
        auth_event: IamAuthEvent,
        event_type: str
    ) -> tuple:
        """
        Calculate risk score from behavioral signals.

        Implements your radius/distance concept:
        Multiple signals combine into overall risk.

        Returns (risk_score, risk_reasons)
        """
        score = 0.0
        reasons = []

        # High risk event types
        if event_type in HIGH_RISK_EVENTS:
            score += 0.4
            reasons.append(
                f"High risk event type: {event_type}"
            )

        # Authentication failure
        if auth_event.outcome in [
            "failure", "denied", "locked"
        ]:
            score += 0.3
            reasons.append(
                f"Authentication failed: "
                f"{auth_event.outcome}"
            )

        # Impossible travel
        if auth_event.is_impossible_travel:
            score += 0.5
            reasons.append(
                f"Impossible travel detected: "
                f"{auth_event.travel_distance_km}km "
                f"in impossible time"
            )

        # New country
        if auth_event.is_new_country:
            score += 0.3
            reasons.append(
                f"Authentication from new country: "
                f"{auth_event.geo.country_name}"
            )

        # MFA not used when expected
        if (
            not auth_event.auth.mfa_used and
            auth_event.outcome == "success"
        ):
            score += 0.2
            reasons.append(
                "Authentication succeeded without MFA"
            )

        # New device
        if auth_event.is_new_device:
            score += 0.2
            reasons.append(
                "Authentication from new device"
            )

        # Tor or proxy
        if auth_event.geo:
            if auth_event.geo.is_tor:
                score += 0.4
                reasons.append(
                    "Authentication through Tor network"
                )
            if auth_event.geo.is_proxy:
                score += 0.2
                reasons.append(
                    "Authentication through proxy"
                )

        return min(score, 1.0), reasons

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    def _calculate_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate distance between two coordinates
        using Haversine formula.

        This is your radius concept —
        measuring how far a user has traveled
        between authentication events.

        Returns distance in kilometers.
        """
        R = 6371  # Earth radius in km

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_lat / 2) ** 2 +
            math.cos(lat1_rad) *
            math.cos(lat2_rad) *
            math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a),
                           math.sqrt(1 - a))

        return R * c

    def _calculate_time_elapsed(
        self,
        timestamp1: str,
        timestamp2: str
    ) -> float:
        """
        Calculate hours elapsed between timestamps.
        Returns 0.0 if timestamps cannot be parsed.
        """
        try:
            fmt = "%Y-%m-%dT%H:%M:%S.%fZ"

            t1 = datetime.strptime(
                timestamp1, fmt
            ).replace(tzinfo=timezone.utc)

            t2 = datetime.strptime(
                timestamp2, fmt
            ).replace(tzinfo=timezone.utc)

            elapsed = abs(
                (t2 - t1).total_seconds()
            ) / 3600

            return elapsed

        except Exception:
            return 0.0

    def _update_user_history(
        self,
        user_email: str,
        geo: GeoLocation,
        event_timestamp:str =""
    ) -> None:
        """Update user location and device history"""
        if geo.latitude and geo.longitude:
            self._user_locations[user_email] = {
                "lat": geo.latitude,
                "lon": geo.longitude,
                "timestamp": event_timestamp or self._now(),
                "country": geo.country_code
            }

        if geo.country_code:
            if user_email not in self._user_countries:
                self._user_countries[user_email] = set()
            self._user_countries[user_email].add(
                geo.country_code
            )

    def _score_to_label(self, score: float) -> str:
        """Convert risk score to label"""
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score > 0.0:
            return "LOW"
        return "UNKNOWN"

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )

    def get_statistics(self) -> dict:
        """Return normalizer statistics"""
        return {
            "events_processed": self.events_processed,
            "high_risk_events": self.high_risk_events,
            "impossible_travel_detected": (
                self.impossible_travel_detected
            ),
            "users_tracked": len(
                self._user_locations
            )
        }