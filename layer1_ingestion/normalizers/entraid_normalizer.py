"""
Layer 1 — Data Ingestion
Microsoft Entra ID Normalizer

WHY THIS FILE EXISTS:
    Microsoft Entra ID is the identity provider
    for 85% of Fortune 500 companies.
    Without this normalizer your platform is
    blind to Microsoft identity events.

    Two log types — two normalizers:
    Sign-in logs  → credential compromise detection
    Audit logs    → persistence and privilege detection

WHAT THIS FILE DOES:
    Transforms raw Entra ID JSON logs into
    IamAuthEvent and IamGovernanceEvent objects
    that your platform already understands.

    The key advantage over other IAM sources:
    Microsoft provides their OWN risk score
    with every sign-in event.
    We combine Microsoft's score with ours
    for higher confidence detections.

HOW IT CONNECTS:
    Layer 1: This normalizer → IamEvent
    Layer 2: IdentityThreatDetector scores it
    Layer 3: user node enriched with MS context
    Layer 4: IntelAgent sees MS attack patterns

Sign-in Log Structure:
{
    "createdDateTime": "2024-03-29T08:42:00Z",
    "userPrincipalName": "jsmith@corp.com",
    "userDisplayName": "John Smith",
    "ipAddress": "10.0.0.155",
    "location": {
        "city": "New York",
        "countryOrRegion": "US",
        "geoCoordinates": {
            "latitude": 40.7128,
            "longitude": -74.0060
        }
    },
    "status": {
        "errorCode": 0,
        "failureReason": null
    },
    "conditionalAccessStatus": "success",
    "authenticationRequirement": "multiFactorAuthentication",
    "riskLevelAggregated": "none",
    "riskState": "none",
    "clientAppUsed": "Browser",
    "appDisplayName": "Microsoft 365",
    "deviceDetail": {
        "deviceId": "device-uuid",
        "displayName": "WKSTN-JSMITH-01",
        "operatingSystem": "Windows10"
    }
}

Audit Log Structure:
{
    "activityDateTime": "2024-03-29T09:21:00Z",
    "activityDisplayName": "Add member to role",
    "category": "RoleManagement",
    "operationType": "Assign",
    "result": "success",
    "initiatedBy": {
        "user": {
            "userPrincipalName": "jsmith@corp.com"
        }
    },
    "targetResources": [
        {
            "displayName": "Global Administrator",
            "type": "Role"
        }
    ]
}
"""

import logging
import math
from datetime import datetime
from datetime import timezone
from typing import Optional

from layer1_ingestion.schema.iam_schema import (
    IamAuthEvent,
    IamGovernanceEvent,
    IamEvent,
    GeoLocation,
    AuthContext
)

logger = logging.getLogger(__name__)


# ============================================================
# ENTRA ID SPECIFIC CONSTANTS
#
# WHY THESE MATTER:
# Microsoft uses string labels for risk levels.
# We convert them to numeric scores so our
# IdentityThreatDetector can combine them
# with our own calculated scores.
# ============================================================

# Microsoft risk level to numeric score mapping
MICROSOFT_RISK_SCORES = {
    "none":    0.0,
    "low":     0.3,
    "medium":  0.6,
    "high":    0.9,
    "hidden":  0.5,
    "unknownFutureValue": 0.5
}

# Error codes that indicate authentication failure
# Full list: docs.microsoft.com/azure/active-directory
# /develop/reference-aadsts-error-codes
FAILURE_ERROR_CODES = {
    50126: "invalid_credentials",
    50053: "account_locked",
    50057: "account_disabled",
    50074: "mfa_required",
    50076: "mfa_required_location",
    70011: "invalid_scope",
    65001:  "consent_required",
    700016: "app_not_found"
}

# Legacy authentication protocols
# These cannot enforce MFA — always elevated risk
# ATT&CK T1078 Valid Accounts via legacy auth
LEGACY_AUTH_PROTOCOLS = {
    "Exchange ActiveSync",
    "SMTP Auth",
    "POP3",
    "IMAP4",
    "Older Office clients",
    "Exchange Web Services",
    "Autodiscover"
}

# High risk audit operations
# These indicate privilege escalation or persistence
HIGH_RISK_AUDIT_OPS = {
    "Add member to role",
    "Add app role assignment to service principal",
    "Add owner to application",
    "Add owner to service principal",
    "Set federation settings on domain",
    "Add unverified domain to company",
    "Disable strong authentication",
    "Update StsRefreshTokenValidFrom",
    "Add service principal credentials",
    "Update application"
}

# Admin roles — if assigned elevate risk significantly
ADMIN_ROLES = {
    "Global Administrator",
    "Privileged Role Administrator",
    "Security Administrator",
    "User Account Administrator",
    "Application Administrator",
    "Cloud Application Administrator",
    "Exchange Administrator",
    "SharePoint Administrator"
}

# Business hours UTC
BUSINESS_HOURS_START = 13
BUSINESS_HOURS_END = 23


class EntraIDSignInNormalizer:
    """
    Normalizes Entra ID Sign-in logs to IamAuthEvent.

    WHY SIGN-IN LOGS:
        Every authentication attempt is logged.
        This is where credential compromise shows
        up first — before the attacker does anything.

        Password spray: many failures → many IPs
        Credential stuffing: sequential failures
        Token theft: success from new location
        Legacy auth abuse: no MFA possible

    KEY FEATURE — Microsoft Risk Score Integration:
        Microsoft runs their own ML on every sign-in.
        We combine their score with ours.
        Two independent systems agreeing = high confidence.

    Usage:
        normalizer = EntraIDSignInNormalizer()
        iam_event = normalizer.normalize(
            raw_signin_log
        )
    """

    def __init__(self):
        # Track failed authentications per user
        # For password spray and stuffing detection
        self._failed_auths = {}

        # Track last known location per user
        # For impossible travel detection
        self._user_locations = {}

        # Track known devices per user
        self._user_devices = {}

        # Statistics
        self.events_processed = 0
        self.high_risk_events = 0
        self.legacy_auth_detected = 0
        self.password_spray_detected = 0

        logger.info(
            "EntraIDSignInNormalizer initialized"
        )

    def normalize(
        self,
        raw_event: dict
    ) -> Optional[IamEvent]:
        """
        Normalize Entra ID Sign-in log event.

        WHAT HAPPENS HERE:
        1. Extract user identity and location
        2. Determine if authentication succeeded
        3. Check for legacy authentication
        4. Detect password spray patterns
        5. Calculate impossible travel
        6. Combine Microsoft risk with our score
        7. Return IamEvent for Layer 2 scoring

        Args:
            raw_event: Raw Entra ID sign-in log dict

        Returns:
            IamEvent or None if extraction fails
        """
        if not raw_event:
            return None

        try:
            # ---- EXTRACT CORE FIELDS ----
            # createdDateTime = when the sign-in happened
            event_time = raw_event.get(
                "createdDateTime", ""
            )
            event_id = raw_event.get("id", "")

            # ---- EXTRACT IDENTITY ----
            # userPrincipalName is the unique identifier
            # in Microsoft's identity system
            # Usually email format: user@domain.com
            upn = raw_event.get(
                "userPrincipalName", ""
            )
            display_name = raw_event.get(
                "userDisplayName", upn
            )
            user_id = raw_event.get("userId", "")

            # ---- EXTRACT LOCATION ----
            # Microsoft provides richer geo data
            # than most other IAM vendors
            ip_address = raw_event.get(
                "ipAddress", ""
            )
            location = raw_event.get("location", {})
            geo = self._extract_geo(
                ip_address, location
            )

            # ---- EXTRACT AUTH RESULT ----
            # errorCode 0 = success
            # Any other code = failure
            status = raw_event.get("status", {})
            error_code = status.get("errorCode", 0)
            outcome = "success" if error_code == 0 \
                else "failure"
            failure_reason = FAILURE_ERROR_CODES.get(
                error_code,
                status.get("failureReason", "")
            )

            # ---- EXTRACT MFA STATUS ----
            # conditionalAccessStatus tells us if
            # Conditional Access policies enforced MFA
            ca_status = raw_event.get(
                "conditionalAccessStatus", ""
            )
            auth_requirement = raw_event.get(
                "authenticationRequirement", ""
            )
            mfa_used = (
                "multiFactorAuthentication"
                in auth_requirement or
                ca_status == "success"
            )

            # ---- DETECT LEGACY AUTHENTICATION ----
            # Legacy protocols cannot enforce MFA
            # Attackers deliberately use them to
            # bypass MFA requirements
            # ATT&CK T1078 Valid Accounts
            client_app = raw_event.get(
                "clientAppUsed", ""
            )
            is_legacy_auth = (
                client_app in LEGACY_AUTH_PROTOCOLS
            )
            if is_legacy_auth:
                self.legacy_auth_detected += 1
                logger.warning(
                    f"Legacy auth detected: "
                    f"{upn} via {client_app}"
                )

            # ---- EXTRACT DEVICE CONTEXT ----
            device = raw_event.get(
                "deviceDetail", {}
            )
            device_id = device.get("deviceId", "")
            device_name = device.get(
                "displayName", ""
            )
            is_new_device = self._is_new_device(
                upn, device_id
            )

            # ---- GET MICROSOFT RISK SCORE ----
            # This is unique to Entra ID
            # Microsoft runs their own behavioral ML
            # and tells us the risk level
            ms_risk_level = raw_event.get(
                "riskLevelAggregated", "none"
            )
            ms_risk_score = MICROSOFT_RISK_SCORES.get(
                ms_risk_level, 0.0
            )
            ms_risk_state = raw_event.get(
                "riskState", "none"
            )

            # ---- BUILD AUTH CONTEXT ----
            auth = AuthContext(
                auth_method=(
                    "legacy" if is_legacy_auth
                    else "modern"
                ),
                mfa_used=mfa_used,
                mfa_method=(
                    "conditional_access"
                    if mfa_used else ""
                ),
                device_id=device_id,
                device_name=device_name,
                is_new_device=is_new_device,
                outcome=outcome
            )

            # ---- BUILD AUTH EVENT ----
            auth_event = IamAuthEvent(
                event_id=event_id,
                event_type="microsoft.signin",
                event_time=event_time,
                source_system="entra_id",
                user_id=user_id,
                user_email=upn,
                user_name=upn,
                user_display_name=display_name,
                action="sign_in",
                outcome=outcome,
                target_app=raw_event.get(
                    "appDisplayName", ""
                ),
                geo=geo,
                auth=auth,
                is_new_device=is_new_device,
                raw_event=raw_event
            )

            # ---- DETECT BEHAVIORAL SIGNALS ----
            self._calculate_behavioral_signals(
                auth_event, upn, geo
            )

            # ---- CALCULATE PASSWORD SPRAY ----
            is_spray = False
            if outcome == "failure":
                is_spray = self._detect_password_spray(
                    upn, ip_address, event_time
                )
                if is_spray:
                    self.password_spray_detected += 1

            # ---- CALCULATE COMBINED RISK ----
            # KEY CONCEPT: We combine two independent
            # risk scores for higher confidence
            our_score, our_reasons = (
                self._calculate_risk(
                    auth_event,
                    ms_risk_score,
                    is_legacy_auth,
                    is_spray,
                    error_code,
                    ms_risk_state
                )
            )

            auth_event.risk_score = our_score
            auth_event.risk_label = (
                self._score_to_label(our_score)
            )
            auth_event.risk_reasons = our_reasons

            # ---- UPDATE HISTORY ----
            if outcome == "success":
                self._update_user_history(
                    upn, geo, device_id, event_time
                )

            # ---- BUILD IAM EVENT ----
            iam_event = IamEvent(
                event_type="auth",
                source_system="entra_id",
                timestamp=event_time,
                host=device_name,
                user=upn,
                auth_event=auth_event,
                overall_risk_score=our_score,
                overall_risk_label=(
                    self._score_to_label(our_score)
                ),
                risk_reasons=our_reasons
            )

            self.events_processed += 1
            if our_score >= 0.7:
                self.high_risk_events += 1

            logger.info(
                f"Entra sign-in normalized: "
                f"{upn} outcome={outcome} "
                f"ms_risk={ms_risk_level} "
                f"our_risk={our_score:.2f}"
            )

            return iam_event

        except Exception as e:
            logger.error(
                f"Entra sign-in normalization "
                f"failed: {e}"
            )
            return None

    # ============================================================
    # BEHAVIORAL DETECTION METHODS
    # ============================================================

    def _calculate_behavioral_signals(
        self,
        auth_event: IamAuthEvent,
        upn: str,
        geo: GeoLocation
    ) -> None:
        """
        Calculate behavioral anomaly signals.

        WHY THIS MATTERS:
        Microsoft provides their risk score but
        we add our own signals on top.
        Two independent systems provide higher
        confidence than either alone.
        """
        # New country detection
        if geo.country_code:
            known = self._user_locations.get(
                upn, {}
            ).get("known_countries", set())

            if known and geo.country_code not in known:
                auth_event.is_new_country = True

        # Impossible travel detection
        last = self._user_locations.get(upn)
        if (
            last and
            geo.latitude and
            geo.longitude and
            last.get("lat") and
            last.get("lon")
        ):
            distance = self._haversine(
                last["lat"], last["lon"],
                geo.latitude, geo.longitude
            )

            elapsed = self._hours_elapsed(
                last["timestamp"],
                auth_event.event_time
            )

            if elapsed > 0:
                speed = distance / elapsed
                auth_event.travel_distance_km = (
                    round(distance, 1)
                )

                if distance > 100 and speed > 800:
                    auth_event.is_impossible_travel = (
                        True
                    )
                    auth_event.travel_speed_kmh = (
                        round(speed, 1)
                    )
                    logger.warning(
                        f"IMPOSSIBLE TRAVEL: {upn} "
                        f"{distance:.0f}km in "
                        f"{elapsed:.2f}h"
                    )

    def _detect_password_spray(
        self,
        upn: str,
        ip_address: str,
        event_time: str
    ) -> bool:
        """
        Detect password spray attack.

        PASSWORD SPRAY vs BRUTE FORCE:

        Brute force: one account, many passwords
            attacker@evil.com tries:
            jsmith → password1
            jsmith → password2
            jsmith → password3
            → Account lockout triggers quickly

        Password spray: many accounts, one password
            attacker@evil.com tries:
            jsmith → Summer2024!
            bjones → Summer2024!
            mwilson → Summer2024!
            → No lockout because each account
              only fails once
            → Much harder to detect
            → This is what we catch here

        ATT&CK T1110.003 Password Spraying
        """
        if upn not in self._failed_auths:
            self._failed_auths[upn] = []

        self._failed_auths[upn].append({
            "time": event_time,
            "ip": ip_address
        })

        # Keep last 50 failures
        self._failed_auths[upn] = (
            self._failed_auths[upn][-50:]
        )

        recent = self._failed_auths[upn][-10:]
        unique_ips = len(set(
            f["ip"] for f in recent
        ))

        # Multiple failures from multiple IPs
        # = spray attack
        if len(recent) >= 5 and unique_ips >= 3:
            return True

        return False

    # ============================================================
    # RISK SCORING
    # ============================================================

    def _calculate_risk(
        self,
        auth_event: IamAuthEvent,
        ms_risk_score: float,
        is_legacy_auth: bool,
        is_spray: bool,
        error_code: int,
        ms_risk_state: str
    ) -> tuple:
        """
        Calculate combined risk score.

        KEY CONCEPT — ENSEMBLE SCORING:
        We combine Microsoft's risk assessment
        with our own behavioral signals.

        Microsoft sees:
        - Global threat intelligence
        - Historical patterns across all tenants
        - Token anomalies
        - Sign-in frequency patterns

        We add:
        - Impossible travel (Haversine)
        - New country detection
        - Legacy auth detection
        - Password spray patterns
        - Cross-source correlation (EDR + IAM)

        Combined = higher confidence than either alone
        """
        score = 0.0
        reasons = []

        # Microsoft's own risk assessment
        # Take their score as our starting point
        if ms_risk_score > 0:
            score += ms_risk_score * 0.6
            reasons.append(
                f"Microsoft risk assessment: "
                f"{ms_risk_state} "
                f"(score: {ms_risk_score:.1f})"
            )

        # Impossible travel — our detection
        if auth_event.is_impossible_travel:
            score += 0.4
            reasons.append(
                f"Impossible travel: "
                f"{auth_event.travel_distance_km}km "
                f"at {auth_event.travel_speed_kmh}km/h"
            )

        # New country
        if auth_event.is_new_country:
            score += 0.3
            reasons.append(
                f"Sign-in from new country: "
                f"{auth_event.geo.country_name if auth_event.geo else 'unknown'}"
            )

        # Legacy authentication — no MFA possible
        if is_legacy_auth:
            score += 0.3
            reasons.append(
                f"Legacy authentication protocol: "
                f"MFA cannot be enforced — "
                f"ATT&CK T1078"
            )

        # Password spray
        if is_spray:
            score += 0.5
            reasons.append(
                "Password spray attack detected: "
                "multiple failures from multiple IPs "
                "— ATT&CK T1110.003"
            )

        # Account locked
        if error_code == 50053:
            score += 0.3
            reasons.append(
                "Account locked — possible "
                "brute force attack"
            )

        # MFA not used on successful auth
        if (
            auth_event.outcome == "success" and
            not auth_event.auth.mfa_used
        ):
            score += 0.2
            reasons.append(
                "Successful authentication "
                "without MFA"
            )

        return min(score, 1.0), reasons

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    def _extract_geo(
        self,
        ip_address: str,
        location: dict
    ) -> GeoLocation:
        """Extract geographic context"""
        coords = location.get(
            "geoCoordinates", {}
        )
        return GeoLocation(
            ip_address=ip_address,
            country_code=location.get(
                "countryOrRegion", ""
            ),
            country_name=location.get(
                "countryOrRegion", ""
            ),
            city=location.get("city", ""),
            latitude=coords.get("latitude"),
            longitude=coords.get("longitude")
        )

    def _is_new_device(
        self,
        upn: str,
        device_id: str
    ) -> bool:
        """Check if this is a new device"""
        if not device_id:
            return False
        known = self._user_devices.get(upn, set())
        return device_id not in known

    def _haversine(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance in km"""
        R = 6371
        lat1r = math.radians(lat1)
        lat2r = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat/2)**2 +
            math.cos(lat1r) *
            math.cos(lat2r) *
            math.sin(dlon/2)**2
        )
        return R * 2 * math.atan2(
            math.sqrt(a), math.sqrt(1-a)
        )

    def _hours_elapsed(
        self,
        t1: str,
        t2: str
    ) -> float:
        """Calculate hours between timestamps"""
        try:
            fmt = "%Y-%m-%dT%H:%M:%SZ"
            dt1 = datetime.strptime(
                t1[:19], "%Y-%m-%dT%H:%M:%S"
            ).replace(tzinfo=timezone.utc)
            dt2 = datetime.strptime(
                t2[:19], "%Y-%m-%dT%H:%M:%S"
            ).replace(tzinfo=timezone.utc)
            return abs(
                (dt2-dt1).total_seconds()
            ) / 3600
        except Exception:
            return 0.0

    def _update_user_history(
        self,
        upn: str,
        geo: GeoLocation,
        device_id: str,
        event_time: str
    ) -> None:
        """Update user location and device history"""
        if upn not in self._user_locations:
            self._user_locations[upn] = {
                "known_countries": set()
            }

        if geo.latitude and geo.longitude:
            self._user_locations[upn].update({
                "lat": geo.latitude,
                "lon": geo.longitude,
                "timestamp": event_time
            })

        if geo.country_code:
            self._user_locations[upn][
                "known_countries"
            ].add(geo.country_code)

        if device_id:
            if upn not in self._user_devices:
                self._user_devices[upn] = set()
            self._user_devices[upn].add(device_id)

    def _score_to_label(self, score: float) -> str:
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score > 0.0:
            return "LOW"
        return "UNKNOWN"

    def get_statistics(self) -> dict:
        return {
            "events_processed": self.events_processed,
            "high_risk_events": self.high_risk_events,
            "legacy_auth_detected": (
                self.legacy_auth_detected
            ),
            "password_spray_detected": (
                self.password_spray_detected
            ),
            "users_tracked": len(
                self._user_locations
            )
        }


# ============================================================
# ENTRA ID AUDIT LOG NORMALIZER
#
# WHY THIS EXISTS:
#   Sign-in logs catch credential abuse.
#   Audit logs catch what happens AFTER
#   the attacker gets in.
#
#   Rogue app registration:
#   Attacker registers an OAuth app with
#   admin consent → persistent backdoor
#   even after password reset
#
#   Admin role assignment:
#   Attacker escalates to Global Admin
#   → full tenant compromise
#
#   These are PERSISTENCE mechanisms.
#   ATT&CK TA0003 Persistence
# ============================================================

class EntraIDAuditNormalizer:
    """
    Normalizes Entra ID Audit logs to
    IamGovernanceEvent objects.

    WHY AUDIT LOGS:
        After an attacker gains access they
        establish persistence.
        Audit logs catch every administrative
        change — app registrations, role
        assignments, policy changes.

        These are the FORENSIC evidence of
        what the attacker did after getting in.
    """

    def __init__(self):
        self.events_processed = 0
        self.high_risk_events = 0
        self.admin_role_assignments = 0
        self.app_registrations = 0

        logger.info(
            "EntraIDAuditNormalizer initialized"
        )

    def normalize(
        self,
        raw_event: dict
    ) -> Optional[IamEvent]:
        """
        Normalize Entra ID Audit log event.
        """
        if not raw_event:
            return None

        try:
            # ---- EXTRACT CORE FIELDS ----
            event_time = raw_event.get(
                "activityDateTime", ""
            )
            event_id = raw_event.get("id", "")
            activity = raw_event.get(
                "activityDisplayName", ""
            )
            category = raw_event.get("category", "")
            result = raw_event.get(
                "result", "unknown"
            )

            # ---- EXTRACT INITIATOR ----
            initiated_by = raw_event.get(
                "initiatedBy", {}
            )
            user_info = initiated_by.get("user", {})
            app_info = initiated_by.get("app", {})

            actor = (
                user_info.get(
                    "userPrincipalName", ""
                ) or
                app_info.get("displayName", "") or
                "unknown"
            )

            # ---- EXTRACT TARGETS ----
            targets = raw_event.get(
                "targetResources", []
            )
            target_name = ""
            target_type = ""
            if targets:
                target_name = targets[0].get(
                    "displayName", ""
                )
                target_type = targets[0].get(
                    "type", ""
                )

            # ---- DETECT HIGH RISK OPERATIONS ----
            is_high_risk = activity in (
                HIGH_RISK_AUDIT_OPS
            )

            # Admin role assignment
            is_admin_role = (
                "role" in activity.lower() and
                any(
                    role.lower() in target_name.lower()
                    for role in ADMIN_ROLES
                )
            )
            if is_admin_role:
                self.admin_role_assignments += 1
                logger.warning(
                    f"Admin role assignment: "
                    f"{actor} → {target_name}"
                )

            # App registration or credential add
            is_app_registration = (
                "application" in activity.lower() or
                "service principal" in activity.lower()
            )
            if is_app_registration:
                self.app_registrations += 1

            # ---- BUILD GOVERNANCE EVENT ----
            governance_event = IamGovernanceEvent(
                event_id=event_id,
                event_type=activity,
                event_time=event_time,
                identity_name=actor,
                entitlement=target_name,
                application=category,
                certification_decision=result,
                is_sod_violation=False,
                raw_event=raw_event
            )

            # ---- RISK SCORING ----
            risk_score, risk_reasons = (
                self._calculate_risk(
                    activity,
                    target_name,
                    is_admin_role,
                    is_high_risk,
                    is_app_registration,
                    event_time
                )
            )

            governance_event.risk_score = risk_score
            governance_event.risk_reasons = (
                risk_reasons
            )

            # ---- BUILD IAM EVENT ----
            iam_event = IamEvent(
                event_type="governance",
                source_system="entra_id_audit",
                timestamp=event_time,
                host="",
                user=actor,
                governance_event=governance_event,
                overall_risk_score=risk_score,
                overall_risk_label=(
                    self._score_to_label(risk_score)
                ),
                risk_reasons=risk_reasons
            )

            self.events_processed += 1
            if risk_score >= 0.6:
                self.high_risk_events += 1

            return iam_event

        except Exception as e:
            logger.error(
                f"Entra audit normalization "
                f"failed: {e}"
            )
            return None

    def _calculate_risk(
        self,
        activity: str,
        target_name: str,
        is_admin_role: bool,
        is_high_risk: bool,
        is_app_registration: bool,
        event_time: str
    ) -> tuple:
        """Calculate risk for audit events"""
        score = 0.0
        reasons = []

        if is_admin_role:
            score += 0.7
            reasons.append(
                f"Admin role assigned: "
                f"{target_name} — "
                f"privilege escalation risk "
                f"ATT&CK T1078"
            )

        if is_high_risk:
            score += 0.4
            reasons.append(
                f"High risk operation: {activity}"
            )

        if is_app_registration:
            score += 0.3
            reasons.append(
                f"App registration/modification: "
                f"{target_name} — "
                f"possible OAuth persistence "
                f"ATT&CK T1528"
            )

        # After hours
        try:
            if "T" in event_time:
                hour = int(
                    event_time.split("T")[1][:2]
                )
                if not (BUSINESS_HOURS_START <= hour < BUSINESS_HOURS_END):
                    score += 0.2
                    reasons.append(
                        "Administrative change "
                        "outside business hours"
                    )
        except Exception:
            pass

        return min(score, 1.0), reasons

    def _score_to_label(self, score: float) -> str:
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score > 0.0:
            return "LOW"
        return "UNKNOWN"

    def get_statistics(self) -> dict:
        return {
            "events_processed": self.events_processed,
            "high_risk_events": self.high_risk_events,
            "admin_role_assignments": (
                self.admin_role_assignments
            ),
            "app_registrations": (
                self.app_registrations
            )
        }