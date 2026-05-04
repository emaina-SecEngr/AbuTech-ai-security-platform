"""
Layer 1 — Data Ingestion
IAM Event Schema

This module extends ECS with Identity and Access
Management specific fields.

Why a Separate IAM Schema:
    ECS covers endpoint and network events well.
    IAM events have unique fields that do not
    exist in standard ECS:

    Authentication context:
        Was MFA used?
        Is this a new device?
        Is this a new country?
        What authentication method was used?

    Identity context:
        What is the user's normal behavior?
        What roles does this user have?
        What is their risk score?

    Access context:
        What resource was accessed?
        What action was performed?
        Was access approved or denied?

    Geographic context:
        Where did the login originate?
        How far from the user's normal location?
        Is impossible travel detected?

These fields power your three IAM detectors:
    1. Impossible Travel Detector
    2. MFA Bypass Detector
    3. Behavioral Anomaly Detector

ETL Mapping:
    Okta fields      → IamAuthEvent
    SailPoint fields → IamGovernanceEvent
    CyberArk fields  → IamPrivilegedEvent
    Vault fields     → IamSecretEvent
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Optional


# ============================================================
# GEOGRAPHIC CONTEXT
# Used for impossible travel detection
# ============================================================

@dataclass
class GeoLocation:
    """
    Geographic location of an authentication event.

    Used to calculate:
    - Distance from last known location
    - Impossible travel detection
    - Country-based risk scoring
    """
    ip_address: str = ""
    country_code: str = ""
    country_name: str = ""
    city: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    is_vpn: bool = False
    is_proxy: bool = False
    is_tor: bool = False
    isp: str = ""


# ============================================================
# AUTHENTICATION CONTEXT
# Used for MFA bypass and anomaly detection
# ============================================================

@dataclass
class AuthContext:
    """
    Authentication method and risk context.

    Captures HOW the user authenticated —
    not just whether they succeeded.

    MFA bypass is one of the most common
    techniques used by threat actors after
    stealing credentials.

    ATT&CK T1556 — Modify Authentication Process
    ATT&CK T1621 — MFA Request Generation
    """
    # Authentication method used
    auth_method: str = ""
    # password, mfa, sso, oauth, certificate, api_key

    # MFA details
    mfa_used: bool = False
    mfa_method: str = ""
    # totp, push, sms, hardware_token, biometric
    mfa_bypassed: bool = False

    # Session context
    session_id: str = ""
    session_duration_seconds: Optional[int] = None

    # Device context
    device_id: str = ""
    device_name: str = ""
    device_type: str = ""
    # desktop, mobile, tablet, api_client
    is_new_device: bool = False
    is_managed_device: bool = False

    # Risk signals
    risk_score: float = 0.0
    risk_reasons: list = field(default_factory=list)

    # Authentication result
    outcome: str = ""
    # success, failure, denied, locked


# ============================================================
# IAM AUTHENTICATION EVENT
# Core event type for Okta, Entra ID, etc
# ============================================================

@dataclass
class IamAuthEvent:
    """
    Normalized IAM authentication event.

    Maps to ECS authentication fields plus
    IAM-specific extensions.

    Sources:
        Okta System Log   → user.authentication.*
        Microsoft Entra   → SignInLogs
        Google Workspace  → login activity
        AWS IAM           → ConsoleLogin CloudTrail

    Key detection fields:
        is_new_country  → impossible travel signal
        is_new_device   → new device signal
        mfa_bypassed    → MFA bypass signal
        risk_score      → combined risk signal
    """

    # Event identification
    event_id: str = ""
    event_type: str = ""
    # user.authentication.sso
    # user.authentication.mfa
    # user.session.start
    # user.account.lock
    event_time: str = ""
    source_system: str = ""
    # okta, entra_id, google, aws

    # Actor (who performed the action)
    user_id: str = ""
    user_name: str = ""
    user_email: str = ""
    user_display_name: str = ""

    # Action
    action: str = ""
    # authenticate, sso, mfa_verify, logout
    outcome: str = ""
    # success, failure, denied

    # Target (what was accessed)
    target_app: str = ""
    target_resource: str = ""
    target_url: str = ""

    # Geographic context
    geo: Optional[GeoLocation] = None

    # Authentication context
    auth: Optional[AuthContext] = None

    # Behavioral signals
    is_new_country: bool = False
    is_new_city: bool = False
    is_new_device: bool = False
    is_outside_hours: bool = False
    is_impossible_travel: bool = False
    travel_distance_km: Optional[float] = None
    travel_speed_kmh: Optional[float] = None

    # Risk assessment
    risk_score: float = 0.0
    risk_label: str = "UNKNOWN"
    risk_reasons: list = field(default_factory=list)

    # Raw event for audit
    raw_event: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_time": self.event_time,
            "source_system": self.source_system,
            "user_name": self.user_name,
            "user_email": self.user_email,
            "action": self.action,
            "outcome": self.outcome,
            "target_app": self.target_app,
            "is_new_country": self.is_new_country,
            "is_new_device": self.is_new_device,
            "is_impossible_travel": (
                self.is_impossible_travel
            ),
            "travel_distance_km": (
                self.travel_distance_km
            ),
            "risk_score": self.risk_score,
            "risk_label": self.risk_label,
            "risk_reasons": self.risk_reasons
        }


# ============================================================
# IAM GOVERNANCE EVENT
# SailPoint ISC — access reviews, role changes
# ============================================================

@dataclass
class IamGovernanceEvent:
    """
    Normalized IAM governance event from SailPoint.

    Governance events tell you whether access
    is APPROPRIATE — not just whether it was used.

    Key detection scenarios:
        Rubber stamp certification:
            Reviewer approved access in < 5 seconds
            without reviewing
            → Access may be over-privileged

        Orphaned account:
            Account has no owner
            → Attack surface for persistence

        SoD violation:
            User has conflicting permissions
            → Fraud or insider threat risk

        Access outlier:
            User has access no peer has
            → Privilege creep
    """
    event_id: str = ""
    event_type: str = ""
    # access_request, certification, role_change,
    # entitlement_revoke, orphan_detected,
    # sod_violation, access_outlier
    event_time: str = ""

    # Identity
    identity_id: str = ""
    identity_name: str = ""

    # Access details
    entitlement: str = ""
    application: str = ""
    role: str = ""

    # Certification details
    certifier_name: str = ""
    certification_decision: str = ""
    # approved, revoked, delegated
    certification_duration_seconds: Optional[int] = None
    is_rubber_stamp: bool = False
    # Decision made in < 5 seconds

    # Risk
    risk_score: float = 0.0
    risk_reasons: list = field(default_factory=list)

    # SoD violation details
    is_sod_violation: bool = False
    conflicting_entitlements: list = field(
        default_factory=list
    )

    raw_event: dict = field(default_factory=dict)


# ============================================================
# IAM PRIVILEGED EVENT
# CyberArk PAM — privileged session activity
# ============================================================

@dataclass
class IamPrivilegedEvent:
    """
    Normalized privileged access event from CyberArk.

    Privileged events are the highest-value
    targets for attackers. Admin credentials
    give complete control.

    Key detection scenarios:
        After-hours privileged access:
            Admin login at 03:00 AM
            → Insider threat or compromise

        New privileged account usage:
            Account never used before today
            → Account discovery or persistence

        Credential checkout without use:
            Password retrieved but session
            never established
            → Credential theft attempt

        Concurrent privileged sessions:
            Same admin account from two locations
            → Account sharing or compromise
    """
    event_id: str = ""
    event_type: str = ""
    # PSM.Connect, PSM.Disconnect,
    # CPM.ChangeCredentials, Vault.AddSafe,
    # Account.Retrieve, Account.Access
    event_time: str = ""

    # Requestor
    user_name: str = ""
    user_domain: str = ""

    # Target privileged account
    target_account: str = ""
    target_system: str = ""
    target_address: str = ""

    # Safe and vault
    safe_name: str = ""
    vault_name: str = ""

    # Session details
    session_id: str = ""
    session_duration_seconds: Optional[int] = None
    is_recorded: bool = False
    is_isolated: bool = False

    # Risk signals
    is_after_hours: bool = False
    is_new_account: bool = False
    is_concurrent_session: bool = False
    credential_retrieved_not_used: bool = False

    risk_score: float = 0.0
    risk_reasons: list = field(default_factory=list)

    raw_event: dict = field(default_factory=dict)


# ============================================================
# IAM SECRET EVENT
# HashiCorp Vault — secret access and rotation
# ============================================================

@dataclass
class IamSecretEvent:
    """
    Normalized secret access event from HashiCorp Vault.

    Vault events tell you which secrets were
    accessed during a compromise window.

    Key detection scenarios:
        Bulk secret access:
            Same identity reading many secrets
            in a short time
            → Credential harvesting

        Secret access after compromise:
            Secret read within minutes of
            malware detection on same host
            → Secrets likely exfiltrated

        Unusual accessor:
            Service account reading secrets
            it has never accessed before
            → Lateral movement via secrets

        Root token usage:
            Vault root token used
            → Critical severity always
    """
    event_id: str = ""
    event_type: str = ""
    # secret.read, secret.write, secret.delete,
    # auth.login, auth.token_create,
    # sys.lease_revoke, audit.enable
    event_time: str = ""

    # Accessor
    accessor_id: str = ""
    accessor_name: str = ""
    accessor_type: str = ""
    # human, service_account, application, root

    # Secret details
    secret_path: str = ""
    secret_mount: str = ""
    # kv, aws, database, pki, ssh
    operation: str = ""
    # read, write, delete, list

    # Token details
    token_type: str = ""
    # service, batch, root
    is_root_token: bool = False
    token_ttl_seconds: Optional[int] = None

    # Risk signals
    is_bulk_access: bool = False
    secrets_accessed_count: int = 0
    is_new_secret_path: bool = False
    is_post_compromise: bool = False

    risk_score: float = 0.0
    risk_reasons: list = field(default_factory=list)

    raw_event: dict = field(default_factory=dict)


# ============================================================
# UNIFIED IAM EVENT
# Single object carrying any IAM event type
# Passed to Layer 2 IAM threat detector
# ============================================================

@dataclass
class IamEvent:
    """
    Unified container for any IAM event type.

    Layer 2 IAM threat detector receives this
    regardless of source system.

    Same pattern as ECSNormalized for EDR events —
    one object that Layer 2 always receives.
    """
    event_type: str = ""
    # auth, governance, privileged, secret

    source_system: str = ""
    # okta, sailpoint, cyberark, vault

    timestamp: str = ""
    host: str = ""
    user: str = ""

    # Event payload — only one is populated
    auth_event: Optional[IamAuthEvent] = None
    governance_event: Optional[IamGovernanceEvent] = None
    privileged_event: Optional[IamPrivilegedEvent] = None
    secret_event: Optional[IamSecretEvent] = None

    # Combined risk from all signals
    overall_risk_score: float = 0.0
    overall_risk_label: str = "UNKNOWN"
    risk_reasons: list = field(default_factory=list)

    def get_primary_event(self):
        """Return the populated event object"""
        if self.auth_event:
            return self.auth_event
        if self.governance_event:
            return self.governance_event
        if self.privileged_event:
            return self.privileged_event
        if self.secret_event:
            return self.secret_event
        return None

    def to_dict(self) -> dict:
        primary = self.get_primary_event()
        return {
            "event_type": self.event_type,
            "source_system": self.source_system,
            "timestamp": self.timestamp,
            "host": self.host,
            "user": self.user,
            "overall_risk_score": (
                self.overall_risk_score
            ),
            "overall_risk_label": (
                self.overall_risk_label
            ),
            "risk_reasons": self.risk_reasons,
            "event_detail": (
                primary.to_dict()
                if primary and hasattr(primary, "to_dict")
                else {}
            )
        }