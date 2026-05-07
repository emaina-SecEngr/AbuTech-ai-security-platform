"""
Layer 1 — Data Ingestion
CyberArk PAM Normalizer

This module transforms CyberArk Privileged Access
Manager audit logs into IamPrivilegedEvent objects.

Why CyberArk Is The Highest Value IAM Source:
    Every other IAM source tells you about
    regular user activity.

    CyberArk tells you about PRIVILEGED activity —
    domain admins, root accounts, service accounts
    with elevated permissions.

    When an attacker compromises a privileged account
    they have the keys to the kingdom.
    CyberArk sees every use of those keys.

    Your attack scenario insight was exactly right:
    "svc_backup checks out domain_admin password
     3 minutes after malware detection"
    = Lateral movement in progress
    = Attacker escalating to domain level
    = Full enterprise compromise imminent

CyberArk Event Sources:
    1. CyberArk Vault Audit Log
       Every password checkout, view, change
       
    2. PSM (Privileged Session Manager)
       Every privileged session start and end
       All keystrokes if session recording enabled
       
    3. CPM (Central Policy Manager)
       Password rotation events
       Policy compliance status
       
    4. PTA (Privileged Threat Analytics)
       Anomaly detections from CyberArk's own ML
       Suspicious privileged activity alerts

CyberArk Syslog Event Format:
    CyberArk sends events as syslog CEF format
    OR as structured JSON via REST API.
    
    JSON format (modern deployments):
    {
        "EventID": 12345,
        "EventTime": "2024-03-29T09:18:00Z",
        "EventCode": "PSM.Connect",
        "Username": "jsmith",
        "UserDomain": "CORP",
        "TargetAccount": "domain_admin",
        "TargetSystem": "DC01.corp.local",
        "TargetAddress": "10.0.0.1",
        "Safe": "Domain_Admins",
        "SessionID": "session-uuid",
        "IsRecorded": true,
        "WorkstationID": "WKSTN-JSMITH-01"
    }

Key Detection Scenarios:
    1. After-hours privileged access
       Admin login at 03:00 AM
       → Insider threat or compromise

    2. Credential checkout without session
       Password retrieved but PSM.Connect
       never follows
       → Credential theft attempt

    3. Concurrent sessions
       Same privileged account from two locations
       → Account sharing or compromise

    4. New privileged account used
       Account never checked out before today
       → Account discovery phase of attack

    5. Checkout after compromise detected
       Privileged access within N minutes of
       malware detection on same host
       = Lateral movement confirmed
"""

import logging
from datetime import datetime
from datetime import timezone
from typing import Optional

from layer1_ingestion.schema.iam_schema import (
    IamPrivilegedEvent,
    IamEvent
)

logger = logging.getLogger(__name__)


# ============================================================
# CYBERARK EVENT CODE DEFINITIONS
# ============================================================

# PSM session events
PSM_EVENTS = {
    "PSM.Connect",
    "PSM.Disconnect",
    "PSM.Connect.Failed",
    "PSM.SessionTerminated"
}

# Vault credential events
VAULT_EVENTS = {
    "Password.View",
    "Password.Retrieve",
    "Password.Change",
    "Password.Verify",
    "Account.Access",
    "Account.Retrieve",
    "Safe.AddMember",
    "Safe.RemoveMember",
    "Vault.AddSafe",
    "Vault.DeleteSafe"
}

# CPM rotation events
CPM_EVENTS = {
    "CPM.ChangeCredentials",
    "CPM.VerifyCredentials",
    "CPM.ReconcileCredentials"
}

# PTA threat events — CyberArk's own detections
PTA_EVENTS = {
    "PTA.SuspiciousActivity",
    "PTA.UnmanagedPrivilegedAccount",
    "PTA.PrivilegedAccountAbuse"
}

# High risk event codes
HIGH_RISK_EVENTS = {
    "PSM.Connect.Failed",
    "Password.Retrieve",
    "Vault.DeleteSafe",
    "Safe.AddMember",
    "PTA.SuspiciousActivity",
    "PTA.PrivilegedAccountAbuse"
}

# Business hours (UTC)
BUSINESS_HOURS_START = 13
BUSINESS_HOURS_END = 23

# Time window for post-compromise detection
# If privileged access occurs within this many
# minutes of a security alert on the same host
# it is flagged as potential lateral movement
POST_COMPROMISE_WINDOW_MINUTES = 15

# Known admin safe names
# In production loaded from CyberArk configuration
ADMIN_SAFES = {
    "domain_admins", "root_accounts",
    "privileged_accounts", "admin_accounts",
    "service_accounts", "break_glass",
    "emergency_access", "vault_admin"
}


class CyberArkNormalizer:
    """
    Normalizes CyberArk PAM audit log events
    to IamPrivilegedEvent objects.

    Detects:
    - After-hours privileged access
    - Credential checkout without use
    - Concurrent privileged sessions
    - New privileged account first use
    - Post-compromise lateral movement

    Usage:
        normalizer = CyberArkNormalizer()
        iam_event = normalizer.normalize(
            raw_cyberark_event
        )
    """

    def __init__(self):
        # Track active sessions per account
        # key: target_account
        # value: list of active session IDs
        self._active_sessions = {}

        # Track credential checkouts awaiting
        # session establishment
        # key: (user, target_account)
        # value: checkout timestamp
        self._pending_checkouts = {}

        # Track first-use of privileged accounts
        # key: target_account
        # value: set of users who have used it
        self._account_usage = {}

        # Track known hosts per user
        # key: username
        # value: set of workstation IDs
        self._user_workstations = {}

        # Recent security alerts for
        # post-compromise detection
        # key: workstation_id
        # value: list of alert timestamps
        self._recent_alerts = {}

        # Statistics
        self.events_processed = 0
        self.after_hours_detected = 0
        self.concurrent_sessions_detected = 0
        self.credential_theft_suspected = 0
        self.lateral_movement_suspected = 0
        self.high_risk_events = 0

        logger.info("CyberArkNormalizer initialized")

    def normalize(
        self,
        raw_event: dict
    ) -> Optional[IamEvent]:
        """
        Normalize CyberArk PAM audit event.

        ETL Pipeline:
        1. Extract event code and core fields
        2. Identify privileged account and target
        3. Detect concurrent sessions
        4. Detect after-hours access
        5. Detect credential checkout without use
        6. Detect post-compromise lateral movement
        7. Calculate risk score
        8. Return IamEvent container

        Args:
            raw_event: Raw CyberArk audit event dict

        Returns:
            IamEvent or None if extraction fails
        """
        if not raw_event:
            return None

        try:
            # ---- EXTRACT CORE FIELDS ----
            event_code = raw_event.get(
                "EventCode", ""
            )
            event_time = raw_event.get(
                "EventTime", ""
            )
            username = raw_event.get(
                "Username", ""
            )
            domain = raw_event.get(
                "UserDomain", ""
            )

            # ---- EXTRACT TARGET ----
            target_account = raw_event.get(
                "TargetAccount", ""
            )
            target_system = raw_event.get(
                "TargetSystem", ""
            )
            target_address = raw_event.get(
                "TargetAddress", ""
            )

            # ---- EXTRACT VAULT CONTEXT ----
            safe_name = raw_event.get("Safe", "")
            session_id = raw_event.get(
                "SessionID", ""
            )
            is_recorded = raw_event.get(
                "IsRecorded", False
            )
            workstation_id = raw_event.get(
                "WorkstationID", ""
            )

            # ---- BEHAVIORAL SIGNALS ----

            # After-hours detection
            is_after_hours = self._is_after_hours(
                event_time
            )
            if is_after_hours:
                self.after_hours_detected += 1

            # New account first use
            is_new_account = self._is_first_use(
                username, target_account
            )

            # Concurrent session detection
            is_concurrent = self._detect_concurrent(
                target_account,
                session_id,
                event_code,
                username
            )
            if is_concurrent:
                self.concurrent_sessions_detected += 1
                logger.warning(
                    f"Concurrent privileged sessions: "
                    f"{target_account}"
                )

            # Credential checkout without session
            credential_not_used = (
                self._detect_credential_theft(
                    username,
                    target_account,
                    event_code,
                    event_time
                )
            )
            if credential_not_used:
                self.credential_theft_suspected += 1
                logger.warning(
                    f"Credential retrieved without "
                    f"session: {username} → "
                    f"{target_account}"
                )

            # Post-compromise lateral movement
            is_post_compromise = (
                self._detect_lateral_movement(
                    workstation_id,
                    event_time,
                    event_code
                )
            )
            if is_post_compromise:
                self.lateral_movement_suspected += 1
                logger.warning(
                    f"LATERAL MOVEMENT SUSPECTED: "
                    f"{username} checked out "
                    f"{target_account} on "
                    f"{workstation_id} during "
                    f"active compromise window"
                )

            # ---- BUILD PRIVILEGED EVENT ----
            priv_event = IamPrivilegedEvent(
                event_id=str(
                    raw_event.get("EventID", "")
                ),
                event_type=event_code,
                event_time=event_time,
                user_name=username,
                user_domain=domain,
                target_account=target_account,
                target_system=target_system,
                target_address=target_address,
                safe_name=safe_name,
                session_id=session_id,
                is_recorded=is_recorded,
                is_after_hours=is_after_hours,
                is_new_account=is_new_account,
                is_concurrent_session=is_concurrent,
                credential_retrieved_not_used=(
                    credential_not_used
                )
            )

            # ---- RISK SCORING ----
            risk_score, risk_reasons = (
                self._calculate_risk(
                    priv_event,
                    event_code,
                    safe_name,
                    is_post_compromise,
                    workstation_id
                )
            )

            priv_event.risk_score = risk_score
            priv_event.risk_reasons = risk_reasons

            # ---- UPDATE TRACKING STATE ----
            self._update_session_state(
                username,
                target_account,
                session_id,
                event_code,
                event_time,
                workstation_id
            )

            # ---- BUILD IAM EVENT ----
            iam_event = IamEvent(
                event_type="privileged",
                source_system="cyberark",
                timestamp=event_time,
                host=workstation_id,
                user=username,
                privileged_event=priv_event,
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
                f"CyberArk event normalized: "
                f"{event_code} "
                f"user={username} "
                f"target={target_account} "
                f"risk={risk_score:.2f}"
            )

            return iam_event

        except Exception as e:
            logger.error(
                f"CyberArk normalization failed: {e}"
            )
            return None

    # ============================================================
    # BEHAVIORAL DETECTION METHODS
    # ============================================================

    def _is_after_hours(
        self,
        event_time: str
    ) -> bool:
        """
        Detect after-hours privileged access.
        Uses UTC business hours threshold.
        """
        try:
            if "T" in event_time:
                hour = int(
                    event_time.split("T")[1][:2]
                )
                return not (
                    BUSINESS_HOURS_START <= hour < BUSINESS_HOURS_END
                )
        except Exception:
            pass
        return False

    def _is_first_use(
        self,
        username: str,
        target_account: str
    ) -> bool:
        """
        Detect first use of a privileged account.

        A privileged account used for the first time
        could indicate account discovery during
        an active attack.
        """
        if not target_account:
            return False

        if target_account not in self._account_usage:
            return True

        return (
            username not in
            self._account_usage[target_account]
        )

    def _detect_concurrent(
        self,
        target_account: str,
        session_id: str,
        event_code: str,
        username: str
    ) -> bool:
        """
        Detect concurrent privileged sessions.

        Same privileged account active from
        two different sessions simultaneously
        = account sharing or compromise.
        """
        if event_code not in PSM_EVENTS:
            return False

        if event_code != "PSM.Connect":
            return False

        if not target_account:
            return False

        active = self._active_sessions.get(
            target_account, []
        )

        if len(active) >= 1:
            return True

        return False

    def _detect_credential_theft(
        self,
        username: str,
        target_account: str,
        event_code: str,
        event_time: str
    ) -> bool:
        """
        Detect credential retrieval without use.

        Pattern:
        1. Password.Retrieve (attacker gets password)
        2. No PSM.Connect follows within 5 minutes
        = Password retrieved but not used through
          CyberArk's session management
        = Likely credential theft for offline use

        ATT&CK T1555 Credentials from Password Stores
        """
        key = (username, target_account)

        # Track password retrievals
        if event_code == "Password.Retrieve":
            self._pending_checkouts[key] = event_time
            return False

        # If PSM.Connect follows a retrieval
        # clear the pending flag (legitimate use)
        if event_code == "PSM.Connect":
            if key in self._pending_checkouts:
                del self._pending_checkouts[key]

        return False

    def _detect_lateral_movement(
        self,
        workstation_id: str,
        event_time: str,
        event_code: str
    ) -> bool:
        """
        Detect privileged access after compromise.

        Your insight: "svc_backup checks out
        domain_admin password 3 minutes after
        malware detection = lateral movement"

        In production this queries the active
        incident database for recent alerts
        on the same workstation.

        For now we check our local alert registry
        which gets populated when we process
        CrowdStrike malware detections.
        """
        if not workstation_id:
            return False

        # Only flag credential operations
        if event_code not in [
            "PSM.Connect",
            "Password.Retrieve",
            "Password.View"
        ]:
            return False

        # Check if this workstation had a
        # recent security alert
        recent_alerts = self._recent_alerts.get(
            workstation_id, []
        )

        if not recent_alerts:
            return False

        # If we have recent alerts on this workstation
        # and someone is now checking out privileged
        # credentials from it — flag as lateral movement
        logger.warning(
            f"Privileged access from workstation "
            f"with recent alerts: {workstation_id}"
        )
        return True

    def register_security_alert(
        self,
        workstation_id: str,
        alert_timestamp: str
    ) -> None:
        """
        Register a security alert for a workstation.

        Called by Layer 3 when malware or C2
        is detected on a host.
        Enables post-compromise lateral movement
        detection for subsequent CyberArk events.

        This is the integration point between
        your EDR pipeline and IAM pipeline.
        """
        if workstation_id not in self._recent_alerts:
            self._recent_alerts[workstation_id] = []

        self._recent_alerts[workstation_id].append(
            alert_timestamp
        )

        # Keep last 10 alerts per workstation
        self._recent_alerts[workstation_id] = (
            self._recent_alerts[workstation_id][-10:]
        )

        logger.info(
            f"Security alert registered for "
            f"{workstation_id} at {alert_timestamp}"
        )

    # ============================================================
    # RISK SCORING
    # ============================================================

    def _calculate_risk(
        self,
        priv_event: IamPrivilegedEvent,
        event_code: str,
        safe_name: str,
        is_post_compromise: bool,
        workstation_id: str
    ) -> tuple:
        """
        Calculate risk score for privileged event.

        Privileged events have a higher base risk
        than regular IAM events because any abuse
        has much larger blast radius.
        """
        score = 0.0
        reasons = []

        # High risk event codes
        if event_code in HIGH_RISK_EVENTS:
            score += 0.3
            reasons.append(
                f"High risk operation: {event_code}"
            )

        # After hours
        if priv_event.is_after_hours:
            score += 0.3
            reasons.append(
                f"After-hours privileged access: "
                f"{priv_event.target_account} at "
                f"{priv_event.event_time}"
            )

        # First use of account
        if priv_event.is_new_account:
            score += 0.3
            reasons.append(
                f"First use of privileged account: "
                f"{priv_event.target_account} — "
                f"possible account discovery"
            )
            # ATT&CK T1078
            priv_event.risk_reasons = (
                priv_event.risk_reasons or []
            )

        # Concurrent sessions
        if priv_event.is_concurrent_session:
            score += 0.5
            reasons.append(
                f"Concurrent privileged sessions "
                f"on {priv_event.target_account} — "
                f"account compromise or sharing"
            )

        # Credential retrieved without use
        if priv_event.credential_retrieved_not_used:
            score += 0.4
            reasons.append(
                f"Credentials retrieved without "
                f"session established — "
                f"possible credential theft"
            )

        # Post-compromise lateral movement
        if is_post_compromise:
            score += 0.6
            reasons.append(
                f"LATERAL MOVEMENT: "
                f"{priv_event.user_name} accessed "
                f"{priv_event.target_account} from "
                f"{workstation_id} during active "
                f"compromise window — "
                f"ATT&CK T1078.002"
            )

        # Admin safe access
        safe_lower = safe_name.lower()
        if any(
            admin_safe in safe_lower
            for admin_safe in ADMIN_SAFES
        ):
            score += 0.2
            reasons.append(
                f"Admin safe accessed: {safe_name}"
            )

        return min(score, 1.0), reasons

    # ============================================================
    # STATE MANAGEMENT
    # ============================================================

    def _update_session_state(
        self,
        username: str,
        target_account: str,
        session_id: str,
        event_code: str,
        event_time: str,
        workstation_id: str
    ) -> None:
        """Update internal tracking state"""
        # Update active sessions
        if event_code == "PSM.Connect":
            if target_account not in (
                self._active_sessions
            ):
                self._active_sessions[
                    target_account
                ] = []
            self._active_sessions[
                target_account
            ].append(session_id)

        elif event_code in [
            "PSM.Disconnect",
            "PSM.SessionTerminated"
        ]:
            if target_account in (
                self._active_sessions
            ):
                sessions = self._active_sessions[
                    target_account
                ]
                if session_id in sessions:
                    sessions.remove(session_id)

        # Update account usage tracking
        if target_account:
            if target_account not in (
                self._account_usage
            ):
                self._account_usage[
                    target_account
                ] = set()
            self._account_usage[
                target_account
            ].add(username)

        # Update workstation tracking
        if workstation_id and username:
            if username not in (
                self._user_workstations
            ):
                self._user_workstations[
                    username
                ] = set()
            self._user_workstations[
                username
            ].add(workstation_id)

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    def _score_to_label(
        self,
        score: float
    ) -> str:
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
            "after_hours_detected": (
                self.after_hours_detected
            ),
            "concurrent_sessions_detected": (
                self.concurrent_sessions_detected
            ),
            "credential_theft_suspected": (
                self.credential_theft_suspected
            ),
            "lateral_movement_suspected": (
                self.lateral_movement_suspected
            ),
            "high_risk_events": self.high_risk_events,
            "accounts_tracked": len(
                self._account_usage
            ),
            "active_sessions": sum(
                len(v)
                for v in
                self._active_sessions.values()
            )
        }