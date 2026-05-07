"""
Layer 1 — Data Ingestion
SailPoint ISC Normalizer

This module transforms SailPoint Identity Security
Cloud events into IamGovernanceEvent objects.

Why SailPoint Is Different From Other IAM Sources:
    Okta tells you: "Did this user authenticate?"
    CyberArk tells you: "Did this user use admin?"
    AWS Secrets tells you: "Did this user read a secret?"
    
    SailPoint tells you: "Should this user HAVE
                          this access AT ALL?"
    
    This is the governance layer — the difference
    between detection and prevention.

SailPoint Event Types We Process:
    ACCESS_REQUEST_APPROVED
        User requested access and it was granted
        Risk: who approved it and how fast?
    
    ACCESS_REQUEST_DENIED
        User requested access and it was denied
        Risk signal: what are they trying to access?
    
    CERTIFICATION_COMPLETED
        Access review completed by manager
        Risk: rubber stamp detection
    
    ENTITLEMENT_REVOKED
        Access removed from user
        Risk: was this a high-privilege entitlement?
    
    ORPHAN_ACCOUNT_DETECTED
        Account with no owner found
        Risk: always high — perfect attacker target
    
    SOD_VIOLATION_DETECTED
        Segregation of Duties conflict found
        Risk: fraud enablement or compliance failure
    
    ACCESS_OUTLIER_DETECTED
        User has access no peer has
        Risk: privilege creep

SailPoint ISC Webhook Payload Structure:
{
    "id": "event-uuid",
    "type": "CERTIFICATION_COMPLETED",
    "created": "2024-03-29T10:00:00.000Z",
    "attributes": {
        "identityId": "jsmith-id-123",
        "identityName": "jsmith",
        "identityDisplayName": "John Smith",
        "certificationName": "Q1 2024 Review",
        "certificationId": "cert-uuid",
        "decision": "APPROVED",
        "decisionMaker": "manager@corp.com",
        "decisionTime": 3,
        "entitlement": "AdminRole",
        "application": "AWS",
        "violationType": null,
        "isOrphan": false
    }
}

Rubber Stamp Detection:
    Your insight: "A stale and orphaned account
    can be a targeted account for breach"
    
    We detect rubber stamps when:
    decisionTime < 5 seconds
    = reviewer did not actually review
    = access should not be trusted
"""

import logging
from typing import Optional

from layer1_ingestion.schema.iam_schema import (
    IamGovernanceEvent,
    IamEvent
)

logger = logging.getLogger(__name__)


# ============================================================
# SAILPOINT EVENT TYPE DEFINITIONS
# ============================================================

# Event types and their risk base scores
GOVERNANCE_EVENT_RISK = {
    "ACCESS_REQUEST_APPROVED":   0.1,
    "ACCESS_REQUEST_DENIED":     0.2,
    "CERTIFICATION_COMPLETED":   0.1,
    "ENTITLEMENT_REVOKED":       0.1,
    "ORPHAN_ACCOUNT_DETECTED":   0.7,
    "SOD_VIOLATION_DETECTED":    0.6,
    "ACCESS_OUTLIER_DETECTED":   0.4,
    "ROLE_ASSIGNMENT_CHANGED":   0.3,
    "POLICY_VIOLATION_DETECTED": 0.5
}

# High privilege entitlements
# In production loaded from SailPoint configuration
HIGH_PRIVILEGE_ENTITLEMENTS = {
    "adminrole", "domain_admin", "global_admin",
    "root", "superuser", "privileged_access",
    "vault_admin", "security_admin",
    "network_admin", "database_admin",
    "aws_admin", "azure_admin",
    "full_access", "read_write_all"
}

# Rubber stamp threshold in seconds
RUBBER_STAMP_THRESHOLD_SECONDS = 5

# High risk applications
HIGH_RISK_APPLICATIONS = {
    "AWS", "Azure", "GCP",
    "Active Directory", "CyberArk",
    "SailPoint", "Workday",
    "SAP", "Oracle", "Salesforce"
}


class SailPointNormalizer:
    """
    Normalizes SailPoint ISC events to
    IamGovernanceEvent objects.

    Key detection capabilities:
    - Rubber stamp certification detection
    - Orphaned account identification
    - SoD violation flagging
    - Access outlier tracking
    - Privilege creep detection

    Usage:
        normalizer = SailPointNormalizer()
        iam_event = normalizer.normalize(
            raw_sailpoint_webhook_event
        )
    """

    def __init__(self):
        # Track certification history per identity
        self._cert_history = {}

        # Track access patterns per identity
        self._access_history = {}

        # Statistics
        self.events_processed = 0
        self.rubber_stamps_detected = 0
        self.orphans_detected = 0
        self.sod_violations_detected = 0
        self.high_risk_events = 0

        logger.info("SailPointNormalizer initialized")

    def normalize(
        self,
        raw_event: dict
    ) -> Optional[IamEvent]:
        """
        Normalize SailPoint ISC event.

        ETL Pipeline:
        1. Extract event type and identity
        2. Map to IamGovernanceEvent schema
        3. Apply governance-specific detection
           - Rubber stamp detection
           - Orphan account detection
           - SoD violation flagging
        4. Calculate risk score
        5. Return IamEvent container

        Args:
            raw_event: Raw SailPoint webhook event

        Returns:
            IamEvent or None if extraction fails
        """
        if not raw_event:
            return None

        try:
            # ---- EXTRACT CORE FIELDS ----
            event_id = raw_event.get("id", "")
            event_type = raw_event.get("type", "")
            created = raw_event.get("created", "")

            if not event_type:
                return None

            # ---- EXTRACT ATTRIBUTES ----
            attrs = raw_event.get("attributes", {})

            identity_id = attrs.get("identityId", "")
            identity_name = attrs.get(
                "identityName", ""
            )
            identity_display = attrs.get(
                "identityDisplayName",
                identity_name
            )

            entitlement = attrs.get("entitlement", "")
            application = attrs.get("application", "")
            role = attrs.get("role", "")

            # ---- CERTIFICATION DETAILS ----
            decision = attrs.get("decision", "")
            certifier = attrs.get(
                "decisionMaker", ""
            )
            cert_duration = attrs.get(
                "decisionTime", None
            )

            # ---- RUBBER STAMP DETECTION ----
            is_rubber_stamp = (
                self._detect_rubber_stamp(
                    decision,
                    cert_duration,
                    event_type
                )
            )

            if is_rubber_stamp:
                self.rubber_stamps_detected += 1
                logger.warning(
                    f"Rubber stamp detected: "
                    f"{certifier} approved "
                    f"{entitlement} for "
                    f"{identity_name} in "
                    f"{cert_duration}s"
                )

            # ---- ORPHAN DETECTION ----
            is_orphan = attrs.get("isOrphan", False)
            if is_orphan or (
                event_type == "ORPHAN_ACCOUNT_DETECTED"
            ):
                is_orphan = True
                self.orphans_detected += 1
                logger.warning(
                    f"Orphaned account: "
                    f"{identity_name}"
                )

            # ---- SOD VIOLATION ----
            is_sod = (
                event_type == "SOD_VIOLATION_DETECTED"
            )
            conflicting = []
            if is_sod:
                self.sod_violations_detected += 1
                conflicting = attrs.get(
                    "conflictingEntitlements", []
                )
                logger.warning(
                    f"SoD violation: "
                    f"{identity_name} has "
                    f"conflicting entitlements"
                )

            # ---- BUILD GOVERNANCE EVENT ----
            governance_event = IamGovernanceEvent(
                event_id=event_id,
                event_type=event_type,
                event_time=created,
                identity_id=identity_id,
                identity_name=identity_name,
                entitlement=entitlement,
                application=application,
                role=role,
                certifier_name=certifier,
                certification_decision=decision,
                certification_duration_seconds=(
                    int(cert_duration)
                    if cert_duration is not None
                    else None
                ),
                is_rubber_stamp=is_rubber_stamp,
                is_sod_violation=is_sod,
                conflicting_entitlements=conflicting,
                raw_event=raw_event
            )

            # ---- RISK SCORING ----
            risk_score, risk_reasons = (
                self._calculate_risk(
                    governance_event,
                    event_type,
                    entitlement,
                    application,
                    is_orphan,
                    is_rubber_stamp,
                    is_sod
                )
            )

            governance_event.risk_score = risk_score
            governance_event.risk_reasons = risk_reasons

            # ---- BUILD IAM EVENT ----
            iam_event = IamEvent(
                event_type="governance",
                source_system="sailpoint",
                timestamp=created,
                host="",
                user=identity_name,
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

            logger.info(
                f"SailPoint event normalized: "
                f"{event_type} "
                f"identity={identity_name} "
                f"entitlement={entitlement} "
                f"risk={risk_score:.2f}"
            )

            return iam_event

        except Exception as e:
            logger.error(
                f"SailPoint normalization failed: {e}"
            )
            return None

    # ============================================================
    # GOVERNANCE DETECTION METHODS
    # ============================================================

    def _detect_rubber_stamp(
        self,
        decision: str,
        duration_seconds,
        event_type: str
    ) -> bool:
        """
        Detect rubber stamp certification.

        A rubber stamp occurs when a reviewer
        approves access without actually reviewing.

        Detection criteria:
        1. Decision was APPROVED
        2. Decision time < 5 seconds
        3. Event is a certification completion

        Why this matters (your insight):
            Stale and orphaned accounts with rubber
            stamp approval are perfect attacker targets.
            The account has more access than it should
            and no one noticed because no one reviewed.

        ATT&CK T1078 Valid Accounts
        """
        if event_type not in [
            "CERTIFICATION_COMPLETED",
            "ACCESS_REQUEST_APPROVED"
        ]:
            return False

        if decision.upper() not in [
            "APPROVED", "CERTIFY"
        ]:
            return False

        if duration_seconds is None:
            return False

        try:
            duration = float(duration_seconds)
            return duration < RUBBER_STAMP_THRESHOLD_SECONDS
        except (ValueError, TypeError):
            return False

    def _is_high_privilege(
        self,
        entitlement: str,
        role: str,
        application: str
    ) -> bool:
        """
        Check if entitlement is high privilege.

        High privilege = if compromised the
        blast radius is much larger.
        """
        combined = (
            f"{entitlement} {role}"
        ).lower()

        for priv in HIGH_PRIVILEGE_ENTITLEMENTS:
            if priv in combined:
                return True

        return application in HIGH_RISK_APPLICATIONS

    # ============================================================
    # RISK SCORING
    # ============================================================

    def _calculate_risk(
        self,
        governance_event: IamGovernanceEvent,
        event_type: str,
        entitlement: str,
        application: str,
        is_orphan: bool,
        is_rubber_stamp: bool,
        is_sod: bool
    ) -> tuple:
        """
        Calculate governance risk score.

        Governance risk is different from
        authentication risk — it is about
        whether access is APPROPRIATE,
        not whether it is anomalous.
        """
        score = GOVERNANCE_EVENT_RISK.get(
            event_type, 0.1
        )
        reasons = []

        # Orphaned account — always high risk
        if is_orphan:
            score = max(score, 0.7)
            reasons.append(
                f"Orphaned account detected: "
                f"{governance_event.identity_name} "
                f"— no owner assigned, "
                f"ideal persistence target"
            )

        # SoD violation — compliance failure
        if is_sod:
            score = max(score, 0.6)
            reasons.append(
                f"Segregation of Duties violation: "
                f"{governance_event.identity_name} "
                f"has conflicting entitlements "
                f"— compliance risk"
            )

        # Rubber stamp certification
        if is_rubber_stamp:
            score += 0.3
            duration = (
                governance_event
                .certification_duration_seconds
            )
            reasons.append(
                f"Rubber stamp certification: "
                f"access approved in {duration}s "
                f"without meaningful review"
            )

        # High privilege entitlement
        if self._is_high_privilege(
            entitlement,
            governance_event.role,
            application
        ):
            score += 0.2
            reasons.append(
                f"High privilege entitlement: "
                f"{entitlement} on {application}"
            )

        # Access denied — what were they trying to get?
        if event_type == "ACCESS_REQUEST_DENIED":
            score += 0.2
            reasons.append(
                f"Access request denied: "
                f"{governance_event.identity_name} "
                f"attempted to access "
                f"{entitlement}"
            )

        # Access outlier
        if event_type == "ACCESS_OUTLIER_DETECTED":
            score += 0.2
            reasons.append(
                f"Access outlier: "
                f"{governance_event.identity_name} "
                f"has access no peers have "
                f"— privilege creep"
            )

        return min(score, 1.0), reasons

    # ============================================================
    # UTILITY METHODS
    # ============================================================

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
            "rubber_stamps_detected": (
                self.rubber_stamps_detected
            ),
            "orphans_detected": self.orphans_detected,
            "sod_violations_detected": (
                self.sod_violations_detected
            ),
            "high_risk_events": self.high_risk_events
        }