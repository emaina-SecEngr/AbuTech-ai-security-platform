"""
Layer 4 — Human in the Loop
HITL Manager

Orchestrates the human approval workflow
within the LangGraph investigation pipeline.

WHEN HITL TRIGGERS:
    After ResponseAgent recommends actions
    IF any action has automated=False
    OR if risk_score >= HITL_THRESHOLD (0.7)
    
    The pipeline PAUSES.
    Approval request created.
    Analyst notified.
    Pipeline resumes after decision.

SR 11-7 REQUIREMENT:
    "Models used to make consequential decisions
     must have documented human oversight."
    
    ISOLATE_HOST = consequential decision.
    RESET_CREDENTIALS = consequential decision.
    BLOCK_C2 = consequential decision.
    
    All require human approval.
    All logged in audit trail.

EU AI ACT ARTICLE 14:
    "High-risk AI systems shall be designed
     to allow natural persons to oversee
     their operation effectively."
    
    Your platform is high-risk AI.
    This manager satisfies Article 14.
"""

import logging
import os
from typing import Optional

from layer4_reasoning.hitl.approval_store import (
    ApprovalStore,
    ApprovalRequest,
    ApprovalStatus
)

logger = logging.getLogger(__name__)

# Risk threshold for requiring human approval
HITL_THRESHOLD = float(
    os.getenv("HITL_THRESHOLD", "0.7")
)

# Actions that always require human approval
ALWAYS_REQUIRE_APPROVAL = {
    "ISOLATE_HOST",
    "RESET_CREDENTIALS",
    "TERMINATE_SESSION",
    "REVOKE_CERTIFICATE",
    "DISABLE_ACCOUNT"
}

# Actions that can be auto-approved
AUTO_APPROVABLE_ACTIONS = {
    "NOTIFY_SECURITY_TEAM",
    "PRESERVE_FORENSIC_EVIDENCE",
    "INCREASE_MONITORING"
}

# Singleton store instance
_approval_store = None


def get_approval_store() -> ApprovalStore:
    """Get or create the approval store"""
    global _approval_store
    if _approval_store is None:
        _approval_store = ApprovalStore()
    return _approval_store


class HITLManager:
    """
    Manages human-in-the-loop approval workflow.
    
    Integrates with the LangGraph investigation
    pipeline to pause execution when human
    approval is required.
    
    Usage:
        manager = HITLManager()
        
        # Check if approval needed
        needs_approval = manager.requires_approval(
            state
        )
        
        # Create approval request
        if needs_approval:
            request = manager.create_approval_request(
                state
            )
            # Pipeline pauses here
            # Analyst approves via dashboard
            # Pipeline resumes
        
        # Check approval status
        status = manager.check_approval_status(
            investigation_id
        )
    """

    def __init__(
        self,
        store: ApprovalStore = None,
        threshold: float = HITL_THRESHOLD
    ):
        self.store = store or get_approval_store()
        self.threshold = threshold

    def requires_approval(
        self,
        state: dict
    ) -> bool:
        """
        Determine if this investigation requires
        human approval before actions are taken.
        
        Returns True if:
        - Risk score >= HITL_THRESHOLD
        - Any response action requires approval
        - Compromise is confirmed
        - C2 is active
        """
        risk_score = float(
            state.get("overall_risk_score", 0) or 0
        )
        if risk_score >= self.threshold:
            return True

        if state.get("compromise_confirmed"):
            return True

        if state.get("c2_confirmed"):
            return True

        actions = state.get("response_actions", [])
        for action in actions:
            action_name = action.get("action", "")
            if action_name in ALWAYS_REQUIRE_APPROVAL:
                return True

        return False

    def create_approval_request(
        self,
        state: dict,
        investigation_id: str = None
    ) -> ApprovalRequest:
        """
        Create an approval request from
        investigation state.
        
        Called by ResponseAgent after generating
        response recommendations.
        """
        inv_id = (
            investigation_id or
            state.get("event_id", "unknown")
        )

        risk_score = float(
            state.get("overall_risk_score", 0) or 0
        )
        severity = state.get(
            "severity_rating", "HIGH"
        ) or "HIGH"

        # Build agent reasoning summary
        reasoning = self._build_reasoning_summary(
            state
        )

        # Get response actions
        actions = state.get("response_actions", [])

        request = self.store.create_approval(
            investigation_id=inv_id,
            event_id=state.get("event_id", ""),
            event_host=state.get("event_host", ""),
            event_user=state.get("event_user", ""),
            recommended_actions=actions,
            risk_score=risk_score,
            severity=severity,
            agent_reasoning=reasoning
        )

        logger.info(
            f"HITL: Created approval {request.approval_id} "
            f"for investigation {inv_id} "
            f"priority={request.priority.value}"
        )

        return request

    def check_approval_status(
        self,
        investigation_id: str
    ) -> dict:
        """
        Check approval status for an investigation.
        
        Returns:
            dict with status and decision details
        """
        approvals = self.store.get_by_investigation(
            investigation_id
        )

        if not approvals:
            return {
                "status": "NO_APPROVAL_REQUIRED",
                "approved": True,
                "message": "No approval request found"
            }

        latest = approvals[-1]

        if latest.status == ApprovalStatus.APPROVED:
            return {
                "status": "APPROVED",
                "approved": True,
                "decided_by": latest.decided_by,
                "decided_at": latest.decided_at,
                "notes": latest.decision_notes,
                "approval_id": latest.approval_id
            }

        elif latest.status == ApprovalStatus.AUTO_APPROVED:
            return {
                "status": "AUTO_APPROVED",
                "approved": True,
                "decided_by": "system",
                "notes": latest.decision_notes,
                "approval_id": latest.approval_id
            }

        elif latest.status == ApprovalStatus.REJECTED:
            return {
                "status": "REJECTED",
                "approved": False,
                "decided_by": latest.decided_by,
                "decided_at": latest.decided_at,
                "notes": latest.decision_notes,
                "approval_id": latest.approval_id
            }

        elif latest.status == ApprovalStatus.PENDING:
            return {
                "status": "PENDING",
                "approved": False,
                "approval_id": latest.approval_id,
                "created_at": latest.created_at,
                "message": (
                    "Waiting for human approval. "
                    f"Priority: {latest.priority.value}"
                )
            }

        return {
            "status": "UNKNOWN",
            "approved": False,
            "message": f"Unknown status: {latest.status}"
        }

    def get_pending_approvals(self) -> list:
        """Get all pending approvals for dashboard"""
        pending = self.store.get_pending()
        return [a.to_dict() for a in pending]

    def approve(
        self,
        approval_id: str,
        analyst: str,
        notes: str = ""
    ) -> dict:
        """
        Approve an action.
        
        Args:
            approval_id: ID of approval request
            analyst: Analyst name/email
            notes: Approval notes
            
        Returns:
            Result dict with success status
        """
        success = self.store.approve(
            approval_id, analyst, notes
        )

        if success:
            logger.info(
                f"HITL: Approved {approval_id} "
                f"by {analyst}"
            )
            return {
                "success": True,
                "approval_id": approval_id,
                "status": "APPROVED",
                "decided_by": analyst,
                "message": (
                    f"Actions approved by {analyst}. "
                    f"Pipeline can proceed."
                )
            }

        return {
            "success": False,
            "approval_id": approval_id,
            "message": "Approval failed or already decided"
        }

    def reject(
        self,
        approval_id: str,
        analyst: str,
        notes: str = ""
    ) -> dict:
        """
        Reject an action.
        
        Args:
            approval_id: ID of approval request
            analyst: Analyst name/email
            notes: Rejection reason
            
        Returns:
            Result dict with success status
        """
        success = self.store.reject(
            approval_id, analyst, notes
        )

        if success:
            logger.info(
                f"HITL: Rejected {approval_id} "
                f"by {analyst}"
            )
            return {
                "success": True,
                "approval_id": approval_id,
                "status": "REJECTED",
                "decided_by": analyst,
                "message": (
                    f"Actions rejected by {analyst}. "
                    f"Manual response required."
                )
            }

        return {
            "success": False,
            "approval_id": approval_id,
            "message": "Rejection failed or already decided"
        }

    def get_audit_trail(
        self,
        limit: int = 100
    ) -> list:
        """
        Get SR 11-7 audit trail.
        Used for regulatory reporting.
        """
        return self.store.get_audit_log(limit)

    def get_stats(self) -> dict:
        """Get HITL statistics for dashboard"""
        return self.store.get_stats()

    def _build_reasoning_summary(
        self,
        state: dict
    ) -> str:
        """Build human-readable reasoning for analyst"""
        parts = []

        risk = float(
            state.get("overall_risk_score", 0) or 0
        )
        parts.append(
            f"Risk score: {risk:.2f}"
        )

        triage = state.get("triage_reasoning", "")
        if triage:
            parts.append(
                f"Triage: {triage[:200]}"
            )

        intel = state.get("intel_summary", "")
        if intel:
            parts.append(
                f"Intel: {intel[:200]}"
            )

        investigation = state.get(
            "investigation_summary", ""
        )
        if investigation:
            parts.append(
                f"Investigation: {investigation[:200]}"
            )

        if state.get("compromise_confirmed"):
            parts.append(
                "⚠️ ACTIVE COMPROMISE CONFIRMED"
            )

        if state.get("c2_confirmed"):
            parts.append(
                "⚠️ C2 COMMUNICATION ACTIVE"
            )

        techniques = state.get(
            "confirmed_techniques", []
        )
        if techniques:
            parts.append(
                f"ATT&CK: {', '.join(techniques)}"
            )

        return " | ".join(parts)