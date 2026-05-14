"""
Layer 4 — Human in the Loop
Approval Store

Stores pending approvals for agent-recommended
actions requiring human review.

SR 11-7 COMPLIANCE:
    Every consequential automated decision
    requires documented human oversight.
    
    This store provides:
    - Complete audit trail of all decisions
    - Who approved/rejected and when
    - Time to decision metrics
    - Override history for regulators

EU AI ACT COMPLIANCE:
    Article 14: Human oversight requirement
    for high-risk AI systems.
    
    Security platforms ARE high-risk AI.
    This store satisfies Article 14.

APPROVAL LIFECYCLE:
    PENDING   → waiting for human decision
    APPROVED  → human approved the action
    REJECTED  → human rejected the action
    EXPIRED   → timed out without decision
    AUTO      → approved automatically
               (low risk, configured threshold)
"""

import json
import logging
import os
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from enum import Enum
from typing import List
from typing import Optional

logger = logging.getLogger(__name__)

APPROVALS_FILE = "data/hitl/pending_approvals.json"
AUDIT_LOG_FILE = "data/hitl/approval_audit_log.json"


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    AUTO_APPROVED = "AUTO_APPROVED"


class ApprovalPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class ApprovalRequest:
    """
    A pending approval request for human review.
    
    Created when an agent recommends an action
    that requires human authorization before
    execution.
    """
    approval_id: str
    investigation_id: str
    event_id: str
    event_host: str
    event_user: str
    recommended_actions: List[dict]
    risk_score: float
    severity: str
    priority: ApprovalPriority
    agent_reasoning: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: str = ""
    decided_at: str = ""
    decided_by: str = ""
    decision_notes: str = ""
    auto_approved: bool = False
    expires_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = _now()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["priority"] = self.priority.value
        return d

    def is_pending(self) -> bool:
        return self.status == ApprovalStatus.PENDING

    def is_critical(self) -> bool:
        return self.priority == ApprovalPriority.CRITICAL


@dataclass
class ApprovalDecision:
    """
    Record of a human decision on an approval.
    SR 11-7 audit trail entry.
    """
    approval_id: str
    decision: ApprovalStatus
    decided_by: str
    decided_at: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.decided_at:
            self.decided_at = _now()


class ApprovalStore:
    """
    Persistent store for approval requests
    and audit trail.

    Usage:
        store = ApprovalStore()

        # Create approval request
        request = store.create_approval(
            investigation_id="inv-001",
            event_id="evt-001",
            event_host="prod-server",
            event_user="svc_backup",
            recommended_actions=[...],
            risk_score=0.95,
            severity="CRITICAL",
            agent_reasoning="..."
        )

        # Get pending approvals
        pending = store.get_pending()

        # Approve
        store.approve(
            approval_id=request.approval_id,
            decided_by="analyst@company.com",
            notes="Verified with SOC lead"
        )

        # Get audit trail
        log = store.get_audit_log()
    """

    def __init__(
        self,
        approvals_file: str = APPROVALS_FILE,
        audit_file: str = AUDIT_LOG_FILE
    ):
        self.approvals_file = approvals_file
        self.audit_file = audit_file
        os.makedirs(
            os.path.dirname(approvals_file),
            exist_ok=True
        )
        os.makedirs(
            os.path.dirname(audit_file),
            exist_ok=True
        )

    def create_approval(
        self,
        investigation_id: str,
        event_id: str,
        event_host: str,
        event_user: str,
        recommended_actions: List[dict],
        risk_score: float,
        severity: str,
        agent_reasoning: str,
        auto_approve_threshold: float = None
    ) -> ApprovalRequest:
        """
        Create a new approval request.

        If risk_score is below auto_approve_threshold
        the request is automatically approved.
        This handles low-risk routine actions.
        """
        import uuid
        approval_id = f"apr_{uuid.uuid4().hex[:8]}"

        priority = self._determine_priority(
            risk_score, severity
        )

        request = ApprovalRequest(
            approval_id=approval_id,
            investigation_id=investigation_id,
            event_id=event_id,
            event_host=event_host,
            event_user=event_user,
            recommended_actions=recommended_actions,
            risk_score=risk_score,
            severity=severity,
            priority=priority,
            agent_reasoning=agent_reasoning
        )

        # Auto-approve low risk if configured
        if (
            auto_approve_threshold and
            risk_score < auto_approve_threshold
        ):
            request.status = (
                ApprovalStatus.AUTO_APPROVED
            )
            request.auto_approved = True
            request.decided_at = _now()
            request.decided_by = "system"
            request.decision_notes = (
                f"Auto-approved: risk {risk_score:.2f} "
                f"below threshold {auto_approve_threshold}"
            )
            logger.info(
                f"Auto-approved {approval_id} "
                f"risk={risk_score:.2f}"
            )
        else:
            logger.info(
                f"Created approval {approval_id} "
                f"priority={priority.value} "
                f"risk={risk_score:.2f}"
            )

        self._save_approval(request)
        self._log_audit(
            approval_id,
            "CREATED",
            "system",
            f"Priority: {priority.value} "
            f"Risk: {risk_score:.2f}"
        )

        return request

    def get_pending(self) -> List[ApprovalRequest]:
        """Get all pending approval requests"""
        approvals = self._load_approvals()
        return [
            a for a in approvals
            if a.status == ApprovalStatus.PENDING
        ]

    def get_all(self) -> List[ApprovalRequest]:
        """Get all approval requests"""
        return self._load_approvals()

    def get_by_id(
        self,
        approval_id: str
    ) -> Optional[ApprovalRequest]:
        """Get specific approval by ID"""
        approvals = self._load_approvals()
        for a in approvals:
            if a.approval_id == approval_id:
                return a
        return None

    def get_by_investigation(
        self,
        investigation_id: str
    ) -> List[ApprovalRequest]:
        """Get all approvals for an investigation"""
        approvals = self._load_approvals()
        return [
            a for a in approvals
            if a.investigation_id == investigation_id
        ]

    def approve(
        self,
        approval_id: str,
        decided_by: str,
        notes: str = ""
    ) -> bool:
        """
        Approve a pending request.

        Args:
            approval_id: ID of approval to approve
            decided_by: Analyst email/name
            notes: Optional approval notes

        Returns:
            True if approved successfully
        """
        return self._update_status(
            approval_id,
            ApprovalStatus.APPROVED,
            decided_by,
            notes
        )

    def reject(
        self,
        approval_id: str,
        decided_by: str,
        notes: str = ""
    ) -> bool:
        """
        Reject a pending request.

        Args:
            approval_id: ID of approval to reject
            decided_by: Analyst email/name
            notes: Reason for rejection

        Returns:
            True if rejected successfully
        """
        return self._update_status(
            approval_id,
            ApprovalStatus.REJECTED,
            decided_by,
            notes
        )

    def get_stats(self) -> dict:
        """Get approval statistics for dashboard"""
        approvals = self._load_approvals()

        total = len(approvals)
        pending = sum(
            1 for a in approvals
            if a.status == ApprovalStatus.PENDING
        )
        approved = sum(
            1 for a in approvals
            if a.status == ApprovalStatus.APPROVED
        )
        rejected = sum(
            1 for a in approvals
            if a.status == ApprovalStatus.REJECTED
        )
        auto_approved = sum(
            1 for a in approvals
            if a.status == ApprovalStatus.AUTO_APPROVED
        )
        critical_pending = sum(
            1 for a in approvals
            if a.status == ApprovalStatus.PENDING
            and a.priority == ApprovalPriority.CRITICAL
        )

        return {
            "total": total,
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "auto_approved": auto_approved,
            "critical_pending": critical_pending,
            "approval_rate": (
                approved / (approved + rejected)
                if (approved + rejected) > 0
                else 0.0
            )
        }

    def get_audit_log(
        self,
        limit: int = 100
    ) -> List[dict]:
        """
        Get SR 11-7 audit trail.

        Returns complete history of all decisions
        for regulatory review.
        """
        try:
            if os.path.exists(self.audit_file):
                with open(self.audit_file) as f:
                    log = json.load(f)
                    return log[-limit:]
        except Exception as e:
            logger.error(
                f"Could not load audit log: {e}"
            )
        return []

    def _update_status(
        self,
        approval_id: str,
        new_status: ApprovalStatus,
        decided_by: str,
        notes: str
    ) -> bool:
        """Update approval status"""
        try:
            approvals = self._load_approvals()
            updated = False

            for approval in approvals:
                if approval.approval_id == approval_id:
                    if not approval.is_pending():
                        logger.warning(
                            f"Approval {approval_id} "
                            f"already decided: "
                            f"{approval.status}"
                        )
                        return False

                    approval.status = new_status
                    approval.decided_at = _now()
                    approval.decided_by = decided_by
                    approval.decision_notes = notes
                    updated = True
                    break

            if updated:
                self._save_approvals(approvals)
                self._log_audit(
                    approval_id,
                    new_status.value,
                    decided_by,
                    notes
                )
                logger.info(
                    f"Approval {approval_id} "
                    f"{new_status.value} by {decided_by}"
                )
                return True

            logger.warning(
                f"Approval {approval_id} not found"
            )
            return False

        except Exception as e:
            logger.error(
                f"Status update failed: {e}"
            )
            return False

    def _determine_priority(
        self,
        risk_score: float,
        severity: str
    ) -> ApprovalPriority:
        """Determine approval priority"""
        if (
            severity == "CRITICAL" or
            risk_score >= 0.9
        ):
            return ApprovalPriority.CRITICAL
        elif (
            severity == "HIGH" or
            risk_score >= 0.7
        ):
            return ApprovalPriority.HIGH
        elif risk_score >= 0.5:
            return ApprovalPriority.MEDIUM
        return ApprovalPriority.LOW

    def _save_approval(
        self,
        approval: ApprovalRequest
    ) -> None:
        """Save single approval to store"""
        approvals = self._load_approvals()
        existing_ids = [
            a.approval_id for a in approvals
        ]
        if approval.approval_id in existing_ids:
            approvals = [
                approval
                if a.approval_id == approval.approval_id
                else a
                for a in approvals
            ]
        else:
            approvals.append(approval)
        self._save_approvals(approvals)

    def _save_approvals(
        self,
        approvals: List[ApprovalRequest]
    ) -> None:
        """Save all approvals to file"""
        try:
            with open(self.approvals_file, "w") as f:
                json.dump(
                    [a.to_dict() for a in approvals],
                    f, indent=2
                )
        except Exception as e:
            logger.error(
                f"Could not save approvals: {e}"
            )

    def _load_approvals(
        self
    ) -> List[ApprovalRequest]:
        """Load approvals from file"""
        try:
            if os.path.exists(self.approvals_file):
                with open(self.approvals_file) as f:
                    data = json.load(f)
                    return [
                        self._dict_to_approval(d)
                        for d in data
                    ]
        except Exception as e:
            logger.debug(
                f"Could not load approvals: {e}"
            )
        return []

    def _dict_to_approval(
        self,
        d: dict
    ) -> ApprovalRequest:
        """Convert dict to ApprovalRequest"""
        d["status"] = ApprovalStatus(
            d.get("status", "PENDING")
        )
        d["priority"] = ApprovalPriority(
            d.get("priority", "HIGH")
        )
        return ApprovalRequest(**d)

    def _log_audit(
        self,
        approval_id: str,
        action: str,
        actor: str,
        notes: str
    ) -> None:
        """Append to SR 11-7 audit log"""
        try:
            log = []
            if os.path.exists(self.audit_file):
                with open(self.audit_file) as f:
                    log = json.load(f)

            log.append({
                "timestamp": _now(),
                "approval_id": approval_id,
                "action": action,
                "actor": actor,
                "notes": notes
            })

            with open(self.audit_file, "w") as f:
                json.dump(log, f, indent=2)

        except Exception as e:
            logger.error(
                f"Audit log failed: {e}"
            )


def _now() -> str:
    return datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )