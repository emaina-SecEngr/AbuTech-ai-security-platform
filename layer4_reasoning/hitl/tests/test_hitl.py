"""
Tests for Human in the Loop components
"""

import pytest
import os
import tempfile

from layer4_reasoning.hitl.approval_store import (
    ApprovalStore,
    ApprovalStatus,
    ApprovalPriority
)
from layer4_reasoning.hitl.hitl_manager import (
    HITLManager,
    ALWAYS_REQUIRE_APPROVAL
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def temp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def approval_store(temp_dir):
    return ApprovalStore(
        approvals_file=f"{temp_dir}/approvals.json",
        audit_file=f"{temp_dir}/audit.json"
    )


@pytest.fixture
def hitl_manager(approval_store):
    return HITLManager(
        store=approval_store,
        threshold=0.7
    )


@pytest.fixture
def critical_state():
    return {
        "event_id": "evt-001",
        "event_host": "prod-server",
        "event_user": "svc_backup",
        "overall_risk_score": 0.95,
        "severity_rating": "CRITICAL",
        "triage_verdict": "INVESTIGATE",
        "triage_reasoning": "Critical risk 0.95",
        "intel_summary": "Tor exit node confirmed",
        "investigation_summary": "Exfil detected",
        "compromise_confirmed": False,
        "c2_confirmed": False,
        "confirmed_techniques": ["T1530", "T1048"],
        "response_actions": [
            {
                "priority": 1,
                "action": "ISOLATE_HOST",
                "target": "prod-server",
                "automated": False
            },
            {
                "priority": 2,
                "action": "NOTIFY_SECURITY_TEAM",
                "target": "SOC",
                "automated": True
            }
        ]
    }


@pytest.fixture
def low_risk_state():
    return {
        "event_id": "evt-002",
        "event_host": "dev-server",
        "event_user": "dev_user",
        "overall_risk_score": 0.3,
        "severity_rating": "LOW",
        "triage_verdict": "MONITOR",
        "triage_reasoning": "Low risk 0.3",
        "intel_summary": "No threat intel",
        "investigation_summary": "Normal activity",
        "compromise_confirmed": False,
        "c2_confirmed": False,
        "confirmed_techniques": [],
        "response_actions": [
            {
                "priority": 1,
                "action": "NOTIFY_SECURITY_TEAM",
                "target": "SOC",
                "automated": True
            }
        ]
    }


# ============================================================
# APPROVAL STORE TESTS
# ============================================================

class TestApprovalStore:

    def test_create_approval(
        self, approval_store, critical_state
    ):
        request = approval_store.create_approval(
            investigation_id="inv-001",
            event_id="evt-001",
            event_host="prod-server",
            event_user="svc_backup",
            recommended_actions=[],
            risk_score=0.95,
            severity="CRITICAL",
            agent_reasoning="High risk"
        )
        assert request.approval_id is not None
        assert request.status == ApprovalStatus.PENDING
        assert request.risk_score == 0.95

    def test_get_pending_after_create(
        self, approval_store
    ):
        approval_store.create_approval(
            investigation_id="inv-002",
            event_id="evt-002",
            event_host="host",
            event_user="user",
            recommended_actions=[],
            risk_score=0.9,
            severity="CRITICAL",
            agent_reasoning="test"
        )
        pending = approval_store.get_pending()
        assert len(pending) >= 1

    def test_approve_request(self, approval_store):
        request = approval_store.create_approval(
            investigation_id="inv-003",
            event_id="evt-003",
            event_host="host",
            event_user="user",
            recommended_actions=[],
            risk_score=0.9,
            severity="HIGH",
            agent_reasoning="test"
        )
        result = approval_store.approve(
            request.approval_id,
            "analyst@company.com",
            "Verified with SOC lead"
        )
        assert result is True

        updated = approval_store.get_by_id(
            request.approval_id
        )
        assert updated.status == ApprovalStatus.APPROVED
        assert updated.decided_by == "analyst@company.com"

    def test_reject_request(self, approval_store):
        request = approval_store.create_approval(
            investigation_id="inv-004",
            event_id="evt-004",
            event_host="host",
            event_user="user",
            recommended_actions=[],
            risk_score=0.8,
            severity="HIGH",
            agent_reasoning="test"
        )
        result = approval_store.reject(
            request.approval_id,
            "analyst@company.com",
            "False positive"
        )
        assert result is True

        updated = approval_store.get_by_id(
            request.approval_id
        )
        assert updated.status == ApprovalStatus.REJECTED

    def test_cannot_approve_twice(
        self, approval_store
    ):
        request = approval_store.create_approval(
            investigation_id="inv-005",
            event_id="evt-005",
            event_host="host",
            event_user="user",
            recommended_actions=[],
            risk_score=0.9,
            severity="HIGH",
            agent_reasoning="test"
        )
        approval_store.approve(
            request.approval_id,
            "analyst1@company.com"
        )
        result = approval_store.approve(
            request.approval_id,
            "analyst2@company.com"
        )
        assert result is False

    def test_auto_approve_low_risk(
        self, approval_store
    ):
        request = approval_store.create_approval(
            investigation_id="inv-006",
            event_id="evt-006",
            event_host="dev-host",
            event_user="dev_user",
            recommended_actions=[],
            risk_score=0.3,
            severity="LOW",
            agent_reasoning="low risk",
            auto_approve_threshold=0.5
        )
        assert request.status == (
            ApprovalStatus.AUTO_APPROVED
        )
        assert request.auto_approved is True

    def test_critical_priority_for_high_risk(
        self, approval_store
    ):
        request = approval_store.create_approval(
            investigation_id="inv-007",
            event_id="evt-007",
            event_host="host",
            event_user="user",
            recommended_actions=[],
            risk_score=0.95,
            severity="CRITICAL",
            agent_reasoning="test"
        )
        assert request.priority == (
            ApprovalPriority.CRITICAL
        )

    def test_get_stats(self, approval_store):
        approval_store.create_approval(
            investigation_id="inv-008",
            event_id="evt-008",
            event_host="host",
            event_user="user",
            recommended_actions=[],
            risk_score=0.9,
            severity="HIGH",
            agent_reasoning="test"
        )
        stats = approval_store.get_stats()
        assert "total" in stats
        assert "pending" in stats
        assert "approved" in stats
        assert stats["total"] >= 1

    def test_audit_log_populated(
        self, approval_store
    ):
        request = approval_store.create_approval(
            investigation_id="inv-009",
            event_id="evt-009",
            event_host="host",
            event_user="user",
            recommended_actions=[],
            risk_score=0.9,
            severity="HIGH",
            agent_reasoning="test"
        )
        approval_store.approve(
            request.approval_id,
            "analyst@company.com"
        )
        log = approval_store.get_audit_log()
        assert len(log) >= 2

    def test_get_by_investigation(
        self, approval_store
    ):
        approval_store.create_approval(
            investigation_id="inv-search",
            event_id="evt-010",
            event_host="host",
            event_user="user",
            recommended_actions=[],
            risk_score=0.9,
            severity="HIGH",
            agent_reasoning="test"
        )
        results = approval_store.get_by_investigation(
            "inv-search"
        )
        assert len(results) >= 1


# ============================================================
# HITL MANAGER TESTS
# ============================================================

class TestHITLManager:

    def test_requires_approval_high_risk(
        self, hitl_manager, critical_state
    ):
        assert hitl_manager.requires_approval(
            critical_state
        ) is True

    def test_no_approval_low_risk(
        self, hitl_manager, low_risk_state
    ):
        assert hitl_manager.requires_approval(
            low_risk_state
        ) is False

    def test_requires_approval_compromise(
        self, hitl_manager
    ):
        state = {
            "overall_risk_score": 0.5,
            "compromise_confirmed": True,
            "c2_confirmed": False,
            "response_actions": []
        }
        assert hitl_manager.requires_approval(
            state
        ) is True

    def test_requires_approval_c2(
        self, hitl_manager
    ):
        state = {
            "overall_risk_score": 0.5,
            "compromise_confirmed": False,
            "c2_confirmed": True,
            "response_actions": []
        }
        assert hitl_manager.requires_approval(
            state
        ) is True

    def test_requires_approval_isolate_action(
        self, hitl_manager
    ):
        state = {
            "overall_risk_score": 0.3,
            "compromise_confirmed": False,
            "c2_confirmed": False,
            "response_actions": [
                {"action": "ISOLATE_HOST"}
            ]
        }
        assert hitl_manager.requires_approval(
            state
        ) is True

    def test_create_approval_request(
        self, hitl_manager, critical_state
    ):
        request = hitl_manager.create_approval_request(
            critical_state,
            investigation_id="test-inv-001"
        )
        assert request is not None
        assert request.approval_id is not None
        assert request.event_host == "prod-server"
        assert request.event_user == "svc_backup"

    def test_check_status_pending(
        self, hitl_manager, critical_state
    ):
        hitl_manager.create_approval_request(
            critical_state,
            investigation_id="status-test-001"
        )
        status = hitl_manager.check_approval_status(
            "status-test-001"
        )
        assert status["status"] == "PENDING"
        assert status["approved"] is False

    def test_check_status_approved(
        self, hitl_manager, critical_state
    ):
        request = hitl_manager.create_approval_request(
            critical_state,
            investigation_id="approve-test-001"
        )
        hitl_manager.approve(
            request.approval_id,
            "analyst@company.com",
            "Approved after review"
        )
        status = hitl_manager.check_approval_status(
            "approve-test-001"
        )
        assert status["status"] == "APPROVED"
        assert status["approved"] is True

    def test_check_status_no_approval(
        self, hitl_manager
    ):
        status = hitl_manager.check_approval_status(
            "nonexistent-inv"
        )
        assert status["approved"] is True

    def test_approve_via_manager(
        self, hitl_manager, critical_state
    ):
        request = hitl_manager.create_approval_request(
            critical_state,
            investigation_id="mgr-approve-001"
        )
        result = hitl_manager.approve(
            request.approval_id,
            "soc.analyst@company.com",
            "Verified threat"
        )
        assert result["success"] is True
        assert result["status"] == "APPROVED"

    def test_reject_via_manager(
        self, hitl_manager, critical_state
    ):
        request = hitl_manager.create_approval_request(
            critical_state,
            investigation_id="mgr-reject-001"
        )
        result = hitl_manager.reject(
            request.approval_id,
            "soc.analyst@company.com",
            "False positive - scheduled maintenance"
        )
        assert result["success"] is True
        assert result["status"] == "REJECTED"

    def test_get_pending_approvals(
        self, hitl_manager, critical_state
    ):
        hitl_manager.create_approval_request(
            critical_state,
            investigation_id="pending-test-001"
        )
        pending = hitl_manager.get_pending_approvals()
        assert len(pending) >= 1

    def test_get_stats(
        self, hitl_manager, critical_state
    ):
        hitl_manager.create_approval_request(
            critical_state,
            investigation_id="stats-test-001"
        )
        stats = hitl_manager.get_stats()
        assert "total" in stats
        assert "pending" in stats
        assert "critical_pending" in stats

    def test_audit_trail(
        self, hitl_manager, critical_state
    ):
        request = hitl_manager.create_approval_request(
            critical_state,
            investigation_id="audit-test-001"
        )
        hitl_manager.approve(
            request.approval_id,
            "auditor@company.com"
        )
        trail = hitl_manager.get_audit_trail()
        assert len(trail) >= 2

    def test_always_require_approval_actions(self):
        assert "ISOLATE_HOST" in ALWAYS_REQUIRE_APPROVAL
        assert "RESET_CREDENTIALS" in ALWAYS_REQUIRE_APPROVAL
        assert "DISABLE_ACCOUNT" in ALWAYS_REQUIRE_APPROVAL