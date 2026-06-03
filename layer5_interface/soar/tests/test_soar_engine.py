"""
Tests for SOAR Playbook Engine,
Actions, and Library
"""

import pytest
from layer5_interface.soar.playbook_actions\
    import PlaybookActions, ActionResult
from layer5_interface.soar.playbook_engine\
    import PlaybookEngine
from layer5_interface.soar.playbook_library\
    import (
        ALL_PLAYBOOKS,
        TECHNIQUE_TO_PLAYBOOK,
        CONTAIN_COMPROMISED_ACCOUNT,
        BLOCK_MALICIOUS_IP,
        CONTAIN_DATA_EXFILTRATION,
        get_playbook,
        get_playbook_for_technique,
        get_all_playbook_names,
        get_auto_execute_playbooks
    )


@pytest.fixture
def actions():
    return PlaybookActions(dry_run=True)


@pytest.fixture
def engine():
    return PlaybookEngine(dry_run=True)


@pytest.fixture
def critical_event():
    return {
        "event_id": "EVT-001",
        "accessor_identity": "svc_backup",
        "accessor_type": "service_account",
        "data_store_name": "prod-pci-data",
        "data_classification": "PCI",
        "source_ip": "185.220.101.45",
        "risk_score": 0.974,
        "risk_reasons": [
            "after_hours", "tor_ip",
            "large_transfer_500mb+"
        ],
        "mitre_techniques": ["T1530"],
        "source_system": "s3_normalizer"
    }


@pytest.fixture
def mfa_event():
    return {
        "event_id": "EVT-002",
        "accessor_identity": "john.smith@bofa.com",
        "accessor_type": "human",
        "source_ip": "185.220.101.45",
        "risk_score": 0.85,
        "mitre_techniques": ["T1621"],
        "source_system": "okta_normalizer"
    }


@pytest.fixture
def low_risk_event():
    return {
        "event_id": "EVT-003",
        "accessor_identity": "john.smith",
        "accessor_type": "human",
        "source_ip": "10.0.0.1",
        "risk_score": 0.10,
        "mitre_techniques": [],
        "source_system": "api_gateway_aws"
    }


# ============================================================
# ACTION TESTS
# ============================================================

class TestPlaybookActions:

    def test_actions_initialize(self, actions):
        assert actions is not None
        assert actions.dry_run is True

    def test_disable_identity(self, actions):
        result = actions.disable_identity(
            "svc_backup", "aws_iam"
        )
        assert result.success is True
        assert result.dry_run is True
        assert result.target == "svc_backup"

    def test_disable_identity_has_rollback(
        self, actions
    ):
        result = actions.disable_identity(
            "svc_backup"
        )
        assert "action" in result.rollback_data
        assert result.rollback_data[
            "action"
        ] == "re_enable"

    def test_revoke_sessions(self, actions):
        result = actions.revoke_sessions(
            "svc_backup"
        )
        assert result.success is True

    def test_force_mfa_reregistration(
        self, actions
    ):
        result = actions.force_mfa_reregistration(
            "john.smith"
        )
        assert result.success is True

    def test_rotate_credentials(self, actions):
        result = actions.rotate_credentials(
            "svc_backup", "cyberark"
        )
        assert result.success is True

    def test_block_ip(self, actions):
        result = actions.block_ip(
            "185.220.101.45", "palo_alto"
        )
        assert result.success is True
        assert result.target == "185.220.101.45"

    def test_block_ip_has_rollback(self, actions):
        result = actions.block_ip(
            "185.220.101.45"
        )
        assert result.rollback_data[
            "action"
        ] == "unblock_ip"

    def test_isolate_endpoint(self, actions):
        result = actions.isolate_endpoint(
            "WORKSTATION-01", "crowdstrike"
        )
        assert result.success is True

    def test_block_domain(self, actions):
        result = actions.block_domain(
            "evil.com"
        )
        assert result.success is True

    def test_restrict_data_store(self, actions):
        result = actions.restrict_data_store(
            "prod-pci-data", "aws_s3"
        )
        assert result.success is True

    def test_snapshot_for_forensics(self, actions):
        result = actions.snapshot_for_forensics(
            "prod-db", "rds"
        )
        assert result.success is True
        assert "snapshot_id" in result.rollback_data

    def test_preserve_logs(self, actions):
        result = actions.preserve_logs(
            "aws_cloudtrail", 24
        )
        assert result.success is True

    def test_page_oncall(self, actions):
        result = actions.page_oncall(
            "Test alert", "HIGH"
        )
        assert result.success is True

    def test_notify_team(self, actions):
        result = actions.notify_team(
            "SOC", "Test message"
        )
        assert result.success is True

    def test_create_incident(self, actions):
        result = actions.create_incident(
            "Test incident", "sentinel"
        )
        assert result.success is True
        assert "incident_id" in result.rollback_data

    def test_start_compliance_timer(self, actions):
        result = actions.start_compliance_timer(
            "PCI-DSS"
        )
        assert result.success is True

    def test_action_result_to_dict(self, actions):
        result = actions.block_ip("1.2.3.4")
        d = result.to_dict()
        assert "action_name" in d
        assert "success" in d
        assert "executed_at" in d

    def test_dry_run_message_simulated(
        self, actions
    ):
        result = actions.disable_identity(
            "test"
        )
        assert "SIMULATED" in result.message


# ============================================================
# PLAYBOOK LIBRARY TESTS
# ============================================================

class TestPlaybookLibrary:

    def test_all_playbooks_exist(self):
        assert len(ALL_PLAYBOOKS) >= 8

    def test_get_playbook_by_name(self):
        pb = get_playbook(
            "contain_compromised_account"
        )
        assert pb.get("name") == (
            "contain_compromised_account"
        )

    def test_get_unknown_playbook(self):
        pb = get_playbook("nonexistent")
        assert pb == {}

    def test_all_playbooks_have_name(self):
        for pb in ALL_PLAYBOOKS.values():
            assert "name" in pb
            assert pb["name"] != ""

    def test_all_playbooks_have_steps(self):
        for pb in ALL_PLAYBOOKS.values():
            assert "steps" in pb
            assert len(pb["steps"]) > 0

    def test_all_playbooks_have_trigger(self):
        for pb in ALL_PLAYBOOKS.values():
            assert "trigger" in pb

    def test_all_steps_have_action(self):
        for pb in ALL_PLAYBOOKS.values():
            for step in pb["steps"]:
                assert "action" in step

    def test_get_playbook_for_t1530(self):
        pb = get_playbook_for_technique("T1530")
        assert pb.get("name") == (
            "contain_data_exfiltration"
        )

    def test_get_playbook_for_t1621(self):
        pb = get_playbook_for_technique("T1621")
        assert pb.get("name") == (
            "respond_mfa_fatigue"
        )

    def test_get_playbook_for_t1078(self):
        pb = get_playbook_for_technique("T1078")
        assert pb.get("name") == (
            "contain_compromised_account"
        )

    def test_get_playbook_unknown_technique(self):
        pb = get_playbook_for_technique("T9999")
        assert pb == {}

    def test_get_all_playbook_names(self):
        names = get_all_playbook_names()
        assert isinstance(names, list)
        assert len(names) >= 8

    def test_auto_execute_playbooks(self):
        auto = get_auto_execute_playbooks()
        assert isinstance(auto, list)
        # block_malicious_ip is auto-execute
        names = [p["name"] for p in auto]
        assert "block_malicious_ip" in names

    def test_block_ip_no_approval(self):
        assert BLOCK_MALICIOUS_IP[
            "requires_approval"
        ] is False

    def test_contain_account_needs_approval(self):
        assert CONTAIN_COMPROMISED_ACCOUNT[
            "requires_approval"
        ] is True

    def test_technique_map_populated(self):
        assert len(TECHNIQUE_TO_PLAYBOOK) > 0


# ============================================================
# ENGINE TESTS
# ============================================================

class TestPlaybookEngine:

    def test_engine_initializes(self, engine):
        assert engine is not None
        assert engine.dry_run is True

    def test_execute_playbook_returns_dict(
        self, engine, critical_event
    ):
        result = engine.execute_playbook(
            playbook=CONTAIN_COMPROMISED_ACCOUNT,
            event=critical_event,
            approved_by="analyst@bank.com"
        )
        assert isinstance(result, dict)

    def test_execute_success(
        self, engine, critical_event
    ):
        result = engine.execute_playbook(
            playbook=CONTAIN_COMPROMISED_ACCOUNT,
            event=critical_event,
            approved_by="analyst@bank.com"
        )
        assert result["success"] is True

    def test_execute_has_execution_id(
        self, engine, critical_event
    ):
        result = engine.execute_playbook(
            playbook=CONTAIN_COMPROMISED_ACCOUNT,
            event=critical_event,
            approved_by="analyst@bank.com"
        )
        assert "execution_id" in result

    def test_execute_completes_steps(
        self, engine, critical_event
    ):
        result = engine.execute_playbook(
            playbook=CONTAIN_COMPROMISED_ACCOUNT,
            event=critical_event,
            approved_by="analyst@bank.com"
        )
        assert result["steps_completed"] > 0

    def test_requires_approval_blocks_system(
        self, engine, critical_event
    ):
        result = engine.execute_playbook(
            playbook=CONTAIN_COMPROMISED_ACCOUNT,
            event=critical_event,
            approved_by="system"
        )
        assert result["success"] is False
        assert "approval" in result[
            "error"
        ].lower()

    def test_auto_execute_no_approval_needed(
        self, engine, critical_event
    ):
        result = engine.execute_playbook(
            playbook=BLOCK_MALICIOUS_IP,
            event=critical_event,
            approved_by="system"
        )
        assert result["success"] is True

    def test_low_risk_fails_trigger(
        self, engine, low_risk_event
    ):
        result = engine.execute_playbook(
            playbook=CONTAIN_COMPROMISED_ACCOUNT,
            event=low_risk_event,
            approved_by="analyst@bank.com"
        )
        assert result["success"] is False
        assert "trigger" in result["error"].lower()

    def test_technique_trigger_match(
        self, engine, critical_event
    ):
        result = engine.execute_playbook(
            playbook=CONTAIN_DATA_EXFILTRATION,
            event=critical_event,
            approved_by="analyst@bank.com"
        )
        assert result["success"] is True

    def test_technique_trigger_mismatch(
        self, engine, mfa_event
    ):
        # MFA event does not have T1530
        result = engine.execute_playbook(
            playbook=CONTAIN_DATA_EXFILTRATION,
            event=mfa_event,
            approved_by="analyst@bank.com"
        )
        assert result["success"] is False

    def test_mfa_playbook_on_mfa_event(
        self, engine, mfa_event
    ):
        from layer5_interface.soar.playbook_library\
            import RESPOND_MFA_FATIGUE
        result = engine.execute_playbook(
            playbook=RESPOND_MFA_FATIGUE,
            event=mfa_event,
            approved_by="analyst@bank.com"
        )
        assert result["success"] is True

    def test_get_execution(
        self, engine, critical_event
    ):
        result = engine.execute_playbook(
            playbook=CONTAIN_COMPROMISED_ACCOUNT,
            event=critical_event,
            approved_by="analyst@bank.com"
        )
        exec_id = result["execution_id"]
        execution = engine.get_execution(exec_id)
        assert execution["execution_id"] == exec_id

    def test_rollback_execution(
        self, engine, critical_event
    ):
        result = engine.execute_playbook(
            playbook=CONTAIN_COMPROMISED_ACCOUNT,
            event=critical_event,
            approved_by="analyst@bank.com"
        )
        exec_id = result["execution_id"]
        rollback = engine.rollback_execution(
            exec_id
        )
        assert rollback["success"] is True

    def test_rollback_unknown_execution(
        self, engine
    ):
        result = engine.rollback_execution(
            "EXEC-UNKNOWN"
        )
        assert result["success"] is False

    def test_param_resolution(
        self, engine, critical_event
    ):
        result = engine.execute_playbook(
            playbook=CONTAIN_COMPROMISED_ACCOUNT,
            event=critical_event,
            approved_by="analyst@bank.com"
        )
        execution = engine.get_execution(
            result["execution_id"]
        )
        # First step disables the identity
        first_step = execution[
            "steps_completed"
        ][0]
        assert "svc_backup" in first_step[
            "message"
        ]

    def test_empty_playbook(
        self, engine, critical_event
    ):
        result = engine.execute_playbook(
            playbook={},
            event=critical_event,
            approved_by="analyst@bank.com"
        )
        assert result["success"] is False

    def test_get_all_executions(
        self, engine, critical_event
    ):
        engine.execute_playbook(
            playbook=BLOCK_MALICIOUS_IP,
            event=critical_event,
            approved_by="system"
        )
        executions = engine.get_all_executions()
        assert len(executions) > 0

    def test_statistics(
        self, engine, critical_event
    ):
        engine.execute_playbook(
            playbook=BLOCK_MALICIOUS_IP,
            event=critical_event,
            approved_by="system"
        )
        stats = engine.get_statistics()
        assert "total_executions" in stats
        assert stats["total_executions"] > 0

    def test_data_exfil_full_playbook(
        self, engine, critical_event
    ):
        result = engine.execute_playbook(
            playbook=CONTAIN_DATA_EXFILTRATION,
            event=critical_event,
            approved_by="analyst@bank.com"
        )
        assert result["success"] is True
        # 8 steps in this playbook
        assert result["steps_completed"] == 8

    def test_dry_run_in_result(
        self, engine, critical_event
    ):
        result = engine.execute_playbook(
            playbook=BLOCK_MALICIOUS_IP,
            event=critical_event,
            approved_by="system"
        )
        assert result["dry_run"] is True