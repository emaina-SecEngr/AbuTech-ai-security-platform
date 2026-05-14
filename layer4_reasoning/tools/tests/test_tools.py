"""
Tests for all 5 Agent Tools
"""

import pytest
import os
import json
import tempfile
from unittest.mock import patch

from layer4_reasoning.tools.ip_reputation import (
    check_ip_reputation
)
from layer4_reasoning.tools.knowledge_graph_tool import (
    query_knowledge_graph
)
from layer4_reasoning.tools.ensemble_tool import (
    get_ensemble_scores
)
from layer4_reasoning.tools.incident_search import (
    search_past_incidents,
    store_incident,
    get_incident_stats
)
from layer4_reasoning.tools.permissions_tool import (
    get_user_permissions
)


# ============================================================
# TOOL 1 — IP REPUTATION TESTS
# ============================================================

class TestIPReputation:

    def test_tor_exit_node_detected(self):
        result = check_ip_reputation("185.220.101.45")
        assert result["score"] >= 90
        assert result["is_tor"] is True
        assert result["is_malicious"] is True

    def test_internal_ip_is_safe(self):
        result = check_ip_reputation("10.0.0.155")
        assert result["score"] == 0
        assert result["is_malicious"] is False
        assert result["is_tor"] is False

    def test_private_range_192(self):
        result = check_ip_reputation("192.168.1.100")
        assert result["score"] == 0
        assert result["is_malicious"] is False

    def test_unknown_ip_returns_clean(self):
        result = check_ip_reputation("8.8.8.8")
        assert result["score"] == 0
        assert result["is_malicious"] is False

    def test_result_has_required_fields(self):
        result = check_ip_reputation("185.220.101.45")
        required = [
            "ip", "score", "is_malicious",
            "is_tor", "categories", "country",
            "isp", "threat_type", "source", "summary"
        ]
        for field in required:
            assert field in result

    def test_empty_ip_returns_safe(self):
        result = check_ip_reputation("")
        assert result["is_malicious"] is False

    def test_none_ip_returns_safe(self):
        result = check_ip_reputation(None)
        assert result["is_malicious"] is False

    def test_summary_not_empty(self):
        result = check_ip_reputation("185.220.101.45")
        assert len(result["summary"]) > 0

    def test_score_range(self):
        result = check_ip_reputation("185.220.101.45")
        assert 0 <= result["score"] <= 100

    def test_source_is_rule_based(self):
        result = check_ip_reputation("185.220.101.45")
        assert result["source"] == "rule_based"

    def test_russian_attacker_ip(self):
        result = check_ip_reputation("45.142.200.1")
        assert result["score"] >= 80
        assert result["is_malicious"] is True


# ============================================================
# TOOL 2 — KNOWLEDGE GRAPH TESTS
# ============================================================

class TestKnowledgeGraphTool:

    def test_empty_entity_returns_empty(self):
        result = query_knowledge_graph("")
        assert result["found"] is False

    def test_none_entity_returns_empty(self):
        result = query_knowledge_graph(None)
        assert result["found"] is False

    def test_result_has_required_fields(self):
        result = query_knowledge_graph("svc_backup")
        required = [
            "entity_id", "entity_type",
            "risk_score", "found",
            "connections", "threat_proximity",
            "threat_connections", "summary"
        ]
        for field in required:
            assert field in result

    def test_service_account_gets_elevated_risk(self):
        result = query_knowledge_graph("svc_backup")
        assert result["risk_score"] >= 0.3

    def test_admin_account_gets_high_risk(self):
        result = query_knowledge_graph("admin_user")
        assert result["risk_score"] >= 0.4

    def test_risk_score_range(self):
        result = query_knowledge_graph("any_entity")
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_summary_not_empty(self):
        result = query_knowledge_graph("svc_backup")
        assert len(result["summary"]) > 0

    def test_connections_is_list(self):
        result = query_knowledge_graph("svc_backup")
        assert isinstance(result["connections"], list)

    def test_backup_account_flagged(self):
        result = query_knowledge_graph("svc_backup")
        assert "backup" in result["summary"].lower()


# ============================================================
# TOOL 3 — ENSEMBLE SCORE TESTS
# ============================================================

class TestEnsembleTool:

    def test_empty_event_returns_empty(self):
        result = get_ensemble_scores({})
        assert result["final_score"] == 0.0

    def test_none_event_returns_empty(self):
        result = get_ensemble_scores(None)
        assert result["final_score"] == 0.0

    def test_result_has_required_fields(self):
        result = get_ensemble_scores({
            "risk_score": 0.5
        })
        required = [
            "final_score", "risk_label",
            "verdict", "model_scores",
            "explanation"
        ]
        for field in required:
            assert field in result

    def test_high_risk_event_scores_high(self):
        result = get_ensemble_scores({
            "risk_score": 0.9,
            "bytes_accessed": 500 * 1024 * 1024,
            "data_store_name": "prod-customer-data",
            "data_path": "pci/card_numbers.csv",
            "event_time": "2024-03-29T03:00:00Z",
            "accessor_identity": "svc_backup"
        }, pii_sensitivity="PCI")
        assert result["final_score"] >= 0.7

    def test_score_range(self):
        result = get_ensemble_scores({
            "risk_score": 0.5
        })
        assert 0.0 <= result["final_score"] <= 1.0

    def test_pci_elevates_score(self):
        base = get_ensemble_scores(
            {"risk_score": 0.5},
            pii_sensitivity="NONE"
        )
        pci = get_ensemble_scores(
            {"risk_score": 0.5},
            pii_sensitivity="PCI"
        )
        assert pci["final_score"] >= base["final_score"]

    def test_explanation_not_empty(self):
        result = get_ensemble_scores({
            "risk_score": 0.7
        })
        assert len(result["explanation"]) > 0


# ============================================================
# TOOL 4 — INCIDENT SEARCH TESTS
# ============================================================

class TestIncidentSearch:

    @pytest.fixture(autouse=True)
    def temp_storage(self, tmp_path, monkeypatch):
        """Use temp directory for incident storage"""
        import layer4_reasoning.tools.incident_search as is_module
        original = is_module.INCIDENTS_FILE
        is_module.INCIDENTS_FILE = str(
            tmp_path / "test_incidents.json"
        )
        yield
        is_module.INCIDENTS_FILE = original

    def test_empty_store_returns_empty_list(self):
        results = search_past_incidents(
            accessor="unknown_user"
        )
        assert results == []

    def test_store_and_retrieve_incident(self):
        store_incident({
            "event_id": "test-001",
            "event_user": "svc_backup",
            "event_host": "prod-server",
            "overall_verdict": "DATA_EXFILTRATION",
            "severity_rating": "CRITICAL",
            "overall_risk_score": 0.95,
            "triage_verdict": "INVESTIGATE",
            "confirmed_techniques": ["T1530"],
            "executive_summary": "Critical exfil detected",
            "response_actions": []
        })
        results = search_past_incidents(
            accessor="svc_backup"
        )
        assert len(results) >= 1
        assert results[0]["accessor"] == "svc_backup"

    def test_search_by_pattern(self):
        store_incident({
            "event_id": "test-002",
            "event_user": "user1",
            "overall_verdict": "DATA_EXFILTRATION",
            "severity_rating": "HIGH",
            "overall_risk_score": 0.75,
            "triage_verdict": "INVESTIGATE",
            "confirmed_techniques": [],
            "executive_summary": "Exfil attempt",
            "response_actions": []
        })
        results = search_past_incidents(
            pattern="exfiltration"
        )
        assert len(results) >= 1

    def test_get_incident_stats_empty(self):
        stats = get_incident_stats()
        assert stats["total"] == 0
        assert stats["repeat_offender"] is False

    def test_repeat_offender_detection(self):
        for i in range(3):
            store_incident({
                "event_id": f"repeat-{i}",
                "event_user": "repeat_user",
                "overall_verdict": "ANOMALY",
                "severity_rating": "HIGH",
                "overall_risk_score": 0.7,
                "triage_verdict": "INVESTIGATE",
                "confirmed_techniques": [],
                "executive_summary": f"Incident {i}",
                "response_actions": []
            })
        stats = get_incident_stats(
            accessor="repeat_user"
        )
        assert stats["repeat_offender"] is True
        assert stats["total"] == 3

    def test_store_returns_true(self):
        result = store_incident({
            "event_id": "store-test",
            "event_user": "test_user",
            "overall_verdict": "NORMAL",
            "severity_rating": "LOW",
            "overall_risk_score": 0.1,
            "triage_verdict": "CLOSE",
            "confirmed_techniques": [],
            "executive_summary": "Normal activity",
            "response_actions": []
        })
        assert result is True


# ============================================================
# TOOL 5 — PERMISSIONS TESTS
# ============================================================

class TestPermissionsTool:

    def test_svc_backup_permissions(self):
        result = get_user_permissions("svc_backup")
        assert result["type"] == "service_account"
        assert result["risk_level"] == "MEDIUM"

    def test_svc_backup_violation_detected(self):
        result = get_user_permissions(
            "svc_backup",
            accessed_resource="prod-customer-data"
        )
        assert len(result["violations"]) > 0

    def test_svc_backup_allowed_resource(self):
        result = get_user_permissions(
            "svc_backup",
            accessed_resource="prod-backup-data"
        )
        assert len(result["violations"]) == 0

    def test_admin_is_privileged(self):
        result = get_user_permissions("admin")
        assert result["is_privileged"] is True
        assert result["risk_level"] == "CRITICAL"

    def test_result_has_required_fields(self):
        result = get_user_permissions("svc_backup")
        required = [
            "user", "type", "permissions",
            "allowed_resources", "risk_level",
            "is_privileged", "violations",
            "summary"
        ]
        for field in required:
            assert field in result

    def test_empty_user_returns_empty(self):
        result = get_user_permissions("")
        assert result["user"] == "unknown"

    def test_pci_access_triggers_violation(self):
        result = get_user_permissions(
            "svc_backup",
            accessed_resource="pci_card_data"
        )
        assert len(result["violations"]) > 0
        assert "SENSITIVE DATA" in str(
            result["violations"]
        )

    def test_summary_not_empty(self):
        result = get_user_permissions("svc_backup")
        assert len(result["summary"]) > 0

    def test_unknown_user_handled(self):
        result = get_user_permissions("unknown_user_xyz")
        assert result["type"] in [
            "standard_user", "privileged_user"
        ]

    def test_violations_is_list(self):
        result = get_user_permissions("svc_backup")
        assert isinstance(result["violations"], list)


# ============================================================
# TOOL REGISTRY TESTS
# ============================================================

class TestToolRegistry:

    def test_all_tools_importable(self):
        from layer4_reasoning.tools.tool_registry import (
            check_ip_reputation,
            query_knowledge_graph,
            get_ensemble_scores,
            search_past_incidents,
            store_incident,
            get_incident_stats,
            get_user_permissions
        )
        assert callable(check_ip_reputation)
        assert callable(query_knowledge_graph)
        assert callable(get_ensemble_scores)
        assert callable(search_past_incidents)
        assert callable(store_incident)
        assert callable(get_incident_stats)
        assert callable(get_user_permissions)

    def test_ip_tool_works_via_registry(self):
        from layer4_reasoning.tools.tool_registry \
            import check_ip_reputation
        result = check_ip_reputation("10.0.0.1")
        assert result["is_malicious"] is False

    def test_permissions_tool_works_via_registry(self):
        from layer4_reasoning.tools.tool_registry \
            import get_user_permissions
        result = get_user_permissions("svc_backup")
        assert result["type"] == "service_account"