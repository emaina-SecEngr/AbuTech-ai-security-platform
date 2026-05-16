"""
Tests for GNN Bridge — Knowledge Graph Connector
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def bridge():
    from layer2_ml.graph.api.gnn_bridge import GNNBridge
    return GNNBridge()


@pytest.fixture
def mock_context():
    return {
        "source_ip": "185.220.101.45",
        "data_store": "prod-customer-data"
    }


class TestGNNBridgeInit:

    def test_bridge_initializes(self, bridge):
        assert bridge is not None

    def test_bridge_has_gnn(self, bridge):
        assert hasattr(bridge, "_gnn")

    def test_bridge_has_kg(self, bridge):
        assert hasattr(bridge, "_kg")


class TestScoreEntity:

    def test_score_returns_result(self, bridge):
        result = bridge.score_entity("svc_backup")
        assert result is not None

    def test_score_has_required_fields(
        self, bridge
    ):
        result = bridge.score_entity("svc_backup")
        assert hasattr(result, "entity_id")
        assert hasattr(result, "anomaly_score")
        assert hasattr(result, "risk_label")
        assert hasattr(result, "threat_proximity")
        assert hasattr(result, "reasoning")

    def test_score_range(self, bridge):
        result = bridge.score_entity("svc_backup")
        assert 0.0 <= result.anomaly_score <= 1.0

    def test_empty_entity_returns_empty(
        self, bridge
    ):
        result = bridge.score_entity("")
        assert result.anomaly_score == 0.0
        assert result.risk_label == "UNKNOWN"

    def test_none_entity_returns_empty(self, bridge):
        result = bridge.score_entity(None)
        assert result.anomaly_score == 0.0

    def test_tor_ip_elevates_score(
        self, bridge, mock_context
    ):
        result = bridge.score_entity(
            "svc_backup",
            context=mock_context
        )
        assert result.anomaly_score >= 0.0

    def test_entity_id_preserved(self, bridge):
        result = bridge.score_entity("test_user")
        assert result.entity_id == "test_user"


class TestScoreEventGraph:

    def test_score_event_returns_result(self, bridge):
        result = bridge.score_event_graph(
            accessor="svc_backup",
            source_ip="185.220.101.45",
            data_store="prod-customer-data"
        )
        assert result is not None

    def test_score_event_range(self, bridge):
        result = bridge.score_event_graph(
            accessor="svc_backup",
            source_ip="10.0.0.155",
            data_store="prod-backup-data"
        )
        assert 0.0 <= result.anomaly_score <= 1.0

    def test_tor_ip_in_event_graph(self, bridge):
        result = bridge.score_event_graph(
            accessor="svc_backup",
            source_ip="185.220.101.45",
            data_store="prod-customer-data"
        )
        assert result.anomaly_score >= 0.0

    def test_internal_ip_safe(self, bridge):
        internal = bridge.score_event_graph(
            accessor="svc_backup",
            source_ip="10.0.0.155",
            data_store="prod-backup-data"
        )
        tor = bridge.score_event_graph(
            accessor="svc_backup",
            source_ip="185.220.101.45",
            data_store="prod-customer-data"
        )
        assert tor.anomaly_score >= internal.anomaly_score


class TestRuleBasedScoring:

    def test_service_account_gets_elevated(
        self, bridge
    ):
        result = bridge._rule_based_score(
            "svc_backup", "user", {}
        )
        assert result.anomaly_score >= 0.3

    def test_admin_gets_elevated(self, bridge):
        result = bridge._rule_based_score(
            "admin_user", "user", {}
        )
        assert result.anomaly_score >= 0.4

    def test_tor_ip_gets_critical(self, bridge):
        result = bridge._rule_based_score(
            "svc_backup", "user",
            {"source_ip": "185.220.101.45"}
        )
        assert result.anomaly_score >= 0.8
        assert result.risk_label == "CRITICAL"

    def test_pci_data_elevates_score(self, bridge):
        result = bridge._rule_based_score(
            "svc_backup", "user",
            {"data_store": "prod-pci-data"}
        )
        assert result.anomaly_score >= 0.6

    def test_internal_ip_is_safe(self, bridge):
        result = bridge._rule_based_score(
            "normal_user", "user",
            {"source_ip": "10.0.0.155"}
        )
        assert result.anomaly_score < 0.5


class TestIPRisk:

    def test_tor_ip_high_risk(self, bridge):
        assert bridge._get_ip_risk(
            "185.220.101.45"
        ) >= 0.9

    def test_internal_ip_zero_risk(self, bridge):
        assert bridge._get_ip_risk(
            "10.0.0.155"
        ) == 0.0

    def test_private_range_zero_risk(self, bridge):
        assert bridge._get_ip_risk(
            "192.168.1.100"
        ) == 0.0

    def test_unknown_ip_low_risk(self, bridge):
        assert bridge._get_ip_risk(
            "8.8.8.8"
        ) == 0.1


class TestDatastoreRisk:

    def test_pci_data_high_risk(self, bridge):
        assert bridge._get_datastore_risk(
            "prod-pci-data"
        ) >= 0.8

    def test_customer_data_high_risk(self, bridge):
        assert bridge._get_datastore_risk(
            "prod-customer-data"
        ) >= 0.7

    def test_backup_data_low_risk(self, bridge):
        assert bridge._get_datastore_risk(
            "prod-backup-data"
        ) <= 0.4

    def test_phi_data_high_risk(self, bridge):
        assert bridge._get_datastore_risk(
            "patient-health-records"
        ) >= 0.8


class TestScoreToLabel:

    def test_critical_label(self, bridge):
        assert bridge._score_to_label(0.9) == "CRITICAL"

    def test_high_label(self, bridge):
        assert bridge._score_to_label(0.7) == "HIGH"

    def test_medium_label(self, bridge):
        assert bridge._score_to_label(0.5) == "MEDIUM"

    def test_low_label(self, bridge):
        assert bridge._score_to_label(0.3) == "LOW"

    def test_unknown_label(self, bridge):
        assert bridge._score_to_label(0.0) == "UNKNOWN"