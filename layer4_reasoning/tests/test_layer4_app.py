"""
Tests for the Layer 4 Reasoning & Investigation dashboard.
"""

import pytest
from fastapi.testclient import TestClient

from layer4_reasoning.api.layer4_app import app


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================
# HEALTH + LAYER INFO
# ============================================================

class TestHealthAndInfo:

    def test_health(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "healthy"
        assert body["components"] >= 10

    def test_layer_info(self, client):
        res = client.get("/api/layer-info")
        assert res.status_code == 200
        body = res.json()
        assert body["layer"] == 4
        assert "what_it_does" in body
        assert "why_it_matters" in body

    def test_dashboard_serves_html(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers[
            "content-type"
        ]
        assert "Layer 4" in res.text


# ============================================================
# COMPONENT CATALOG
# ============================================================

class TestComponentCatalog:

    def test_list_components(self, client):
        res = client.get("/api/components")
        assert res.status_code == 200
        body = res.json()
        assert "categories" in body
        assert "status_meta" in body

    def test_catalog_stats(self, client):
        res = client.get("/api/components")
        body = res.json()
        stats = body["catalog_stats"]
        assert stats["total_components"] >= 10
        assert stats["agents"] >= 1
        assert stats["tools"] >= 1

    def test_all_components_have_required_fields(
        self, client
    ):
        res = client.get("/api/components")
        body = res.json()
        for group in body["categories"]:
            for c in group["components"]:
                assert "name" in c
                assert "tech" in c
                assert "does" in c
                assert "value" in c
                assert "status" in c
                assert "icon" in c


# ============================================================
# COMPONENT DETAIL
# ============================================================

class TestComponentDetail:

    def test_get_investigation_graph(self, client):
        res = client.get(
            "/api/components/investigation_graph"
        )
        assert res.status_code == 200
        body = res.json()
        assert body["tech"] == "LangGraph"

    def test_get_hitl_gate(self, client):
        res = client.get(
            "/api/components/hitl_manager"
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "gate"

    def test_get_triage_agent(self, client):
        res = client.get(
            "/api/components/triage_agent"
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "agent"

    def test_get_ensemble_tool(self, client):
        res = client.get(
            "/api/components/ensemble_tool"
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "tool"

    def test_component_not_found(self, client):
        res = client.get(
            "/api/components/nonexistent"
        )
        assert res.status_code == 404


# ============================================================
# HITL GATE EMPHASIS
# ============================================================

class TestGovernanceGate:

    def test_gate_components_exist(self, client):
        res = client.get("/api/components")
        body = res.json()
        gate_components = []
        for group in body["categories"]:
            for c in group["components"]:
                if c["status"] == "gate":
                    gate_components.append(c)
        # HITL manager + approval store
        assert len(gate_components) >= 2