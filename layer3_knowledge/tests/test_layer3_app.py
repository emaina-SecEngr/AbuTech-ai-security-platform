"""
Tests for the Layer 3 Knowledge & Enrichment dashboard.
"""

import pytest
from fastapi.testclient import TestClient

from layer3_knowledge.api.layer3_app import app


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
        assert body["enrichers"] >= 10

    def test_layer_info(self, client):
        res = client.get("/api/layer-info")
        assert res.status_code == 200
        body = res.json()
        assert body["layer"] == 3
        assert "what_it_does" in body
        assert "why_it_matters" in body

    def test_dashboard_serves_html(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers[
            "content-type"
        ]
        assert "Layer 3" in res.text


# ============================================================
# ENRICHER CATALOG
# ============================================================

class TestEnricherCatalog:

    def test_list_enrichers(self, client):
        res = client.get("/api/enrichers")
        assert res.status_code == 200
        body = res.json()
        assert "categories" in body
        assert "status_meta" in body

    def test_catalog_stats(self, client):
        res = client.get("/api/enrichers")
        body = res.json()
        stats = body["catalog_stats"]
        assert stats["total_enrichers"] >= 10
        assert stats["intel_sources"] >= 5

    def test_all_enrichers_have_required_fields(
        self, client
    ):
        res = client.get("/api/enrichers")
        body = res.json()
        for group in body["categories"]:
            for e in group["enrichers"]:
                assert "name" in e
                assert "source" in e
                assert "adds" in e
                assert "how" in e
                assert "value" in e
                assert "status" in e
                assert "icon" in e


# ============================================================
# ENRICHER DETAIL
# ============================================================

class TestEnricherDetail:

    def test_get_mitre(self, client):
        res = client.get("/api/enrichers/mitre")
        assert res.status_code == 200
        body = res.json()
        assert body["name"] == "MITRE ATT&CK Enricher"

    def test_get_cisa(self, client):
        res = client.get("/api/enrichers/cisa")
        assert res.status_code == 200
        body = res.json()
        assert "KEV" in body["name"]

    def test_get_ciem(self, client):
        res = client.get("/api/enrichers/ciem")
        assert res.status_code == 200
        body = res.json()
        assert body["category"] == "Cloud entitlement"

    def test_get_security_graph(self, client):
        res = client.get(
            "/api/enrichers/security_graph"
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "graph"

    def test_enricher_not_found(self, client):
        res = client.get("/api/enrichers/nonexistent")
        assert res.status_code == 404