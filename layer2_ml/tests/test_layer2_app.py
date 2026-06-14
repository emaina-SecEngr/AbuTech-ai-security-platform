"""
Tests for the Layer 2 ML Detection dashboard app.
"""

import pytest
from fastapi.testclient import TestClient

from layer2_ml.api.layer2_app import app


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
        assert body["detection_models"] >= 8

    def test_layer_info(self, client):
        res = client.get("/api/layer-info")
        assert res.status_code == 200
        body = res.json()
        assert body["layer"] == 2
        assert "what_it_does" in body
        assert "why_it_matters" in body

    def test_dashboard_serves_html(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers[
            "content-type"
        ]
        assert "Layer 2" in res.text


# ============================================================
# MODELS CATALOG
# ============================================================

class TestModelsCatalog:

    def test_list_models(self, client):
        res = client.get("/api/models")
        assert res.status_code == 200
        body = res.json()
        assert "categories" in body
        assert "status_meta" in body
        assert "learning_meta" in body

    def test_catalog_stats(self, client):
        res = client.get("/api/models")
        body = res.json()
        stats = body["catalog_stats"]
        assert stats["total_models"] >= 8
        assert "by_learning" in stats

    def test_has_supervised_and_unsupervised(
        self, client
    ):
        res = client.get("/api/models")
        body = res.json()
        learning = body["catalog_stats"]["by_learning"]
        # We have both learning types represented
        assert learning.get("supervised", 0) >= 1
        assert learning.get("unsupervised", 0) >= 1

    def test_all_models_have_required_fields(
        self, client
    ):
        res = client.get("/api/models")
        body = res.json()
        for group in body["categories"]:
            for m in group["models"]:
                assert "name" in m
                assert "algorithm" in m
                assert "tech_stack" in m
                assert "learning_type" in m
                assert "detects" in m
                assert "how_it_scores" in m
                assert "rules_miss" in m
                assert "mitre" in m
                assert "status" in m

    def test_all_models_have_tech_stack(self, client):
        res = client.get("/api/models")
        body = res.json()
        for group in body["categories"]:
            for m in group["models"]:
                assert isinstance(m["tech_stack"], list)
                assert len(m["tech_stack"]) >= 1


# ============================================================
# MODEL DETAIL
# ============================================================

class TestModelDetail:

    def test_get_gnn(self, client):
        res = client.get("/api/models/gnn")
        assert res.status_code == 200
        body = res.json()
        assert body["name"] == "GNN Detector"
        assert "PyTorch" in body["tech_stack"]

    def test_get_phishing(self, client):
        res = client.get("/api/models/phishing")
        assert res.status_code == 200
        body = res.json()
        assert body["mitre"] == "T1566 (Phishing)"

    def test_get_isolation_forest_unsupervised(
        self, client
    ):
        res = client.get(
            "/api/models/isolation_forest"
        )
        body = res.json()
        assert body["learning_type"] == "unsupervised"

    def test_get_intrusion_supervised(self, client):
        res = client.get(
            "/api/models/intrusion_detection"
        )
        body = res.json()
        assert body["learning_type"] == "supervised"

    def test_model_not_found(self, client):
        res = client.get("/api/models/nonexistent")
        assert res.status_code == 404