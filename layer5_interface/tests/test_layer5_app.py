"""
Tests for the Layer 5 Response & Interface dashboard.
"""

import pytest
from fastapi.testclient import TestClient

from layer5_interface.api.layer5_app import app


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
        assert body["cloud_executors"] >= 6

    def test_layer_info(self, client):
        res = client.get("/api/layer-info")
        assert res.status_code == 200
        body = res.json()
        assert body["layer"] == 5
        assert "what_it_does" in body
        assert "why_it_matters" in body

    def test_dashboard_serves_html(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers[
            "content-type"
        ]
        assert "Layer 5" in res.text


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
        assert stats["total_components"] >= 14
        assert stats["cloud_executors"] == 6
        assert stats["siem_integrations"] >= 3

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
                # the Splunk typo guard: value is a
                # string, never a tuple
                assert isinstance(c["value"], str)


# ============================================================
# COMPONENT DETAIL
# ============================================================

class TestComponentDetail:

    def test_get_playbook_engine(self, client):
        res = client.get(
            "/api/components/playbook_engine"
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "soar"

    def test_get_ec2_quarantine_executor(
        self, client
    ):
        res = client.get(
            "/api/components/ec2_quarantine"
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "executor"

    def test_get_revoke_sts(self, client):
        res = client.get(
            "/api/components/revoke_sts_sessions"
        )
        assert res.status_code == 200
        body = res.json()
        assert "boto3" in body["tech"]

    def test_get_siem_router(self, client):
        res = client.get(
            "/api/components/siem_router"
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "siem"

    def test_component_not_found(self, client):
        res = client.get(
            "/api/components/nonexistent"
        )
        assert res.status_code == 404


# ============================================================
# THE SIX CLOUD EXECUTORS
# ============================================================

class TestCloudExecutors:

    def test_all_six_executors_present(self, client):
        res = client.get("/api/components")
        body = res.json()
        executors = []
        for group in body["categories"]:
            for c in group["components"]:
                if c["status"] == "executor":
                    executors.append(c["id"])
        for expected in [
            "ec2_quarantine",
            "forensic_capture",
            "perimeter_block",
            "port_exposure_remediation",
            "revoke_iam_credentials",
            "revoke_sts_sessions",
        ]:
            assert expected in executors