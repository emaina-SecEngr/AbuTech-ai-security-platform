"""
Tests for the Layer 1 FastAPI Ingestion App.

Uses FastAPI's TestClient so the API endpoints are
covered by the same test suite as the rest of the
platform.
"""

import pytest
from fastapi.testclient import TestClient

from layer1_ingestion.api.ingestion_app import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def s3_event():
    return {
        "eventName": "GetObject",
        "eventSource": "s3.amazonaws.com",
        "requestParameters": {
            "bucketName": "prod-pci-data"
        },
        "sourceIPAddress": "185.220.101.45",
        "userIdentity": {"userName": "svc_backup"}
    }


@pytest.fixture
def guardduty_event():
    return {
        "type": "CryptoCurrency:EC2/BitcoinTool.B!DNS",
        "severity": 8.0,
        "title": "Crypto mining detected",
        "resource": {
            "resourceType": "Instance",
            "instanceDetails": {"instanceId": "i-0abc"}
        },
        "service": {"count": 1}
    }


# ============================================================
# HEALTH + CATALOG
# ============================================================

class TestHealthAndCatalog:

    def test_health(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "healthy"
        assert body["supported_sources"] >= 7

    def test_list_sources(self, client):
        res = client.get("/api/sources")
        assert res.status_code == 200
        body = res.json()
        assert body["catalog_stats"][
            "total_sources"
        ] == 28
        assert "categories" in body
        assert "methods" in body

    def test_source_detail(self, client):
        res = client.get("/api/sources/guardduty")
        assert res.status_code == 200
        body = res.json()
        assert body["name"] == "GuardDuty"
        assert body["method"] == "push"
        assert "routable_today" in body

    def test_source_detail_not_found(self, client):
        res = client.get("/api/sources/nonexistent")
        assert res.status_code == 404

    def test_dashboard_serves_html(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers[
            "content-type"
        ]
        assert "Layer 1" in res.text


# ============================================================
# INGESTION — WITH SOURCE (PUSH)
# ============================================================

class TestIngestWithSource:

    def test_ingest_s3(self, client, s3_event):
        res = client.post(
            "/api/ingest/s3", json=s3_event
        )
        assert res.status_code == 200
        body = res.json()
        assert body["accepted"] is True
        assert body["source"] == "s3"

    def test_ingest_guardduty(
        self, client, guardduty_event
    ):
        res = client.post(
            "/api/ingest/guardduty",
            json=guardduty_event
        )
        assert res.status_code == 200
        body = res.json()
        assert body["accepted"] is True
        assert body["mitre_technique"] == "T1496"

    def test_ingest_returns_normalized(
        self, client, s3_event
    ):
        res = client.post(
            "/api/ingest/s3", json=s3_event
        )
        body = res.json()
        assert "normalized_event" in body
        assert "risk_score" in body

    def test_ingest_unknown_source_rejected(
        self, client, s3_event
    ):
        res = client.post(
            "/api/ingest/not_a_source",
            json=s3_event
        )
        assert res.status_code == 422
        assert res.json()["accepted"] is False

    def test_ingest_invalid_json(self, client):
        res = client.post(
            "/api/ingest/s3",
            content=b"not json",
            headers={"content-type": "application/json"}
        )
        assert res.status_code == 400


# ============================================================
# INGESTION — INFERRED (STREAM)
# ============================================================

class TestIngestInferred:

    def test_ingest_inferred_guardduty(
        self, client, guardduty_event
    ):
        res = client.post(
            "/api/ingest", json=guardduty_event
        )
        assert res.status_code == 200
        body = res.json()
        assert body["accepted"] is True
        assert body["source"] == "guardduty"

    def test_ingest_inferred_unknown(self, client):
        res = client.post(
            "/api/ingest", json={"mystery": "data"}
        )
        assert res.status_code == 422


# ============================================================
# INGESTION — BATCH (PULL)
# ============================================================

class TestIngestBatch:

    def test_ingest_batch(
        self, client, s3_event, guardduty_event
    ):
        res = client.post(
            "/api/ingest/batch",
            json=[s3_event, guardduty_event]
        )
        assert res.status_code == 200
        body = res.json()
        assert body["accepted"] == 2
        assert body["submitted"] == 2

    def test_ingest_batch_skips_failures(
        self, client, s3_event
    ):
        res = client.post(
            "/api/ingest/batch",
            json=[s3_event, {"mystery": "x"}]
        )
        body = res.json()
        assert body["accepted"] == 1
        assert body["dropped"] == 1

    def test_ingest_batch_not_array(self, client):
        res = client.post(
            "/api/ingest/batch",
            json={"not": "array"}
        )
        assert res.status_code == 400


# ============================================================
# STATS
# ============================================================

class TestStats:

    def test_stats_endpoint(
        self, client, s3_event
    ):
        client.post("/api/ingest/s3", json=s3_event)
        res = client.get("/api/stats")
        assert res.status_code == 200
        body = res.json()
        assert "pipeline" in body
        assert "detector" in body
        assert "router" in body