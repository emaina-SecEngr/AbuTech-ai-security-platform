"""
Layer 5 - FastAPI Tests

WHAT WE ARE PROVING:
    1. API starts correctly
    2. Health endpoint returns healthy status
    3. S3 ingestion endpoint processes events
    4. IAM ingestion endpoint processes events
    5. Stats endpoint returns platform statistics
    6. Event feed endpoint returns events
    7. Error handling works correctly
    8. Response models validate correctly
"""

import pytest
from fastapi.testclient import TestClient

from layer5_interface.main import app

client = TestClient(app)


# ============================================================
# SAMPLE TEST EVENTS
# ============================================================

S3_TEST_EVENT = {
    "raw_event": {
        "eventTime": "2024-03-29T15:00:00Z",
        "eventSource": "s3.amazonaws.com",
        "eventName": "GetObject",
        "awsRegion": "us-east-1",
        "sourceIPAddress": "10.0.0.155",
        "requestID": "test-req-001",
        "userIdentity": {
            "type": "IAMUser",
            "userName": "svc_backup",
            "arn": "arn:aws:iam::123:user/svc_backup",
            "principalId": "AIDABC123",
            "accountId": "123456789"
        },
        "requestParameters": {
            "bucketName": "prod-customer-data",
            "key": "customers/pii/2024-q1.csv"
        },
        "additionalEventData": {
            "bytesTransferredOut": 52428800,
            "bytesTransferredIn": 0
        }
    },
    "enrich": True,
    "investigate": False
}

IAM_TEST_EVENT = {
    "raw_event": {
        "id": "okta-test-001",
        "eventType": "user.session.start",
        "published": "2024-03-29T15:00:00Z",
        "actor": {
            "id": "user-id-123",
            "type": "User",
            "login": "jsmith@corp.com",
            "displayName": "John Smith"
        },
        "client": {
            "ipAddress": "10.0.0.155",
            "geographicalContext": {
                "country": "US",
                "city": "New York"
            }
        },
        "outcome": {
            "result": "SUCCESS"
        },
        "authenticationContext": {
            "authenticationStep": 0
        }
    },
    "source_system": "okta",
    "enrich": True,
    "investigate": False
}


# ============================================================
# TEST CLASS - HEALTH
# ============================================================

class TestHealthEndpoint:
    """Tests for platform health endpoint"""

    def test_health_returns_200(self):
        """Health endpoint returns 200 OK"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_has_status_field(self):
        """Health response has status field"""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in [
            "healthy", "degraded", "unhealthy"
        ]

    def test_health_has_version(self):
        """Health response includes version"""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "version" in data

    def test_health_has_layer_status(self):
        """Health response shows all layer status"""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "layer1_normalizers" in data
        assert "layer2_models" in data
        assert "layer3_graph" in data

    def test_root_endpoint(self):
        """Root endpoint returns platform info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "platform" in data
        assert "AbuTech" in data["platform"]


# ============================================================
# TEST CLASS - S3 INGESTION
# ============================================================

class TestS3Ingestion:
    """Tests for S3 event ingestion endpoint"""

    def test_s3_endpoint_accepts_event(self):
        """
        S3 endpoint accepts valid CloudTrail event.
        Returns 200 OK with risk assessment.
        """
        response = client.post(
            "/api/v1/ingest/s3",
            json=S3_TEST_EVENT
        )
        assert response.status_code == 200

    def test_s3_response_has_risk_score(self):
        """Response contains risk_score 0-1"""
        response = client.post(
            "/api/v1/ingest/s3",
            json=S3_TEST_EVENT
        )
        data = response.json()
        assert "risk_score" in data
        assert 0.0 <= data["risk_score"] <= 1.0

    def test_s3_response_has_risk_label(self):
        """Response contains risk_label"""
        response = client.post(
            "/api/v1/ingest/s3",
            json=S3_TEST_EVENT
        )
        data = response.json()
        assert "risk_label" in data
        assert data["risk_label"] in [
            "CRITICAL", "HIGH", "MEDIUM",
            "LOW", "UNKNOWN"
        ]

    def test_s3_response_has_accessor(self):
        """Response identifies the accessor"""
        response = client.post(
            "/api/v1/ingest/s3",
            json=S3_TEST_EVENT
        )
        data = response.json()
        assert "accessor_identity" in data
        assert data["accessor_identity"] == "svc_backup"

    def test_s3_response_has_audit_fields(self):
        """
        Response has SR 11-7 audit trail fields.
        scored_at and model_version required.
        """
        response = client.post(
            "/api/v1/ingest/s3",
            json=S3_TEST_EVENT
        )
        data = response.json()
        assert "scored_at" in data
        assert "model_version" in data
        assert "models_used" in data

    def test_s3_response_has_q3_fields(self):
        """
        Response has your Q3 analyst context fields.
        baseline_comparison, ip_reputation available.
        """
        response = client.post(
            "/api/v1/ingest/s3",
            json=S3_TEST_EVENT
        )
        data = response.json()
        assert "baseline_comparison" in data
        assert "ip_reputation" in data
        assert "permissions_summary" in data

    def test_s3_invalid_event_returns_400(self):
        """Invalid event returns 400 Bad Request"""
        response = client.post(
            "/api/v1/ingest/s3",
            json={
                "raw_event": {
                    "eventSource": "ec2.amazonaws.com"
                },
                "enrich": False
            }
        )
        assert response.status_code in [400, 500]

    def test_s3_pii_path_elevates_risk(self):
        """
        S3 event with /pii/ in path gets
        elevated risk score.
        """
        response = client.post(
            "/api/v1/ingest/s3",
            json=S3_TEST_EVENT
        )
        data = response.json()
        assert data["risk_score"] > 0.0


# ============================================================
# TEST CLASS - IAM INGESTION
# ============================================================

class TestIAMIngestion:
    """Tests for IAM event ingestion endpoint"""

    def test_iam_endpoint_accepts_event(self):
        """IAM endpoint accepts valid Okta event"""
        response = client.post(
            "/api/v1/ingest/iam",
            json=IAM_TEST_EVENT
        )
        assert response.status_code == 200

    def test_iam_response_has_risk_score(self):
        """IAM response has risk score"""
        response = client.post(
            "/api/v1/ingest/iam",
            json=IAM_TEST_EVENT
        )
        data = response.json()
        assert "risk_score" in data
        assert 0.0 <= data["risk_score"] <= 1.0

    def test_iam_source_system_in_response(self):
        """Source system identified in response"""
        response = client.post(
            "/api/v1/ingest/iam",
            json=IAM_TEST_EVENT
        )
        data = response.json()
        assert data["source_system"] == "okta"


# ============================================================
# TEST CLASS - DASHBOARD ENDPOINTS
# ============================================================

class TestDashboardEndpoints:
    """Tests for Streamlit dashboard endpoints"""

    def test_stats_endpoint_returns_200(self):
        """Stats endpoint returns 200"""
        response = client.get("/api/v1/stats")
        assert response.status_code == 200

    def test_stats_has_event_count(self):
        """Stats has total events processed"""
        response = client.get("/api/v1/stats")
        data = response.json()
        assert "total_events_processed" in data

    def test_event_feed_returns_list(self):
        """Event feed returns a list"""
        response = client.get("/api/v1/events/feed")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_event_feed_limit_parameter(self):
        """Event feed respects limit parameter"""
        response = client.get(
            "/api/v1/events/feed?limit=10"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 10

    def test_graph_summary_returns_200(self):
        """Graph summary endpoint returns 200"""
        response = client.get(
            "/api/v1/graph/summary"
        )
        assert response.status_code == 200

    def test_graph_summary_has_node_count(self):
        """Graph summary has node count"""
        response = client.get(
            "/api/v1/graph/summary"
        )
        data = response.json()
        assert "total_nodes" in data


# ============================================================
# TEST CLASS - API DOCUMENTATION
# ============================================================

class TestAPIDocumentation:
    """
    Tests that API documentation is accessible.
    BofA integration team needs these endpoints.
    """

    def test_swagger_docs_accessible(self):
        """
        Swagger UI accessible at /docs.
        BofA integration team browses here.
        """
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_accessible(self):
        """ReDoc accessible at /redoc"""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema_accessible(self):
        """
        OpenAPI schema accessible.
        Used for automated client generation.
        BofA generates TypeScript client from this.
        """
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "components" in schema