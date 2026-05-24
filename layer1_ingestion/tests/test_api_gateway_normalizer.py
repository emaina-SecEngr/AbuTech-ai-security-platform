"""
Tests for API Gateway Normalizer
"""
import pytest
from layer1_ingestion.normalizers.api_gateway_normalizer import (
    APIGatewayNormalizer,
    HTTP_METHOD_RISK,
    HTTP_STATUS_RISK,
    HIGH_VALUE_API_ENDPOINTS,
    SUSPICIOUS_USER_AGENTS
)


@pytest.fixture
def normalizer():
    return APIGatewayNormalizer()


@pytest.fixture
def aws_apigw_scraping():
    return {
        "apiId": "abc123",
        "resourcePath": "/api/customers",
        "httpMethod": "GET",
        "sourceIp": "185.220.101.45",
        "userAgent": "python-requests/2.28.0",
        "apiKey": "key-abc123",
        "status": 200,
        "responseLength": "52428800",
        "requestTime": "2026-05-21T03:00:00Z",
        "stage": "prod",
        "requestId": "req-001"
    }


@pytest.fixture
def aws_apigw_brute_force():
    return {
        "apiId": "auth123",
        "resourcePath": "/api/auth/login",
        "httpMethod": "POST",
        "sourceIp": "45.142.100.10",
        "userAgent": "curl/7.88.0",
        "apiKey": "",
        "status": 401,
        "responseLength": "128",
        "requestTime": "2026-05-21T03:00:00Z",
        "stage": "prod",
        "requestId": "req-002"
    }


@pytest.fixture
def aws_apigw_normal():
    return {
        "apiId": "api123",
        "resourcePath": "/api/health",
        "httpMethod": "GET",
        "sourceIp": "10.0.0.1",
        "userAgent": "Mozilla/5.0 Chrome/120",
        "apiKey": "key-user123",
        "status": 200,
        "responseLength": "256",
        "requestTime": "2026-05-21T09:00:00Z",
        "stage": "prod",
        "requestId": "req-003"
    }


@pytest.fixture
def azure_apim_event():
    return {
        "apiId": "banking-api",
        "operationId": "GetCustomerAccounts",
        "requestMethod": "GET",
        "requestUrl": (
            "https://api.bofa.com/api/accounts"
            "?customerId=12345"
        ),
        "callerIpAddress": "185.220.101.45",
        "responseCode": 200,
        "subscriptionId": "sub-abc123",
        "productId": "banking-product",
        "userId": "user-456",
        "apiRegion": "East US",
        "requestSize": 256,
        "responseSize": 524288000,
        "duration": 1500.0,
        "time": "2026-05-21T03:00:00Z"
    }


@pytest.fixture
def azure_apim_forbidden():
    return {
        "apiId": "admin-api",
        "operationId": "AdminOperation",
        "requestMethod": "DELETE",
        "requestUrl": (
            "https://api.bofa.com/api/admin/users/123"
        ),
        "callerIpAddress": "10.0.0.1",
        "responseCode": 403,
        "subscriptionId": "sub-limited",
        "productId": "standard",
        "userId": "user-789",
        "requestSize": 128,
        "responseSize": 256,
        "duration": 50.0,
        "time": "2026-05-21T14:00:00Z"
    }


@pytest.fixture
def kong_event():
    return {
        "client_ip": "185.220.101.45",
        "request": {
            "method": "GET",
            "uri": "/api/v2/accounts",
            "url": (
                "http://api.internal/api/v2/accounts"
            ),
            "size": 256,
            "headers": {
                "user-agent": (
                    "python-requests/2.28.0"
                ),
                "x-forwarded-for": "185.220.101.45"
            }
        },
        "response": {
            "status": 200,
            "size": 1048576
        },
        "route": {"name": "accounts-route"},
        "service": {"name": "accounts-service"},
        "consumer": {
            "id": "consumer-abc",
            "username": "svc_mobile_app"
        },
        "latencies": {"request": 250},
        "started_at": "2026-05-21T03:00:00Z"
    }


@pytest.fixture
def api_abuse_events():
    return [
        {
            "source_ip": "185.220.101.45",
            "api_path": "/api/auth/login",
            "api_status": 401,
            "api_key": ""
        }
    ] * 60 + [
        {
            "source_ip": "185.220.101.45",
            "api_path": "/api/auth/login",
            "api_status": 200,
            "api_key": "stolen-key"
        }
    ] * 5


@pytest.fixture
def scraping_events():
    return [
        {
            "source_ip": "45.142.100.10",
            "api_path": "/api/customers",
            "api_status": 200,
            "api_key": "key-abc"
        }
    ] * 150


# ============================================================
# INITIALIZATION TESTS
# ============================================================

class TestInitialization:

    def test_normalizer_initializes(self, normalizer):
        assert normalizer is not None

    def test_http_method_risk_populated(self):
        assert "GET" in HTTP_METHOD_RISK
        assert "DELETE" in HTTP_METHOD_RISK
        assert HTTP_METHOD_RISK["DELETE"] > (
            HTTP_METHOD_RISK["GET"]
        )

    def test_status_risk_populated(self):
        assert 401 in HTTP_STATUS_RISK
        assert 403 in HTTP_STATUS_RISK
        assert 429 in HTTP_STATUS_RISK

    def test_high_value_endpoints_populated(self):
        assert "/api/accounts" in (
            HIGH_VALUE_API_ENDPOINTS
        )
        assert "/api/payments" in (
            HIGH_VALUE_API_ENDPOINTS
        )

    def test_suspicious_agents_populated(self):
        assert "python-requests" in (
            SUSPICIOUS_USER_AGENTS
        )
        assert "sqlmap" in SUSPICIOUS_USER_AGENTS


# ============================================================
# VENDOR DETECTION TESTS
# ============================================================

class TestVendorDetection:

    def test_detects_aws_apigw(
        self, normalizer, aws_apigw_scraping
    ):
        vendor = normalizer._detect_vendor(
            aws_apigw_scraping
        )
        assert vendor == "aws_apigw"

    def test_detects_azure_apim(
        self, normalizer, azure_apim_event
    ):
        vendor = normalizer._detect_vendor(
            azure_apim_event
        )
        assert vendor == "azure_apim"

    def test_detects_kong(
        self, normalizer, kong_event
    ):
        vendor = normalizer._detect_vendor(kong_event)
        assert vendor == "kong"

    def test_unknown_vendor_generic(self, normalizer):
        vendor = normalizer._detect_vendor(
            {"path": "/api/test"}
        )
        assert vendor == "generic"


# ============================================================
# AWS API GATEWAY TESTS
# ============================================================

class TestAWSAPIGateway:

    def test_normalize_returns_dict(
        self, normalizer, aws_apigw_scraping
    ):
        result = normalizer.normalize(
            aws_apigw_scraping
        )
        assert isinstance(result, dict)

    def test_required_fields_present(
        self, normalizer, aws_apigw_scraping
    ):
        result = normalizer.normalize(
            aws_apigw_scraping
        )
        required = [
            "accessor_identity", "source_ip",
            "risk_score", "risk_reasons",
            "source_system", "api_method",
            "api_path", "api_status"
        ]
        for field in required:
            assert field in result

    def test_large_response_elevated(
    self, normalizer, aws_apigw_scraping
):
     result = normalizer.normalize(
        aws_apigw_scraping
    )
     assert result["risk_score"] >= 0.30

    def test_tor_ip_elevated(
        self, normalizer, aws_apigw_scraping
    ):
        result = normalizer.normalize(
            aws_apigw_scraping
        )
        reasons = str(result["risk_reasons"])
        assert "tor" in reasons.lower()

    def test_suspicious_ua_elevated(
        self, normalizer, aws_apigw_scraping
    ):
        result = normalizer.normalize(
            aws_apigw_scraping
        )
        reasons = str(result["risk_reasons"])
        assert "suspicious_user_agent" in reasons

    def test_401_auth_failure_elevated(
        self, normalizer, aws_apigw_brute_force
    ):
        result = normalizer.normalize(
            aws_apigw_brute_force
        )
        reasons = str(result["risk_reasons"])
        assert "authentication_failure" in reasons

    def test_high_value_endpoint_flagged(
        self, normalizer, aws_apigw_scraping
    ):
        result = normalizer.normalize(
            aws_apigw_scraping
        )
        reasons = str(result["risk_reasons"])
        assert "high_value_api_endpoint" in reasons

    def test_normal_low_risk(
        self, normalizer, aws_apigw_normal
    ):
        result = normalizer.normalize(aws_apigw_normal)
        assert result["risk_score"] <= 0.40

    def test_pii_classification(
        self, normalizer, aws_apigw_scraping
    ):
        result = normalizer.normalize(
            aws_apigw_scraping
        )
        assert result["data_classification"] == "PII"

    def test_source_system_aws(
        self, normalizer, aws_apigw_scraping
    ):
        result = normalizer.normalize(
            aws_apigw_scraping
        )
        assert "aws" in result["source_system"]

    def test_empty_event_safe(self, normalizer):
        result = normalizer.normalize({})
        assert result["risk_score"] == 0.0

    def test_none_event_safe(self, normalizer):
        result = normalizer.normalize(None)
        assert result is not None

    def test_risk_score_capped(
        self, normalizer, aws_apigw_scraping
    ):
        result = normalizer.normalize(
            aws_apigw_scraping
        )
        assert result["risk_score"] <= 1.0


# ============================================================
# AZURE APIM TESTS
# ============================================================

class TestAzureAPIM:

    def test_azure_large_response_elevated(
        self, normalizer, azure_apim_event
    ):
        result = normalizer.normalize(azure_apim_event)
        assert result["risk_score"] >= 0.25

    def test_azure_forbidden_elevated(
        self, normalizer, azure_apim_forbidden
    ):
        result = normalizer.normalize(
            azure_apim_forbidden
        )
        reasons = str(result["risk_reasons"])
        assert "authorization_failure" in reasons

    def test_azure_source_system(
        self, normalizer, azure_apim_event
    ):
        result = normalizer.normalize(azure_apim_event)
        assert "azure" in result["source_system"]

    def test_azure_accounts_pii(
        self, normalizer, azure_apim_event
    ):
        result = normalizer.normalize(azure_apim_event)
        assert result["data_classification"] in [
            "PII", "PCI", "UNKNOWN"
        ]

    def test_azure_delete_elevated(
        self, normalizer, azure_apim_forbidden
    ):
        result = normalizer.normalize(
            azure_apim_forbidden
        )
        reasons = str(result["risk_reasons"])
        assert "DELETE" in reasons


# ============================================================
# KONG TESTS
# ============================================================

class TestKong:

    def test_kong_returns_dict(
        self, normalizer, kong_event
    ):
        result = normalizer.normalize(kong_event)
        assert isinstance(result, dict)

    def test_kong_source_system(
        self, normalizer, kong_event
    ):
        result = normalizer.normalize(kong_event)
        assert "kong" in result["source_system"]

    def test_kong_tor_elevated(
        self, normalizer, kong_event
    ):
        result = normalizer.normalize(kong_event)
        reasons = str(result["risk_reasons"])
        assert "tor" in reasons.lower()

    def test_kong_consumer_captured(
        self, normalizer, kong_event
    ):
        result = normalizer.normalize(kong_event)
        assert "svc_mobile_app" in str(
            result.get("accessor_identity", "")
        )


# ============================================================
# API ABUSE DETECTION TESTS
# ============================================================

class TestAPIAbuseDetection:

    def test_abuse_returns_dict(
        self, normalizer, api_abuse_events
    ):
        result = normalizer.detect_api_abuse(
            api_abuse_events
        )
        assert isinstance(result, dict)

    def test_credential_stuffing_detected(
        self, normalizer, api_abuse_events
    ):
        result = normalizer.detect_api_abuse(
            api_abuse_events
        )
        assert result["abuse_detected"] is True
        assert result["abuse_type"] == (
            "credential_stuffing"
        )

    def test_scraping_detected(
        self, normalizer, scraping_events
    ):
        result = normalizer.detect_api_abuse(
            scraping_events, threshold=100
        )
        assert result["abuse_detected"] is True

    def test_no_abuse_empty(self, normalizer):
        result = normalizer.detect_api_abuse([])
        assert result["abuse_detected"] is False

    def test_abuse_has_recommendation(
        self, normalizer, api_abuse_events
    ):
        result = normalizer.detect_api_abuse(
            api_abuse_events
        )
        assert "recommendation" in result

    def test_auth_failure_rate_calculated(
        self, normalizer, api_abuse_events
    ):
        result = normalizer.detect_api_abuse(
            api_abuse_events
        )
        assert "auth_failure_rate" in result
        assert result["auth_failure_rate"] > 0

    def test_abuse_risk_score_elevated(
        self, normalizer, api_abuse_events
    ):
        result = normalizer.detect_api_abuse(
            api_abuse_events
        )
        assert result["abuse_risk_score"] >= 0.50


# ============================================================
# CLASSIFICATION TESTS
# ============================================================

class TestClassification:

    def test_pci_path_detected(self, normalizer):
        cls = normalizer._detect_classification(
            "/api/payments/card"
        )
        assert cls == "PCI"

    def test_pii_path_detected(self, normalizer):
        cls = normalizer._detect_classification(
            "/api/customers/accounts"
        )
        assert cls == "PII"

    def test_unknown_path(self, normalizer):
     cls = normalizer._detect_classification(
        "/api/status"
    )
     assert cls == "UNKNOWN"

    def test_high_value_accounts(self, normalizer):
        assert normalizer._is_high_value_endpoint(
            "/api/accounts/123"
        )

    def test_health_not_high_value(self, normalizer):
        assert not normalizer._is_high_value_endpoint(
            "/api/health"
        )