"""
Layer 1 — Data Ingestion
API Gateway Security Normalizer

Converts API Gateway logs from multiple
vendors into DataAccessEvent format.

VENDORS SUPPORTED:
    AWS API Gateway:
        CloudWatch execution logs
        Access logs (custom format)
        API key usage tracking
        
    Azure API Management (APIM):
        Diagnostic logs
        Request/response logs
        Subscription tracking
        
    Kong API Gateway:
        JSON access logs
        Plugin event logs
        Rate limiting events
        
    Generic REST API:
        Common log formats
        JSON structured logs

WHY API GATEWAY SECURITY MATTERS FOR BANKS:
    Banks expose APIs to partners, fintech apps,
    mobile apps, and internal services.
    APIs are the primary attack surface in 2026.
    
    ATTACK SCENARIOS YOUR PLATFORM CATCHES:
    
    API KEY THEFT AND ABUSE:
    Attacker steals API key from GitHub.
    Makes 10,000 requests to /api/customers.
    IsolationForest: 10,000 requests in 1 hour.
    
    CREDENTIAL STUFFING VIA API:
    Bot tries 1M username/password combos.
    Via /api/auth/login endpoint.
    LSTM: velocity pattern detection.
    
    DATA SCRAPING:
    Attacker calls /api/accounts repeatedly.
    Slowly downloads entire customer database.
    LSTM: slow scraping pattern over days.
    
    BROKEN OBJECT LEVEL AUTH (BOLA):
    User A calls /api/accounts/12345 (their account)
    Then calls /api/accounts/12346 (not their account)
    IsolationForest: unauthorized object access.
    
    PRIVILEGE ESCALATION VIA API:
    User calls admin endpoints they should not access.
    IdentityDetector: role violation.
    
    API ENUMERATION:
    Attacker probes all possible endpoints.
    High 404 error rate.
    IsolationForest: endpoint enumeration pattern.

USAGE:
    normalizer = APIGatewayNormalizer()
    event = normalizer.normalize(raw_event)
    event = normalizer.normalize_aws_apigw(event)
    event = normalizer.normalize_azure_apim(event)
    campaign = normalizer.detect_api_abuse(events)
"""

import logging
import re
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)

# HTTP method risk levels
HTTP_METHOD_RISK = {
    "GET":     0.10,
    "POST":    0.20,
    "PUT":     0.25,
    "PATCH":   0.25,
    "DELETE":  0.35,
    "HEAD":    0.15,
    "OPTIONS": 0.10,
    "TRACE":   0.40,
}

# HTTP response code risk
HTTP_STATUS_RISK = {
    # Success
    200: 0.05,
    201: 0.10,
    204: 0.05,

    # Redirect
    301: 0.10,
    302: 0.10,

    # Client errors
    400: 0.20,
    401: 0.45,
    403: 0.50,
    404: 0.20,
    405: 0.25,
    429: 0.55,

    # Server errors
    500: 0.30,
    502: 0.20,
    503: 0.20,
}

# High value API endpoints
HIGH_VALUE_API_ENDPOINTS = [
    "/api/accounts", "/api/customers",
    "/api/payments", "/api/transfers",
    "/api/cards", "/api/loans",
    "/api/users", "/api/admin",
    "/api/auth", "/api/login",
    "/api/token", "/api/oauth",
    "/v1/accounts", "/v2/accounts",
    "/api/v1/users", "/api/v2/users",
    "/internal", "/management",
    "/api/pii", "/api/records"
]

# Suspicious user agents
SUSPICIOUS_USER_AGENTS = [
    "python-requests", "curl", "wget",
    "scrapy", "go-http-client", "okhttp",
    "java/", "libwww-perl", "masscan",
    "nikto", "sqlmap", "burpsuite",
    "dirbuster", "gobuster", "ffuf"
]


class APIGatewayNormalizer:
    """
    Normalizes API Gateway security events
    from multiple vendors into DataAccessEvent.

    Handles: AWS API Gateway, Azure APIM,
    Kong, Generic REST APIs.
    """

    def __init__(self):
        self.source_system = "api_gateway"

    def normalize(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize API gateway log.
        Auto-detects vendor.

        Args:
            raw_event: API gateway log dict

        Returns:
            DataAccessEvent compatible dict
        """
        if not raw_event:
            return self._empty_event()

        vendor = self._detect_vendor(raw_event)

        if vendor == "aws_apigw":
            return self.normalize_aws_apigw(raw_event)
        elif vendor == "azure_apim":
            return self.normalize_azure_apim(raw_event)
        elif vendor == "kong":
            return self.normalize_kong(raw_event)
        else:
            return self._normalize_generic(raw_event)

    def normalize_aws_apigw(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize AWS API Gateway CloudWatch log.
        Supports both execution logs and access logs.
        """
        request_id = raw_event.get(
            "requestId",
            raw_event.get("request_id", "")
        )
        api_id = raw_event.get(
            "apiId",
            raw_event.get("api_id", "")
        )
        resource_path = raw_event.get(
            "resourcePath",
            raw_event.get("path", "")
        )
        http_method = raw_event.get(
            "httpMethod",
            raw_event.get("method", "GET")
        ).upper()
        source_ip = raw_event.get(
            "sourceIp",
            raw_event.get("ip", "")
        )
        user_agent = raw_event.get(
            "userAgent",
            raw_event.get("user_agent", "")
        )
        api_key = raw_event.get(
            "apiKey",
            raw_event.get("api_key", "")
        )
        status = int(
            raw_event.get(
                "status",
                raw_event.get("statusCode", 200)
            ) or 200
        )
        response_length = int(
            raw_event.get(
                "responseLength",
                raw_event.get("bytes", 0)
            ) or 0
        )
        request_time = raw_event.get(
            "requestTime",
            raw_event.get("timestamp", _now())
        )
        stage = raw_event.get("stage", "")
        query_params = raw_event.get(
            "queryStringParameters", {}
        )
        identity = raw_event.get(
            "identity",
            raw_event.get("authorizer", {})
        )

        accessor = (
            api_key or
            (identity.get("cognitoIdentityId", "")
             if isinstance(identity, dict) else "") or
            source_ip or
            "unknown"
        )

        risk_score, risk_reasons = (
            self._calculate_api_risk(
                http_method, resource_path,
                status, source_ip, user_agent,
                response_length, raw_event
            )
        )

        return {
            "accessor_identity": accessor,
            "accessor_type": (
                "api_key" if api_key else "human"
            ),
            "data_store_name": (
                f"aws-apigw-{api_id}"
                if api_id else "aws-api-gateway"
            ),
            "data_path": (
                f"{http_method} {resource_path}"
            ),
            "data_classification": (
                self._detect_classification(
                    resource_path
                )
            ),
            "bytes_accessed": response_length,
            "event_time": str(request_time),
            "source_ip": source_ip,
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "api_gateway_aws",
            "raw_event": raw_event,
            "api_vendor": "aws_apigw",
            "api_method": http_method,
            "api_path": resource_path,
            "api_status": status,
            "api_key": api_key[:20] + "..." if (
                api_key and len(api_key) > 20
            ) else api_key,
            "api_user_agent": user_agent[:200],
            "api_stage": stage,
            "api_request_id": request_id,
            "api_response_bytes": response_length
        }

    def normalize_azure_apim(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize Azure API Management diagnostic log.
        """
        api_id = raw_event.get(
            "apiId",
            raw_event.get("api_id", "")
        )
        operation_id = raw_event.get(
            "operationId",
            raw_event.get("operation", "")
        )
        method = raw_event.get(
            "requestMethod",
            raw_event.get("method", "GET")
        ).upper()
        url = raw_event.get(
            "requestUrl",
            raw_event.get("url", "")
        )
        path = self._extract_path(url)
        source_ip = raw_event.get(
            "callerIpAddress",
            raw_event.get("ip", "")
        )
        status = int(
            raw_event.get(
                "responseCode",
                raw_event.get("status", 200)
            ) or 200
        )
        subscription_id = raw_event.get(
            "subscriptionId", ""
        )
        product_id = raw_event.get("productId", "")
        user_id = raw_event.get("userId", "")
        api_region = raw_event.get("apiRegion", "")
        request_size = int(
            raw_event.get("requestSize", 0) or 0
        )
        response_size = int(
            raw_event.get("responseSize", 0) or 0
        )
        duration = float(
            raw_event.get("duration", 0) or 0
        )

        accessor = (
            user_id or subscription_id or
            source_ip or "unknown"
        )

        risk_score, risk_reasons = (
            self._calculate_api_risk(
                method, path, status,
                source_ip, "",
                response_size, raw_event
            )
        )

        if duration > 30000:
            risk_score = min(risk_score + 0.10, 1.0)
            risk_reasons.append(
                f"slow_api_response:{duration}ms"
            )

        return {
            "accessor_identity": accessor,
            "accessor_type": (
                "api_key" if subscription_id
                else "human"
            ),
            "data_store_name": (
                f"azure-apim-{api_id}"
                if api_id else "azure-apim"
            ),
            "data_path": f"{method} {path}",
            "data_classification": (
                self._detect_classification(path)
            ),
            "bytes_accessed": response_size,
            "event_time": raw_event.get(
                "time",
                raw_event.get("timestamp", _now())
            ),
            "source_ip": source_ip,
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "api_gateway_azure",
            "raw_event": raw_event,
            "api_vendor": "azure_apim",
            "api_method": method,
            "api_path": path,
            "api_status": status,
            "api_subscription_id": subscription_id,
            "api_product_id": product_id,
            "api_user_id": user_id,
            "api_duration_ms": duration,
            "api_response_bytes": response_size
        }

    def normalize_kong(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize Kong API Gateway log.
        Kong is popular for microservices.
        """
        request = raw_event.get("request", {})
        response = raw_event.get("response", {})
        route = raw_event.get("route", {})
        service = raw_event.get("service", {})
        consumer = raw_event.get("consumer", {})

        method = (
            request.get("method", "GET")
            if isinstance(request, dict)
            else "GET"
        ).upper()

        uri = (
            request.get("uri", "")
            if isinstance(request, dict)
            else ""
        )
        path = (
            request.get("url", uri)
            if isinstance(request, dict)
            else uri
        )
        path = self._extract_path(str(path))

        source_ip = (
            raw_event.get("client_ip", "") or
            (request.get("headers", {}).get(
                "x-forwarded-for", ""
            ) if isinstance(request, dict) else "")
        )

        status = int(
            (response.get("status", 200)
             if isinstance(response, dict)
             else raw_event.get("status", 200)
             ) or 200
        )

        request_size = int(
            (request.get("size", 0)
             if isinstance(request, dict)
             else 0) or 0
        )
        response_size = int(
            (response.get("size", 0)
             if isinstance(response, dict)
             else 0) or 0
        )

        consumer_id = (
            consumer.get("id", "")
            if isinstance(consumer, dict)
            else ""
        )
        consumer_username = (
            consumer.get("username", "")
            if isinstance(consumer, dict)
            else ""
        )
        service_name = (
            service.get("name", "")
            if isinstance(service, dict)
            else ""
        )
        latency = raw_event.get("latencies", {})
        total_latency = (
            latency.get("request", 0)
            if isinstance(latency, dict)
            else 0
        )

        accessor = (
            consumer_username or
            consumer_id or
            source_ip or "unknown"
        )

        user_agent = (
            request.get("headers", {}).get(
                "user-agent", ""
            )
            if isinstance(request, dict)
            else ""
        )

        risk_score, risk_reasons = (
            self._calculate_api_risk(
                method, path, status,
                source_ip, user_agent,
                response_size, raw_event
            )
        )

        return {
            "accessor_identity": accessor,
            "accessor_type": (
                "api_key" if consumer_id
                else "human"
            ),
            "data_store_name": (
                service_name or "kong-api-gateway"
            ),
            "data_path": f"{method} {path}",
            "data_classification": (
                self._detect_classification(path)
            ),
            "bytes_accessed": response_size,
            "event_time": raw_event.get(
                "started_at",
                raw_event.get("timestamp", _now())
            ),
            "source_ip": source_ip,
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "api_gateway_kong",
            "raw_event": raw_event,
            "api_vendor": "kong",
            "api_method": method,
            "api_path": path,
            "api_status": status,
            "api_consumer_id": consumer_id,
            "api_service": service_name,
            "api_latency_ms": total_latency,
            "api_response_bytes": response_size
        }

    def detect_api_abuse(
        self,
        events: list,
        window_minutes: int = 60,
        threshold: int = 100
    ) -> dict:
        """
        Detect API abuse patterns.
        Credential stuffing, scraping, enumeration.

        Args:
            events: List of normalized API events
            window_minutes: Analysis window
            threshold: Min requests to flag

        Returns:
            dict with abuse analysis
        """
        if not events:
            return {
                "abuse_detected": False,
                "event_count": 0
            }

        ip_counts = {}
        endpoint_counts = {}
        status_counts = {}
        key_counts = {}

        for event in events:
            ip = event.get("source_ip", "")
            path = event.get("api_path", "")
            status = event.get("api_status", 200)
            key = event.get("api_key", "")

            if ip:
                ip_counts[ip] = (
                    ip_counts.get(ip, 0) + 1
                )
            if path:
                endpoint_counts[path] = (
                    endpoint_counts.get(path, 0) + 1
                )
            if status:
                status_counts[status] = (
                    status_counts.get(status, 0) + 1
                )
            if key:
                key_counts[key] = (
                    key_counts.get(key, 0) + 1
                )

        total = len(events)
        auth_failures = (
            status_counts.get(401, 0) +
            status_counts.get(403, 0)
        )
        not_found = status_counts.get(404, 0)
        rate_limited = status_counts.get(429, 0)

        top_ip = max(
            ip_counts.items(),
            key=lambda x: x[1],
            default=("", 0)
        )
        top_endpoint = max(
            endpoint_counts.items(),
            key=lambda x: x[1],
            default=("", 0)
        )

        abuse_type = "none"
        abuse_risk = 0.0

        if auth_failures / max(total, 1) > 0.50:
            abuse_type = "credential_stuffing"
            abuse_risk = 0.90

        elif not_found / max(total, 1) > 0.40:
            abuse_type = "api_enumeration"
            abuse_risk = 0.70

        elif (
            top_ip[1] >= threshold and
            top_endpoint[1] >= threshold * 0.8
        ):
            abuse_type = "data_scraping"
            abuse_risk = 0.80

        elif rate_limited > 50:
            abuse_type = "rate_limit_abuse"
            abuse_risk = 0.65

        abuse_detected = abuse_type != "none"

        return {
            "abuse_detected": abuse_detected,
            "abuse_type": abuse_type,
            "abuse_risk_score": abuse_risk,
            "event_count": total,
            "auth_failure_count": auth_failures,
            "auth_failure_rate": (
                auth_failures / max(total, 1)
            ),
            "not_found_count": not_found,
            "rate_limited_count": rate_limited,
            "top_attacker_ip": top_ip[0],
            "top_attacker_requests": top_ip[1],
            "top_endpoint": top_endpoint[0],
            "unique_ips": len(ip_counts),
            "recommendation": (
                f"BLOCK {top_ip[0]}. "
                f"Abuse type: {abuse_type}. "
                f"Risk: {abuse_risk:.2f}."
                if abuse_detected
                else "Normal API activity"
            )
        }

    def _calculate_api_risk(
        self,
        method: str,
        path: str,
        status: int,
        source_ip: str,
        user_agent: str,
        response_bytes: int,
        raw_event: dict
    ) -> tuple:
        """Calculate API gateway event risk"""
        method_risk = HTTP_METHOD_RISK.get(
            method.upper(), 0.20
        )
        status_risk = HTTP_STATUS_RISK.get(
            status, 0.20
        )
        risk = max(method_risk, status_risk)
        reasons = [
            f"http_method:{method}",
            f"http_status:{status}"
        ]

        if self._is_high_value_endpoint(path):
            risk = min(risk + 0.15, 1.0)
            reasons.append(
                "high_value_api_endpoint"
            )

        if source_ip.startswith("185.220"):
            risk = min(risk + 0.30, 1.0)
            reasons.append("tor_exit_node")

        ua_lower = user_agent.lower()
        for suspicious_ua in SUSPICIOUS_USER_AGENTS:
            if suspicious_ua in ua_lower:
                risk = min(risk + 0.15, 1.0)
                reasons.append(
                    f"suspicious_user_agent:"
                    f"{suspicious_ua}"
                )
                break

        resp_mb = response_bytes / (1024 * 1024)
        if resp_mb > 50:
            risk = min(risk + 0.20, 1.0)
            reasons.append(
                f"large_api_response:"
                f"{resp_mb:.1f}mb"
            )

        if status == 401:
            risk = min(risk + 0.10, 1.0)
            reasons.append("authentication_failure")

        if status == 403:
            risk = min(risk + 0.15, 1.0)
            reasons.append("authorization_failure")

        if status == 429:
            risk = min(risk + 0.20, 1.0)
            reasons.append("rate_limit_exceeded")

        return min(risk, 1.0), reasons

    def _detect_vendor(
        self, raw_event: dict
    ) -> str:
        """Auto-detect API gateway vendor"""
        if "apiId" in raw_event and (
            "resourcePath" in raw_event or
            "sourceIp" in raw_event
        ):
            return "aws_apigw"

        if "apiKey" in raw_event and (
            "requestId" in raw_event
        ):
            return "aws_apigw"

        if "callerIpAddress" in raw_event and (
            "responseCode" in raw_event or
            "operationId" in raw_event
        ):
            return "azure_apim"

        if "subscriptionId" in raw_event and (
            "productId" in raw_event
        ):
            return "azure_apim"

        if "consumer" in raw_event and (
            "route" in raw_event or
            "service" in raw_event
        ):
            return "kong"

        if "latencies" in raw_event:
            return "kong"

        return "generic"

    def _is_high_value_endpoint(
        self, path: str
    ) -> bool:
        """Check if API path is high value"""
        path_lower = path.lower()
        return any(
            ep in path_lower
            for ep in HIGH_VALUE_API_ENDPOINTS
        )

    def _detect_classification(
        self, path: str
    ) -> str:
        """Detect data classification from path"""
        path_lower = path.lower()

        if any(
            kw in path_lower
            for kw in [
                "pci", "card", "payment",
                "credit", "billing", "transaction"
            ]
        ):
            return "PCI"

        if any(
            kw in path_lower
            for kw in [
                "phi", "health", "medical",
                "patient", "hipaa"
            ]
        ):
            return "PHI"

        if any(
            kw in path_lower
            for kw in [
                "pii", "customer", "user",
                "account", "personal", "member"
            ]
        ):
            return "PII"

        return "UNKNOWN"

    def _extract_path(self, url: str) -> str:
        """Extract path from full URL"""
        if not url:
            return ""
        try:
            if "://" in url:
                path_start = url.index("/", 8)
                path = url[path_start:]
                if "?" in path:
                    path = path[:path.index("?")]
                return path
        except (ValueError, IndexError):
            pass
        return url[:200]

    def _normalize_generic(
        self, raw_event: dict
    ) -> dict:
        """Generic API gateway normalization"""
        method = (
            raw_event.get("method", "") or
            raw_event.get("httpMethod", "") or
            raw_event.get("Method", "GET")
        ).upper()
        path = (
            raw_event.get("path", "") or
            raw_event.get("url", "") or
            raw_event.get("resourcePath", "")
        )
        status = int(
            raw_event.get("status", "") or
            raw_event.get("statusCode", 200) or 200
        )
        source_ip = (
            raw_event.get("ip", "") or
            raw_event.get("sourceIp", "") or
            raw_event.get("callerIpAddress", "")
        )
        user_agent = raw_event.get(
            "userAgent", ""
        )
        response_bytes = int(
            raw_event.get("responseLength", 0) or
            raw_event.get("bytes", 0) or 0
        )

        risk_score, risk_reasons = (
            self._calculate_api_risk(
                method, path, status,
                source_ip, user_agent,
                response_bytes, raw_event
            )
        )

        return {
            "accessor_identity": (
                source_ip or "unknown"
            ),
            "accessor_type": "api_key",
            "data_store_name": "api-gateway",
            "data_path": f"{method} {path}",
            "data_classification": (
                self._detect_classification(path)
            ),
            "bytes_accessed": response_bytes,
            "event_time": raw_event.get(
                "timestamp", _now()
            ),
            "source_ip": source_ip,
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "api_gateway_generic",
            "raw_event": raw_event,
            "api_vendor": "unknown",
            "api_method": method,
            "api_path": path,
            "api_status": status
        }

    def _empty_event(self) -> dict:
        return {
            "accessor_identity": "unknown",
            "accessor_type": "api_key",
            "data_store_name": "unknown",
            "data_path": "",
            "data_classification": "UNKNOWN",
            "bytes_accessed": 0,
            "event_time": _now(),
            "source_ip": "",
            "risk_score": 0.0,
            "risk_reasons": [],
            "source_system": "api_gateway",
            "raw_event": {}
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")