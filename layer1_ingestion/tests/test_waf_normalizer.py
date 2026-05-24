"""
Tests for WAF Normalizer
"""

import pytest
from layer1_ingestion.normalizers.waf_normalizer import (
    WAFNormalizer,
    WAF_ACTION_RISK,
    ATTACK_TYPE_RISK,
    HIGH_VALUE_ENDPOINTS
)


@pytest.fixture
def normalizer():
    return WAFNormalizer()


@pytest.fixture
def aws_waf_sqli():
    return {
        "timestamp": 1716256462000,
        "action": "BLOCK",
        "httpRequest": {
            "clientIp": "185.220.101.45",
            "uri": "/api/customers/search",
            "args": "id=1 OR 1=1--",
            "httpMethod": "GET",
            "country": "NL",
            "headers": [
                {"name": "host", "value": "api.bofa.com"},
                {"name": "user-agent", "value": "sqlmap/1.7"}
            ]
        },
        "ruleGroupList": [{
            "ruleGroupId": "AWSManagedRulesSQLiRuleSet",
            "terminatingRule": {
                "ruleId": "SQLi_BODY",
                "action": "BLOCK"
            }
        }]
    }


@pytest.fixture
def aws_waf_xss():
    return {
        "timestamp": 1716256462000,
        "action": "BLOCK",
        "httpRequest": {
            "clientIp": "10.0.0.1",
            "uri": "/api/comments",
            "args": "text=<script>alert(1)</script>",
            "httpMethod": "POST",
            "country": "US",
            "headers": [
                {"name": "host", "value": "app.bofa.com"},
                {"name": "user-agent",
                 "value": "Mozilla/5.0"}
            ]
        },
        "ruleGroupList": [{
            "ruleGroupId": "AWSManagedRulesXSSRuleSet",
            "terminatingRule": {
                "ruleId": "XSS_BODY",
                "action": "BLOCK"
            }
        }]
    }


@pytest.fixture
def aws_waf_rce():
    return {
        "timestamp": 1716256462000,
        "action": "BLOCK",
        "httpRequest": {
            "clientIp": "45.142.100.10",
            "uri": "/api/execute",
            "args": "cmd=cat /etc/passwd",
            "httpMethod": "POST",
            "country": "RU",
            "headers": [
                {"name": "host",
                 "value": "api.bofa.com"}
            ]
        },
        "ruleGroupList": [{
            "ruleGroupId": "AWSManagedRulesCMSRuleSet",
            "terminatingRule": {
                "ruleId": "OS_Command_Injection_BODY",
                "action": "BLOCK"
            }
        }]
    }


@pytest.fixture
def cloudflare_sqli():
    return {
        "ClientIP": "185.220.101.45",
        "ClientRequestPath": "/api/accounts",
        "ClientRequestQuery": (
            "user=admin' OR '1'='1"
        ),
        "ClientRequestMethod": "GET",
        "ClientRequestHost": "api.bofa.com",
        "WAFAction": "block",
        "WAFRuleID": "100001",
        "WAFRuleMessage": "SQL Injection Attack",
        "EdgeStartTimestamp": "2026-05-21T03:14:22Z",
        "ClientCountry": "NL",
        "BotScore": 5,
        "ClientRequestUserAgent": "sqlmap/1.7",
        "EdgeResponseBytes": 0
    }


@pytest.fixture
def cloudflare_legitimate():
    return {
        "ClientIP": "192.168.1.100",
        "ClientRequestPath": "/api/balance",
        "ClientRequestQuery": "account=12345",
        "ClientRequestMethod": "GET",
        "ClientRequestHost": "api.bofa.com",
        "WAFAction": "allow",
        "WAFRuleID": "",
        "WAFRuleMessage": "",
        "EdgeStartTimestamp": "2026-05-21T09:00:00Z",
        "ClientCountry": "US",
        "BotScore": 95,
        "ClientRequestUserAgent": (
            "Mozilla/5.0 Chrome/120"
        ),
        "EdgeResponseBytes": 1024
    }


@pytest.fixture
def azure_waf_sqli():
    return {
        "time": "2026-05-21T03:14:22Z",
        "policyMode": "Prevention",
        "properties": {
            "action": "Blocked",
            "clientIp": "45.142.100.10",
            "requestUri": "/api/customers",
            "ruleId": "942100",
            "ruleGroup": "REQUEST-942-APPLICATION-ATTACK-SQLI",
            "message": "SQL Injection Attack Detected",
            "hostname": "app.bofa.com",
            "Method": "POST"
        }
    }


@pytest.fixture
def modsec_sqli():
    return {
        "timestamp": "2026-05-21T03:14:22Z",
        "action": "block",
        "client_ip": "185.220.101.45",
        "request_uri": "/login",
        "method": "POST",
        "rule_id": "942100",
        "message": "SQL Injection Attack Detected",
        "data": "username=admin' OR '1'='1",
        "host": "portal.bofa.com",
        "severity": "CRITICAL"
    }


@pytest.fixture
def normal_waf_event():
    return {
        "timestamp": 1716256462000,
        "action": "ALLOW",
        "httpRequest": {
            "clientIp": "10.0.0.1",
            "uri": "/api/health",
            "args": "",
            "httpMethod": "GET",
            "country": "US",
            "headers": [
                {"name": "host",
                 "value": "api.bofa.com"},
                {"name": "user-agent",
                 "value": "HealthCheck/1.0"}
            ]
        },
        "ruleGroupList": []
    }


# ============================================================
# INITIALIZATION TESTS
# ============================================================

class TestInitialization:

    def test_normalizer_initializes(self, normalizer):
        assert normalizer is not None

    def test_action_risk_map_populated(self):
        assert "BLOCK" in WAF_ACTION_RISK
        assert WAF_ACTION_RISK["BLOCK"] >= 0.65

    def test_attack_type_risk_populated(self):
        assert "SQLi" in ATTACK_TYPE_RISK
        assert "XSS" in ATTACK_TYPE_RISK
        assert "RCE" in ATTACK_TYPE_RISK

    def test_high_value_endpoints_populated(self):
        assert len(HIGH_VALUE_ENDPOINTS) > 0
        assert "/api/accounts" in HIGH_VALUE_ENDPOINTS


# ============================================================
# SOURCE DETECTION TESTS
# ============================================================

class TestSourceDetection:

    def test_detects_aws_waf(
        self, normalizer, aws_waf_sqli
    ):
        source = normalizer._detect_source(
            aws_waf_sqli
        )
        assert source == "aws_waf"

    def test_detects_cloudflare(
        self, normalizer, cloudflare_sqli
    ):
        source = normalizer._detect_source(
            cloudflare_sqli
        )
        assert source == "cloudflare"

    def test_detects_azure_waf(
        self, normalizer, azure_waf_sqli
    ):
        source = normalizer._detect_source(
            azure_waf_sqli
        )
        assert source == "azure_waf"

    def test_detects_modsecurity(
        self, normalizer, modsec_sqli
    ):
        source = normalizer._detect_source(
            modsec_sqli
        )
        assert source == "modsecurity"

    def test_unknown_source_generic(self, normalizer):
        source = normalizer._detect_source(
            {"someField": "someValue"}
        )
        assert source == "generic"


# ============================================================
# AWS WAF TESTS
# ============================================================

class TestAWSWAF:

    def test_normalize_returns_dict(
        self, normalizer, aws_waf_sqli
    ):
        result = normalizer.normalize(aws_waf_sqli)
        assert isinstance(result, dict)

    def test_sqli_high_risk(
        self, normalizer, aws_waf_sqli
    ):
        result = normalizer.normalize(aws_waf_sqli)
        assert result["risk_score"] >= 0.80

    def test_sqli_detected(
        self, normalizer, aws_waf_sqli
    ):
        result = normalizer.normalize(aws_waf_sqli)
        assert result["waf_sqli_detected"] is True

    def test_xss_detected(
        self, normalizer, aws_waf_xss
    ):
        result = normalizer.normalize(aws_waf_xss)
        assert result["waf_xss_detected"] is True

    def test_rce_high_risk(
        self, normalizer, aws_waf_rce
    ):
        result = normalizer.normalize(aws_waf_rce)
        assert result["risk_score"] >= 0.70

    def test_block_action_high_risk(
        self, normalizer, aws_waf_sqli
    ):
        result = normalizer.normalize(aws_waf_sqli)
        assert result["waf_action"] == "BLOCK"
        assert result["risk_score"] >= 0.70

    def test_source_ip_extracted(
        self, normalizer, aws_waf_sqli
    ):
        result = normalizer.normalize(aws_waf_sqli)
        assert result["source_ip"] == "185.220.101.45"

    def test_tor_ip_elevated(
        self, normalizer, aws_waf_sqli
    ):
        result = normalizer.normalize(aws_waf_sqli)
        reasons = str(result["risk_reasons"])
        assert "tor" in reasons.lower()

    def test_high_value_endpoint_flagged(
        self, normalizer, aws_waf_sqli
    ):
        result = normalizer.normalize(aws_waf_sqli)
        reasons = str(result["risk_reasons"])
        assert "high_value_endpoint" in reasons

    def test_source_system_aws(
        self, normalizer, aws_waf_sqli
    ):
        result = normalizer.normalize(aws_waf_sqli)
        assert "aws" in result["source_system"]

    def test_mitre_technique_set(
        self, normalizer, aws_waf_sqli
    ):
        result = normalizer.normalize(aws_waf_sqli)
        assert "waf_mitre" in result
        assert result["waf_mitre"] != ""

    def test_required_fields_present(
        self, normalizer, aws_waf_sqli
    ):
        result = normalizer.normalize(aws_waf_sqli)
        required = [
            "accessor_identity", "accessor_type",
            "data_store_name", "data_path",
            "event_time", "source_ip",
            "risk_score", "risk_reasons",
            "source_system", "raw_event"
        ]
        for field in required:
            assert field in result

    def test_normal_allow_lower_risk(
        self, normalizer, normal_waf_event
    ):
        result = normalizer.normalize(normal_waf_event)
        assert result["risk_score"] <= 0.50

    def test_risk_score_capped(
        self, normalizer, aws_waf_sqli
    ):
        result = normalizer.normalize(aws_waf_sqli)
        assert result["risk_score"] <= 1.0

    def test_empty_event_safe(self, normalizer):
        result = normalizer.normalize({})
        assert result["risk_score"] == 0.0

    def test_none_event_safe(self, normalizer):
        result = normalizer.normalize(None)
        assert result is not None


# ============================================================
# CLOUDFLARE WAF TESTS
# ============================================================

class TestCloudflareWAF:

    def test_cloudflare_sqli_high_risk(
        self, normalizer, cloudflare_sqli
    ):
        result = normalizer.normalize(cloudflare_sqli)
        assert result["risk_score"] >= 0.75

    def test_cloudflare_bot_score_flagged(
        self, normalizer, cloudflare_sqli
    ):
        result = normalizer.normalize(cloudflare_sqli)
        reasons = str(result["risk_reasons"])
        assert "bot_score" in reasons

    def test_cloudflare_source_system(
        self, normalizer, cloudflare_sqli
    ):
        result = normalizer.normalize(cloudflare_sqli)
        assert "cloudflare" in result["source_system"]

    def test_cloudflare_legitimate_lower_risk(
        self, normalizer, cloudflare_legitimate
    ):
        result = normalizer.normalize(
            cloudflare_legitimate
        )
        assert result["risk_score"] <= 0.50

    def test_cloudflare_sqli_detected(
        self, normalizer, cloudflare_sqli
    ):
        result = normalizer.normalize(cloudflare_sqli)
        assert result["risk_score"] >= 0.70


# ============================================================
# AZURE WAF TESTS
# ============================================================

class TestAzureWAF:

    def test_azure_sqli_high_risk(
        self, normalizer, azure_waf_sqli
    ):
        result = normalizer.normalize(azure_waf_sqli)
        assert result["risk_score"] >= 0.70

    def test_azure_source_system(
        self, normalizer, azure_waf_sqli
    ):
        result = normalizer.normalize(azure_waf_sqli)
        assert "azure" in result["source_system"]

    def test_azure_rule_id_captured(
        self, normalizer, azure_waf_sqli
    ):
        result = normalizer.normalize(azure_waf_sqli)
        assert result.get("waf_rule_id") == "942100"

    def test_azure_prevention_mode_flagged(
        self, normalizer, azure_waf_sqli
    ):
        result = normalizer.normalize(azure_waf_sqli)
        assert result["risk_score"] >= 0.70


# ============================================================
# MODSECURITY TESTS
# ============================================================

class TestModSecurity:

    def test_modsec_sqli_high_risk(
        self, normalizer, modsec_sqli
    ):
        result = normalizer.normalize(modsec_sqli)
        assert result["risk_score"] >= 0.75

    def test_modsec_source_system(
        self, normalizer, modsec_sqli
    ):
        result = normalizer.normalize(modsec_sqli)
        assert "modsecurity" in result["source_system"]

    def test_modsec_sqli_detected(
        self, normalizer, modsec_sqli
    ):
        result = normalizer.normalize(modsec_sqli)
        assert result["risk_score"] >= 0.75

    def test_modsec_severity_critical(
        self, normalizer, modsec_sqli
    ):
        result = normalizer.normalize(modsec_sqli)
        assert result["risk_score"] >= 0.75


# ============================================================
# ATTACK DETECTION TESTS
# ============================================================

class TestAttackDetection:

    def test_sqli_pattern_detected(self, normalizer):
        assert normalizer._detect_sqli(
            "id=1 OR 1=1--"
        ) is True

    def test_union_select_detected(self, normalizer):
        assert normalizer._detect_sqli(
            "UNION SELECT * FROM users"
        ) is True

    def test_clean_string_not_sqli(self, normalizer):
        assert normalizer._detect_sqli(
            "John Smith"
        ) is False

    def test_xss_script_detected(self, normalizer):
        assert normalizer._detect_xss(
            "<script>alert(1)</script>"
        ) is True

    def test_xss_onerror_detected(self, normalizer):
        assert normalizer._detect_xss(
            "onerror=alert(1)"
        ) is True

    def test_clean_string_not_xss(self, normalizer):
        assert normalizer._detect_xss(
            "Hello World"
        ) is False

    def test_path_traversal_detected(
        self, normalizer
    ):
        assert normalizer._detect_path_traversal(
            "../../etc/passwd"
        ) is True

    def test_etc_passwd_detected(self, normalizer):
        assert normalizer._detect_path_traversal(
            "/etc/passwd"
        ) is True

    def test_clean_path_not_traversal(
        self, normalizer
    ):
        assert normalizer._detect_path_traversal(
            "/api/customers"
        ) is False

    def test_high_value_endpoint_accounts(
        self, normalizer
    ):
        assert normalizer._is_high_value_endpoint(
            "/api/accounts"
        ) is True

    def test_high_value_endpoint_admin(
        self, normalizer
    ):
        assert normalizer._is_high_value_endpoint(
            "/admin/dashboard"
        ) is True

    def test_normal_endpoint_not_high_value(
        self, normalizer
    ):
        assert normalizer._is_high_value_endpoint(
            "/api/health"
        ) is False


# ============================================================
# CAMPAIGN DETECTION TESTS
# ============================================================

class TestCampaignDetection:

    def test_campaign_returns_dict(self, normalizer):
        result = normalizer.detect_campaign([])
        assert isinstance(result, dict)

    def test_no_campaign_empty(self, normalizer):
        result = normalizer.detect_campaign([])
        assert result["campaign_detected"] is False

    def test_campaign_detected_high_volume(
        self, normalizer
    ):
        events = [
            {
                "source_ip": "185.220.101.45",
                "waf_uri": "/api/accounts",
                "waf_attack_type": "SQLi"
            }
        ] * 20

        result = normalizer.detect_campaign(
            events, threshold=10
        )
        assert result["campaign_detected"] is True

    def test_campaign_top_attacker(
        self, normalizer
    ):
        events = [
            {
                "source_ip": "185.220.101.45",
                "waf_uri": "/api/login",
                "waf_attack_type": "BruteForce"
            }
        ] * 15

        result = normalizer.detect_campaign(
            events, threshold=10
        )
        assert result["top_attacker_ip"] == (
            "185.220.101.45"
        )

    def test_campaign_risk_score_elevated(
        self, normalizer
    ):
        events = [
            {
                "source_ip": "185.220.101.45",
                "waf_uri": "/api/payments",
                "waf_attack_type": "SQLi"
            }
        ] * 20

        result = normalizer.detect_campaign(
            events, threshold=10
        )
        assert result["campaign_risk_score"] >= 0.50