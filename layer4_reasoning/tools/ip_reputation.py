"""
Layer 4 — Agent Tools
Tool 1: IP Reputation Checker

Queries AbuseIPDB for IP threat intelligence.
Used by IntelAgent to get real threat data
instead of guessing from indicators alone.

IN PRODUCTION:
    Real AbuseIPDB API call.
    Requires ABUSEIPDB_API_KEY env variable.
    Rate limit: 1000 requests/day on free tier.

IN DEVELOPMENT/TESTING:
    Falls back to rule-based lookup.
    Known bad ranges return high scores.
    No API key required.

USAGE BY AGENTS:
    result = check_ip_reputation("185.220.101.45")
    print(result["score"])        # 97
    print(result["is_tor"])       # True
    print(result["categories"])   # ["VPN", "Tor"]
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Known malicious IP ranges
# In production: replaced by real AbuseIPDB data
KNOWN_BAD_RANGES = {
    "185.220": {
        "score": 97,
        "categories": ["Tor Exit Node", "VPN"],
        "is_tor": True,
        "country": "DE",
        "isp": "Tor Network",
        "threat_type": "tor_exit_node"
    },
    "162.247": {
        "score": 95,
        "categories": ["Tor Exit Node"],
        "is_tor": True,
        "country": "US",
        "isp": "Tor Network",
        "threat_type": "tor_exit_node"
    },
    "198.96": {
        "score": 88,
        "categories": ["Proxy", "VPN"],
        "is_tor": False,
        "country": "US",
        "isp": "Hosting Provider",
        "threat_type": "proxy"
    },
    "45.142": {
        "score": 92,
        "categories": ["Hacking", "Brute Force"],
        "is_tor": False,
        "country": "RU",
        "isp": "Unknown",
        "threat_type": "attacker"
    }
}

# Known safe IP ranges
KNOWN_SAFE_RANGES = {
    "10.": {"score": 0, "categories": [], "is_tor": False,
            "country": "PRIVATE", "isp": "Internal",
            "threat_type": "internal"},
    "192.168.": {"score": 0, "categories": [], "is_tor": False,
                 "country": "PRIVATE", "isp": "Internal",
                 "threat_type": "internal"},
    "172.16.": {"score": 0, "categories": [], "is_tor": False,
                "country": "PRIVATE", "isp": "Internal",
                "threat_type": "internal"},
}


def check_ip_reputation(
    ip: str,
    api_key: str = None
) -> dict:
    """
    Check IP reputation using AbuseIPDB.

    Args:
        ip: IP address to check
        api_key: AbuseIPDB API key (optional)
                 Falls back to rule-based if not set

    Returns:
        dict with:
            ip: str
            score: int (0-100, higher = more malicious)
            is_malicious: bool (score >= 50)
            is_tor: bool
            categories: list of threat categories
            country: str
            isp: str
            threat_type: str
            source: str (api or rule_based)
            summary: str (human readable)
    """
    if not ip:
        return _empty_result(ip)

    # Try real API first if key available
    api_key = api_key or os.getenv("ABUSEIPDB_API_KEY")
    if api_key:
        result = _query_abuseipdb(ip, api_key)
        if result:
            return result

    # Fall back to rule-based lookup
    return _rule_based_lookup(ip)


def _query_abuseipdb(ip: str, api_key: str) -> Optional[dict]:
    """
    Query real AbuseIPDB API.
    Returns None if API call fails.
    """
    try:
        import requests
        url = "https://api.abuseipdb.com/api/v2/check"
        headers = {
            "Accept": "application/json",
            "Key": api_key
        }
        params = {
            "ipAddress": ip,
            "maxAgeInDays": 90,
            "verbose": True
        }
        response = requests.get(
            url, headers=headers,
            params=params, timeout=5
        )
        if response.status_code == 200:
            data = response.json().get("data", {})
            score = data.get(
                "abuseConfidenceScore", 0
            )
            categories = _parse_categories(
                data.get("reports", [])
            )
            result = {
                "ip": ip,
                "score": score,
                "is_malicious": score >= 50,
                "is_tor": _check_tor_categories(
                    categories
                ),
                "categories": categories,
                "country": data.get(
                    "countryCode", "Unknown"
                ),
                "isp": data.get("isp", "Unknown"),
                "threat_type": _determine_threat_type(
                    score, categories
                ),
                "source": "abuseipdb_api",
                "summary": _build_summary(
                    ip, score, categories,
                    _check_tor_categories(categories)
                )
            }
            logger.info(
                f"AbuseIPDB: {ip} score={score}"
            )
            return result
    except Exception as e:
        logger.warning(
            f"AbuseIPDB API failed for {ip}: {e}"
        )
    return None


def _rule_based_lookup(ip: str) -> dict:
    """
    Rule-based IP reputation lookup.
    Used when AbuseIPDB API is not available.
    """
    # Check known safe ranges first
    for prefix, data in KNOWN_SAFE_RANGES.items():
        if ip.startswith(prefix):
            return {
                "ip": ip,
                "score": data["score"],
                "is_malicious": False,
                "is_tor": data["is_tor"],
                "categories": data["categories"],
                "country": data["country"],
                "isp": data["isp"],
                "threat_type": data["threat_type"],
                "source": "rule_based",
                "summary": (
                    f"IP {ip} is internal/private. "
                    f"No threat intelligence."
                )
            }

    # Check known bad ranges
    ip_prefix = ".".join(ip.split(".")[:2])
    if ip_prefix in KNOWN_BAD_RANGES:
        data = KNOWN_BAD_RANGES[ip_prefix]
        return {
            "ip": ip,
            "score": data["score"],
            "is_malicious": data["score"] >= 50,
            "is_tor": data["is_tor"],
            "categories": data["categories"],
            "country": data["country"],
            "isp": data["isp"],
            "threat_type": data["threat_type"],
            "source": "rule_based",
            "summary": _build_summary(
                ip, data["score"],
                data["categories"],
                data["is_tor"]
            )
        }

    # Unknown IP
    return {
        "ip": ip,
        "score": 0,
        "is_malicious": False,
        "is_tor": False,
        "categories": [],
        "country": "Unknown",
        "isp": "Unknown",
        "threat_type": "unknown",
        "source": "rule_based",
        "summary": (
            f"IP {ip} — no threat intelligence found. "
            f"Manual verification recommended."
        )
    }


def _parse_categories(reports: list) -> list:
    """Parse categories from AbuseIPDB reports"""
    category_map = {
        10: "Tor Exit Node",
        9: "Open Proxy",
        14: "DDoS Attack",
        18: "Brute Force",
        22: "SSH Attack",
        23: "IoT Targeted"
    }
    categories = set()
    for report in reports:
        for cat_id in report.get("categories", []):
            if cat_id in category_map:
                categories.add(category_map[cat_id])
    return list(categories)


def _check_tor_categories(categories: list) -> bool:
    """Check if IP is a Tor exit node"""
    tor_indicators = [
        "Tor Exit Node", "Tor", "tor"
    ]
    return any(
        t in categories
        for t in tor_indicators
    )


def _determine_threat_type(
    score: int,
    categories: list
) -> str:
    """Determine primary threat type"""
    if _check_tor_categories(categories):
        return "tor_exit_node"
    if score >= 90:
        return "high_confidence_attacker"
    if score >= 70:
        return "suspicious_host"
    if score >= 50:
        return "low_confidence_threat"
    return "clean"


def _build_summary(
    ip: str,
    score: int,
    categories: list,
    is_tor: bool
) -> str:
    """Build human-readable summary for agents"""
    if score >= 90:
        severity = "CRITICAL THREAT"
    elif score >= 70:
        severity = "HIGH THREAT"
    elif score >= 50:
        severity = "SUSPICIOUS"
    else:
        severity = "CLEAN"

    summary = f"IP {ip}: {severity} (AbuseIPDB: {score}/100). "

    if is_tor:
        summary += "Confirmed Tor exit node. "
        summary += "Traffic anonymized — attribution difficult. "

    if categories:
        summary += f"Categories: {', '.join(categories)}. "

    if score >= 70:
        summary += (
            "Recommend blocking at firewall "
            "and investigating all connections."
        )

    return summary


def _empty_result(ip: str) -> dict:
    """Return empty result for invalid input"""
    return {
        "ip": ip or "unknown",
        "score": 0,
        "is_malicious": False,
        "is_tor": False,
        "categories": [],
        "country": "Unknown",
        "isp": "Unknown",
        "threat_type": "unknown",
        "source": "rule_based",
        "summary": "No IP provided for lookup."
    }