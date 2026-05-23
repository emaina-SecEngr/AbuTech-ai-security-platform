"""
Layer 3 — Knowledge Graph
AlienVault OTX Feed Integration

Open Threat Exchange — community threat intelligence.
Free. Updated continuously by security community.

WHAT OTX PROVIDES:
    Pulses: curated threat intelligence packages.
    Each pulse contains:
    - IOCs (Indicators of Compromise)
    - Malicious IP addresses
    - Malicious domains
    - File hashes (malware)
    - URLs (phishing, C2)
    - CVE references
    
    CONTRIBUTORS:
    5 million+ security researchers worldwide.
    Updated in real time.
    Free API access.

WHY THIS MATTERS FOR YOUR PLATFORM:
    When 185.220.101.45 appears in an event:
    OTX says: "Known Tor exit node.
    Seen in 847 malicious pulses.
    Associated with: APT29, ransomware, C2."
    
    Your GNN scores the connection higher.
    Your agents have real context.
    Not just "this IP looks suspicious."
    But "this IP is confirmed malicious."

USAGE:
    feed = OTXFeed()
    result = feed.get_ip_reputation("185.220.101.45")
    result = feed.get_domain_reputation("evil.com")
    result = feed.get_url_reputation("http://evil.com/payload")
    pulses = feed.get_latest_pulses(tags=["financial"])

API: otx.alienvault.com
COST: FREE with registration
KEY: Set OTX_API_KEY environment variable
"""

import logging
import os
from datetime import datetime
from datetime import timezone
from typing import Optional

logger = logging.getLogger(__name__)

OTX_API_BASE = "https://otx.alienvault.com/api/v1"


class OTXFeed:
    """
    AlienVault OTX threat intelligence feed.
    Community-sourced IOC database.
    Free API with registration.
    """

    def __init__(self, api_key: str = None):
        self.api_key = (
            api_key or
            os.getenv("OTX_API_KEY", "")
        )

    def get_ip_reputation(
        self, ip: str
    ) -> dict:
        """
        Get threat intelligence for an IP address.

        Args:
            ip: IP address to check

        Returns:
            dict with reputation data and risk score
        """
        if not ip:
            return self._empty_ip_result(ip)

        if not self.api_key:
            return self._simulated_ip_result(ip)

        try:
            import requests

            url = (
                f"{OTX_API_BASE}/indicators/IPv4/"
                f"{ip}/general"
            )
            headers = {"X-OTX-API-KEY": self.api_key}

            response = requests.get(
                url, headers=headers, timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                return self._format_ip_result(
                    ip, data
                )

            return self._empty_ip_result(ip)

        except Exception as e:
            logger.error(
                f"OTX IP lookup failed for {ip}: {e}"
            )
            return self._empty_ip_result(ip)

    def get_domain_reputation(
        self, domain: str
    ) -> dict:
        """
        Get threat intelligence for a domain.

        Args:
            domain: Domain to check

        Returns:
            dict with reputation data and risk score
        """
        if not domain:
            return self._empty_domain_result(domain)

        if not self.api_key:
            return self._simulated_domain_result(
                domain
            )

        try:
            import requests

            url = (
                f"{OTX_API_BASE}/indicators/domain/"
                f"{domain}/general"
            )
            headers = {"X-OTX-API-KEY": self.api_key}

            response = requests.get(
                url, headers=headers, timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                return self._format_domain_result(
                    domain, data
                )

            return self._empty_domain_result(domain)

        except Exception as e:
            logger.error(
                f"OTX domain lookup failed: {e}"
            )
            return self._empty_domain_result(domain)

    def get_latest_pulses(
        self,
        tags: list = None,
        limit: int = 20
    ) -> list:
        """
        Get latest threat intelligence pulses.
        Filter by tags for financial sector focus.

        Args:
            tags: Filter pulses by tags
                  e.g. ["financial", "banking", "apt"]
            limit: Max pulses to return

        Returns:
            List of threat pulse dicts
        """
        if not self.api_key:
            return self._simulated_pulses()

        try:
            import requests

            url = f"{OTX_API_BASE}/pulses/subscribed"
            headers = {"X-OTX-API-KEY": self.api_key}
            params = {"limit": limit}

            if tags:
                params["tags"] = ",".join(tags)

            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                pulses = data.get("results", [])
                return [
                    self._format_pulse(p)
                    for p in pulses
                ]

            return []

        except Exception as e:
            logger.error(
                f"OTX pulse fetch failed: {e}"
            )
            return []

    def get_financial_sector_pulses(self) -> list:
        """
        Get pulses specifically targeting
        financial sector. Used to update
        knowledge graph with banking threats.
        """
        financial_tags = [
            "financial", "banking", "fintech",
            "swift", "atm", "credit-card",
            "financial-fraud", "apt"
        ]
        return self.get_latest_pulses(
            tags=financial_tags
        )

    def extract_iocs_for_graph(
        self,
        pulses: list
    ) -> dict:
        """
        Extract IOCs from pulses for knowledge graph.

        Returns:
            dict with categorized IOCs ready for
            SecurityKnowledgeGraph ingestion
        """
        iocs = {
            "malicious_ips": set(),
            "malicious_domains": set(),
            "malicious_urls": set(),
            "malware_hashes": set(),
            "threat_actors": set(),
            "cves": set()
        }

        for pulse in pulses:
            for ioc in pulse.get("indicators", []):
                ioc_type = ioc.get("type", "")
                ioc_value = ioc.get("indicator", "")

                if not ioc_value:
                    continue

                if ioc_type == "IPv4":
                    iocs["malicious_ips"].add(ioc_value)
                elif ioc_type in ["domain", "hostname"]:
                    iocs["malicious_domains"].add(
                        ioc_value
                    )
                elif ioc_type == "URL":
                    iocs["malicious_urls"].add(ioc_value)
                elif ioc_type in [
                    "FileHash-MD5", "FileHash-SHA1",
                    "FileHash-SHA256"
                ]:
                    iocs["malware_hashes"].add(ioc_value)
                elif ioc_type == "CVE":
                    iocs["cves"].add(ioc_value)

        # Convert sets to lists for JSON serialization
        return {
            k: list(v) for k, v in iocs.items()
        }

    def _format_ip_result(
        self, ip: str, data: dict
    ) -> dict:
        """Format OTX IP response"""
        pulse_count = data.get(
            "pulse_info", {}
        ).get("count", 0)

        reputation = data.get("reputation", 0)

        if pulse_count >= 10 or reputation < -1:
            risk_score = 0.90
            risk_label = "CRITICAL"
        elif pulse_count >= 5:
            risk_score = 0.75
            risk_label = "HIGH"
        elif pulse_count >= 1:
            risk_score = 0.50
            risk_label = "MEDIUM"
        else:
            risk_score = 0.10
            risk_label = "LOW"

        return {
            "ip": ip,
            "pulse_count": pulse_count,
            "reputation": reputation,
            "country": data.get(
                "country_name", ""
            ),
            "asn": data.get("asn", ""),
            "risk_score": risk_score,
            "risk_label": risk_label,
            "source": "otx",
            "checked_at": _now()
        }

    def _format_domain_result(
        self, domain: str, data: dict
    ) -> dict:
        """Format OTX domain response"""
        pulse_count = data.get(
            "pulse_info", {}
        ).get("count", 0)

        if pulse_count >= 5:
            risk_score = 0.85
            risk_label = "HIGH"
        elif pulse_count >= 1:
            risk_score = 0.55
            risk_label = "MEDIUM"
        else:
            risk_score = 0.10
            risk_label = "LOW"

        return {
            "domain": domain,
            "pulse_count": pulse_count,
            "risk_score": risk_score,
            "risk_label": risk_label,
            "source": "otx",
            "checked_at": _now()
        }

    def _format_pulse(self, pulse: dict) -> dict:
        """Format OTX pulse for platform"""
        return {
            "id": pulse.get("id", ""),
            "name": pulse.get("name", ""),
            "description": pulse.get(
                "description", ""
            )[:500],
            "tags": pulse.get("tags", []),
            "indicators": pulse.get(
                "indicators", []
            ),
            "targeted_countries": pulse.get(
                "targeted_countries", []
            ),
            "adversary": pulse.get("adversary", ""),
            "malware_families": pulse.get(
                "malware_families", []
            ),
            "attack_ids": pulse.get(
                "attack_ids", []
            ),
            "created": pulse.get("created", ""),
            "modified": pulse.get("modified", ""),
            "indicator_count": pulse.get(
                "indicator_count", 0
            )
        }

    def _simulated_ip_result(self, ip: str) -> dict:
        """Simulated result for testing"""
        if ip.startswith("185.220"):
            return {
                "ip": ip,
                "pulse_count": 847,
                "reputation": -3,
                "country": "Netherlands",
                "asn": "AS60729",
                "risk_score": 0.97,
                "risk_label": "CRITICAL",
                "source": "otx_simulated",
                "notes": "Known Tor exit node",
                "checked_at": _now()
            }
        if ip.startswith("45.142"):
            return {
                "ip": ip,
                "pulse_count": 234,
                "reputation": -2,
                "country": "Russia",
                "asn": "AS47764",
                "risk_score": 0.88,
                "risk_label": "HIGH",
                "source": "otx_simulated",
                "checked_at": _now()
            }
        if ip.startswith("10.") or ip.startswith("192.168."):
            return {
                "ip": ip,
                "pulse_count": 0,
                "reputation": 0,
                "country": "Internal",
                "risk_score": 0.0,
                "risk_label": "INTERNAL",
                "source": "otx_simulated",
                "checked_at": _now()
            }
        return {
            "ip": ip,
            "pulse_count": 0,
            "reputation": 0,
            "risk_score": 0.10,
            "risk_label": "UNKNOWN",
            "source": "otx_simulated",
            "checked_at": _now()
        }

    def _simulated_domain_result(
        self, domain: str
    ) -> dict:
        """Simulated domain result for testing"""
        if any(
            kw in domain.lower()
            for kw in ["evil", "malware", "c2",
                       "duckdns", "ngrok"]
        ):
            return {
                "domain": domain,
                "pulse_count": 45,
                "risk_score": 0.90,
                "risk_label": "HIGH",
                "source": "otx_simulated",
                "checked_at": _now()
            }
        return {
            "domain": domain,
            "pulse_count": 0,
            "risk_score": 0.05,
            "risk_label": "CLEAN",
            "source": "otx_simulated",
            "checked_at": _now()
        }

    def _simulated_pulses(self) -> list:
        """Simulated pulses for testing"""
        return [
            {
                "id": "sim-001",
                "name": "APT29 Financial Sector Campaign",
                "description": "APT29 targeting US banks via spearphishing",
                "tags": ["apt29", "financial", "banking"],
                "indicators": [
                    {"type": "IPv4", "indicator": "198.51.100.42"},
                    {"type": "domain", "indicator": "evil-bank-login.com"},
                    {"type": "FileHash-SHA256",
                     "indicator": "abc123def456"}
                ],
                "adversary": "APT29",
                "attack_ids": ["T1566.002", "T1078"],
                "indicator_count": 3
            }
        ]

    def _empty_ip_result(self, ip: str) -> dict:
        """Empty result for invalid input"""
        return {
            "ip": ip,
            "pulse_count": 0,
            "risk_score": 0.0,
            "risk_label": "UNKNOWN",
            "source": "otx"
        }

    def _empty_domain_result(
        self, domain: str
    ) -> dict:
        """Empty result for invalid input"""
        return {
            "domain": domain,
            "pulse_count": 0,
            "risk_score": 0.0,
            "risk_label": "UNKNOWN",
            "source": "otx"
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")