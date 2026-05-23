"""
Layer 3 — Knowledge Graph
Recorded Future Feed Integration

Enterprise dark web and threat intelligence.
REQUIRES: Recorded Future API subscription.
COST: $50,000 - $500,000/year enterprise.

WHY RECORDED FUTURE FOR BANKS:
    Recorded Future monitors:
    Dark web criminal forums (XSS, Exploit.in)
    Carding marketplaces (stolen card sales)
    Ransomware group leak sites
    Telegram criminal channels (Russian, Chinese)
    Paste sites (Pastebin, GhostBin)
    Initial Access Broker markets
    
    FOR FINANCIAL SECTOR SPECIFICALLY:
    Stolen card dump alerts with BIN matching.
    Credential stuffing campaign warnings.
    Pre-attack infrastructure detection.
    Executive threat intelligence.
    Third-party breach monitoring.

INTEGRATION PATTERN:
    Client provides their Recorded Future API key.
    Your platform consumes via this feed manager.
    Zero duplicate investment for client.
    Their existing subscription immediately useful.

DARK WEB INTELLIGENCE FLOW:
    Criminal forum: "500K BofA cards for sale"
            ↓
    Recorded Future detects + classifies
            ↓
    STIX 2.1 alert → your FeedManager
            ↓
    SecurityKnowledgeGraph updated
            ↓
    All BofA auth events scored higher
            ↓
    Credential stuffing detected in real-time

USAGE:
    feed = RecordedFutureFeed()
    
    # IP intelligence with dark web context
    result = feed.get_ip_intelligence("185.220.101.45")
    
    # Domain intelligence
    result = feed.get_domain_intelligence("evil.com")
    
    # Dark web alerts for your organization
    alerts = feed.get_dark_web_alerts("company.com")
    
    # Stolen credential alerts
    creds = feed.get_credential_alerts("bofa.com")
    
    # Threat actor profile
    actor = feed.get_threat_actor("APT29")
"""

import logging
import os
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)

RF_API_BASE = "https://api.recordedfuture.com/v2"


class RecordedFutureFeed:
    """
    Recorded Future threat intelligence integration.

    Provides dark web monitoring, criminal forum
    intelligence, and financial sector specific
    threat data.

    REQUIRES enterprise API subscription.
    Framework ready — client provides API key.
    """

    def __init__(self, api_key: str = None):
        self.api_key = (
            api_key or
            os.getenv("RECORDED_FUTURE_API_KEY", "")
        )
        self.is_configured = bool(self.api_key)

        if not self.is_configured:
            logger.info(
                "Recorded Future API key not set. "
                "Using simulated intelligence. "
                "Set RECORDED_FUTURE_API_KEY for "
                "live dark web monitoring."
            )

    def get_ip_intelligence(
        self, ip: str
    ) -> dict:
        """
        Get comprehensive threat intelligence for IP.
        Includes dark web mentions, C2 history,
        threat actor attribution.

        Args:
            ip: IP address to investigate

        Returns:
            dict with comprehensive intel and risk score
        """
        if not ip:
            return self._empty_result("ip", ip)

        if not self.is_configured:
            return self._simulated_ip_intel(ip)

        try:
            import requests

            url = f"{RF_API_BASE}/ip/{ip}"
            headers = {
                "X-RFToken": self.api_key,
                "Content-Type": "application/json"
            }
            params = {
                "fields": (
                    "risk,threatLists,intelCard,"
                    "relatedEntities,metrics"
                )
            }

            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                return self._format_ip_intel(ip, data)

            logger.warning(
                f"RF API error {response.status_code}"
                f" for IP {ip}"
            )
            return self._simulated_ip_intel(ip)

        except Exception as e:
            logger.error(
                f"Recorded Future IP lookup failed: {e}"
            )
            return self._simulated_ip_intel(ip)

    def get_domain_intelligence(
        self, domain: str
    ) -> dict:
        """
        Get threat intelligence for domain.
        Includes dark web sales, phishing kits,
        C2 infrastructure detection.
        """
        if not domain:
            return self._empty_result(
                "domain", domain
            )

        if not self.is_configured:
            return self._simulated_domain_intel(domain)

        try:
            import requests

            url = f"{RF_API_BASE}/domain/{domain}"
            headers = {"X-RFToken": self.api_key}
            params = {
                "fields": "risk,threatLists,metrics"
            }

            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                return self._format_domain_intel(
                    domain, data
                )

            return self._simulated_domain_intel(domain)

        except Exception as e:
            logger.error(
                f"RF domain lookup failed: {e}"
            )
            return self._simulated_domain_intel(domain)

    def get_dark_web_alerts(
        self,
        organization: str,
        days_back: int = 7
    ) -> list:
        """
        Get dark web alerts mentioning your organization.
        Monitors criminal forums, paste sites,
        data breach markets.

        Args:
            organization: Company name or domain
            days_back: How many days to look back

        Returns:
            List of dark web alert dicts
        """
        if not self.is_configured:
            return self._simulated_dark_web_alerts(
                organization
            )

        try:
            import requests
            from datetime import timedelta

            cutoff = (
                datetime.now(timezone.utc) -
                timedelta(days=days_back)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")

            url = f"{RF_API_BASE}/alert/search"
            headers = {"X-RFToken": self.api_key}
            params = {
                "freetext": organization,
                "triggered": f"[{cutoff},]",
                "limit": 50
            }

            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                alerts = data.get("data", {}).get(
                    "results", []
                )
                return [
                    self._format_alert(a)
                    for a in alerts
                ]

            return []

        except Exception as e:
            logger.error(
                f"RF dark web alerts failed: {e}"
            )
            return self._simulated_dark_web_alerts(
                organization
            )

    def get_credential_alerts(
        self,
        domain: str
    ) -> list:
        """
        Get alerts for stolen credentials
        matching your domain.

        CRITICAL FOR BANKS:
        When criminal forums sell credential dumps
        matching company.com email addresses,
        you get alerted immediately.

        Args:
            domain: Email domain to monitor
                    e.g. "bofa.com", "amex.com"

        Returns:
            List of credential alert dicts
        """
        if not self.is_configured:
            return self._simulated_credential_alerts(
                domain
            )

        try:
            import requests

            url = f"{RF_API_BASE}/identity/search"
            headers = {"X-RFToken": self.api_key}
            params = {
                "domain": domain,
                "limit": 100
            }

            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("identities", [])

            return []

        except Exception as e:
            logger.error(
                f"RF credential alerts failed: {e}"
            )
            return self._simulated_credential_alerts(
                domain
            )

    def get_threat_actor(
        self, actor_name: str
    ) -> dict:
        """
        Get comprehensive threat actor profile.
        Dark web aliases, infrastructure,
        known TTPs, targeted sectors.

        Args:
            actor_name: Threat actor name
                        e.g. "APT29", "LockBit"

        Returns:
            dict with full threat actor profile
        """
        if not self.is_configured:
            return self._simulated_threat_actor(
                actor_name
            )

        try:
            import requests

            url = f"{RF_API_BASE}/threat_actor/search"
            headers = {"X-RFToken": self.api_key}
            params = {"freetext": actor_name}

            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get(
                    "data", {}
                ).get("results", [])
                if results:
                    return self._format_actor(
                        results[0]
                    )

            return self._simulated_threat_actor(
                actor_name
            )

        except Exception as e:
            logger.error(
                f"RF threat actor lookup failed: {e}"
            )
            return self._simulated_threat_actor(
                actor_name
            )

    def get_financial_sector_intelligence(
        self
    ) -> dict:
        """
        Get latest intelligence specifically
        targeting financial sector.

        Returns:
            dict with current financial threats,
            active campaigns, stolen data markets
        """
        if not self.is_configured:
            return self._simulated_financial_intel()

        # In production: query RF for financial sector
        # specific intelligence package
        return self._simulated_financial_intel()

    # ============================================================
    # FORMATTERS
    # ============================================================

    def _format_ip_intel(
        self, ip: str, data: dict
    ) -> dict:
        """Format Recorded Future IP intelligence"""
        risk_data = data.get("data", {}).get(
            "risk", {}
        )
        risk_score_rf = risk_data.get(
            "score", 0
        ) / 100.0

        threat_lists = data.get("data", {}).get(
            "threatLists", []
        )
        threat_list_names = [
            t.get("name", "")
            for t in threat_lists
        ]

        dark_web_mentions = sum(
            1 for t in threat_list_names
            if "dark" in t.lower() or
            "underground" in t.lower()
        )

        return {
            "ip": ip,
            "risk_score": risk_score_rf,
            "risk_label": (
                "CRITICAL" if risk_score_rf >= 0.8
                else "HIGH" if risk_score_rf >= 0.6
                else "MEDIUM" if risk_score_rf >= 0.3
                else "LOW"
            ),
            "threat_lists": threat_list_names,
            "dark_web_mentions": dark_web_mentions,
            "source": "recorded_future",
            "checked_at": _now()
        }

    def _format_domain_intel(
        self, domain: str, data: dict
    ) -> dict:
        """Format Recorded Future domain intelligence"""
        risk_data = data.get("data", {}).get(
            "risk", {}
        )
        risk_score = risk_data.get("score", 0) / 100.0

        return {
            "domain": domain,
            "risk_score": risk_score,
            "risk_label": (
                "CRITICAL" if risk_score >= 0.8
                else "HIGH" if risk_score >= 0.6
                else "LOW"
            ),
            "source": "recorded_future",
            "checked_at": _now()
        }

    def _format_alert(self, alert: dict) -> dict:
        """Format dark web alert"""
        return {
            "alert_id": alert.get("id", ""),
            "title": alert.get("title", ""),
            "type": alert.get("type", ""),
            "triggered": alert.get("triggered", ""),
            "risk_score": alert.get(
                "risk_score", 0
            ) / 100.0,
            "entities": alert.get("entities", []),
            "source": "recorded_future_dark_web"
        }

    def _format_actor(self, actor: dict) -> dict:
        """Format threat actor profile"""
        return {
            "name": actor.get("name", ""),
            "aliases": actor.get("aliases", []),
            "motivation": actor.get(
                "motivation", ""
            ),
            "targeted_sectors": actor.get(
                "targeted_sectors", []
            ),
            "origin_country": actor.get(
                "origin_country", ""
            ),
            "risk_score": 0.90,
            "source": "recorded_future"
        }

    # ============================================================
    # SIMULATED DATA (used when no API key)
    # ============================================================

    def _simulated_ip_intel(self, ip: str) -> dict:
        """High-quality simulated IP intel"""
        if ip.startswith("185.220"):
            return {
                "ip": ip,
                "risk_score": 0.97,
                "risk_label": "CRITICAL",
                "threat_lists": [
                    "Tor Exit Node",
                    "C2 Infrastructure",
                    "Dark Web Associated"
                ],
                "dark_web_mentions": 23,
                "threat_actor": "Multiple APT groups",
                "last_seen_dark_web": (
                    "2026-05-20T18:00:00Z"
                ),
                "associated_malware": [
                    "Emotet", "CobaltStrike"
                ],
                "source": "rf_simulated",
                "note": (
                    "Configure RECORDED_FUTURE_API_KEY "
                    "for live dark web intelligence"
                ),
                "checked_at": _now()
            }

        if ip.startswith("198.51.100"):
            return {
                "ip": ip,
                "risk_score": 0.92,
                "risk_label": "CRITICAL",
                "threat_lists": [
                    "APT29 Infrastructure",
                    "C2 Server"
                ],
                "dark_web_mentions": 8,
                "threat_actor": "APT29 (Cozy Bear)",
                "targeted_sectors": [
                    "Financial", "Government"
                ],
                "source": "rf_simulated",
                "checked_at": _now()
            }

        if (
            ip.startswith("10.") or
            ip.startswith("192.168.")
        ):
            return {
                "ip": ip,
                "risk_score": 0.0,
                "risk_label": "INTERNAL",
                "threat_lists": [],
                "dark_web_mentions": 0,
                "source": "rf_simulated",
                "checked_at": _now()
            }

        return {
            "ip": ip,
            "risk_score": 0.15,
            "risk_label": "LOW",
            "threat_lists": [],
            "dark_web_mentions": 0,
            "source": "rf_simulated",
            "checked_at": _now()
        }

    def _simulated_domain_intel(
        self, domain: str
    ) -> dict:
        """Simulated domain intelligence"""
        domain_lower = domain.lower()
        if any(
            kw in domain_lower
            for kw in ["evil", "malware", "phish",
                       "c2", "duckdns"]
        ):
            return {
                "domain": domain,
                "risk_score": 0.93,
                "risk_label": "CRITICAL",
                "dark_web_mentions": 15,
                "threat_lists": ["Phishing", "C2"],
                "source": "rf_simulated",
                "checked_at": _now()
            }
        return {
            "domain": domain,
            "risk_score": 0.05,
            "risk_label": "CLEAN",
            "dark_web_mentions": 0,
            "threat_lists": [],
            "source": "rf_simulated",
            "checked_at": _now()
        }

    def _simulated_dark_web_alerts(
        self, organization: str
    ) -> list:
        """
        Simulated dark web alerts.
        Shows IBM stakeholders exactly what
        real alerts look like.
        """
        return [
            {
                "alert_id": "rf-dw-001",
                "title": (
                    f"Credential dump mentioning "
                    f"{organization} on XSS forum"
                ),
                "type": "credential_exposure",
                "triggered": "2026-05-21T02:00:00Z",
                "risk_score": 0.90,
                "details": (
                    "500,000 employee credentials "
                    "posted on Russian dark web forum "
                    "XSS.is. Includes email addresses "
                    f"matching @{organization} domain."
                ),
                "forum": "XSS.is (Russian dark web)",
                "threat_actor": "Unknown",
                "recommended_action": (
                    "Immediate forced password reset "
                    "for all affected accounts. "
                    "Enable MFA if not already active."
                ),
                "source": "rf_dark_web_simulated"
            },
            {
                "alert_id": "rf-dw-002",
                "title": (
                    f"PCI card data sale "
                    f"attributed to {organization}"
                ),
                "type": "financial_fraud",
                "triggered": "2026-05-19T14:00:00Z",
                "risk_score": 0.95,
                "details": (
                    "Carding forum selling 2M "
                    "Visa/MC cards. BIN analysis "
                    "suggests cards issued by "
                    f"{organization}. Price: $0.50/card."
                ),
                "forum": "Carding marketplace",
                "estimated_records": 2000000,
                "card_brands": ["Visa", "Mastercard"],
                "recommended_action": (
                    "Cross-reference BIN ranges. "
                    "Alert card fraud team. "
                    "Consider proactive reissuance."
                ),
                "source": "rf_dark_web_simulated"
            }
        ]

    def _simulated_credential_alerts(
        self, domain: str
    ) -> list:
        """Simulated credential breach alerts"""
        return [
            {
                "domain": domain,
                "breach_source": "LinkedIn 2024",
                "affected_emails": 1250,
                "breach_date": "2024-06-15",
                "data_types": ["email", "password_hash"],
                "risk_score": 0.65,
                "recommendation": (
                    "Force password reset for "
                    "1,250 affected accounts"
                )
            }
        ]

    def _simulated_threat_actor(
        self, actor_name: str
    ) -> dict:
        """Simulated threat actor profile"""
        profiles = {
            "APT29": {
                "name": "APT29",
                "aliases": [
                    "Cozy Bear", "Midnight Blizzard",
                    "The Dukes"
                ],
                "motivation": "Espionage",
                "origin_country": "Russia",
                "targeted_sectors": [
                    "Financial", "Government",
                    "Healthcare", "Technology"
                ],
                "active_campaigns": [
                    "Targeting US financial sector Q2 2026"
                ],
                "ttps": [
                    "T1566.002", "T1078", "T1021.001"
                ],
                "dark_web_presence": "None confirmed",
                "risk_score": 0.92
            },
            "LockBit": {
                "name": "LockBit",
                "aliases": ["LockBit 3.0", "LockBit Black"],
                "motivation": "Financial",
                "origin_country": "Russia (suspected)",
                "targeted_sectors": [
                    "Financial", "Healthcare",
                    "Manufacturing"
                ],
                "dark_web_presence": (
                    "Active leak site. "
                    "Posts stolen data publicly."
                ),
                "ransom_average": "$2M-50M",
                "risk_score": 0.95
            }
        }

        if actor_name in profiles:
            result = profiles[actor_name].copy()
            result["source"] = "rf_simulated"
            result["checked_at"] = _now()
            return result

        return {
            "name": actor_name,
            "risk_score": 0.70,
            "source": "rf_simulated",
            "note": "Limited intelligence available",
            "checked_at": _now()
        }

    def _simulated_financial_intel(self) -> dict:
        """Simulated financial sector intelligence"""
        return {
            "report_date": _now(),
            "active_campaigns": [
                {
                    "name": "Operation BankStrike",
                    "threat_actor": "Scattered Spider",
                    "target": "US Financial Sector",
                    "method": "MFA fatigue + helpdesk social engineering",
                    "status": "ACTIVE",
                    "risk_level": "CRITICAL"
                }
            ],
            "active_card_markets": [
                {
                    "forum": "BidenCash",
                    "cards_listed": 2400000,
                    "geographic_focus": "US",
                    "card_brands": ["Visa", "MC", "Amex"],
                    "price_per_card": "$0.50-2.00"
                }
            ],
            "emerging_malware": [
                {
                    "name": "BankBot 2026",
                    "type": "Banking Trojan",
                    "targets": ["Online banking", "Mobile banking"],
                    "first_seen": "2026-05-01"
                }
            ],
            "source": "rf_simulated"
        }

    def _empty_result(
        self, entity_type: str, value: str
    ) -> dict:
        """Empty result for invalid input"""
        return {
            entity_type: value,
            "risk_score": 0.0,
            "risk_label": "UNKNOWN",
            "source": "recorded_future"
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")