"""
Layer 3 — Knowledge Graph
Threat Intelligence Feed Manager

This module integrates free open-source threat
intelligence feeds into your knowledge graph.

Feed Sources Implemented:
    1. AbuseIPDB
       Community-reported IP abuse database
       Free API key required
       https://www.abuseipdb.com

    2. Feodo Tracker (Abuse.ch)
       Emotet, TrickBot, Dridex C2 infrastructure
       No API key required
       https://feodotracker.abuse.ch

    3. URLhaus (Abuse.ch)
       Malicious URL database
       No API key required
       https://urlhaus.abuse.ch

Architecture Position:
    ThreatEnricher calls FeedManager
    FeedManager queries external APIs
    Results cached to respect rate limits
    Knowledge graph updated with live intelligence

Production Upgrade Path:
    Step 1: Free APIs (this file)
    Step 2: VirusTotal + MISP + STIX/TAXII
    Step 3: CISA PDF ingestion via LLM extraction
    Step 4: Commercial feeds (Recorded Future etc)

Secret Management:
    API keys retrieved via SecretsManager
    Development: .env file
    Production:  CyberArk AAM or HashiCorp Vault
    Code never changes between environments
"""

import json
import logging
import os
import time
from datetime import datetime
from datetime import timezone
from datetime import timedelta
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================
# CACHE CONFIGURATION
# API calls are expensive — cache results locally
# ============================================================

# How long to cache results before re-querying
CACHE_TTL_SECONDS = {
    "abuseipdb": 3600,      # 1 hour
    "feodo": 3600,           # 1 hour
    "urlhaus": 1800,         # 30 minutes
}

# Free tier rate limits
RATE_LIMITS = {
    "abuseipdb": {
        "requests_per_day": 1000,
        "min_seconds_between": 0.1
    },
    "virustotal": {
        "requests_per_minute": 4,
        "min_seconds_between": 15
    }
}


class ThreatFeedManager:
    """
    Manages external threat intelligence feed queries.

    Provides a unified interface for querying multiple
    threat intelligence sources with:
    - Automatic caching to respect rate limits
    - Combined risk scoring from multiple sources
    - Graceful degradation when feeds are unavailable
    - Full audit trail of enrichment sources

    Usage:
        manager = ThreatFeedManager()

        # Check an IP
        intel = manager.check_ip("185.220.101.45")
        print(intel["risk_score"])     # 0.95
        print(intel["malware_family"]) # "Emotet"

        # Check a domain
        intel = manager.check_domain("xjf8k2mp.duckdns.org")
        print(intel["is_malicious"])   # True
    """

    def __init__(
        self,
        cache_dir: str = ".threat_cache",
        abuseipdb_key: str = None
    ):
        """
        Initialize feed manager.

        Args:
            cache_dir: Directory for caching API responses
            abuseipdb_key: AbuseIPDB API key
                          If None reads from environment
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Get API key from environment or parameter
        self.abuseipdb_key = (
            abuseipdb_key or
            os.getenv("ABUSEIPDB_API_KEY", "")
        )

        # In-memory cache for current session
        self.ip_cache = {}
        self.domain_cache = {}
        self.url_cache = {}

        # Feodo blocklist cache
        self._feodo_blocklist = None
        self._feodo_loaded_at = None

        # URLhaus cache
        self._urlhaus_cache = {}

        # Statistics
        self.api_calls_made = 0
        self.cache_hits = 0
        self.enrichments_applied = 0

        logger.info("ThreatFeedManager initialized")

    # ============================================================
    # PRIMARY PUBLIC METHODS
    # ============================================================

    def check_ip(
        self,
        ip_address: str
    ) -> dict:
        """
        Check an IP address against all configured feeds.

        Queries:
        1. AbuseIPDB (if API key configured)
        2. Feodo Tracker (always available)
        3. Local known malicious list

        Returns combined intelligence with unified
        risk score on 0.0 to 1.0 scale.

        Args:
            ip_address: IP to check

        Returns:
            Intelligence dictionary with risk_score,
            sources, and threat context
        """
        # Check in-memory cache first
        if ip_address in self.ip_cache:
            cached = self.ip_cache[ip_address]
            if self._is_cache_valid(
                cached.get("cached_at"),
                CACHE_TTL_SECONDS["abuseipdb"]
            ):
                self.cache_hits += 1
                logger.debug(
                    f"Cache hit for IP: {ip_address}"
                )
                return cached

        intel = {
            "ip": ip_address,
            "risk_score": 0.0,
            "is_malicious": False,
            "malware_family": None,
            "threat_type": None,
            "sources": {},
            "tags": [],
            "cached_at": self._now()
        }

        # ---- QUERY ABUSEIPDB ----
        if self.abuseipdb_key:
            abuse_result = self._query_abuseipdb(
                ip_address
            )
            intel["sources"]["abuseipdb"] = abuse_result

            if abuse_result.get("success"):
                # Translate 0-100 to 0.0-1.0
                abuse_score = (
                    abuse_result["abuse_confidence"]
                    / 100.0
                )
                intel["risk_score"] = max(
                    intel["risk_score"], abuse_score
                )

                if abuse_result.get("is_tor"):
                    intel["tags"].append("tor_exit_node")
                    intel["threat_type"] = "tor_exit_node"
                    # Tor exit nodes get elevated score
                    intel["risk_score"] = max(
                        intel["risk_score"], 0.75
                    )

                if abuse_result.get("total_reports", 0) > 10:
                    intel["tags"].append(
                        "community_reported"
                    )

        # ---- QUERY FEODO TRACKER ----
        feodo_result = self._check_feodo_tracker(
            ip_address
        )
        intel["sources"]["feodo"] = feodo_result

        if feodo_result.get("is_c2"):
            # Confirmed C2 infrastructure
            # Override risk to near maximum
            intel["risk_score"] = max(
                intel["risk_score"], 0.95
            )
            intel["is_malicious"] = True
            intel["malware_family"] = (
                feodo_result.get("malware", "unknown")
            )
            intel["threat_type"] = "c2_infrastructure"
            intel["tags"].append("confirmed_c2")
            intel["tags"].append(
                f"malware:{intel['malware_family']}"
            )

            logger.warning(
                f"CONFIRMED C2: {ip_address} "
                f"({intel['malware_family']})"
            )

        # Set is_malicious based on final score
        if intel["risk_score"] >= 0.5:
            intel["is_malicious"] = True

        # Cache result
        self.ip_cache[ip_address] = intel
        self.enrichments_applied += 1

        logger.info(
            f"IP checked: {ip_address} → "
            f"risk={intel['risk_score']:.2f} "
            f"malicious={intel['is_malicious']}"
        )

        return intel

    def check_domain(
        self,
        domain: str
    ) -> dict:
        """
        Check a domain against URLhaus and other feeds.

        URLhaus tracks malware distribution URLs.
        If a domain is actively distributing malware
        it appears in URLhaus.

        Args:
            domain: Domain name to check

        Returns:
            Intelligence dictionary
        """
        if domain in self.domain_cache:
            cached = self.domain_cache[domain]
            if self._is_cache_valid(
                cached.get("cached_at"),
                CACHE_TTL_SECONDS["urlhaus"]
            ):
                self.cache_hits += 1
                return cached

        intel = {
            "domain": domain,
            "risk_score": 0.0,
            "is_malicious": False,
            "malware_family": None,
            "threat_type": None,
            "sources": {},
            "tags": [],
            "cached_at": self._now()
        }

        # ---- QUERY URLHAUS ----
        urlhaus_result = self._query_urlhaus_domain(
            domain
        )
        intel["sources"]["urlhaus"] = urlhaus_result

        if urlhaus_result.get("found"):
            intel["risk_score"] = max(
                intel["risk_score"], 0.90
            )
            intel["is_malicious"] = True
            intel["threat_type"] = "malware_distribution"
            intel["tags"].append("urlhaus_confirmed")

            if urlhaus_result.get("malware"):
                intel["malware_family"] = (
                    urlhaus_result["malware"]
                )
                intel["tags"].append(
                    f"malware:{intel['malware_family']}"
                )

        if intel["risk_score"] >= 0.5:
            intel["is_malicious"] = True

        self.domain_cache[domain] = intel

        return intel

    def check_url(
        self,
        url: str
    ) -> dict:
        """
        Check a URL against URLhaus directly.

        Used by phishing detection module to
        verify if a URL is actively malicious.

        Args:
            url: Full URL to check

        Returns:
            Intelligence dictionary
        """
        if url in self.url_cache:
            return self.url_cache[url]

        intel = {
            "url": url,
            "risk_score": 0.0,
            "is_malicious": False,
            "malware_family": None,
            "sources": {},
            "cached_at": self._now()
        }

        urlhaus_result = self._query_urlhaus_url(url)
        intel["sources"]["urlhaus"] = urlhaus_result

        if urlhaus_result.get("found"):
            intel["risk_score"] = 0.95
            intel["is_malicious"] = True
            intel["malware_family"] = (
                urlhaus_result.get("malware")
            )

        self.url_cache[url] = intel
        return intel

    def enrich_knowledge_graph(
        self,
        graph
    ) -> dict:
        """
        Enrich all IP and domain nodes in the graph
        with live threat intelligence.

        This replaces the static dictionary lookup
        in ThreatEnricher with live feed queries.

        Args:
            graph: SecurityKnowledgeGraph instance

        Returns:
            Enrichment summary
        """
        from layer3_knowledge.graph.security_graph import (
            NodeType
        )

        results = {
            "ips_checked": 0,
            "ips_enriched": 0,
            "domains_checked": 0,
            "domains_enriched": 0,
            "c2_confirmed": 0,
            "malware_families": []
        }

        # ---- ENRICH IPs ----
        ip_nodes = graph.get_nodes_by_type(
            NodeType.IP_ADDRESS
        )

        for node in ip_nodes:
            results["ips_checked"] += 1
            intel = self.check_ip(node.label)

            if intel["is_malicious"]:
                # Update node risk score
                node.update_risk(intel["risk_score"])

                # Add threat context to node
                node.properties.update({
                    "feed_risk_score": (
                        intel["risk_score"]
                    ),
                    "feed_threat_type": (
                        intel["threat_type"]
                    ),
                    "feed_malware_family": (
                        intel["malware_family"]
                    ),
                    "feed_tags": intel["tags"],
                    "feed_sources": list(
                        intel["sources"].keys()
                    ),
                    "feed_enriched_at": self._now()
                })

                results["ips_enriched"] += 1

                if intel.get("malware_family"):
                    results["c2_confirmed"] += 1
                    if (intel["malware_family"]
                            not in
                            results["malware_families"]):
                        results["malware_families"].append(
                            intel["malware_family"]
                        )

                logger.info(
                    f"Graph IP enriched: "
                    f"{node.label} → "
                    f"risk={intel['risk_score']:.2f}"
                )

        # ---- ENRICH DOMAINS ----
        domain_nodes = graph.get_nodes_by_type(
            NodeType.DOMAIN
        )

        for node in domain_nodes:
            results["domains_checked"] += 1
            intel = self.check_domain(node.label)

            if intel["is_malicious"]:
                node.update_risk(intel["risk_score"])
                node.properties.update({
                    "feed_risk_score": (
                        intel["risk_score"]
                    ),
                    "feed_threat_type": (
                        intel["threat_type"]
                    ),
                    "feed_malware_family": (
                        intel["malware_family"]
                    ),
                    "feed_enriched_at": self._now()
                })

                results["domains_enriched"] += 1

        logger.info(
            f"Graph enrichment complete: {results}"
        )

        return results

    def get_statistics(self) -> dict:
        """Return feed manager statistics"""
        return {
            "api_calls_made": self.api_calls_made,
            "cache_hits": self.cache_hits,
            "enrichments_applied": (
                self.enrichments_applied
            ),
            "ips_cached": len(self.ip_cache),
            "domains_cached": len(self.domain_cache),
            "abuseipdb_configured": bool(
                self.abuseipdb_key
            ),
            "feodo_loaded": (
                self._feodo_blocklist is not None
            )
        }

    # ============================================================
    # PRIVATE FEED QUERY METHODS
    # ============================================================

    def _query_abuseipdb(
        self,
        ip_address: str
    ) -> dict:
        """
        Query AbuseIPDB API for IP reputation.

        Free tier: 1000 requests per day
        Get API key: https://www.abuseipdb.com/register

        Returns:
            abuse_confidence: 0-100 confidence score
            total_reports: community report count
            is_tor: whether IP is Tor exit node
            country: originating country
        """
        try:
            import requests

            self.api_calls_made += 1
            last_call_time = getattr(
                self, "_last_abuseipdb_call", 0
            )
            elapsed = time.time() - last_call_time
            min_interval = RATE_LIMITS["abuseipdb"][
                "min_seconds_between"
            ]

            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            response = requests.get(
                "https://api.abuseipdb.com/api/v2/check",
                headers={
                    "Key": self.abuseipdb_key,
                    "Accept": "application/json"
                },
                params={
                    "ipAddress": ip_address,
                    "maxAgeInDays": 90,
                    "verbose": True
                },
                timeout=10
            )

            self._last_abuseipdb_call = time.time()

            if response.status_code == 200:
                data = response.json().get("data", {})
                return {
                    "success": True,
                    "abuse_confidence": data.get(
                        "abuseConfidenceScore", 0
                    ),
                    "total_reports": data.get(
                        "totalReports", 0
                    ),
                    "is_tor": data.get("isTor", False),
                    "country": data.get(
                        "countryCode", ""
                    ),
                    "isp": data.get("isp", ""),
                    "last_reported": data.get(
                        "lastReportedAt", ""
                    )
                }
            elif response.status_code == 401:
                logger.warning(
                    "AbuseIPDB: Invalid API key"
                )
                return {
                    "success": False,
                    "error": "Invalid API key"
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                }

        except ImportError:
            logger.warning(
                "requests library not available"
            )
            return {
                "success": False,
                "error": "requests not installed"
            }
        except Exception as e:
            logger.warning(
                f"AbuseIPDB query failed: {e}"
            )
            return {
                "success": False,
                "error": str(e)
            }

    def _check_feodo_tracker(
        self,
        ip_address: str
    ) -> dict:
        """
        Check Feodo Tracker for C2 infrastructure.

        Feodo Tracker tracks Command and Control
        servers for Emotet, TrickBot, Dridex, QakBot.
        No API key required — downloads JSON blocklist.

        Updated every few minutes by Abuse.ch.
        This is real-time C2 intelligence.

        Returns:
            is_c2: whether IP is confirmed C2
            malware: malware family name
            first_seen: when first observed
            last_online: when last seen active
            port: C2 port number
        """
        try:
            # Refresh blocklist if stale
            if not self._is_feodo_current():
                self._load_feodo_blocklist()

            if not self._feodo_blocklist:
                return {"is_c2": False, "error": "not_loaded"}

            # Check IP against blocklist
            for entry in self._feodo_blocklist:
                if entry.get("ip_address") == ip_address:
                    return {
                        "is_c2": True,
                        "malware": entry.get(
                            "malware", "unknown"
                        ),
                        "first_seen": entry.get(
                            "first_seen", ""
                        ),
                        "last_online": entry.get(
                            "last_online", ""
                        ),
                        "port": entry.get("port", 0),
                        "status": entry.get("status", "")
                    }

            return {"is_c2": False}

        except Exception as e:
            logger.warning(
                f"Feodo check failed for {ip_address}: {e}"
            )
            return {"is_c2": False, "error": str(e)}

    def _load_feodo_blocklist(self) -> None:
        """
        Download Feodo Tracker JSON blocklist.

        Cached locally to avoid repeated downloads.
        Refreshed every hour automatically.
        """
        try:
            import requests

            logger.info("Loading Feodo Tracker blocklist")
            self.api_calls_made += 1

            response = requests.get(
                "https://feodotracker.abuse.ch/"
                "downloads/ipblocklist.json",
                timeout=15
            )

            if response.status_code == 200:
                self._feodo_blocklist = response.json()
                self._feodo_loaded_at = time.time()

                logger.info(
                    f"Feodo blocklist loaded: "
                    f"{len(self._feodo_blocklist)} entries"
                )
            else:
                logger.warning(
                    f"Feodo download failed: "
                    f"HTTP {response.status_code}"
                )

        except Exception as e:
            logger.warning(
                f"Failed to load Feodo blocklist: {e}"
            )

    def _query_urlhaus_domain(
        self,
        domain: str
    ) -> dict:
        """
        Query URLhaus for malicious domains.

        URLhaus tracks URLs used for malware
        distribution. If a domain is actively
        serving malware it appears here.

        No API key required.
        API endpoint: https://urlhaus-api.abuse.ch/v1/
        """
        try:
            import requests

            self.api_calls_made += 1

            response = requests.post(
                "https://urlhaus-api.abuse.ch/v1/host/",
                data={"host": domain},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("query_status") == "is_host":
                    urls = data.get("urls", [])
                    malware_types = list(set(
                        url.get("tags", [None])[0]
                        for url in urls
                        if url.get("tags")
                    ))

                    return {
                        "found": True,
                        "url_count": len(urls),
                        "malware": (
                            malware_types[0]
                            if malware_types else None
                        ),
                        "first_seen": data.get(
                            "first_seen", ""
                        )
                    }

            return {"found": False}

        except Exception as e:
            logger.warning(
                f"URLhaus domain query failed: {e}"
            )
            return {"found": False, "error": str(e)}

    def _query_urlhaus_url(self, url: str) -> dict:
        """Query URLhaus for specific URL"""
        try:
            import requests

            self.api_calls_made += 1

            response = requests.post(
                "https://urlhaus-api.abuse.ch/v1/url/",
                data={"url": url},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("query_status") == "is_host":
                    return {
                        "found": True,
                        "malware": data.get(
                            "tags", [None]
                        )[0]
                    }

            return {"found": False}

        except Exception as e:
            return {"found": False, "error": str(e)}

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    def _is_feodo_current(self) -> bool:
        """Check if Feodo blocklist needs refreshing"""
        if self._feodo_blocklist is None:
            return False
        if self._feodo_loaded_at is None:
            return False
        age = time.time() - self._feodo_loaded_at
        return age < CACHE_TTL_SECONDS["feodo"]

    def _is_cache_valid(
        self,
        cached_at: str,
        ttl_seconds: int
    ) -> bool:
        """Check if cached result is still valid"""
        if not cached_at:
            return False
        try:
            cached_time = datetime.fromisoformat(
                cached_at.replace("Z", "+00:00")
            )
            age = (
                datetime.now(timezone.utc) - cached_time
            ).total_seconds()
            return age < ttl_seconds
        except Exception:
            return False

    def _now(self) -> str:
        """Return current UTC timestamp"""
        return datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )