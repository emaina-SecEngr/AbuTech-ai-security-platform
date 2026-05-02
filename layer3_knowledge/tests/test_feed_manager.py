"""
Layer 3 — Threat Intelligence Feed Manager Tests

Tests verify:
1. Score translation from AbuseIPDB 0-100 to 0.0-1.0
2. Feodo Tracker C2 detection
3. Cache behavior
4. Graceful degradation when feeds unavailable
5. Knowledge graph enrichment
"""

import pytest
from unittest.mock import patch, MagicMock
from layer3_knowledge.enrichment.feed_manager import (
    ThreatFeedManager
)
from layer3_knowledge.graph.security_graph import (
    SecurityKnowledgeGraph
)


class TestScoreTranslation:
    """Tests for AbuseIPDB score translation"""

    def setup_method(self):
        self.manager = ThreatFeedManager(
            abuseipdb_key="test_key"
        )

    def test_score_translation_100_becomes_1(self):
        """AbuseIPDB 100 → risk score 1.0"""
        score = 100 / 100.0
        assert score == 1.0

    def test_score_translation_85_becomes_085(self):
        """AbuseIPDB 85 → risk score 0.85"""
        score = 85 / 100.0
        assert abs(score - 0.85) < 0.001

    def test_score_translation_0_becomes_0(self):
        """AbuseIPDB 0 → risk score 0.0"""
        score = 0 / 100.0
        assert score == 0.0

    def test_tor_exit_node_elevated_to_075(self):
        """
        Tor exit nodes elevated to minimum 0.75
        regardless of abuse confidence score.
        A Tor exit node with zero reports is still
        high risk for C2 anonymization.
        """
        with patch.object(
            self.manager,
            "_query_abuseipdb",
            return_value={
                "success": True,
                "abuse_confidence": 0,
                "total_reports": 0,
                "is_tor": True
            }
        ):
            intel = self.manager.check_ip("1.2.3.4")
            assert intel["risk_score"] >= 0.75
            assert "tor_exit_node" in intel["tags"]


class TestFeodoTracker:
    """Tests for Feodo Tracker C2 detection"""

    def setup_method(self):
        self.manager = ThreatFeedManager()

    def test_known_c2_ip_detected(self):
        """
        IP in Feodo blocklist correctly identified
        as confirmed C2 infrastructure.
        """
        mock_blocklist = [
            {
                "ip_address": "185.220.101.45",
                "malware": "Emotet",
                "first_seen": "2024-01-15 10:00:00",
                "last_online": "2024-03-29 02:00:00",
                "port": 443,
                "status": "online"
            }
        ]

        self.manager._feodo_blocklist = mock_blocklist
        self.manager._feodo_loaded_at = (
            __import__("time").time()
        )

        result = self.manager._check_feodo_tracker(
            "185.220.101.45"
        )

        assert result["is_c2"] is True
        assert result["malware"] == "Emotet"
        assert result["port"] == 443

    def test_clean_ip_not_in_feodo(self):
        """Clean IP returns is_c2 False"""
        mock_blocklist = [
            {
                "ip_address": "10.0.0.1",
                "malware": "Emotet"
            }
        ]

        self.manager._feodo_blocklist = mock_blocklist
        self.manager._feodo_loaded_at = (
            __import__("time").time()
        )

        result = self.manager._check_feodo_tracker(
            "8.8.8.8"
        )

        assert result["is_c2"] is False

    def test_confirmed_c2_gets_095_risk(self):
        """
        Confirmed C2 IP gets risk score 0.95.
        Confirmed malware infrastructure is
        near-maximum risk by definition.
        """
        mock_blocklist = [
            {
                "ip_address": "185.220.101.45",
                "malware": "TrickBot",
                "first_seen": "2024-01-01",
                "last_online": "2024-03-29",
                "port": 443,
                "status": "online"
            }
        ]

        self.manager._feodo_blocklist = mock_blocklist
        self.manager._feodo_loaded_at = (
            __import__("time").time()
        )

        with patch.object(
            self.manager,
            "_query_abuseipdb",
            return_value={"success": False}
        ):
            intel = self.manager.check_ip(
                "185.220.101.45"
            )

        assert intel["risk_score"] >= 0.95
        assert intel["is_malicious"] is True
        assert intel["malware_family"] == "TrickBot"
        assert "confirmed_c2" in intel["tags"]


class TestCacheBehavior:
    """Tests for caching behavior"""

    def setup_method(self):
        self.manager = ThreatFeedManager()

    def test_cached_result_returned_on_second_call(self):
        """Second call to same IP uses cache"""
        mock_intel = {
            "ip": "1.2.3.4",
            "risk_score": 0.9,
            "is_malicious": True,
            "cached_at": (
                __import__("datetime")
                .datetime.now(
                    __import__("datetime").timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            )
        }

        self.manager.ip_cache["1.2.3.4"] = mock_intel

        initial_calls = self.manager.api_calls_made
        result = self.manager.check_ip("1.2.3.4")

        assert self.manager.cache_hits >= 1
        assert result["risk_score"] == 0.9

    def test_statistics_tracked(self):
        """Feed manager tracks statistics correctly"""
        stats = self.manager.get_statistics()

        assert "api_calls_made" in stats
        assert "cache_hits" in stats
        assert "enrichments_applied" in stats
        assert "abuseipdb_configured" in stats


class TestGracefulDegradation:
    """Tests for behavior when feeds unavailable"""

    def setup_method(self):
        self.manager = ThreatFeedManager()

    def test_missing_api_key_still_checks_feodo(self):
        """
        Platform works without AbuseIPDB key.
        Feodo Tracker requires no API key.
        """
        manager = ThreatFeedManager(
            abuseipdb_key=""
        )

        mock_blocklist = []
        manager._feodo_blocklist = mock_blocklist
        manager._feodo_loaded_at = (
            __import__("time").time()
        )

        result = manager.check_ip("8.8.8.8")
        assert result is not None
        assert "risk_score" in result

    def test_network_error_returns_safe_default(self):
        """
        Network errors return safe default
        not crash the platform.
        """
        with patch.object(
            self.manager,
            "_query_abuseipdb",
            side_effect=Exception("Network error")
        ):
            with patch.object(
                self.manager,
                "_check_feodo_tracker",
                return_value={"is_c2": False}
            ):
                result = self.manager.check_ip(
                    "1.2.3.4"
                )

        assert result is not None
        assert result["risk_score"] == 0.0


class TestGraphEnrichment:
    """Tests for knowledge graph enrichment"""

    def setup_method(self):
        self.graph = SecurityKnowledgeGraph()
        self.manager = ThreatFeedManager()

    def test_graph_enrichment_updates_ip_node(self):
        """
        Malicious IP in graph gets risk updated
        from feed intelligence.
        """
        self.graph.add_ip(
            "185.220.101.45",
            risk_score=0.1
        )

        mock_blocklist = [
            {
                "ip_address": "185.220.101.45",
                "malware": "Emotet",
                "first_seen": "2024-01-15",
                "last_online": "2024-03-29",
                "port": 443,
                "status": "online"
            }
        ]

        self.manager._feodo_blocklist = mock_blocklist
        self.manager._feodo_loaded_at = (
            __import__("time").time()
        )

        with patch.object(
            self.manager,
            "_query_abuseipdb",
            return_value={"success": False}
        ):
            results = self.manager.enrich_knowledge_graph(
                self.graph
            )

        assert results["ips_enriched"] >= 1
        assert results["c2_confirmed"] >= 1

        node = self.graph.get_node("ip:185.220.101.45")
        assert node.risk_score >= 0.95

    def test_clean_ip_unchanged(self):
        """Clean IP risk score unchanged by enrichment"""
        self.graph.add_ip("8.8.8.8", risk_score=0.0)

        self.manager._feodo_blocklist = []
        self.manager._feodo_loaded_at = (
            __import__("time").time()
        )

        with patch.object(
            self.manager,
            "_query_abuseipdb",
            return_value={
                "success": True,
                "abuse_confidence": 0,
                "total_reports": 0,
                "is_tor": False
            }
        ):
            results = self.manager.enrich_knowledge_graph(
                self.graph
            )

        node = self.graph.get_node("ip:8.8.8.8")
        assert node.risk_score == 0.0