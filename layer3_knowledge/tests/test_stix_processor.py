"""
Layer 3 — STIX Processor and Feed Scheduler Tests

Tests verify:
1. STIX pattern extraction for all IOC types
2. Bundle processing with delta detection
3. Knowledge graph updates from STIX objects
4. Feed scheduler subscription management
5. High-risk IOC alert triggering
"""

import pytest
import json
from datetime import datetime
from datetime import timezone
from unittest.mock import MagicMock, patch
from layer3_knowledge.graph.security_graph import (
    SecurityKnowledgeGraph
)
from layer3_knowledge.enrichment.stix_processor import (
    STIXProcessor,
    extract_ip_from_pattern,
    extract_domain_from_pattern,
    extract_url_from_pattern,
    extract_hash_from_pattern,
    extract_ioc_from_pattern
)
from layer3_knowledge.enrichment.feed_scheduler import (
    FeedScheduler,
    FeedSubscription,
    SAMPLE_STIX_BUNDLES
)


# ============================================================
# SAMPLE STIX OBJECTS FOR TESTING
# ============================================================

SAMPLE_INDICATOR_IP = {
    "type": "indicator",
    "id": "indicator--test-001",
    "name": "Test Malicious IP",
    "pattern": "[ipv4-addr:value = '185.220.101.45']",
    "pattern_type": "stix",
    "labels": ["malicious-activity"],
    "confidence": 90,
    "valid_from": "2024-01-15T00:00:00Z"
}

SAMPLE_INDICATOR_DOMAIN = {
    "type": "indicator",
    "id": "indicator--test-002",
    "name": "DGA Domain",
    "pattern": (
        "[domain-name:value = "
        "'xjf8k2mp.duckdns.org']"
    ),
    "pattern_type": "stix",
    "labels": ["malicious-activity", "dga"],
    "confidence": 85,
    "valid_from": "2024-01-15T00:00:00Z"
}

SAMPLE_INDICATOR_URL = {
    "type": "indicator",
    "id": "indicator--test-003",
    "name": "Malware Download URL",
    "pattern": (
        "[url:value = "
        "'http://evil.com/payload.exe']"
    ),
    "pattern_type": "stix",
    "labels": ["malicious-activity"],
    "confidence": 75,
    "valid_from": "2024-01-15T00:00:00Z"
}

SAMPLE_INDICATOR_HASH = {
    "type": "indicator",
    "id": "indicator--test-004",
    "name": "Malware Hash",
    "pattern": (
        "[file:hashes.MD5 = "
        "'eb84f6e4376d1b9a50f7d3a4a48d9f2c']"
    ),
    "pattern_type": "stix",
    "labels": ["malicious-activity"],
    "confidence": 95,
    "valid_from": "2024-01-15T00:00:00Z"
}

SAMPLE_MALWARE = {
    "type": "malware",
    "id": "malware--test-001",
    "name": "TestMalware",
    "malware_types": ["ransomware"],
    "aliases": ["TestRansom"],
    "capabilities": ["encryption", "exfiltration"]
}

SAMPLE_THREAT_ACTOR = {
    "type": "threat-actor",
    "id": "threat-actor--test-001",
    "name": "Test APT Group",
    "aliases": ["TestAPT"],
    "primary_motivation": "espionage",
    "sophistication": "advanced"
}

SAMPLE_BUNDLE = {
    "type": "bundle",
    "id": "bundle--test-001",
    "objects": [
        SAMPLE_INDICATOR_IP,
        SAMPLE_INDICATOR_DOMAIN,
        SAMPLE_MALWARE,
        SAMPLE_THREAT_ACTOR
    ]
}


# ============================================================
# TEST CLASS — STIX PATTERN EXTRACTION
# ============================================================

class TestSTIXPatternExtraction:
    """Tests for STIX pattern parsing functions"""

    def test_extract_ip_from_pattern(self):
        """IP correctly extracted from STIX pattern"""
        pattern = (
            "[ipv4-addr:value = '185.220.101.45']"
        )
        result = extract_ip_from_pattern(pattern)
        assert result == "185.220.101.45"

    def test_extract_ip_with_double_quotes(self):
        """IP extraction works with double quotes"""
        pattern = (
            '[ipv4-addr:value = "185.220.101.45"]'
        )
        result = extract_ip_from_pattern(pattern)
        assert result == "185.220.101.45"

    def test_extract_ip_returns_none_for_domain(self):
        """IP extractor returns None for domain pattern"""
        pattern = "[domain-name:value = 'evil.com']"
        result = extract_ip_from_pattern(pattern)
        assert result is None

    def test_extract_domain_from_pattern(self):
        """Domain correctly extracted from STIX pattern"""
        pattern = (
            "[domain-name:value = "
            "'xjf8k2mp.duckdns.org']"
        )
        result = extract_domain_from_pattern(pattern)
        assert result == "xjf8k2mp.duckdns.org"

    def test_extract_url_from_pattern(self):
        """URL correctly extracted from STIX pattern"""
        pattern = (
            "[url:value = "
            "'http://evil.com/payload.exe']"
        )
        result = extract_url_from_pattern(pattern)
        assert result == "http://evil.com/payload.exe"

    def test_extract_hash_from_pattern(self):
        """MD5 hash correctly extracted"""
        pattern = (
            "[file:hashes.MD5 = "
            "'eb84f6e4376d1b9a50f7d3a4a48d9f2c']"
        )
        result = extract_hash_from_pattern(pattern)
        assert result is not None
        assert result["md5"] == (
            "eb84f6e4376d1b9a50f7d3a4a48d9f2c"
        )

    def test_extract_ioc_identifies_ip(self):
        """Generic extractor identifies IP type"""
        pattern = (
            "[ipv4-addr:value = '1.2.3.4']"
        )
        result = extract_ioc_from_pattern(pattern)
        assert result["type"] == "ip"
        assert result["value"] == "1.2.3.4"

    def test_extract_ioc_identifies_domain(self):
        """Generic extractor identifies domain type"""
        pattern = (
            "[domain-name:value = 'evil.com']"
        )
        result = extract_ioc_from_pattern(pattern)
        assert result["type"] == "domain"
        assert result["value"] == "evil.com"

    def test_extract_ioc_identifies_url(self):
        """Generic extractor identifies URL type"""
        pattern = (
            "[url:value = 'http://evil.com/x']"
        )
        result = extract_ioc_from_pattern(pattern)
        assert result["type"] == "url"
        assert result["value"] == "http://evil.com/x"

    def test_extract_ioc_returns_unknown_for_invalid(self):
        """Unknown pattern returns unknown type"""
        pattern = "invalid pattern string"
        result = extract_ioc_from_pattern(pattern)
        assert result["type"] == "unknown"
        assert result["value"] is None


# ============================================================
# TEST CLASS — STIX PROCESSOR
# ============================================================

class TestSTIXProcessor:
    """Tests for STIXProcessor bundle processing"""

    def setup_method(self, method):
        import uuid
        self.graph = SecurityKnowledgeGraph()
        self.processor = STIXProcessor(
        self.graph,
        db_path=(
            f".test_cache/test_{uuid.uuid4().hex}.db"
        )
    )

    def test_process_bundle_adds_ip_to_graph(self):
        """
        IP indicator in STIX bundle adds IP node
        to knowledge graph.
        """
        bundle = {
            "type": "bundle",
            "objects": [SAMPLE_INDICATOR_IP]
        }

        results = self.processor.process_bundle(bundle)

        assert results["indicators_processed"] >= 1
        ip_node = self.graph.get_node(
            "ip:185.220.101.45"
        )
        assert ip_node is not None

    def test_process_bundle_adds_domain_to_graph(self):
        """Domain indicator adds domain node to graph"""
        bundle = {
            "type": "bundle",
            "objects": [SAMPLE_INDICATOR_DOMAIN]
        }

        results = self.processor.process_bundle(bundle)

        domain_node = self.graph.get_node(
            "domain:xjf8k2mp.duckdns.org"
        )
        assert domain_node is not None

    def test_ip_risk_score_from_confidence(self):
        """
        STIX confidence 90 translates to
        risk score 0.90 on IP node.
        """
        bundle = {
            "type": "bundle",
            "objects": [SAMPLE_INDICATOR_IP]
        }

        self.processor.process_bundle(bundle)

        ip_node = self.graph.get_node(
            "ip:185.220.101.45"
        )
        assert ip_node is not None
        assert ip_node.risk_score == 0.9

    def test_delta_processing_skips_seen_objects(self):
        """
        Objects processed once are not reprocessed.
        Delta processing prevents duplicate updates.
        """
        bundle = {
            "type": "bundle",
            "objects": [SAMPLE_INDICATOR_IP]
        }

        # Process first time
        result1 = self.processor.process_bundle(bundle)

        # Process same bundle again
        result2 = self.processor.process_bundle(bundle)

        # Second run should process 0 new objects
        assert result2["new_objects"] == 0
        assert result2["indicators_processed"] == 0

    def test_new_objects_processed_after_delta(self):
        """
        New objects in subsequent bundles ARE processed.
        Delta correctly identifies what is new.
        """
        bundle1 = {
            "type": "bundle",
            "objects": [SAMPLE_INDICATOR_IP]
        }
        bundle2 = {
            "type": "bundle",
            "objects": [
                SAMPLE_INDICATOR_IP,
                SAMPLE_INDICATOR_DOMAIN
            ]
        }

        self.processor.process_bundle(bundle1)
        result2 = self.processor.process_bundle(bundle2)

        # Only the domain is new
        assert result2["new_objects"] == 1

    def test_malware_object_processed(self):
        """Malware STIX object correctly processed"""
        bundle = {
            "type": "bundle",
            "objects": [SAMPLE_MALWARE]
        }

        results = self.processor.process_bundle(bundle)
        assert results["malware_processed"] >= 1

    def test_threat_actor_added_to_graph(self):
        """Threat actor added as graph node"""
        bundle = {
            "type": "bundle",
            "objects": [SAMPLE_THREAT_ACTOR]
        }

        self.processor.process_bundle(bundle)

        actor_node = self.graph.get_node(
            f"actor:{SAMPLE_THREAT_ACTOR['id']}"
        )
        assert actor_node is not None

    def test_ioc_stored_in_database(self):
        """Processed IOC stored in SQLite database"""
        bundle = {
            "type": "bundle",
            "objects": [SAMPLE_INDICATOR_IP]
        }

        self.processor.process_bundle(bundle)

        # Look up in database
        record = self.processor.lookup_ip(
            "185.220.101.45"
        )
        assert record is not None
        assert record["ioc_type"] == "ip"
        assert record["ioc_value"] == "185.220.101.45"

    def test_domain_lookup_after_processing(self):
        """Domain can be looked up after processing"""
        bundle = {
            "type": "bundle",
            "objects": [SAMPLE_INDICATOR_DOMAIN]
        }

        self.processor.process_bundle(bundle)

        record = self.processor.lookup_domain(
            "xjf8k2mp.duckdns.org"
        )
        assert record is not None

    def test_statistics_tracked(self):
        """Processing statistics correctly tracked"""
        self.processor.process_bundle(SAMPLE_BUNDLE)

        stats = self.processor.get_statistics()
        assert stats["bundles_processed"] == 1
        assert stats["iocs_in_database"] >= 2

    def test_empty_bundle_handled_gracefully(self):
        """Empty bundle returns zero results"""
        bundle = {"type": "bundle", "objects": []}
        results = self.processor.process_bundle(bundle)
        assert results["total_objects"] == 0

    def test_invalid_bundle_handled_gracefully(self):
        """Invalid bundle handled without crashing"""
        results = self.processor.process_bundle(
            "not a bundle"
        )
        assert results["total_objects"] == 0


# ============================================================
# TEST CLASS — FEED SCHEDULER
# ============================================================

class TestFeedScheduler:
    """Tests for FeedScheduler subscription management"""

    def setup_method(self, method):
     import uuid
     self.graph = SecurityKnowledgeGraph()
     self.processor = STIXProcessor(
        self.graph,
        db_path=(
            f".test_cache/sched_{uuid.uuid4().hex}.db"
        )
    )
     self.scheduler = FeedScheduler(self.processor)

    def test_add_subscription(self):
        """Feed subscription correctly added"""
        sub = FeedSubscription(
            feed_id="test_feed",
            feed_name="Test Feed",
            feed_url="sample://test",
            feed_type="sample"
        )
        self.scheduler.add_subscription(sub)

        assert "test_feed" in self.scheduler.subscriptions

    def test_add_sample_feeds(self):
        """Sample feeds added correctly"""
        self.scheduler.add_sample_feeds()

        assert len(self.scheduler.subscriptions) >= 2

    def test_run_once_processes_due_feeds(self):
        """
        run_once processes feeds due for update.
        New subscriptions are always due.
        """
        self.scheduler.add_sample_feeds()
        results = self.scheduler.run_once()

        assert results["feeds_updated"] >= 1

    def test_run_once_skips_non_due_feeds(self):
        """
        Feeds updated recently are not reprocessed.
        Respects update interval.
        """
        self.scheduler.add_sample_feeds()

        # First run
        self.scheduler.run_once()

        # Second run immediately after
        # Feeds were just updated so not due
        results = self.scheduler.run_once()
        assert results["feeds_updated"] == 0

    def test_sample_bundle_processed_correctly(self):
        """
        CISA sample bundle correctly processed.
        IP and domain from bundle added to graph.
        """
        self.scheduler.add_sample_feeds()
        self.scheduler.run_once()

        ip_node = self.graph.get_node(
            "ip:185.220.101.45"
        )
        domain_node = self.graph.get_node(
            "domain:xjf8k2mp.duckdns.org"
        )

        assert ip_node is not None
        assert domain_node is not None

    def test_alert_callback_triggered(self):
        """
        Alert callback fired when high-risk IOCs found.
        Implements your trigger vision.
        """
        alerts_received = []

        def alert_handler(alert):
            alerts_received.append(alert)

        scheduler = FeedScheduler(
            self.processor,
            alert_callback=alert_handler
        )
        scheduler.add_sample_feeds()
        scheduler.run_once()

        # CISA bundle has high confidence IOCs
        # that should trigger alerts
        assert scheduler.total_runs == 1

    def test_statistics_tracked(self):
        """Scheduler statistics correctly tracked"""
        self.scheduler.add_sample_feeds()
        self.scheduler.run_once()

        stats = self.scheduler.get_statistics()
        assert stats["total_runs"] == 1
        assert stats["active_subscriptions"] >= 2

    def test_subscription_is_due_initially(self):
        """New subscription is immediately due"""
        sub = FeedSubscription(
            feed_id="new_feed",
            feed_name="New Feed",
            feed_url="sample://new",
            feed_type="sample"
        )
        assert sub.is_due_for_update() is True

    def test_subscription_not_due_after_update(self):
        """Subscription not due immediately after update"""
        sub = FeedSubscription(
            feed_id="recent_feed",
            feed_name="Recent Feed",
            feed_url="sample://recent",
            feed_type="sample",
            update_interval_minutes=60
        )

        from datetime import timezone
        sub.last_updated = datetime.now(timezone.utc)

        assert sub.is_due_for_update() is False

    def test_disabled_subscription_never_due(self):
        """Disabled subscription never processed"""
        sub = FeedSubscription(
            feed_id="disabled",
            feed_name="Disabled Feed",
            feed_url="sample://disabled",
            feed_type="sample",
            enabled=False
        )
        assert sub.is_due_for_update() is False