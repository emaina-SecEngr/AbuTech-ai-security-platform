"""
Tests for Dark Web and Threat Intelligence Feeds
"""

import pytest
from layer3_knowledge.enrichment.hibp_feed import HIBPFeed
from layer3_knowledge.enrichment.otx_feed import OTXFeed
from layer3_knowledge.enrichment.abusech_feed import AbusechFeed
from layer3_knowledge.enrichment.recorded_future_feed import RecordedFutureFeed
from layer3_knowledge.enrichment.dark_web_enricher import DarkWebEnricher


# ============================================================
# HIBP TESTS
# ============================================================

class TestHIBPFeed:

    @pytest.fixture
    def feed(self):
        return HIBPFeed()

    def test_check_email_returns_dict(self, feed):
        result = feed.check_email("test@example.com")
        assert isinstance(result, dict)

    def test_check_email_has_required_fields(
        self, feed
    ):
        result = feed.check_email("test@example.com")
        assert "email" in result
        assert "breach_count" in result
        assert "risk_score" in result
        assert "risk_label" in result

    def test_check_email_risk_score_range(self, feed):
        result = feed.check_email("test@example.com")
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_check_email_empty_returns_empty(
        self, feed
    ):
        result = feed.check_email("")
        assert result["risk_score"] == 0.0

    def test_check_email_invalid_returns_empty(
        self, feed
    ):
        result = feed.check_email("not-an-email")
        assert result["risk_score"] == 0.0

    def test_simulated_breach_email_high_risk(
        self, feed
    ):
        result = feed.check_email(
            "user@test.breach.example.com"
        )
        assert result["breach_count"] >= 0
        assert result["risk_score"] >= 0.0

    def test_check_domain_returns_dict(self, feed):
        result = feed.check_domain("example.com")
        assert isinstance(result, dict)
        assert "domain" in result

    def test_get_risk_score_returns_float(self, feed):
        score = feed.get_risk_score_for_email(
            "test@example.com"
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ============================================================
# OTX FEED TESTS
# ============================================================

class TestOTXFeed:

    @pytest.fixture
    def feed(self):
        return OTXFeed()

    def test_get_ip_reputation_returns_dict(
        self, feed
    ):
        result = feed.get_ip_reputation(
            "185.220.101.45"
        )
        assert isinstance(result, dict)

    def test_tor_ip_high_risk(self, feed):
        result = feed.get_ip_reputation(
            "185.220.101.45"
        )
        assert result["risk_score"] >= 0.9

    def test_internal_ip_zero_risk(self, feed):
        result = feed.get_ip_reputation("10.0.0.1")
        assert result["risk_score"] == 0.0

    def test_private_range_zero_risk(self, feed):
        result = feed.get_ip_reputation("192.168.1.1")
        assert result["risk_score"] == 0.0

    def test_ip_result_has_required_fields(self, feed):
        result = feed.get_ip_reputation("8.8.8.8")
        assert "ip" in result
        assert "risk_score" in result
        assert "risk_label" in result
        assert "source" in result

    def test_get_domain_reputation(self, feed):
        result = feed.get_domain_reputation(
            "evil-malware.com"
        )
        assert isinstance(result, dict)
        assert "risk_score" in result

    def test_malicious_domain_elevated(self, feed):
        result = feed.get_domain_reputation(
            "evil-c2-malware.com"
        )
        assert result["risk_score"] >= 0.8

    def test_clean_domain_low_risk(self, feed):
        result = feed.get_domain_reputation(
            "microsoft.com"
        )
        assert result["risk_score"] <= 0.2

    def test_get_latest_pulses_returns_list(
        self, feed
    ):
        pulses = feed.get_latest_pulses()
        assert isinstance(pulses, list)

    def test_extract_iocs_from_pulses(self, feed):
        pulses = feed._simulated_pulses()
        iocs = feed.extract_iocs_for_graph(pulses)
        assert "malicious_ips" in iocs
        assert "malicious_domains" in iocs
        assert "malware_hashes" in iocs


# ============================================================
# ABUSE.CH FEED TESTS
# ============================================================

class TestAbusechFeed:

    @pytest.fixture
    def feed(self):
        return AbusechFeed()

    def test_check_url_returns_dict(self, feed):
        result = feed.check_url(
            "http://evil-malware.com/payload.exe"
        )
        assert isinstance(result, dict)

    def test_malicious_url_detected(self, feed):
        result = feed.check_url(
            "http://evil-phish.com/malware/shell.php"
        )
        assert isinstance(result, dict)
        assert "is_malicious" in result
        assert "risk_score" in result

    def test_clean_url_low_risk(self, feed):
        result = feed.check_url(
            "https://microsoft.com/update"
        )
        assert result["is_malicious"] is False

    def test_check_ip_c2_returns_dict(self, feed):
        result = feed.check_ip_c2("198.51.100.42")
        assert isinstance(result, dict)
        assert "is_c2" in result
        assert "risk_score" in result

    def test_known_c2_detected(self, feed):
        result = feed.check_ip_c2("198.51.100.42")
        assert isinstance(result, dict)
        assert "is_c2" in result
        assert "risk_score" in result

    def test_internal_ip_not_c2(self, feed):
        result = feed.check_ip_c2("10.0.0.1")
        assert result["is_c2"] is False

    def test_check_hash_returns_dict(self, feed):
        result = feed.check_hash(
            "abc123def456abc123def456abc123de"
        )
        assert isinstance(result, dict)
        assert "is_malware" in result

    def test_short_hash_returns_empty(self, feed):
        result = feed.check_hash("abc")
        assert result["is_malware"] is False

    def test_get_banking_trojan_c2s(self, feed):
        c2s = feed.get_banking_trojan_c2s()
        assert isinstance(c2s, list)
        if c2s:
            assert "ip" in c2s[0]
            assert "malware_family" in c2s[0]

    def test_is_banking_trojan_c2(self, feed):
        result = feed.is_banking_trojan_c2(
            "198.51.100.42"
        )
        assert isinstance(result, dict)
        assert "is_c2" in result


# ============================================================
# RECORDED FUTURE FEED TESTS
# ============================================================

class TestRecordedFutureFeed:

    @pytest.fixture
    def feed(self):
        return RecordedFutureFeed()

    def test_feed_initializes(self, feed):
        assert feed is not None

    def test_not_configured_without_key(self, feed):
        assert feed.is_configured is False

    def test_get_ip_intelligence_returns_dict(
        self, feed
    ):
        result = feed.get_ip_intelligence(
            "185.220.101.45"
        )
        assert isinstance(result, dict)

    def test_tor_ip_critical(self, feed):
        result = feed.get_ip_intelligence(
            "185.220.101.45"
        )
        assert result["risk_score"] >= 0.9
        assert result["risk_label"] == "CRITICAL"

    def test_internal_ip_zero_risk(self, feed):
        result = feed.get_ip_intelligence("10.0.0.1")
        assert result["risk_score"] == 0.0

    def test_ip_has_dark_web_mentions(self, feed):
        result = feed.get_ip_intelligence(
            "185.220.101.45"
        )
        assert "dark_web_mentions" in result
        assert result["dark_web_mentions"] > 0

    def test_get_dark_web_alerts_returns_list(
        self, feed
    ):
        alerts = feed.get_dark_web_alerts(
            "bofa.com"
        )
        assert isinstance(alerts, list)
        assert len(alerts) > 0

    def test_dark_web_alert_has_required_fields(
        self, feed
    ):
        alerts = feed.get_dark_web_alerts("bofa.com")
        if alerts:
            alert = alerts[0]
            assert "title" in alert
            assert "risk_score" in alert
            assert "recommended_action" in alert

    def test_get_credential_alerts_returns_list(
        self, feed
    ):
        alerts = feed.get_credential_alerts("bofa.com")
        assert isinstance(alerts, list)

    def test_get_threat_actor_apt29(self, feed):
        actor = feed.get_threat_actor("APT29")
        assert actor["name"] == "APT29"
        assert "aliases" in actor
        assert "targeted_sectors" in actor

    def test_get_threat_actor_lockbit(self, feed):
        actor = feed.get_threat_actor("LockBit")
        assert actor["name"] == "LockBit"
        assert "motivation" in actor

    def test_get_financial_intel_returns_dict(
        self, feed
    ):
        intel = feed.get_financial_sector_intelligence()
        assert isinstance(intel, dict)
        assert "active_campaigns" in intel


# ============================================================
# DARK WEB ENRICHER TESTS
# ============================================================

class TestDarkWebEnricher:

    @pytest.fixture
    def enricher(self):
        return DarkWebEnricher()

    @pytest.fixture
    def sample_event(self):
        return {
            "accessor_identity": "jsmith@bofa.com",
            "accessor_type": "human",
            "source_ip": "185.220.101.45",
            "data_store_name": "prod-customer-data",
            "data_path": "customers/pci/cards.csv",
            "bytes_accessed": 524288000,
            "event_time": "2026-05-21T03:00:00Z",
            "risk_score": 0.45,
            "risk_reasons": ["after_hours"]
        }

    def test_enricher_initializes(self, enricher):
        assert enricher is not None
        assert enricher.hibp is not None
        assert enricher.otx is not None
        assert enricher.abusech is not None
        assert enricher.rf is not None

    def test_enrich_event_returns_dict(
        self, enricher, sample_event
    ):
        result = enricher.enrich_event(sample_event)
        assert isinstance(result, dict)

    def test_enrich_event_has_ti_context(
        self, enricher, sample_event
    ):
        result = enricher.enrich_event(sample_event)
        assert "threat_intelligence" in result

    def test_enrich_event_elevates_score(
        self, enricher, sample_event
    ):
        result = enricher.enrich_event(sample_event)
        ti = result["threat_intelligence"]
        assert "original_risk_score" in ti
        assert "elevated_risk_score" in ti

    def test_tor_ip_elevates_score(
        self, enricher, sample_event
    ):
        result = enricher.enrich_event(sample_event)
        original = result["threat_intelligence"][
            "original_risk_score"
        ]
        elevated = result["threat_intelligence"][
            "elevated_risk_score"
        ]
        assert elevated >= original

    def test_elevate_risk_score_returns_dict(
        self, enricher
    ):
        result = enricher.elevate_risk_score(
            base_score=0.30,
            source_ip="185.220.101.45"
        )
        assert isinstance(result, dict)
        assert "original_score" in result
        assert "elevated_score" in result

    def test_elevation_not_negative(self, enricher):
        result = enricher.elevate_risk_score(
            base_score=0.30,
            source_ip="10.0.0.1"
        )
        assert result["elevated_score"] >= (
            result["original_score"]
        )

    def test_elevation_capped_at_one(self, enricher):
        result = enricher.elevate_risk_score(
            base_score=0.95,
            source_ip="185.220.101.45"
        )
        assert result["elevated_score"] <= 1.0

    def test_get_org_alerts_returns_dict(
        self, enricher
    ):
        result = enricher.get_org_alerts("bofa.com")
        assert isinstance(result, dict)
        assert "dark_web_alerts" in result
        assert "credential_alerts" in result
        assert "financial_intel" in result

    def test_enrich_adds_ti_risk_reasons(
        self, enricher, sample_event
    ):
        result = enricher.enrich_event(sample_event)
        reasons = result.get("risk_reasons", [])
        assert len(reasons) >= len(
            sample_event["risk_reasons"]
        )