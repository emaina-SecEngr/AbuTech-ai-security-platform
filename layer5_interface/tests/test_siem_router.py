"""
Tests for SIEM Router and Destinations
"""

import asyncio
import pytest
from layer5_interface.siem.siem_router import (
    SIEMRouter
)
from layer5_interface.siem.sentinel_destination import (
    SentinelDestination
)
from layer5_interface.siem.splunk_destination import (
    SplunkDestination
)
from layer5_interface.siem.qradar_destination import (
    QRadarDestination
)


@pytest.fixture
def router():
    return SIEMRouter()


@pytest.fixture
def sentinel():
    return SentinelDestination()


@pytest.fixture
def splunk():
    return SplunkDestination()


@pytest.fixture
def qradar():
    return QRadarDestination()


@pytest.fixture
def critical_event():
    return {
        "accessor_identity": "svc_backup",
        "accessor_type": "service_account",
        "data_store_name": "prod-customer-data",
        "data_path": "customers/pci/cards.csv",
        "data_classification": "PCI",
        "bytes_accessed": 524288000,
        "event_time": "2026-05-21T03:00:00Z",
        "source_ip": "185.220.101.45",
        "risk_score": 0.974,
        "risk_reasons": [
            "after_hours", "tor_ip",
            "large_volume", "pci_data"
        ],
        "source_system": "s3_normalizer",
        "verdict": "DATA_EXFILTRATION",
        "mitre_techniques": ["T1530", "T1048"],
        "hitl_status": "PENDING",
        "agent_summary": (
            "svc_backup compromised. "
            "Tor exit node confirmed."
        ),
        "investigation_id": "INV-001"
    }


@pytest.fixture
def low_risk_event():
    return {
        "accessor_identity": "john.smith",
        "accessor_type": "human",
        "data_store_name": "dev-logs",
        "data_path": "debug/app.log",
        "data_classification": "INTERNAL",
        "bytes_accessed": 1024,
        "event_time": "2026-05-21T09:00:00Z",
        "source_ip": "10.0.0.1",
        "risk_score": 0.15,
        "risk_reasons": [],
        "source_system": "s3_normalizer"
    }


@pytest.fixture
def batch_events(critical_event, low_risk_event):
    return [critical_event, low_risk_event] * 5


# ============================================================
# SENTINEL DESTINATION TESTS
# ============================================================

class TestSentinelDestination:

    def test_initializes(self, sentinel):
        assert sentinel is not None
        assert sentinel.name == "sentinel"

    def test_not_configured_without_creds(
        self, sentinel
    ):
        assert sentinel.is_configured is False

    def test_should_receive_all_events(
        self, sentinel
    ):
        assert sentinel.should_receive(0.10) is True
        assert sentinel.should_receive(0.50) is True
        assert sentinel.should_receive(0.95) is True

    def test_format_for_sentinel(
        self, sentinel, critical_event
    ):
        formatted = sentinel._format_for_sentinel(
            critical_event
        )
        assert isinstance(formatted, dict)
        assert "abutech_risk_score" in formatted
        assert "abutech_accessor_identity" in formatted
        assert "TimeGenerated" in formatted

    def test_format_risk_score_correct(
        self, sentinel, critical_event
    ):
        formatted = sentinel._format_for_sentinel(
            critical_event
        )
        assert formatted["abutech_risk_score"] == 0.974

    def test_format_risk_label_critical(
        self, sentinel, critical_event
    ):
        formatted = sentinel._format_for_sentinel(
            critical_event
        )
        assert formatted["abutech_risk_label"] == (
            "CRITICAL"
        )

    def test_score_to_label_critical(self, sentinel):
        assert sentinel._score_to_label(0.90) == (
            "CRITICAL"
        )

    def test_score_to_label_high(self, sentinel):
        assert sentinel._score_to_label(0.65) == "HIGH"

    def test_score_to_label_medium(self, sentinel):
        assert sentinel._score_to_label(0.40) == (
            "MEDIUM"
        )

    def test_score_to_label_low(self, sentinel):
        assert sentinel._score_to_label(0.10) == "LOW"

    def test_circuit_breaker_opens(self, sentinel):
        sentinel._circuit_threshold = 2
        sentinel._record_failure()
        sentinel._record_failure()
        assert sentinel._circuit_open is True

    def test_circuit_breaker_resets_on_success(
        self, sentinel
    ):
        sentinel._circuit_open = True
        sentinel._record_success()
        assert sentinel._circuit_open is False
        assert sentinel._failure_count == 0

    def test_buffer_event(self, sentinel, critical_event):
        sentinel._buffer_event(critical_event)
        assert len(sentinel._buffer) == 1

    def test_buffer_max_size(
        self, sentinel, critical_event
    ):
        sentinel._max_buffer = 3
        for _ in range(5):
            sentinel._buffer_event(critical_event)
        assert len(sentinel._buffer) == 3

    def test_get_status_returns_dict(self, sentinel):
        status = sentinel.get_status()
        assert isinstance(status, dict)
        assert "name" in status
        assert "configured" in status
        assert "circuit_open" in status

    @pytest.mark.asyncio
    async def test_send_simulated(
        self, sentinel, critical_event
    ):
        result = await sentinel.send(critical_event)
        assert result is True

    @pytest.mark.asyncio
    async def test_send_batch_simulated(
        self, sentinel, batch_events
    ):
        result = await sentinel.send_batch(batch_events)
        assert isinstance(result, dict)
        assert "sent" in result

    @pytest.mark.asyncio
    async def test_health_check_simulated(self, sentinel):
        health = await sentinel.health_check()
        assert isinstance(health, dict)
        assert "healthy" in health


# ============================================================
# SPLUNK DESTINATION TESTS
# ============================================================

class TestSplunkDestination:

    def test_initializes(self, splunk):
        assert splunk is not None
        assert splunk.name == "splunk"

    def test_not_configured_without_creds(self, splunk):
        assert splunk.is_configured is False

    def test_should_receive_high_risk(self, splunk):
        assert splunk.should_receive(0.80) is True
        assert splunk.should_receive(0.50) is True

    def test_should_not_receive_low_risk(self, splunk):
        assert splunk.should_receive(0.10) is False

    def test_format_for_splunk(
        self, splunk, critical_event
    ):
        formatted = splunk._format_for_splunk(
            critical_event
        )
        assert isinstance(formatted, dict)
        assert "event" in formatted
        assert "sourcetype" in formatted
        assert "index" in formatted

    def test_format_sourcetype(
        self, splunk, critical_event
    ):
        formatted = splunk._format_for_splunk(
            critical_event
        )
        assert formatted["sourcetype"] == (
            "abutech:security:event"
        )

    def test_format_ecs_structure(
        self, splunk, critical_event
    ):
        formatted = splunk._format_for_splunk(
            critical_event
        )
        event = formatted["event"]
        assert "user" in event
        assert "source" in event
        assert "abutech" in event

    def test_format_risk_score_in_event(
        self, splunk, critical_event
    ):
        formatted = splunk._format_for_splunk(
            critical_event
        )
        abutech = formatted["event"]["abutech"]
        assert abutech["risk_score"] == 0.974

    def test_circuit_breaker(self, splunk):
        splunk._circuit_threshold = 2
        splunk._record_failure()
        splunk._record_failure()
        assert splunk._circuit_open is True

    def test_get_status(self, splunk):
        status = splunk.get_status()
        assert "name" in status
        assert "index" in status

    @pytest.mark.asyncio
    async def test_send_simulated(
        self, splunk, critical_event
    ):
        result = await splunk.send(critical_event)
        assert result is True

    @pytest.mark.asyncio
    async def test_send_batch_simulated(
        self, splunk, batch_events
    ):
        result = await splunk.send_batch(batch_events)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_health_check(self, splunk):
        health = await splunk.health_check()
        assert "healthy" in health


# ============================================================
# QRADAR DESTINATION TESTS
# ============================================================

class TestQRadarDestination:

    def test_initializes(self, qradar):
        assert qradar is not None
        assert qradar.name == "qradar"

    def test_not_configured_without_creds(
        self, qradar
    ):
        assert qradar.is_configured is False

    def test_should_receive_critical(self, qradar):
        assert qradar.should_receive(0.90) is True
        assert qradar.should_receive(0.65) is True

    def test_should_not_receive_low(self, qradar):
        assert qradar.should_receive(0.30) is False

    def test_format_for_qradar(
        self, qradar, critical_event
    ):
        formatted = qradar._format_for_qradar(
            critical_event
        )
        assert isinstance(formatted, dict)
        assert "abutech_risk_score" in formatted
        assert "abutech_verdict" in formatted
        assert "abutech_accessor" in formatted

    def test_format_risk_score(
        self, qradar, critical_event
    ):
        formatted = qradar._format_for_qradar(
            critical_event
        )
        assert formatted["abutech_risk_score"] == 0.974

    def test_circuit_breaker(self, qradar):
        qradar._circuit_threshold = 2
        qradar._record_failure()
        qradar._record_failure()
        assert qradar._circuit_open is True

    @pytest.mark.asyncio
    async def test_send_simulated(
        self, qradar, critical_event
    ):
        result = await qradar.send(critical_event)
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check(self, qradar):
        health = await qradar.health_check()
        assert "healthy" in health


# ============================================================
# SIEM ROUTER TESTS
# ============================================================

class TestSIEMRouter:

    def test_initializes(self, router):
        assert router is not None
        assert router.routing_mode == "tiered"

    def test_has_all_destinations(self, router):
        assert router.sentinel is not None
        assert router.splunk is not None
        assert router.qradar is not None

    def test_get_status_returns_dict(self, router):
        status = router.get_status()
        assert isinstance(status, dict)
        assert "routing_mode" in status
        assert "destinations" in status
        assert "statistics" in status

    def test_routing_decision_tiered_critical(
        self, router
    ):
        decision = router.get_routing_decision(0.974)
        assert decision["sentinel"] is True
        assert decision["splunk"] is True
        assert decision["qradar"] is True

    def test_routing_decision_tiered_low(self, router):
        decision = router.get_routing_decision(0.10)
        assert decision["sentinel"] is True
        assert decision["splunk"] is False
        assert decision["qradar"] is False

    def test_routing_decision_active_active(self, router):
        router.routing_mode = "active_active"
        decision = router.get_routing_decision(0.10)
        assert decision["sentinel"] is True
        assert decision["splunk"] is True
        assert decision["qradar"] is True

    def test_update_routing_mode_valid(self, router):
        result = router.update_routing_mode(
            "active_active"
        )
        assert router.routing_mode == "active_active"
        assert "new_mode" in result

    def test_update_routing_mode_invalid(self, router):
        result = router.update_routing_mode(
            "invalid_mode"
        )
        assert "error" in result
        assert router.routing_mode == "tiered"

    def test_statistics_initialized(self, router):
        stats = router._stats
        assert stats["total_routed"] == 0
        assert stats["sentinel_sent"] == 0
        assert stats["splunk_sent"] == 0

    def test_configured_destinations_simulated(
        self, router
    ):
        destinations = router._configured_destinations()
        assert isinstance(destinations, list)
        assert len(destinations) > 0

    @pytest.mark.asyncio
    async def test_route_tiered_critical(
        self, router, critical_event
    ):
        result = await router.route(critical_event)
        assert isinstance(result, dict)
        assert "sentinel" in result
        assert "splunk" in result
        assert "qradar" in result

    @pytest.mark.asyncio
    async def test_route_tiered_low_risk(
        self, router, low_risk_event
    ):
        result = await router.route(low_risk_event)
        assert isinstance(result, dict)
        assert result["sentinel"] is True

    @pytest.mark.asyncio
    async def test_route_active_active(
        self, router, critical_event
    ):
        router.routing_mode = "active_active"
        result = await router.route(critical_event)
        assert result["sentinel"] is True
        assert result["splunk"] is True
        assert result["qradar"] is True

    @pytest.mark.asyncio
    async def test_route_active_passive(
        self, router, critical_event
    ):
        router.routing_mode = "active_passive"
        result = await router.route(critical_event)
        assert isinstance(result, dict)
        assert "failover" in result

    @pytest.mark.asyncio
    async def test_route_batch(
        self, router, batch_events
    ):
        result = await router.route_batch(batch_events)
        assert isinstance(result, dict)
        assert "total" in result
        assert result["total"] == len(batch_events)

    @pytest.mark.asyncio
    async def test_route_empty_event(self, router):
        result = await router.route({})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_health_check_all(self, router):
        health = await router.health_check_all()
        assert isinstance(health, dict)
        assert "overall_healthy" in health
        assert "sentinel" in health
        assert "splunk" in health
        assert "qradar" in health

    @pytest.mark.asyncio
    async def test_statistics_increment(
        self, router, critical_event
    ):
        initial = router._stats["total_routed"]
        await router.route(critical_event)
        assert router._stats["total_routed"] == (
            initial + 1
        )