"""
Layer 5 — Interface
SIEM Router

Routes enriched security events to multiple
SIEM destinations simultaneously.

WHAT THE SIEM ROUTER DOES:
    Takes one enriched DataAccessEvent.
    Routes to ALL configured SIEMs in parallel.
    Handles failures gracefully.
    Zero data loss via buffering.
    
    BEFORE (single SIEM):
    Your Platform → Sentinel
    
    AFTER (dual/triple SIEM):
    Your Platform → Sentinel (all events)
                 → Splunk   (HIGH+ events)
                 → QRadar   (CRITICAL events)

THREE ROUTING PATTERNS:

PATTERN 1 — ACTIVE-ACTIVE:
    All SIEMs receive ALL events.
    Maximum redundancy.
    Zero blind spots.
    Higher cost.
    
    BEST FOR: PCI-DSS compliance.
    "All security events must be logged."

PATTERN 2 — TIERED (our default):
    Sentinel: all events (primary).
    Splunk: HIGH+ events (SOC workflow).
    QRadar: CRITICAL only (IBM clients).
    
    Cost optimized.
    Still redundant for critical events.

PATTERN 3 — ACTIVE-PASSIVE:
    Primary SIEM gets all events.
    Secondary only on primary failure.
    Circuit breaker triggers failover.
    
    BEST FOR: Budget-conscious deployments.

CIRCUIT BREAKER PATTERN:
    Each destination has its own circuit breaker.
    3 failures → circuit opens.
    Events buffered to Azure Service Bus.
    After 60s → circuit tries to close.
    If healthy → drain buffer to SIEM.
    Zero data loss guaranteed.

USAGE:
    router = SIEMRouter()
    
    # Route single event
    results = await router.route(enriched_event)
    
    # Route batch
    results = await router.route_batch(events)
    
    # Health check all SIEMs
    health = await router.health_check_all()
    
    # Get routing status
    status = router.get_status()
"""

import asyncio
import logging
import os
from datetime import datetime
from datetime import timezone

from layer5_interface.siem.sentinel_destination\
    import SentinelDestination
from layer5_interface.siem.splunk_destination\
    import SplunkDestination
from layer5_interface.siem.qradar_destination\
    import QRadarDestination

logger = logging.getLogger(__name__)


class SIEMRouter:
    """
    Routes enriched security events to
    multiple SIEM destinations.

    Supports: Microsoft Sentinel, Splunk,
    IBM QRadar, and any future SIEM.

    Handles:
    - Parallel delivery to all configured SIEMs
    - Circuit breaker per destination
    - Zero data loss buffering
    - Health monitoring
    - Tiered routing by risk score
    """

    def __init__(
        self,
        routing_mode: str = "tiered",
        min_risk_for_splunk: float = 0.3,
        min_risk_for_qradar: float = 0.6
    ):
        """
        Initialize SIEM Router.

        Args:
            routing_mode: "active_active", "tiered",
                          or "active_passive"
            min_risk_for_splunk: Min risk score
                for Splunk delivery
            min_risk_for_qradar: Min risk score
                for QRadar delivery
        """
        self.routing_mode = routing_mode
        self.min_risk_for_splunk = (
            min_risk_for_splunk
        )
        self.min_risk_for_qradar = (
            min_risk_for_qradar
        )

        # Initialize destinations
        self.sentinel = SentinelDestination()
        self.splunk = SplunkDestination()
        self.qradar = QRadarDestination()

        # All destinations in priority order
        self.destinations = [
            self.sentinel,
            self.splunk,
            self.qradar
        ]

        # Routing statistics
        self._stats = {
            "total_routed": 0,
            "sentinel_sent": 0,
            "splunk_sent": 0,
            "qradar_sent": 0,
            "total_failed": 0,
            "total_buffered": 0
        }

        logger.info(
            f"SIEM Router initialized: "
            f"mode={routing_mode} "
            f"destinations="
            f"{self._configured_destinations()}"
        )

    async def route(
        self,
        event: dict,
        risk_score: float = None
    ) -> dict:
        """
        Route enriched event to configured SIEMs.
        Runs all deliveries in parallel.

        Args:
            event: Enriched DataAccessEvent dict
            risk_score: Override risk score for routing

        Returns:
            dict with delivery results per SIEM
        """
        if not event:
            return {
                "sentinel": False,
                "splunk": False,
                "qradar": False
            }

        score = risk_score or float(
            event.get("risk_score", 0.0) or 0.0
        )

        self._stats["total_routed"] += 1

        if self.routing_mode == "active_active":
            return await self._route_active_active(
                event
            )
        elif self.routing_mode == "active_passive":
            return await self._route_active_passive(
                event
            )
        else:
            return await self._route_tiered(
                event, score
            )

    async def route_batch(
        self,
        events: list,
        risk_score_override: float = None
    ) -> dict:
        """
        Route batch of events.
        More efficient than individual routing.

        Args:
            events: List of enriched events
            risk_score_override: Apply same score to all

        Returns:
            dict with total sent/failed per SIEM
        """
        if not events:
            return {
                "total": 0,
                "sentinel": {"sent": 0, "failed": 0},
                "splunk": {"sent": 0, "failed": 0},
                "qradar": {"sent": 0, "failed": 0}
            }

        sentinel_events = events
        splunk_events = [
            e for e in events
            if float(
                e.get("risk_score", 0.0) or 0.0
            ) >= self.min_risk_for_splunk
        ]
        qradar_events = [
            e for e in events
            if float(
                e.get("risk_score", 0.0) or 0.0
            ) >= self.min_risk_for_qradar
        ]

        sentinel_task = self.sentinel.send_batch(
            sentinel_events
        )
        splunk_task = self.splunk.send_batch(
            splunk_events
        )
        qradar_task = self.qradar.send_batch(
            qradar_events
        )

        results = await asyncio.gather(
            sentinel_task,
            splunk_task,
            qradar_task,
            return_exceptions=True
        )

        sentinel_result = (
            results[0] if not isinstance(
                results[0], Exception
            ) else {"sent": 0, "failed": len(events)}
        )
        splunk_result = (
            results[1] if not isinstance(
                results[1], Exception
            ) else {"sent": 0, "failed": len(events)}
        )
        qradar_result = (
            results[2] if not isinstance(
                results[2], Exception
            ) else {"sent": 0, "failed": len(events)}
        )

        return {
            "total": len(events),
            "sentinel": sentinel_result,
            "splunk": splunk_result,
            "qradar": qradar_result
        }

    async def health_check_all(self) -> dict:
        """
        Check health of all SIEM destinations.
        Called by FastAPI health endpoint.

        Returns:
            dict with health status per SIEM
        """
        sentinel_health = self.sentinel.health_check()
        splunk_health = self.splunk.health_check()
        qradar_health = self.qradar.health_check()

        results = await asyncio.gather(
            sentinel_health,
            splunk_health,
            qradar_health,
            return_exceptions=True
        )

        sentinel_h = (
            results[0] if not isinstance(
                results[0], Exception
            ) else {"healthy": False, "error": str(results[0])}
        )
        splunk_h = (
            results[1] if not isinstance(
                results[1], Exception
            ) else {"healthy": False, "error": str(results[1])}
        )
        qradar_h = (
            results[2] if not isinstance(
                results[2], Exception
            ) else {"healthy": False, "error": str(results[2])}
        )

        overall_healthy = (
            sentinel_h.get("healthy", False) or
            splunk_h.get("healthy", False)
        )

        return {
            "overall_healthy": overall_healthy,
            "routing_mode": self.routing_mode,
            "sentinel": sentinel_h,
            "splunk": splunk_h,
            "qradar": qradar_h,
            "configured_destinations": (
                self._configured_destinations()
            ),
            "checked_at": _now()
        }

    def get_status(self) -> dict:
        """
        Get current router status.
        Includes statistics and circuit breaker states.
        """
        return {
            "routing_mode": self.routing_mode,
            "configured_destinations": (
                self._configured_destinations()
            ),
            "destinations": {
                "sentinel": self.sentinel.get_status(),
                "splunk": self.splunk.get_status(),
                "qradar": self.qradar.get_status()
            },
            "statistics": self._stats,
            "routing_rules": {
                "sentinel": "all events",
                "splunk": (
                    f"risk >= "
                    f"{self.min_risk_for_splunk}"
                ),
                "qradar": (
                    f"risk >= "
                    f"{self.min_risk_for_qradar}"
                )
            }
        }

    def update_routing_mode(
        self, mode: str
    ) -> dict:
        """
        Update routing mode at runtime.
        No restart needed.

        Args:
            mode: active_active, tiered,
                  or active_passive

        Returns:
            Updated status dict
        """
        valid_modes = [
            "active_active",
            "tiered",
            "active_passive"
        ]
        if mode not in valid_modes:
            return {
                "error": f"Invalid mode: {mode}",
                "valid_modes": valid_modes
            }

        old_mode = self.routing_mode
        self.routing_mode = mode
        logger.info(
            f"SIEM routing mode changed: "
            f"{old_mode} → {mode}"
        )
        return {
            "previous_mode": old_mode,
            "new_mode": mode,
            "updated_at": _now()
        }

    def get_routing_decision(
        self, risk_score: float
    ) -> dict:
        """
        Preview routing decision for a risk score.
        Useful for debugging and demonstrations.

        Args:
            risk_score: Risk score 0.0-1.0

        Returns:
            dict showing which SIEMs would receive
        """
        if self.routing_mode == "active_active":
            return {
                "sentinel": True,
                "splunk": True,
                "qradar": True,
                "mode": "active_active",
                "reason": "All events → all SIEMs"
            }
        elif self.routing_mode == "tiered":
            return {
                "sentinel": True,
                "splunk": (
                    risk_score >= self.min_risk_for_splunk
                ),
                "qradar": (
                    risk_score >= self.min_risk_for_qradar
                ),
                "mode": "tiered",
                "risk_score": risk_score,
                "reason": (
                    f"Sentinel: always | "
                    f"Splunk: >= {self.min_risk_for_splunk} | "
                    f"QRadar: >= {self.min_risk_for_qradar}"
                )
            }
        else:
            sentinel_healthy = not (
                self.sentinel._circuit_open
            )
            return {
                "sentinel": sentinel_healthy,
                "splunk": not sentinel_healthy,
                "qradar": False,
                "mode": "active_passive",
                "reason": (
                    "Primary: Sentinel | "
                    "Failover: Splunk"
                )
            }

    # ============================================================
    # PRIVATE ROUTING METHODS
    # ============================================================

    async def _route_tiered(
        self,
        event: dict,
        risk_score: float
    ) -> dict:
        """
        Tiered routing based on risk score.
        Sentinel: all events.
        Splunk: HIGH+ events.
        QRadar: CRITICAL+ events.
        """
        tasks = {}

        tasks["sentinel"] = self.sentinel.send(event)

        async def _false():
            return False

        if risk_score >= self.min_risk_for_splunk:
            tasks["splunk"] = self.splunk.send(event)
        else:
            tasks["splunk"] = _false()

        if risk_score >= self.min_risk_for_qradar:
            tasks["qradar"] = self.qradar.send(event)
        else:
            tasks["qradar"] = _false()

        results = await asyncio.gather(
            tasks["sentinel"],
            tasks["splunk"],
            tasks["qradar"],
            return_exceptions=True
        )

        sentinel_ok = (
            results[0] is True
            if not isinstance(results[0], Exception)
            else False
        )
        splunk_ok = (
            results[1] is True
            if not isinstance(results[1], Exception)
            else False
        )
        qradar_ok = (
            results[2] is True
            if not isinstance(results[2], Exception)
            else False
        )

        if sentinel_ok:
            self._stats["sentinel_sent"] += 1
        if splunk_ok:
            self._stats["splunk_sent"] += 1
        if qradar_ok:
            self._stats["qradar_sent"] += 1

        return {
            "sentinel": sentinel_ok,
            "splunk": splunk_ok,
            "qradar": qradar_ok,
            "risk_score": risk_score,
            "routing_mode": "tiered"
        }

    async def _route_active_active(
        self, event: dict
    ) -> dict:
        """All SIEMs receive all events"""
        results = await asyncio.gather(
            self.sentinel.send(event),
            self.splunk.send(event),
            self.qradar.send(event),
            return_exceptions=True
        )

        sentinel_ok = results[0] is True
        splunk_ok = results[1] is True
        qradar_ok = results[2] is True

        if sentinel_ok:
            self._stats["sentinel_sent"] += 1
        if splunk_ok:
            self._stats["splunk_sent"] += 1
        if qradar_ok:
            self._stats["qradar_sent"] += 1

        return {
            "sentinel": sentinel_ok,
            "splunk": splunk_ok,
            "qradar": qradar_ok,
            "routing_mode": "active_active"
        }

    async def _route_active_passive(
        self, event: dict
    ) -> dict:
        """
        Active-passive: try Sentinel first.
        If Sentinel fails → Splunk.
        """
        sentinel_ok = await self.sentinel.send(event)

        if sentinel_ok:
            self._stats["sentinel_sent"] += 1
            return {
                "sentinel": True,
                "splunk": False,
                "qradar": False,
                "routing_mode": "active_passive",
                "failover": False
            }

        logger.warning(
            "Sentinel failed. Failing over to Splunk."
        )
        splunk_ok = await self.splunk.send(event)

        if splunk_ok:
            self._stats["splunk_sent"] += 1

        return {
            "sentinel": False,
            "splunk": splunk_ok,
            "qradar": False,
            "routing_mode": "active_passive",
            "failover": True
        }

    def _configured_destinations(self) -> list:
        """Return list of configured destination names"""
        configured = []
        if self.sentinel.is_configured:
            configured.append("sentinel")
        if self.splunk.is_configured:
            configured.append("splunk")
        if self.qradar.is_configured:
            configured.append("qradar")
        if not configured:
            configured = ["simulated"]
        return configured


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")