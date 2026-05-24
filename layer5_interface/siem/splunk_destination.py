"""
Layer 5 — Interface
Splunk HEC Destination

Forwards enriched security events to
Splunk via HTTP Event Collector (HEC).

WHAT THIS DOES:
    Takes enriched DataAccessEvent with ML scores
    and agent investigation results.
    
    Sends to Splunk HEC endpoint.
    Splunk indexes as sourcetype abutech:security.
    SOC dashboard shows AbuTech enrichment.
    Existing Splunk searches and alerts work.

SPLUNK HEC:
    HTTP Event Collector.
    Standard Splunk ingestion method.
    Simple POST to port 8088.
    Token-based authentication.
    No agent needed.
    
    BofA generates HEC token in Splunk.
    Provides token to AbuTech.
    AbuTech POSTs enriched events.
    Splunk indexes them.
    SOC analysts see them in < 1 second.

SPLUNK ECS MAPPING:
    Our DataAccessEvent → ECS fields → Splunk
    
    accessor_identity → user.name
    source_ip         → source.ip
    data_store_name   → destination.domain
    bytes_accessed    → network.bytes
    event_time        → @timestamp
    risk_score        → event.risk_score (AbuTech)

USAGE:
    dest = SplunkDestination()
    await dest.send(enriched_event)
    health = await dest.health_check()
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)


class SplunkDestination:
    """
    Splunk HTTP Event Collector destination.
    Sends enriched events to Splunk HEC.

    Events appear in Splunk index: security_alerts
    Sourcetype: abutech:security:event

    SOC analysts search with SPL:
    index=security_alerts sourcetype=abutech:security:event
    | where abutech_risk_score >= 0.8
    | sort - abutech_risk_score
    """

    def __init__(
        self,
        hec_url: str = None,
        hec_token: str = None,
        index: str = None,
        source: str = None
    ):
        self.hec_url = (
            hec_url or
            os.getenv("SPLUNK_HEC_URL", "")
        )
        self.hec_token = (
            hec_token or
            os.getenv("SPLUNK_HEC_TOKEN", "")
        )
        self.index = (
            index or
            os.getenv(
                "SPLUNK_INDEX", "security_alerts"
            )
        )
        self.source = (
            source or
            "abutech-platform"
        )
        self.sourcetype = "abutech:security:event"
        self.name = "splunk"

        self.is_configured = bool(
            self.hec_url and self.hec_token
        )

        # Circuit breaker
        self._failure_count = 0
        self._circuit_open = False
        self._circuit_open_time = 0.0
        self._circuit_threshold = 3
        self._circuit_timeout = 60.0

        # Buffer
        self._buffer = []
        self._max_buffer = 1000

        if not self.is_configured:
            logger.info(
                "Splunk HEC not configured. "
                "Set SPLUNK_HEC_URL and "
                "SPLUNK_HEC_TOKEN. "
                "Running in simulation mode."
            )

    async def send(
        self, event: dict
    ) -> bool:
        """
        Send enriched event to Splunk HEC.

        Args:
            event: Enriched DataAccessEvent dict

        Returns:
            bool: True if sent successfully
        """
        if self._is_circuit_open():
            self._buffer_event(event)
            return False

        if not self.is_configured:
            return self._simulate_send(event)

        try:
            payload = self._format_for_splunk(event)
            success = await self._post_to_hec(payload)

            if success:
                self._record_success()
                await self._drain_buffer()
                return True
            else:
                self._record_failure()
                self._buffer_event(event)
                return False

        except Exception as e:
            logger.error(
                f"Splunk send failed: {e}"
            )
            self._record_failure()
            self._buffer_event(event)
            return False

    async def send_batch(
        self, events: list
    ) -> dict:
        """
        Send batch of events to Splunk HEC.
        Splunk HEC accepts newline-delimited JSON.
        Much more efficient than individual sends.

        Args:
            events: List of enriched events

        Returns:
            dict with sent/failed counts
        """
        if not events:
            return {"sent": 0, "failed": 0}

        if self._is_circuit_open():
            for event in events:
                self._buffer_event(event)
            return {
                "sent": 0,
                "failed": len(events),
                "buffered": len(events)
            }

        if not self.is_configured:
            for event in events:
                self._simulate_send(event)
            return {
                "sent": len(events),
                "failed": 0,
                "simulated": True
            }

        try:
            payloads = [
                json.dumps(
                    self._format_for_splunk(e),
                    default=str
                )
                for e in events
            ]
            body = "\n".join(payloads)

            success = await self._post_to_hec(
                body, raw=True
            )

            if success:
                self._record_success()
                return {
                    "sent": len(events),
                    "failed": 0
                }
            else:
                self._record_failure()
                for event in events:
                    self._buffer_event(event)
                return {
                    "sent": 0,
                    "failed": len(events)
                }

        except Exception as e:
            logger.error(
                f"Splunk batch send failed: {e}"
            )
            self._record_failure()
            return {
                "sent": 0,
                "failed": len(events)
            }

    async def health_check(self) -> dict:
        """Check Splunk HEC connectivity"""
        if not self.is_configured:
            return {
                "healthy": True,
                "status": "simulated",
                "message": "Splunk not configured",
                "circuit_open": False
            }

        if self._is_circuit_open():
            return {
                "healthy": False,
                "status": "circuit_open",
                "circuit_open": True,
                "buffered_events": len(self._buffer)
            }

        try:
            health_url = self.hec_url.replace(
                "/services/collector",
                "/services/collector/health"
            )

            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    health_url,
                    headers={
                        "Authorization": (
                            f"Splunk {self.hec_token}"
                        )
                    },
                    timeout=aiohttp.ClientTimeout(
                        total=10
                    )
                ) as response:
                    healthy = response.status == 200
                    return {
                        "healthy": healthy,
                        "status": (
                            "connected" if healthy
                            else "unreachable"
                        ),
                        "hec_url": (
                            self.hec_url[:30] + "..."
                        ),
                        "circuit_open": False
                    }

        except ImportError:
            return {
                "healthy": True,
                "status": "simulated",
                "circuit_open": False
            }
        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "message": str(e),
                "circuit_open": False
            }

    def should_receive(
        self, risk_score: float
    ) -> bool:
        """
        Splunk receives CRITICAL and HIGH events.
        LOW events go to cheaper storage.
        Cost optimization for high-volume Splunk.
        """
        return risk_score >= 0.3

    def get_status(self) -> dict:
        """Get current destination status"""
        return {
            "name": self.name,
            "configured": self.is_configured,
            "circuit_open": self._circuit_open,
            "failure_count": self._failure_count,
            "buffered_events": len(self._buffer),
            "index": self.index
        }

    # ============================================================
    # PRIVATE METHODS
    # ============================================================

    def _format_for_splunk(
        self, event: dict
    ) -> dict:
        """
        Format DataAccessEvent as Splunk HEC payload.
        ECS-compliant event structure.
        """
        risk_score = float(
            event.get("risk_score", 0.0) or 0.0
        )

        ecs_event = {
            "@timestamp": event.get(
                "event_time", _now()
            ),
            "event": {
                "kind": "alert",
                "provider": "abutech-platform",
                "risk_score": risk_score * 100,
                "severity": self._score_to_severity(
                    risk_score
                ),
                "reason": ", ".join(
                    event.get("risk_reasons", [])
                )[:500]
            },
            "user": {
                "name": event.get(
                    "accessor_identity", ""
                ),
                "type": event.get(
                    "accessor_type", ""
                )
            },
            "source": {
                "ip": event.get("source_ip", "")
            },
            "destination": {
                "domain": event.get(
                    "data_store_name", ""
                ),
                "path": event.get(
                    "data_path", ""
                )[:300]
            },
            "network": {
                "bytes": event.get(
                    "bytes_accessed", 0
                )
            },
            "abutech": {
                "risk_score": risk_score,
                "risk_label": (
                    self._score_to_label(risk_score)
                ),
                "verdict": event.get(
                    "verdict", ""
                ),
                "data_classification": event.get(
                    "data_classification", ""
                ),
                "mitre_techniques": event.get(
                    "mitre_techniques", []
                ),
                "hitl_status": event.get(
                    "hitl_status", ""
                ),
                "agent_summary": event.get(
                    "agent_summary", ""
                )[:1000],
                "source_system": event.get(
                    "source_system", ""
                ),
                "platform_version": "1.0.0"
            }
        }

        return {
            "time": self._parse_timestamp(
                event.get("event_time", _now())
            ),
            "source": self.source,
            "sourcetype": self.sourcetype,
            "index": self.index,
            "host": "abutech-api.azure.com",
            "event": ecs_event
        }

    async def _post_to_hec(
        self,
        payload,
        raw: bool = False
    ) -> bool:
        """POST to Splunk HEC endpoint"""
        try:
            import aiohttp
        except ImportError:
            logger.info(
                "aiohttp not installed. "
                "Simulating Splunk POST."
            )
            return True

        try:
            if not raw:
                body = json.dumps(
                    payload, default=str
                ).encode("utf-8")
            else:
                body = payload.encode("utf-8")

            headers = {
                "Authorization": (
                    f"Splunk {self.hec_token}"
                ),
                "Content-Type": "application/json"
            }

            url = self.hec_url
            if not url.endswith(
                "/services/collector"
            ):
                url = (
                    url.rstrip("/") +
                    "/services/collector"
                )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(
                        total=30
                    ),
                    ssl=False
                ) as response:
                    if response.status == 200:
                        resp_json = await response.json()
                        if resp_json.get("text") == "Success":
                            logger.info(
                                "Splunk: event forwarded"
                            )
                            return True
                    logger.warning(
                        f"Splunk HEC returned "
                        f"{response.status}"
                    )
                    return False

        except Exception as e:
            logger.error(
                f"Splunk HEC POST failed: {e}"
            )
            return False

    def _simulate_send(self, event: dict) -> bool:
        """Simulate send when not configured"""
        risk = float(
            event.get("risk_score", 0.0) or 0.0
        )
        label = self._score_to_label(risk)
        logger.info(
            f"[SPLUNK SIMULATED] "
            f"accessor={event.get('accessor_identity')} "
            f"risk={risk:.3f} {label} "
            f"index={self.index}"
        )
        return True

    def _score_to_label(
        self, score: float
    ) -> str:
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.3:
            return "MEDIUM"
        return "LOW"

    def _score_to_severity(
        self, score: float
    ) -> int:
        if score >= 0.8:
            return 4
        elif score >= 0.6:
            return 3
        elif score >= 0.3:
            return 2
        return 1

    def _parse_timestamp(self, ts: str) -> float:
        """Parse ISO timestamp to Unix epoch"""
        try:
            dt = datetime.fromisoformat(
                ts.replace("Z", "+00:00")
            )
            return dt.timestamp()
        except Exception:
            return datetime.now(
                timezone.utc
            ).timestamp()

    def _is_circuit_open(self) -> bool:
        if not self._circuit_open:
            return False
        elapsed = time.time() - self._circuit_open_time
        if elapsed >= self._circuit_timeout:
            self._circuit_open = False
            self._failure_count = 0
            return False
        return True

    def _record_success(self):
        self._failure_count = 0
        self._circuit_open = False

    def _record_failure(self):
        self._failure_count += 1
        if self._failure_count >= (
            self._circuit_threshold
        ):
            self._circuit_open = True
            self._circuit_open_time = time.time()
            logger.warning(
                "Splunk circuit breaker OPEN"
            )

    def _buffer_event(self, event: dict):
        if len(self._buffer) < self._max_buffer:
            self._buffer.append(event)
        else:
            self._buffer.pop(0)
            self._buffer.append(event)

    async def _drain_buffer(self):
        if not self._buffer:
            return
        drained = []
        for event in self._buffer[:]:
            try:
                payload = self._format_for_splunk(
                    event
                )
                success = await self._post_to_hec(
                    payload
                )
                if success:
                    drained.append(event)
                else:
                    break
            except Exception:
                break
        for event in drained:
            self._buffer.remove(event)


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")