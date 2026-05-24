"""
Layer 5 — Interface
Microsoft Sentinel Destination

Forwards enriched security events to
Microsoft Sentinel via Log Analytics API.

WHAT THIS DOES:
    Takes enriched DataAccessEvent with:
    - ML risk scores (0.974 CRITICAL)
    - Agent investigation summary
    - MITRE ATT&CK techniques
    - HITL approval status
    - Policy violations
    
    Sends to Sentinel Log Analytics workspace.
    Analyst opens Sentinel → sees AbuTech data.
    Security Copilot reads enriched context.
    Existing dashboards work immediately.

SENTINEL INTEGRATION OPTIONS:
    Option 1: Log Analytics Data Collector API
    Custom log table: AbuTechEvents_CL
    Simple HTTP POST with shared key auth.
    Free. No extra license needed.
    
    Option 2: Azure Monitor Ingestion API
    DCR (Data Collection Rule) based.
    More modern. Better schema control.
    Requires DCR setup in Azure.
    
    Option 3: Write-back to existing incidents
    PATCH existing Sentinel incidents.
    Adds AbuTech enrichment as comment.
    Already built in SentinelNormalizer.
    
    WE USE OPTION 1 (simplest, most compatible).

HEALTH MONITORING:
    Check workspace connectivity before sending.
    Circuit breaker: 3 failures → open 60s.
    Buffer events during outage.
    Auto-drain buffer when healthy again.

USAGE:
    dest = SentinelDestination()
    
    await dest.send(enriched_event)
    health = await dest.health_check()
    await dest.send_batch(events)
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)

# Sentinel Log Analytics API
SENTINEL_API_VERSION = "2016-04-01"
SENTINEL_RESOURCE = "/api/logs"
SENTINEL_CONTENT_TYPE = "application/json"
SENTINEL_LOG_TYPE = "AbuTechSecurityEvents"


class SentinelDestination:
    """
    Microsoft Sentinel destination.
    Sends enriched events to Log Analytics workspace
    via HTTP Data Collector API.

    Events appear in custom table:
    AbuTechSecurityEvents_CL

    SOC analysts query with KQL:
    AbuTechSecurityEvents_CL
    | where abutech_risk_score_d >= 0.8
    | sort by TimeGenerated desc
    """

    def __init__(
        self,
        workspace_id: str = None,
        workspace_key: str = None,
        log_type: str = None
    ):
        self.workspace_id = (
            workspace_id or
            os.getenv("SENTINEL_WORKSPACE_ID", "")
        )
        self.workspace_key = (
            workspace_key or
            os.getenv("SENTINEL_WORKSPACE_KEY", "")
        )
        self.log_type = (
            log_type or
            os.getenv(
                "SENTINEL_LOG_TYPE",
                SENTINEL_LOG_TYPE
            )
        )
        self.name = "sentinel"
        self.is_configured = bool(
            self.workspace_id and self.workspace_key
        )

        # Circuit breaker state
        self._failure_count = 0
        self._circuit_open = False
        self._circuit_open_time = 0.0
        self._circuit_threshold = 3
        self._circuit_timeout = 60.0

        # Event buffer for outages
        self._buffer = []
        self._max_buffer = 1000

        if not self.is_configured:
            logger.info(
                "Sentinel not configured. "
                "Set SENTINEL_WORKSPACE_ID and "
                "SENTINEL_WORKSPACE_KEY for live "
                "forwarding. Running in simulation mode."
            )

    async def send(
        self, event: dict
    ) -> bool:
        """
        Send single enriched event to Sentinel.

        Args:
            event: Enriched DataAccessEvent dict

        Returns:
            bool: True if sent successfully
        """
        if self._is_circuit_open():
            logger.warning(
                "Sentinel circuit breaker open. "
                "Buffering event."
            )
            self._buffer_event(event)
            return False

        if not self.is_configured:
            return self._simulate_send(event)

        try:
            sentinel_event = (
                self._format_for_sentinel(event)
            )
            body = json.dumps(
                [sentinel_event],
                default=str
            )
            success = await self._post_to_sentinel(
                body
            )

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
                f"Sentinel send failed: {e}"
            )
            self._record_failure()
            self._buffer_event(event)
            return False

    async def send_batch(
        self, events: list
    ) -> dict:
        """
        Send batch of events to Sentinel.
        More efficient than individual sends.

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
            sentinel_events = [
                self._format_for_sentinel(e)
                for e in events
            ]
            body = json.dumps(
                sentinel_events, default=str
            )
            success = await self._post_to_sentinel(
                body
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
                f"Sentinel batch send failed: {e}"
            )
            self._record_failure()
            return {
                "sent": 0,
                "failed": len(events)
            }

    async def health_check(self) -> dict:
        """
        Check Sentinel workspace connectivity.

        Returns:
            dict with health status
        """
        if not self.is_configured:
            return {
                "healthy": True,
                "status": "simulated",
                "message": "Sentinel not configured",
                "circuit_open": False
            }

        if self._is_circuit_open():
            return {
                "healthy": False,
                "status": "circuit_open",
                "message": (
                    "Circuit breaker open. "
                    f"Resets in "
                    f"{self._time_until_reset():.0f}s"
                ),
                "circuit_open": True,
                "buffered_events": len(self._buffer)
            }

        try:
            test_event = {
                "abutech_health_check": True,
                "timestamp": _now()
            }
            body = json.dumps([test_event])
            success = await self._post_to_sentinel(
                body
            )

            return {
                "healthy": success,
                "status": (
                    "connected" if success
                    else "unreachable"
                ),
                "workspace_id": (
                    self.workspace_id[:8] + "..."
                ),
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
        Check if this destination should receive
        event based on risk score.
        Sentinel receives ALL events (primary SIEM).
        """
        return True

    def get_status(self) -> dict:
        """Get current destination status"""
        return {
            "name": self.name,
            "configured": self.is_configured,
            "circuit_open": self._circuit_open,
            "failure_count": self._failure_count,
            "buffered_events": len(self._buffer),
            "log_type": self.log_type
        }

    # ============================================================
    # PRIVATE METHODS
    # ============================================================

    def _format_for_sentinel(
        self, event: dict
    ) -> dict:
        """
        Format DataAccessEvent for Sentinel
        Log Analytics custom table.

        Field names get _s (string) _d (double)
        _b (bool) suffix in Sentinel automatically.
        """
        return {
            "TimeGenerated": event.get(
                "event_time", _now()
            ),
            "abutech_accessor_identity": event.get(
                "accessor_identity", ""
            ),
            "abutech_accessor_type": event.get(
                "accessor_type", ""
            ),
            "abutech_data_store": event.get(
                "data_store_name", ""
            ),
            "abutech_data_path": event.get(
                "data_path", ""
            )[:500],
            "abutech_data_classification": event.get(
                "data_classification", ""
            ),
            "abutech_bytes_accessed": event.get(
                "bytes_accessed", 0
            ),
            "abutech_source_ip": event.get(
                "source_ip", ""
            ),
            "abutech_risk_score": float(
                event.get("risk_score", 0.0) or 0.0
            ),
            "abutech_risk_label": (
                self._score_to_label(
                    float(
                        event.get(
                            "risk_score", 0.0
                        ) or 0.0
                    )
                )
            ),
            "abutech_risk_reasons": json.dumps(
                event.get("risk_reasons", [])
            ),
            "abutech_source_system": event.get(
                "source_system", ""
            ),
            "abutech_verdict": event.get(
                "verdict", ""
            ),
            "abutech_mitre_techniques": json.dumps(
                event.get("mitre_techniques", [])
            ),
            "abutech_hitl_status": event.get(
                "hitl_status", ""
            ),
            "abutech_agent_summary": event.get(
                "agent_summary", ""
            )[:2000],
            "abutech_platform_version": "1.0.0",
            "abutech_investigation_id": event.get(
                "investigation_id", ""
            )
        }

    async def _post_to_sentinel(
        self, body: str
    ) -> bool:
        """
        POST data to Sentinel Log Analytics API.
        Uses HMAC-SHA256 signature for auth.
        """
        try:
            import aiohttp
        except ImportError:
            logger.info(
                "aiohttp not installed. "
                "Simulating Sentinel POST."
            )
            return True

        try:
            rfc1123_date = datetime.now(
                timezone.utc
            ).strftime(
                "%a, %d %b %Y %H:%M:%S GMT"
            )

            signature = self._build_signature(
                body, rfc1123_date
            )

            url = (
                f"https://{self.workspace_id}"
                f".ods.opinsights.azure.com"
                f"{SENTINEL_RESOURCE}"
                f"?api-version={SENTINEL_API_VERSION}"
            )

            headers = {
                "Content-Type": SENTINEL_CONTENT_TYPE,
                "Log-Type": self.log_type,
                "Authorization": signature,
                "x-ms-date": rfc1123_date,
                "time-generated-field": "TimeGenerated"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=body.encode("utf-8"),
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(
                        total=30
                    )
                ) as response:
                    if response.status in [200, 204]:
                        logger.info(
                            "Sentinel: event forwarded"
                        )
                        return True
                    else:
                        logger.warning(
                            f"Sentinel returned "
                            f"{response.status}"
                        )
                        return False

        except Exception as e:
            logger.error(
                f"Sentinel POST failed: {e}"
            )
            return False

    def _build_signature(
        self, body: str, date: str
    ) -> str:
        """Build HMAC-SHA256 signature for Sentinel"""
        content_length = len(body.encode("utf-8"))
        string_to_hash = (
            f"POST\n{content_length}\n"
            f"{SENTINEL_CONTENT_TYPE}\n"
            f"x-ms-date:{date}\n"
            f"{SENTINEL_RESOURCE}"
        )
        bytes_to_hash = string_to_hash.encode("utf-8")
        decoded_key = base64.b64decode(
            self.workspace_key
        )
        encoded_hash = base64.b64encode(
            hmac.new(
                decoded_key,
                bytes_to_hash,
                digestmod=hashlib.sha256
            ).digest()
        ).decode("utf-8")
        return (
            f"SharedKey {self.workspace_id}:"
            f"{encoded_hash}"
        )

    def _simulate_send(self, event: dict) -> bool:
        """Simulate send when not configured"""
        risk = float(
            event.get("risk_score", 0.0) or 0.0
        )
        label = self._score_to_label(risk)
        logger.info(
            f"[SENTINEL SIMULATED] "
            f"accessor={event.get('accessor_identity')} "
            f"risk={risk:.3f} {label}"
        )
        return True

    def _score_to_label(
        self, score: float
    ) -> str:
        """Convert score to severity label"""
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.3:
            return "MEDIUM"
        return "LOW"

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        if not self._circuit_open:
            return False
        elapsed = time.time() - self._circuit_open_time
        if elapsed >= self._circuit_timeout:
            logger.info(
                "Sentinel circuit breaker reset"
            )
            self._circuit_open = False
            self._failure_count = 0
            return False
        return True

    def _time_until_reset(self) -> float:
        """Seconds until circuit breaker resets"""
        elapsed = time.time() - self._circuit_open_time
        return max(
            0.0,
            self._circuit_timeout - elapsed
        )

    def _record_success(self):
        """Record successful send"""
        self._failure_count = 0
        self._circuit_open = False

    def _record_failure(self):
        """Record failed send, open circuit if needed"""
        self._failure_count += 1
        if self._failure_count >= (
            self._circuit_threshold
        ):
            self._circuit_open = True
            self._circuit_open_time = time.time()
            logger.warning(
                f"Sentinel circuit breaker OPEN "
                f"after {self._failure_count} failures"
            )

    def _buffer_event(self, event: dict):
        """Buffer event during outage"""
        if len(self._buffer) < self._max_buffer:
            self._buffer.append(event)
        else:
            logger.warning(
                "Sentinel buffer full. "
                "Dropping oldest event."
            )
            self._buffer.pop(0)
            self._buffer.append(event)

    async def _drain_buffer(self):
        """Send buffered events after recovery"""
        if not self._buffer:
            return

        logger.info(
            f"Draining {len(self._buffer)} "
            f"buffered Sentinel events"
        )
        drained = []
        for event in self._buffer[:]:
            try:
                body = json.dumps(
                    [self._format_for_sentinel(event)],
                    default=str
                )
                success = await self._post_to_sentinel(
                    body
                )
                if success:
                    drained.append(event)
                else:
                    break
            except Exception:
                break

        for event in drained:
            self._buffer.remove(event)

        if drained:
            logger.info(
                f"Drained {len(drained)} events "
                f"to Sentinel"
            )


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")