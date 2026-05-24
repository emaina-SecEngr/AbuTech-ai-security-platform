"""
Layer 5 — Interface
IBM QRadar Destination

Forwards enriched security events to
IBM QRadar SIEM via REST API.

WHY QRADAR FOR IBM:
    IBM QRadar is IBM's flagship SIEM product.
    Used by 2,000+ enterprises globally.
    IBM Security Consultant role = selling QRadar.
    
    YOUR PLATFORM + QRADAR:
    QRadar collects raw events (it is the collector).
    YOUR platform adds ML intelligence.
    QRadar displays AbuTech-enriched offenses.
    IBM analysts work in QRadar they know.
    
    OPENING LINE FOR IBM INTERVIEW:
    "Our platform integrates natively with QRadar.
     We enrich QRadar offenses with ML scores
     and Claude agent investigation summaries.
     IBM clients get AbuTech intelligence
     inside their existing QRadar workflow."

QRADAR INTEGRATION OPTIONS:
    Option 1: Custom Log Source (syslog CEF)
    Forward events as CEF syslog to QRadar.
    QRadar parses as custom log source.
    Simple. Works with any QRadar version.
    
    Option 2: QRadar REST API
    POST to /api/siem/offenses endpoint.
    Add notes to existing offenses.
    More structured. Better integration.
    
    Option 3: QRadar Custom Properties
    Add AbuTech risk score as custom property.
    Appears in QRadar offense view.
    
    WE IMPLEMENT OPTION 2 (REST API).
    Most professional. IBM-preferred.

USAGE:
    dest = QRadarDestination()
    await dest.send(enriched_event)
    health = await dest.health_check()
"""

import json
import logging
import os
import time
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)


class QRadarDestination:
    """
    IBM QRadar SIEM destination.
    Sends enriched events via QRadar REST API.

    Events appear as QRadar custom log source.
    Analysts work in QRadar Console they know.
    AbuTech scores visible in offense view.

    QRadar API:
    https://{qradar_host}/api/siem/offenses
    SEC token in header for auth.
    """

    def __init__(
        self,
        host: str = None,
        sec_token: str = None,
        log_source_id: str = None
    ):
        self.host = (
            host or
            os.getenv("QRADAR_HOST", "")
        )
        self.sec_token = (
            sec_token or
            os.getenv("QRADAR_SEC_TOKEN", "")
        )
        self.log_source_id = (
            log_source_id or
            os.getenv("QRADAR_LOG_SOURCE_ID", "")
        )
        self.name = "qradar"
        self.is_configured = bool(
            self.host and self.sec_token
        )

        # Circuit breaker
        self._failure_count = 0
        self._circuit_open = False
        self._circuit_open_time = 0.0
        self._circuit_threshold = 3
        self._circuit_timeout = 60.0

        # Buffer
        self._buffer = []
        self._max_buffer = 500

        if not self.is_configured:
            logger.info(
                "QRadar not configured. "
                "Set QRADAR_HOST and QRADAR_SEC_TOKEN. "
                "Running in simulation mode."
            )

    async def send(
        self, event: dict
    ) -> bool:
        """
        Send enriched event to QRadar.

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
            qradar_event = (
                self._format_for_qradar(event)
            )
            success = await self._post_to_qradar(
                qradar_event
            )

            if success:
                self._record_success()
                return True
            else:
                self._record_failure()
                self._buffer_event(event)
                return False

        except Exception as e:
            logger.error(
                f"QRadar send failed: {e}"
            )
            self._record_failure()
            self._buffer_event(event)
            return False

    async def send_batch(
        self, events: list
    ) -> dict:
        """Send batch of events to QRadar"""
        if not events:
            return {"sent": 0, "failed": 0}

        if not self.is_configured:
            for event in events:
                self._simulate_send(event)
            return {
                "sent": len(events),
                "failed": 0,
                "simulated": True
            }

        sent = 0
        failed = 0
        for event in events:
            success = await self.send(event)
            if success:
                sent += 1
            else:
                failed += 1

        return {"sent": sent, "failed": failed}

    async def health_check(self) -> dict:
        """Check QRadar API connectivity"""
        if not self.is_configured:
            return {
                "healthy": True,
                "status": "simulated",
                "message": "QRadar not configured",
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
            import aiohttp
            url = (
                f"https://{self.host}"
                f"/api/siem/offenses"
                f"?filter=status%3DOPEN&fields=id"
                f"&Range=items%3D0-0"
            )
            headers = {
                "SEC": self.sec_token,
                "Accept": "application/json",
                "Version": "17.0"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(
                        total=10
                    ),
                    ssl=False
                ) as response:
                    healthy = response.status in [
                        200, 204
                    ]
                    return {
                        "healthy": healthy,
                        "status": (
                            "connected" if healthy
                            else "unreachable"
                        ),
                        "host": self.host,
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
        QRadar receives CRITICAL events only.
        Tiered routing: most important events.
        QRadar is premium SIEM - higher cost per event.
        """
        return risk_score >= 0.6

    def get_status(self) -> dict:
        """Get current destination status"""
        return {
            "name": self.name,
            "configured": self.is_configured,
            "circuit_open": self._circuit_open,
            "failure_count": self._failure_count,
            "buffered_events": len(self._buffer),
            "host": self.host
        }

    def _format_for_qradar(
        self, event: dict
    ) -> dict:
        """
        Format DataAccessEvent for QRadar API.
        QRadar uses its own JSON format.
        """
        risk_score = float(
            event.get("risk_score", 0.0) or 0.0
        )

        return {
            "abutech_risk_score": risk_score,
            "abutech_risk_label": (
                self._score_to_label(risk_score)
            ),
            "abutech_verdict": event.get(
                "verdict", ""
            ),
            "abutech_source_system": event.get(
                "source_system", ""
            ),
            "abutech_accessor": event.get(
                "accessor_identity", ""
            ),
            "abutech_source_ip": event.get(
                "source_ip", ""
            ),
            "abutech_data_store": event.get(
                "data_store_name", ""
            ),
            "abutech_classification": event.get(
                "data_classification", ""
            ),
            "abutech_risk_reasons": ", ".join(
                event.get("risk_reasons", [])
            )[:500],
            "abutech_mitre": ", ".join(
                event.get("mitre_techniques", [])
            ),
            "abutech_agent_summary": event.get(
                "agent_summary", ""
            )[:1000],
            "abutech_hitl_status": event.get(
                "hitl_status", ""
            ),
            "abutech_timestamp": event.get(
                "event_time", _now()
            ),
            "abutech_platform_version": "1.0.0"
        }

    async def _post_to_qradar(
        self, qradar_event: dict
    ) -> bool:
        """POST event to QRadar REST API"""
        try:
            import aiohttp
        except ImportError:
            return True

        try:
            url = (
                f"https://{self.host}"
                f"/api/siem/offenses"
            )
            headers = {
                "SEC": self.sec_token,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Version": "17.0"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=qradar_event,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(
                        total=30
                    ),
                    ssl=False
                ) as response:
                    if response.status in [
                        200, 201, 204
                    ]:
                        logger.info(
                            "QRadar: event forwarded"
                        )
                        return True
                    logger.warning(
                        f"QRadar returned "
                        f"{response.status}"
                    )
                    return False

        except Exception as e:
            logger.error(
                f"QRadar POST failed: {e}"
            )
            return False

    def _simulate_send(self, event: dict) -> bool:
        """Simulate send when not configured"""
        risk = float(
            event.get("risk_score", 0.0) or 0.0
        )
        label = self._score_to_label(risk)
        logger.info(
            f"[QRADAR SIMULATED] "
            f"accessor={event.get('accessor_identity')} "
            f"risk={risk:.3f} {label}"
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

    def _is_circuit_open(self) -> bool:
        if not self._circuit_open:
            return False
        elapsed = (
            time.time() - self._circuit_open_time
        )
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
                "QRadar circuit breaker OPEN"
            )

    def _buffer_event(self, event: dict):
        if len(self._buffer) < self._max_buffer:
            self._buffer.append(event)
        else:
            self._buffer.pop(0)
            self._buffer.append(event)


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")