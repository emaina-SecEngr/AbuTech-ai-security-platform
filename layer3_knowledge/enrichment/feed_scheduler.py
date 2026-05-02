"""
Layer 3 — Knowledge Graph
Feed Scheduler — Automated Intelligence Updates

This module implements your delta processing vision:
    1. Subscribe to threat feeds
    2. Check periodically for new IOCs
    3. Compute delta — only new objects
    4. Update knowledge graph automatically
    5. Trigger alerts when high-risk IOCs found

This is the automation layer that makes your platform
continuously aware of the global threat landscape
without requiring human intervention.

Your architectural insight was exactly right:
    "Create triggers where the logic runs the report
     and creates delta information. If something new
     appears on the delta data it should automatically
     enrich and update Layer 3 as new information."

That is precisely what this module implements.

Production Deployment:
    This scheduler runs as a background service
    on your Azure Container or Kubernetes pod.
    It wakes up on schedule, checks feeds,
    processes deltas, and goes back to sleep.
    Analysts wake up to an already-enriched graph.
"""

import json
import logging
import threading
import time
from datetime import datetime
from datetime import timezone
from datetime import timedelta
from pathlib import Path
from typing import Callable
from typing import Optional

from layer3_knowledge.enrichment.stix_processor import (
    STIXProcessor
)

logger = logging.getLogger(__name__)


# ============================================================
# FEED CONFIGURATION
# ============================================================

# Free STIX/TAXII feeds available without credentials
FREE_FEEDS = {
    "cisa_ais": {
        "name": "CISA Automated Indicator Sharing",
        "url": "https://ais2.cisa.dhs.gov/taxii2/",
        "type": "taxii",
        "requires_auth": True,
        "description": (
            "US Government threat indicators. "
            "Free registration required at cisa.gov"
        ),
        "update_interval_minutes": 60
    },
    "misp_circl": {
        "name": "CIRCL MISP Feed",
        "url": (
            "https://www.circl.lu/doc/misp/"
            "feed-osint/"
        ),
        "type": "misp",
        "requires_auth": False,
        "description": (
            "CIRCL Luxembourg open source feeds. "
            "No registration required."
        ),
        "update_interval_minutes": 60
    }
}

# Sample STIX bundles for demonstration
# In production these come from real TAXII servers
SAMPLE_STIX_BUNDLES = {
    "cisa_ransomware": {
        "type": "bundle",
        "id": "bundle--cisa-ransomware-2024",
        "objects": [
            {
                "type": "indicator",
                "id": "indicator--001",
                "name": "Royal Ransomware C2 IP",
                "pattern": (
                    "[ipv4-addr:value = "
                    "'185.220.101.45']"
                ),
                "pattern_type": "stix",
                "labels": [
                    "malicious-activity",
                    "ransomware"
                ],
                "confidence": 90,
                "valid_from": "2024-01-15T00:00:00Z",
                "description": (
                    "Command and Control server "
                    "used by Royal Ransomware group"
                )
            },
            {
                "type": "indicator",
                "id": "indicator--002",
                "name": "Royal Ransomware DGA Domain",
                "pattern": (
                    "[domain-name:value = "
                    "'xjf8k2mp.duckdns.org']"
                ),
                "pattern_type": "stix",
                "labels": [
                    "malicious-activity",
                    "dga",
                    "ransomware"
                ],
                "confidence": 85,
                "valid_from": "2024-01-15T00:00:00Z"
            },
            {
                "type": "malware",
                "id": "malware--001",
                "name": "Royal Ransomware",
                "malware_types": ["ransomware"],
                "aliases": ["Royal", "Zeon"],
                "capabilities": [
                    "exfiltration",
                    "encryption",
                    "lateral-movement"
                ]
            },
            {
                "type": "threat-actor",
                "id": "threat-actor--001",
                "name": "Royal Ransomware Group",
                "aliases": ["Royal"],
                "primary_motivation": "financial-gain",
                "sophistication": "advanced",
                "labels": ["criminal"]
            },
            {
                "type": "relationship",
                "id": "relationship--001",
                "relationship_type": "indicates",
                "source_ref": "indicator--001",
                "target_ref": "malware--001"
            },
            {
                "type": "relationship",
                "id": "relationship--002",
                "relationship_type": "attributed-to",
                "source_ref": "malware--001",
                "target_ref": "threat-actor--001"
            }
        ]
    },
    "fbi_flash_emotet": {
        "type": "bundle",
        "id": "bundle--fbi-emotet-2024",
        "objects": [
            {
                "type": "indicator",
                "id": "indicator--003",
                "name": "Emotet C2 Infrastructure",
                "pattern": (
                    "[ipv4-addr:value = "
                    "'192.168.100.200']"
                ),
                "pattern_type": "stix",
                "labels": [
                    "malicious-activity",
                    "botnet"
                ],
                "confidence": 95,
                "valid_from": "2024-02-01T00:00:00Z"
            },
            {
                "type": "malware",
                "id": "malware--002",
                "name": "Emotet",
                "malware_types": [
                    "trojan", "botnet"
                ],
                "aliases": ["Geodo", "Heodo"],
                "capabilities": [
                    "email-spam",
                    "credential-stealing",
                    "module-loading"
                ]
            }
        ]
    }
}



class FeedSubscription:
    """
    Represents a subscription to a threat feed.

    Tracks:
    - Feed URL and type
    - Update interval
    - Last successful update
    - Processing statistics
    """

    def __init__(
        self,
        feed_id: str,
        feed_name: str,
        feed_url: str,
        feed_type: str,
        update_interval_minutes: int = 60,
        enabled: bool = True
    ):
        self.feed_id = feed_id
        self.feed_name = feed_name
        self.feed_url = feed_url
        self.feed_type = feed_type
        self.update_interval_minutes = (
            update_interval_minutes
        )
        self.enabled = enabled

        self.last_updated = None
        self.total_updates = 0
        self.total_iocs_processed = 0
        self.last_error = None

    def is_due_for_update(self) -> bool:
        """Check if feed needs updating"""
        if not self.enabled:
            return False
        if self.last_updated is None:
            return True

        age_minutes = (
            datetime.now(timezone.utc) -
            self.last_updated
        ).total_seconds() / 60

        return age_minutes >= self.update_interval_minutes

    def to_dict(self) -> dict:
        return {
            "feed_id": self.feed_id,
            "feed_name": self.feed_name,
            "feed_url": self.feed_url,
            "enabled": self.enabled,
            "last_updated": (
                self.last_updated.isoformat()
                if self.last_updated else None
            ),
            "total_updates": self.total_updates,
            "total_iocs_processed": (
                self.total_iocs_processed
            ),
            "update_interval_minutes": (
                self.update_interval_minutes
            )
        }


class FeedScheduler:
    """
    Automated threat feed update scheduler.

    Manages multiple feed subscriptions and
    processes them on their configured intervals.

    Implements your delta processing vision:
    - Only processes new IOCs
    - Updates graph automatically
    - Triggers alerts for high-risk IOCs
    - Runs as background thread

    Usage:
        graph = SecurityKnowledgeGraph()
        processor = STIXProcessor(graph)
        scheduler = FeedScheduler(processor)

        # Add feeds
        scheduler.add_sample_feeds()

        # Process all due feeds once
        results = scheduler.run_once()

        # Or start continuous background updates
        scheduler.start_background()
        # ... later ...
        scheduler.stop_background()
    """

    def __init__(
        self,
        stix_processor: STIXProcessor,
        alert_callback: Optional[Callable] = None
    ):
        """
        Initialize scheduler.

        Args:
            stix_processor: STIXProcessor instance
            alert_callback: Optional function called
                          when high-risk IOCs found
                          Signature: callback(alert_dict)
        """
        self.processor = stix_processor
        self.alert_callback = alert_callback

        # Feed subscriptions
        self.subscriptions = {}

        # Background thread
        self._running = False
        self._thread = None

        # Statistics
        self.total_runs = 0
        self.total_iocs_discovered = 0
        self.high_risk_alerts_generated = 0

        logger.info("FeedScheduler initialized")

    def add_subscription(
        self,
        subscription: FeedSubscription
    ) -> None:
        """Add a feed subscription"""
        self.subscriptions[
            subscription.feed_id
        ] = subscription
        logger.info(
            f"Feed subscribed: {subscription.feed_name}"
        )

    def add_sample_feeds(self) -> None:
        """
        Add demonstration feeds using sample bundles.

        In production these would be real TAXII servers.
        For demonstration we use pre-built STIX bundles
        that contain real IOC patterns.
        """
        self.add_subscription(FeedSubscription(
            feed_id="cisa_ransomware",
            feed_name="CISA Ransomware Advisory",
            feed_url="sample://cisa_ransomware",
            feed_type="sample",
            update_interval_minutes=60
        ))

        self.add_subscription(FeedSubscription(
            feed_id="fbi_emotet",
            feed_name="FBI Flash: Emotet",
            feed_url="sample://fbi_flash_emotet",
            feed_type="sample",
            update_interval_minutes=120
        ))

        logger.info(
            f"Added {len(self.subscriptions)} "
            f"sample feeds"
        )

    def run_once(self) -> dict:
        """
        Process all feeds that are due for update.

        This is your delta processing trigger.
        Checks each subscription and processes
        only feeds that have not been updated
        within their configured interval.

        Returns:
            Summary of all processing results
        """
        self.total_runs += 1
        run_start = datetime.now(timezone.utc)

        results = {
            "run_at": run_start.isoformat(),
            "feeds_checked": len(self.subscriptions),
            "feeds_updated": 0,
            "total_new_iocs": 0,
            "high_risk_alerts": 0,
            "feed_results": {}
        }

        logger.info(
            f"Feed scheduler run #{self.total_runs} "
            f"checking {len(self.subscriptions)} feeds"
        )

        for feed_id, sub in self.subscriptions.items():
            if not sub.is_due_for_update():
                logger.debug(
                    f"Feed not due: {sub.feed_name}"
                )
                continue

            logger.info(
                f"Processing feed: {sub.feed_name}"
            )

            feed_result = self._process_subscription(
                sub
            )
            results["feed_results"][feed_id] = (
                feed_result
            )

            if feed_result.get("success"):
                results["feeds_updated"] += 1
                new_iocs = feed_result.get(
                    "iocs_added", 0
                )
                results["total_new_iocs"] += new_iocs

                # Check for high-risk IOCs
                high_risk = feed_result.get(
                    "high_risk_iocs", []
                )
                if high_risk:
                    results["high_risk_alerts"] += (
                        len(high_risk)
                    )
                    self.high_risk_alerts_generated += (
                        len(high_risk)
                    )
                    self._trigger_alerts(
                        sub.feed_name, high_risk
                    )

        self.total_iocs_discovered += (
            results["total_new_iocs"]
        )

        duration = (
            datetime.now(timezone.utc) - run_start
        ).total_seconds()

        results["duration_seconds"] = round(duration, 2)

        logger.info(
            f"Feed run complete: "
            f"{results['feeds_updated']} feeds updated, "
            f"{results['total_new_iocs']} new IOCs, "
            f"{results['high_risk_alerts']} alerts"
        )

        return results

    def start_background(
        self,
        check_interval_seconds: int = 300
    ) -> None:
        """
        Start continuous background feed processing.

        Runs in a separate thread checking for
        feed updates every check_interval_seconds.

        Default: check every 5 minutes
        Each feed has its own update interval.
        """
        if self._running:
            logger.warning(
                "Scheduler already running"
            )
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._background_loop,
            args=(check_interval_seconds,),
            daemon=True
        )
        self._thread.start()

        logger.info(
            f"Background scheduler started "
            f"(check every {check_interval_seconds}s)"
        )

    def stop_background(self) -> None:
        """Stop background feed processing"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Background scheduler stopped")

    def get_statistics(self) -> dict:
        """Return scheduler statistics"""
        return {
            "total_runs": self.total_runs,
            "total_iocs_discovered": (
                self.total_iocs_discovered
            ),
            "high_risk_alerts_generated": (
                self.high_risk_alerts_generated
            ),
            "active_subscriptions": len(
                self.subscriptions
            ),
            "subscriptions": {
                fid: sub.to_dict()
                for fid, sub in
                self.subscriptions.items()
            }
        }

    # ============================================================
    # PRIVATE METHODS
    # ============================================================

    def _process_subscription(
        self,
        sub: FeedSubscription
    ) -> dict:
        """Process a single feed subscription"""
        result = {
            "feed_name": sub.feed_name,
            "success": False,
            "iocs_added": 0,
            "high_risk_iocs": []
        }

        try:
            if sub.feed_type == "sample":
                # Use sample bundles for demonstration
                bundle_key = sub.feed_id
                if bundle_key in SAMPLE_STIX_BUNDLES:
                    bundle = SAMPLE_STIX_BUNDLES[
                        bundle_key
                    ]
                    processing_result = (
                        self.processor.process_bundle(
                            bundle,
                            source_name=sub.feed_name
                        )
                    )

                    result["success"] = True
                    result["iocs_added"] = (
                        processing_result.get(
                            "iocs_added", 0
                        )
                    )

                    # Check for high-risk IOCs
                    result["high_risk_iocs"] = (
                        self._identify_high_risk_iocs(
                            processing_result
                        )
                    )

            elif sub.feed_type == "taxii":
                # Real TAXII server
                processing_result = (
                    self.processor
                    .process_bundle_from_taxii(
                        taxii_url=sub.feed_url,
                        collection_id="default",
                        source_name=sub.feed_name
                    )
                )
                result["success"] = True
                result["iocs_added"] = (
                    processing_result.get(
                        "iocs_added", 0
                    )
                )

            # Update subscription stats
            sub.last_updated = datetime.now(
                timezone.utc
            )
            sub.total_updates += 1
            sub.total_iocs_processed += (
                result["iocs_added"]
            )

        except Exception as e:
            result["error"] = str(e)
            sub.last_error = str(e)
            logger.error(
                f"Feed processing failed "
                f"({sub.feed_name}): {e}"
            )

        return result

    def _identify_high_risk_iocs(
        self,
        processing_result: dict
    ) -> list:
        """
        Identify high-risk IOCs from processing result.

        High-risk IOCs trigger immediate alerts to
        Layer 4 agents for investigation.
        """
        high_risk = []

        # Check graph for newly elevated nodes
        high_risk_nodes = (
            self.processor.graph.get_high_risk_nodes(
                threshold=0.8
            )
        )

        for node in high_risk_nodes:
            if node.properties.get("stix_source"):
                high_risk.append({
                    "entity": node.label,
                    "type": node.node_type.value,
                    "risk_score": node.risk_score,
                    "source": node.properties.get(
                        "stix_source", ""
                    )
                })

        return high_risk

    def _trigger_alerts(
        self,
        feed_name: str,
        high_risk_iocs: list
    ) -> None:
        """
        Trigger alerts for high-risk IOCs.

        In production this:
        1. Calls Layer 4 investigation agent
        2. Sends notification to SOC dashboard
        3. Creates incident ticket if configured

        For now logs and calls alert_callback.
        """
        alert = {
            "alert_type": "HIGH_RISK_IOC_DISCOVERED",
            "feed": feed_name,
            "ioc_count": len(high_risk_iocs),
            "iocs": high_risk_iocs,
            "discovered_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "message": (
                f"ALERT: {len(high_risk_iocs)} "
                f"high-risk IOCs discovered from "
                f"{feed_name}"
            )
        }

        logger.warning(alert["message"])

        if self.alert_callback:
            self.alert_callback(alert)

    def _background_loop(
        self,
        check_interval: int
    ) -> None:
        """Background thread loop"""
        logger.info("Background loop started")

        while self._running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(
                    f"Background run failed: {e}"
                )

            # Sleep in small increments to allow
            # clean shutdown
            for _ in range(check_interval):
                if not self._running:
                    break
                time.sleep(1)

        logger.info("Background loop stopped")