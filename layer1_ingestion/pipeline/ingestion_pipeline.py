"""
Layer 1 — Data Ingestion
Ingestion Pipeline

The single entry point for Layer 1. This is what
the rest of the platform calls to turn any raw
security event into a normalized event ready for
Layer 2 ML scoring.

IT TIES TOGETHER:
    SourceDetector  — identifies the source
    IngestionRouter — dispatches to the normalizer

THE THREE INGESTION PATHS ALL LAND HERE:
    PUSH (webhook):  source is known from the URL
                     path → pass it as a hint.
    PULL (scheduled): the fetcher knows the source
                     it polled → pass it as a hint.
    STREAM (syslog):  mixed events, source unknown
                     → let the detector infer it.

THE FLOW:
    raw_event (+ optional source hint)
        → detect source (if no hint)
        → route to normalizer
        → normalized event
        → return to caller (Layer 2)

BATCH SUPPORT:
    ingest_batch() processes a list of events,
    skipping any that fail so one bad event never
    stops the batch.

USAGE:
    pipeline = IngestionPipeline()

    # PUSH - source known from webhook path
    event = pipeline.ingest(raw, source="crowdstrike")

    # STREAM - source inferred
    event = pipeline.ingest(raw)

    # BATCH
    events = pipeline.ingest_batch([e1, e2, e3])
"""

import logging

from layer1_ingestion.pipeline.source_detector\
    import SourceDetector
from layer1_ingestion.pipeline.ingestion_router\
    import IngestionRouter

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    The unified Layer 1 entry point: detect the
    source of a raw event and route it to the
    correct normalizer, returning a standard event.
    """

    def __init__(self):
        self.detector = SourceDetector()
        self.router = IngestionRouter()

        # Pipeline statistics
        self.received = 0
        self.normalized = 0
        self.dropped = 0

    def ingest(
        self,
        raw_event: dict,
        source: str = None
    ):
        """
        Ingest a single raw event.

        Args:
            raw_event: The raw event dict
            source: Optional explicit source name
                    (from webhook path / topic).
                    If None, the source is inferred.

        Returns:
            Normalized event dict, or None if the
            event could not be normalized.
        """
        self.received += 1

        # Validate input
        if not raw_event or not isinstance(
            raw_event, dict
        ):
            self.dropped += 1
            logger.warning(
                "Pipeline received invalid event"
            )
            return None

        # Step 1: identify the source
        identified = self.detector.detect(
            raw_event, hint=source
        )

        # Step 2: route to the normalizer
        normalized = self.router.route(
            raw_event, identified
        )

        # Step 3: track outcome
        if normalized is None:
            self.dropped += 1
            return None

        # Tag the event with the pipeline-identified
        # source for downstream traceability
        if isinstance(normalized, dict):
            normalized.setdefault(
                "ingestion_source", identified
            )

        self.normalized += 1
        return normalized

    def ingest_batch(
        self,
        raw_events: list,
        source: str = None
    ) -> list:
        """
        Ingest a list of raw events.

        Failed events are skipped, not fatal.

        Args:
            raw_events: List of raw event dicts
            source: Optional shared source hint for
                    the whole batch (e.g. all from
                    the same CloudTrail pull)

        Returns:
            List of normalized events (failures
            excluded)
        """
        if not raw_events:
            return []

        results = []
        for raw in raw_events:
            normalized = self.ingest(raw, source=source)
            if normalized is not None:
                results.append(normalized)

        return results

    def get_statistics(self) -> dict:
        """
        Return full pipeline statistics, including
        detector and router breakdowns.
        """
        drop_rate = (
            self.dropped / self.received * 100
            if self.received > 0 else 0
        )
        return {
            "pipeline": {
                "received": self.received,
                "normalized": self.normalized,
                "dropped": self.dropped,
                "drop_rate_pct": round(drop_rate, 2)
            },
            "detector": self.detector.get_statistics(),
            "router": self.router.get_statistics()
        }

    def supported_sources(self) -> list:
        """Sources the pipeline can currently route"""
        return self.router.supported_sources()