"""
Layer 1 — Data Ingestion
Base Normalizer Class

Every data source normalizer inherits from this class.
It provides shared functionality that all normalizers need:

    - Input validation
    - Timestamp conversion
    - Event ID generation
    - Error handling
    - Audit logging
    - Output validation

A normalizer that inherits this class only needs to
implement one method: normalize_event()

That method contains the source-specific transformation
logic. Everything else is handled here automatically.

Usage:
    class CrowdStrikeNormalizer(BaseNormalizer):
        def normalize_event(self, raw_event):
            # CrowdStrike-specific logic here
            pass
"""

import hashlib
import logging
import uuid
from abc import ABC
from abc import abstractmethod
from datetime import datetime
from datetime import timezone
from typing import Optional

from layer1_ingestion.schema.ecs_schema import ECSNormalized


# Configure logger for this module
logger = logging.getLogger(__name__)


class NormalizationError(Exception):
    """
    Raised when a raw event cannot be normalized.

    This is a specific exception class rather than
    a generic Exception so calling code can catch
    normalization failures specifically without
    accidentally catching other errors.

    In production your pipeline catches this and:
    1. Logs the failed event for investigation
    2. Sends it to a dead letter queue
    3. Continues processing other events
    4. Never crashes the pipeline
    """
    pass


class BaseNormalizer(ABC):
    """
    Abstract base class for all security event normalizers.

    Abstract means you cannot instantiate this class directly.
    You must create a subclass that implements normalize_event().

    This design enforces a contract:
    Every normalizer MUST implement normalize_event()
    or Python will raise a TypeError at instantiation.

    This catches missing implementations at startup
    rather than at runtime when processing real events.
    """

    def __init__(self, source_name: str):
        """
        Initialize the normalizer.

        source_name: Human readable name for this data source
                     e.g. "crowdstrike_edr", "okta_system_log"
                     Used in logging and normalized event metadata
        """
        self.source_name = source_name
        self.logger = logging.getLogger(
            f"{__name__}.{source_name}"
        )

        # Statistics tracking
        # These counters help you monitor normalizer
        # health in production
        self.events_processed = 0
        self.events_failed = 0
        self.events_succeeded = 0

    # ============================================================
    # ABSTRACT METHOD
    # Every subclass MUST implement this method
    # This is where source-specific logic lives
    # ============================================================

    @abstractmethod
    def normalize_event(self, raw_event: dict) -> ECSNormalized:
        """
        Transform a raw source event into ECS format.

        This is the only method subclasses must implement.
        All other methods in this class are inherited
        and work automatically.

        Args:
            raw_event: Raw event dict from the data source
                      Structure varies per source

        Returns:
            ECSNormalized: Standardized ECS event

        Raises:
            NormalizationError: If event cannot be normalized
        """
        pass

    # ============================================================
    # SHARED METHODS
    # All normalizers inherit these automatically
    # ============================================================

    def normalize(self, raw_event: dict) -> Optional[ECSNormalized]:
        """
        Main entry point for normalization.

        This method wraps normalize_event() with:
        - Input validation
        - Error handling
        - Statistics tracking
        - Audit logging

        Calling code always calls normalize()
        never normalize_event() directly.

        Returns None if normalization fails
        so the pipeline can skip and continue
        rather than crashing.
        """
        self.events_processed += 1

        # Step 1: Validate input
        if not self._validate_input(raw_event):
            self.events_failed += 1
            return None

        # Step 2: Attempt normalization
        try:
            normalized = self.normalize_event(raw_event)

            # Step 3: Add metadata the subclass cannot add
            normalized.normalized_at = self._get_timestamp()
            normalized.raw_event_hash = self._hash_event(
                raw_event
            )

            # Step 4: Validate output
            if not self._validate_output(normalized):
                self.events_failed += 1
                return None

            # Step 5: Log success
            self.events_succeeded += 1
            self.logger.debug(
                f"Normalized event {normalized.event.id} "
                f"from {self.source_name}"
            )

            return normalized

        except NormalizationError as e:
            # Expected normalization failures
            # Log and continue
            self.events_failed += 1
            self.logger.warning(
                f"Normalization failed for {self.source_name}: "
                f"{str(e)}"
            )
            return None

        except Exception as e:
            # Unexpected errors
            # Log with full context for debugging
            self.events_failed += 1
            self.logger.error(
                f"Unexpected error normalizing "
                f"{self.source_name} event: {str(e)}",
                exc_info=True
            )
            return None

    def convert_unix_ms_to_iso(
        self,
        unix_ms: int
    ) -> str:
        """
        Convert Unix millisecond timestamp to ISO 8601 UTC.

        This is used by multiple normalizers because many
        security data sources use Unix timestamps.

        CrowdStrike uses Unix milliseconds.
        Syslog uses Unix seconds.
        Windows events use Windows FILETIME format.

        All of them need to become ISO 8601 UTC for ECS.

        Args:
            unix_ms: Unix timestamp in milliseconds
                     e.g. 1711678663000

        Returns:
            ISO 8601 UTC string
            e.g. "2025-03-29T02:17:43.000000Z"
        """
        try:
            # Convert milliseconds to seconds
            unix_seconds = unix_ms / 1000

            # Create UTC datetime object
            dt = datetime.fromtimestamp(
                unix_seconds,
                tz=timezone.utc
            )

            # Return ISO 8601 format
            return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        except (ValueError, TypeError, OSError) as e:
            raise NormalizationError(
                f"Cannot convert timestamp {unix_ms}: {e}"
            )

    def convert_unix_seconds_to_iso(
        self,
        unix_seconds: int
    ) -> str:
        """
        Convert Unix second timestamp to ISO 8601 UTC.
        Used for Syslog and other second-precision sources.
        """
        return self.convert_unix_ms_to_iso(unix_seconds * 1000)

    def split_domain_username(
        self,
        username: str
    ) -> tuple:
        """
        Split domain-prefixed username into components.

        Handles these common formats:
            CORP\\jsmith    → ("CORP", "jsmith")
            CORP/jsmith     → ("CORP", "jsmith")
            jsmith          → (None, "jsmith")
            jsmith@corp.com → ("corp.com", "jsmith")

        This is used by EDR, Windows Event, and
        Active Directory normalizers.

        Your IAM background means you understand
        why separating domain from username matters -
        the same username can exist in multiple domains
        and must be disambiguated for accurate identity
        correlation in the knowledge graph.

        Args:
            username: Raw username string from source

        Returns:
            Tuple of (domain, username)
        """
        if not username:
            return None, None

        # Handle DOMAIN\\username format
        if "\\" in username:
            parts = username.split("\\", 1)
            return parts[0], parts[1]

        # Handle DOMAIN/username format
        if "/" in username and "@" not in username:
            parts = username.split("/", 1)
            return parts[0], parts[1]

        # Handle username@domain format
        if "@" in username:
            parts = username.split("@", 1)
            return parts[1], parts[0]

        # No domain prefix
        return None, username

    def extract_filename_from_path(
        self,
        file_path: str
    ) -> str:
        """
        Extract filename from a full file path.

        Handles both Windows and Linux paths:
            C:\\Windows\\System32\\powershell.exe
            → powershell.exe

            \\Device\\HarddiskVolume3\\Windows\\powershell.exe
            → powershell.exe

            /usr/bin/python3
            → python3

        Args:
            file_path: Full file path string

        Returns:
            Filename only
        """
        if not file_path:
            return ""

        # Normalize path separators
        normalized_path = file_path.replace("\\", "/")

        # Extract filename
        return normalized_path.split("/")[-1]

    def clean_windows_device_path(
        self,
        device_path: str
    ) -> str:
        """
        Convert Windows device path to human readable path.
        """
        if not device_path:
            return ""

        import re

        cleaned = re.sub(
            r"\\Device\\HarddiskVolume\d+",
            "C:",
            device_path
        )

        return cleaned

    def generate_event_id(self) -> str:
        """
        Generate a unique ID for this normalized event.

        Uses UUID4 for guaranteed uniqueness across
        all events from all sources.

        This ID is used by:
        - Elasticsearch to deduplicate events
        - Knowledge graph to reference specific events
        - Audit trails to track specific event processing
        """
        return str(uuid.uuid4())

    def get_statistics(self) -> dict:
        """
        Return normalization statistics for monitoring.

        Called by your MLOps monitoring layer to track
        normalizer health in production.

        If success rate drops below a threshold it signals
        that the data source format may have changed
        and the normalizer needs updating.
        """
        total = self.events_processed
        success_rate = (
            self.events_succeeded / total * 100
            if total > 0 else 0
        )

        return {
            "source": self.source_name,
            "events_processed": self.events_processed,
            "events_succeeded": self.events_succeeded,
            "events_failed": self.events_failed,
            "success_rate_pct": round(success_rate, 2)
        }

    # ============================================================
    # PRIVATE METHODS
    # Internal helpers not called by outside code
    # ============================================================

    def _validate_input(self, raw_event: dict) -> bool:
        """
        Validate raw event before attempting normalization.

        Catches the most common input problems:
        - None input
        - Empty dict
        - Non-dict input

        Returns True if valid, False if invalid.
        Logs specific reason for rejection.
        """
        if raw_event is None:
            self.logger.warning(
                f"{self.source_name}: Received None event"
            )
            return False

        if not isinstance(raw_event, dict):
            self.logger.warning(
                f"{self.source_name}: Event is not a dict, "
                f"got {type(raw_event)}"
            )
            return False

        if len(raw_event) == 0:
            self.logger.warning(
                f"{self.source_name}: Received empty event"
            )
            return False

        return True

    def _validate_output(
        self,
        normalized: ECSNormalized
    ) -> bool:
        """
        Validate normalized event meets ECS requirements.

        Checks that all required fields are populated.
        Required fields defined in ECSNormalized.get_required_fields()

        This is your data quality gate.
        Events that do not meet the schema do not
        flow to Layer 2 ML models.
        """
        if normalized is None:
            return False

        # Check required fields exist
        event_dict = normalized.to_dict()

        missing_fields = []

        for field in ECSNormalized.get_required_fields():
            # Navigate nested fields using dot notation
            parts = field.split(".")
            current = event_dict

            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    current = None
                    break

            if current is None:
                missing_fields.append(field)

        if missing_fields:
            self.logger.warning(
                f"{self.source_name}: Normalized event "
                f"missing required fields: {missing_fields}"
            )
            return False

        return True

    def _get_timestamp(self) -> str:
        """Return current UTC timestamp in ISO 8601 format"""
        return datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )

    def _hash_event(self, raw_event: dict) -> str:
        """
        Generate SHA256 hash of raw event.

        Used for:
        - Deduplication: same event arriving twice
          has the same hash
        - Audit trail: proves which raw event
          produced which normalized event
        - Forensics: verify event integrity
        """
        import json
        event_str = json.dumps(raw_event, sort_keys=True)
        return hashlib.sha256(
            event_str.encode()
        ).hexdigest()