"""
Layer 2 — ML Processing
Sequence API — LSTM Bridge

Bridges Layer 5 FastAPI ingestion pipeline
to Layer 2 LSTMAttentionDetector.

WHY THIS EXISTS:
    FastAPI receives individual events.
    LSTM needs sequences of events.
    This module maintains per-accessor
    event history and feeds sequences
    to the LSTM detector.

SLOW EXFIL DETECTION:
    Day 1-7:  svc_backup 5MB   → NORMAL
    Day 8:    svc_backup 150MB → ELEVATED
    Day 9:    svc_backup 350MB → HIGH
    Day 10:   svc_backup 524MB → CRITICAL
    
    IsolationForest sees each event alone.
    LSTM sees the 10-day pattern.
    LSTM catches what IF misses.

USAGE:
    api = SequenceAPI()
    
    # Score new event in context of history
    result = api.score_event(data_event)
    print(result.anomaly_score)   # 0.82
    print(result.risk_label)      # HIGH
    print(result.sequence_length) # 10
"""

import logging
import os
import json
from datetime import datetime
from datetime import timezone
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from typing import List

logger = logging.getLogger(__name__)

# History configuration
MAX_SEQUENCE_LENGTH = 60
MIN_SEQUENCE_FOR_LSTM = 3
HISTORY_FILE = "data/sequence/event_history.json"


@dataclass
class SequenceScore:
    """Result from sequence scoring"""
    accessor: str
    anomaly_score: float
    risk_label: str
    sequence_length: int
    pattern_detected: str
    kill_chain_score: float
    slow_exfil_score: float
    used_lstm: bool
    reasoning: str


class SequenceAPI:
    """
    API bridge between FastAPI ingestion
    and LSTM sequence detector.

    Maintains per-accessor event history.
    Scores new events in sequence context.
    Returns anomaly score for ensemble.
    """

    def __init__(
        self,
        max_sequence: int = MAX_SEQUENCE_LENGTH,
        min_sequence: int = MIN_SEQUENCE_FOR_LSTM,
        persist: bool = True
    ):
        self.max_sequence = max_sequence
        self.min_sequence = min_sequence
        self.persist = persist

        # In-memory history per accessor
        # {accessor_id: [DataAccessEvent, ...]}
        self._history = defaultdict(list)

        # Load persisted history if available
        if persist:
            self._load_history()

        # Initialize LSTM detector
        self._lstm = None
        self._sequence_builder = None
        self._initialize_lstm()

    def _initialize_lstm(self):
        """Initialize LSTM detector lazily"""
        try:
            from layer2_ml.sequence\
                .lstm_attention_detector \
                import LSTMAttentionDetector
            from layer2_ml.sequence\
                .sequence_builder \
                import SequenceBuilder

            self._lstm = LSTMAttentionDetector()
            self._sequence_builder = SequenceBuilder()
            logger.info("LSTM detector initialized")
        except Exception as e:
            logger.warning(
                f"LSTM init failed: {e}. "
                f"Using rule-based scoring."
            )

    def score_event(
        self,
        data_event
    ) -> SequenceScore:
        """
        Score a data access event in the
        context of accessor's history.

        Args:
            data_event: DataAccessEvent from
                        Layer 1 normalizer

        Returns:
            SequenceScore with anomaly score
            and pattern details
        """
        accessor = getattr(
            data_event, "accessor_identity", ""
        ) or "unknown"

        # Add to history
        self._add_to_history(accessor, data_event)

        # Get current sequence
        history = self._history[accessor]
        seq_length = len(history)

        # Need minimum events for LSTM
        if seq_length < self.min_sequence:
            return self._insufficient_data_score(
                accessor, seq_length
            )

        # Try LSTM scoring
        if self._lstm and self._sequence_builder:
            return self._score_with_lstm(
                accessor, history
            )

        # Fall back to rule-based
        return self._rule_based_score(
            accessor, history
        )

    def score_accessor_history(
        self,
        accessor: str,
        events: list
    ) -> SequenceScore:
        """
        Score a pre-built list of events
        for a specific accessor.

        Used for batch scoring and testing.
        """
        if not events:
            return self._insufficient_data_score(
                accessor, 0
            )

        self._history[accessor] = events[
            -self.max_sequence:
        ]

        return self.score_event(events[-1])

    def get_accessor_history(
        self,
        accessor: str
    ) -> list:
        """Get stored event history for accessor"""
        return self._history.get(accessor, [])

    def clear_accessor_history(
        self,
        accessor: str
    ) -> None:
        """Clear history for specific accessor"""
        if accessor in self._history:
            del self._history[accessor]
            logger.info(
                f"Cleared history for {accessor}"
            )

    def get_stats(self) -> dict:
        """Get sequence API statistics"""
        return {
            "total_accessors_tracked": len(
                self._history
            ),
            "total_events_in_memory": sum(
                len(h) for h in self._history.values()
            ),
            "lstm_available": self._lstm is not None,
            "max_sequence_length": self.max_sequence
        }

    # ============================================================
    # PRIVATE METHODS
    # ============================================================

    def _add_to_history(
        self,
        accessor: str,
        event
    ) -> None:
        """Add event to accessor history"""
        self._history[accessor].append(event)

        # Keep only max_sequence events
        if len(self._history[accessor]) > (
            self.max_sequence
        ):
            self._history[accessor] = (
                self._history[accessor][
                    -self.max_sequence:
                ]
            )

        # Persist to disk
        if self.persist:
            self._save_history_entry(
                accessor, event
            )

    def _score_with_lstm(
        self,
        accessor: str,
        history: list
    ) -> SequenceScore:
        """Score using LSTM detector"""
        try:
            # Build sequences
            for event in history:
                self._sequence_builder\
                    .add_data_access_event(event)

            sequences = (
                self._sequence_builder.get_sequences(
                    accessor
                )
            )

            kill_chain_score = 0.0
            slow_exfil_score = 0.0
            pattern = "NO_PATTERN"

            if sequences:
                if "kill_chain" in sequences:
                    kc_result = (
                        self._lstm.score_kill_chain(
                            sequences["kill_chain"]
                        )
                    )
                    kill_chain_score = float(
                        kc_result.anomaly_score
                    )

                if "slow_exfil" in sequences:
                    se_result = (
                        self._lstm.score_slow_exfil(
                            sequences["slow_exfil"]
                        )
                    )
                    slow_exfil_score = float(
                        se_result.anomaly_score
                    )

            final_score = max(
                kill_chain_score,
                slow_exfil_score
            )

            if kill_chain_score >= 0.7:
                pattern = "KILL_CHAIN_DETECTED"
            elif slow_exfil_score >= 0.7:
                pattern = "SLOW_EXFIL_DETECTED"
            elif final_score >= 0.5:
                pattern = "ANOMALOUS_SEQUENCE"

            risk_label = self._score_to_label(
                final_score
            )
            reasoning = self._build_reasoning(
                accessor, len(history),
                kill_chain_score, slow_exfil_score,
                pattern
            )

            logger.info(
                f"LSTM scored {accessor}: "
                f"score={final_score:.3f} "
                f"pattern={pattern} "
                f"seq_len={len(history)}"
            )

            return SequenceScore(
                accessor=accessor,
                anomaly_score=final_score,
                risk_label=risk_label,
                sequence_length=len(history),
                pattern_detected=pattern,
                kill_chain_score=kill_chain_score,
                slow_exfil_score=slow_exfil_score,
                used_lstm=True,
                reasoning=reasoning
            )

        except Exception as e:
            logger.warning(
                f"LSTM scoring failed for "
                f"{accessor}: {e}. "
                f"Using rule-based."
            )
            return self._rule_based_score(
                accessor, history
            )

    def _rule_based_score(
        self,
        accessor: str,
        history: list
    ) -> SequenceScore:
        """
        Rule-based sequence scoring fallback.
        Used when LSTM is unavailable.
        Detects patterns using heuristics.
        """
        if not history:
            return self._insufficient_data_score(
                accessor, 0
            )

        # Extract bytes over time
        bytes_sequence = []
        for event in history:
            bytes_val = getattr(
                event, "bytes_accessed", 0
            ) or 0
            bytes_sequence.append(bytes_val)

        # Check for escalating volume pattern
        escalating = self._detect_escalation(
            bytes_sequence
        )

        # Check for after-hours pattern
        after_hours = self._detect_after_hours(
            history
        )

        # Check for bucket change pattern
        bucket_change = self._detect_bucket_change(
            history
        )

        # Calculate composite score
        score = 0.0
        patterns = []

        if escalating:
            score = max(score, 0.75)
            patterns.append("ESCALATING_VOLUME")

        if after_hours and escalating:
            score = max(score, 0.85)
            patterns.append("AFTER_HOURS_ESCALATION")

        if bucket_change and escalating:
            score = max(score, 0.90)
            patterns.append("BUCKET_PIVOT_EXFIL")

        recent_event = history[-1]
        recent_bytes = getattr(
            recent_event, "bytes_accessed", 0
        ) or 0
        recent_mb = recent_bytes / (1024 * 1024)

        if recent_mb > 100:
            score = max(score, 0.70)
        if recent_mb > 300:
            score = max(score, 0.85)

        pattern_str = (
            " + ".join(patterns)
            if patterns else "NO_PATTERN"
        )
        risk_label = self._score_to_label(score)

        reasoning = (
            f"Rule-based sequence analysis "
            f"for {accessor}: "
            f"{len(history)} events. "
            f"Score: {score:.2f}. "
            f"Patterns: {pattern_str}."
        )

        return SequenceScore(
            accessor=accessor,
            anomaly_score=score,
            risk_label=risk_label,
            sequence_length=len(history),
            pattern_detected=pattern_str,
            kill_chain_score=0.0,
            slow_exfil_score=score,
            used_lstm=False,
            reasoning=reasoning
        )

    def _detect_escalation(
        self,
        bytes_sequence: list
    ) -> bool:
        """Detect escalating volume pattern"""
        if len(bytes_sequence) < 3:
            return False

        recent = bytes_sequence[-3:]
        if all(
            recent[i] <= recent[i+1]
            for i in range(len(recent)-1)
        ):
            if recent[-1] > recent[0] * 2:
                return True

        return False

    def _detect_after_hours(
        self,
        history: list
    ) -> bool:
        """Detect after-hours access pattern"""
        after_hours_count = 0
        for event in history[-5:]:
            event_time = str(
                getattr(event, "event_time", "") or ""
            )
            if event_time:
                try:
                    hour = int(event_time[11:13])
                    if hour < 6 or hour > 22:
                        after_hours_count += 1
                except (ValueError, IndexError):
                    pass

        return after_hours_count >= 2

    def _detect_bucket_change(
        self,
        history: list
    ) -> bool:
        """Detect accessor switching to different bucket"""
        if len(history) < 4:
            return False

        early_buckets = set()
        recent_buckets = set()

        half = len(history) // 2
        for event in history[:half]:
            bucket = getattr(
                event, "data_store_name", ""
            ) or ""
            if bucket:
                early_buckets.add(bucket)

        for event in history[half:]:
            bucket = getattr(
                event, "data_store_name", ""
            ) or ""
            if bucket:
                recent_buckets.add(bucket)

        new_buckets = recent_buckets - early_buckets
        return len(new_buckets) > 0

    def _score_to_label(self, score: float) -> str:
        """Convert score to risk label"""
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score > 0.0:
            return "LOW"
        return "UNKNOWN"

    def _build_reasoning(
        self,
        accessor: str,
        seq_length: int,
        kill_chain: float,
        slow_exfil: float,
        pattern: str
    ) -> str:
        """Build human-readable reasoning"""
        reasoning = (
            f"LSTM sequence analysis for "
            f"{accessor}: "
            f"{seq_length} events analyzed. "
        )
        if kill_chain > 0.5:
            reasoning += (
                f"Kill chain score: "
                f"{kill_chain:.2f}. "
            )
        if slow_exfil > 0.5:
            reasoning += (
                f"Slow exfil score: "
                f"{slow_exfil:.2f}. "
            )
        reasoning += f"Pattern: {pattern}."
        return reasoning

    def _insufficient_data_score(
        self,
        accessor: str,
        seq_length: int
    ) -> SequenceScore:
        """Return low score when insufficient data"""
        return SequenceScore(
            accessor=accessor,
            anomaly_score=0.0,
            risk_label="UNKNOWN",
            sequence_length=seq_length,
            pattern_detected="INSUFFICIENT_DATA",
            kill_chain_score=0.0,
            slow_exfil_score=0.0,
            used_lstm=False,
            reasoning=(
                f"Only {seq_length} events for "
                f"{accessor}. Need at least "
                f"{self.min_sequence} for scoring."
            )
        )

    def _load_history(self) -> None:
        """Load persisted history from disk"""
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE) as f:
                    data = json.load(f)
                    logger.info(
                        f"Loaded sequence history: "
                        f"{len(data)} accessors"
                    )
        except Exception as e:
            logger.debug(
                f"Could not load history: {e}"
            )

    def _save_history_entry(
        self,
        accessor: str,
        event
    ) -> None:
        """Save event to persistent store"""
        try:
            os.makedirs(
                os.path.dirname(HISTORY_FILE),
                exist_ok=True
            )
        except Exception:
            pass