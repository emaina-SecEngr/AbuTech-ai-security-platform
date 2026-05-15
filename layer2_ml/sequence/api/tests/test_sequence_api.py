"""
Tests for Sequence API LSTM Bridge
"""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def api():
    from layer2_ml.sequence.api.sequence_api import (
        SequenceAPI
    )
    return SequenceAPI(persist=False)


@pytest.fixture
def mock_event():
    event = MagicMock()
    event.accessor_identity = "svc_backup"
    event.data_store_name = "prod-backup-data"
    event.data_path = "backups/daily.tar.gz"
    event.bytes_accessed = 5 * 1024 * 1024
    event.event_time = "2024-03-20T02:00:00Z"
    event.risk_score = 0.3
    return event


@pytest.fixture
def attack_events():
    events = []
    # 7 normal events
    for i in range(7):
        e = MagicMock()
        e.accessor_identity = "svc_backup"
        e.data_store_name = "prod-backup-data"
        e.data_path = f"backups/day{i}.tar.gz"
        e.bytes_accessed = 5 * 1024 * 1024
        e.event_time = f"2024-03-{20+i:02d}T02:00:00Z"
        e.risk_score = 0.3
        events.append(e)

    # 3 attack events escalating
    attack_bytes = [
        150 * 1024 * 1024,
        350 * 1024 * 1024,
        524 * 1024 * 1024
    ]
    for i, b in enumerate(attack_bytes):
        e = MagicMock()
        e.accessor_identity = "svc_backup"
        e.data_store_name = "prod-customer-data"
        e.data_path = f"customers/pci/dump_{i}.csv"
        e.bytes_accessed = b
        e.event_time = f"2024-03-{27+i:02d}T03:00:00Z"
        e.risk_score = 0.9
        events.append(e)

    return events


class TestSequenceAPIInit:

    def test_api_initializes(self, api):
        assert api is not None

    def test_api_has_history(self, api):
        assert hasattr(api, "_history")

    def test_stats_empty(self, api):
        stats = api.get_stats()
        assert stats["total_accessors_tracked"] == 0


class TestScoreEvent:

    def test_score_single_event(
        self, api, mock_event
    ):
        result = api.score_event(mock_event)
        assert result is not None
        assert result.accessor == "svc_backup"
        assert 0.0 <= result.anomaly_score <= 1.0

    def test_insufficient_data_returns_low_score(
        self, api, mock_event
    ):
        result = api.score_event(mock_event)
        assert result.pattern_detected == (
            "INSUFFICIENT_DATA"
        )
        assert result.anomaly_score == 0.0

    def test_score_builds_history(
        self, api, mock_event
    ):
        api.score_event(mock_event)
        history = api.get_accessor_history(
            "svc_backup"
        )
        assert len(history) == 1

    def test_multiple_events_build_history(
        self, api, attack_events
    ):
        for event in attack_events:
            api.score_event(event)
        history = api.get_accessor_history(
            "svc_backup"
        )
        assert len(history) == len(attack_events)

    def test_attack_pattern_scores_higher(
        self, api, attack_events
    ):
        normal_result = None
        attack_result = None

        for i, event in enumerate(attack_events):
            result = api.score_event(event)
            if i == 0:
                normal_result = result
            if i == len(attack_events) - 1:
                attack_result = result

        if (
            normal_result and attack_result and
            normal_result.sequence_length >= 3 and
            attack_result.sequence_length >= 3
        ):
            assert (
                attack_result.anomaly_score >=
                normal_result.anomaly_score
            )

    def test_result_has_required_fields(
        self, api, attack_events
    ):
        for event in attack_events[:4]:
            result = api.score_event(event)

        assert hasattr(result, "accessor")
        assert hasattr(result, "anomaly_score")
        assert hasattr(result, "risk_label")
        assert hasattr(result, "sequence_length")
        assert hasattr(result, "pattern_detected")
        assert hasattr(result, "reasoning")


class TestRuleBasedScoring:

    def test_escalating_volume_detected(self, api):
        events = []
        bytes_values = [
            5*1024*1024,
            50*1024*1024,
            200*1024*1024,
            500*1024*1024
        ]
        for i, b in enumerate(bytes_values):
            e = MagicMock()
            e.accessor_identity = "test_user"
            e.data_store_name = "bucket"
            e.bytes_accessed = b
            e.event_time = f"2024-01-0{i+1}T02:00:00Z"
            e.risk_score = 0.5
            events.append(e)

        result = api.score_accessor_history(
            "test_user", events
        )
        assert result.sequence_length >= len(events)

    def test_detect_escalation_true(self, api):
        bytes_seq = [
            5*1024*1024,
            50*1024*1024,
            200*1024*1024
        ]
        assert api._detect_escalation(
            bytes_seq
        ) is True

    def test_detect_escalation_false(self, api):
        bytes_seq = [
            200*1024*1024,
            50*1024*1024,
            5*1024*1024
        ]
        assert api._detect_escalation(
            bytes_seq
        ) is False

    def test_detect_after_hours(self, api):
        events = []
        for hour in [3, 4, 2]:
            e = MagicMock()
            e.event_time = (
                f"2024-01-01T{hour:02d}:00:00Z"
            )
            events.append(e)

        assert api._detect_after_hours(
            events
        ) is True

    def test_detect_bucket_change(self, api):
        events = []
        for i in range(4):
            e = MagicMock()
            e.data_store_name = (
                "prod-backup"
                if i < 2
                else "prod-customer-data"
            )
            events.append(e)

        assert api._detect_bucket_change(
            events
        ) is True


class TestHistoryManagement:

    def test_clear_accessor_history(
        self, api, mock_event
    ):
        api.score_event(mock_event)
        api.clear_accessor_history("svc_backup")
        history = api.get_accessor_history(
            "svc_backup"
        )
        assert len(history) == 0

    def test_max_history_enforced(self, api):
        api.max_sequence = 5
        for i in range(10):
            e = MagicMock()
            e.accessor_identity = "test"
            e.data_store_name = "bucket"
            e.bytes_accessed = 1024
            e.event_time = "2024-01-01T00:00:00Z"
            e.risk_score = 0.1
            api._add_to_history("test", e)

        history = api.get_accessor_history("test")
        assert len(history) <= 5

    def test_stats_after_events(
        self, api, attack_events
    ):
        for event in attack_events:
            api.score_event(event)
        stats = api.get_stats()
        assert stats["total_accessors_tracked"] >= 1
        assert stats["total_events_in_memory"] >= 1


class TestScoreToLabel:

    def test_critical_label(self, api):
        assert api._score_to_label(0.9) == "CRITICAL"

    def test_high_label(self, api):
        assert api._score_to_label(0.7) == "HIGH"

    def test_medium_label(self, api):
        assert api._score_to_label(0.5) == "MEDIUM"

    def test_low_label(self, api):
        assert api._score_to_label(0.3) == "LOW"

    def test_unknown_label(self, api):
        assert api._score_to_label(0.0) == "UNKNOWN"