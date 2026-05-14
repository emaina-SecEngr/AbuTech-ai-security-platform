"""
Tests for Agent Memory Store
"""

import pytest
import os
import tempfile
from unittest.mock import patch


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def temp_memory_dir(tmp_path):
    """Use temp directory for memory storage"""
    return str(tmp_path / "test_memory")


@pytest.fixture
def memory_store(temp_memory_dir):
    """Create memory store with temp directory"""
    from layer4_reasoning.memory.memory_store import (
        AgentMemoryStore
    )
    store = AgentMemoryStore(
        persist_directory=temp_memory_dir,
        collection_name="test_investigations"
    )
    return store


@pytest.fixture
def sample_state():
    """Sample investigation state"""
    return {
        "event_id": "test-001",
        "event_host": "prod-customer-data",
        "event_user": "svc_backup",
        "overall_verdict": "DATA_EXFILTRATION",
        "overall_risk_score": 0.95,
        "severity_rating": "CRITICAL",
        "triage_verdict": "INVESTIGATE",
        "triage_reasoning": (
            "Critical risk score 0.95. "
            "Service account accessing PCI data "
            "from Tor exit node at 3am."
        ),
        "intel_summary": (
            "Activity attributed to insider threat. "
            "Tor exit node confirmed."
        ),
        "investigation_summary": (
            "svc_backup accessed prod-customer-data "
            "containing PCI data. 524MB transferred."
        ),
        "confirmed_techniques": [
            "T1530", "T1048", "T1078"
        ],
        "threat_actor_identified": None,
        "malware_family_confirmed": None,
        "compromise_confirmed": False
    }


@pytest.fixture
def sample_state_2():
    """Second sample state for search tests"""
    return {
        "event_id": "test-002",
        "event_host": "prod-rds-database",
        "event_user": "admin_user",
        "overall_verdict": "IDENTITY_COMPROMISE",
        "overall_risk_score": 0.88,
        "severity_rating": "HIGH",
        "triage_verdict": "INVESTIGATE",
        "triage_reasoning": (
            "Admin account accessed RDS database "
            "from unusual location."
        ),
        "intel_summary": (
            "Possible credential theft. "
            "Admin account behavior anomalous."
        ),
        "investigation_summary": (
            "admin_user accessed production database "
            "from foreign IP address."
        ),
        "confirmed_techniques": [
            "T1078", "T1021.001"
        ],
        "threat_actor_identified": "APT29",
        "malware_family_confirmed": None,
        "compromise_confirmed": True
    }


# ============================================================
# INITIALIZATION TESTS
# ============================================================

class TestInitialization:

    def test_store_creates_directory(
        self, temp_memory_dir
    ):
        from layer4_reasoning.memory.memory_store \
            import AgentMemoryStore
        store = AgentMemoryStore(
            persist_directory=temp_memory_dir
        )
        assert os.path.exists(temp_memory_dir)

    def test_store_initializes_successfully(
        self, memory_store
    ):
        assert memory_store is not None

    def test_backend_is_set(self, memory_store):
        assert hasattr(memory_store, "use_chromadb")

    def test_json_fallback_works(
        self, temp_memory_dir
    ):
        from layer4_reasoning.memory.memory_store \
            import AgentMemoryStore
        with patch.dict(
            "sys.modules", {"chromadb": None}
        ):
            store = AgentMemoryStore(
                persist_directory=temp_memory_dir
            )
            assert store is not None


# ============================================================
# STORE TESTS
# ============================================================

class TestStoreInvestigation:

    def test_store_returns_true(
        self, memory_store, sample_state
    ):
        result = memory_store.store_investigation(
            sample_state
        )
        assert result is True

    def test_store_with_custom_id(
        self, memory_store, sample_state
    ):
        result = memory_store.store_investigation(
            sample_state,
            investigation_id="custom-001"
        )
        assert result is True

    def test_store_multiple_investigations(
        self, memory_store, sample_state,
        sample_state_2
    ):
        r1 = memory_store.store_investigation(
            sample_state
        )
        r2 = memory_store.store_investigation(
            sample_state_2
        )
        assert r1 is True
        assert r2 is True

    def test_store_empty_state(self, memory_store):
        result = memory_store.store_investigation({})
        assert result is True

    def test_stats_after_store(
        self, memory_store, sample_state
    ):
        memory_store.store_investigation(sample_state)
        stats = memory_store.get_stats()
        assert stats["total_investigations"] >= 1


# ============================================================
# SEARCH TESTS
# ============================================================

class TestSearchSimilar:

    def test_search_empty_store(self, memory_store):
        memory_store.clear_all()
        results = memory_store.search_similar(
            "service account data exfiltration"
        )
        assert results == []

    def test_search_finds_stored(
        self, memory_store, sample_state
    ):
        memory_store.store_investigation(
            sample_state,
            investigation_id="search-test-001"
        )
        results = memory_store.search_similar(
            "service account PCI data exfiltration"
        )
        assert len(results) >= 1

    def test_search_returns_list(
        self, memory_store, sample_state
    ):
        memory_store.store_investigation(sample_state)
        results = memory_store.search_similar(
            "security incident"
        )
        assert isinstance(results, list)

    def test_search_result_has_fields(
        self, memory_store, sample_state
    ):
        memory_store.store_investigation(
            sample_state,
            investigation_id="field-test-001"
        )
        results = memory_store.search_similar(
            "svc_backup exfiltration"
        )
        if results:
            result = results[0]
            assert "id" in result
            assert "document" in result
            assert "metadata" in result

    def test_search_n_results_limit(
        self, memory_store, sample_state,
        sample_state_2
    ):
        memory_store.store_investigation(
            sample_state,
            investigation_id="limit-001"
        )
        memory_store.store_investigation(
            sample_state_2,
            investigation_id="limit-002"
        )
        results = memory_store.search_similar(
            "security incident investigation",
            n_results=1
        )
        assert len(results) <= 1

    def test_search_with_severity_filter(
        self, memory_store, sample_state
    ):
        memory_store.store_investigation(
            sample_state,
            investigation_id="sev-test-001"
        )
        results = memory_store.search_similar(
            "critical security incident",
            filter_severity="CRITICAL"
        )
        assert isinstance(results, list)


# ============================================================
# ENTITY HISTORY TESTS
# ============================================================

class TestEntityHistory:

    def test_empty_history(self, memory_store):
        profile = memory_store.get_entity_history(
            "unknown_user"
        )
        assert profile["total_incidents"] == 0
        assert profile["repeat_offender"] is False

    def test_history_after_store(
        self, memory_store, sample_state
    ):
        memory_store.store_investigation(
            sample_state,
            investigation_id="hist-001"
        )
        profile = memory_store.get_entity_history(
            "svc_backup"
        )
        assert isinstance(profile, dict)
        assert "total_incidents" in profile
        assert "repeat_offender" in profile
        assert "risk_level" in profile
        assert "summary" in profile

    def test_repeat_offender_detection(
        self, memory_store
    ):
        from layer4_reasoning.memory.memory_store \
            import AgentMemoryStore
        state_template = {
            "event_host": "prod-data",
            "event_user": "repeat_user",
            "overall_verdict": "DATA_EXFILTRATION",
            "overall_risk_score": 0.9,
            "severity_rating": "CRITICAL",
            "triage_verdict": "INVESTIGATE",
            "triage_reasoning": "repeat user critical",
            "intel_summary": "repeat pattern",
            "investigation_summary": (
                "repeat_user accessed prod-data"
            ),
            "confirmed_techniques": ["T1530"],
            "threat_actor_identified": None,
            "malware_family_confirmed": None,
            "compromise_confirmed": False
        }

        for i in range(3):
            state = {
                **state_template,
                "event_id": f"repeat-{i}"
            }
            memory_store.store_investigation(
                state,
                investigation_id=f"repeat-inv-{i}"
            )

        profile = memory_store.get_entity_history(
            "repeat_user"
        )
        assert profile["total_incidents"] >= 3
        assert profile["repeat_offender"] is True

    def test_profile_has_required_fields(
        self, memory_store, sample_state
    ):
        memory_store.store_investigation(sample_state)
        profile = memory_store.get_entity_history(
            "svc_backup"
        )
        required = [
            "entity", "entity_type",
            "total_incidents", "critical_incidents",
            "high_incidents", "repeat_offender",
            "escalating", "risk_level",
            "recent_incidents", "summary"
        ]
        for field in required:
            assert field in profile


# ============================================================
# STATS TESTS
# ============================================================

class TestStats:

    def test_stats_empty_store(self, memory_store):
        stats = memory_store.get_stats()
        assert "total_investigations" in stats
        assert "backend" in stats

    def test_stats_after_store(
        self, memory_store, sample_state
    ):
        memory_store.store_investigation(sample_state)
        stats = memory_store.get_stats()
        assert stats["total_investigations"] >= 1

    def test_stats_has_backend_info(
        self, memory_store
    ):
        stats = memory_store.get_stats()
        assert stats["backend"] in [
            "chromadb", "json_fallback"
        ]


# ============================================================
# DOCUMENT BUILDING TESTS
# ============================================================

class TestDocumentBuilding:

    def test_document_contains_host(
        self, memory_store, sample_state
    ):
        doc = memory_store._build_document(
            sample_state
        )
        assert "prod-customer-data" in doc

    def test_document_contains_user(
        self, memory_store, sample_state
    ):
        doc = memory_store._build_document(
            sample_state
        )
        assert "svc_backup" in doc

    def test_document_contains_verdict(
        self, memory_store, sample_state
    ):
        doc = memory_store._build_document(
            sample_state
        )
        assert "DATA_EXFILTRATION" in doc

    def test_metadata_has_required_fields(
        self, memory_store, sample_state
    ):
        meta = memory_store._build_metadata(
            sample_state
        )
        required = [
            "event_host", "event_user",
            "severity", "verdict",
            "risk_score", "timestamp"
        ]
        for field in required:
            assert field in meta

    def test_metadata_risk_score_is_float(
        self, memory_store, sample_state
    ):
        meta = memory_store._build_metadata(
            sample_state
        )
        assert isinstance(meta["risk_score"], float)

    def test_empty_state_builds_document(
        self, memory_store
    ):
        doc = memory_store._build_document({})
        assert isinstance(doc, str)


# ============================================================
# CLEAR AND DELETE TESTS
# ============================================================

class TestClearAndDelete:

    def test_clear_all(
        self, memory_store, sample_state
    ):
        memory_store.store_investigation(
            sample_state,
            investigation_id="clear-001"
        )
        result = memory_store.clear_all()
        assert result is True

    def test_delete_investigation(
        self, memory_store, sample_state
    ):
        memory_store.store_investigation(
            sample_state,
            investigation_id="delete-001"
        )
        result = memory_store.delete_investigation(
            "delete-001"
        )
        assert result is True