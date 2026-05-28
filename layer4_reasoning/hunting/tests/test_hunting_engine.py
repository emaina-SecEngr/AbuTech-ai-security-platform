"""
Tests for Threat Hunting Hypothesis Engine
and KQL Repository
"""

import pytest
from layer5_interface.analytics.kql_repository\
    import KQLRepository, KQL_QUERIES
from layer4_reasoning.hunting.hypothesis_engine\
    import (
        HypothesisEngine,
        KEYWORD_TECHNIQUE_MAP,
        ACTOR_TECHNIQUE_MAP
    )


@pytest.fixture
def repo():
    return KQLRepository()


@pytest.fixture
def engine():
    return HypothesisEngine()


@pytest.fixture
def scattered_spider_advisory():
    return (
        "FS-ISAC Advisory AA-2026-001: "
        "Scattered Spider is actively targeting "
        "US financial institutions via MFA fatigue "
        "attacks against Okta and Entra ID. "
        "Threat actor generates rapid push "
        "notifications to overwhelm users. "
        "Known IOCs: 185.220.101.45, 45.142.100.10. "
        "Affected sectors: Banking and Finance."
    )


@pytest.fixture
def ransomware_advisory():
    return (
        "CISA Alert AA-2026-002: "
        "LockBit successor group actively "
        "conducting ransomware campaigns against "
        "financial institutions. "
        "Initial access via phishing emails "
        "with malicious attachments. "
        "Followed by data exfiltration "
        "before encryption. "
        "IOCs: evil-domain.com, malware.exe hash "
        "a1b2c3d4e5f6789012345678901234567890abcd"
    )


@pytest.fixture
def structured_intel():
    return {
        "threat_actor": "Scattered Spider",
        "technique": "MFA fatigue",
        "target": "Okta and Entra ID environments",
        "iocs": [
            "185.220.101.45",
            "45.142.100.10"
        ],
        "source": "FS-ISAC"
    }


@pytest.fixture
def batch_intel():
    return [
        {
            "threat_actor": "Scattered Spider",
            "technique": "MFA fatigue",
            "target": "Financial sector Okta",
            "source": "FS-ISAC"
        },
        {
            "threat_actor": "Unknown",
            "technique": "DNS tunneling",
            "target": "Corporate network",
            "source": "Internal"
        },
        {
            "threat_actor": "LockBit",
            "technique": "ransomware",
            "target": "Windows servers",
            "source": "CISA"
        }
    ]


# ============================================================
# KQL REPOSITORY TESTS
# ============================================================

class TestKQLRepository:

    def test_repo_initializes(self, repo):
        assert repo is not None

    def test_queries_loaded(self, repo):
        assert len(repo.get_all_queries()) > 0

    def test_has_minimum_queries(self, repo):
        assert len(repo.get_all_queries()) >= 10

    def test_get_query_by_id(self, repo):
        result = repo.get_query("KQL-001")
        assert isinstance(result, dict)
        assert result.get("query_id") == "KQL-001"

    def test_unknown_query_empty(self, repo):
        result = repo.get_query("KQL-999")
        assert result == {}

    def test_get_by_technique_t1621(self, repo):
        results = repo.get_by_technique("T1621")
        assert len(results) > 0
        for r in results:
            assert "T1621" in r.get(
                "technique_id", ""
            )

    def test_get_by_technique_t1530(self, repo):
        results = repo.get_by_technique("T1530")
        assert len(results) > 0

    def test_get_by_technique_t1110(self, repo):
        results = repo.get_by_technique("T1110")
        assert len(results) > 0

    def test_get_by_tactic(self, repo):
        results = repo.get_by_tactic(
            "Credential Access"
        )
        assert len(results) > 0

    def test_get_by_severity_critical(self, repo):
        results = repo.get_by_severity("CRITICAL")
        assert len(results) > 0

    def test_get_by_severity_high(self, repo):
        results = repo.get_by_severity("HIGH")
        assert len(results) > 0

    def test_get_by_data_source(self, repo):
        results = repo.get_by_data_source(
            "SigninLogs"
        )
        assert len(results) > 0

    def test_get_by_threat_actor(self, repo):
        results = repo.get_by_threat_actor(
            "Scattered Spider"
        )
        assert len(results) > 0

    def test_search_mfa(self, repo):
        results = repo.search("MFA")
        assert len(results) > 0

    def test_search_exfiltration(self, repo):
        results = repo.search("exfiltration")
        assert len(results) > 0

    def test_search_returns_list(self, repo):
        results = repo.search("unknown_xyz_999")
        assert isinstance(results, list)

    def test_financial_queries_returned(self, repo):
        results = repo.get_financial_sector_queries()
        assert len(results) >= 5

    def test_all_queries_have_kql(self, repo):
        for q in repo.get_all_queries():
            assert len(
                q.get("kql", "").strip()
            ) > 0

    def test_all_queries_have_technique(self, repo):
        for q in repo.get_all_queries():
            assert q.get("technique_id", "") != ""

    def test_all_queries_have_title(self, repo):
        for q in repo.get_all_queries():
            assert q.get("title", "") != ""

    def test_statistics_returns_dict(self, repo):
        stats = repo.get_statistics()
        assert isinstance(stats, dict)
        assert "total_queries" in stats
        assert "by_tactic" in stats
        assert "by_severity" in stats

    def test_kql_contains_where(self, repo):
        for q in repo.get_all_queries():
            assert "where" in q.get(
                "kql", ""
            ).lower()

    def test_kql_contains_summarize_or_project(
        self, repo
    ):
        for q in repo.get_all_queries():
            kql = q.get("kql", "").lower()
            assert (
                "summarize" in kql or
                "project" in kql
            )


# ============================================================
# HYPOTHESIS ENGINE TESTS
# ============================================================

class TestHypothesisEngine:

    def test_engine_initializes(self, engine):
        assert engine is not None

    def test_keyword_map_populated(self):
        assert len(KEYWORD_TECHNIQUE_MAP) > 0
        assert "mfa fatigue" in KEYWORD_TECHNIQUE_MAP
        assert "brute force" in KEYWORD_TECHNIQUE_MAP

    def test_actor_map_populated(self):
        assert len(ACTOR_TECHNIQUE_MAP) > 0
        assert "scattered spider" in (
            ACTOR_TECHNIQUE_MAP
        )

    def test_generate_from_text_returns_dict(
        self, engine, scattered_spider_advisory
    ):
        result = engine.generate_from_text(
            scattered_spider_advisory
        )
        assert isinstance(result, dict)

    def test_hypothesis_has_required_fields(
        self, engine, scattered_spider_advisory
    ):
        result = engine.generate_from_text(
            scattered_spider_advisory
        )
        required = [
            "hypothesis_id", "title",
            "threat_actor", "mitre_technique",
            "mitre_tactic", "confidence",
            "priority", "hypothesis_statement",
            "kql_query", "affected_systems",
            "iocs", "hunt_status"
        ]
        for field in required:
            assert field in result

    def test_scattered_spider_detected(
        self, engine, scattered_spider_advisory
    ):
        result = engine.generate_from_text(
            scattered_spider_advisory
        )
        assert "Scattered" in result.get(
            "threat_actor", ""
        )

    def test_mfa_technique_detected(
        self, engine, scattered_spider_advisory
    ):
        result = engine.generate_from_text(
            scattered_spider_advisory
        )
        assert result.get(
            "mitre_technique"
        ) == "T1621"

    def test_iocs_extracted(
        self, engine, scattered_spider_advisory
    ):
        result = engine.generate_from_text(
            scattered_spider_advisory
        )
        assert len(result.get("iocs", [])) > 0

    def test_kql_query_provided(
        self, engine, scattered_spider_advisory
    ):
        result = engine.generate_from_text(
            scattered_spider_advisory
        )
        assert len(
            result.get("kql_query", "").strip()
        ) > 0

    def test_hunt_status_pending(
        self, engine, scattered_spider_advisory
    ):
        result = engine.generate_from_text(
            scattered_spider_advisory
        )
        assert result.get(
            "hunt_status"
        ) == "PENDING"

    def test_confidence_between_0_and_1(
        self, engine, scattered_spider_advisory
    ):
        result = engine.generate_from_text(
            scattered_spider_advisory
        )
        confidence = result.get("confidence", 0)
        assert 0.0 <= confidence <= 1.0

    def test_financial_advisory_high_priority(
        self, engine, scattered_spider_advisory
    ):
        result = engine.generate_from_text(
            scattered_spider_advisory,
            source="FS-ISAC"
        )
        assert result.get("priority") in [
            "HIGH", "CRITICAL"
        ]

    def test_generate_from_intel(
        self, engine, structured_intel
    ):
        result = engine.generate_from_intel(
            structured_intel
        )
        assert isinstance(result, dict)
        assert result.get(
            "mitre_technique"
        ) == "T1621"

    def test_generate_from_intel_actor(
        self, engine, structured_intel
    ):
        result = engine.generate_from_intel(
            structured_intel
        )
        assert "Scattered" in result.get(
            "threat_actor", ""
        )

    def test_generate_batch(
        self, engine, batch_intel
    ):
        results = engine.generate_batch(batch_intel)
        assert isinstance(results, list)
        assert len(results) == len(batch_intel)

    def test_batch_all_have_technique(
        self, engine, batch_intel
    ):
        results = engine.generate_batch(batch_intel)
        for r in results:
            assert r.get("mitre_technique") != ""

    def test_hypothesis_stored(
        self, engine, scattered_spider_advisory
    ):
        result = engine.generate_from_text(
            scattered_spider_advisory
        )
        hyp_id = result.get("hypothesis_id")
        retrieved = engine.get_hypothesis(hyp_id)
        assert retrieved.get(
            "hypothesis_id"
        ) == hyp_id

    def test_get_pending_hypotheses(
        self, engine, scattered_spider_advisory
    ):
        engine.generate_from_text(
            scattered_spider_advisory
        )
        pending = engine.get_pending_hypotheses()
        assert len(pending) > 0

    def test_update_hunt_status(
        self, engine, scattered_spider_advisory
    ):
        result = engine.generate_from_text(
            scattered_spider_advisory
        )
        hyp_id = result.get("hypothesis_id")
        updated = engine.update_hunt_status(
            hyp_id,
            "CONFIRMED",
            "Found 3 affected accounts"
        )
        assert updated.get(
            "hunt_status"
        ) == "CONFIRMED"
        assert updated.get("convert_to_rule") is True

    def test_empty_text_safe(self, engine):
        result = engine.generate_from_text("")
        assert result == {}

    def test_empty_intel_safe(self, engine):
        result = engine.generate_from_intel({})
        assert result == {}

    def test_statistics_returns_dict(self, engine):
        engine.generate_from_text(
            "MFA fatigue attack against Okta"
        )
        stats = engine.get_statistics()
        assert isinstance(stats, dict)
        assert "total_hypotheses" in stats

    def test_ransomware_advisory(
        self, engine, ransomware_advisory
    ):
        result = engine.generate_from_text(
            ransomware_advisory
        )
        assert isinstance(result, dict)
        assert result.get(
            "hunt_status"
        ) == "PENDING"

    def test_hypothesis_id_unique(
        self, engine
    ):
        r1 = engine.generate_from_text(
            "MFA fatigue attack"
        )
        r2 = engine.generate_from_text(
            "Brute force attack"
        )
        assert r1.get("hypothesis_id") != (
            r2.get("hypothesis_id")
        )

    def test_hypothesis_statement_not_empty(
        self, engine, scattered_spider_advisory
    ):
        result = engine.generate_from_text(
            scattered_spider_advisory
        )
        assert len(
            result.get(
                "hypothesis_statement", ""
            )
        ) > 0

    def test_affected_systems_extracted(
        self, engine, scattered_spider_advisory
    ):
        result = engine.generate_from_text(
            scattered_spider_advisory
        )
        assert len(
            result.get("affected_systems", [])
        ) > 0

    def test_get_hypotheses_by_technique(
        self, engine
    ):
        engine.generate_from_intel({
            "technique": "T1621",
            "target": "Okta",
            "source": "test"
        })
        results = engine.get_hypotheses_by_technique(
            "T1621"
        )
        assert len(results) > 0