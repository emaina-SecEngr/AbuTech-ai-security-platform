"""
Tests for MITRE ATT&CK Enricher
"""

import pytest
from layer3_knowledge.enrichment.mitre_enricher import (
    MITREEnricher,
    SIGNAL_TO_TECHNIQUE,
    SOURCE_TO_TECHNIQUE
)


@pytest.fixture
def enricher():
    return MITREEnricher()


@pytest.fixture
def s3_exfil_event():
    """Classic S3 data exfiltration event"""
    return {
        "accessor_identity": "svc_backup",
        "accessor_type": "service_account",
        "data_store_name": "prod-customer-pci-data",
        "data_path": "customers/cards/export.csv",
        "data_classification": "PCI",
        "bytes_accessed": 524288000,
        "event_time": "2026-05-21T03:00:00Z",
        "source_ip": "185.220.101.45",
        "risk_score": 0.974,
        "risk_reasons": [
            "after_hours",
            "tor_ip",
            "large_transfer_500mb+",
            "high_value_endpoint"
        ],
        "source_system": "s3_normalizer"
    }


@pytest.fixture
def mfa_fatigue_event():
    """MFA fatigue attack event"""
    return {
        "accessor_identity": "john.smith@bofa.com",
        "accessor_type": "human",
        "data_store_name": "azure-ad",
        "data_path": "SigninLogs",
        "data_classification": "INTERNAL",
        "bytes_accessed": 0,
        "event_time": "2026-05-21T09:00:00Z",
        "source_ip": "185.220.101.45",
        "risk_score": 0.88,
        "risk_reasons": [
            "mfa_fatigue",
            "tor_ip",
            "impossible_travel"
        ],
        "source_system": "okta_normalizer"
    }


@pytest.fixture
def waf_sqli_event():
    """WAF SQL injection event"""
    return {
        "accessor_identity": "185.220.101.45",
        "accessor_type": "api_key",
        "data_store_name": "api.bofa.com",
        "data_path": "GET /api/customers/search",
        "data_classification": "PII",
        "bytes_accessed": 0,
        "event_time": "2026-05-21T03:00:00Z",
        "source_ip": "185.220.101.45",
        "risk_score": 0.85,
        "risk_reasons": [
            "sql_injection_pattern_detected",
            "waf_action:BLOCK",
            "high_value_endpoint_targeted"
        ],
        "source_system": "waf_aws"
    }


@pytest.fixture
def iac_public_s3_event():
    """IaC misconfiguration - public S3"""
    return {
        "accessor_identity": (
            "aws_s3_bucket.prod_customer_data"
        ),
        "accessor_type": "service_account",
        "data_store_name": "terraform/s3/main.tf",
        "data_path": "aws_s3_bucket.prod:1",
        "data_classification": "PCI",
        "bytes_accessed": 0,
        "event_time": "2026-05-21T09:00:00Z",
        "source_ip": "",
        "risk_score": 0.85,
        "risk_reasons": [
            "iac_severity:HIGH",
            "critical_check:CKV_AWS_20",
            "public_exposure_risk"
        ],
        "source_system": "iac_checkov"
    }


@pytest.fixture
def low_risk_event():
    """Normal low risk event"""
    return {
        "accessor_identity": "john.smith",
        "accessor_type": "human",
        "data_store_name": "dev-logs",
        "data_path": "GET /api/health",
        "data_classification": "INTERNAL",
        "bytes_accessed": 1024,
        "event_time": "2026-05-21T09:00:00Z",
        "source_ip": "10.0.0.1",
        "risk_score": 0.10,
        "risk_reasons": [],
        "source_system": "api_gateway_aws"
    }


@pytest.fixture
def batch_events(
    s3_exfil_event, mfa_fatigue_event,
    waf_sqli_event, iac_public_s3_event,
    low_risk_event
):
    return [
        s3_exfil_event, mfa_fatigue_event,
        waf_sqli_event, iac_public_s3_event,
        low_risk_event
    ]


# ============================================================
# INITIALIZATION TESTS
# ============================================================

class TestInitialization:

    def test_enricher_initializes(self, enricher):
        assert enricher is not None

    def test_database_loaded(self, enricher):
        stats = enricher.get_database_stats()
        assert stats["techniques_count"] > 0

    def test_signal_map_populated(self):
        assert len(SIGNAL_TO_TECHNIQUE) > 0
        assert "after_hours" in SIGNAL_TO_TECHNIQUE
        assert "tor_ip" in SIGNAL_TO_TECHNIQUE

    def test_source_map_populated(self):
        assert len(SOURCE_TO_TECHNIQUE) > 0
        assert "s3_normalizer" in SOURCE_TO_TECHNIQUE
        assert "waf_aws" in SOURCE_TO_TECHNIQUE

    def test_database_has_tactics(self, enricher):
        stats = enricher.get_database_stats()
        assert stats["tactics_count"] > 0

    def test_database_has_mitigations(self, enricher):
        stats = enricher.get_database_stats()
        assert stats["mitigations_count"] > 0


# ============================================================
# TECHNIQUE LOOKUP TESTS
# ============================================================

class TestTechniqueLookup:

    def test_get_technique_returns_dict(
        self, enricher
    ):
        result = enricher.get_technique("T1530")
        assert isinstance(result, dict)

    def test_t1530_data_cloud_storage(
        self, enricher
    ):
        tech = enricher.get_technique("T1530")
        assert tech.get("id") == "T1530"
        assert "Cloud Storage" in tech.get(
            "name", ""
        )

    def test_t1078_valid_accounts(self, enricher):
        tech = enricher.get_technique("T1078")
        assert tech.get("id") == "T1078"
        assert "Valid Accounts" in tech.get(
            "name", ""
        )

    def test_t1110_brute_force(self, enricher):
        tech = enricher.get_technique("T1110")
        assert tech.get("id") == "T1110"

    def test_t1621_mfa_fatigue(self, enricher):
        tech = enricher.get_technique("T1621")
        assert tech.get("id") == "T1621"

    def test_unknown_technique_empty(self, enricher):
        result = enricher.get_technique("T9999")
        assert result == {}

    def test_technique_has_mitigations(
        self, enricher
    ):
        tech = enricher.get_technique("T1530")
        assert "mitigations" in tech
        assert isinstance(tech["mitigations"], list)

    def test_technique_has_tactic_refs(
        self, enricher
    ):
        tech = enricher.get_technique("T1530")
        assert "tactic_refs" in tech
        assert len(tech["tactic_refs"]) > 0

    def test_technique_has_description(
        self, enricher
    ):
        tech = enricher.get_technique("T1530")
        assert len(tech.get("description", "")) > 0


# ============================================================
# MITIGATION TESTS
# ============================================================

class TestMitigations:

    def test_get_mitigations_returns_list(
        self, enricher
    ):
        result = enricher.get_mitigations("T1530")
        assert isinstance(result, list)

    def test_t1530_has_mitigations(self, enricher):
        mitigations = enricher.get_mitigations(
            "T1530"
        )
        assert len(mitigations) > 0

    def test_mitigation_has_id(self, enricher):
        mitigations = enricher.get_mitigations(
            "T1078"
        )
        if mitigations:
            assert "id" in mitigations[0]
            assert mitigations[0]["id"].startswith("M")

    def test_mitigation_has_name(self, enricher):
        mitigations = enricher.get_mitigations(
            "T1078"
        )
        if mitigations:
            assert "name" in mitigations[0]
            assert len(
                mitigations[0]["name"]
            ) > 0


# ============================================================
# EVENT MAPPING TESTS
# ============================================================

class TestEventMapping:

    def test_map_event_returns_dict(
        self, enricher, s3_exfil_event
    ):
        result = enricher.map_event(s3_exfil_event)
        assert isinstance(result, dict)

    def test_map_event_has_techniques(
        self, enricher, s3_exfil_event
    ):
        result = enricher.map_event(s3_exfil_event)
        assert "techniques" in result
        assert len(result["techniques"]) > 0

    def test_s3_exfil_maps_to_t1530(
        self, enricher, s3_exfil_event
    ):
        result = enricher.map_event(s3_exfil_event)
        technique_ids = result.get(
            "technique_ids", []
        )
        assert "T1530" in technique_ids

    def test_mfa_fatigue_maps_to_t1621(
        self, enricher, mfa_fatigue_event
    ):
        result = enricher.map_event(
            mfa_fatigue_event
        )
        technique_ids = result.get(
            "technique_ids", []
        )
        assert "T1621" in technique_ids

    def test_sqli_maps_to_t1190(
        self, enricher, waf_sqli_event
    ):
        result = enricher.map_event(waf_sqli_event)
        technique_ids = result.get(
            "technique_ids", []
        )
        assert "T1190" in technique_ids

    def test_iac_public_maps_to_t1530(
        self, enricher, iac_public_s3_event
    ):
        result = enricher.map_event(
            iac_public_s3_event
        )
        technique_ids = result.get(
            "technique_ids", []
        )
        assert "T1530" in technique_ids

    def test_map_has_primary_tactic(
        self, enricher, s3_exfil_event
    ):
        result = enricher.map_event(s3_exfil_event)
        assert "primary_tactic" in result

    def test_map_has_mitigations(
        self, enricher, s3_exfil_event
    ):
        result = enricher.map_event(s3_exfil_event)
        assert "all_mitigations" in result

    def test_map_has_procedures(
        self, enricher, s3_exfil_event
    ):
        result = enricher.map_event(s3_exfil_event)
        assert "procedures" in result
        assert len(result["procedures"]) > 0

    def test_map_has_confidence(
        self, enricher, s3_exfil_event
    ):
        result = enricher.map_event(s3_exfil_event)
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_low_risk_still_maps(
        self, enricher, low_risk_event
    ):
        result = enricher.map_event(low_risk_event)
        assert isinstance(result, dict)
        assert "techniques" in result

    def test_empty_event_safe(self, enricher):
        result = enricher.map_event({})
        assert isinstance(result, dict)


# ============================================================
# ENRICH EVENT TESTS
# ============================================================

class TestEnrichEvent:

    def test_enrich_returns_dict(
        self, enricher, s3_exfil_event
    ):
        result = enricher.enrich_event(
            s3_exfil_event
        )
        assert isinstance(result, dict)

    def test_enrich_adds_mitre_context(
        self, enricher, s3_exfil_event
    ):
        result = enricher.enrich_event(
            s3_exfil_event
        )
        assert "mitre_context" in result

    def test_enrich_adds_technique_ids(
        self, enricher, s3_exfil_event
    ):
        result = enricher.enrich_event(
            s3_exfil_event
        )
        assert "mitre_techniques" in result
        assert isinstance(
            result["mitre_techniques"], list
        )

    def test_enrich_adds_tactic_name(
        self, enricher, s3_exfil_event
    ):
        result = enricher.enrich_event(
            s3_exfil_event
        )
        assert "mitre_tactic" in result

    def test_enrich_preserves_original_fields(
        self, enricher, s3_exfil_event
    ):
        result = enricher.enrich_event(
            s3_exfil_event
        )
        assert result["accessor_identity"] == (
            "svc_backup"
        )
        assert result["risk_score"] == 0.974

    def test_enrich_none_safe(self, enricher):
        result = enricher.enrich_event(None)
        assert result is None


# ============================================================
# COVERAGE REPORT TESTS
# ============================================================

class TestCoverageReport:

    def test_coverage_returns_dict(
        self, enricher, batch_events
    ):
        enriched = [
            enricher.enrich_event(e)
            for e in batch_events
        ]
        report = enricher.get_coverage_report(
            enriched
        )
        assert isinstance(report, dict)

    def test_coverage_has_required_fields(
        self, enricher, batch_events
    ):
        enriched = [
            enricher.enrich_event(e)
            for e in batch_events
        ]
        report = enricher.get_coverage_report(
            enriched
        )
        required = [
            "total_events",
            "techniques_detected",
            "coverage_percentage",
            "tactics_covered"
        ]
        for field in required:
            assert field in report

    def test_coverage_event_count(
        self, enricher, batch_events
    ):
        enriched = [
            enricher.enrich_event(e)
            for e in batch_events
        ]
        report = enricher.get_coverage_report(
            enriched
        )
        assert report["total_events"] == len(
            batch_events
        )

    def test_coverage_detects_techniques(
        self, enricher, batch_events
    ):
        enriched = [
            enricher.enrich_event(e)
            for e in batch_events
        ]
        report = enricher.get_coverage_report(
            enriched
        )
        assert report["techniques_detected"] > 0

    def test_empty_events_safe(self, enricher):
        report = enricher.get_coverage_report([])
        assert report["total_events"] == 0


# ============================================================
# SEARCH AND UTILITY TESTS
# ============================================================

class TestSearchAndUtility:

    def test_search_techniques_returns_list(
        self, enricher
    ):
        results = enricher.search_techniques("DNS")
        assert isinstance(results, list)

    def test_search_finds_relevant(self, enricher):
        results = enricher.search_techniques(
            "cloud storage"
        )
        assert len(results) > 0

    def test_financial_techniques_returns_list(
        self, enricher
    ):
        results = (
            enricher.get_financial_sector_techniques()
        )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_financial_includes_t1530(
        self, enricher
    ):
        results = (
            enricher.get_financial_sector_techniques()
        )
        ids = [t.get("id") for t in results]
        assert "T1530" in ids

    def test_financial_includes_t1621(
        self, enricher
    ):
        results = (
            enricher.get_financial_sector_techniques()
        )
        ids = [t.get("id") for t in results]
        assert "T1621" in ids

    def test_get_tactic_returns_dict(
        self, enricher
    ):
        result = enricher.get_tactic("TA0010")
        assert isinstance(result, dict)

    def test_exfiltration_tactic(self, enricher):
        tactic = enricher.get_tactic("TA0010")
        assert tactic.get("name") == "Exfiltration"

    def test_database_stats_returns_dict(
        self, enricher
    ):
        stats = enricher.get_database_stats()
        assert isinstance(stats, dict)
        assert "techniques_count" in stats
        assert "tactics_count" in stats