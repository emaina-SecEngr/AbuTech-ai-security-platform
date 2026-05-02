"""
Layer 3 — CISA Ingester Tests
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from layer3_knowledge.graph.security_graph import (
    SecurityKnowledgeGraph
)
from layer3_knowledge.enrichment.stix_processor import (
    STIXProcessor
)
from layer3_knowledge.enrichment.cisa_ingester import (
    CISAIngester
)

SAMPLE_ADVISORY_TEXT = """
--- Page 1 ---
CISA Advisory AA24-131A
Royal Ransomware Indicators of Compromise

SUMMARY
The FBI and CISA are releasing this advisory
about Royal ransomware.

--- Page 2 ---
TECHNICAL DETAILS

Threat Actor: Royal Ransomware Group
Malware Family: Royal, Zeon

IP Addresses:
185.220.101.45 - Primary C2 server
185.220.101.46 - Secondary C2 server

IP Ranges:
185.220.0.0/16 - Tor exit node range

Domains:
xjf8k2mp.duckdns.org - DGA domain for C2

URL:
http://evil.com/royal_payload.exe

--- Page 3 ---
INDICATORS OF COMPROMISE

MITRE ATT&CK Techniques:
T1486 - Data Encrypted for Impact
T1059.001 - PowerShell
"""

SAMPLE_TEXT_WITH_HASH = (
    "a3f8d2c1e5b7a9f0d4c2e6b8a1f3d5c7"
    "e9b2a4f6d8c0e2a4f6b8d0c2e4a6f8b0"
)


class TestPDFExtraction:

    def setup_method(self, method):
        import uuid
        self.graph = SecurityKnowledgeGraph()
        self.processor = STIXProcessor(
            self.graph,
            db_path=(
                f".test_cache/cisa_{uuid.uuid4().hex}.db"
            )
        )
        self.ingester = CISAIngester(
            self.processor,
            anthropic_api_key=""
        )

    def test_extract_text_from_nonexistent_pdf(self):
        """Nonexistent PDF returns empty string"""
        result = self.ingester.extract_text_only(
            "nonexistent.pdf"
        )
        assert result == ""

    def test_process_nonexistent_pdf_returns_error(self):
        """Nonexistent PDF returns error result"""
        result = self.ingester.process_pdf(
            "nonexistent.pdf"
        )
        assert result["success"] is False
        assert result["error"] == "not_found"


class TestRuleBasedExtraction:

    def setup_method(self, method):
        import uuid
        self.graph = SecurityKnowledgeGraph()
        self.processor = STIXProcessor(
            self.graph,
            db_path=(
                f".test_cache/rule_{uuid.uuid4().hex}.db"
            )
        )
        self.ingester = CISAIngester(
            self.processor,
            anthropic_api_key=""
        )

    def test_extract_ip_addresses(self):
        """IP addresses extracted from advisory text"""
        extracted = self.ingester._rule_based_extraction(
            SAMPLE_ADVISORY_TEXT
        )
        ip_values = [
            e["value"]
            for e in
            extracted["indicators"]["ip_addresses"]
        ]
        assert "185.220.101.45" in ip_values
        assert "185.220.101.46" in ip_values

    def test_extract_cidr_ranges(self):
        """CIDR ranges extracted from advisory text"""
        extracted = self.ingester._rule_based_extraction(
            SAMPLE_ADVISORY_TEXT
        )
        cidrs = [
            e["cidr"]
            for e in
            extracted["indicators"]["ip_ranges"]
        ]
        assert "185.220.0.0/16" in cidrs

    def test_extract_domains(self):
        """Domain names extracted from advisory text"""
        extracted = self.ingester._rule_based_extraction(
            SAMPLE_ADVISORY_TEXT
        )
        domains = [
            e["value"]
            for e in
            extracted["indicators"]["domains"]
        ]
        assert any("duckdns.org" in d for d in domains)

    def test_extract_sha256_hashes(self):
        """SHA256 hashes extracted from text"""
        extracted = self.ingester._rule_based_extraction(
            SAMPLE_TEXT_WITH_HASH
        )
        hashes = extracted["indicators"]["file_hashes"]
        assert len(hashes) > 0
        sha256_values = [
            h.get("sha256", "") for h in hashes
        ]
        assert any(len(h) == 64 for h in sha256_values)

    def test_extract_attack_techniques(self):
        """ATT&CK technique IDs extracted"""
        extracted = self.ingester._rule_based_extraction(
            SAMPLE_ADVISORY_TEXT
        )
        technique_ids = [
            t["technique_id"]
            for t in extracted["attack_techniques"]
        ]
        assert "T1486" in technique_ids
        assert "T1059.001" in technique_ids

    def test_excluded_domains_not_extracted(self):
        """Known legitimate domains not extracted"""
        text = (
            SAMPLE_ADVISORY_TEXT +
            "\nvisit cisa.gov\nalso microsoft.com"
        )
        extracted = self.ingester._rule_based_extraction(
            text
        )
        domains = [
            e["value"]
            for e in
            extracted["indicators"]["domains"]
        ]
        assert "cisa.gov" not in domains
        assert "microsoft.com" not in domains


class TestCIDRRangeHandling:

    def setup_method(self, method):
        import uuid
        self.graph = SecurityKnowledgeGraph()
        self.processor = STIXProcessor(
            self.graph,
            db_path=(
                f".test_cache/cidr_{uuid.uuid4().hex}.db"
            )
        )
        self.ingester = CISAIngester(
            self.processor,
            anthropic_api_key=""
        )

    def test_store_valid_cidr_range(self):
        """Valid CIDR range stored successfully"""
        self.ingester._store_ip_range(
            "185.220.0.0/16", "Tor", 85, "CISA"
        )
        assert "185.220.0.0/16" in self.ingester.ip_ranges

    def test_invalid_cidr_not_stored(self):
        """Invalid CIDR rejected gracefully"""
        self.ingester._store_ip_range(
            "not_a_cidr", "Test", 80, "test"
        )
        assert "not_a_cidr" not in self.ingester.ip_ranges

    def test_ip_matches_stored_range(self):
        """IP in stored range correctly identified"""
        self.ingester._store_ip_range(
            "185.220.0.0/16", "Tor", 85, "CISA"
        )
        result = self.ingester.check_ip_against_ranges(
            "185.220.101.45"
        )
        assert result is not None
        assert result["matched_range"] == "185.220.0.0/16"
        assert result["risk_score"] == 0.85

    def test_ip_outside_range_returns_none(self):
        """IP outside stored range returns None"""
        self.ingester._store_ip_range(
            "10.0.0.0/8", "Private", 50, "test"
        )
        result = self.ingester.check_ip_against_ranges(
            "185.220.101.45"
        )
        assert result is None

    def test_multiple_ranges_checked(self):
        """IP checked against all stored ranges"""
        self.ingester._store_ip_range(
            "10.0.0.0/8", "Private", 30, "test"
        )
        self.ingester._store_ip_range(
            "185.220.0.0/16", "Tor", 85, "CISA"
        )
        result = self.ingester.check_ip_against_ranges(
            "185.220.101.45"
        )
        assert result is not None
        assert result["matched_range"] == "185.220.0.0/16"

    def test_invalid_ip_returns_none(self):
        """Invalid IP address returns None"""
        result = self.ingester.check_ip_against_ranges(
            "not_an_ip"
        )
        assert result is None

    def test_cidr_confidence_to_risk_score(self):
        """Confidence 85 → risk score 0.85"""
        self.ingester._store_ip_range(
            "192.168.0.0/16", "Test", 85, "test"
        )
        intel = self.ingester.ip_ranges["192.168.0.0/16"]
        assert intel["risk_score"] == 0.85


class TestSTIXConversion:

    def setup_method(self, method):
        import uuid
        self.graph = SecurityKnowledgeGraph()
        self.processor = STIXProcessor(
            self.graph,
            db_path=(
                f".test_cache/stix_{uuid.uuid4().hex}.db"
            )
        )
        self.ingester = CISAIngester(
            self.processor,
            anthropic_api_key=""
        )

    def _make_extracted(self, ips=None, domains=None):
        return {
            "advisory_title": "Test",
            "advisory_id": "",
            "threat_actors": [],
            "malware_families": [],
            "attack_techniques": [],
            "indicators": {
                "ip_addresses": ips or [],
                "ip_ranges": [],
                "domains": domains or [],
                "urls": [],
                "file_hashes": []
            },
            "summary": ""
        }

    def test_stix_bundle_created(self):
        """Extracted IOCs converted to STIX bundle"""
        extracted = self._make_extracted(
            ips=[{
                "value": "1.2.3.4",
                "context": "C2",
                "confidence": 90
            }],
            domains=[{
                "value": "evil.com",
                "context": "Phishing",
                "confidence": 85
            }]
        )
        bundle = self.ingester._convert_to_stix(
            extracted, "test"
        )
        assert bundle["type"] == "bundle"
        assert len(bundle["objects"]) >= 2

    def test_ip_indicator_correct_pattern(self):
        """IP IOC creates correct STIX pattern"""
        extracted = self._make_extracted(
            ips=[{
                "value": "185.220.101.45",
                "context": "C2",
                "confidence": 90
            }]
        )
        bundle = self.ingester._convert_to_stix(
            extracted, "test"
        )
        patterns = [
            obj.get("pattern", "")
            for obj in bundle["objects"]
            if obj.get("type") == "indicator"
        ]
        assert any(
            "185.220.101.45" in p for p in patterns
        )

    def test_stix_bundle_ingested_by_processor(self):
        """STIX bundle ingested adds graph nodes"""
        extracted = self.ingester._rule_based_extraction(
            SAMPLE_ADVISORY_TEXT
        )
        bundle = self.ingester._convert_to_stix(
            extracted, "test"
        )
        results = self.processor.process_bundle(
            bundle, "test"
        )
        assert results["indicators_processed"] >= 1

    def test_full_pipeline_without_claude(self):
        """Full pipeline from text to graph"""
        extracted = self.ingester._rule_based_extraction(
            SAMPLE_ADVISORY_TEXT
        )
        bundle = self.ingester._convert_to_stix(
            extracted, "test"
        )
        self.processor.process_bundle(bundle, "test")
        stats = self.graph.get_statistics()
        assert stats["total_nodes"] >= 1


class TestIngesterStatistics:

    def setup_method(self, method):
        import uuid
        self.graph = SecurityKnowledgeGraph()
        self.processor = STIXProcessor(
            self.graph,
            db_path=(
                f".test_cache/stat_{uuid.uuid4().hex}.db"
            )
        )
        self.ingester = CISAIngester(
            self.processor,
            anthropic_api_key=""
        )

    def test_initial_statistics_zero(self):
        """Statistics start at zero"""
        stats = self.ingester.get_statistics()
        assert stats["pdfs_processed"] == 0
        assert stats["iocs_extracted"] == 0
        assert stats["ip_ranges_stored"] == 0

    def test_ip_range_count_tracked(self):
        """IP range storage count tracked"""
        self.ingester._store_ip_range(
            "185.220.0.0/16", "Test", 80, "test"
        )
        self.ingester._store_ip_range(
            "10.0.0.0/8", "Private", 30, "test"
        )
        stats = self.ingester.get_statistics()
        assert stats["ip_ranges_stored"] == 2

    def test_claude_response_parsing_valid_json(self):
        """Valid JSON correctly parsed"""
        valid_json = json.dumps({
            "advisory_title": "Test",
            "advisory_id": "",
            "published_date": "",
            "threat_actors": [],
            "malware_families": [],
            "attack_techniques": [],
            "indicators": {
                "ip_addresses": [],
                "ip_ranges": [],
                "domains": [],
                "urls": [],
                "file_hashes": []
            },
            "summary": "Test"
        })
        result = self.ingester._parse_claude_response(
            valid_json
        )
        assert result is not None
        assert result["advisory_title"] == "Test"

    def test_claude_response_markdown_json(self):
        """Markdown-wrapped JSON correctly parsed"""
        markdown_json = (
            "```json\n"
            '{"advisory_title": "Test", '
            '"advisory_id": "", '
            '"published_date": "", '
            '"threat_actors": [], '
            '"malware_families": [], '
            '"attack_techniques": [], '
            '"indicators": {'
            '"ip_addresses": [], '
            '"ip_ranges": [], '
            '"domains": [], '
            '"urls": [], '
            '"file_hashes": []}, '
            '"summary": "Test"}\n'
            "```"
        )
        result = self.ingester._parse_claude_response(
            markdown_json
        )
        assert result is not None

    def test_invalid_json_returns_none(self):
        """Invalid JSON returns None gracefully"""
        result = self.ingester._parse_claude_response(
            "not json at all"
        )
        assert result is None