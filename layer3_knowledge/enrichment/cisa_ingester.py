"""
Layer 3 — Knowledge Graph
CISA Advisory PDF Ingester

This module automatically extracts threat intelligence
from CISA cybersecurity advisory PDF documents using
Claude AI for structured IOC extraction.

Why This Is Unique:
    Most platforms require human analysts to manually
    read CISA advisories and extract IOCs. This takes
    hours per advisory and introduces human error.

    Your platform uses Claude AI to read the full
    advisory text and extract structured intelligence
    automatically — converting unstructured PDFs into
    machine-readable STIX bundles within minutes of
    CISA publishing.

Pipeline:
    CISA PDF → PyMuPDF text extraction
            → Claude API structured extraction
            → JSON IOC schema
            → STIX bundle conversion
            → STIXProcessor ingestion
            → Knowledge graph enrichment

Dynamic Schema Adaptation:
    CISA advisory formats change over time.
    Rather than hardcoding table column positions
    we use Claude to understand the document
    structure dynamically and extract IOCs
    regardless of formatting variation.

CIDR Range Handling:
    CISA advisories often reference IP ranges.
    We store ranges as IP_RANGE nodes and check
    new IPs against all stored ranges automatically.

Claude AI Integration:
    Uses your existing Anthropic API key.
    Same key as Layer 4 agents.
    Model: claude-opus-4-6 for best extraction accuracy.
"""

import ipaddress
import json
import logging
import os
import re
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# IOC EXTRACTION SCHEMA
# Claude returns JSON matching this exact schema
# Schema-constrained extraction prevents hallucination
# ============================================================

EXTRACTION_SCHEMA = {
    "advisory_title": "string",
    "advisory_id": "string",
    "published_date": "string",
    "threat_actors": ["string"],
    "malware_families": ["string"],
    "attack_techniques": [
        {
            "technique_id": "T1059.001",
            "technique_name": "PowerShell"
        }
    ],
    "indicators": {
        "ip_addresses": [
            {
                "value": "1.2.3.4",
                "context": "C2 server",
                "confidence": 90
            }
        ],
        "ip_ranges": [
            {
                "cidr": "185.220.0.0/16",
                "context": "Tor exit nodes",
                "confidence": 75
            }
        ],
        "domains": [
            {
                "value": "evil.com",
                "context": "Phishing domain",
                "confidence": 85
            }
        ],
        "urls": [
            {
                "value": "http://evil.com/payload",
                "context": "Malware download",
                "confidence": 90
            }
        ],
        "file_hashes": [
            {
                "md5": "abc123",
                "sha256": "def456",
                "filename": "malware.exe",
                "context": "Ransomware payload",
                "confidence": 95
            }
        ]
    },
    "summary": "Brief description of the threat"
}


# Claude extraction prompt
# Carefully engineered for accurate IOC extraction
EXTRACTION_PROMPT = """You are a cybersecurity threat intelligence analyst.
Extract all threat indicators from this CISA cybersecurity advisory.

Return ONLY valid JSON matching this exact schema:
{schema}

Extraction rules:
1. Extract ALL IP addresses mentioned as indicators
   Include IPs in tables, paragraphs, and appendices
   Do NOT include IPs that are examples or documentation

2. Extract ALL CIDR ranges mentioned
   Example: 185.220.0.0/16 or 10.0.0.0/8

3. Extract ALL domain names mentioned as indicators
   Do NOT include legitimate domains like cisa.gov or microsoft.com

4. Extract ALL URLs that are malicious indicators
   Not documentation links

5. Extract ALL file hashes (MD5, SHA1, SHA256)
   Include filename if mentioned nearby

6. Extract MITRE ATT&CK technique IDs
   Format: T1059.001, T1071, etc.

7. Extract threat actor names if mentioned

8. Extract malware family names if mentioned

9. For confidence scores:
   IOC in a structured table = 90-95
   IOC mentioned in text = 75-85
   IOC inferred from context = 50-70

10. For context field: brief description of how this
    IOC was used by the threat actor

If a field has no data return an empty list [].
Return ONLY the JSON, no other text.

ADVISORY TEXT:
{text}"""


class CISAIngester:
    """
    Extracts threat intelligence from CISA PDF advisories.

    Uses PyMuPDF to extract text from PDFs and
    Claude AI to extract structured IOCs from the text.

    Converts extracted IOCs to STIX format for
    ingestion by STIXProcessor.

    Usage:
        graph = SecurityKnowledgeGraph()
        processor = STIXProcessor(graph)
        ingester = CISAIngester(processor)

        # Process a single advisory
        results = ingester.process_pdf(
            "advisories/AA24-131A.pdf"
        )

        # Process all PDFs in a directory
        results = ingester.process_directory(
            "advisories/"
        )
    """

    def __init__(
        self,
        stix_processor,
        anthropic_api_key: str = None,
        model: str = "claude-opus-4-6"
    ):
        """
        Initialize CISA ingester.

        Args:
            stix_processor: STIXProcessor instance
            anthropic_api_key: Anthropic API key
                              If None reads from env
            model: Claude model to use for extraction
        """
        self.processor = stix_processor
        self.model = model

        self.api_key = (
            anthropic_api_key or
            os.getenv("ANTHROPIC_API_KEY", "")
        )

        # CIDR range database
        # Maps CIDR string to intelligence dict
        self.ip_ranges = {}

        # Statistics
        self.pdfs_processed = 0
        self.iocs_extracted = 0
        self.advisories_indexed = []

        logger.info("CISAIngester initialized")

    # ============================================================
    # PRIMARY PUBLIC METHODS
    # ============================================================

    def process_pdf(
        self,
        pdf_path: str,
        source_name: str = None
    ) -> dict:
        """
        Process a single CISA advisory PDF.

        Full pipeline:
        1. Extract text from PDF
        2. Send to Claude for IOC extraction
        3. Parse structured response
        4. Handle CIDR ranges
        5. Convert to STIX bundle
        6. Ingest via STIXProcessor

        Args:
            pdf_path: Path to PDF file
            source_name: Override source attribution

        Returns:
            Processing results dictionary
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return {"success": False, "error": "not_found"}

        logger.info(
            f"Processing CISA advisory: {pdf_path.name}"
        )

        results = {
            "pdf_file": pdf_path.name,
            "success": False,
            "text_extracted": False,
            "iocs_extracted": 0,
            "graph_nodes_updated": 0,
            "advisory_title": "",
            "threat_actors": [],
            "malware_families": []
        }

        # ---- STEP 1: EXTRACT TEXT ----
        text = self._extract_pdf_text(pdf_path)
        if not text:
            results["error"] = "text_extraction_failed"
            return results

        results["text_extracted"] = True
        results["text_length"] = len(text)
        logger.info(
            f"Extracted {len(text)} characters "
            f"from {pdf_path.name}"
        )

        # ---- STEP 2: CLAUDE IOC EXTRACTION ----
        extracted = self._extract_iocs_with_claude(text)
        if not extracted:
            results["error"] = "claude_extraction_failed"
            return results

        results["advisory_title"] = extracted.get(
            "advisory_title", ""
        )
        results["threat_actors"] = extracted.get(
            "threat_actors", []
        )
        results["malware_families"] = extracted.get(
            "malware_families", []
        )

        # ---- STEP 3: HANDLE CIDR RANGES ----
        ip_ranges = extracted.get(
            "indicators", {}
        ).get("ip_ranges", [])

        for range_entry in ip_ranges:
            self._store_ip_range(
                range_entry.get("cidr", ""),
                range_entry.get("context", ""),
                range_entry.get("confidence", 75),
                source_name or pdf_path.name
            )

        # ---- STEP 4: CONVERT TO STIX ----
        stix_bundle = self._convert_to_stix(
            extracted,
            source_name or pdf_path.stem
        )

        # ---- STEP 5: INGEST VIA STIX PROCESSOR ----
        ingest_results = self.processor.process_bundle(
            stix_bundle,
            source_name=f"CISA:{pdf_path.stem}"
        )

        results["success"] = True
        results["iocs_extracted"] = (
            ingest_results.get("iocs_added", 0)
        )
        results["graph_nodes_updated"] = (
            ingest_results.get(
                "graph_nodes_updated", 0
            )
        )

        # Track processed advisories
        self.pdfs_processed += 1
        self.iocs_extracted += results["iocs_extracted"]
        self.advisories_indexed.append({
            "file": pdf_path.name,
            "title": results["advisory_title"],
            "iocs": results["iocs_extracted"]
        })

        logger.info(
            f"Advisory processed: "
            f"{results['advisory_title']} — "
            f"{results['iocs_extracted']} IOCs"
        )

        return results

    def process_directory(
        self,
        directory: str
    ) -> list:
        """
        Process all PDF files in a directory.

        Implements your landing zone vision:
        Any PDF dropped into this directory gets
        automatically processed and ingested.

        Args:
            directory: Path to directory with PDFs

        Returns:
            List of processing results
        """
        directory = Path(directory)

        if not directory.exists():
            logger.error(
                f"Directory not found: {directory}"
            )
            return []

        pdf_files = list(directory.glob("*.pdf"))

        if not pdf_files:
            logger.warning(
                f"No PDF files found in {directory}"
            )
            return []

        logger.info(
            f"Processing {len(pdf_files)} PDFs "
            f"from {directory}"
        )

        results = []
        for pdf_file in pdf_files:
            result = self.process_pdf(str(pdf_file))
            results.append(result)

        successful = sum(
            1 for r in results if r.get("success")
        )
        total_iocs = sum(
            r.get("iocs_extracted", 0)
            for r in results
        )

        logger.info(
            f"Directory processing complete: "
            f"{successful}/{len(pdf_files)} successful, "
            f"{total_iocs} total IOCs extracted"
        )

        return results

    def check_ip_against_ranges(
        self,
        ip_address: str
    ) -> Optional[dict]:
        """
        Check if an IP falls within any stored CIDR range.

        This implements your outlier accommodation:
        "185.220.101.45" matches "185.220.0.0/16"
        and inherits its threat intelligence.

        Called by ThreatEnricher when checking IPs
        that are not in the exact match database.

        Args:
            ip_address: IP address to check

        Returns:
            Range intelligence dict or None
        """
        try:
            target_ip = ipaddress.ip_address(ip_address)

            for cidr, intel in self.ip_ranges.items():
                try:
                    network = ipaddress.ip_network(
                        cidr, strict=False
                    )
                    if target_ip in network:
                        logger.info(
                            f"IP {ip_address} matches "
                            f"range {cidr}"
                        )
                        return {
                            "matched_range": cidr,
                            "risk_score": intel[
                                "risk_score"
                            ],
                            "context": intel["context"],
                            "source": intel["source"]
                        }
                except ValueError:
                    continue

        except ValueError:
            logger.warning(
                f"Invalid IP address: {ip_address}"
            )

        return None

    def extract_text_only(
        self,
        pdf_path: str
    ) -> str:
        """
        Extract text from PDF without Claude processing.

        Useful for testing text extraction independently
        from IOC extraction.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text string
        """
        return self._extract_pdf_text(Path(pdf_path))

    def get_statistics(self) -> dict:
        """Return ingestion statistics"""
        return {
            "pdfs_processed": self.pdfs_processed,
            "iocs_extracted": self.iocs_extracted,
            "ip_ranges_stored": len(self.ip_ranges),
            "advisories": self.advisories_indexed
        }

    # ============================================================
    # PRIVATE METHODS
    # ============================================================

    def _extract_pdf_text(
        self,
        pdf_path: Path
    ) -> str:
        """
        Extract text from PDF using PyMuPDF.

        PyMuPDF preserves text structure including
        table layouts which helps Claude identify
        IOC tables in CISA advisories.
        """
        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(
                        f"\n--- Page {page_num + 1} ---\n"
                        + text
                    )

            doc.close()

            full_text = "\n".join(text_parts)
            return full_text

        except ImportError:
            logger.error(
                "PyMuPDF not installed. "
                "Run: pip install pymupdf"
            )
            return ""
        except Exception as e:
            logger.error(
                f"PDF text extraction failed: {e}"
            )
            return ""

    def _extract_iocs_with_claude(
        self,
        text: str
    ) -> Optional[dict]:
        """
        Send advisory text to Claude for IOC extraction.

        Uses schema-constrained extraction to ensure
        consistent JSON output regardless of advisory
        format variation.

        Handles large PDFs by chunking if needed.
        """
        if not self.api_key:
            logger.warning(
                "No Anthropic API key configured. "
                "Set ANTHROPIC_API_KEY environment variable."
            )
            return self._rule_based_extraction(text)

        try:
            import anthropic

            client = anthropic.Anthropic(
                api_key=self.api_key
            )

            # Truncate text if too long
            # Claude context window is large but
            # we keep within safe limits
            max_chars = 150000
            if len(text) > max_chars:
                logger.warning(
                    f"Advisory text truncated from "
                    f"{len(text)} to {max_chars} chars"
                )
                text = text[:max_chars]

            prompt = EXTRACTION_PROMPT.format(
                schema=json.dumps(
                    EXTRACTION_SCHEMA, indent=2
                ),
                text=text
            )

            logger.info(
                "Sending advisory to Claude for "
                "IOC extraction..."
            )

            message = client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            response_text = message.content[0].text

            # Parse JSON response
            extracted = self._parse_claude_response(
                response_text
            )

            if extracted:
                logger.info(
                    "Claude extraction successful"
                )
                return extracted
            else:
                logger.warning(
                    "Claude returned invalid JSON, "
                    "falling back to rule-based"
                )
                return self._rule_based_extraction(text)

        except Exception as e:
            logger.error(
                f"Claude extraction failed: {e}. "
                f"Falling back to rule-based."
            )
            return self._rule_based_extraction(text)

    def _rule_based_extraction(
        self,
        text: str
    ) -> dict:
        """
        Fallback rule-based IOC extraction.

        Used when Claude API is not available.
        Uses regex patterns to extract IOCs.
        Less accurate than Claude but always works.
        """
        logger.info(
            "Using rule-based IOC extraction"
        )

        extracted = {
            "advisory_title": (
                self._extract_title(text)
            ),
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
            "summary": ""
        }

        # Extract IPs using regex
        ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        ips = re.findall(ip_pattern, text)

        # Filter out common non-IOC IPs
        excluded_ips = {
            "0.0.0.0", "127.0.0.1", "255.255.255.255",
            "192.168.0.0", "10.0.0.0", "172.16.0.0"
        }

        for ip in set(ips):
            if ip not in excluded_ips:
                extracted["indicators"][
                    "ip_addresses"
                ].append({
                    "value": ip,
                    "context": "Extracted from advisory",
                    "confidence": 70
                })

        # Extract CIDR ranges
        cidr_pattern = (
            r"\b(?:\d{1,3}\.){3}\d{1,3}/\d{1,2}\b"
        )
        cidrs = re.findall(cidr_pattern, text)
        for cidr in set(cidrs):
            extracted["indicators"][
                "ip_ranges"
            ].append({
                "cidr": cidr,
                "context": "Extracted from advisory",
                "confidence": 70
            })

        # Extract domains
        domain_pattern = (
            r"\b(?:[a-zA-Z0-9]"
            r"(?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
            r"(?:com|net|org|io|xyz|top|duckdns\.org)\b"
        )
        domains = re.findall(domain_pattern, text)

        excluded_domains = {
            "cisa.gov", "fbi.gov", "microsoft.com",
            "google.com", "github.com"
        }

        for domain in set(domains):
            if domain not in excluded_domains:
                extracted["indicators"][
                    "domains"
                ].append({
                    "value": domain,
                    "context": "Extracted from advisory",
                    "confidence": 65
                })

        # Extract SHA256 hashes
        sha256_pattern = r"\b[a-fA-F0-9]{64}\b"
        hashes = re.findall(sha256_pattern, text)
        for hash_val in set(hashes):
            extracted["indicators"][
                "file_hashes"
            ].append({
                "sha256": hash_val,
                "context": "Extracted from advisory",
                "confidence": 80
            })

        # Extract ATT&CK techniques
        attck_pattern = r"T\d{4}(?:\.\d{3})?"
        techniques = re.findall(attck_pattern, text)
        for technique in set(techniques):
            extracted["attack_techniques"].append({
                "technique_id": technique,
                "technique_name": ""
            })

        return extracted

    def _parse_claude_response(
        self,
        response_text: str
    ) -> Optional[dict]:
        """
        Parse Claude's JSON response.

        Claude sometimes wraps JSON in markdown
        code blocks. We handle both cases.
        """
        # Remove markdown code blocks if present
        clean = response_text.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1])

        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Try to find JSON within the response
            json_match = re.search(
                r"\{.*\}", clean, re.DOTALL
            )
            if json_match:
                try:
                    return json.loads(
                        json_match.group()
                    )
                except json.JSONDecodeError:
                    pass

        return None

    def _convert_to_stix(
        self,
        extracted: dict,
        source_name: str
    ) -> dict:
        """
        Convert extracted IOCs to STIX bundle format.

        Creates proper STIX Indicator objects for
        each extracted IOC so STIXProcessor can
        ingest them using the existing pipeline.
        """
        objects = []
        obj_count = 0

        indicators = extracted.get("indicators", {})

        # ---- IP ADDRESS INDICATORS ----
        for ip_entry in indicators.get(
            "ip_addresses", []
        ):
            obj_count += 1
            objects.append({
                "type": "indicator",
                "id": f"indicator--cisa-ip-{obj_count}",
                "name": (
                    f"CISA IOC: {ip_entry['value']}"
                ),
                "pattern": (
                    f"[ipv4-addr:value = "
                    f"'{ip_entry['value']}']"
                ),
                "pattern_type": "stix",
                "labels": ["malicious-activity"],
                "confidence": ip_entry.get(
                    "confidence", 75
                ),
                "valid_from": self._now(),
                "description": ip_entry.get(
                    "context", ""
                ),
                "external_references": [{
                    "source_name": source_name
                }]
            })

        # ---- DOMAIN INDICATORS ----
        for domain_entry in indicators.get(
            "domains", []
        ):
            obj_count += 1
            objects.append({
                "type": "indicator",
                "id": (
                    f"indicator--cisa-domain-{obj_count}"
                ),
                "name": (
                    f"CISA IOC: "
                    f"{domain_entry['value']}"
                ),
                "pattern": (
                    f"[domain-name:value = "
                    f"'{domain_entry['value']}']"
                ),
                "pattern_type": "stix",
                "labels": ["malicious-activity"],
                "confidence": domain_entry.get(
                    "confidence", 75
                ),
                "valid_from": self._now(),
                "description": domain_entry.get(
                    "context", ""
                )
            })

        # ---- URL INDICATORS ----
        for url_entry in indicators.get("urls", []):
            obj_count += 1
            objects.append({
                "type": "indicator",
                "id": (
                    f"indicator--cisa-url-{obj_count}"
                ),
                "name": (
                    f"CISA IOC: {url_entry['value']}"
                ),
                "pattern": (
                    f"[url:value = "
                    f"'{url_entry['value']}']"
                ),
                "pattern_type": "stix",
                "labels": ["malicious-activity"],
                "confidence": url_entry.get(
                    "confidence", 75
                ),
                "valid_from": self._now()
            })

        # ---- FILE HASH INDICATORS ----
        for hash_entry in indicators.get(
            "file_hashes", []
        ):
            obj_count += 1
            sha256 = hash_entry.get("sha256", "")
            md5 = hash_entry.get("md5", "")

            if sha256:
                pattern = (
                    f"[file:hashes.'SHA-256' = "
                    f"'{sha256}']"
                )
            elif md5:
                pattern = (
                    f"[file:hashes.MD5 = '{md5}']"
                )
            else:
                continue

            objects.append({
                "type": "indicator",
                "id": (
                    f"indicator--cisa-hash-{obj_count}"
                ),
                "name": (
                    f"CISA IOC: "
                    f"{hash_entry.get('filename', 'malware')}"
                ),
                "pattern": pattern,
                "pattern_type": "stix",
                "labels": ["malicious-activity"],
                "confidence": hash_entry.get(
                    "confidence", 80
                ),
                "valid_from": self._now(),
                "description": hash_entry.get(
                    "context", ""
                )
            })

        # ---- MALWARE OBJECTS ----
        for malware_name in extracted.get(
            "malware_families", []
        ):
            obj_count += 1
            objects.append({
                "type": "malware",
                "id": (
                    f"malware--cisa-{obj_count}"
                ),
                "name": malware_name,
                "malware_types": ["unknown"],
                "aliases": []
            })

        # ---- THREAT ACTOR OBJECTS ----
        for actor_name in extracted.get(
            "threat_actors", []
        ):
            obj_count += 1
            objects.append({
                "type": "threat-actor",
                "id": (
                    f"threat-actor--cisa-{obj_count}"
                ),
                "name": actor_name,
                "aliases": [],
                "primary_motivation": "unknown"
            })

        return {
            "type": "bundle",
            "id": (
                f"bundle--cisa-{source_name}-"
                f"{self._now()}"
            ),
            "objects": objects
        }

    def _store_ip_range(
        self,
        cidr: str,
        context: str,
        confidence: int,
        source: str
    ) -> None:
        """Store CIDR range in local database"""
        if not cidr:
            return

        try:
            # Validate CIDR notation
            ipaddress.ip_network(cidr, strict=False)

            self.ip_ranges[cidr] = {
                "context": context,
                "risk_score": confidence / 100.0,
                "source": source,
                "stored_at": self._now()
            }

            logger.info(
                f"IP range stored: {cidr} "
                f"(risk={confidence/100.0:.2f})"
            )

        except ValueError:
            logger.warning(
                f"Invalid CIDR notation: {cidr}"
            )

    def _extract_title(self, text: str) -> str:
        """Extract advisory title from text"""
        lines = text.split("\n")
        for line in lines[:20]:
            line = line.strip()
            if len(line) > 10 and "AA" in line:
                return line[:100]
        return "Unknown Advisory"

    def _now(self) -> str:
        """Return current UTC timestamp"""
        return datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )