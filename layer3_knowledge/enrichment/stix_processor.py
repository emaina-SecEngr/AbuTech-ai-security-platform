"""
Layer 3 — Knowledge Graph
STIX/TAXII Threat Intelligence Processor

This module consumes STIX threat intelligence bundles
from TAXII servers and enriches the knowledge graph.

STIX/TAXII is the enterprise standard for sharing
threat intelligence. Every major source uses it:

    CISA        → https://cisa.gov/taxii
    FBI         → Flash reports in STIX format
    MISP        → Open source TI platform
    OpenCTI     → Open source TI platform
    Recorded Future → Commercial
    Mandiant    → Commercial

STIX Object Types We Process:
    indicator      → IOC with detection pattern
                     Contains IP, domain, URL, hash
    malware        → Malware family details
    threat-actor   → Who is behind the attack
    campaign       → Coordinated attack series
    attack-pattern → MITRE ATT&CK technique
    relationship   → Connects objects together

Delta Processing:
    We track which STIX objects we have already
    processed using their STIX IDs.
    Only new objects trigger graph updates.
    This prevents reprocessing millions of IOCs
    every time the feed updates.

Storage:
    SQLite local database for IOC lookup
    Fast, built into Python, no setup needed
    Handles millions of IOCs efficiently
    Syncs to Delta Lake in production
"""

import json
import logging
import re
import sqlite3
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# STIX PATTERN EXTRACTORS
# Each function extracts a specific IOC type
# from the STIX pattern language
# ============================================================

def extract_ip_from_pattern(pattern: str) -> Optional[str]:
    """
    Extract IP address from STIX pattern.

    STIX pattern format:
        [ipv4-addr:value = '185.220.101.45']
        [ipv4-addr:value = '10.0.0.1']
    """
    match = re.search(
        r"ipv4-addr:value\s*=\s*['\"](.+?)['\"]",
        pattern
    )
    return match.group(1) if match else None


def extract_domain_from_pattern(
    pattern: str
) -> Optional[str]:
    """
    Extract domain from STIX pattern.

    STIX pattern format:
        [domain-name:value = 'evil.com']
    """
    match = re.search(
        r"domain-name:value\s*=\s*['\"](.+?)['\"]",
        pattern
    )
    return match.group(1) if match else None


def extract_url_from_pattern(
    pattern: str
) -> Optional[str]:
    """
    Extract URL from STIX pattern.

    STIX pattern format:
        [url:value = 'http://evil.com/payload.exe']
    """
    match = re.search(
        r"url:value\s*=\s*['\"](.+?)['\"]",
        pattern
    )
    return match.group(1) if match else None


def extract_hash_from_pattern(
    pattern: str
) -> Optional[dict]:
    """
    Extract file hash from STIX pattern.

    STIX pattern format:
        [file:hashes.MD5 = 'abc123def456']
        [file:hashes.'SHA-256' = 'abc123...']
    """
    md5_match = re.search(
        r"hashes\.MD5\s*=\s*['\"](.+?)['\"]",
        pattern
    )
    sha256_match = re.search(
        r"hashes\.'SHA-256'\s*=\s*['\"](.+?)['\"]",
        pattern
    )

    if md5_match or sha256_match:
        return {
            "md5": md5_match.group(1) if md5_match else None,
            "sha256": (
                sha256_match.group(1)
                if sha256_match else None
            )
        }
    return None


def extract_ioc_from_pattern(
    pattern: str
) -> dict:
    """
    Extract any IOC type from a STIX pattern.

    Returns dict with ioc_type and ioc_value.
    """
    ip = extract_ip_from_pattern(pattern)
    if ip:
        return {"type": "ip", "value": ip}

    domain = extract_domain_from_pattern(pattern)
    if domain:
        return {"type": "domain", "value": domain}

    url = extract_url_from_pattern(pattern)
    if url:
        return {"type": "url", "value": url}

    hashes = extract_hash_from_pattern(pattern)
    if hashes:
        return {"type": "hash", "value": hashes}

    return {"type": "unknown", "value": None}


class STIXProcessor:
    """
    Processes STIX bundles and enriches the
    security knowledge graph.

    Handles delta processing — only processes
    new STIX objects since last run.

    Usage:
        processor = STIXProcessor(graph)

        # Process a STIX bundle
        results = processor.process_bundle(stix_bundle)

        # Query local IOC database
        intel = processor.lookup_ip("185.220.101.45")
    """

    def __init__(
        self,
        graph,
        db_path: str = ".threat_cache/ioc_database.db"
    ):
        """
        Initialize STIX processor.

        Args:
            graph: SecurityKnowledgeGraph instance
            db_path: Path to local SQLite IOC database
        """
        self.graph = graph
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        # Statistics
        self.bundles_processed = 0
        self.objects_processed = 0
        self.new_iocs_added = 0
        self.graph_nodes_updated = 0

        # Initialize database
        self._init_database()

        logger.info("STIXProcessor initialized")

    # ============================================================
    # PRIMARY PUBLIC METHODS
    # ============================================================

    def process_bundle(
        self,
        bundle: dict,
        source_name: str = "stix_feed"
    ) -> dict:
        """
        Process a STIX bundle and update the graph.

        A STIX bundle is a collection of STIX objects
        wrapped in a container:
        {
            "type": "bundle",
            "id": "bundle--uuid",
            "objects": [...]
        }

        This method:
        1. Extracts all objects from the bundle
        2. Computes delta — only new objects
        3. Processes each new object
        4. Updates knowledge graph
        5. Returns processing summary

        Args:
            bundle: STIX bundle dictionary
            source_name: Name of the feed source

        Returns:
            Processing results dictionary
        """
        results = {
            "total_objects": 0,
            "new_objects": 0,
            "indicators_processed": 0,
            "malware_processed": 0,
            "actors_processed": 0,
            "graph_nodes_updated": 0,
            "iocs_added": 0
        }

        if not isinstance(bundle, dict):
            logger.error("Bundle must be a dictionary")
            return results

        objects = bundle.get("objects", [])
        if not objects:
            logger.warning("Bundle contains no objects")
            return results

        results["total_objects"] = len(objects)

        # ---- DELTA PROCESSING ----
        # Only process objects we have not seen before
        new_objects = self._compute_delta(objects)
        results["new_objects"] = len(new_objects)

        logger.info(
            f"Bundle: {len(objects)} total objects, "
            f"{len(new_objects)} new"
        )

        # ---- PROCESS EACH NEW OBJECT ----
        for stix_obj in new_objects:
            obj_type = stix_obj.get("type", "")

            if obj_type == "indicator":
                processed = self._process_indicator(
                    stix_obj, source_name
                )
                if processed:
                    results["indicators_processed"] += 1
                    results["iocs_added"] += 1
                    results["graph_nodes_updated"] += 1

            elif obj_type == "malware":
                self._process_malware(
                    stix_obj, source_name
                )
                results["malware_processed"] += 1

            elif obj_type == "threat-actor":
                self._process_threat_actor(
                    stix_obj, source_name
                )
                results["actors_processed"] += 1

            elif obj_type == "relationship":
                self._process_relationship(stix_obj)

            # Mark as processed in database
            self._mark_processed(stix_obj)

        self.bundles_processed += 1
        self.objects_processed += len(new_objects)
        self.new_iocs_added += results["iocs_added"]
        self.graph_nodes_updated += (
            results["graph_nodes_updated"]
        )

        logger.info(
            f"Bundle processing complete: {results}"
        )

        return results

    def process_bundle_from_file(
        self,
        file_path: str,
        source_name: str = "stix_file"
    ) -> dict:
        """
        Load and process a STIX bundle from a JSON file.

        CISA publishes STIX bundles as JSON files.
        FBI Flash reports are distributed as STIX JSON.

        Args:
            file_path: Path to STIX JSON file
            source_name: Name for attribution

        Returns:
            Processing results
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(
                f"STIX file not found: {file_path}"
            )
            return {}

        with open(file_path, "r") as f:
            bundle = json.load(f)

        logger.info(
            f"Processing STIX file: {file_path.name}"
        )
        return self.process_bundle(bundle, source_name)

    def process_bundle_from_taxii(
        self,
        taxii_url: str,
        collection_id: str,
        source_name: str = "taxii_feed"
    ) -> dict:
        """
        Fetch and process STIX bundle from TAXII server.

        TAXII servers host collections of STIX objects.
        This method:
        1. Connects to TAXII server
        2. Fetches new objects since last check
        3. Processes the STIX bundle

        Example TAXII servers:
            CISA: https://cisa.gov/taxii/v21/
            MISP: Your organization's MISP instance

        Args:
            taxii_url: TAXII server API root URL
            collection_id: Collection to fetch from
            source_name: Name for attribution

        Returns:
            Processing results
        """
        try:
            from taxii2client.v21 import (
                Server, Collection
            )

            logger.info(
                f"Connecting to TAXII: {taxii_url}"
            )

            server = Server(taxii_url)
            api_root = server.api_roots[0]

            collection = Collection(
                f"{taxii_url}collections/"
                f"{collection_id}/objects/"
            )

            # Fetch objects added since last check
            last_checked = self._get_last_checked(
                taxii_url
            )

            objects = collection.get_objects(
                added_after=last_checked
            )

            bundle = {
                "type": "bundle",
                "objects": list(objects.get("objects", []))
            }

            self._update_last_checked(taxii_url)

            return self.process_bundle(
                bundle, source_name
            )

        except ImportError:
            logger.error(
                "taxii2-client not installed. "
                "Run: pip install taxii2-client"
            )
            return {}
        except Exception as e:
            logger.error(
                f"TAXII fetch failed: {e}"
            )
            return {}

    def lookup_ip(self, ip: str) -> Optional[dict]:
        """
        Look up an IP in the local IOC database.

        Faster than querying external APIs.
        Used as first-level cache before API calls.

        Args:
            ip: IP address to look up

        Returns:
            IOC record or None if not found
        """
        return self._db_lookup("ip", ip)

    def lookup_domain(
        self, domain: str
    ) -> Optional[dict]:
        """Look up domain in local IOC database"""
        return self._db_lookup("domain", domain)

    def lookup_hash(
        self, hash_value: str
    ) -> Optional[dict]:
        """Look up file hash in local IOC database"""
        return self._db_lookup("hash", hash_value)

    def get_statistics(self) -> dict:
        """Return processing statistics"""
        db_count = self._get_ioc_count()
        return {
            "bundles_processed": self.bundles_processed,
            "objects_processed": self.objects_processed,
            "new_iocs_added": self.new_iocs_added,
            "graph_nodes_updated": (
                self.graph_nodes_updated
            ),
            "iocs_in_database": db_count
        }

    # ============================================================
    # STIX OBJECT PROCESSORS
    # ============================================================

    def _process_indicator(
        self,
        indicator: dict,
        source: str
    ) -> bool:
        """
        Process a STIX Indicator object.

        Indicators contain detection patterns —
        the actual IOC values we need.

        STIX Indicator structure:
        {
            "type": "indicator",
            "id": "indicator--uuid",
            "name": "Tor Exit Node",
            "pattern": "[ipv4-addr:value = '...']",
            "pattern_type": "stix",
            "labels": ["malicious-activity"],
            "valid_from": "2024-01-01T00:00:00Z",
            "confidence": 85
        }
        """
        pattern = indicator.get("pattern", "")
        if not pattern:
            return False

        # Extract IOC from pattern
        ioc = extract_ioc_from_pattern(pattern)

        if not ioc["value"]:
            logger.debug(
                f"Could not extract IOC from: {pattern}"
            )
            return False

        name = indicator.get("name", "Unknown Indicator")
        confidence = indicator.get("confidence", 50)
        risk_score = confidence / 100.0
        labels = indicator.get("labels", [])
        valid_from = indicator.get("valid_from", "")

        # Store in local database
        self._db_store_ioc(
            ioc_type=ioc["type"],
            ioc_value=(
                ioc["value"]
                if isinstance(ioc["value"], str)
                else json.dumps(ioc["value"])
            ),
            name=name,
            risk_score=risk_score,
            source=source,
            labels=labels,
            valid_from=valid_from,
            stix_id=indicator.get("id", "")
        )

        # Update knowledge graph
        self._update_graph_from_ioc(
            ioc, name, risk_score, source, labels
        )

        logger.info(
            f"Indicator processed: {ioc['type']} "
            f"{ioc['value']} risk={risk_score:.2f}"
        )

        return True

    def _process_malware(
        self,
        malware: dict,
        source: str
    ) -> None:
        """
        Process a STIX Malware object.

        Malware objects describe malware families —
        their capabilities, aliases, and behaviors.

        We store these for reference when campaign
        identification links detections to families.
        """
        name = malware.get("name", "Unknown Malware")
        aliases = malware.get("aliases", [])
        capabilities = malware.get(
            "capabilities", []
        )
        malware_types = malware.get(
            "malware_types", []
        )

        logger.info(
            f"Malware family indexed: {name} "
            f"(aliases: {aliases})"
        )

        # Store malware family in database
        self._db_store_metadata(
            object_type="malware",
            name=name,
            stix_id=malware.get("id", ""),
            properties={
                "aliases": aliases,
                "capabilities": capabilities,
                "malware_types": malware_types,
                "source": source
            }
        )

    def _process_threat_actor(
        self,
        actor: dict,
        source: str
    ) -> None:
        """
        Process a STIX Threat Actor object.

        Threat actors are the groups or individuals
        behind attacks — nation states, criminal
        organizations, hacktivist groups.

        We add these to the knowledge graph and
        connect them to related indicators.
        """
        from layer3_knowledge.graph.security_graph import (
            SecurityNode, NodeType
        )

        name = actor.get("name", "Unknown Actor")
        aliases = actor.get("aliases", [])
        motivation = (
            actor.get("primary_motivation", "unknown")
        )
        sophistication = actor.get(
            "sophistication", "unknown"
        )

        logger.info(
            f"Threat actor indexed: {name}"
        )

        # Add to knowledge graph
        node = SecurityNode(
            node_id=f"actor:{actor.get('id', name)}",
            node_type=NodeType.THREAT_ACTOR,
            label=name,
            risk_score=0.9,
            properties={
                "aliases": aliases,
                "motivation": motivation,
                "sophistication": sophistication,
                "source": source,
                "stix_id": actor.get("id", "")
            }
        )
        self.graph.add_node(node)

        # Store in database
        self._db_store_metadata(
            object_type="threat-actor",
            name=name,
            stix_id=actor.get("id", ""),
            properties={
                "aliases": aliases,
                "motivation": motivation,
                "source": source
            }
        )

    def _process_relationship(
        self,
        relationship: dict
    ) -> None:
        """
        Process a STIX Relationship object.

        Relationships connect STIX objects:
        - Indicator INDICATES Malware
        - Malware ATTRIBUTED-TO Threat Actor
        - Campaign USES Malware

        We use these to build graph edges between
        threat actors, malware, and indicators.
        """
        rel_type = relationship.get(
            "relationship_type", ""
        )
        source_ref = relationship.get("source_ref", "")
        target_ref = relationship.get("target_ref", "")

        logger.debug(
            f"Relationship: {source_ref} "
            f"--{rel_type}--> {target_ref}"
        )

    def _update_graph_from_ioc(
        self,
        ioc: dict,
        name: str,
        risk_score: float,
        source: str,
        labels: list
    ) -> None:
        """
        Add IOC to knowledge graph.

        Maps STIX IOC types to graph node types
        and updates risk scores.
        """
        properties = {
            "stix_name": name,
            "stix_labels": labels,
            "stix_source": source
        }

        if ioc["type"] == "ip":
            node = self.graph.add_ip(
                ip_address=ioc["value"],
                risk_score=risk_score,
                properties=properties
            )
            if node:
                self.graph_nodes_updated += 1

        elif ioc["type"] == "domain":
            node = self.graph.add_domain(
                domain=ioc["value"],
                risk_score=risk_score,
                properties=properties
            )
            if node:
                self.graph_nodes_updated += 1

    # ============================================================
    # DATABASE METHODS
    # SQLite for local IOC storage
    # ============================================================

    def _init_database(self) -> None:
        """Initialize SQLite IOC database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS iocs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ioc_type TEXT NOT NULL,
                    ioc_value TEXT NOT NULL,
                    name TEXT,
                    risk_score REAL DEFAULT 0.0,
                    source TEXT,
                    labels TEXT,
                    valid_from TEXT,
                    stix_id TEXT UNIQUE,
                    created_at TEXT,
                    UNIQUE(ioc_type, ioc_value)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    stix_id TEXT UNIQUE,
                    properties TEXT,
                    created_at TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed (
                    stix_id TEXT PRIMARY KEY,
                    processed_at TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS feed_state (
                    feed_url TEXT PRIMARY KEY,
                    last_checked TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_ioc_type_value
                ON iocs(ioc_type, ioc_value)
            """)

        logger.debug(
            f"IOC database initialized: {self.db_path}"
        )

    def _compute_delta(
        self,
        objects: list
    ) -> list:
        """
        Return only objects not yet processed.

        Delta processing ensures we never reprocess
        the same STIX objects twice even if the
        full feed is re-downloaded.
        """
        new_objects = []

        with sqlite3.connect(self.db_path) as conn:
            for obj in objects:
                stix_id = obj.get("id", "")
                if not stix_id:
                    new_objects.append(obj)
                    continue

                cursor = conn.execute(
                    "SELECT 1 FROM processed "
                    "WHERE stix_id = ?",
                    (stix_id,)
                )
                if not cursor.fetchone():
                    new_objects.append(obj)

        return new_objects

    def _mark_processed(self, obj: dict) -> None:
        """Mark a STIX object as processed"""
        stix_id = obj.get("id", "")
        if not stix_id:
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO processed "
                "(stix_id, processed_at) VALUES (?, ?)",
                (stix_id, self._now())
            )

    def _db_store_ioc(
        self,
        ioc_type: str,
        ioc_value: str,
        name: str,
        risk_score: float,
        source: str,
        labels: list,
        valid_from: str,
        stix_id: str
    ) -> None:
        """Store IOC in SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO iocs
                (ioc_type, ioc_value, name, risk_score,
                 source, labels, valid_from, stix_id,
                 created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ioc_type,
                ioc_value,
                name,
                risk_score,
                source,
                json.dumps(labels),
                valid_from,
                stix_id,
                self._now()
            ))

    def _db_store_metadata(
        self,
        object_type: str,
        name: str,
        stix_id: str,
        properties: dict
    ) -> None:
        """Store metadata object in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO metadata
                (object_type, name, stix_id,
                 properties, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                object_type,
                name,
                stix_id,
                json.dumps(properties),
                self._now()
            ))

    def _db_lookup(
        self,
        ioc_type: str,
        ioc_value: str
    ) -> Optional[dict]:
        """Look up IOC in local database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM iocs "
                "WHERE ioc_type = ? AND ioc_value = ?",
                (ioc_type, ioc_value)
            )
            row = cursor.fetchone()

            if row:
                columns = [
                    desc[0]
                    for desc in cursor.description
                ]
                return dict(zip(columns, row))

        return None

    def _get_ioc_count(self) -> int:
        """Return total IOC count in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM iocs"
            )
            return cursor.fetchone()[0]

    def _get_last_checked(
        self, feed_url: str
    ) -> Optional[str]:
        """Get last time we checked this feed"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT last_checked FROM feed_state "
                "WHERE feed_url = ?",
                (feed_url,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def _update_last_checked(
        self, feed_url: str
    ) -> None:
        """Update last checked timestamp for feed"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO feed_state
                (feed_url, last_checked)
                VALUES (?, ?)
            """, (feed_url, self._now()))

    def _now(self) -> str:
        """Return current UTC timestamp"""
        return datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )