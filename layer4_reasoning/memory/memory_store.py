"""
Layer 4 — Agent Memory
ChromaDB Vector Memory Store

Provides persistent memory for security agents
using vector similarity search.

WHY VECTOR MEMORY:
    Regular search: exact keyword match.
    Vector search: semantic similarity.
    
    "svc_backup accessed PCI data via Tor"
    finds similar incidents like:
    "service account exfiltrated customer records"
    even though keywords don't match exactly.
    
    This is how agents detect patterns
    that rule-based systems miss.

TWO BACKENDS:
    ChromaDB: production vector database
              persistent across sessions
              semantic similarity search
              
    JSON fallback: when ChromaDB unavailable
                   same interface
                   keyword matching only
                   always works

USAGE:
    memory = AgentMemoryStore()
    
    # Store investigation
    memory.store_investigation(state)
    
    # Find similar past incidents
    similar = memory.search_similar(
        "service account accessed PCI data",
        n_results=5
    )
    
    # Get repeat offender history
    history = memory.get_entity_history(
        "svc_backup"
    )
"""

import json
import logging
import os
from datetime import datetime
from datetime import timezone
from typing import List
from typing import Optional

logger = logging.getLogger(__name__)

# Storage paths
MEMORY_DIR = "data/memory"
JSON_FALLBACK_FILE = "data/memory/investigations.json"


class AgentMemoryStore:
    """
    Persistent memory store for security agents.
    
    Uses ChromaDB for vector similarity search
    with JSON fallback when ChromaDB unavailable.
    
    Agents use this to:
    1. Store completed investigations
    2. Find similar past incidents
    3. Detect repeat offenders
    4. Build context across sessions
    """

    def __init__(
        self,
        persist_directory: str = MEMORY_DIR,
        collection_name: str = "security_investigations"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.use_chromadb = False

        os.makedirs(persist_directory, exist_ok=True)
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB or fall back to JSON"""
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            self.collection = (
                self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={
                        "description": (
                            "AbuTech security investigations"
                        )
                    }
                )
            )
            self.use_chromadb = True
            logger.info(
                f"ChromaDB initialized at "
                f"{self.persist_directory}"
            )

        except ImportError:
            logger.info(
                "ChromaDB not available. "
                "Using JSON fallback store."
            )
            self.use_chromadb = False
        except Exception as e:
            logger.warning(
                f"ChromaDB init failed: {e}. "
                f"Using JSON fallback."
            )
            self.use_chromadb = False

    def store_investigation(
        self,
        state: dict,
        investigation_id: str = None
    ) -> bool:
        """
        Store a completed investigation in memory.
        
        Creates a searchable document from the
        investigation state for future similarity
        search.
        
        Args:
            state: InvestigationState dict
            investigation_id: Optional custom ID
            
        Returns:
            True if stored successfully
        """
        try:
            inv_id = investigation_id or (
                f"inv_{state.get('event_id', 'unknown')}"
                f"_{self._timestamp()}"
            )

            # Build searchable document text
            document = self._build_document(state)

            # Build metadata for filtering
            metadata = self._build_metadata(state)

            if self.use_chromadb:
                return self._store_chromadb(
                    inv_id, document, metadata
                )
            else:
                return self._store_json(
                    inv_id, document, metadata, state
                )

        except Exception as e:
            logger.error(
                f"Failed to store investigation: {e}"
            )
            return False

    def search_similar(
        self,
        query: str,
        n_results: int = 5,
        filter_severity: str = None
    ) -> List[dict]:
        """
        Search for investigations similar to query.
        
        Uses vector similarity (ChromaDB) or
        keyword matching (JSON fallback).
        
        Args:
            query: Natural language description
                   of the incident to find
            n_results: Max results to return
            filter_severity: Optional severity filter
                            (CRITICAL/HIGH/MEDIUM/LOW)
                            
        Returns:
            List of similar past investigations
        """
        try:
            if self.use_chromadb:
                return self._search_chromadb(
                    query, n_results, filter_severity
                )
            else:
                return self._search_json(
                    query, n_results, filter_severity
                )
        except Exception as e:
            logger.error(
                f"Search failed: {e}"
            )
            return []

    def get_entity_history(
        self,
        entity: str,
        entity_type: str = "user",
        max_results: int = 10
    ) -> dict:
        """
        Get investigation history for a specific entity.
        
        Used by agents to detect repeat offenders
        and build entity risk profiles.
        
        Args:
            entity: Username, IP, or hostname
            entity_type: user/ip/host
            max_results: Max investigations to return
            
        Returns:
            dict with history and risk profile
        """
        try:
            query = (
                f"{entity_type} {entity} "
                f"security incident investigation"
            )
            results = self.search_similar(
                query, n_results=max_results
            )

            # Filter to exact entity matches
            entity_results = [
                r for r in results
                if entity.lower() in
                r.get("document", "").lower()
                or entity.lower() in
                str(r.get("metadata", {})).lower()
            ]

            return self._build_entity_profile(
                entity, entity_type, entity_results
            )

        except Exception as e:
            logger.error(
                f"Entity history failed: {e}"
            )
            return self._empty_profile(entity)

    def get_stats(self) -> dict:
        """
        Get memory store statistics.
        
        Returns:
            dict with total investigations,
            breakdown by severity, backend type
        """
        try:
            if self.use_chromadb:
                count = self.collection.count()
                return {
                    "total_investigations": count,
                    "backend": "chromadb",
                    "persist_directory": (
                        self.persist_directory
                    ),
                    "collection": self.collection_name
                }
            else:
                investigations = self._load_json()
                return {
                    "total_investigations": (
                        len(investigations)
                    ),
                    "backend": "json_fallback",
                    "storage_file": JSON_FALLBACK_FILE
                }
        except Exception as e:
            logger.error(f"Stats failed: {e}")
            return {
                "total_investigations": 0,
                "backend": "unknown",
                "error": str(e)
            }

    def delete_investigation(
        self,
        investigation_id: str
    ) -> bool:
        """Delete a specific investigation"""
        try:
            if self.use_chromadb:
                self.collection.delete(
                    ids=[investigation_id]
                )
                return True
            else:
                investigations = self._load_json()
                investigations = [
                    i for i in investigations
                    if i.get("id") != investigation_id
                ]
                self._save_json(investigations)
                return True
        except Exception as e:
            logger.error(
                f"Delete failed: {e}"
            )
            return False

    def clear_all(self) -> bool:
        """Clear all stored investigations"""
        try:
            if self.use_chromadb:
                self.client.delete_collection(
                    self.collection_name
                )
                self.collection = (
                    self.client.get_or_create_collection(
                        name=self.collection_name
                    )
                )
            else:
                self._save_json([])
            logger.info("Memory store cleared")
            return True
        except Exception as e:
            logger.error(f"Clear failed: {e}")
            return False

    # ============================================================
    # CHROMADB METHODS
    # ============================================================

    def _store_chromadb(
        self,
        inv_id: str,
        document: str,
        metadata: dict
    ) -> bool:
        """Store investigation in ChromaDB"""
        try:
            # Check if ID already exists
            existing = self.collection.get(
                ids=[inv_id]
            )
            if existing["ids"]:
                self.collection.update(
                    ids=[inv_id],
                    documents=[document],
                    metadatas=[metadata]
                )
            else:
                self.collection.add(
                    ids=[inv_id],
                    documents=[document],
                    metadatas=[metadata]
                )
            logger.info(
                f"Stored investigation {inv_id} "
                f"in ChromaDB"
            )
            return True
        except Exception as e:
            logger.error(
                f"ChromaDB store failed: {e}"
            )
            return False

    def _search_chromadb(
        self,
        query: str,
        n_results: int,
        filter_severity: str = None
    ) -> List[dict]:
        """Search ChromaDB with vector similarity"""
        try:
            count = self.collection.count()
            if count == 0:
                return []

            where = None
            if filter_severity:
                where = {
                    "severity": filter_severity
                }

            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, count),
                where=where
            )

            formatted = []
            for i in range(
                len(results["ids"][0])
            ):
                formatted.append({
                    "id": results["ids"][0][i],
                    "document": (
                        results["documents"][0][i]
                    ),
                    "metadata": (
                        results["metadatas"][0][i]
                    ),
                    "distance": (
                        results["distances"][0][i]
                        if results.get("distances")
                        else None
                    )
                })

            return formatted

        except Exception as e:
            logger.error(
                f"ChromaDB search failed: {e}"
            )
            return []

    # ============================================================
    # JSON FALLBACK METHODS
    # ============================================================

    def _store_json(
        self,
        inv_id: str,
        document: str,
        metadata: dict,
        state: dict
    ) -> bool:
        """Store investigation in JSON file"""
        try:
            investigations = self._load_json()

            record = {
                "id": inv_id,
                "document": document,
                "metadata": metadata,
                "timestamp": self._timestamp()
            }

            # Update if exists
            existing_ids = [
                i.get("id") for i in investigations
            ]
            if inv_id in existing_ids:
                investigations = [
                    record if i.get("id") == inv_id
                    else i
                    for i in investigations
                ]
            else:
                investigations.append(record)

            # Keep last 1000
            if len(investigations) > 1000:
                investigations = investigations[-1000:]

            self._save_json(investigations)
            logger.info(
                f"Stored investigation {inv_id} "
                f"in JSON store"
            )
            return True

        except Exception as e:
            logger.error(
                f"JSON store failed: {e}"
            )
            return False

    def _search_json(
        self,
        query: str,
        n_results: int,
        filter_severity: str = None
    ) -> List[dict]:
        """Search JSON store with keyword matching"""
        try:
            investigations = self._load_json()

            if filter_severity:
                investigations = [
                    i for i in investigations
                    if i.get("metadata", {}).get(
                        "severity"
                    ) == filter_severity
                ]

            query_words = set(
                query.lower().split()
            )
            scored = []

            for inv in investigations:
                doc_words = set(
                    inv.get(
                        "document", ""
                    ).lower().split()
                )
                score = len(
                    query_words & doc_words
                )
                if score > 0:
                    scored.append({
                        **inv,
                        "distance": 1.0 / (
                            1.0 + score
                        )
                    })

            scored.sort(
                key=lambda x: x["distance"]
            )
            return scored[:n_results]

        except Exception as e:
            logger.error(
                f"JSON search failed: {e}"
            )
            return []

    def _load_json(self) -> list:
        """Load investigations from JSON file"""
        try:
            if os.path.exists(JSON_FALLBACK_FILE):
                with open(
                    JSON_FALLBACK_FILE, "r"
                ) as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(
                f"Could not load JSON store: {e}"
            )
        return []

    def _save_json(
        self,
        investigations: list
    ) -> None:
        """Save investigations to JSON file"""
        try:
            os.makedirs(MEMORY_DIR, exist_ok=True)
            with open(
                JSON_FALLBACK_FILE, "w"
            ) as f:
                json.dump(
                    investigations, f, indent=2
                )
        except Exception as e:
            logger.error(
                f"Could not save JSON store: {e}"
            )

    # ============================================================
    # HELPER METHODS
    # ============================================================

    def _build_document(self, state: dict) -> str:
        """
        Build searchable text document from state.
        
        This text is what gets embedded into
        vector space for similarity search.
        The more descriptive, the better the search.
        """
        parts = []

        # Core event info
        host = state.get("event_host", "")
        user = state.get("event_user", "")
        verdict = state.get("overall_verdict", "")
        severity = state.get("severity_rating", "")

        if host:
            parts.append(f"host {host}")
        if user:
            parts.append(f"user {user}")
        if verdict:
            parts.append(f"verdict {verdict}")
        if severity:
            parts.append(
                f"severity {severity}"
            )

        # Triage
        triage = state.get("triage_reasoning", "")
        if triage:
            parts.append(triage[:500])

        # Intel
        intel = state.get("intel_summary", "")
        if intel:
            parts.append(intel[:500])

        # Investigation
        investigation = state.get(
            "investigation_summary", ""
        )
        if investigation:
            parts.append(investigation[:500])

        # ATT&CK techniques
        techniques = state.get(
            "confirmed_techniques", []
        )
        if techniques:
            parts.append(
                f"techniques {' '.join(techniques)}"
            )

        # Threat actor
        actor = state.get(
            "threat_actor_identified", ""
        )
        if actor:
            parts.append(
                f"threat actor {actor}"
            )

        # Malware
        malware = state.get(
            "malware_family_confirmed", ""
        )
        if malware:
            parts.append(f"malware {malware}")

        return " ".join(parts)

    def _build_metadata(self, state: dict) -> dict:
        """
        Build metadata dict for filtering.
        
        ChromaDB metadata must be
        str, int, float, or bool only.
        No nested dicts or lists.
        """
        return {
            "event_host": str(
                state.get("event_host", "")
            ),
            "event_user": str(
                state.get("event_user", "")
            ),
            "severity": str(
                state.get("severity_rating", "")
            ),
            "verdict": str(
                state.get("overall_verdict", "")
            ),
            "triage_verdict": str(
                state.get("triage_verdict", "")
            ),
            "risk_score": float(
                state.get(
                    "overall_risk_score", 0.0
                ) or 0.0
            ),
            "compromise_confirmed": bool(
                state.get(
                    "compromise_confirmed", False
                )
            ),
            "timestamp": self._timestamp()
        }

    def _build_entity_profile(
        self,
        entity: str,
        entity_type: str,
        results: List[dict]
    ) -> dict:
        """Build entity risk profile from history"""
        if not results:
            return self._empty_profile(entity)

        severities = [
            r.get("metadata", {}).get(
                "severity", ""
            )
            for r in results
        ]
        critical_count = severities.count("CRITICAL")
        high_count = severities.count("HIGH")

        repeat_offender = len(results) >= 3
        escalating = (
            len(results) >= 2 and
            critical_count > 0
        )

        risk_level = "LOW"
        if critical_count >= 2:
            risk_level = "CRITICAL"
        elif critical_count >= 1 or high_count >= 2:
            risk_level = "HIGH"
        elif len(results) >= 3:
            risk_level = "MEDIUM"

        summary = (
            f"Entity {entity} ({entity_type}): "
            f"{len(results)} past investigations. "
        )
        if repeat_offender:
            summary += (
                f"⚠️ REPEAT OFFENDER. "
            )
        if escalating:
            summary += (
                f"Pattern is ESCALATING. "
            )
        summary += (
            f"Risk level: {risk_level}."
        )

        return {
            "entity": entity,
            "entity_type": entity_type,
            "total_incidents": len(results),
            "critical_incidents": critical_count,
            "high_incidents": high_count,
            "repeat_offender": repeat_offender,
            "escalating": escalating,
            "risk_level": risk_level,
            "recent_incidents": results[:5],
            "summary": summary
        }

    def _empty_profile(self, entity: str) -> dict:
        """Return empty profile for unknown entity"""
        return {
            "entity": entity,
            "entity_type": "unknown",
            "total_incidents": 0,
            "critical_incidents": 0,
            "high_incidents": 0,
            "repeat_offender": False,
            "escalating": False,
            "risk_level": "UNKNOWN",
            "recent_incidents": [],
            "summary": (
                f"No history found for {entity}."
            )
        }

    def _timestamp(self) -> str:
        """Return current UTC timestamp"""
        return datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")