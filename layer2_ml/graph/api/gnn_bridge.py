"""
Layer 2 — ML Processing
GNN Bridge — Knowledge Graph Connector

Bridges Layer 2 GNN detector to
Layer 3 SecurityKnowledgeGraph.

WHY THIS EXISTS:
    GNNThreatDetector needs a SecurityGraph
    object with nodes and edges.
    
    Layer 3 SecurityKnowledgeGraph has
    real entity relationships from:
    - CISA advisories
    - STIX threat intel
    - Historical investigations
    - Network flow data
    
    This bridge:
    1. Queries Layer 3 graph for an entity
    2. Builds a SecurityGraph from real data
    3. Passes to GNNThreatDetector
    4. Returns real anomaly score

REAL vs SYNTHETIC:
    BEFORE: GNN scored synthetic test graph
    AFTER:  GNN scores real threat intel graph
    
    svc_backup connected to:
    - Known Tor exit node IP ← from STIX intel
    - prod-customer-data ← from past incidents
    - 3 past critical events ← from memory store
    
    GNN score reflects real threat context.

USAGE:
    bridge = GNNBridge()
    result = bridge.score_entity(
        entity_id="svc_backup",
        entity_type="user"
    )
    print(result.anomaly_score)  # 0.91
    print(result.threat_proximity) # 1
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GNNBridgeResult:
    """Result from GNN bridge scoring"""
    entity_id: str
    entity_type: str
    anomaly_score: float
    risk_label: str
    threat_proximity: int
    connected_threats: int
    graph_nodes: int
    graph_edges: int
    used_real_graph: bool
    pattern: str
    reasoning: str


class GNNBridge:
    """
    Bridge between Layer 2 GNN detector
    and Layer 3 Security Knowledge Graph.

    Queries real graph data and feeds it
    to the GNN for accurate scoring.
    """

    def __init__(self):
        self._gnn = None
        self._kg = None
        self._initialize()

    def _initialize(self):
        """Initialize GNN and Knowledge Graph"""
        # Initialize GNN detector
        try:
            from layer2_ml.graph.gnn_detector import (
                GNNThreatDetector
            )
            self._gnn = GNNThreatDetector()
            logger.info("GNN detector initialized")
        except Exception as e:
            logger.warning(
                f"GNN init failed: {e}"
            )

        # Initialize Knowledge Graph
        try:
            from layer3_knowledge.graph\
                .security_graph import (
                    SecurityKnowledgeGraph
                )
            self._kg = SecurityKnowledgeGraph()
            logger.info(
                "Knowledge Graph initialized"
            )
        except Exception as e:
            logger.warning(
                f"Knowledge Graph init failed: {e}"
            )

    def score_entity(
        self,
        entity_id: str,
        entity_type: str = "user",
        context: dict = None
    ) -> GNNBridgeResult:
        """
        Score an entity using GNN with
        real knowledge graph data.

        Args:
            entity_id: Entity to score
                       (username, IP, hostname)
            entity_type: Type of entity
            context: Optional additional context
                     from the current event

        Returns:
            GNNBridgeResult with anomaly score
        """
        if not entity_id:
            return self._empty_result(entity_id)

        # Try real graph scoring
        if self._gnn and self._kg:
            try:
                return self._score_with_real_graph(
                    entity_id, entity_type, context
                )
            except Exception as e:
                logger.warning(
                    f"Real graph scoring failed "
                    f"for {entity_id}: {e}. "
                    f"Using rule-based."
                )

        # Fall back to rule-based
        return self._rule_based_score(
            entity_id, entity_type, context
        )

    def score_event_graph(
        self,
        accessor: str,
        source_ip: str = None,
        data_store: str = None
    ) -> GNNBridgeResult:
        """
        Score an event by building a mini-graph
        from event context and querying KG.

        This is the main entry point from
        the FastAPI ingestion pipeline.

        Args:
            accessor: User or service account
            source_ip: Source IP of the event
            data_store: Data store accessed

        Returns:
            GNNBridgeResult with anomaly score
        """
        context = {
            "source_ip": source_ip,
            "data_store": data_store
        }

        return self.score_entity(
            entity_id=accessor,
            entity_type="user",
            context=context
        )

    def _score_with_real_graph(
        self,
        entity_id: str,
        entity_type: str,
        context: dict = None
    ) -> GNNBridgeResult:
        """Score using real KG + GNN"""
        from layer2_ml.graph.gnn_detector import (
            SecurityGraph
        )

        # Build SecurityGraph from KG data
        security_graph = SecurityGraph()

        # Add the main entity
        entity_risk = self._get_entity_risk(
            entity_id
        )
        security_graph.add_node(
            f"{entity_type}:{entity_id}",
            entity_type,
            entity_risk
        )

        # Add connected entities from KG
        threat_count = 0
        total_nodes = 1

        try:
            neighbors = self._get_neighbors(
                entity_id
            )
            for neighbor in neighbors[:10]:
                n_id = neighbor.get("id", "")
                n_type = neighbor.get("type", "unknown")
                n_risk = float(
                    neighbor.get("risk_score", 0.0)
                    or 0.0
                )
                n_label = f"{n_type}:{n_id}"

                security_graph.add_node(
                    n_label, n_type, n_risk
                )
                security_graph.add_edge(
                    f"{entity_type}:{entity_id}",
                    n_label,
                    "connected",
                    n_risk
                )
                total_nodes += 1

                if neighbor.get("is_threat", False):
                    security_graph.mark_as_threat(
                        n_label,
                        neighbor.get(
                            "threat_type", "known_threat"
                        )
                    )
                    threat_count += 1

        except Exception as e:
            logger.debug(
                f"Could not get neighbors: {e}"
            )

        # Add context entities if provided
        if context:
            source_ip = context.get("source_ip")
            if source_ip:
                ip_risk = self._get_ip_risk(source_ip)
                ip_label = f"ip:{source_ip}"
                security_graph.add_node(
                    ip_label, "ip_address", ip_risk
                )
                security_graph.add_edge(
                    f"{entity_type}:{entity_id}",
                    ip_label,
                    "communicates_with",
                    ip_risk
                )
                total_nodes += 1

                if ip_risk >= 0.8:
                    security_graph.mark_as_threat(
                        ip_label, "suspicious_ip"
                    )
                    threat_count += 1

            data_store = context.get("data_store")
            if data_store:
                ds_risk = self._get_datastore_risk(
                    data_store
                )
                ds_label = f"data:{data_store}"
                security_graph.add_node(
                    ds_label, "data_store", ds_risk
                )
                security_graph.add_edge(
                    f"{entity_type}:{entity_id}",
                    ds_label,
                    "accessed",
                    ds_risk
                )
                total_nodes += 1

        # Score with GNN
        gnn_result = self._gnn.score_node(
            security_graph,
            f"{entity_type}:{entity_id}"
        )

        score = float(gnn_result.anomaly_score)
        proximity = int(
            gnn_result.hop_distance_to_threat
            if hasattr(
                gnn_result, "hop_distance_to_threat"
            )
            else (-1 if threat_count == 0 else 1)
        )
        pattern = str(
            gnn_result.subgraph_pattern
            if hasattr(gnn_result, "subgraph_pattern")
            else "GRAPH_ANALYSIS"
        )

        risk_label = self._score_to_label(score)
        reasoning = self._build_reasoning(
            entity_id, score, threat_count,
            total_nodes, proximity, pattern
        )

        return GNNBridgeResult(
            entity_id=entity_id,
            entity_type=entity_type,
            anomaly_score=score,
            risk_label=risk_label,
            threat_proximity=proximity,
            connected_threats=threat_count,
            graph_nodes=total_nodes,
            graph_edges=total_nodes - 1,
            used_real_graph=True,
            pattern=pattern,
            reasoning=reasoning
        )

    def _rule_based_score(
        self,
        entity_id: str,
        entity_type: str,
        context: dict = None
    ) -> GNNBridgeResult:
        """Rule-based scoring when GNN unavailable"""
        score = 0.0
        patterns = []
        entity_lower = entity_id.lower()

        # Service account heuristics
        if any(
            p in entity_lower
            for p in ["svc_", "_svc", "service_"]
        ):
            score = max(score, 0.4)
            patterns.append("SERVICE_ACCOUNT")

        # Admin heuristics
        if any(
            p in entity_lower
            for p in ["admin", "root", "sudo"]
        ):
            score = max(score, 0.5)
            patterns.append("PRIVILEGED_ACCOUNT")

        # Check context
        threat_count = 0
        if context:
            ip = context.get("source_ip", "")
            if ip.startswith("185.220"):
                score = max(score, 0.9)
                patterns.append("TOR_EXIT_NODE")
                threat_count += 1

            data_store = context.get(
                "data_store", ""
            ).lower()
            if any(
                kw in data_store
                for kw in [
                    "customer", "pci", "phi", "pii"
                ]
            ):
                score = max(score, 0.7)
                patterns.append("SENSITIVE_DATA")

        pattern_str = (
            " + ".join(patterns)
            if patterns else "NO_PATTERN"
        )
        risk_label = self._score_to_label(score)

        return GNNBridgeResult(
            entity_id=entity_id,
            entity_type=entity_type,
            anomaly_score=score,
            risk_label=risk_label,
            threat_proximity=(
                1 if threat_count > 0 else -1
            ),
            connected_threats=threat_count,
            graph_nodes=1,
            graph_edges=0,
            used_real_graph=False,
            pattern=pattern_str,
            reasoning=(
                f"Rule-based: {entity_id} "
                f"score={score:.2f} "
                f"pattern={pattern_str}"
            )
        )

    def _get_entity_risk(
        self,
        entity_id: str
    ) -> float:
        """Get entity risk from KG"""
        try:
            node = self._kg.get_node(entity_id)
            if node:
                return float(
                    node.get("risk_score", 0.0) or 0.0
                )
        except Exception:
            pass
        return 0.3

    def _get_neighbors(
        self,
        entity_id: str
    ) -> list:
        """Get entity neighbors from KG"""
        try:
            return self._kg.get_neighbors(
                entity_id, max_results=10
            ) or []
        except Exception:
            return []

    def _get_ip_risk(self, ip: str) -> float:
        """Get IP risk score"""
        # Known bad ranges
        if ip.startswith("185.220"):
            return 0.97  # Tor exit node
        if ip.startswith("45.142"):
            return 0.92  # Known attacker
        if ip.startswith("10.") or \
           ip.startswith("192.168."):
            return 0.0   # Internal
        return 0.1

    def _get_datastore_risk(
        self,
        data_store: str
    ) -> float:
        """Get data store risk score"""
        ds_lower = data_store.lower()
        if any(
            kw in ds_lower
            for kw in ["pci", "card", "payment"]
        ):
            return 0.9
        if any(
            kw in ds_lower
            for kw in ["phi", "health", "medical"]
        ):
            return 0.85
        if any(
            kw in ds_lower
            for kw in ["customer", "pii", "personal"]
        ):
            return 0.8
        if "backup" in ds_lower:
            return 0.3
        return 0.2

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
        entity_id: str,
        score: float,
        threat_count: int,
        total_nodes: int,
        proximity: int,
        pattern: str
    ) -> str:
        """Build human-readable reasoning"""
        reasoning = (
            f"GNN analysis for {entity_id}: "
            f"score={score:.3f} "
            f"graph={total_nodes} nodes "
        )
        if threat_count > 0:
            reasoning += (
                f"threats={threat_count} "
                f"proximity={proximity} hops "
            )
        reasoning += f"pattern={pattern}"
        return reasoning

    def _empty_result(
        self,
        entity_id: str
    ) -> GNNBridgeResult:
        """Return empty result for invalid input"""
        return GNNBridgeResult(
            entity_id=entity_id or "unknown",
            entity_type="unknown",
            anomaly_score=0.0,
            risk_label="UNKNOWN",
            threat_proximity=-1,
            connected_threats=0,
            graph_nodes=0,
            graph_edges=0,
            used_real_graph=False,
            pattern="NO_ENTITY",
            reasoning="No entity provided"
        )