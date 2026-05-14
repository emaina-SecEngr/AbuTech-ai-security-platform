"""
Layer 2 — ML Processing
Graph Neural Network Threat Detector

RESEARCH CONTEXT:
    Implements graph-based threat detection
    using Message Passing Neural Networks (MPNN).

    "Graph Neural Networks for Enterprise Security:
     Structural Anomaly Detection Across
     Heterogeneous Security Knowledge Graphs"

    Eliud Maina — University of Arizona
    Abuhari Consulting Services LLC

WHY GNN FOR SECURITY:

    All previous models look at EVENTS (flat data).
    GNN looks at GRAPH STRUCTURE (relationships).

    The PATTERN OF CONNECTIONS reveals attacks
    that individual events cannot show:

    Lateral Movement:
        star pattern centered on compromised host
        multiple users accessed from one machine

    Data Exfiltration:
        fan-out from user to multiple sensitive stores
        + connection to external IP

    Supply Chain Attack:
        vendor node suddenly connected to C2
        + spreading to many internal systems

    Account Takeover:
        new nodes suddenly connected to
        established identity node

HOW GNN WORKS — MESSAGE PASSING:

    Step 1: Each node has a 5-feature vector.
    Step 2: Each node sends features to neighbors.
    Step 3: Each node aggregates neighbor messages.
    Step 4: Node representation updated.
    Step 5: Repeat for N layers (hops).

    After 3 layers: each node knows about
    everything 3 connections away.
    Threat intelligence propagates through graph.
    Known bad IP → connected host → connected user
    All elevated in risk automatically.

NODE FEATURE VECTOR (5 dimensions):
    [degree_centrality, recency_score,
     edge_type_entropy, mean_interaction_weight,
     cumulative_risk_score]

    Your Q1 answers implemented exactly.

EVASION HARDENING (your Q2 answers):
    1. Threat intel propagation (guilt by association)
    2. Behavioral fingerprinting (features betray attacker)
    3. Edge type encoding (operation intent revealed)

DISAGREEMENT FRAMEWORK (your Q3 answer):
    GNN flags, LSTM does not → structural anomaly
    LSTM flags, GNN does not → behavioral anomaly
    Both flag → CRITICAL multi-dimensional compromise
    Neither flags → requires third detection dimension
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Node feature size
NODE_FEATURE_SIZE = 5

# GNN hyperparameters
GNN_HIDDEN_SIZE = 32
GNN_NUM_LAYERS = 3
GNN_OUTPUT_SIZE = 16


# ============================================================
# NODE TYPES AND EDGE TYPES
#
# Maps your existing Layer 3 knowledge graph
# node and edge types to numeric encodings
# for GNN processing.
# ============================================================

NODE_TYPE_ENCODING = {
    "user":          0,
    "host":          1,
    "ip_address":    2,
    "domain":        3,
    "process":       4,
    "data_store":    5,
    "secret":        6,
    "threat_actor":  7,
    "vulnerability": 8,
    "file":          9,
    "unknown":       10
}

EDGE_TYPE_ENCODING = {
    "accessed":          0.2,
    "connected_to":      0.3,
    "belongs_to":        0.1,
    "checked_out":       0.6,
    "authenticated_from": 0.4,
    "resolved_to":       0.3,
    "contains":          0.2,
    "exploits":          0.9,
    "communicates_with": 0.5,
    "lateral_move":      0.8,
    "exfiltrated_to":    0.9,
    "unknown":           0.4
}

# Sensitivity of node types
# Used for risk propagation
NODE_SENSITIVITY = {
    "user":          0.5,
    "host":          0.4,
    "ip_address":    0.3,
    "domain":        0.3,
    "process":       0.4,
    "data_store":    0.7,
    "secret":        0.9,
    "threat_actor":  1.0,
    "vulnerability": 0.8,
    "file":          0.5,
    "unknown":       0.5
}


# ============================================================
# GNN RESULT
# ============================================================

@dataclass
class GNNDetectionResult:
    """
    Result from GNN graph anomaly detection.

    UNIQUE FIELDS VS OTHER DETECTORS:

    subgraph_pattern:
        What graph pattern was detected.
        "lateral_movement", "data_exfiltration",
        "supply_chain", "account_takeover"
        Tells analyst WHAT TYPE of attack.

    connected_threat_actors:
        Threat actors reachable from flagged node.
        Guilt by association — your Q2 answer.
        Node is suspicious because of WHO
        it is connected to.

    node_scores:
        Risk score per node in subgraph.
        Shows WHICH nodes are most suspicious.
        Analyst knows where to focus.

    hop_distance_to_threat:
        How many hops to nearest known threat.
        1 hop = directly connected to threat.
        3 hops = 3 degrees of separation.
        Used for risk scoring.
    """
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    confidence: str = "LOW"
    subgraph_pattern: str = "UNKNOWN"

    # Graph-specific fields
    node_id: str = ""
    node_type: str = ""
    node_scores: dict = field(default_factory=dict)
    connected_threat_actors: list = field(
        default_factory=list
    )
    hop_distance_to_threat: int = -1

    # Investigation context
    risk_reasons: list = field(default_factory=list)
    subgraph_nodes: list = field(default_factory=list)
    subgraph_edges: list = field(default_factory=list)
    scored_at: str = ""

    def to_dict(self) -> dict:
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "confidence": self.confidence,
            "subgraph_pattern": self.subgraph_pattern,
            "node_id": self.node_id,
            "node_type": self.node_type,
            "connected_threat_actors": (
                self.connected_threat_actors
            ),
            "hop_distance_to_threat": (
                self.hop_distance_to_threat
            ),
            "risk_reasons": self.risk_reasons,
            "scored_at": self.scored_at
        }


# ============================================================
# SECURITY GRAPH REPRESENTATION
#
# Lightweight graph structure for GNN processing.
# In production connect to your Layer 3
# NetworkX knowledge graph directly.
# ============================================================

class SecurityGraph:
    """
    Lightweight security knowledge graph
    for GNN processing.

    Represents nodes and edges with features.
    Supports message passing operations.

    IN PRODUCTION:
    Connect directly to Layer 3 NetworkX graph:
        from layer3_knowledge.graph.security_graph
            import SecurityKnowledgeGraph
        sg = SecurityKnowledgeGraph()
        gnn_graph = SecurityGraph.from_networkx(
            sg.graph
        )
    """

    def __init__(self):
        # Node storage
        # key: node_id
        # value: {type, features, risk_score}
        self.nodes = {}

        # Adjacency list
        # key: node_id
        # value: list of (neighbor_id, edge_type,
        #                  edge_weight)
        self.adjacency = defaultdict(list)

        # Known threat nodes
        # From your threat intelligence feeds
        self.threat_nodes = set()

        # Node count for statistics
        self.edge_count = 0

    def add_node(
        self,
        node_id: str,
        node_type: str,
        risk_score: float = 0.0,
        features: dict = None
    ) -> None:
        """
        Add a node to the security graph.

        NODE FEATURES (your Q1 answer):
        Computed from the node's history.
        Updated as new events arrive.
        """
        self.nodes[node_id] = {
            "type": node_type,
            "risk_score": risk_score,
            "features": features or {},
            "degree": 0,
            "last_seen": datetime.now(
                timezone.utc
            ).isoformat(),
            "edge_types_seen": set(),
            "interaction_weights": []
        }

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0
    ) -> None:
        """
        Add a directed edge between nodes.

        EDGE TYPE MATTERS (your Q2 answer):
        ACCESSED vs BULK_EXPORTED vs EXFILTRATED
        Operation intent encoded in edge type.
        Attacker cannot hide what operation
        they performed.
        """
        # Auto-create nodes if not exist
        if source_id not in self.nodes:
            self.add_node(source_id, "unknown")
        if target_id not in self.nodes:
            self.add_node(target_id, "unknown")

        self.adjacency[source_id].append(
            (target_id, edge_type, weight)
        )

        # Update node statistics
        self.nodes[source_id]["degree"] += 1
        self.nodes[source_id][
            "edge_types_seen"
        ].add(edge_type)
        self.nodes[source_id][
            "interaction_weights"
        ].append(weight)

        self.edge_count += 1

    def mark_as_threat(
        self,
        node_id: str,
        threat_type: str = "known_bad"
    ) -> None:
        """
        Mark a node as a known threat.

        CALLED WHEN:
        IP appears in AbuseIPDB feed.
        Domain appears in Feodo Tracker.
        Process matches malware hash.
        User appears in breach database.

        EFFECT:
        GNN propagates threat signal to
        all nodes connected to this node.
        Guilt by association implemented.
        """
        self.threat_nodes.add(node_id)
        if node_id in self.nodes:
            self.nodes[node_id]["risk_score"] = 1.0
            self.nodes[node_id]["is_threat"] = True
            self.nodes[node_id][
                "threat_type"
            ] = threat_type

    def get_node_feature_vector(
        self,
        node_id: str
    ) -> np.ndarray:
        """
        Compute 5-feature vector for a node.

        YOUR Q1 FEATURES:
        [0] degree_centrality
        [1] recency_score
        [2] edge_type_entropy
        [3] mean_interaction_weight
        [4] cumulative_risk_score

        All normalized to 0.0-1.0.
        """
        if node_id not in self.nodes:
            return np.zeros(
                NODE_FEATURE_SIZE,
                dtype=np.float32
            )

        node = self.nodes[node_id]
        total_nodes = max(len(self.nodes), 1)

        # [0] Degree centrality
        # How many connections does this node have?
        # Normalized by total nodes in graph.
        # High centrality = hub node.
        # Compromised hub = high blast radius.
        degree = node.get("degree", 0)
        degree_centrality = min(
            1.0, degree / max(total_nodes - 1, 1)
        )

        # [1] Recency score
        # How recently was this node active?
        # 1.0 = very recent (last hour)
        # 0.0 = not seen recently (days ago)
        # Recent activity = more relevant to
        # current investigation.
        recency = 0.5  # default
        last_seen = node.get("last_seen", "")
        if last_seen:
            try:
                last_dt = datetime.fromisoformat(
                    last_seen.replace("Z", "+00:00")
                )
                now = datetime.now(timezone.utc)
                hours_ago = (
                    now - last_dt
                ).total_seconds() / 3600
                recency = max(
                    0.0,
                    1.0 - (hours_ago / 24.0)
                )
            except Exception:
                pass

        # [2] Edge type entropy
        # How diverse are the types of connections?
        # YOUR Q1: Diversity Coefficient
        # Low entropy = predictable behavior (normal)
        # High entropy = diverse behavior (suspicious)
        edge_types = node.get("edge_types_seen", set())
        edge_entropy = 0.0
        if edge_types:
            n_types = len(edge_types)
            if n_types > 1:
                p = 1.0 / n_types
                edge_entropy = -n_types * p * math.log2(
                    p
                )
                edge_entropy = min(
                    1.0,
                    edge_entropy / math.log2(
                        len(EDGE_TYPE_ENCODING)
                    )
                )

        # [3] Mean interaction weight
        # Average strength of connections.
        # High-weight connections = more significant.
        weights = node.get("interaction_weights", [])
        mean_weight = float(
            np.mean(weights) if weights else 0.5
        )
        mean_weight = min(1.0, mean_weight)

        # [4] Cumulative risk score
        # Aggregated risk from all events
        # associated with this node.
        # YOUR Q1: the missing 5th feature.
        cumulative_risk = float(
            node.get("risk_score", 0.0)
        )

        return np.array([
            degree_centrality,
            recency,
            edge_entropy,
            mean_weight,
            cumulative_risk
        ], dtype=np.float32)

    def get_neighbors(
        self,
        node_id: str,
        max_hops: int = 2
    ) -> dict:
        """
        Get all neighbors within max_hops.
        Returns {node_id: hop_distance}.
        Used for threat proximity calculation.
        """
        visited = {node_id: 0}
        queue = [(node_id, 0)]

        while queue:
            current, hops = queue.pop(0)
            if hops >= max_hops:
                continue

            for neighbor, edge_type, weight in (
                self.adjacency.get(current, [])
            ):
                if neighbor not in visited:
                    visited[neighbor] = hops + 1
                    queue.append(
                        (neighbor, hops + 1)
                    )

        return visited

    def get_hop_distance_to_threat(
        self,
        node_id: str,
        max_hops: int = 3
    ) -> int:
        """
        Find shortest path to nearest threat node.

        YOUR Q2 ANSWER — GUILT BY ASSOCIATION:
        Node 3 hops from known bad IP → elevated risk.
        Node 1 hop from known C2 → high risk.
        Node 0 hops (IS the threat) → critical.

        This is the graph equivalent of your
        threat intelligence enrichment in Layer 3.
        """
        if not self.threat_nodes:
            return -1

        if node_id in self.threat_nodes:
            return 0

        neighbors = self.get_neighbors(
            node_id, max_hops
        )

        min_distance = -1
        for neighbor, distance in neighbors.items():
            if neighbor in self.threat_nodes:
                if (
                    min_distance == -1 or
                    distance < min_distance
                ):
                    min_distance = distance

        return min_distance

    @classmethod
    def from_networkx(cls, nx_graph):
        """
        Create SecurityGraph from your existing
        Layer 3 NetworkX knowledge graph.

        IN PRODUCTION USE THIS METHOD.
        Connects GNN directly to knowledge graph.
        No data duplication.

        Example:
            from layer3_knowledge.graph
                .security_graph import (
                    SecurityKnowledgeGraph
                )
            skg = SecurityKnowledgeGraph()
            graph = SecurityGraph.from_networkx(
                skg.graph
            )
        """
        sg = cls()

        for node_id, attrs in nx_graph.nodes(
            data=True
        ):
            node_type = attrs.get("type", "unknown")
            risk_score = attrs.get(
                "risk_score", 0.0
            )
            sg.add_node(
                str(node_id),
                node_type,
                float(risk_score),
                attrs
            )

        for source, target, attrs in (
            nx_graph.edges(data=True)
        ):
            edge_type = attrs.get(
                "relationship", "unknown"
            )
            weight = attrs.get("weight", 1.0)
            sg.add_edge(
                str(source),
                str(target),
                edge_type,
                float(weight)
            )

        return sg

    def get_statistics(self) -> dict:
        return {
            "num_nodes": len(self.nodes),
            "num_edges": self.edge_count,
            "num_threat_nodes": len(self.threat_nodes),
            "node_types": list(set(
                n["type"]
                for n in self.nodes.values()
            ))
        }


# ============================================================
# MESSAGE PASSING NEURAL NETWORK
# ============================================================

def build_mpnn():
    """
    Build Message Passing Neural Network.

    WHY MPNN NOT PyTorch Geometric:
    PyG requires complex installation.
    Pure PyTorch MPNN works for proof of concept.
    Same mathematical operation.
    Production: migrate to PyG or DGL.

    ARCHITECTURE:
        Input:   node features (5 per node)
        Layer 1: aggregate 1-hop neighborhood
        Layer 2: aggregate 2-hop neighborhood
        Layer 3: aggregate 3-hop neighborhood
        Output:  node embedding (16 dimensions)
        Score:   anomaly score per node

    WHY 3 LAYERS:
        Layer 1: knows about direct connections
        Layer 2: knows about connections of
                 connections
        Layer 3: knows about 3-hop neighborhood
                 Enough to detect most attack patterns.
                 More layers = more computation.
                 3 is the sweet spot for security.
    """
    try:
        import torch
        import torch.nn as nn

        class SecurityMPNN(nn.Module):
            def __init__(
                self,
                input_size=NODE_FEATURE_SIZE,
                hidden_size=GNN_HIDDEN_SIZE,
                output_size=GNN_OUTPUT_SIZE,
                num_layers=GNN_NUM_LAYERS
            ):
                super().__init__()
                self.num_layers = num_layers

                # MESSAGE FUNCTION:
                # Transforms neighbor features
                # before aggregation.
                # Learns which neighbor features
                # are most relevant.
                self.message_layers = nn.ModuleList([
                    nn.Linear(
                        input_size if i == 0
                        else hidden_size,
                        hidden_size
                    )
                    for i in range(num_layers)
                ])

                # AGGREGATION + UPDATE:
                # Combines own features with
                # aggregated neighbor messages.
                self.update_layers = nn.ModuleList([
                    nn.Linear(
                        hidden_size * 2,
                        hidden_size
                    )
                    for _ in range(num_layers)
                ])

                # OUTPUT:
                # Maps node embedding to
                # anomaly score.
                self.output_layer = nn.Sequential(
                    nn.Linear(hidden_size, output_size),
                    nn.ReLU(),
                    nn.Linear(output_size, 1),
                    nn.Sigmoid()
                )

                self.activation = nn.ReLU()

            def forward(
                self,
                node_features,
                adjacency_matrix
            ):
                """
                Forward pass through MPNN.

                node_features:    (N, 5)
                                  N nodes, 5 features
                adjacency_matrix: (N, N)
                                  1 if connected, else 0

                Returns:
                    node_scores: (N, 1)
                                 anomaly score per node
                    node_embeddings: (N, hidden)
                                     learned representations
                """
                import torch

                h = node_features

                for layer_idx in range(self.num_layers):
                    # MESSAGE PASSING:
                    # Transform features
                    messages = self.activation(
                        self.message_layers[layer_idx](h)
                    )

                    # AGGREGATION:
                    # Sum neighbor messages.
                    # adjacency_matrix @ messages =
                    # sum of neighbor embeddings
                    aggregated = torch.mm(
                        adjacency_matrix, messages
                    )

                    # Normalize by degree
                    # Prevents high-degree nodes
                    # from dominating
                    degree = adjacency_matrix.sum(
                        dim=1, keepdim=True
                    ).clamp(min=1)
                    aggregated = aggregated / degree

                    # UPDATE:
                    # Combine own embedding
                    # with aggregated neighbors
                    combined = torch.cat([h, aggregated], dim=1) if (
                        h.shape[1] == aggregated.shape[1]
                    ) else torch.cat([
                        messages, aggregated
                    ], dim=1)

                    h = self.activation(
                        self.update_layers[layer_idx](
                            combined
                        )
                    )

                # Score each node
                node_scores = self.output_layer(h)
                return node_scores, h

        return SecurityMPNN()

    except ImportError:
        return None


# ============================================================
# GNN DETECTOR
# ============================================================

class GNNThreatDetector:
    """
    Graph Neural Network threat detector.

    Detects structural anomalies in the
    security knowledge graph that indicate:
    - Lateral movement
    - Data exfiltration
    - Supply chain attacks
    - Account takeover

    COMPLEMENTS LSTM + ATTENTION:
    LSTM detects TEMPORAL anomalies (sequences)
    GNN detects STRUCTURAL anomalies (graph patterns)

    YOUR Q3 ANSWER:
    Disagreement is diagnostic not a failure.
    GNN-only → investigate graph structure.
    LSTM-only → investigate behavioral sequence.
    Both → CRITICAL multi-dimensional compromise.

    Usage:
        detector = GNNThreatDetector()
        graph = SecurityGraph()
        # populate graph from events
        result = detector.score_node(
            graph, "user:jsmith"
        )
    """

    def __init__(
        self,
        anomaly_threshold: float = 0.6,
        threat_hop_threshold: int = 2
    ):
        self.anomaly_threshold = anomaly_threshold
        self.threat_hop_threshold = threat_hop_threshold

        self.model = None
        self._is_trained = False

        # Statistics
        self.nodes_scored = 0
        self.anomalies_detected = 0
        self.lateral_movement_detected = 0
        self.exfiltration_detected = 0

        logger.info(
            "GNNThreatDetector initialized"
        )

    def train(
        self,
        normal_graphs: list,
        epochs: int = 30
    ) -> dict:
        """
        Train GNN on normal graph structures.

        UNSUPERVISED APPROACH:
        Train on normal graphs only.
        Learn what normal structure looks like.
        Anomalous structures = high reconstruction error.

        normal_graphs: list of SecurityGraph objects
                      representing normal activity
        """
        history = {
            "final_loss": None,
            "epochs_run": 0,
            "graphs_trained": 0
        }

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            self.model = build_mpnn()
            if self.model is None:
                return history

            optimizer = optim.Adam(
                self.model.parameters(),
                lr=0.001
            )
            criterion = nn.MSELoss()

            losses = []

            for epoch in range(epochs):
                epoch_loss = 0.0
                n_graphs = 0

                for graph in normal_graphs:
                    if not graph.nodes:
                        continue

                    # Build tensors from graph
                    node_features, adj_matrix = (
                        self._graph_to_tensors(graph)
                    )

                    if node_features is None:
                        continue

                    optimizer.zero_grad()

                    # Forward pass
                    scores, embeddings = self.model(
                        node_features, adj_matrix
                    )

                    # Target: all normal = 0
                    target = torch.zeros_like(scores)
                    loss = criterion(scores, target)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_graphs += 1

                if n_graphs > 0:
                    avg_loss = epoch_loss / n_graphs
                    losses.append(avg_loss)

                    if epoch % 10 == 0:
                        logger.debug(
                            f"GNN epoch {epoch}: "
                            f"loss={avg_loss:.6f}"
                        )

            self._is_trained = True
            history["final_loss"] = (
                losses[-1] if losses else None
            )
            history["epochs_run"] = epochs
            history["graphs_trained"] = len(
                normal_graphs
            )

            logger.info(
                f"GNN trained on "
                f"{len(normal_graphs)} graphs"
            )

        except Exception as e:
            logger.error(
                f"GNN training failed: {e}"
            )

        return history

    def score_node(
        self,
        graph: SecurityGraph,
        node_id: str
    ) -> GNNDetectionResult:
        """
        Score a specific node for anomalies.

        COMBINES:
        1. ML-based structural scoring (if trained)
        2. Rule-based pattern detection (always)
        3. Threat proximity scoring (guilt by association)
        """
        result = GNNDetectionResult(
            node_id=node_id,
            node_type=graph.nodes.get(
                node_id, {}
            ).get("type", "unknown"),
            scored_at=self._now()
        )

        if node_id not in graph.nodes:
            return result

        # Threat proximity (guilt by association)
        hop_distance = graph.get_hop_distance_to_threat(
            node_id
        )
        result.hop_distance_to_threat = hop_distance

        # Find connected threat actors
        if hop_distance >= 0:
            neighbors = graph.get_neighbors(
                node_id, hop_distance + 1
            )
            for neighbor in neighbors:
                if neighbor in graph.threat_nodes:
                    result.connected_threat_actors.append(
                        neighbor
                    )

        # ML scoring if model trained
        ml_score = 0.0
        if self._is_trained and self.model is not None:
            ml_score = self._score_with_model(
                graph, node_id
            )

        # Rule-based pattern detection
        pattern_score, pattern_name, pattern_reasons = (
            self._detect_graph_patterns(graph, node_id)
        )

        # Threat proximity score
        proximity_score = self._score_threat_proximity(
            hop_distance
        )

        # Combine scores
        combined_score = max(
            ml_score,
            pattern_score,
            proximity_score
        )

        result.anomaly_score = round(
            combined_score, 3
        )
        result.is_anomaly = (
            combined_score >= self.anomaly_threshold
        )
        result.subgraph_pattern = pattern_name
        result.risk_reasons = pattern_reasons.copy()
        result.confidence = (
            "HIGH" if combined_score >= 0.8
            else "MEDIUM" if combined_score >= 0.6
            else "LOW"
        )

        # Add threat proximity reason
        if proximity_score > 0:
            result.risk_reasons.append(
                f"Threat proximity: node is "
                f"{hop_distance} hop(s) from "
                f"known threat node. "
                f"Connected threats: "
                f"{result.connected_threat_actors}"
            )

        if result.is_anomaly:
            self.anomalies_detected += 1
            if pattern_name == "LATERAL_MOVEMENT":
                self.lateral_movement_detected += 1
            elif pattern_name == "DATA_EXFILTRATION":
                self.exfiltration_detected += 1

        self.nodes_scored += 1

        return result

    def score_graph(
        self,
        graph: SecurityGraph
    ) -> dict:
        """
        Score ALL nodes in the graph.
        Returns dict of node_id → GNNDetectionResult.
        Used for full graph sweep.
        """
        results = {}
        for node_id in graph.nodes:
            results[node_id] = self.score_node(
                graph, node_id
            )
        return results

    # ============================================================
    # GRAPH PATTERN DETECTION
    # ============================================================

    def _detect_graph_patterns(
        self,
        graph: SecurityGraph,
        node_id: str
    ) -> tuple:
        """
        Detect known attack graph patterns.

        PATTERNS DETECTED:

        LATERAL MOVEMENT:
            Star pattern — one node connected
            to many hosts and user accounts.
            Attacker pivoting through network.

        DATA EXFILTRATION:
            Fan-out to multiple sensitive stores
            plus connection to external IP.
            Data being stolen.

        ACCOUNT_TAKEOVER:
            New nodes suddenly connected to
            established identity node.
            New device, new IP, new location.

        SUPPLY_CHAIN:
            Vendor/software node connected
            to unexpected internal systems
            AND external C2.
        """
        node = graph.nodes.get(node_id, {})
        node_type = node.get("type", "unknown")
        score = 0.0
        pattern = "NORMAL"
        reasons = []

        neighbors = graph.adjacency.get(node_id, [])
        neighbor_types = [
            graph.nodes.get(n, {}).get("type", "unknown")
            for n, _, _ in neighbors
        ]

        # ---- LATERAL MOVEMENT DETECTION ----
        # Many hosts connected to one node
        host_connections = neighbor_types.count("host")
        user_connections = neighbor_types.count("user")

        if (
            host_connections >= 3 or
            (host_connections >= 2 and
             user_connections >= 2)
        ):
            score = max(score, 0.75)
            pattern = "LATERAL_MOVEMENT"
            reasons.append(
                f"Lateral movement pattern: "
                f"node connected to {host_connections} "
                f"hosts and {user_connections} users. "
                f"Star topology indicates "
                f"pivoting activity. "
                f"ATT&CK TA0008"
            )

        # ---- DATA EXFILTRATION DETECTION ----
        # Multiple data stores + external IP
        data_store_connections = (
            neighbor_types.count("data_store")
        )
        ip_connections = neighbor_types.count(
            "ip_address"
        )

        if (
            data_store_connections >= 2 and
            ip_connections >= 1
        ):
            # Check if any IP is external
            external_ips = []
            for n_id, edge_type, weight in neighbors:
                n_type = graph.nodes.get(
                    n_id, {}
                ).get("type", "")
                if n_type == "ip_address":
                    n_risk = graph.nodes.get(
                        n_id, {}
                    ).get("risk_score", 0)
                    if n_risk > 0.5:
                        external_ips.append(n_id)

            if external_ips or ip_connections >= 1:
                score = max(score, 0.80)
                pattern = "DATA_EXFILTRATION"
                reasons.append(
                    f"Data exfiltration pattern: "
                    f"node accessed "
                    f"{data_store_connections} "
                    f"data stores and connected to "
                    f"{ip_connections} IP addresses. "
                    f"Fan-out topology indicates "
                    f"data staging for exfiltration. "
                    f"ATT&CK TA0010"
                )

        # ---- ACCOUNT TAKEOVER DETECTION ----
        # New high-risk connections to user node
        if node_type == "user":
            node_risk = float(
                node.get("risk_score", 0)
            )
            if node_risk > 0.6:
                new_ip_connections = sum(
                    1 for n_id, _, _ in neighbors
                    if graph.nodes.get(
                        n_id, {}
                    ).get("type") == "ip_address" and
                    graph.nodes.get(
                        n_id, {}
                    ).get("risk_score", 0) > 0.5
                )

                if new_ip_connections >= 1:
                    score = max(score, 0.70)
                    pattern = "ACCOUNT_TAKEOVER"
                    reasons.append(
                        f"Account takeover pattern: "
                        f"high-risk user node "
                        f"connected to "
                        f"{new_ip_connections} "
                        f"suspicious IP addresses. "
                        f"ATT&CK T1078"
                    )

        # ---- UNUSUAL EDGE DIVERSITY ----
        # High entropy of edge types = reconnaissance
        edge_types = [
            et for _, et, _ in neighbors
        ]
        if len(set(edge_types)) >= 4:
            score = max(score, 0.55)
            if pattern == "NORMAL":
                pattern = "RECONNAISSANCE"
            reasons.append(
                f"High edge type diversity: "
                f"{len(set(edge_types))} different "
                f"operation types detected. "
                f"Possible reconnaissance activity. "
                f"ATT&CK TA0043"
            )

        return score, pattern, reasons

    def _score_threat_proximity(
        self,
        hop_distance: int
    ) -> float:
        """
        Score based on proximity to known threats.

        YOUR Q2 ANSWER — GUILT BY ASSOCIATION:
        0 hops (IS threat):     1.0 critical
        1 hop (direct connect): 0.8 high
        2 hops:                 0.5 medium
        3 hops:                 0.3 low
        No connection:          0.0

        This propagates threat intelligence
        from your Layer 3 feeds through the graph.
        Same mechanism as:
        Layer 3 ThreatEnricher.enrich_with_feeds()
        But applied at graph structure level.
        """
        if hop_distance == 0:
            return 1.0
        elif hop_distance == 1:
            return 0.8
        elif hop_distance == 2:
            return 0.5
        elif hop_distance == 3:
            return 0.3
        else:
            return 0.0

    def _score_with_model(
        self,
        graph: SecurityGraph,
        node_id: str
    ) -> float:
        """Score node using trained MPNN"""
        try:
            import torch

            node_features, adj_matrix = (
                self._graph_to_tensors(graph)
            )

            if node_features is None:
                return 0.0

            self.model.eval()
            with torch.no_grad():
                scores, _ = self.model(
                    node_features, adj_matrix
                )

            # Get score for specific node
            node_ids = list(graph.nodes.keys())
            if node_id in node_ids:
                idx = node_ids.index(node_id)
                return float(scores[idx].item())

        except Exception as e:
            logger.debug(
                f"Model scoring failed: {e}"
            )

        return 0.0

    def _graph_to_tensors(
        self,
        graph: SecurityGraph
    ) -> tuple:
        """
        Convert SecurityGraph to PyTorch tensors.

        Returns:
            node_features: (N, 5) tensor
            adj_matrix:    (N, N) tensor
        """
        try:
            import torch

            if not graph.nodes:
                return None, None

            node_ids = list(graph.nodes.keys())
            n = len(node_ids)
            node_to_idx = {
                nid: i for i, nid in enumerate(node_ids)
            }

            # Build feature matrix
            features = np.stack([
                graph.get_node_feature_vector(nid)
                for nid in node_ids
            ])

            # Build adjacency matrix
            adj = np.zeros((n, n), dtype=np.float32)
            for source_id, neighbors in (
                graph.adjacency.items()
            ):
                if source_id not in node_to_idx:
                    continue
                i = node_to_idx[source_id]
                for target_id, edge_type, weight in (
                    neighbors
                ):
                    if target_id in node_to_idx:
                        j = node_to_idx[target_id]
                        edge_weight = EDGE_TYPE_ENCODING.get(
                            edge_type, 0.4
                        )
                        adj[i][j] = edge_weight

            return (
                torch.FloatTensor(features),
                torch.FloatTensor(adj)
            )

        except Exception as e:
            logger.debug(
                f"Graph to tensor failed: {e}"
            )
            return None, None

    def get_statistics(self) -> dict:
        return {
            "model_trained": self._is_trained,
            "nodes_scored": self.nodes_scored,
            "anomalies_detected": (
                self.anomalies_detected
            ),
            "lateral_movement_detected": (
                self.lateral_movement_detected
            ),
            "exfiltration_detected": (
                self.exfiltration_detected
            ),
            "anomaly_threshold": (
                self.anomaly_threshold
            )
        }

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )


# ============================================================
# SYNTHETIC GRAPH GENERATORS
# ============================================================

def generate_normal_graph() -> SecurityGraph:
    """
    Generate a synthetic normal activity graph.

    Normal characteristics:
    - Low connectivity (predictable paths)
    - No connections to threat nodes
    - Low edge type diversity
    - Regular access patterns
    """
    graph = SecurityGraph()

    # Normal user accessing expected resources
    graph.add_node("user:analyst", "user", 0.1)
    graph.add_node("host:laptop_001", "host", 0.1)
    graph.add_node("app:salesforce", "data_store", 0.2)
    graph.add_node("app:email", "data_store", 0.1)
    graph.add_node("ip:10.0.0.155", "ip_address", 0.1)

    graph.add_edge(
        "user:analyst", "host:laptop_001",
        "belongs_to", 0.9
    )
    graph.add_edge(
        "user:analyst", "app:salesforce",
        "accessed", 0.3
    )
    graph.add_edge(
        "user:analyst", "app:email",
        "accessed", 0.3
    )
    graph.add_edge(
        "host:laptop_001", "ip:10.0.0.155",
        "connected_to", 0.5
    )

    return graph


def generate_lateral_movement_graph() -> SecurityGraph:
    """
    Generate a synthetic lateral movement graph.

    Attacker pivoting from compromised host
    to multiple other systems.
    """
    graph = SecurityGraph()

    # Compromised host connecting to everything
    graph.add_node("host:compromised", "host", 0.9)
    graph.add_node("user:admin1", "user", 0.3)
    graph.add_node("user:admin2", "user", 0.3)
    graph.add_node("user:admin3", "user", 0.3)
    graph.add_node("host:dc01", "host", 0.2)
    graph.add_node("host:fileserver", "host", 0.2)
    graph.add_node(
        "ip:185.220.101.45", "ip_address", 0.9
    )

    # Star pattern from compromised host
    graph.add_edge(
        "host:compromised", "user:admin1",
        "lateral_move", 0.8
    )
    graph.add_edge(
        "host:compromised", "user:admin2",
        "lateral_move", 0.8
    )
    graph.add_edge(
        "host:compromised", "user:admin3",
        "lateral_move", 0.8
    )
    graph.add_edge(
        "host:compromised", "host:dc01",
        "connected_to", 0.7
    )
    graph.add_edge(
        "host:compromised", "host:fileserver",
        "connected_to", 0.7
    )
    graph.add_edge(
        "host:compromised", "ip:185.220.101.45",
        "communicates_with", 0.9
    )

    # Mark C2 as threat
    graph.mark_as_threat(
        "ip:185.220.101.45", "c2_server"
    )

    return graph


def generate_exfiltration_graph() -> SecurityGraph:
    """
    Generate a synthetic data exfiltration graph.
    """
    graph = SecurityGraph()

    graph.add_node("user:svc_backup", "user", 0.6)
    graph.add_node(
        "data:customer_pii", "data_store", 0.8
    )
    graph.add_node(
        "data:payment_cards", "data_store", 0.9
    )
    graph.add_node(
        "data:employee_records", "data_store", 0.7
    )
    graph.add_node(
        "ip:185.220.101.45", "ip_address", 0.9
    )

    # Fan-out to multiple sensitive stores
    graph.add_edge(
        "user:svc_backup", "data:customer_pii",
        "accessed", 0.8
    )
    graph.add_edge(
        "user:svc_backup", "data:payment_cards",
        "accessed", 0.9
    )
    graph.add_edge(
        "user:svc_backup", "data:employee_records",
        "accessed", 0.7
    )
    graph.add_edge(
        "user:svc_backup", "ip:185.220.101.45",
        "communicates_with", 0.9
    )

    graph.mark_as_threat(
        "ip:185.220.101.45", "exfil_destination"
    )

    return graph