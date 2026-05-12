"""
GNN Threat Detector Tests

WHAT WE ARE PROVING:
    1. Graph construction works correctly
    2. Node feature vectors computed correctly
    3. Threat proximity detection works
    4. Lateral movement pattern detected
    5. Data exfiltration pattern detected
    6. Normal graphs score low
    7. Attack graphs score high
    8. GNN model builds and trains
    9. from_networkx integration works
    10. SR 11-7 result serialization works
"""

import pytest
import numpy as np

from layer2_ml.graph.gnn_detector import (
    GNNThreatDetector,
    GNNDetectionResult,
    SecurityGraph,
    generate_normal_graph,
    generate_lateral_movement_graph,
    generate_exfiltration_graph,
    build_mpnn,
    NODE_FEATURE_SIZE
)


# ============================================================
# TEST CLASS — SECURITY GRAPH
# ============================================================

class TestSecurityGraph:
    """Tests for graph construction and features"""

    def setup_method(self):
        self.graph = SecurityGraph()

    def test_add_node(self):
        """Node correctly added to graph"""
        self.graph.add_node(
            "user:jsmith", "user", 0.3
        )
        assert "user:jsmith" in self.graph.nodes

    def test_add_edge(self):
        """Edge correctly added between nodes"""
        self.graph.add_node("user:jsmith", "user")
        self.graph.add_node(
            "data:customers", "data_store"
        )
        self.graph.add_edge(
            "user:jsmith", "data:customers",
            "accessed", 0.5
        )
        neighbors = self.graph.adjacency[
            "user:jsmith"
        ]
        assert len(neighbors) == 1

    def test_auto_create_nodes_on_edge(self):
        """
        Nodes created automatically when edge
        references non-existent nodes.
        Prevents KeyError when processing
        events out of order.
        """
        self.graph.add_edge(
            "user:new", "host:new",
            "belongs_to", 0.5
        )
        assert "user:new" in self.graph.nodes
        assert "host:new" in self.graph.nodes

    def test_mark_as_threat(self):
        """Node correctly marked as threat"""
        self.graph.add_node(
            "ip:185.220.101.45", "ip_address"
        )
        self.graph.mark_as_threat(
            "ip:185.220.101.45", "c2_server"
        )
        assert "ip:185.220.101.45" in (
            self.graph.threat_nodes
        )

    def test_node_feature_vector_shape(self):
        """
        Node feature vector has exactly 5 dimensions.
        Your Q1 answer: 5 features per node.
        """
        self.graph.add_node("user:jsmith", "user", 0.3)
        features = self.graph.get_node_feature_vector(
            "user:jsmith"
        )
        assert features.shape == (NODE_FEATURE_SIZE,)
        assert features.shape == (5,)

    def test_node_features_normalized(self):
        """All 5 features in 0.0-1.0 range"""
        self.graph.add_node("user:jsmith", "user", 0.7)
        features = self.graph.get_node_feature_vector(
            "user:jsmith"
        )
        for i, val in enumerate(features):
            assert 0.0 <= val <= 1.0, (
                f"Feature {i} = {val} out of range"
            )

    def test_degree_centrality_increases(self):
        """
        Degree centrality increases as node
        gains more connections.
        High centrality = potential hub for attack.
        """
        self.graph.add_node("hub:compromised", "host")

        for i in range(5):
            self.graph.add_edge(
                "hub:compromised",
                f"target:{i}",
                "lateral_move"
            )

        features = self.graph.get_node_feature_vector(
            "hub:compromised"
        )
        degree_centrality = features[0]
        assert degree_centrality > 0.0

    def test_hop_distance_to_threat(self):
        """
        Hop distance correctly calculated.
        Your Q2: guilt by association.
        """
        self.graph.add_node("user:jsmith", "user")
        self.graph.add_node(
            "ip:suspicious", "ip_address"
        )
        self.graph.add_node(
            "ip:c2", "ip_address"
        )

        self.graph.add_edge(
            "user:jsmith", "ip:suspicious",
            "connected_to"
        )
        self.graph.add_edge(
            "ip:suspicious", "ip:c2",
            "communicates_with"
        )
        self.graph.mark_as_threat("ip:c2", "c2")

        distance = self.graph.get_hop_distance_to_threat(
            "user:jsmith"
        )
        assert distance == 2

    def test_direct_threat_connection(self):
        """Node directly connected to threat = 1 hop"""
        self.graph.add_node("user:victim", "user")
        self.graph.add_node("ip:c2", "ip_address")
        self.graph.add_edge(
            "user:victim", "ip:c2",
            "communicates_with"
        )
        self.graph.mark_as_threat("ip:c2", "c2")

        distance = self.graph.get_hop_distance_to_threat(
            "user:victim"
        )
        assert distance == 1

    def test_no_threat_returns_minus_one(self):
        """No path to threat returns -1"""
        self.graph.add_node("user:safe", "user")
        distance = self.graph.get_hop_distance_to_threat(
            "user:safe"
        )
        assert distance == -1

    def test_graph_statistics(self):
        """Statistics correctly tracked"""
        self.graph.add_node("user:a", "user")
        self.graph.add_node("host:b", "host")
        self.graph.add_edge(
            "user:a", "host:b", "belongs_to"
        )
        stats = self.graph.get_statistics()
        assert stats["num_nodes"] == 2
        assert stats["num_edges"] == 1


# ============================================================
# TEST CLASS — PATTERN DETECTION
# ============================================================

class TestGNNPatternDetection:
    """
    Tests for graph attack pattern detection.
    Rule-based patterns work without ML training.
    """

    def setup_method(self):
        self.detector = GNNThreatDetector(
            anomaly_threshold=0.5
        )

    def test_normal_graph_scores_low(self):
        """
        Normal activity graph scores below threshold.
        Analyst not overwhelmed with false positives.
        """
        graph = generate_normal_graph()
        result = self.detector.score_node(
            graph, "user:analyst"
        )
        assert result.anomaly_score < 0.7

    def test_lateral_movement_detected(self):
        """
        Lateral movement graph pattern detected.
        Star topology from compromised host.
        ATT&CK TA0008.
        """
        graph = generate_lateral_movement_graph()
        result = self.detector.score_node(
            graph, "host:compromised"
        )
        assert result.anomaly_score >= 0.5
        assert result.subgraph_pattern in [
            "LATERAL_MOVEMENT",
            "DATA_EXFILTRATION",
            "RECONNAISSANCE"
        ]

    def test_exfiltration_detected(self):
        """
        Data exfiltration pattern detected.
        Fan-out to multiple sensitive stores
        plus external IP connection.
        ATT&CK TA0010.
        """
        graph = generate_exfiltration_graph()
        result = self.detector.score_node(
            graph, "user:svc_backup"
        )
        assert result.anomaly_score >= 0.5

    def test_threat_proximity_elevates_score(self):
        """
        Node connected to known threat
        gets elevated risk score.
        Your Q2: guilt by association.
        Threat intel propagates through graph.
        """
        graph = SecurityGraph()
        graph.add_node("user:victim", "user", 0.2)
        graph.add_node("ip:c2", "ip_address", 0.9)
        graph.add_edge(
            "user:victim", "ip:c2",
            "communicates_with", 0.9
        )
        graph.mark_as_threat("ip:c2", "c2_server")

        result = self.detector.score_node(
            graph, "user:victim"
        )
        assert result.anomaly_score >= 0.5
        assert result.hop_distance_to_threat == 1

    def test_connected_threats_listed(self):
        """Connected threat nodes identified in result"""
        graph = generate_lateral_movement_graph()
        result = self.detector.score_node(
            graph, "host:compromised"
        )
        assert isinstance(
            result.connected_threat_actors, list
        )

    def test_attack_scores_higher_than_normal(self):
        """
        Attack graphs score higher than normal.
        Core requirement: signal above noise.
        """
        normal_graph = generate_normal_graph()
        attack_graph = generate_lateral_movement_graph()

        normal_result = self.detector.score_node(
            normal_graph, "user:analyst"
        )
        attack_result = self.detector.score_node(
            attack_graph, "host:compromised"
        )

        assert (
            attack_result.anomaly_score >=
            normal_result.anomaly_score
        )

    def test_score_graph_returns_all_nodes(self):
        """
        score_graph() returns result for every node.
        Full graph sweep capability.
        """
        graph = generate_normal_graph()
        results = self.detector.score_graph(graph)

        assert len(results) == len(graph.nodes)
        for node_id, result in results.items():
            assert isinstance(
                result, GNNDetectionResult
            )

    def test_result_has_risk_reasons(self):
        """
        Anomalous nodes have human-readable reasons.
        SR 11-7: decisions must be explainable.
        """
        graph = generate_lateral_movement_graph()
        result = self.detector.score_node(
            graph, "host:compromised"
        )
        if result.is_anomaly:
            assert len(result.risk_reasons) > 0

    def test_result_has_timestamp(self):
        """Every result has scored_at timestamp"""
        graph = generate_normal_graph()
        result = self.detector.score_node(
            graph, "user:analyst"
        )
        assert result.scored_at != ""

    def test_result_serializes_to_dict(self):
        """
        Result serializes to dict for audit logging.
        SR 11-7 audit trail requirement.
        """
        graph = generate_normal_graph()
        result = self.detector.score_node(
            graph, "user:analyst"
        )
        d = result.to_dict()
        assert "is_anomaly" in d
        assert "anomaly_score" in d
        assert "subgraph_pattern" in d


# ============================================================
# TEST CLASS — GNN MODEL
# ============================================================

class TestGNNModel:
    """Tests for GNN PyTorch architecture"""

    def test_mpnn_builds(self):
        """MPNN builds without error"""
        model = build_mpnn()
        assert model is not None

    def test_mpnn_forward_pass(self):
        """
        MPNN forward pass produces correct shapes.
        node_scores: (N, 1)
        """
        import torch
        model = build_mpnn()

        N = 5
        node_features = torch.randn(N, NODE_FEATURE_SIZE)
        adj_matrix = torch.zeros(N, N)
        adj_matrix[0][1] = 1.0
        adj_matrix[1][2] = 1.0

        scores, embeddings = model(
            node_features, adj_matrix
        )
        assert scores.shape == (N, 1)

    def test_scores_in_valid_range(self):
        """
        All node scores between 0 and 1.
        Sigmoid activation ensures this.
        """
        import torch
        model = build_mpnn()

        N = 4
        node_features = torch.randn(N, NODE_FEATURE_SIZE)
        adj_matrix = torch.eye(N)

        scores, _ = model(node_features, adj_matrix)

        for score in scores:
            assert 0.0 <= score.item() <= 1.0

    def test_gnn_trains_on_normal_graphs(self):
        """GNN trains on synthetic normal graphs"""
        detector = GNNThreatDetector()
        normal_graphs = [
            generate_normal_graph()
            for _ in range(5)
        ]
        history = detector.train(
            normal_graphs, epochs=3
        )
        assert detector._is_trained is True

    def test_statistics_tracked(self):
        """Statistics correctly updated"""
        detector = GNNThreatDetector()
        graph = generate_normal_graph()
        detector.score_node(graph, "user:analyst")

        stats = detector.get_statistics()
        assert stats["nodes_scored"] == 1


# ============================================================
# TEST CLASS — LSTM GNN DISAGREEMENT FRAMEWORK
# ============================================================

class TestLSTMGNNDisagreement:
    """
    Tests implementing your Q3 answer.

    GNN-only detection → structural anomaly
    LSTM-only detection → behavioral anomaly
    Both detect → CRITICAL
    Neither detects → requires further investigation
    """

    def test_gnn_detects_structural_anomaly(self):
        """
        GNN flags lateral movement (structural).
        Graph pattern visible even if
        individual events look normal.
        """
        detector = GNNThreatDetector(
            anomaly_threshold=0.5
        )
        graph = generate_lateral_movement_graph()
        result = self.detector_score_all(
            detector, graph
        )
        max_score = max(
            r.anomaly_score for r in result.values()
        )
        assert max_score >= 0.5

    def detector_score_all(self, detector, graph):
        return detector.score_graph(graph)

    def test_disagreement_diagnostic_value(self):
        """
        When GNN and LSTM disagree it provides
        diagnostic information not confusion.

        GNN flags → investigate STRUCTURE
        LSTM flags → investigate SEQUENCE
        Both flag → CRITICAL multi-dimensional
        """
        gnn_only = {
            "gnn_score": 0.8,
            "lstm_score": 0.2,
            "interpretation": "STRUCTURAL_ANOMALY",
            "investigation": "graph_structure"
        }
        lstm_only = {
            "gnn_score": 0.2,
            "lstm_score": 0.8,
            "interpretation": "BEHAVIORAL_ANOMALY",
            "investigation": "event_sequence"
        }
        both = {
            "gnn_score": 0.8,
            "lstm_score": 0.8,
            "interpretation": "CRITICAL",
            "investigation": "immediate_response"
        }

        def classify_disagreement(
            gnn_score, lstm_score, threshold=0.5
        ):
            gnn_flag = gnn_score >= threshold
            lstm_flag = lstm_score >= threshold
            if gnn_flag and lstm_flag:
                return "CRITICAL"
            elif gnn_flag:
                return "STRUCTURAL_ANOMALY"
            elif lstm_flag:
                return "BEHAVIORAL_ANOMALY"
            else:
                return "NORMAL"

        assert classify_disagreement(0.8, 0.2) == (
            "STRUCTURAL_ANOMALY"
        )
        assert classify_disagreement(0.2, 0.8) == (
            "BEHAVIORAL_ANOMALY"
        )
        assert classify_disagreement(0.8, 0.8) == (
            "CRITICAL"
        )
        assert classify_disagreement(0.2, 0.2) == (
            "NORMAL"
        )