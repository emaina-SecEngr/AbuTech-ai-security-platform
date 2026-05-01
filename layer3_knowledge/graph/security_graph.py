"""
Layer 3 — Knowledge Graph
Core Security Graph Data Structure

This module defines the graph data structure that
connects all security entities across your platform.

Why a Graph Instead of a Database Table:
    A relational database answers: "What happened?"
    A knowledge graph answers: "How does everything
    connect and what does that mean?"

    Traditional SIEM query:
        SELECT * FROM events
        WHERE host = 'WKSTN-JSMITH-01'
        Returns: a list of unconnected events

    Knowledge graph query:
        "Show me everything connected to
        WKSTN-JSMITH-01 within 24 hours"
        Returns: a connected subgraph showing
        malware execution → C2 DNS → C2 IP →
        threat actor → campaign → other victims

Graph Theory Basics Applied to Security:
    Node = any security entity
           (host, user, IP, domain, process, file)
    Edge = relationship between entities
           (connected_to, resolved, executed, belongs_to)
    Path = chain of relationships
           (the attack kill chain)
    Subgraph = cluster of related entities
               (the attack campaign infrastructure)

Technology:
    NetworkX — Python graph library
    In-memory for development and testing
    Neo4j — production graph database (Layer 5 upgrade)
"""

import logging
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from enum import Enum
from typing import Optional
import networkx as nx

logger = logging.getLogger(__name__)


# ============================================================
# ENTITY AND RELATIONSHIP DEFINITIONS
# ============================================================

class NodeType(Enum):
    """
    Types of security entities in the graph.
    Every node has exactly one type.
    """
    HOST = "host"
    USER = "user"
    PROCESS = "process"
    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    FILE = "file"
    ALERT = "alert"
    THREAT_ACTOR = "threat_actor"
    CAMPAIGN = "campaign"
    VULNERABILITY = "vulnerability"


class EdgeType(Enum):
    """
    Types of relationships between entities.
    Every edge has exactly one type.
    """
    # Process relationships
    EXECUTED = "executed"           # Host executed Process
    SPAWNED = "spawned"             # Process spawned Process
    LOADED = "loaded"               # Process loaded File

    # Network relationships
    CONNECTED_TO = "connected_to"   # Process/Host → IP
    RESOLVED = "resolved"           # Process resolved Domain
    RESOLVES_TO = "resolves_to"     # Domain resolves to IP

    # Identity relationships
    AUTHENTICATED_AS = "authenticated_as"  # Host → User
    BELONGS_TO = "belongs_to"       # Host belongs to User

    # Threat intelligence relationships
    ATTRIBUTED_TO = "attributed_to" # IP/Domain → Actor
    PART_OF = "part_of"             # Entity → Campaign
    TARGETS = "targets"             # Actor targets Host/User

    # Detection relationships
    TRIGGERED = "triggered"         # Event triggered Alert
    INDICATES = "indicates"         # Entity indicates Alert


@dataclass
class SecurityNode:
    """
    A security entity node in the knowledge graph.

    Every entity in your security environment
    becomes a node — hosts, users, IPs, domains,
    processes, files, threat actors, campaigns.

    The risk_score starts at the Layer 2 ML score
    and increases as the graph adds context.
    This is where graph enrichment happens —
    an IP that looks benign alone becomes HIGH risk
    when connected to DGA domains and malware.
    """
    node_id: str           # Unique identifier
    node_type: NodeType    # What kind of entity
    label: str             # Human readable name

    # Risk assessment — updated by graph enrichment
    risk_score: float = 0.0
    risk_label: str = "UNKNOWN"

    # Timestamps
    first_seen: str = ""
    last_seen: str = ""

    # Source of this node
    data_source: str = ""

    # Additional properties
    properties: dict = field(default_factory=dict)

    def update_risk(self, new_score: float) -> None:
        """
        Update risk score — always take the highest.
        Risk can only increase through enrichment
        never decrease. If any evidence points to
        malicious activity the node stays elevated.
        """
        if new_score > self.risk_score:
            self.risk_score = new_score
            self.risk_label = self._score_to_label(
                new_score
            )

    def _score_to_label(self, score: float) -> str:
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score > 0.0:
            return "LOW"
        return "UNKNOWN"

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "risk_score": self.risk_score,
            "risk_label": self.risk_label,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "data_source": self.data_source,
            "properties": self.properties
        }


@dataclass
class SecurityEdge:
    """
    A relationship between two security entities.

    Edges encode what happened between entities.
    The combination of nodes and edges tells
    the complete attack story.

    Example:
        powershell.exe (node)
        SPAWNED_BY (edge)
        MSBuild.exe (node)
        → T1127 Trusted Developer Utilities

        svchost.exe (node)
        RESOLVED (edge)
        xjf8k2mp.duckdns.org (node)
        RESOLVES_TO (edge)
        185.220.101.45 (node)
        → Active C2 communication path
    """
    source_id: str
    target_id: str
    edge_type: EdgeType

    # When this relationship was observed
    timestamp: str = ""
    first_seen: str = ""
    last_seen: str = ""

    # Confidence in this relationship
    confidence: float = 1.0

    # Additional context
    properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "properties": self.properties
        }


class SecurityKnowledgeGraph:
    """
    The core knowledge graph for your security platform.

    Wraps NetworkX DiGraph with security-specific
    methods for adding entities, querying relationships,
    and computing risk propagation.

    This graph grows continuously as events flow
    through your platform. Every normalized event
    from Layer 1 and every detection from Layer 2
    adds nodes and edges to this graph.

    Usage:
        graph = SecurityKnowledgeGraph()

        # Add entities
        graph.add_host("WKSTN-JSMITH-01")
        graph.add_ip("185.220.101.45", risk_score=0.9)

        # Add relationships
        graph.add_edge(
            source_id="svchost.exe:WKSTN-JSMITH-01",
            target_id="185.220.101.45",
            edge_type=EdgeType.CONNECTED_TO
        )

        # Query
        neighbors = graph.get_neighbors(
            "185.220.101.45"
        )
        risk = graph.compute_node_risk(
            "185.220.101.45"
        )
    """

    def __init__(self):
        # NetworkX directed graph
        # Directed because relationships have direction
        # svchost CONNECTED_TO 185.x.x.x
        # not 185.x.x.x CONNECTED_TO svchost
        self.graph = nx.DiGraph()

        # Node registry for fast lookup
        self.nodes: dict[str, SecurityNode] = {}

        # Statistics
        self.total_events_processed = 0
        self.total_nodes = 0
        self.total_edges = 0

        logger.info("SecurityKnowledgeGraph initialized")

    # ============================================================
    # NODE MANAGEMENT
    # ============================================================

    def add_node(
        self,
        node: SecurityNode
    ) -> SecurityNode:
        """
        Add or update a node in the graph.

        If the node already exists updates its
        risk score and last_seen timestamp.
        This prevents duplicate nodes while
        continuously enriching existing ones.
        """
        if node.node_id in self.nodes:
            # Update existing node
            existing = self.nodes[node.node_id]
            existing.update_risk(node.risk_score)
            existing.last_seen = (
                self._current_timestamp()
            )
            # Merge properties
            existing.properties.update(
                node.properties
            )
            return existing

        # Set timestamps for new node
        now = self._current_timestamp()
        node.first_seen = now
        node.last_seen = now

        # Add to graph and registry
        self.nodes[node.node_id] = node
        self.graph.add_node(
            node.node_id,
            **node.to_dict()
        )
        self.total_nodes += 1

        return node

    def add_host(
        self,
        hostname: str,
        risk_score: float = 0.0,
        properties: dict = None
    ) -> SecurityNode:
        """Add a host entity to the graph"""
        node = SecurityNode(
            node_id=f"host:{hostname}",
            node_type=NodeType.HOST,
            label=hostname,
            risk_score=risk_score,
            properties=properties or {}
        )
        return self.add_node(node)

    def add_user(
        self,
        username: str,
        domain: str = "",
        risk_score: float = 0.0
    ) -> SecurityNode:
        """Add a user identity entity"""
        full_name = (
            f"{domain}\\{username}"
            if domain else username
        )
        node = SecurityNode(
            node_id=f"user:{full_name}",
            node_type=NodeType.USER,
            label=full_name,
            risk_score=risk_score,
            properties={
                "username": username,
                "domain": domain
            }
        )
        return self.add_node(node)

    def add_process(
        self,
        process_name: str,
        host: str,
        risk_score: float = 0.0,
        properties: dict = None
    ) -> SecurityNode:
        """Add a process entity"""
        node_id = f"process:{process_name}:{host}"
        node = SecurityNode(
            node_id=node_id,
            node_type=NodeType.PROCESS,
            label=process_name,
            risk_score=risk_score,
            properties={
                "process_name": process_name,
                "host": host,
                **(properties or {})
            }
        )
        return self.add_node(node)

    def add_ip(
        self,
        ip_address: str,
        risk_score: float = 0.0,
        properties: dict = None
    ) -> SecurityNode:
        """Add an IP address entity"""
        node = SecurityNode(
            node_id=f"ip:{ip_address}",
            node_type=NodeType.IP_ADDRESS,
            label=ip_address,
            risk_score=risk_score,
            properties=properties or {}
        )
        return self.add_node(node)

    def add_domain(
        self,
        domain: str,
        risk_score: float = 0.0,
        properties: dict = None
    ) -> SecurityNode:
        """Add a domain name entity"""
        node = SecurityNode(
            node_id=f"domain:{domain}",
            node_type=NodeType.DOMAIN,
            label=domain,
            risk_score=risk_score,
            properties=properties or {}
        )
        return self.add_node(node)

    def add_alert(
        self,
        alert_id: str,
        alert_type: str,
        risk_score: float = 0.0,
        properties: dict = None
    ) -> SecurityNode:
        """Add a detection alert entity"""
        node = SecurityNode(
            node_id=f"alert:{alert_id}",
            node_type=NodeType.ALERT,
            label=alert_type,
            risk_score=risk_score,
            properties={
                "alert_type": alert_type,
                **(properties or {})
            }
        )
        return self.add_node(node)

    # ============================================================
    # EDGE MANAGEMENT
    # ============================================================

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        confidence: float = 1.0,
        properties: dict = None,
        timestamp: str = None
    ) -> Optional[SecurityEdge]:
        """
        Add a directed relationship between two nodes.

        Both nodes must exist in the graph.
        Returns None if either node is missing.
        """
        if source_id not in self.nodes:
            logger.warning(
                f"Source node not found: {source_id}"
            )
            return None

        if target_id not in self.nodes:
            logger.warning(
                f"Target node not found: {target_id}"
            )
            return None

        now = timestamp or self._current_timestamp()

        edge = SecurityEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            timestamp=now,
            first_seen=now,
            last_seen=now,
            confidence=confidence,
            properties=properties or {}
        )

        edge_data = edge.to_dict()
        self.graph.add_edge(
    source_id,
    target_id,
    **edge_data
        )
        self.total_edges += 1

        return edge

    # ============================================================
    # GRAPH QUERIES
    # ============================================================

    def get_node(
        self,
        node_id: str
    ) -> Optional[SecurityNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)

    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both"
    ) -> list:
        """
        Get all neighbors of a node.

        direction:
            "out"  = nodes this node points to
            "in"   = nodes that point to this node
            "both" = all connected nodes
        """
        if node_id not in self.graph:
            return []

        neighbors = []

        if direction in ("out", "both"):
            for neighbor_id in self.graph.successors(
                node_id
            ):
                node = self.nodes.get(neighbor_id)
                if node:
                    neighbors.append(node)

        if direction in ("in", "both"):
            for neighbor_id in self.graph.predecessors(
                node_id
            ):
                node = self.nodes.get(neighbor_id)
                if node:
                    neighbors.append(node)

        return neighbors

    def get_attack_path(
        self,
        source_id: str,
        target_id: str
    ) -> list:
        """
        Find the shortest path between two entities.

        Used to trace attack kill chains:
            How did the attacker get from
            initial access to target?

        Returns list of node IDs in the path.
        """
        try:
            path = nx.shortest_path(
                self.graph,
                source_id,
                target_id
            )
            return path
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []

    def get_connected_component(
        self,
        node_id: str
    ) -> list:
        """
        Get all nodes reachable from a given node.

        Used to find the complete blast radius
        of a compromise:
            "Show me everything connected to
            this compromised host"

        Returns list of SecurityNode objects.
        """
        if node_id not in self.graph:
            return []

        # Use undirected version for reachability
        undirected = self.graph.to_undirected()

        try:
            component = nx.node_connected_component(
                undirected, node_id
            )
            return [
                self.nodes[n]
                for n in component
                if n in self.nodes
            ]
        except nx.NodeNotFound:
            return []

    def get_high_risk_nodes(
        self,
        threshold: float = 0.6
    ) -> list:
        """
        Return all nodes above a risk threshold.

        Used by Layer 4 agents to identify
        which entities need investigation.
        """
        return [
            node for node in self.nodes.values()
            if node.risk_score >= threshold
        ]

    def get_nodes_by_type(
        self,
        node_type: NodeType
    ) -> list:
        """Return all nodes of a specific type"""
        return [
            node for node in self.nodes.values()
            if node.node_type == node_type
        ]

    # ============================================================
    # RISK PROPAGATION
    # ============================================================

    def propagate_risk(
        self,
        node_id: str,
        decay_factor: float = 0.7
    ) -> None:
        """
        Propagate risk from a high-risk node to neighbors.

        When a node is confirmed malicious its risk
        propagates to connected nodes with a decay factor.

        This is how the graph changes the C2 IP score:
            xjf8k2mp.duckdns.org → DGA score: 0.90
            propagates to →
            185.220.101.45 → propagated risk: 0.63
            which is HIGH even though Layer 2 scored BENIGN

        decay_factor: how much risk decreases per hop
                     0.7 means each hop reduces by 30%
        """
        if node_id not in self.nodes:
            return

        source_node = self.nodes[node_id]
        propagated_risk = (
            source_node.risk_score * decay_factor
        )

        if propagated_risk < 0.1:
            return

        for neighbor_id in self.graph.successors(
            node_id
        ):
            if neighbor_id in self.nodes:
                self.nodes[neighbor_id].update_risk(
                    propagated_risk
                )
                logger.debug(
                    f"Risk propagated: "
                    f"{node_id} → {neighbor_id} "
                    f"({propagated_risk:.2f})"
                )

    def propagate_all_risks(
        self,
        threshold: float = 0.5
    ) -> int:
        """
        Propagate risk from all high-risk nodes.

        Called after adding new events to update
        the risk landscape across the entire graph.

        Returns count of nodes that had risk updated.
        """
        high_risk_nodes = self.get_high_risk_nodes(
            threshold
        )
        updated_count = 0

        for node in high_risk_nodes:
            before_count = len(
                self.get_neighbors(node.node_id)
            )
            self.propagate_risk(node.node_id)
            updated_count += before_count

        return updated_count

    # ============================================================
    # STATISTICS AND REPORTING
    # ============================================================

    def get_statistics(self) -> dict:
        """Return graph statistics for monitoring"""
        node_type_counts = {}
        for node in self.nodes.values():
            type_name = node.node_type.value
            node_type_counts[type_name] = (
                node_type_counts.get(type_name, 0) + 1
            )

        high_risk = self.get_high_risk_nodes(0.6)

        return {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "node_types": node_type_counts,
            "high_risk_count": len(high_risk),
            "events_processed": (
                self.total_events_processed
            ),
            "graph_density": (
                nx.density(self.graph)
                if self.total_nodes > 1 else 0
            )
        }

    def get_summary(self) -> str:
        """Return human-readable graph summary"""
        stats = self.get_statistics()
        high_risk = self.get_high_risk_nodes(0.6)

        summary = (
            f"Security Knowledge Graph Summary\n"
            f"  Nodes: {stats['total_nodes']}\n"
            f"  Edges: {stats['total_edges']}\n"
            f"  High Risk Entities: "
            f"{stats['high_risk_count']}\n"
        )

        if high_risk:
            summary += "\nHigh Risk Entities:\n"
            for node in sorted(
                high_risk,
                key=lambda x: x.risk_score,
                reverse=True
            )[:5]:
                summary += (
                    f"  {node.risk_label}: "
                    f"{node.label} "
                    f"({node.node_type.value}) "
                    f"score={node.risk_score:.2f}\n"
                )

        return summary

    # ============================================================
    # PRIVATE METHODS
    # ============================================================

    def _current_timestamp(self) -> str:
        """Return current UTC timestamp"""
        return datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )