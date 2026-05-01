"""
Layer 3 — Knowledge Graph Tests

Tests verify four things:

1. GRAPH STRUCTURE
   Can nodes and edges be added correctly?
   Do duplicate nodes get merged not duplicated?
   Do edges connect correct node pairs?

2. RISK PROPAGATION
   Does risk propagate from malicious nodes?
   Does the decay factor work correctly?
   Does the IP enrichment change its risk score?

3. GRAPH BUILDER
   Does it correctly extract entities from ECS events?
   Does it create correct parent-child relationships?
   Does it connect processes to IPs and domains?

4. THREAT ENRICHER
   Does it correctly identify known malicious IPs?
   Does it correctly identify known domains?
   Does it identify campaign patterns?
"""

import pytest
from unittest.mock import MagicMock

from layer3_knowledge.graph.security_graph import (
    SecurityKnowledgeGraph,
    SecurityNode,
    NodeType,
    EdgeType
)
from layer3_knowledge.graph.graph_builder import (
    GraphBuilder
)
from layer3_knowledge.enrichment.threat_enricher import (
    ThreatEnricher,
    KNOWN_MALICIOUS_IPS,
    KNOWN_MALICIOUS_DOMAINS
)


# ============================================================
# MOCK EVENT BUILDERS
# ============================================================

def make_mock_process_ecs(
    hostname="WKSTN-JSMITH-01",
    username="jsmith",
    domain="CORP",
    process_name="powershell.exe",
    parent_name="MSBuild.exe",
    command_line="powershell.exe -enc JABj",
    severity=75
):
    """Build mock ECS process event"""
    mock_parent = MagicMock()
    mock_parent.name = parent_name

    mock_process = MagicMock()
    mock_process.name = process_name
    mock_process.command_line = command_line
    mock_process.integrity_level = "Medium"
    mock_process.parent = mock_parent

    mock_host = MagicMock()
    mock_host.hostname = hostname
    mock_host.ip = "10.0.0.155"
    mock_host.os_platform = "windows"

    mock_user = MagicMock()
    mock_user.name = username
    mock_user.domain = domain

    mock_event = MagicMock()
    mock_event.category = "process"
    mock_event.severity = severity
    mock_event.id = "test-event-001"
    mock_event.dataset = "crowdstrike.process"

    mock_ecs = MagicMock()
    mock_ecs.host = mock_host
    mock_ecs.user = mock_user
    mock_ecs.process = mock_process
    mock_ecs.event = mock_event
    mock_ecs.timestamp = "2024-03-29T02:17:43Z"
    mock_ecs.source = None
    mock_ecs.destination = None

    return mock_ecs


def make_mock_network_ecs(
    hostname="WKSTN-JSMITH-01",
    dest_ip="185.220.101.45",
    dest_port=443
):
    """Build mock ECS network event"""
    mock_dest = MagicMock()
    mock_dest.ip = dest_ip
    mock_dest.port = dest_port
    mock_dest.domain = None

    mock_host = MagicMock()
    mock_host.hostname = hostname
    mock_host.ip = "10.0.0.155"
    mock_host.os_platform = "windows"

    mock_user = MagicMock()
    mock_user.name = "jsmith"
    mock_user.domain = "CORP"

    mock_process = MagicMock()
    mock_process.name = "svchost.exe"

    mock_event = MagicMock()
    mock_event.category = "network"
    mock_event.severity = 0
    mock_event.id = "test-event-002"
    mock_event.dataset = "crowdstrike.network"

    mock_ecs = MagicMock()
    mock_ecs.host = mock_host
    mock_ecs.user = mock_user
    mock_ecs.process = mock_process
    mock_ecs.event = mock_event
    mock_ecs.timestamp = "2024-03-29T02:18:43Z"
    mock_ecs.source = MagicMock()
    mock_ecs.source.ip = "10.0.0.155"
    mock_ecs.destination = mock_dest

    return mock_ecs


def make_mock_dns_ecs(
    hostname="WKSTN-JSMITH-01",
    domain="xjf8k2mp.duckdns.org",
    process_name="svchost.exe"
):
    """Build mock ECS DNS event"""
    mock_dest = MagicMock()
    mock_dest.domain = domain
    mock_dest.ip = None

    mock_host = MagicMock()
    mock_host.hostname = hostname
    mock_host.ip = "10.0.0.155"
    mock_host.os_platform = "windows"

    mock_user = MagicMock()
    mock_user.name = "jsmith"
    mock_user.domain = "CORP"

    mock_process = MagicMock()
    mock_process.name = process_name

    mock_event = MagicMock()
    mock_event.category = "dns"
    mock_event.severity = 0
    mock_event.id = "test-event-003"
    mock_event.dataset = "crowdstrike.dns"

    mock_ecs = MagicMock()
    mock_ecs.host = mock_host
    mock_ecs.user = mock_user
    mock_ecs.process = mock_process
    mock_ecs.event = mock_event
    mock_ecs.timestamp = "2024-03-29T02:19:03Z"
    mock_ecs.source = None
    mock_ecs.destination = mock_dest

    return mock_ecs


def make_mock_routing_result(
    is_malicious=False,
    malware_risk=0.0,
    is_dga=False,
    dga_risk=0.0
):
    """Build mock Layer 2 routing result"""
    result = MagicMock()
    result.overall_verdict = "BENIGN"
    result.overall_risk_score = 0.0

    if is_malicious:
        mal = MagicMock()
        mal.is_malicious = True
        mal.risk_score = malware_risk
        mal.process_name = "powershell.exe"
        mal.attack_techniques = [
            "T1059.001", "T1566.001"
        ]
        mal.malware_indicators = [
            "Encoded PowerShell detected"
        ]
        mal.scored_at = "2024-03-29T02:17:43Z"
        result.malware_result = mal
        result.overall_verdict = "MALWARE"
        result.overall_risk_score = malware_risk
    else:
        result.malware_result = None

    if is_dga:
        dns = MagicMock()
        dns.is_dga = True
        dns.risk_score = dga_risk
        dns.domain = "xjf8k2mp.duckdns.org"
        dns.dga_family = "dynamic_dns_abuse"
        dns.dga_indicators = [
            "Dynamic DNS provider detected"
        ]
        dns.scored_at = "2024-03-29T02:19:03Z"
        result.dns_result = dns
        result.overall_verdict = "DGA_DOMAIN"
        result.overall_risk_score = dga_risk
    else:
        result.dns_result = None

    result.intrusion_result = None

    return result


# ============================================================
# TEST CLASS — GRAPH STRUCTURE
# ============================================================

class TestGraphStructure:
    """Tests for SecurityKnowledgeGraph core operations"""

    def setup_method(self):
        self.graph = SecurityKnowledgeGraph()

    def test_add_host_node(self):
        """Host node correctly added to graph"""
        node = self.graph.add_host("WKSTN-TEST-01")
        assert node is not None
        assert node.node_type == NodeType.HOST
        assert node.label == "WKSTN-TEST-01"

    def test_add_ip_node(self):
        """IP address node correctly added"""
        node = self.graph.add_ip("185.220.101.45")
        assert node is not None
        assert node.node_type == NodeType.IP_ADDRESS
        assert node.label == "185.220.101.45"

    def test_add_domain_node(self):
        """Domain node correctly added"""
        node = self.graph.add_domain(
            "xjf8k2mp.duckdns.org"
        )
        assert node is not None
        assert node.node_type == NodeType.DOMAIN

    def test_duplicate_node_merged(self):
        """
        Adding same node twice merges not duplicates.
        Graph should have one node not two.
        """
        self.graph.add_host("WKSTN-TEST-01")
        self.graph.add_host("WKSTN-TEST-01")
        assert self.graph.total_nodes == 1

    def test_duplicate_node_updates_risk(self):
        """
        Adding same node with higher risk updates score.
        Risk can only increase never decrease.
        """
        self.graph.add_ip(
            "185.220.101.45", risk_score=0.3
        )
        self.graph.add_ip(
            "185.220.101.45", risk_score=0.9
        )

        node = self.graph.get_node("ip:185.220.101.45")
        assert node.risk_score == 0.9

    def test_risk_never_decreases(self):
        """
        Adding same node with lower risk keeps high risk.
        Evidence of malice is permanent.
        """
        self.graph.add_ip(
            "185.220.101.45", risk_score=0.9
        )
        self.graph.add_ip(
            "185.220.101.45", risk_score=0.1
        )

        node = self.graph.get_node("ip:185.220.101.45")
        assert node.risk_score == 0.9

    def test_add_edge_between_nodes(self):
        """Edge correctly connects two nodes"""
        self.graph.add_host("WKSTN-TEST-01")
        self.graph.add_ip("185.220.101.45")

        edge = self.graph.add_edge(
            source_id="host:WKSTN-TEST-01",
            target_id="ip:185.220.101.45",
            edge_type=EdgeType.CONNECTED_TO
        )

        assert edge is not None
        assert self.graph.total_edges == 1

    def test_edge_requires_existing_nodes(self):
        """
        Edge cannot connect nonexistent nodes.
        Returns None gracefully.
        """
        edge = self.graph.add_edge(
            source_id="host:NONEXISTENT",
            target_id="ip:185.220.101.45",
            edge_type=EdgeType.CONNECTED_TO
        )
        assert edge is None

    def test_get_neighbors(self):
        """Get neighbors returns connected nodes"""
        self.graph.add_host("WKSTN-TEST-01")
        self.graph.add_ip("185.220.101.45")
        self.graph.add_edge(
            source_id="host:WKSTN-TEST-01",
            target_id="ip:185.220.101.45",
            edge_type=EdgeType.CONNECTED_TO
        )

        neighbors = self.graph.get_neighbors(
            "host:WKSTN-TEST-01",
            direction="out"
        )

        assert len(neighbors) == 1
        assert neighbors[0].label == "185.220.101.45"

    def test_get_high_risk_nodes(self):
        """High risk nodes correctly filtered"""
        self.graph.add_ip(
            "185.220.101.45", risk_score=0.9
        )
        self.graph.add_ip(
            "10.0.0.1", risk_score=0.1
        )

        high_risk = self.graph.get_high_risk_nodes(0.6)

        assert len(high_risk) == 1
        assert high_risk[0].label == "185.220.101.45"

    def test_get_nodes_by_type(self):
        """Nodes correctly filtered by type"""
        self.graph.add_host("WKSTN-01")
        self.graph.add_host("WKSTN-02")
        self.graph.add_ip("10.0.0.1")

        hosts = self.graph.get_nodes_by_type(
            NodeType.HOST
        )
        ips = self.graph.get_nodes_by_type(
            NodeType.IP_ADDRESS
        )

        assert len(hosts) == 2
        assert len(ips) == 1


# ============================================================
# TEST CLASS — RISK PROPAGATION
# ============================================================

class TestRiskPropagation:
    """Tests for graph-based risk propagation"""

    def setup_method(self):
        self.graph = SecurityKnowledgeGraph()

    def test_risk_propagates_to_neighbors(self):
        """
        High risk node propagates risk to neighbors.
        DGA domain should increase connected IP risk.
        """
        self.graph.add_domain(
            "xjf8k2mp.duckdns.org",
            risk_score=0.9
        )
        self.graph.add_ip(
            "185.220.101.45",
            risk_score=0.1
        )
        self.graph.add_edge(
            source_id="domain:xjf8k2mp.duckdns.org",
            target_id="ip:185.220.101.45",
            edge_type=EdgeType.RESOLVES_TO
        )

        # Propagate risk
        self.graph.propagate_risk(
            "domain:xjf8k2mp.duckdns.org"
        )

        ip_node = self.graph.get_node(
            "ip:185.220.101.45"
        )
        # Risk should have increased from 0.1
        assert ip_node.risk_score > 0.1

    def test_risk_decay_factor_applied(self):
        """
        Risk decreases with decay factor per hop.
        0.9 × 0.7 = 0.63 for one hop.
        """
        self.graph.add_domain(
            "evil.duckdns.org",
            risk_score=0.9
        )
        self.graph.add_ip(
            "1.2.3.4",
            risk_score=0.0
        )
        self.graph.add_edge(
            source_id="domain:evil.duckdns.org",
            target_id="ip:1.2.3.4",
            edge_type=EdgeType.RESOLVES_TO
        )

        self.graph.propagate_risk(
            "domain:evil.duckdns.org",
            decay_factor=0.7
        )

        ip_node = self.graph.get_node("ip:1.2.3.4")
        expected = 0.9 * 0.7
        assert abs(ip_node.risk_score - expected) < 0.01

    def test_attack_path_found(self):
        """
        Shortest path found between connected nodes.
        Used to trace attack kill chains.
        """
        self.graph.add_host("WKSTN-01")
        self.graph.add_process("powershell.exe", "WKSTN-01")
        self.graph.add_ip("185.220.101.45")

        self.graph.add_edge(
            "host:WKSTN-01",
            "process:powershell.exe:WKSTN-01",
            EdgeType.EXECUTED
        )
        self.graph.add_edge(
            "process:powershell.exe:WKSTN-01",
            "ip:185.220.101.45",
            EdgeType.CONNECTED_TO
        )

        path = self.graph.get_attack_path(
            "host:WKSTN-01",
            "ip:185.220.101.45"
        )

        assert len(path) == 3
        assert path[0] == "host:WKSTN-01"
        assert path[-1] == "ip:185.220.101.45"


# ============================================================
# TEST CLASS — GRAPH BUILDER
# ============================================================

class TestGraphBuilder:
    """Tests for GraphBuilder entity extraction"""

    def setup_method(self):
        self.graph = SecurityKnowledgeGraph()
        self.builder = GraphBuilder(self.graph)

    def test_process_event_adds_host(self):
        """Process event adds host to graph"""
        ecs_event = make_mock_process_ecs()
        routing_result = make_mock_routing_result(
            is_malicious=True,
            malware_risk=0.97
        )

        self.builder.process_routing_result(
            ecs_event, routing_result
        )

        host = self.graph.get_node(
            "host:WKSTN-JSMITH-01"
        )
        assert host is not None
        assert host.node_type == NodeType.HOST

    def test_process_event_adds_user(self):
        """Process event adds user to graph"""
        ecs_event = make_mock_process_ecs()
        routing_result = make_mock_routing_result()

        self.builder.process_routing_result(
            ecs_event, routing_result
        )

        user = self.graph.get_node("user:CORP\\jsmith")
        assert user is not None
        assert user.node_type == NodeType.USER

    def test_process_event_adds_process_node(self):
        """Process event adds process entity"""
        ecs_event = make_mock_process_ecs()
        routing_result = make_mock_routing_result(
            is_malicious=True,
            malware_risk=0.97
        )

        self.builder.process_routing_result(
            ecs_event, routing_result
        )

        process = self.graph.get_node(
            "process:powershell.exe:WKSTN-JSMITH-01"
        )
        assert process is not None
        assert process.node_type == NodeType.PROCESS

    def test_process_event_adds_parent_process(self):
        """
        Process event adds parent process node.
        MSBuild spawning PowerShell is visible
        as graph relationship.
        """
        ecs_event = make_mock_process_ecs()
        routing_result = make_mock_routing_result(
            is_malicious=True,
            malware_risk=0.97
        )

        self.builder.process_routing_result(
            ecs_event, routing_result
        )

        parent = self.graph.get_node(
            "process:MSBuild.exe:WKSTN-JSMITH-01"
        )
        assert parent is not None

    def test_malware_risk_applied_to_process(self):
        """
        Layer 2 malware risk score applied to
        process node in graph.
        """
        ecs_event = make_mock_process_ecs()
        routing_result = make_mock_routing_result(
            is_malicious=True,
            malware_risk=0.97
        )

        self.builder.process_routing_result(
            ecs_event, routing_result
        )

        process = self.graph.get_node(
            "process:powershell.exe:WKSTN-JSMITH-01"
        )
        assert process.risk_score == 0.97

    def test_network_event_adds_ip(self):
        """Network event adds destination IP"""
        ecs_event = make_mock_network_ecs()
        routing_result = make_mock_routing_result()

        self.builder.process_routing_result(
            ecs_event, routing_result
        )

        ip = self.graph.get_node("ip:185.220.101.45")
        assert ip is not None
        assert ip.node_type == NodeType.IP_ADDRESS

    def test_dns_event_adds_domain(self):
        """DNS event adds domain entity"""
        ecs_event = make_mock_dns_ecs()
        routing_result = make_mock_routing_result(
            is_dga=True,
            dga_risk=0.9
        )

        self.builder.process_routing_result(
            ecs_event, routing_result
        )

        domain = self.graph.get_node(
            "domain:xjf8k2mp.duckdns.org"
        )
        assert domain is not None
        assert domain.node_type == NodeType.DOMAIN

    def test_dga_risk_applied_to_domain(self):
        """
        Layer 2 DGA risk score applied to
        domain node in graph.
        """
        ecs_event = make_mock_dns_ecs()
        routing_result = make_mock_routing_result(
            is_dga=True,
            dga_risk=0.9
        )

        self.builder.process_routing_result(
            ecs_event, routing_result
        )

        domain = self.graph.get_node(
            "domain:xjf8k2mp.duckdns.org"
        )
        assert domain.risk_score == 0.9

    def test_malware_alert_node_created(self):
        """Alert node created for malware detection"""
        ecs_event = make_mock_process_ecs()
        routing_result = make_mock_routing_result(
            is_malicious=True,
            malware_risk=0.97
        )

        self.builder.process_routing_result(
            ecs_event, routing_result
        )

        alert_nodes = self.graph.get_nodes_by_type(
            NodeType.ALERT
        )
        assert len(alert_nodes) > 0

    def test_events_processed_counter(self):
        """Events processed counter increments"""
        ecs_event = make_mock_process_ecs()
        routing_result = make_mock_routing_result()

        self.builder.process_routing_result(
            ecs_event, routing_result
        )
        self.builder.process_routing_result(
            ecs_event, routing_result
        )

        assert self.builder.events_processed == 2


# ============================================================
# TEST CLASS — THREAT ENRICHER
# ============================================================

class TestThreatEnricher:
    """Tests for ThreatEnricher intelligence enrichment"""

    def setup_method(self):
        self.graph = SecurityKnowledgeGraph()
        self.enricher = ThreatEnricher(self.graph)

    def test_known_malicious_ip_enriched(self):
        """
        Known malicious IP gets enriched with
        threat intelligence context.
        185.220.101.45 is a known Tor exit node.
        """
        self.graph.add_ip(
            "185.220.101.45",
            risk_score=0.1
        )
        result = self.enricher.enrich_ip(
            "185.220.101.45"
        )

        assert result is True

        node = self.graph.get_node("ip:185.220.101.45")
        assert node.risk_score > 0.5
        assert "threat_type" in node.properties

    def test_unknown_ip_not_enriched(self):
        """Unknown IP returns False gracefully"""
        self.graph.add_ip("10.0.0.1", risk_score=0.0)
        result = self.enricher.enrich_ip("10.0.0.1")
        assert result is False

    def test_known_domain_pattern_enriched(self):
        """
        Domain matching known malicious pattern enriched.
        duckdns.org is a known DGA provider.
        """
        self.graph.add_domain(
            "xjf8k2mp.duckdns.org",
            risk_score=0.3
        )
        result = self.enricher.enrich_domain(
            "xjf8k2mp.duckdns.org"
        )

        assert result is True

        node = self.graph.get_node(
            "domain:xjf8k2mp.duckdns.org"
        )
        assert node.risk_score > 0.3
        assert "threat_type" in node.properties

    def test_legitimate_domain_not_enriched(self):
        """google.com not enriched as malicious"""
        self.graph.add_domain(
            "google.com",
            risk_score=0.0
        )
        result = self.enricher.enrich_domain("google.com")
        assert result is False

    def test_enrich_all_processes_all_entities(self):
        """enrich_all enriches all IP and domain nodes"""
        self.graph.add_ip("185.220.101.45")
        self.graph.add_ip("10.0.0.1")
        self.graph.add_domain("xjf8k2mp.duckdns.org")
        self.graph.add_domain("google.com")

        results = self.enricher.enrich_all()

        assert results["ips_enriched"] >= 1
        assert results["domains_enriched"] >= 1

    def test_host_enrichment_from_connections(self):
        """
        Host risk increases when connected to
        malicious entities.
        Core value of Layer 3 graph enrichment.
        """
        # Add host with low initial risk
        self.graph.add_host(
            "WKSTN-JSMITH-01",
            risk_score=0.1
        )

        # Add malicious IP with high risk
        self.graph.add_ip(
            "185.220.101.45",
            risk_score=0.9
        )

        # Connect host to malicious IP
        self.graph.add_edge(
            "host:WKSTN-JSMITH-01",
            "ip:185.220.101.45",
            EdgeType.CONNECTED_TO
        )

        # Enrich host based on connections
        enrichment = self.enricher.enrich_host_from_graph(
            "WKSTN-JSMITH-01"
        )

        assert enrichment["host_risk_score"] > 0.1
        assert len(
            enrichment["malicious_connections"]
        ) > 0

    def test_threat_summary_generated(self):
        """
        Comprehensive threat summary generated
        from graph state.
        """
        self.graph.add_ip(
            "185.220.101.45",
            risk_score=0.9
        )
        self.graph.add_domain(
            "xjf8k2mp.duckdns.org",
            risk_score=0.9
        )

        summary = self.enricher.get_threat_summary()

        assert "total_entities" in summary
        assert "high_risk_entities" in summary
        assert summary["high_risk_entities"] >= 2

    def test_nonexistent_ip_returns_false(self):
        """Enriching nonexistent IP returns False"""
        result = self.enricher.enrich_ip("999.999.999.999")
        assert result is False

    def test_nonexistent_domain_returns_false(self):
        """Enriching nonexistent domain returns False"""
        result = self.enricher.enrich_domain(
            "notingraph.com"
        )
        assert result is False