"""
Layer 3 — Knowledge Graph
Graph Builder

This module takes Layer 2 RoutingResults and builds
the security knowledge graph by extracting entities
and relationships from each detection result.

This is the bridge between Layer 2 and Layer 3.

Every time an event flows through the pipeline:
    1. Layer 1 normalizes the raw event
    2. Layer 2 scores it with ML models
    3. Layer 3 GraphBuilder extracts entities
       and adds them to the knowledge graph

The GraphBuilder knows:
    - Which fields in ECSNormalized become nodes
    - Which relationships between fields become edges
    - How Layer 2 scores translate to node risk

Over time the graph accumulates:
    - All hosts seen in your environment
    - All users and their behaviors
    - All processes that have executed
    - All IPs and domains contacted
    - All threat detections and their context
    - All relationships between these entities

This accumulated graph is what Layer 4 LLM agents
query when investigating alerts.
"""

import logging
from datetime import timezone
from datetime import datetime
from typing import Optional

from layer3_knowledge.graph.security_graph import (
    SecurityKnowledgeGraph,
    SecurityNode,
    NodeType,
    EdgeType
)

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds security knowledge graph from pipeline events.

    Takes Layer 2 RoutingResults and ECSNormalized events
    and extracts entities and relationships into the graph.

    Usage:
        graph = SecurityKnowledgeGraph()
        builder = GraphBuilder(graph)

        # After each pipeline event
        builder.process_routing_result(
            ecs_event, routing_result
        )

        # Get graph summary
        print(graph.get_summary())
    """

    def __init__(
        self,
        graph: SecurityKnowledgeGraph
    ):
        self.graph = graph
        self.events_processed = 0

    def process_routing_result(
        self,
        ecs_event,
        routing_result
    ) -> dict:
        """
        Process a complete pipeline result into the graph.

        Extracts all entities and relationships from
        both the normalized ECS event and the Layer 2
        detection results.

        Args:
            ecs_event: ECSNormalized from Layer 1
            routing_result: RoutingResult from Layer 2

        Returns:
            Dictionary of nodes added/updated
        """
        if ecs_event is None:
            return {}

        added = {}

        # ---- EXTRACT BASE ENTITIES ----
        # These come from the normalized ECS event
        host_node = self._extract_host(ecs_event)
        user_node = self._extract_user(ecs_event)

        if host_node:
            added["host"] = host_node
        if user_node:
            added["user"] = user_node

        # Connect host to user
        if host_node and user_node:
            self.graph.add_edge(
                source_id=host_node.node_id,
                target_id=user_node.node_id,
                edge_type=EdgeType.BELONGS_TO
            )

        # ---- EXTRACT EVENT-SPECIFIC ENTITIES ----
        category = ecs_event.event.category

        if category == "process":
            process_nodes = self._extract_process_entities(
                ecs_event, routing_result, host_node
            )
            added.update(process_nodes)

        elif category == "network":
            network_nodes = self._extract_network_entities(
                ecs_event, routing_result, host_node
            )
            added.update(network_nodes)

        elif category == "dns":
            dns_nodes = self._extract_dns_entities(
                ecs_event, routing_result, host_node
            )
            added.update(dns_nodes)

        # ---- ADD ALERT NODES ----
        if routing_result:
            alert_nodes = self._extract_alert_nodes(
                routing_result
            )
            added.update(alert_nodes)

            # Connect alerts to host
            if host_node:
                for alert_key, alert_node in (
                    alert_nodes.items()
                ):
                    self.graph.add_edge(
                        source_id=host_node.node_id,
                        target_id=alert_node.node_id,
                        edge_type=EdgeType.TRIGGERED
                    )

        # ---- PROPAGATE RISK ----
        self.graph.propagate_all_risks(threshold=0.5)

        self.events_processed += 1
        self.graph.total_events_processed += 1

        logger.debug(
            f"Processed {category} event. "
            f"Graph: {self.graph.total_nodes} nodes, "
            f"{self.graph.total_edges} edges"
        )

        return added

    # ============================================================
    # ENTITY EXTRACTORS
    # ============================================================

    def _extract_host(
        self,
        ecs_event
    ) -> Optional[SecurityNode]:
        """Extract host entity from ECS event"""
        if not ecs_event.host:
            return None

        hostname = ecs_event.host.hostname or ""
        if not hostname:
            return None

        # Host risk starts at Layer 1 severity
        severity = ecs_event.event.severity or 0
        risk_score = severity / 100.0

        return self.graph.add_host(
            hostname=hostname,
            risk_score=risk_score,
            properties={
                "os_platform": (
                    ecs_event.host.os_platform or ""
                ),
                "ip": ecs_event.host.ip or ""
            }
        )

    def _extract_user(
        self,
        ecs_event
    ) -> Optional[SecurityNode]:
        """Extract user identity entity"""
        if not ecs_event.user:
            return None

        username = ecs_event.user.name or ""
        if not username:
            return None

        domain = ecs_event.user.domain or ""

        return self.graph.add_user(
            username=username,
            domain=domain
        )

    def _extract_process_entities(
        self,
        ecs_event,
        routing_result,
        host_node
    ) -> dict:
        """
        Extract process and parent process entities.

        Creates nodes for both the process and its
        parent then connects them with SPAWNED edge.

        This parent-child relationship is the most
        important security signal in the graph —
        MSBuild → PowerShell is visible as a graph
        pattern that triggers threat hunting rules.
        """
        added = {}

        if not ecs_event.process:
            return added

        # Get malware risk score from Layer 2
        malware_risk = 0.0
        if routing_result and routing_result.malware_result:
            malware_risk = (
                routing_result.malware_result.risk_score
            )

        # ---- PROCESS NODE ----
        process_name = ecs_event.process.name or ""
        if process_name and host_node:
            properties = {
                "command_line": (
                    ecs_event.process.command_line or ""
                )[:200],
                "integrity_level": (
                    ecs_event.process.integrity_level
                    or ""
                )
            }

            if routing_result and (
                routing_result.malware_result
            ):
                mal = routing_result.malware_result
                properties["malware_verdict"] = (
                    "MALICIOUS" if mal.is_malicious
                    else "BENIGN"
                )
                properties["attack_techniques"] = (
                    mal.attack_techniques
                )

            process_node = self.graph.add_process(
                process_name=process_name,
                host=host_node.label,
                risk_score=malware_risk,
                properties=properties
            )
            added["process"] = process_node

            # Connect host to process
            self.graph.add_edge(
                source_id=host_node.node_id,
                target_id=process_node.node_id,
                edge_type=EdgeType.EXECUTED
            )

            # ---- PARENT PROCESS NODE ----
            if ecs_event.process.parent:
                parent_name = (
                    ecs_event.process.parent.name or ""
                )
                if parent_name:
                    parent_node = self.graph.add_process(
                        process_name=parent_name,
                        host=host_node.label,
                        risk_score=0.0,
                        properties={
                            "role": "parent_process"
                        }
                    )
                    added["parent_process"] = parent_node

                    # Connect parent to child
                    # parent SPAWNED child
                    self.graph.add_edge(
                        source_id=parent_node.node_id,
                        target_id=process_node.node_id,
                        edge_type=EdgeType.SPAWNED,
                        properties={
                            "relationship": (
                                "parent_spawned_child"
                            )
                        }
                    )

        return added

    def _extract_network_entities(
        self,
        ecs_event,
        routing_result,
        host_node
    ) -> dict:
        """
        Extract IP address entities from network events.

        Creates nodes for source and destination IPs
        and connects the process that made the connection.
        """
        added = {}

        # Get intrusion risk score
        intrusion_risk = 0.0
        if routing_result and (
            routing_result.intrusion_result
        ):
            intrusion_risk = (
                routing_result.intrusion_result.risk_score
            )

        # ---- DESTINATION IP NODE ----
        if ecs_event.destination:
            dest_ip = ecs_event.destination.ip or ""
            if dest_ip:
                properties = {
                    "port": ecs_event.destination.port,
                }
                if routing_result and (
                    routing_result.intrusion_result
                ):
                    intr = routing_result.intrusion_result
                    properties["intrusion_verdict"] = (
                        "ATTACK" if intr.is_attack
                        else "BENIGN"
                    )

                ip_node = self.graph.add_ip(
                    ip_address=dest_ip,
                    risk_score=intrusion_risk,
                    properties=properties
                )
                added["destination_ip"] = ip_node

                # Connect host to destination IP
                if host_node:
                    self.graph.add_edge(
                        source_id=host_node.node_id,
                        target_id=ip_node.node_id,
                        edge_type=EdgeType.CONNECTED_TO,
                        properties={
                            "port": (
                                ecs_event.destination.port
                            )
                        }
                    )

                # Connect process to destination IP
                if ecs_event.process:
                    process_name = (
                        ecs_event.process.name or ""
                    )
                    if process_name and host_node:
                        proc_id = (
                            f"process:{process_name}"
                            f":{host_node.label}"
                        )
                        if proc_id in self.graph.nodes:
                            self.graph.add_edge(
                                source_id=proc_id,
                                target_id=ip_node.node_id,
                                edge_type=EdgeType.CONNECTED_TO
                            )

        return added

    def _extract_dns_entities(
        self,
        ecs_event,
        routing_result,
        host_node
    ) -> dict:
        """
        Extract domain entities from DNS events.

        Creates domain node and connects it to
        the process that queried it and any
        IP address it resolves to.

        This creates the critical graph path:
        svchost.exe → RESOLVED → xjf8k2mp.duckdns.org
        xjf8k2mp.duckdns.org → RESOLVES_TO → 185.x.x.x
        """
        added = {}

        # Get DGA risk score
        dga_risk = 0.0
        if routing_result and routing_result.dns_result:
            dga_risk = routing_result.dns_result.risk_score

        # ---- DOMAIN NODE ----
        if ecs_event.destination:
            domain = ecs_event.destination.domain or ""
            if domain:
                properties = {}
                if routing_result and (
                    routing_result.dns_result
                ):
                    dns = routing_result.dns_result
                    properties["dga_verdict"] = (
                        "DGA" if dns.is_dga
                        else "LEGITIMATE"
                    )
                    properties["dga_family"] = (
                        dns.dga_family
                    )
                    properties["dga_indicators"] = (
                        dns.dga_indicators
                    )

                domain_node = self.graph.add_domain(
                    domain=domain,
                    risk_score=dga_risk,
                    properties=properties
                )
                added["domain"] = domain_node

                # Connect host to domain
                if host_node:
                    self.graph.add_edge(
                        source_id=host_node.node_id,
                        target_id=domain_node.node_id,
                        edge_type=EdgeType.RESOLVED
                    )

                # Connect process to domain
                if ecs_event.process:
                    process_name = (
                        ecs_event.process.name or ""
                    )
                    if process_name and host_node:
                        proc_id = (
                            f"process:{process_name}"
                            f":{host_node.label}"
                        )
                        # Add process if not exists
                        if proc_id not in self.graph.nodes:
                            self.graph.add_process(
                                process_name=process_name,
                                host=host_node.label
                            )
                            self.graph.add_edge(
                                source_id=host_node.node_id,
                                target_id=proc_id,
                                edge_type=EdgeType.EXECUTED
                            )

                        self.graph.add_edge(
                            source_id=proc_id,
                            target_id=domain_node.node_id,
                            edge_type=EdgeType.RESOLVED
                        )

        return added

    def _extract_alert_nodes(
        self,
        routing_result
    ) -> dict:
        """
        Create alert nodes from Layer 2 detections.

        Each positive detection becomes an alert
        node in the graph connected to the entities
        that triggered it.
        """
        added = {}

        if routing_result.malware_result:
            mal = routing_result.malware_result
            if mal.is_malicious:
                alert_node = self.graph.add_alert(
                    alert_id=f"malware:{mal.scored_at}",
                    alert_type="MALWARE_DETECTED",
                    risk_score=mal.risk_score,
                    properties={
                        "process": mal.process_name,
                        "techniques": mal.attack_techniques,
                        "indicators": mal.malware_indicators
                    }
                )
                added["malware_alert"] = alert_node

        if routing_result.dns_result:
            dns = routing_result.dns_result
            if dns.is_dga:
                alert_node = self.graph.add_alert(
                    alert_id=f"dga:{dns.scored_at}",
                    alert_type="DGA_DOMAIN_DETECTED",
                    risk_score=dns.risk_score,
                    properties={
                        "domain": dns.domain,
                        "dga_family": dns.dga_family,
                        "indicators": dns.dga_indicators
                    }
                )
                added["dga_alert"] = alert_node

        if routing_result.intrusion_result:
            intr = routing_result.intrusion_result
            if intr.is_attack:
                alert_node = self.graph.add_alert(
                    alert_id=f"intrusion:{intr.scored_at}",
                    alert_type="INTRUSION_DETECTED",
                    risk_score=intr.risk_score,
                    properties={
                        "explanation": intr.explanation
                    }
                )
                added["intrusion_alert"] = alert_node

        return added

    def get_statistics(self) -> dict:
        """Return builder statistics"""
        return {
            "events_processed": self.events_processed,
            "graph_stats": self.graph.get_statistics()
        }