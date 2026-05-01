"""
Layer 3 — Knowledge Graph
Threat Intelligence Enricher

This module enriches knowledge graph entities with
external threat intelligence data.

What Enrichment Does:
    After GraphBuilder adds raw entities to the graph
    the ThreatEnricher adds intelligence context:

    IP 185.220.101.45 (raw from CrowdStrike)
        ↓ enriched with ↓
    Known Tor exit node
    Associated with APT29 infrastructure
    Seen in 47 other organizations
    Part of campaign: "SolarWinds-like 2024"

    Domain xjf8k2mp.duckdns.org (raw from DNS)
        ↓ enriched with ↓
    DGA family: Emotet variant
    First seen: 2024-01-15
    Associated C2 IPs: [185.x.x.x, 192.x.x.x]
    Threat actor: TA542 (Mummy Spider)

Enrichment Sources:
    1. Built-in intelligence (this file)
       Known malicious IPs, domains, threat actors
       Updated manually and via threat feeds

    2. VirusTotal API (production upgrade)
       Real-time IP and domain reputation
       File hash analysis
       Malware family identification

    3. MISP / OpenCTI (enterprise upgrade)
       Structured threat intelligence
       STIX format threat actor profiles
       Campaign tracking

    4. Your own SOC data
       Historical incidents from your environment
       Custom threat actor profiles
       Industry-specific indicators

ATT&CK Coverage:
    T1588  Obtain Capabilities
    T1583  Acquire Infrastructure
    T1584  Compromise Infrastructure
"""

import logging
from typing import Optional
from layer3_knowledge.graph.security_graph import (
    SecurityKnowledgeGraph,
    SecurityNode,
    NodeType,
    EdgeType
)

logger = logging.getLogger(__name__)


# ============================================================
# BUILT-IN THREAT INTELLIGENCE
# In production these come from threat feed APIs
# ============================================================

# Known malicious IP ranges and addresses
# Sources: Abuse.ch, Feodo Tracker, Emerging Threats
KNOWN_MALICIOUS_IPS = {
    "185.220.101.45": {
        "type": "tor_exit_node",
        "actor": "multiple",
        "tags": ["tor", "anonymization", "c2_relay"],
        "risk_score": 0.85,
        "description": (
            "Known Tor exit node frequently used "
            "for C2 communication"
        )
    },
    "185.220.101.46": {
        "type": "tor_exit_node",
        "actor": "multiple",
        "tags": ["tor", "anonymization"],
        "risk_score": 0.85,
        "description": "Known Tor exit node"
    },
    "192.168.100.0": {
        "type": "test",
        "actor": "test",
        "tags": ["test"],
        "risk_score": 0.1,
        "description": "Test IP"
    }
}

# Known malicious domain patterns
KNOWN_MALICIOUS_DOMAINS = {
    "duckdns.org": {
        "type": "dynamic_dns_provider",
        "tags": ["dyndns", "c2_infrastructure"],
        "risk_score": 0.4,
        "description": (
            "Dynamic DNS provider frequently abused "
            "for C2 infrastructure"
        )
    },
    "no-ip.com": {
        "type": "dynamic_dns_provider",
        "tags": ["dyndns", "c2_infrastructure"],
        "risk_score": 0.4,
        "description": "Dynamic DNS provider"
    }
}

# Known threat actor profiles
THREAT_ACTORS = {
    "TA542": {
        "name": "Mummy Spider",
        "aliases": ["Emotet", "Geodo"],
        "motivation": "financial",
        "ttps": [
            "T1566.001",  # Spearphishing
            "T1059.001",  # PowerShell
            "T1547.001",  # Registry Run Keys
            "T1071.001"   # Web Protocols C2
        ],
        "description": (
            "Financially motivated threat actor "
            "operating Emotet botnet"
        )
    },
    "APT29": {
        "name": "Cozy Bear",
        "aliases": ["The Dukes", "Grizzly Steppe"],
        "motivation": "espionage",
        "ttps": [
            "T1566.002",  # Spearphishing Link
            "T1059.001",  # PowerShell
            "T1078",      # Valid Accounts
            "T1021.001"   # Remote Desktop
        ],
        "description": (
            "Russian SVR-linked espionage group "
            "targeting government and think tanks"
        )
    },
    "FIN7": {
        "name": "Carbanak Group",
        "aliases": ["Navigator Group"],
        "motivation": "financial",
        "ttps": [
            "T1566.001",  # Spearphishing
            "T1059.005",  # VBScript
            "T1218.011",  # Rundll32
            "T1041"       # Exfil over C2
        ],
        "description": (
            "Financially motivated group targeting "
            "retail, hospitality, finance sectors"
        )
    }
}

# Known C2 ports
SUSPICIOUS_PORTS = {
    4444: {
        "service": "Metasploit default",
        "risk": 0.9
    },
    1337: {
        "service": "Common backdoor port",
        "risk": 0.8
    },
    31337: {
        "service": "Elite backdoor port",
        "risk": 0.9
    },
    9001: {
        "service": "Tor OR port",
        "risk": 0.7
    },
    9030: {
        "service": "Tor directory port",
        "risk": 0.7
    },
    8080: {
        "service": "Common C2 proxy port",
        "risk": 0.5
    }
}

# Known malware families and their indicators
MALWARE_FAMILIES = {
    "emotet": {
        "indicators": [
            "msbuild spawning powershell",
            "encoded powershell",
            "duckdns domains"
        ],
        "ttps": ["T1566.001", "T1059.001"],
        "actor": "TA542"
    },
    "cobalt_strike": {
        "indicators": [
            "powershell iex download",
            "beacon traffic pattern",
            "amsi bypass"
        ],
        "ttps": ["T1059.001", "T1071.001"],
        "actor": "multiple"
    },
    "trickbot": {
        "indicators": [
            "svchost network connections",
            "dynamic dns abuse",
            "lateral movement"
        ],
        "ttps": ["T1021.002", "T1078"],
        "actor": "TA505"
    }
}


class ThreatEnricher:
    """
    Enriches knowledge graph entities with
    threat intelligence context.

    Takes existing graph nodes and adds:
    - Threat actor attribution
    - Campaign association
    - Malware family identification
    - Risk score updates based on TI

    Usage:
        enricher = ThreatEnricher(graph)
        enricher.enrich_all()
        enricher.enrich_ip("185.220.101.45")
        enricher.enrich_domain("xjf8k2mp.duckdns.org")
    """

    def __init__(
        self,
        graph: SecurityKnowledgeGraph
    ):
        self.graph = graph
        self.enrichments_applied = 0

    def enrich_all(self) -> dict:
        """
        Enrich all entities in the graph.

        Called after processing a batch of events
        to apply all available threat intelligence.

        Returns dictionary of enrichment counts.
        """
        results = {
            "ips_enriched": 0,
            "domains_enriched": 0,
            "actors_added": 0,
            "campaigns_identified": 0
        }

        # Enrich all IPs
        ip_nodes = self.graph.get_nodes_by_type(
            NodeType.IP_ADDRESS
        )
        for node in ip_nodes:
            if self.enrich_ip(node.label):
                results["ips_enriched"] += 1

        # Enrich all domains
        domain_nodes = self.graph.get_nodes_by_type(
            NodeType.DOMAIN
        )
        for node in domain_nodes:
            if self.enrich_domain(node.label):
                results["domains_enriched"] += 1

        # Identify campaigns from alert patterns
        campaigns = self.identify_campaigns()
        results["campaigns_identified"] = len(campaigns)

        logger.info(
            f"Enrichment complete: {results}"
        )

        return results

    def enrich_ip(
        self,
        ip_address: str
    ) -> bool:
        """
        Enrich an IP address node with threat intel.

        Checks built-in intelligence and updates
        risk score if malicious intelligence found.

        Args:
            ip_address: IP address string

        Returns:
            True if enrichment was applied
        """
        node_id = f"ip:{ip_address}"
        node = self.graph.get_node(node_id)

        if node is None:
            return False

        enriched = False

        # Check exact IP match
        if ip_address in KNOWN_MALICIOUS_IPS:
            intel = KNOWN_MALICIOUS_IPS[ip_address]
            node.update_risk(intel["risk_score"])
            node.properties.update({
                "threat_type": intel["type"],
                "threat_actor": intel["actor"],
                "threat_tags": intel["tags"],
                "threat_description": (
                    intel["description"]
                ),
                "enriched": True
            })
            enriched = True
            self.enrichments_applied += 1

            logger.info(
                f"IP enriched: {ip_address} → "
                f"{intel['type']} "
                f"(risk: {intel['risk_score']})"
            )

        # Check suspicious ports
        port = node.properties.get("port")
        if port and port in SUSPICIOUS_PORTS:
            port_intel = SUSPICIOUS_PORTS[port]
            node.update_risk(port_intel["risk"])
            node.properties["suspicious_port"] = (
                port_intel["service"]
            )
            enriched = True

        return enriched

    def enrich_domain(
        self,
        domain: str
    ) -> bool:
        """
        Enrich a domain node with threat intel.

        Checks if domain matches known malicious
        patterns — exact match or suffix match.

        Args:
            domain: Domain name string

        Returns:
            True if enrichment was applied
        """
        node_id = f"domain:{domain}"
        node = self.graph.get_node(node_id)

        if node is None:
            return False

        enriched = False
        domain_lower = domain.lower()

        # Check suffix match against known patterns
        for pattern, intel in (
            KNOWN_MALICIOUS_DOMAINS.items()
        ):
            if domain_lower.endswith(pattern):
                node.update_risk(intel["risk_score"])
                node.properties.update({
                    "threat_type": intel["type"],
                    "threat_tags": intel["tags"],
                    "threat_description": (
                        intel["description"]
                    ),
                    "matched_pattern": pattern,
                    "enriched": True
                })
                enriched = True
                self.enrichments_applied += 1

                logger.info(
                    f"Domain enriched: {domain} → "
                    f"{intel['type']} "
                    f"(matched: {pattern})"
                )
                break

        return enriched

    def identify_campaigns(self) -> list:
        """
        Identify likely attack campaigns from
        patterns in the knowledge graph.

        Looks for combinations of indicators that
        match known malware family patterns.

        Returns list of identified campaign dicts.
        """
        campaigns = []

        # Get all alert nodes
        alert_nodes = self.graph.get_nodes_by_type(
            NodeType.ALERT
        )

        if not alert_nodes:
            return campaigns

        # Collect all alert types
        alert_types = set()
        for alert in alert_nodes:
            alert_types.add(
                alert.properties.get("alert_type", "")
            )
            techniques = alert.properties.get(
                "techniques", []
            )
            for technique in techniques:
                alert_types.add(technique)

        # Check against malware family patterns
        for family_name, family_data in (
            MALWARE_FAMILIES.items()
        ):
            matches = 0
            for indicator in family_data["indicators"]:
                # Check if any alert matches
                for alert in alert_nodes:
                    props = str(
                        alert.properties
                    ).lower()
                    if indicator.lower() in props:
                        matches += 1
                        break

            # If 2+ indicators match identify campaign
            if matches >= 2:
                campaign = {
                    "family": family_name,
                    "confidence": matches / len(
                        family_data["indicators"]
                    ),
                    "actor": family_data["actor"],
                    "ttps": family_data["ttps"],
                    "matched_indicators": matches
                }
                campaigns.append(campaign)

                logger.info(
                    f"Campaign identified: "
                    f"{family_name} "
                    f"(confidence: "
                    f"{campaign['confidence']:.0%})"
                )

                # Add threat actor to graph
                self._add_threat_actor(
                    family_data["actor"],
                    family_name
                )

        return campaigns

    def enrich_host_from_graph(
        self,
        hostname: str
    ) -> dict:
        """
        Enrich a host based on its graph connections.

        Calculates host risk from all connected
        malicious entities — the graph-enriched
        score that surpasses simple ML scoring.

        This is the core value of Layer 3:
            Host connected to malicious IP
            + DGA domain
            + malware process
            = much higher combined risk than any
              single detection alone

        Args:
            hostname: Hostname to enrich

        Returns:
            Enrichment summary dictionary
        """
        host_node_id = f"host:{hostname}"
        host_node = self.graph.get_node(host_node_id)

        if host_node is None:
            return {}

        # Get all connected entities
        neighbors = self.graph.get_neighbors(
            host_node_id,
            direction="both"
        )

        # Calculate combined risk
        malicious_connections = []
        max_neighbor_risk = 0.0

        for neighbor in neighbors:
            if neighbor.risk_score >= 0.5:
                malicious_connections.append({
                    "entity": neighbor.label,
                    "type": neighbor.node_type.value,
                    "risk": neighbor.risk_score,
                    "risk_label": neighbor.risk_label
                })
                if neighbor.risk_score > max_neighbor_risk:
                    max_neighbor_risk = (
                        neighbor.risk_score
                    )

        # Update host risk based on connections
        # If connected to HIGH risk entities
        # host risk increases proportionally
        if malicious_connections:
            connection_risk = (
                max_neighbor_risk * 0.8
            )
            host_node.update_risk(connection_risk)

        enrichment = {
            "hostname": hostname,
            "host_risk_score": host_node.risk_score,
            "host_risk_label": host_node.risk_label,
            "malicious_connections": malicious_connections,
            "total_connections": len(neighbors),
            "threat_connections": len(
                malicious_connections
            )
        }

        logger.info(
            f"Host enriched: {hostname} → "
            f"risk={host_node.risk_score:.2f} "
            f"({len(malicious_connections)} threats)"
        )

        return enrichment

    def get_threat_summary(self) -> dict:
        """
        Generate a comprehensive threat summary
        from the knowledge graph.

        Used by Layer 4 LLM agents as context
        for investigation and reasoning.
        """
        high_risk_nodes = (
            self.graph.get_high_risk_nodes(0.6)
        )

        # Group by type
        threats_by_type = {}
        for node in high_risk_nodes:
            type_name = node.node_type.value
            if type_name not in threats_by_type:
                threats_by_type[type_name] = []
            threats_by_type[type_name].append({
                "entity": node.label,
                "risk_score": node.risk_score,
                "risk_label": node.risk_label,
                "properties": node.properties
            })

        # Get alert nodes
        alert_nodes = self.graph.get_nodes_by_type(
            NodeType.ALERT
        )

        # Get graph statistics
        stats = self.graph.get_statistics()

        return {
            "total_entities": stats["total_nodes"],
            "total_relationships": stats["total_edges"],
            "high_risk_entities": len(high_risk_nodes),
            "threats_by_type": threats_by_type,
            "active_alerts": len(alert_nodes),
            "alert_details": [
                {
                    "type": n.properties.get(
                        "alert_type", ""
                    ),
                    "risk": n.risk_score,
                    "details": n.properties
                }
                for n in alert_nodes
            ],
            "enrichments_applied": (
                self.enrichments_applied
            )
        }

    # ============================================================
    # PRIVATE METHODS
    # ============================================================

    def _add_threat_actor(
        self,
        actor_id: str,
        campaign_name: str
    ) -> Optional[SecurityNode]:
        """
        Add threat actor node to graph.

        Called when campaign identification
        links activity to a known threat actor.
        """
        if actor_id not in THREAT_ACTORS:
            return None

        actor_data = THREAT_ACTORS[actor_id]

        node = SecurityNode(
            node_id=f"actor:{actor_id}",
            node_type=NodeType.THREAT_ACTOR,
            label=actor_data["name"],
            risk_score=0.9,
            properties={
                "actor_id": actor_id,
                "aliases": actor_data["aliases"],
                "motivation": actor_data["motivation"],
                "ttps": actor_data["ttps"],
                "description": (
                    actor_data["description"]
                ),
                "identified_campaign": campaign_name
            }
        )

        actor_node = self.graph.add_node(node)

        # Connect actor to relevant alerts
        alert_nodes = self.graph.get_nodes_by_type(
            NodeType.ALERT
        )
        for alert in alert_nodes:
            self.graph.add_edge(
                source_id=actor_node.node_id,
                target_id=alert.node_id,
                edge_type=EdgeType.TRIGGERS
                if hasattr(EdgeType, "TRIGGERS")
                else EdgeType.TRIGGERED,
                properties={
                    "campaign": campaign_name
                }
            )

        logger.info(
            f"Threat actor added: "
            f"{actor_data['name']} ({actor_id})"
        )

        return actor_node