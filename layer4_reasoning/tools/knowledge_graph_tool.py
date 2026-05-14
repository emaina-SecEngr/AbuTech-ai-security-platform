"""
Layer 4 — Agent Tools
Tool 2: Knowledge Graph Query Tool

Queries Layer 3 SecurityKnowledgeGraph
for entity context and relationships.

Used by IntelAgent and InvestigationAgent
to get real graph data instead of guessing.

USAGE BY AGENTS:
    result = query_knowledge_graph("svc_backup")
    print(result["risk_score"])      # 0.94
    print(result["connections"])     # [...]
    print(result["threat_proximity"]) # 1
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def query_knowledge_graph(
    entity_id: str,
    entity_type: str = None,
    max_connections: int = 10
) -> dict:
    """
    Query the security knowledge graph
    for an entity and its connections.

    Args:
        entity_id: Entity to look up
                   (IP, username, hostname, etc.)
        entity_type: Optional type hint
                     (user, ip_address, host, etc.)
        max_connections: Max connections to return

    Returns:
        dict with:
            entity_id: str
            entity_type: str
            risk_score: float
            found: bool
            connections: list of connected entities
            threat_proximity: int (hops to nearest threat)
            threat_connections: list of threat nodes
            summary: str
    """
    if not entity_id:
        return _empty_result(entity_id)

    try:
        from layer3_knowledge.graph.security_graph \
            import SecurityKnowledgeGraph

        kg = SecurityKnowledgeGraph()

        # Try to find entity in graph
        node_data = kg.get_node(entity_id)

        if node_data:
            return _build_result_from_graph(
                kg, entity_id, node_data,
                max_connections
            )
        else:
            # Entity not in graph yet
            # Use rule-based heuristics
            return _rule_based_lookup(
                entity_id, entity_type
            )

    except Exception as e:
        logger.warning(
            f"Knowledge graph query failed "
            f"for {entity_id}: {e}"
        )
        return _rule_based_lookup(
            entity_id, entity_type
        )


def _build_result_from_graph(
    kg,
    entity_id: str,
    node_data: dict,
    max_connections: int
) -> dict:
    """Build result from real graph data"""
    try:
        # Get connections
        connections = kg.get_neighbors(
            entity_id, max_connections
        )

        # Get threat proximity
        threat_proximity = kg.get_threat_distance(
            entity_id
        )

        # Get threat connections
        threat_connections = [
            c for c in connections
            if c.get("is_threat", False)
        ]

        risk_score = float(
            node_data.get("risk_score", 0.0)
        )
        entity_type = node_data.get(
            "entity_type", "unknown"
        )

        summary = _build_graph_summary(
            entity_id, entity_type,
            risk_score, connections,
            threat_proximity, threat_connections
        )

        return {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "risk_score": risk_score,
            "found": True,
            "connections": connections[
                :max_connections
            ],
            "threat_proximity": threat_proximity,
            "threat_connections": threat_connections,
            "summary": summary
        }

    except Exception as e:
        logger.warning(
            f"Failed to build graph result: {e}"
        )
        return _empty_result(entity_id)


def _rule_based_lookup(
    entity_id: str,
    entity_type: str = None
) -> dict:
    """
    Rule-based fallback when graph unavailable.
    Uses heuristics to estimate risk.
    """
    risk_score = 0.0
    threat_type = "unknown"
    summary_parts = []

    entity_lower = entity_id.lower()

    # Service account patterns
    if any(
        pattern in entity_lower
        for pattern in ["svc_", "service_", "_svc"]
    ):
        risk_score = 0.4
        threat_type = "service_account"
        summary_parts.append(
            "Service account — elevated scrutiny required"
        )

    # Admin patterns
    if any(
        pattern in entity_lower
        for pattern in ["admin", "root", "sudo"]
    ):
        risk_score = max(risk_score, 0.5)
        threat_type = "privileged_account"
        summary_parts.append(
            "Privileged account — high blast radius"
        )

    # Backup patterns with data access
    if "backup" in entity_lower:
        risk_score = max(risk_score, 0.45)
        summary_parts.append(
            "Backup service — should only access backup data"
        )

    summary = (
        f"Entity {entity_id} (rule-based): "
        + " | ".join(summary_parts)
        if summary_parts
        else f"Entity {entity_id}: no risk indicators found"
    )

    return {
        "entity_id": entity_id,
        "entity_type": entity_type or threat_type,
        "risk_score": risk_score,
        "found": False,
        "connections": [],
        "threat_proximity": -1,
        "threat_connections": [],
        "summary": summary
    }


def _build_graph_summary(
    entity_id: str,
    entity_type: str,
    risk_score: float,
    connections: list,
    threat_proximity: int,
    threat_connections: list
) -> str:
    """Build human readable graph summary"""
    summary = (
        f"Entity {entity_id} ({entity_type}): "
        f"risk={risk_score:.2f}, "
        f"{len(connections)} connections. "
    )

    if threat_proximity == 0:
        summary += "IS a known threat node. "
    elif threat_proximity == 1:
        summary += (
            f"DIRECTLY connected to "
            f"{len(threat_connections)} threat node(s). "
        )
    elif threat_proximity == 2:
        summary += "2 hops from nearest threat. "
    elif threat_proximity > 2:
        summary += f"{threat_proximity} hops from threats. "
    else:
        summary += "No path to known threats found. "

    if risk_score >= 0.8:
        summary += "CRITICAL risk — investigate immediately."
    elif risk_score >= 0.6:
        summary += "HIGH risk — investigate within 1 hour."

    return summary


def _empty_result(entity_id: str) -> dict:
    """Return empty result for invalid input"""
    return {
        "entity_id": entity_id or "unknown",
        "entity_type": "unknown",
        "risk_score": 0.0,
        "found": False,
        "connections": [],
        "threat_proximity": -1,
        "threat_connections": [],
        "summary": "No entity provided for lookup."
    }