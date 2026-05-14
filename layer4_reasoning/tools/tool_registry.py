"""
Layer 4 — Agent Tools
Tool Registry

Central registry for all agent tools.
Agents import from here instead of
individual tool files.

USAGE:
    from layer4_reasoning.tools.tool_registry import (
        check_ip_reputation,
        query_knowledge_graph,
        get_ensemble_scores,
        search_past_incidents,
        get_user_permissions,
        get_incident_stats,
        store_incident
    )
"""

from layer4_reasoning.tools.ip_reputation import (
    check_ip_reputation
)
from layer4_reasoning.tools.knowledge_graph_tool import (
    query_knowledge_graph
)
from layer4_reasoning.tools.ensemble_tool import (
    get_ensemble_scores
)
from layer4_reasoning.tools.incident_search import (
    search_past_incidents,
    store_incident,
    get_incident_stats
)
from layer4_reasoning.tools.permissions_tool import (
    get_user_permissions
)

__all__ = [
    "check_ip_reputation",
    "query_knowledge_graph",
    "get_ensemble_scores",
    "search_past_incidents",
    "store_incident",
    "get_incident_stats",
    "get_user_permissions"
]