"""
Layer 4 — LLM Reasoning
Agent State Definition

This module defines the shared state that flows
between all agents in the investigation graph.

Why Shared State Matters:
    In LangGraph every agent reads from and writes to
    a shared state object. This is how agents
    communicate without calling each other directly.

    Triage Agent writes:  state["verdict"] = "INVESTIGATE"
    Intel Agent reads:    state["verdict"]
    Intel Agent writes:   state["threat_actor"] = "TA542"
    Report Agent reads:   state["threat_actor"]

    The state is the single source of truth for the
    entire investigation. Every agent sees the complete
    picture of what previous agents discovered.

State Design Principles:
    1. Immutable history — agents append to lists
       never overwrite. Every finding is preserved.

    2. Confidence tracking — every finding includes
       a confidence score. Low confidence findings
       are flagged for human review.

    3. Complete audit trail — every agent logs its
       reasoning. The report agent uses this to
       generate a transparent investigation report.

    4. Layer integration — state carries context
       from all three previous layers so agents
       have full platform intelligence available.
"""

from typing import TypedDict
from typing import Optional
from typing import Annotated
import operator


class InvestigationState(TypedDict):
    """
    Shared state flowing through the investigation graph.

    Every agent reads from this state and writes
    findings back to it. LangGraph manages state
    transitions between agents.

    Fields are organized by which agent populates them:
    - Input fields: populated before graph starts
    - Triage fields: populated by TriageAgent
    - Intel fields: populated by IntelAgent
    - Investigation fields: populated by InvestigationAgent
    - Response fields: populated by ResponseAgent
    - Report fields: populated by ReportAgent
    """

    # ============================================================
    # INPUT FIELDS
    # Populated before the graph starts
    # Carry Layer 1, 2, 3 context into the agents
    # ============================================================

    # Original event data from Layer 1
    event_id: str
    event_category: str
    event_host: str
    event_user: str
    event_timestamp: str

    # Layer 2 ML scores
    overall_risk_score: float
    overall_verdict: str
    malware_risk: Optional[float]
    intrusion_risk: Optional[float]
    dga_risk: Optional[float]

    # Layer 2 detection details
    malware_indicators: list
    attack_techniques: list
    dga_indicators: list

    # Layer 3 knowledge graph context
    graph_node_count: int
    graph_edge_count: int
    high_risk_entities: list
    threat_connections: list
    host_risk_score: float

    # Layer 3 threat intelligence
    known_threat_actor: Optional[str]
    known_malware_family: Optional[str]
    ti_enrichments: list

    # ============================================================
    # TRIAGE FIELDS
    # Populated by TriageAgent
    # ============================================================

    # Primary triage decision
    triage_verdict: Optional[str]
    # INVESTIGATE — proceed to full investigation
    # MONITOR     — watch but do not escalate
    # CLOSE       — false positive, no action needed

    triage_confidence: Optional[float]
    triage_reasoning: Optional[str]
    triage_priority: Optional[str]
    # CRITICAL, HIGH, MEDIUM, LOW

    # ============================================================
    # INTEL FIELDS
    # Populated by IntelAgent
    # ============================================================

    threat_actor_identified: Optional[str]
    threat_actor_confidence: Optional[float]
    campaign_identified: Optional[str]
    c2_confirmed: Optional[bool]
    malware_family_confirmed: Optional[str]
    intel_summary: Optional[str]

    # ATT&CK technique mapping
    confirmed_techniques: list

    # ============================================================
    # INVESTIGATION FIELDS
    # Populated by InvestigationAgent
    # ============================================================

    attack_timeline: list
    # List of dicts: {timestamp, event, significance}

    compromise_confirmed: Optional[bool]
    initial_access_vector: Optional[str]
    lateral_movement_detected: Optional[bool]
    data_exfiltration_suspected: Optional[bool]
    blast_radius: list
    # List of affected entities

    investigation_summary: Optional[str]

    # ============================================================
    # RESPONSE FIELDS
    # Populated by ResponseAgent
    # ============================================================

    response_actions: list
    # List of recommended/taken actions

    containment_priority: Optional[str]
    isolation_recommended: Optional[bool]
    credential_reset_recommended: Optional[bool]
    response_summary: Optional[str]

    # ============================================================
    # REPORT FIELDS
    # Populated by ReportAgent
    # ============================================================

    final_report: Optional[str]
    executive_summary: Optional[str]
    severity_rating: Optional[str]

    # ============================================================
    # AUDIT TRAIL
    # Every agent appends to this list
    # Provides complete reasoning transparency
    # ============================================================

    agent_log: Annotated[list, operator.add]
    # operator.add means LangGraph merges lists
    # rather than overwriting them


def create_initial_state(
    routing_result,
    ecs_event,
    graph_summary: dict = None,
    threat_summary: dict = None
) -> InvestigationState:
    """
    Create initial investigation state from
    Layer 2 routing result and Layer 3 context.

    This function bridges Layer 2 and Layer 3
    output into the Layer 4 agent state format.

    Args:
        routing_result: Layer 2 RoutingResult
        ecs_event: ECSNormalized from Layer 1
        graph_summary: Layer 3 graph statistics
        threat_summary: Layer 3 threat intelligence

    Returns:
        InvestigationState ready for agent graph
    """

    # Extract Layer 2 scores
    malware_risk = None
    malware_indicators = []
    attack_techniques = []

    if routing_result.malware_result:
        mal = routing_result.malware_result
        malware_risk = mal.risk_score
        malware_indicators = getattr(
            mal, "malware_indicators", []
        )
        attack_techniques = getattr(
            mal, "attack_techniques", []
        )

    intrusion_risk = None
    if routing_result.intrusion_result:
        intrusion_risk = (
            routing_result.intrusion_result.risk_score
        )

    dga_risk = None
    dga_indicators = []
    if routing_result.dns_result:
        dns = routing_result.dns_result
        dga_risk = dns.risk_score
        dga_indicators = getattr(
            dns, "dga_indicators", []
        )

    # Extract Layer 1 event context
    event_host = ""
    event_user = ""
    if ecs_event.host:
        event_host = ecs_event.host.hostname or ""
    if ecs_event.user:
        event_user = (
            f"{ecs_event.user.domain}\\"
            f"{ecs_event.user.name}"
        )

    # Extract Layer 3 context
    graph_node_count = 0
    graph_edge_count = 0
    high_risk_entities = []
    threat_connections = []
    host_risk_score = routing_result.overall_risk_score

    if graph_summary:
        graph_node_count = graph_summary.get(
            "total_nodes", 0
        )
        graph_edge_count = graph_summary.get(
            "total_edges", 0
        )

    if threat_summary:
        high_risk_entities = []
        for entity_type, entities in (
            threat_summary.get(
                "threats_by_type", {}
            ).items()
        ):
            for entity in entities:
                high_risk_entities.append({
                    "type": entity_type,
                    "entity": entity["entity"],
                    "risk": entity["risk_score"]
                })

        threat_connections = high_risk_entities

    # Extract known threat intel
    known_threat_actor = None
    known_malware_family = None
    ti_enrichments = []

    if threat_summary:
        for alert in threat_summary.get(
            "alert_details", []
        ):
            props = alert.get("details", {})
            if props.get("malware_family"):
                known_malware_family = (
                    props["malware_family"]
                )

    return InvestigationState(
        # Input fields
        event_id=ecs_event.event.id or "unknown",
        event_category=(
            ecs_event.event.category or "unknown"
        ),
        event_host=event_host,
        event_user=event_user,
        event_timestamp=ecs_event.timestamp or "",

        # Layer 2 scores
        overall_risk_score=(
            routing_result.overall_risk_score
        ),
        overall_verdict=routing_result.overall_verdict,
        malware_risk=malware_risk,
        intrusion_risk=intrusion_risk,
        dga_risk=dga_risk,

        # Layer 2 details
        malware_indicators=malware_indicators,
        attack_techniques=attack_techniques,
        dga_indicators=dga_indicators,

        # Layer 3 graph context
        graph_node_count=graph_node_count,
        graph_edge_count=graph_edge_count,
        high_risk_entities=high_risk_entities,
        threat_connections=threat_connections,
        host_risk_score=host_risk_score,

        # Layer 3 threat intel
        known_threat_actor=known_threat_actor,
        known_malware_family=known_malware_family,
        ti_enrichments=ti_enrichments,

        # Initialize agent output fields
        triage_verdict=None,
        triage_confidence=None,
        triage_reasoning=None,
        triage_priority=None,

        threat_actor_identified=None,
        threat_actor_confidence=None,
        campaign_identified=None,
        c2_confirmed=None,
        malware_family_confirmed=None,
        intel_summary=None,
        confirmed_techniques=[],

        attack_timeline=[],
        compromise_confirmed=None,
        initial_access_vector=None,
        lateral_movement_detected=None,
        data_exfiltration_suspected=None,
        blast_radius=[],
        investigation_summary=None,

        response_actions=[],
        containment_priority=None,
        isolation_recommended=None,
        credential_reset_recommended=None,
        response_summary=None,

        final_report=None,
        executive_summary=None,
        severity_rating=None,

        agent_log=[]
    )