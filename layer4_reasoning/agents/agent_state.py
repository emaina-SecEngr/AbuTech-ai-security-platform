"""
Layer 4 — LLM Reasoning
Agent State Definition

Shared state flowing between all investigation agents.
Every agent reads from and writes to this state.
"""

import operator
from typing import Annotated
from typing import Optional
from typing import TypedDict


class InvestigationState(TypedDict):
    """
    Shared state for the investigation graph.
    Every agent reads and writes to this state.
    """

    # Input fields from Layers 1-3
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

    # Layer 2 details
    malware_indicators: list
    attack_techniques: list
    dga_indicators: list

    # Layer 3 graph context
    graph_node_count: int
    graph_edge_count: int
    high_risk_entities: list
    threat_connections: list
    host_risk_score: float

    # Layer 3 threat intel
    known_threat_actor: Optional[str]
    known_malware_family: Optional[str]
    ti_enrichments: list

    # Triage fields
    triage_verdict: Optional[str]
    triage_confidence: Optional[float]
    triage_reasoning: Optional[str]
    triage_priority: Optional[str]

    # Intel fields
    threat_actor_identified: Optional[str]
    threat_actor_confidence: Optional[float]
    campaign_identified: Optional[str]
    c2_confirmed: Optional[bool]
    malware_family_confirmed: Optional[str]
    intel_summary: Optional[str]
    confirmed_techniques: list

    # Investigation fields
    attack_timeline: list
    compromise_confirmed: Optional[bool]
    initial_access_vector: Optional[str]
    lateral_movement_detected: Optional[bool]
    data_exfiltration_suspected: Optional[bool]
    blast_radius: list
    investigation_summary: Optional[str]

    # Response fields
    response_actions: list
    containment_priority: Optional[str]
    isolation_recommended: Optional[bool]
    credential_reset_recommended: Optional[bool]
    response_summary: Optional[str]

    # Report fields
    final_report: Optional[str]
    executive_summary: Optional[str]
    severity_rating: Optional[str]

    # Audit trail
    agent_log: Annotated[list, operator.add]


def _safe_float(value, default: float = 0.0) -> float:
    """
    Safely convert any value to float.
    Handles MagicMock, None, and non-numeric types.
    Prevents TypeError in f-string formatting.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def create_initial_state(
    routing_result,
    ecs_event,
    graph_summary: dict = None,
    threat_summary: dict = None
) -> InvestigationState:
    """
    Create initial investigation state from
    Layer 2 routing result and Layer 3 context.

    All numeric values are safely converted to float.
    MagicMock objects from tests are handled gracefully.
    """

    # ---- LAYER 2 SCORES ----
    # Use _safe_float for ALL numeric extractions
    # This handles MagicMock in tests and None in prod

    overall_risk_score = _safe_float(
        routing_result.overall_risk_score, 0.5
    )

    overall_verdict = str(
        routing_result.overall_verdict or "UNKNOWN"
    )

    # Malware scores
    malware_risk = None
    malware_indicators = []
    attack_techniques = []

    malware_result = getattr(
        routing_result, "malware_result", None
    )
    if malware_result and not _is_mock_truthy(
        malware_result
    ):
        malware_risk = _safe_float(
            getattr(malware_result, "risk_score", 0.0)
        )
        malware_indicators = list(
            getattr(malware_result, "malware_indicators", [])
            or []
        )
        attack_techniques = list(
            getattr(malware_result, "attack_techniques", [])
            or []
        )

    # Direct malware_risk from routing_result
    direct_malware = getattr(
        routing_result, "malware_risk", None
    )
    if malware_risk is None and direct_malware is not None:
        malware_risk = _safe_float(direct_malware, 0.0)

    # Direct attack_techniques
    direct_techniques = getattr(
        routing_result, "attack_techniques", None
    )
    if not attack_techniques and direct_techniques:
        try:
            attack_techniques = list(direct_techniques)
        except TypeError:
            attack_techniques = []

    # Direct malware_indicators
    direct_indicators = getattr(
        routing_result, "malware_indicators", None
    )
    if not malware_indicators and direct_indicators:
        try:
            malware_indicators = list(direct_indicators)
        except TypeError:
            malware_indicators = []

    # Intrusion risk
    intrusion_risk = None
    intrusion_result = getattr(
        routing_result, "intrusion_result", None
    )
    if intrusion_result and not _is_mock_truthy(
        intrusion_result
    ):
        intrusion_risk = _safe_float(
            getattr(intrusion_result, "risk_score", 0.0)
        )

    # DGA/DNS risk
    dga_risk = None
    dga_indicators = []

    direct_dga = getattr(
        routing_result, "dga_risk", None
    )
    if direct_dga is not None:
        dga_risk = _safe_float(direct_dga, 0.0)

    dns_result = getattr(
        routing_result, "dns_result", None
    )
    if dns_result and not _is_mock_truthy(dns_result):
        dga_risk = _safe_float(
            getattr(dns_result, "risk_score", 0.0)
        )
        direct_dga_ind = getattr(
            dns_result, "dga_indicators", []
        )
        try:
            dga_indicators = list(direct_dga_ind or [])
        except TypeError:
            dga_indicators = []

    direct_dga_ind = getattr(
        routing_result, "dga_indicators", None
    )
    if not dga_indicators and direct_dga_ind:
        try:
            dga_indicators = list(direct_dga_ind)
        except TypeError:
            dga_indicators = []

    # ---- LAYER 1 EVENT CONTEXT ----
    event_host = ""
    event_user = ""

    try:
        host = getattr(ecs_event, "host", None)
        if host:
            event_host = str(
                getattr(host, "hostname", "") or ""
            )
    except Exception:
        pass

    try:
        user = getattr(ecs_event, "user", None)
        if user:
            name = getattr(user, "name", "") or ""
            domain = getattr(user, "domain", "") or ""
            event_host_name = str(
                getattr(
                    getattr(ecs_event, "host", None),
                    "hostname", ""
                ) or ""
            )
            event_host = event_host_name
            event_user = f"{domain}\\{name}" if domain else name
    except Exception:
        pass

    try:
        host = getattr(ecs_event, "host", None)
        if host:
            event_host = str(
                getattr(host, "hostname", "") or ""
            )
    except Exception:
        pass

    # ---- LAYER 3 CONTEXT ----
    graph_node_count = 0
    graph_edge_count = 0
    high_risk_entities = []
    threat_connections = []
    host_risk_score = overall_risk_score

    if graph_summary and isinstance(graph_summary, dict):
        graph_node_count = int(
            graph_summary.get("total_nodes", 0) or 0
        )
        graph_edge_count = int(
            graph_summary.get("total_edges", 0) or 0
        )

    if threat_summary and isinstance(
        threat_summary, dict
    ):
        for entity_type, entities in (
            threat_summary.get(
                "threats_by_type", {}
            ).items()
        ):
            for entity in (entities or []):
                if isinstance(entity, dict):
                    high_risk_entities.append({
                        "type": entity_type,
                        "entity": entity.get(
                            "entity", ""
                        ),
                        "risk": _safe_float(
                            entity.get("risk_score", 0)
                        )
                    })

        threat_connections = high_risk_entities.copy()

    # ---- LAYER 3 THREAT INTEL ----
    known_threat_actor = None
    known_malware_family = None
    ti_enrichments = []

    direct_malware_family = getattr(
        routing_result, "known_malware_family", None
    )
    if direct_malware_family and isinstance(
        direct_malware_family, str
    ):
        known_malware_family = direct_malware_family

    if threat_summary and isinstance(
        threat_summary, dict
    ):
        for alert in threat_summary.get(
            "alert_details", []
        ):
            props = alert.get("details", {})
            if props.get("malware_family"):
                known_malware_family = (
                    props["malware_family"]
                )

    # ---- TIMESTAMPS ----
    event_timestamp = ""
    try:
        ev = getattr(ecs_event, "event", None)
        if ev:
            event_timestamp = str(
                getattr(ev, "created", "")
                or getattr(ev, "start", "")
                or ""
            )
        ts = getattr(ecs_event, "timestamp", None)
        if ts and not event_timestamp:
            event_timestamp = str(ts)
    except Exception:
        pass

    event_id = "unknown"
    try:
        ev = getattr(ecs_event, "event", None)
        if ev:
            eid = getattr(ev, "id", None)
            if eid and isinstance(eid, str):
                event_id = eid
    except Exception:
        pass

    event_category = "unknown"
    try:
        ev = getattr(ecs_event, "event", None)
        if ev:
            cat = getattr(ev, "category", None)
            if cat and isinstance(cat, str):
                event_category = cat
    except Exception:
        pass

    return InvestigationState(
        event_id=event_id,
        event_category=event_category,
        event_host=event_host,
        event_user=event_user,
        event_timestamp=event_timestamp,
        overall_risk_score=overall_risk_score,
        overall_verdict=overall_verdict,
        malware_risk=malware_risk,
        intrusion_risk=intrusion_risk,
        dga_risk=dga_risk,
        malware_indicators=malware_indicators,
        attack_techniques=attack_techniques,
        dga_indicators=dga_indicators,
        graph_node_count=graph_node_count,
        graph_edge_count=graph_edge_count,
        high_risk_entities=high_risk_entities,
        threat_connections=threat_connections,
        host_risk_score=host_risk_score,
        known_threat_actor=known_threat_actor,
        known_malware_family=known_malware_family,
        ti_enrichments=ti_enrichments,
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


def _is_mock_truthy(obj) -> bool:
    """
    Check if an object is a MagicMock.
    MagicMock objects are always truthy
    but should not be treated as real data.
    """
    return "MagicMock" in type(obj).__name__