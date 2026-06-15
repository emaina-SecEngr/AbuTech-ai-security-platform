"""
Layer 4 — Reasoning & Investigation
Reasoning Catalog

The single source of truth describing every component
in Layer 4: the specialist agents, the orchestration
graph, the agent tools, memory, hypothesis hunting,
and the human-in-the-loop approval gate.

Drives both the /api/components endpoint and the live
Layer 4 dashboard.

DESIGN NOTE (honest framing):
    The agents run today via rule-based reasoning and
    enhance with LLM reasoning when an Anthropic key is
    provided (the use_llm path) - the same graceful-
    degradation pattern as the phishing and DNS
    detectors. So agents are marked "built": they run
    now, LLM reasoning is pluggable.
"""

LAYER_INFO = {
    "layer": 4,
    "name": "Reasoning & Investigation",
    "tagline": (
        "Specialist agents investigate each enriched "
        "event — reasoning, gathering context, and "
        "recommending a response — with a human-in-the-"
        "loop gate before any action is taken."
    ),
    "what_it_does": (
        "Layer 4 takes an enriched event and "
        "investigates it. An orchestration graph runs "
        "specialist agents that triage the alert, pull "
        "threat intelligence, query the knowledge graph "
        "and permissions, form hypotheses, and "
        "recommend a response - then a human approves "
        "before anything executes."
    ),
    "why_it_matters": (
        "Detection finds threats; investigation decides "
        "what they mean and what to do. Automating the "
        "investigation cuts response time, while the "
        "human-in-the-loop gate keeps a person in "
        "control of consequential actions - the key "
        "AI-governance control."
    ),
    "how_it_works": (
        "The investigation graph coordinates specialist "
        "agents over a shared state. Each agent uses "
        "tools (scoring, IP reputation, knowledge graph, "
        "permissions, incident search) and reasons "
        "rule-first, with LLM reasoning layered on when "
        "available. Recommendations pass through the "
        "HITL approval gate before reaching the "
        "response layer."
    ),
}


STATUS_META = {
    "built": {
        "label": "Built + tested",
        "color": "#378ADD",
    },
    "agent": {
        "label": "Agent",
        "color": "#7C5CDB",
    },
    "tool": {
        "label": "Agent tool",
        "color": "#1D9E75",
    },
    "gate": {
        "label": "HITL governance gate",
        "color": "#C2492B",
    },
}


COMPONENT_CATALOG = [
    # --- Orchestration ---
    {
        "id": "investigation_graph",
        "name": "Investigation Graph",
        "category": "Orchestration",
        "tech": "LangGraph",
        "icon": "ti-sitemap",
        "does": (
            "Coordinates the specialist agents over a "
            "shared investigation state, routing the "
            "event through triage, intel, and analysis."
        ),
        "value": (
            "Turns a single alert into a structured, "
            "multi-agent investigation - consistent and "
            "repeatable every time."
        ),
        "status": "built",
    },
    {
        "id": "agent_state",
        "name": "Agent State",
        "category": "Orchestration",
        "tech": "Shared state object",
        "icon": "ti-versions",
        "does": (
            "Holds the shared investigation state every "
            "agent reads from and writes to."
        ),
        "value": (
            "Agents build on each other's findings "
            "instead of working in isolation."
        ),
        "status": "built",
    },

    # --- Specialist agents ---
    {
        "id": "triage_agent",
        "name": "Triage Agent",
        "category": "Specialist agents",
        "tech": "Rule-based + LLM (pluggable)",
        "icon": "ti-urgent",
        "does": (
            "Evaluates the alert, assigns a verdict and "
            "priority, and decides whether deeper "
            "investigation is warranted."
        ),
        "value": (
            "Filters noise fast so analysts focus on "
            "what matters - the first triage decision "
            "automated."
        ),
        "status": "agent",
    },
    {
        "id": "intel_agent",
        "name": "Intel Agent",
        "category": "Specialist agents",
        "tech": "Rule-based + LLM (pluggable)",
        "icon": "ti-radar",
        "does": (
            "Pulls threat-intelligence context for the "
            "event and summarizes what is known about "
            "the indicators and actors."
        ),
        "value": (
            "Brings the enrichment context into the "
            "investigation as a readable summary."
        ),
        "status": "agent",
    },
    {
        "id": "specialist_agents",
        "name": "Specialist Agent Suite",
        "category": "Specialist agents",
        "tech": "Rule-based + LLM (pluggable)",
        "icon": "ti-users-group",
        "does": (
            "The full set of specialist agents - triage, "
            "intel, context, forensics, and response - "
            "each focused on one part of the "
            "investigation."
        ),
        "value": (
            "Mirrors how a real SOC divides an "
            "investigation across specialists, automated."
        ),
        "status": "agent",
    },

    # --- Agent tools ---
    {
        "id": "ensemble_tool",
        "name": "Ensemble Scoring Tool",
        "category": "Agent tools",
        "tech": "Calls Layer 2 ensemble",
        "icon": "ti-stack-2",
        "does": (
            "Lets an agent request the ML ensemble score "
            "for an event during investigation."
        ),
        "value": (
            "Agents can re-score and dig into the "
            "detection evidence on demand."
        ),
        "status": "tool",
    },
    {
        "id": "ip_reputation",
        "name": "IP Reputation Tool",
        "category": "Agent tools",
        "tech": "Reputation lookup",
        "icon": "ti-map-pin-search",
        "does": (
            "Looks up the reputation and context of an "
            "IP address involved in the event."
        ),
        "value": (
            "Quick verdict on whether a source IP is "
            "known-bad."
        ),
        "status": "tool",
    },
    {
        "id": "knowledge_graph_tool",
        "name": "Knowledge Graph Tool",
        "category": "Agent tools",
        "tech": "Queries Layer 3 graph",
        "icon": "ti-binary-tree-2",
        "does": (
            "Lets an agent query the security knowledge "
            "graph for related entities and events."
        ),
        "value": (
            "Reveals the attack path - what else this "
            "event connects to."
        ),
        "status": "tool",
    },
    {
        "id": "permissions_tool",
        "name": "Permissions Tool",
        "category": "Agent tools",
        "tech": "Cloud entitlement check",
        "icon": "ti-key",
        "does": (
            "Checks the cloud permissions of an identity "
            "in the investigation."
        ),
        "value": (
            "Shows the blast radius - what a compromised "
            "identity could actually do."
        ),
        "status": "tool",
    },
    {
        "id": "incident_search",
        "name": "Incident Search Tool",
        "category": "Agent tools",
        "tech": "Historical incident search",
        "icon": "ti-history-toggle",
        "does": (
            "Searches past incidents for similar events "
            "and prior resolutions."
        ),
        "value": (
            "Brings institutional memory into each "
            "investigation - have we seen this before?"
        ),
        "status": "tool",
    },
    {
        "id": "tool_registry",
        "name": "Tool Registry",
        "category": "Agent tools",
        "tech": "Tool registration",
        "icon": "ti-plug",
        "does": (
            "Registers the tools available to agents so "
            "the orchestration can wire them in."
        ),
        "value": (
            "Makes the toolset extensible - add a tool, "
            "agents can use it."
        ),
        "status": "tool",
    },

    # --- Hunting + memory ---
    {
        "id": "hypothesis_engine",
        "name": "Hypothesis Engine",
        "category": "Threat hunting",
        "tech": "Hypothesis-driven hunting",
        "icon": "ti-bulb",
        "does": (
            "Generates and tests threat-hunting "
            "hypotheses against the available evidence."
        ),
        "value": (
            "Proactive hunting - looks for threats "
            "rather than waiting for alerts."
        ),
        "status": "built",
    },
    {
        "id": "memory_store",
        "name": "Investigation Memory",
        "category": "Threat hunting",
        "tech": "Agent memory store",
        "icon": "ti-database-heart",
        "does": (
            "Stores investigation context and findings "
            "so agents retain memory across steps."
        ),
        "value": (
            "Investigations build context over time "
            "instead of starting from scratch."
        ),
        "status": "built",
    },

    # --- HITL gate ---
    {
        "id": "hitl_manager",
        "name": "Human-in-the-Loop Gate",
        "category": "Governance control",
        "tech": "Approval gate",
        "icon": "ti-hand-stop",
        "does": (
            "Holds consequential recommendations for "
            "human approval before they reach the "
            "response layer."
        ),
        "value": (
            "Keeps a person in control of irreversible "
            "actions - the core AI-governance control, "
            "mapping to SR 11-7 and EU AI Act human "
            "oversight."
        ),
        "status": "gate",
    },
    {
        "id": "approval_store",
        "name": "Approval Store",
        "category": "Governance control",
        "tech": "Approval persistence + audit",
        "icon": "ti-checklist",
        "does": (
            "Persists every approval decision with a "
            "full audit trail."
        ),
        "value": (
            "Every human decision is recorded - the "
            "evidence trail auditors and regulators "
            "require."
        ),
        "status": "gate",
    },
]


def get_components() -> list:
    return COMPONENT_CATALOG


def get_component(component_id: str) -> dict:
    for c in COMPONENT_CATALOG:
        if c["id"] == component_id:
            return c
    return {}


def get_categories() -> list:
    cats = {}
    for c in COMPONENT_CATALOG:
        cats.setdefault(c["category"], []).append(c)
    return [
        {"category": cat, "components": cs}
        for cat, cs in cats.items()
    ]


def catalog_stats() -> dict:
    by_status = {}
    agents = 0
    tools = 0
    for c in COMPONENT_CATALOG:
        by_status[c["status"]] = (
            by_status.get(c["status"], 0) + 1
        )
        if c["status"] == "agent":
            agents += 1
        if c["status"] == "tool":
            tools += 1
    return {
        "total_components": len(COMPONENT_CATALOG),
        "agents": agents,
        "tools": tools,
        "by_status": by_status,
    }