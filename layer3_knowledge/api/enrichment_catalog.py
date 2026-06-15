"""
Layer 3 — Knowledge & Enrichment
Enrichment Catalog

The single source of truth describing every enrichment
component in Layer 3: what it adds, the source it draws
on, how it enriches an event, and its build status.

Drives both the /api/enrichers endpoint and the live
Layer 3 dashboard.
"""

LAYER_INFO = {
    "layer": 3,
    "name": "Knowledge & Enrichment",
    "tagline": (
        "Turns a scored event into actionable "
        "intelligence — mapping it to MITRE ATT&CK, "
        "enriching it with seven threat-intel sources, "
        "and correlating it in a knowledge graph."
    ),
    "what_it_does": (
        "Layer 3 takes a scored event from the ML layer "
        "and adds context. It maps the event to MITRE "
        "ATT&CK, checks it against live threat "
        "intelligence, flags breached-credential and "
        "dark-web exposure, and correlates it with "
        "related events in a security knowledge graph."
    ),
    "why_it_matters": (
        "A score tells you something is risky. "
        "Enrichment tells you why, who is behind it, "
        "and what to do. This is the difference between "
        "an alert and actionable intelligence an analyst "
        "can act on immediately."
    ),
    "how_it_works": (
        "Each enricher attaches context to the event. "
        "The MITRE enricher adds technique, tactic, and "
        "mitigations; the threat-intel feeds add IOC "
        "matches and actor attribution; the graph "
        "correlates the event with related entities to "
        "reveal the attack path."
    ),
}


STATUS_META = {
    "built": {
        "label": "Built + tested",
        "color": "#378ADD",
    },
    "feed": {
        "label": "Feed integration",
        "color": "#1D9E75",
    },
    "graph": {
        "label": "Knowledge graph",
        "color": "#7C5CDB",
    },
}


ENRICHER_CATALOG = [
    # --- MITRE ---
    {
        "id": "mitre",
        "name": "MITRE ATT&CK Enricher",
        "category": "Framework mapping",
        "source": "MITRE ATT&CK knowledge base",
        "icon": "ti-target-arrow",
        "adds": (
            "The attack technique, tactic, and "
            "recommended mitigations for an event."
        ),
        "how": (
            "Maps the event's signals to ATT&CK "
            "techniques, then looks up the tactic, "
            "related techniques, and mitigations from "
            "the knowledge base."
        ),
        "value": (
            "Translates a raw detection into the "
            "language security teams and regulators "
            "use — explainable, standardized."
        ),
        "status": "built",
    },

    # --- Threat intel feeds ---
    {
        "id": "cisa",
        "name": "CISA KEV Ingester",
        "category": "Threat intelligence",
        "source": "CISA Known Exploited Vulnerabilities",
        "icon": "ti-shield-bolt",
        "adds": (
            "Whether a vulnerability is on CISA's "
            "actively-exploited list."
        ),
        "how": (
            "Ingests the CISA KEV catalog and flags "
            "events touching known-exploited CVEs — the "
            "highest-priority vulnerabilities."
        ),
        "value": (
            "Prioritizes the vulnerabilities attackers "
            "are actually exploiting right now, not "
            "just theoretical ones."
        ),
        "status": "feed",
    },
    {
        "id": "abusech",
        "name": "abuse.ch Feed",
        "category": "Threat intelligence",
        "source": "abuse.ch (URLhaus, ThreatFox)",
        "icon": "ti-virus",
        "adds": (
            "Malware, C2, and malicious-URL indicator "
            "matches."
        ),
        "how": (
            "Matches event IPs, domains, and hashes "
            "against abuse.ch malware and command-and-"
            "control indicators."
        ),
        "value": (
            "Confirms when an event involves a known "
            "malicious infrastructure, with no guesswork."
        ),
        "status": "feed",
    },
    {
        "id": "otx",
        "name": "AlienVault OTX",
        "category": "Threat intelligence",
        "source": "AlienVault Open Threat Exchange",
        "icon": "ti-world-bolt",
        "adds": (
            "Community threat-intelligence pulse "
            "matches and IOC context."
        ),
        "how": (
            "Checks event indicators against OTX pulses "
            "to attach community-sourced threat context."
        ),
        "value": (
            "Crowdsourced intelligence broadens coverage "
            "beyond any single vendor feed."
        ),
        "status": "feed",
    },
    {
        "id": "recorded_future",
        "name": "Recorded Future",
        "category": "Threat intelligence",
        "source": "Recorded Future risk scores",
        "icon": "ti-chart-dots",
        "adds": (
            "Commercial-grade risk scores and actor "
            "attribution for indicators."
        ),
        "how": (
            "Enriches indicators with Recorded Future "
            "risk scoring and threat-actor context."
        ),
        "value": (
            "Premium intelligence adds confidence and "
            "attribution to high-stakes detections."
        ),
        "status": "feed",
    },
    {
        "id": "hibp",
        "name": "Have I Been Pwned",
        "category": "Credential exposure",
        "source": "HIBP breached-credential data",
        "icon": "ti-key-off",
        "adds": (
            "Whether an identity's credentials appear "
            "in a known breach."
        ),
        "how": (
            "Checks accounts involved in an event "
            "against breached-credential datasets."
        ),
        "value": (
            "Flags accounts at elevated risk because "
            "their credentials are already exposed."
        ),
        "status": "feed",
    },
    {
        "id": "dark_web",
        "name": "Dark Web Enricher",
        "category": "Credential exposure",
        "source": "Dark web exposure data",
        "icon": "ti-spy",
        "adds": (
            "Whether an identity or asset is exposed on "
            "dark web markets or forums."
        ),
        "how": (
            "Correlates event identities and assets with "
            "dark web exposure indicators."
        ),
        "value": (
            "Early warning that an identity or asset is "
            "being discussed or sold by attackers."
        ),
        "status": "feed",
    },
    {
        "id": "threat_enricher",
        "name": "Threat Enricher",
        "category": "Threat intelligence",
        "source": "Aggregated feeds",
        "icon": "ti-radar-2",
        "adds": (
            "A consolidated threat verdict combining all "
            "feed matches for an event."
        ),
        "how": (
            "Aggregates the individual feed results into "
            "one enriched threat context on the event."
        ),
        "value": (
            "One unified threat verdict instead of "
            "scattered feed hits — easier to act on."
        ),
        "status": "built",
    },

    # --- Feed infrastructure ---
    {
        "id": "stix",
        "name": "STIX / TAXII Processor",
        "category": "Intel infrastructure",
        "source": "STIX 2.x threat-intel standard",
        "icon": "ti-file-code-2",
        "adds": (
            "Standardized parsing of structured threat "
            "intelligence."
        ),
        "how": (
            "Parses STIX objects so intelligence from "
            "any TAXII source can be ingested in a "
            "common format."
        ),
        "value": (
            "Industry-standard intel interchange — "
            "integrates with any compliant feed."
        ),
        "status": "built",
    },
    {
        "id": "feed_manager",
        "name": "Feed Manager + Scheduler",
        "category": "Intel infrastructure",
        "source": "All threat-intel feeds",
        "icon": "ti-refresh",
        "adds": (
            "Orchestrated, scheduled refresh of every "
            "threat-intel source."
        ),
        "how": (
            "Manages and schedules feed pulls so "
            "intelligence stays current without manual "
            "refresh."
        ),
        "value": (
            "Keeps enrichment fresh automatically — "
            "stale intel misses active threats."
        ),
        "status": "built",
    },

    # --- CIEM ---
    {
        "id": "ciem",
        "name": "CIEM Enricher",
        "category": "Cloud entitlement",
        "source": "Cloud IAM entitlement analysis",
        "icon": "ti-cloud-lock",
        "adds": (
            "Cloud identity and entitlement context — "
            "excess permissions, privilege risk."
        ),
        "how": (
            "Analyzes the cloud entitlements of the "
            "identity in an event to flag excessive or "
            "risky permissions."
        ),
        "value": (
            "Reveals whether a compromised identity has "
            "dangerous over-permissions to exploit — "
            "the blast radius."
        ),
        "status": "built",
    },

    # --- Knowledge graph ---
    {
        "id": "security_graph",
        "name": "Security Knowledge Graph",
        "category": "Correlation",
        "source": "Correlated events + entities",
        "icon": "ti-binary-tree-2",
        "adds": (
            "The relationships between events, "
            "identities, and assets — the attack path."
        ),
        "how": (
            "Builds a graph of entities and events so "
            "related activity is connected, revealing "
            "multi-step attacks no single event shows."
        ),
        "value": (
            "Turns isolated alerts into a connected "
            "attack story — the full picture, not "
            "scattered dots."
        ),
        "status": "graph",
    },
]


def get_enrichers() -> list:
    return ENRICHER_CATALOG


def get_enricher(enricher_id: str) -> dict:
    for e in ENRICHER_CATALOG:
        if e["id"] == enricher_id:
            return e
    return {}


def get_categories() -> list:
    cats = {}
    for e in ENRICHER_CATALOG:
        cats.setdefault(e["category"], []).append(e)
    return [
        {"category": c, "enrichers": es}
        for c, es in cats.items()
    ]


def catalog_stats() -> dict:
    by_status = {}
    feed_count = 0
    for e in ENRICHER_CATALOG:
        by_status[e["status"]] = (
            by_status.get(e["status"], 0) + 1
        )
        if e["category"] == "Threat intelligence" or (
            e["category"] == "Credential exposure"
        ):
            feed_count += 1
    return {
        "total_enrichers": len(ENRICHER_CATALOG),
        "intel_sources": feed_count,
        "by_status": by_status,
    }