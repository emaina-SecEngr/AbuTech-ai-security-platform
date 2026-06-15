"""
Layer 5 — Response & Interface
Response Catalog

The single source of truth describing every component
in Layer 5: the SOAR playbook engine and its parts, the
SIEM integrations, the analytics repository, and the six
real cloud-execution scripts that carry out remediation.

Drives both the /api/components endpoint and the live
Layer 5 dashboard.

THE TWO-PART STORY:
    DECISION / ORCHESTRATION (the brain) - the SOAR
    engine selects and sequences playbooks.
    EXECUTION (the hands) - the six boto3 cloud scripts
    perform the actual remediation, dry-run safe and
    reversible.
    The human-in-the-loop gate (Layer 4) sits between
    decision and execution.
"""

LAYER_INFO = {
    "layer": 5,
    "name": "Response & Interface",
    "tagline": (
        "Turns an approved decision into action — a "
        "SOAR engine orchestrates the response, six "
        "real cloud scripts execute it, and the result "
        "is pushed to the enterprise SIEM."
    ),
    "what_it_does": (
        "Layer 5 executes the response. The SOAR engine "
        "selects the right playbook for the threat and "
        "sequences the steps; real boto3 cloud scripts "
        "perform the remediation; and the outcome is "
        "routed to enterprise SIEMs for visibility and "
        "record-keeping."
    ),
    "why_it_matters": (
        "Detection and investigation are worthless "
        "without action. This layer closes the loop - "
        "fast, consistent, auditable response - while "
        "keeping execution separate from decision so a "
        "human stays in control of irreversible actions."
    ),
    "how_it_works": (
        "The engine maps a threat to a playbook recipe "
        "and runs its actions in order, with rollback. "
        "Safe actions can auto-execute; irreversible "
        "ones wait for human approval. The cloud scripts "
        "are the execution arm - dry-run by default, "
        "reversible, and unit-tested against mocked AWS."
    ),
}


STATUS_META = {
    "built": {
        "label": "Built + tested",
        "color": "#378ADD",
    },
    "soar": {
        "label": "SOAR orchestration",
        "color": "#7C5CDB",
    },
    "siem": {
        "label": "SIEM integration",
        "color": "#1D9E75",
    },
    "executor": {
        "label": "Real cloud executor",
        "color": "#C28A2B",
    },
}


COMPONENT_CATALOG = [
    # --- SOAR orchestration ---
    {
        "id": "playbook_engine",
        "name": "Playbook Engine",
        "category": "SOAR orchestration",
        "tech": "Python orchestration",
        "icon": "ti-engine",
        "does": (
            "Runs a playbook's actions in order, handles "
            "failures and rollback, and enforces the "
            "approval gate on irreversible steps."
        ),
        "value": (
            "Safe, consistent automated response - the "
            "same threat gets the same correct handling "
            "every time."
        ),
        "status": "soar",
    },
    {
        "id": "playbook_library",
        "name": "Playbook Library",
        "category": "SOAR orchestration",
        "tech": "Threat-to-playbook recipes",
        "icon": "ti-books",
        "does": (
            "Maps each threat type and MITRE technique "
            "to an ordered recipe of response actions."
        ),
        "value": (
            "Different threats get different responses - "
            "data exfil isn't handled like cryptomining."
        ),
        "status": "soar",
    },
    {
        "id": "playbook_actions",
        "name": "Playbook Actions",
        "category": "SOAR orchestration",
        "tech": "Atomic response actions",
        "icon": "ti-bolt",
        "does": (
            "The reusable building-block actions - "
            "isolate, block, snapshot, notify - that "
            "playbooks are composed from."
        ),
        "value": (
            "Build an action once, reuse it across every "
            "playbook - clean and extensible."
        ),
        "status": "soar",
    },

    # --- Cloud executors (the real hands) ---
    {
        "id": "ec2_quarantine",
        "name": "EC2 Quarantine",
        "category": "Cloud executors (boto3)",
        "tech": "boto3 · EC2 security groups",
        "icon": "ti-shield-lock",
        "does": (
            "Network-isolates a compromised EC2 instance "
            "by swapping its security groups to deny-all, "
            "keeping it alive for forensics."
        ),
        "value": (
            "Stops the attacker without destroying "
            "evidence - mirrors CrowdStrike containment. "
            "Reversible, dry-run safe, tested."
        ),
        "status": "executor",
    },
    {
        "id": "forensic_capture",
        "name": "Forensic Capture",
        "category": "Cloud executors (boto3)",
        "tech": "boto3 · EBS snapshots",
        "icon": "ti-camera",
        "does": (
            "Snapshots every EBS volume on a compromised "
            "instance with chain-of-custody tags."
        ),
        "value": (
            "Preserves disk evidence that survives even "
            "if the instance is destroyed. Tested."
        ),
        "status": "executor",
    },
    {
        "id": "perimeter_block",
        "name": "Perimeter Block",
        "category": "Cloud executors (boto3)",
        "tech": "boto3 · NACL / WAF",
        "icon": "ti-ban",
        "does": (
            "Blocks a malicious IP at the cloud edge - "
            "NACL for network attacks, WAF for web "
            "attacks - extracted from a GuardDuty finding."
        ),
        "value": (
            "Matches the enforcement layer to the attack. "
            "Reversible, dry-run safe, tested."
        ),
        "status": "executor",
    },
    {
        "id": "port_exposure_remediation",
        "name": "Port Exposure Remediation",
        "category": "Cloud executors (boto3)",
        "tech": "boto3 · security-group rules",
        "icon": "ti-door",
        "does": (
            "Closes security-group rules exposing SSH, "
            "RDP, or databases to the whole internet - "
            "surgically removing only the offending rule."
        ),
        "value": (
            "Shuts the most common cloud misconfiguration "
            "before scanners exploit it. Tested."
        ),
        "status": "executor",
    },
    {
        "id": "revoke_iam_credentials",
        "name": "Revoke IAM Credentials",
        "category": "Cloud executors (boto3)",
        "tech": "boto3 · IAM",
        "icon": "ti-key-off",
        "does": (
            "Contains a compromised IAM user - "
            "deactivates the leaked key, attaches a "
            "deny-all policy, and revokes active sessions."
        ),
        "value": (
            "Kills the #1 cloud attack vector - leaked "
            "credentials - in one shot. Tested."
        ),
        "status": "executor",
    },
    {
        "id": "revoke_sts_sessions",
        "name": "Revoke STS Sessions",
        "category": "Cloud executors (boto3)",
        "tech": "boto3 · IAM time-based deny",
        "icon": "ti-clock-off",
        "does": (
            "Neutralizes stolen EC2 instance-role tokens "
            "by denying any token issued before the "
            "revocation moment."
        ),
        "value": (
            "Kills credentials you can't delete, without "
            "taking the legitimate workload offline. "
            "Tested."
        ),
        "status": "executor",
    },

    # --- SIEM integrations ---
    {
        "id": "siem_router",
        "name": "SIEM Router",
        "category": "SIEM integration",
        "tech": "Multi-SIEM routing",
        "icon": "ti-router",
        "does": (
            "Routes enriched events and response outcomes "
            "to the configured enterprise SIEM(s)."
        ),
        "value": (
            "Fits into the existing security stack "
            "instead of replacing it."
        ),
        "status": "siem",
    },
    {
        "id": "splunk_destination",
        "name": "Splunk Destination",
        "category": "SIEM integration",
        "tech": "Splunk HEC",
        "icon": "ti-brand-splunk",
        "does": (
            "Delivers events to Splunk in its expected "
            "format."
        ),
        "value": (
            "Native Splunk integration for teams that "
            "run it."
        ),
        "status": "siem",
    },
    {
        "id": "sentinel_destination",
        "name": "Microsoft Sentinel Destination",
        "category": "SIEM integration",
        "tech": "Sentinel / Log Analytics",
        "icon": "ti-brand-azure",
        "does": (
            "Delivers events to Microsoft Sentinel."
        ),
        "value": (
            "Native integration for Azure-centric SOCs."
        ),
        "status": "siem",
    },
    {
        "id": "qradar_destination",
        "name": "IBM QRadar Destination",
        "category": "SIEM integration",
        "tech": "QRadar",
        "icon": "ti-brand-ibm",
        "does": (
            "Delivers events to IBM QRadar."
        ),
        "value": (
            "Native integration for QRadar shops."
        ),
        "status": "siem",
    },

    # --- Analytics + interface ---
    {
        "id": "kql_repository",
        "name": "KQL Detection Repository",
        "category": "Analytics & interface",
        "tech": "KQL queries",
        "icon": "ti-search",
        "does": (
            "A library of KQL detection queries for "
            "hunting and analytics in Sentinel / Log "
            "Analytics."
        ),
        "value": (
            "Ready-made detections analysts can run "
            "immediately."
        ),
        "status": "built",
    },
    {
        "id": "api_routes",
        "name": "Response API",
        "category": "Analytics & interface",
        "tech": "FastAPI",
        "icon": "ti-api",
        "does": (
            "The API surface for the response layer - "
            "exposing actions, status, and data models."
        ),
        "value": (
            "Programmatic access so the platform "
            "integrates with other systems."
        ),
        "status": "built",
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
    executors = 0
    siems = 0
    for c in COMPONENT_CATALOG:
        by_status[c["status"]] = (
            by_status.get(c["status"], 0) + 1
        )
        if c["status"] == "executor":
            executors += 1
        if c["status"] == "siem":
            siems += 1
    return {
        "total_components": len(COMPONENT_CATALOG),
        "cloud_executors": executors,
        "siem_integrations": siems,
        "by_status": by_status,
    }