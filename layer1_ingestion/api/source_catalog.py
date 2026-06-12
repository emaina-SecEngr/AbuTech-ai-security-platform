"""
Layer 1 — Data Ingestion
Source Catalog

The single source of truth describing every data
source the platform ingests: its category, ingestion
method, a short description, and a display icon.

This catalog drives both:
    - The /api/sources API endpoint
    - The live ingestion dashboard

INGESTION METHODS:
    push   - the source sends events to us (webhook)
    pull   - we fetch on a schedule (API poll)
    stream - continuous flow (syslog / Kafka)

Each entry's "source" key matches the canonical
source name used by the SourceDetector and
IngestionRouter, so the catalog stays in lockstep
with what the pipeline can actually route.
"""

# Ingestion method metadata (label + color for UI)
INGESTION_METHODS = {
    "push": {
        "label": "Push",
        "color": "#378ADD",
        "description": (
            "The source sends events to our webhook "
            "endpoint the moment they occur."
        ),
    },
    "pull": {
        "label": "Pull",
        "color": "#1D9E75",
        "description": (
            "We poll the source's API on a schedule "
            "and fetch new events since the last check."
        ),
    },
    "stream": {
        "label": "Stream",
        "color": "#D85A30",
        "description": (
            "Events flow continuously to our syslog "
            "or Kafka listener in real time."
        ),
    },
}


# The full source catalog.
# "source" must match SourceDetector KNOWN_SOURCES.
SOURCE_CATALOG = [
    # --- Data access ---
    {
        "source": "s3", "name": "Amazon S3",
        "category": "Data access", "method": "pull",
        "icon": "ti-bucket",
        "description": (
            "S3 object-access events from CloudTrail, "
            "fetched via the AWS API."
        ),
    },
    {
        "source": "rds", "name": "Amazon RDS",
        "category": "Data access", "method": "pull",
        "icon": "ti-database",
        "description": (
            "Database access and admin events from "
            "CloudTrail."
        ),
    },
    {
        "source": "snowflake", "name": "Snowflake",
        "category": "Data access", "method": "pull",
        "icon": "ti-snowflake",
        "description": (
            "Query and access history pulled from the "
            "Snowflake API."
        ),
    },
    {
        "source": "sharepoint", "name": "SharePoint",
        "category": "Data access", "method": "pull",
        "icon": "ti-folder-share",
        "description": (
            "File and site access from the M365 audit "
            "log."
        ),
    },
    {
        "source": "oracle", "name": "Oracle DB",
        "category": "Data access", "method": "pull",
        "icon": "ti-database",
        "description": (
            "Database audit records collected on a "
            "schedule."
        ),
    },

    # --- Secrets and vault ---
    {
        "source": "aws_secrets",
        "name": "AWS Secrets Manager",
        "category": "Secrets and vault", "method": "pull",
        "icon": "ti-key",
        "description": (
            "Secret retrieval and rotation events via "
            "the AWS API."
        ),
    },
    {
        "source": "azure_keyvault",
        "name": "Azure Key Vault",
        "category": "Secrets and vault", "method": "pull",
        "icon": "ti-key",
        "description": (
            "Secret and certificate access from Azure "
            "diagnostics."
        ),
    },

    # --- Identity and access ---
    {
        "source": "okta", "name": "Okta",
        "category": "Identity and access",
        "method": "push", "icon": "ti-user-shield",
        "description": (
            "System-log events pushed via Okta event "
            "hooks."
        ),
    },
    {
        "source": "entraid",
        "name": "Microsoft Entra ID",
        "category": "Identity and access",
        "method": "pull", "icon": "ti-user-shield",
        "description": (
            "Sign-in and audit logs from the Microsoft "
            "Graph API."
        ),
    },
    {
        "source": "cyberark", "name": "CyberArk",
        "category": "Identity and access",
        "method": "pull", "icon": "ti-lock-access",
        "description": (
            "Privileged session and vault activity "
            "events."
        ),
    },
    {
        "source": "sailpoint", "name": "SailPoint",
        "category": "Identity and access",
        "method": "pull", "icon": "ti-id-badge-2",
        "description": (
            "Identity lifecycle and access "
            "certification events."
        ),
    },
    {
        "source": "sentinel",
        "name": "Microsoft Sentinel",
        "category": "Identity and access",
        "method": "pull", "icon": "ti-shield-check",
        "description": (
            "SIEM incidents and analytics-rule alerts."
        ),
    },

    # --- Network ---
    {
        "source": "firewall",
        "name": "Firewall (Palo Alto)",
        "category": "Network", "method": "stream",
        "icon": "ti-wall",
        "description": (
            "Traffic and threat logs streamed via "
            "syslog."
        ),
    },
    {
        "source": "waf", "name": "WAF",
        "category": "Network", "method": "stream",
        "icon": "ti-shield-half",
        "description": (
            "Web request blocks and rule matches."
        ),
    },
    {
        "source": "api_gateway", "name": "API Gateway",
        "category": "Network", "method": "pull",
        "icon": "ti-api",
        "description": (
            "API access logs fetched on a schedule."
        ),
    },
    {
        "source": "network_flow", "name": "Network flow",
        "category": "Network", "method": "stream",
        "icon": "ti-affiliate",
        "description": (
            "NetFlow/IPFIX records streamed "
            "continuously."
        ),
    },

    # --- Email ---
    {
        "source": "email_gateway",
        "name": "Email gateway",
        "category": "Email", "method": "push",
        "icon": "ti-mail-forward",
        "description": (
            "Phishing and threat alerts pushed on "
            "detection."
        ),
    },
    {
        "source": "email", "name": "Email (mailbox)",
        "category": "Email", "method": "pull",
        "icon": "ti-mail",
        "description": (
            "Mailbox audit events from the M365 audit "
            "log."
        ),
    },

    # --- Cloud posture ---
    {
        "source": "cspm", "name": "CSPM",
        "category": "Cloud posture", "method": "pull",
        "icon": "ti-cloud-cog",
        "description": (
            "Misconfiguration findings from Wiz / "
            "Prisma Cloud."
        ),
    },
    {
        "source": "iac", "name": "IaC scan",
        "category": "Cloud posture", "method": "push",
        "icon": "ti-file-code",
        "description": (
            "Checkov / tfsec results pushed from "
            "CI/CD."
        ),
    },
    {
        "source": "gcp", "name": "GCP audit",
        "category": "Cloud posture", "method": "pull",
        "icon": "ti-brand-google",
        "description": (
            "GCP cloud audit logs pulled from the API."
        ),
    },

    # --- Endpoint and runtime ---
    {
        "source": "crowdstrike", "name": "CrowdStrike",
        "category": "Endpoint and runtime",
        "method": "push", "icon": "ti-device-desktop",
        "description": (
            "Falcon EDR detections pushed via webhook."
        ),
    },
    {
        "source": "kubernetes", "name": "Kubernetes",
        "category": "Endpoint and runtime",
        "method": "stream", "icon": "ti-box",
        "description": (
            "Audit logs and Falco runtime alerts "
            "streamed continuously."
        ),
    },
    {
        "source": "cwpp", "name": "CWPP",
        "category": "Endpoint and runtime",
        "method": "pull", "icon": "ti-box-multiple",
        "description": (
            "Prisma / Aqua / Falcon workload runtime "
            "findings."
        ),
    },

    # --- Azure-native and DLP ---
    {
        "source": "defender_cloud",
        "name": "Defender for Cloud",
        "category": "Azure-native and DLP",
        "method": "pull", "icon": "ti-cloud-lock",
        "description": (
            "Azure CNAPP alerts across all Defender "
            "plans."
        ),
    },
    {
        "source": "purview_dlp", "name": "Purview DLP",
        "category": "Azure-native and DLP",
        "method": "pull", "icon": "ti-file-shield",
        "description": (
            "Data-loss events across M365 and "
            "endpoint DLP."
        ),
    },

    # --- AWS-native ---
    {
        "source": "guardduty", "name": "GuardDuty",
        "category": "AWS-native", "method": "push",
        "icon": "ti-radar",
        "description": (
            "ML threat findings pushed via EventBridge."
        ),
    },
    {
        "source": "security_hub", "name": "Security Hub",
        "category": "AWS-native", "method": "pull",
        "icon": "ti-shield-search",
        "description": (
            "Aggregated ASFF findings: Inspector, "
            "Macie, Config, Access Analyzer."
        ),
    },
]


def get_catalog() -> list:
    """Return the full source catalog."""
    return SOURCE_CATALOG


def get_categories() -> list:
    """Return catalog grouped by category."""
    categories = {}
    for entry in SOURCE_CATALOG:
        cat = entry["category"]
        categories.setdefault(cat, []).append(entry)
    return [
        {"category": cat, "sources": sources}
        for cat, sources in categories.items()
    ]


def get_source(source_name: str) -> dict:
    """Return a single source entry by name."""
    for entry in SOURCE_CATALOG:
        if entry["source"] == source_name:
            return entry
    return {}


def catalog_stats() -> dict:
    """Return summary counts for the catalog."""
    by_method = {}
    by_category = {}
    for entry in SOURCE_CATALOG:
        by_method[entry["method"]] = (
            by_method.get(entry["method"], 0) + 1
        )
        by_category[entry["category"]] = (
            by_category.get(entry["category"], 0) + 1
        )
    return {
        "total_sources": len(SOURCE_CATALOG),
        "by_method": by_method,
        "by_category": by_category,
    }