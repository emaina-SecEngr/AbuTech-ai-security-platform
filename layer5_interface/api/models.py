"""
Layer 5 - Interface
API Request and Response Models

PURPOSE:
    Defines the data shapes for all API endpoints.
    Uses Pydantic for validation and documentation.

WHY PYDANTIC:
    FastAPI is built on Pydantic.
    Every field is validated automatically.
    Bad data rejected before processing.
    API documentation generated automatically.
    BofA integration team gets exact schemas.

REQUEST MODELS:
    What the API accepts as input.
    S3 events, IAM events, investigation requests.

RESPONSE MODELS:
    What the API returns as output.
    Risk assessments, investigation reports,
    platform health status.

SR 11-7 COMPLIANCE:
    Every response includes model_version.
    Every response includes scored_at timestamp.
    Complete audit trail per decision.
"""

from datetime import datetime
from datetime import timezone
from typing import Optional
from pydantic import BaseModel
from pydantic import Field


# ============================================================
# REQUEST MODELS
# ============================================================

class S3IngestRequest(BaseModel):
    """
    Raw AWS S3 CloudTrail event for ingestion.

    Accepts the exact format that AWS CloudTrail
    produces. No transformation needed before sending.

    Example:
    {
        "eventTime": "2024-03-29T09:19:00Z",
        "eventSource": "s3.amazonaws.com",
        "eventName": "GetObject",
        "userIdentity": {
            "type": "IAMUser",
            "userName": "svc_backup"
        },
        "requestParameters": {
            "bucketName": "prod-customer-data",
            "key": "customers/pii/2024-q1.csv"
        },
        "additionalEventData": {
            "bytesTransferredOut": 52428800
        },
        "sourceIPAddress": "185.220.101.45"
    }
    """
    raw_event: dict = Field(
        description="Raw AWS CloudTrail S3 event"
    )
    enrich: bool = Field(
        default=True,
        description=(
            "Whether to enrich with threat intel. "
            "Set False for high-volume batch ingestion."
        )
    )
    investigate: bool = Field(
        default=False,
        description=(
            "Whether to trigger Layer 4 investigation. "
            "Only for high-priority events."
        )
    )


class IAMIngestRequest(BaseModel):
    """
    Raw IAM event for ingestion.
    Accepts events from any IAM source:
    Okta, Entra ID, CyberArk, SailPoint.
    """
    raw_event: dict = Field(
        description="Raw IAM event from any vendor"
    )
    source_system: str = Field(
        description=(
            "Source vendor: okta, entra_id, "
            "cyberark, sailpoint, aws_secrets, "
            "azure_keyvault"
        )
    )
    enrich: bool = Field(default=True)
    investigate: bool = Field(default=False)


class RDSIngestRequest(BaseModel):
    """Raw RDS Database Activity Stream event"""
    raw_event: dict = Field(
        description="Raw RDS activity stream event"
    )
    enrich: bool = Field(default=True)
    investigate: bool = Field(default=False)


class SnowflakeIngestRequest(BaseModel):
    """Raw Snowflake Query History event"""
    raw_event: dict = Field(
        description="Raw Snowflake query history event"
    )
    enrich: bool = Field(default=True)
    investigate: bool = Field(default=False)


class InvestigationRequest(BaseModel):
    """
    Request to investigate a specific event.
    Triggers Layer 4 LLM agents.
    """
    event_id: str = Field(
        description="Event ID to investigate"
    )
    priority: str = Field(
        default="medium",
        description="Priority: low, medium, high, critical"
    )
    context: Optional[dict] = Field(
        default=None,
        description="Additional context for investigation"
    )


class GraphQueryRequest(BaseModel):
    """Request to query the knowledge graph"""
    node_id: Optional[str] = Field(
        default=None,
        description="Specific node to query"
    )
    max_hops: int = Field(
        default=2,
        description="Graph traversal depth"
    )
    include_threats: bool = Field(
        default=True,
        description="Include threat intelligence nodes"
    )


# ============================================================
# RESPONSE MODELS
# ============================================================

class RiskAssessment(BaseModel):
    """
    Risk assessment for a single event.
    Returned by all ingest endpoints.

    YOUR Q3 ANSWER — THREE KEY FIELDS:
    baseline_comparison: "Normal is 5MB/day"
    permissions_summary: "Full Administrator access"
    ip_reputation: "Known Tor exit node, score 97"
    """
    event_id: str
    risk_score: float = Field(
        ge=0.0, le=1.0,
        description="Risk score 0.0-1.0"
    )
    risk_label: str = Field(
        description="CRITICAL, HIGH, MEDIUM, LOW"
    )
    risk_reasons: list = Field(
        description="Human-readable risk reasons"
    )

    # YOUR Q3 — THREE IMMEDIATE CONTEXT FIELDS
    baseline_comparison: Optional[str] = Field(
        default=None,
        description=(
            "Comparison to normal baseline. "
            "Example: Normal 5MB/day vs 500MB now."
        )
    )
    permissions_summary: Optional[str] = Field(
        default=None,
        description=(
            "IAM permissions of the accessor. "
            "Example: Full Administrator access."
        )
    )
    ip_reputation: Optional[str] = Field(
        default=None,
        description=(
            "IP reputation from threat feeds. "
            "Example: Known Tor exit node score 97."
        )
    )

    # Sensitivity finding
    sensitivity_label: Optional[str] = Field(
        default=None,
        description="PII, PHI, PCI, PFI, NONE"
    )
    data_types_found: list = Field(
        default_factory=list,
        description="Types of sensitive data detected"
    )

    # Source information
    source_system: str
    accessor_identity: str
    event_time: str

    # SR 11-7 compliance fields
    model_version: str = "1.0.0"
    scored_at: str = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )
    models_used: list = Field(
        default_factory=list,
        description="ML models that scored this event"
    )


class InvestigationReport(BaseModel):
    """
    Full investigation report from Layer 4 agents.
    Generated by LLM investigation pipeline.
    """
    event_id: str
    investigation_id: str
    risk_score: float
    risk_label: str

    # Attack narrative
    summary: str = Field(
        description="One paragraph attack summary"
    )
    attack_type: str = Field(
        description="Classification of attack type"
    )
    mitre_techniques: list = Field(
        default_factory=list,
        description="Relevant ATT&CK technique IDs"
    )

    # Timeline
    attack_timeline: list = Field(
        default_factory=list,
        description="Chronological attack events"
    )

    # Response
    immediate_actions: list = Field(
        default_factory=list,
        description="Actions to take right now"
    )
    containment_steps: list = Field(
        default_factory=list,
        description="Steps to contain the threat"
    )

    # Context
    affected_systems: list = Field(
        default_factory=list
    )
    affected_data: list = Field(
        default_factory=list
    )
    threat_actor_attribution: Optional[str] = None

    # Metadata
    investigated_at: str = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )
    investigation_duration_ms: Optional[int] = None


class GraphSummary(BaseModel):
    """Knowledge graph summary"""
    total_nodes: int
    total_edges: int
    high_risk_nodes: int
    threat_nodes: int
    node_type_distribution: dict
    top_risk_entities: list = Field(
        default_factory=list,
        description="Top 10 highest risk entities"
    )
    recent_threats: list = Field(
        default_factory=list,
        description="Recently detected threats"
    )
    last_updated: str


class ModelHealthStatus(BaseModel):
    """
    Status of all ML models.
    Used by health endpoint and dashboard.
    SR 11-7 ongoing monitoring display.
    """
    model_name: str
    is_trained: bool
    last_trained: Optional[str] = None
    performance_metric: Optional[float] = None
    threshold: Optional[float] = None
    status: str = Field(
        description="OK, WARNING, CRITICAL, UNKNOWN"
    )


class PlatformHealth(BaseModel):
    """
    Overall platform health status.
    Used by Docker health check and dashboard.
    """
    status: str = Field(
        description="healthy, degraded, unhealthy"
    )
    version: str = "1.0.0"
    uptime_seconds: Optional[float] = None

    # Component health
    layer1_normalizers: str = "unknown"
    layer2_models: str = "unknown"
    layer3_graph: str = "unknown"
    layer4_agents: str = "unknown"

    # Model statuses
    model_health: list = Field(
        default_factory=list
    )

    # Test coverage
    tests_passing: Optional[int] = None
    test_coverage_pct: Optional[float] = None

    # Timestamp
    checked_at: str = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )


class EventFeedItem(BaseModel):
    """
    Single item in the real-time event feed.
    Displayed in Streamlit dashboard.
    """
    event_id: str
    event_time: str
    source_system: str
    accessor_identity: str
    risk_score: float
    risk_label: str
    risk_color: str = Field(
        description="red, orange, yellow, green"
    )
    summary: str = Field(
        description="One-line event summary"
    )
    requires_investigation: bool = False


class PlatformStats(BaseModel):
    """
    Platform statistics for dashboard.
    """
    total_events_processed: int = 0
    events_last_hour: int = 0
    critical_alerts: int = 0
    high_alerts: int = 0
    medium_alerts: int = 0
    low_alerts: int = 0
    models_active: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    pii_findings_today: int = 0
    threats_detected_today: int = 0