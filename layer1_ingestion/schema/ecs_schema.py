"""
Layer 1 — Data Ingestion
Elastic Common Schema (ECS) Definition

This module defines the standard data structure that every
normalizer in the platform must output. Think of it as the
contract between Layer 1 and all layers above it.

Every security event — regardless of source — becomes an
ECSEvent object. This ensures Layer 2 ML models, Layer 3
knowledge graph, and Layer 4 LLM reasoning always receive
data in a consistent, predictable format.

Reference: https://www.elastic.co/guide/en/ecs/current/index.html
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Optional
from datetime import datetime


# ============================================================
# NESTED DATA STRUCTURES
# Each represents a logical grouping of related fields
# in the ECS specification
# ============================================================

@dataclass
class ECSEvent:
    """
    Core event metadata.
    Every security event has these fundamental attributes.
    
    category: Type of event
               network, authentication, process, file, dns
    
    type: What happened
          connection, start, end, allowed, denied, info
    
    severity: Numeric severity 0-100
              Populated by Layer 2 ML models after ingestion
    
    risk_score: Overall risk 0-100
                Populated by Layer 2 ML models after ingestion
    """
    id: str
    category: str
    type: str
    dataset: str
    provider: str
    created: str
    severity: Optional[int] = None
    risk_score: Optional[float] = None
    outcome: Optional[str] = None


@dataclass
class ECSSource:
    """
    Source of network connection or event origin.
    Used for network events and authentication events.
    """
    ip: Optional[str] = None
    port: Optional[int] = None
    bytes: Optional[int] = None
    domain: Optional[str] = None
    geo_country: Optional[str] = None


@dataclass
class ECSDestination:
    """
    Destination of network connection.
    Used for network events.
    """
    ip: Optional[str] = None
    port: Optional[int] = None
    bytes: Optional[int] = None
    domain: Optional[str] = None
    geo_country: Optional[str] = None


@dataclass
class ECSHost:
    """
    The machine where the event occurred.
    
    hostname: Machine name e.g. WKSTN-JSMITH-01
    ip: Machine IP address
    os_platform: windows, linux, macos
    
    This field connects events across sources.
    The same hostname appears in EDR events,
    network flows, and authentication logs -
    allowing Layer 3 knowledge graph to link them.
    """
    hostname: Optional[str] = None
    ip: Optional[str] = None
    os_platform: Optional[str] = None
    os_version: Optional[str] = None
    domain: Optional[str] = None


@dataclass
class ECSUser:
    """
    The identity involved in the event.

    Connecting user identity to behavior is where
    your IAM expertise becomes directly valuable.
    The same user appears in Okta auth logs,
    CyberArk privileged sessions, and EDR process
    events - the knowledge graph links them all
    through this standardized user structure.

    name: Username without domain e.g. jsmith
    domain: Domain name e.g. CORP
    email: Full email e.g. jsmith@company.com
    """
    name: Optional[str] = None
    domain: Optional[str] = None
    email: Optional[str] = None
    full_name: Optional[str] = None


@dataclass
class ECSParentProcess:
    """
    The process that spawned the current process.

    Parent-child process relationships are one of
    the most valuable signals in endpoint security.

    MSBuild.exe spawning powershell.exe is suspicious.
    cmd.exe spawning powershell.exe is suspicious.
    explorer.exe spawning powershell.exe is normal.

    Your Layer 2 malware classifier uses this
    relationship as a high-value feature.
    """
    name: Optional[str] = None
    pid: Optional[int] = None
    executable: Optional[str] = None
    command_line: Optional[str] = None


@dataclass
class ECSProcess:
    """
    Process execution details.
    Critical for malware detection and
    endpoint threat hunting.

    name: Process filename e.g. powershell.exe
    executable: Full path e.g. C:\\Windows\\System32\\powershell.exe
    command_line: Full command with arguments
    args_count: Number of arguments passed
    parent: The process that spawned this one
    """
    name: Optional[str] = None
    pid: Optional[int] = None
    executable: Optional[str] = None
    command_line: Optional[str] = None
    args_count: Optional[int] = None
    integrity_level: Optional[str] = None
    parent: Optional[ECSParentProcess] = None


@dataclass
class ECSNetwork:
    """
    Network communication details.
    Used for network flow events and
    process network connections.

    Basic fields populated by all network normalizers.
    Flow statistics fields populated by network flow
    normalizers like Zeek and NetFlow.
    These statistics are what feed the Layer 2
    network intrusion detection model.
    """
    # Basic connection fields
    protocol: Optional[str] = None
    transport: Optional[str] = None
    direction: Optional[str] = None
    bytes: Optional[int] = None
    packets: Optional[int] = None

    # Flow statistics - populated by network flow normalizer
    # These map directly to CICIDS2017 features
    duration_ms: Optional[float] = None
    fwd_packets: Optional[int] = None
    bwd_packets: Optional[int] = None
    fwd_bytes: Optional[int] = None
    bwd_bytes: Optional[int] = None
    fwd_packet_len_max: Optional[float] = None
    fwd_packet_len_mean: Optional[float] = None
    bwd_packet_len_max: Optional[float] = None
    bwd_packet_len_mean: Optional[float] = None
    flow_bytes_per_sec: Optional[float] = None
    flow_packets_per_sec: Optional[float] = None
    iat_mean: Optional[float] = None
    iat_std: Optional[float] = None
    fwd_iat_mean: Optional[float] = None
    bwd_iat_mean: Optional[float] = None

    # TCP flag counts
    syn_flags: Optional[int] = None
    rst_flags: Optional[int] = None
    psh_flags: Optional[int] = None
    ack_flags: Optional[int] = None

    # Window size
    init_win_bytes_fwd: Optional[int] = None


@dataclass
class ECSThreat:
    """
    Threat intelligence enrichment fields.

    These fields are EMPTY when events first enter
    Layer 1. They get populated as the event flows
    through the platform:

    Layer 2 ML models populate:
        indicator.confidence
        indicator.type

    Layer 3 knowledge graph populates:
        tactic, technique, actor

    Layer 4 LLM reasoning populates:
        enriched analysis connecting all signals
    """
    indicator_type: Optional[str] = None
    indicator_confidence: Optional[int] = None
    tactic_name: Optional[str] = None
    tactic_id: Optional[str] = None
    technique_name: Optional[str] = None
    technique_id: Optional[str] = None
    actor: Optional[str] = None
    campaign: Optional[str] = None


@dataclass
class ECSNormalized:
    """
    The complete normalized security event.

    This is the output of every normalizer.
    This is the input to every ML model.
    This is what flows through your entire platform.

    Every field has a clear purpose.
    Every layer knows exactly where to find data.
    No layer needs to know where the event came from.

    Fields marked Optional are populated by layers
    above Layer 1 as the event is enriched.
    """

    # Timestamp - always UTC ISO 8601
    timestamp: str

    # Core event metadata
    event: ECSEvent

    # Source identity and machine
    user: ECSUser
    host: ECSHost

    # Network fields (populated for network events)
    source: Optional[ECSSource] = None
    destination: Optional[ECSDestination] = None
    network: Optional[ECSNetwork] = None

    # Process fields (populated for endpoint events)
    process: Optional[ECSProcess] = None

    # Threat enrichment (populated by layers 2-4)
    threat: Optional[ECSThreat] = None

    # Metadata about the normalization itself
    data_source: str = ""
    normalized: bool = True
    normalized_at: str = ""
    raw_event_hash: str = ""

    def to_dict(self) -> dict:
        """
        Convert to dictionary for storage and transmission.
        Used when writing to Elasticsearch or Delta Lake.
        """
        def _serialize(obj):
            if obj is None:
                return None
            if hasattr(obj, '__dataclass_fields__'):
                return {
                    k: _serialize(v)
                    for k, v in obj.__dict__.items()
                    if v is not None
                }
            return obj

        return {
            "@timestamp": self.timestamp,
            "event": _serialize(self.event),
            "user": _serialize(self.user),
            "host": _serialize(self.host),
            "source": _serialize(self.source),
            "destination": _serialize(self.destination),
            "network": _serialize(self.network),
            "process": _serialize(self.process),
            "threat": _serialize(self.threat),
            "labels": {
                "data_source": self.data_source,
                "normalized": self.normalized,
                "normalized_at": self.normalized_at
            }
        }

    @classmethod
    def get_required_fields(cls) -> list:
        """
        Returns list of fields that must be present
        in every normalized event regardless of source.
        Used by data validation tests.
        """
        return [
            "@timestamp",
            "event.id",
            "event.category",
            "event.type",
            "event.dataset",
            "event.provider",
            "labels.data_source",
            "labels.normalized"
        ]