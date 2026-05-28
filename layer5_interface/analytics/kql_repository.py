"""
Layer 5 — Interface
KQL Query Repository

A library of pre-built KQL threat hunting
queries organized by MITRE ATT&CK technique.

WHY THIS EXISTS:
    SOC analysts spend hours writing KQL.
    The same queries get rewritten repeatedly.
    This repository gives every analyst
    immediate access to battle-tested queries.

    IBM VALUE:
    IBM consultants arrive at client sites
    and immediately run proven hunts.
    No time wasted writing queries from scratch.
    Day one value delivery.

QUERY ORGANIZATION:
    By MITRE ATT&CK technique (primary)
    By threat actor (secondary)
    By data source (tertiary)

EACH QUERY CONTAINS:
    query_id:      Unique identifier
    title:         Human readable name
    description:   What it detects
    technique_id:  MITRE ATT&CK technique
    tactic:        MITRE ATT&CK tactic
    threat_actors: Known actors using this TTP
    data_sources:  Sentinel tables required
    kql:           The actual KQL query
    false_positives: Known FP scenarios
    severity:      Expected alert severity
    validated:     Last validation date

USAGE:
    repo = KQLRepository()

    # Get query by technique
    queries = repo.get_by_technique("T1621")

    # Search queries
    results = repo.search("MFA fatigue")

    # Get all financial sector queries
    queries = repo.get_financial_sector_queries()

    # Get query by ID
    query = repo.get_query("KQL-001")
"""

import logging
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)


# ============================================================
# QUERY DEFINITIONS
# Full library of financial sector KQL queries
# ============================================================

KQL_QUERIES = [

    # --------------------------------------------------------
    # CREDENTIAL ACCESS
    # --------------------------------------------------------
    {
        "query_id": "KQL-001",
        "title": "MFA Fatigue Attack Detection",
        "description": (
            "Detects MFA fatigue attacks where "
            "an attacker generates rapid MFA "
            "push notifications to overwhelm "
            "the user into approving access. "
            "Primary TTP of Scattered Spider."
        ),
        "technique_id": "T1621",
        "tactic": "Credential Access",
        "threat_actors": [
            "Scattered Spider",
            "UNC3944"
        ],
        "data_sources": ["SigninLogs"],
        "severity": "HIGH",
        "kql": """
SigninLogs
| where TimeGenerated > ago(1h)
| where ResultType == "500121"
| summarize
    MFADenials = count(),
    UniqueIPs = dcount(IPAddress),
    Locations = make_set(Location),
    FirstSeen = min(TimeGenerated),
    LastSeen = max(TimeGenerated)
    by UserPrincipalName,
       bin(TimeGenerated, 15m)
| where MFADenials > 5
| extend
    DurationMinutes = datetime_diff(
        'minute', LastSeen, FirstSeen
    ),
    IsMultiLocation = array_length(Locations) > 1
| project
    TimeGenerated,
    UserPrincipalName,
    MFADenials,
    UniqueIPs,
    Locations,
    DurationMinutes,
    IsMultiLocation
| sort by MFADenials desc
""",
        "false_positives": [
            "User with poor connectivity",
            "Legitimate MFA app issues"
        ],
        "validated": "2026-05-27",
        "abutech_model": "Identity Detector"
    },

    {
        "query_id": "KQL-002",
        "title": "Brute Force Login Detection",
        "description": (
            "Detects brute force attacks with "
            "multiple failed login attempts "
            "from a single IP address."
        ),
        "technique_id": "T1110",
        "tactic": "Credential Access",
        "threat_actors": ["Multiple"],
        "data_sources": ["SigninLogs"],
        "severity": "MEDIUM",
        "kql": """
SigninLogs
| where TimeGenerated > ago(1h)
| where ResultType != "0"
| summarize
    FailureCount = count(),
    DistinctUsers = dcount(UserPrincipalName),
    DistinctIPs = dcount(IPAddress),
    UserList = make_set(UserPrincipalName, 10)
    by IPAddress,
       bin(TimeGenerated, 5m)
| where FailureCount > 10
| extend
    AttackType = iff(
        DistinctUsers > 5,
        "Password Spray",
        "Brute Force"
    )
| project
    TimeGenerated,
    IPAddress,
    FailureCount,
    DistinctUsers,
    AttackType,
    UserList
| sort by FailureCount desc
""",
        "false_positives": [
            "Misconfigured service account",
            "User testing wrong password"
        ],
        "validated": "2026-05-27",
        "abutech_model": "Identity Detector"
    },

    {
        "query_id": "KQL-003",
        "title": "Credential Stuffing via API",
        "description": (
            "Detects credential stuffing attacks "
            "against API authentication endpoints. "
            "High 401 rate from single IP indicates "
            "automated credential testing."
        ),
        "technique_id": "T1110.004",
        "tactic": "Credential Access",
        "threat_actors": ["Multiple"],
        "data_sources": ["AzureDiagnostics"],
        "severity": "HIGH",
        "kql": """
AzureDiagnostics
| where TimeGenerated > ago(1h)
| where Category == "FrontdoorAccessLog"
| where httpStatusCode_d == 401
| summarize
    FailedRequests = count(),
    DistinctEndpoints = dcount(requestUri_s),
    DistinctUsers = dcount(userPrincipalName_s)
    by clientIp_s,
       bin(TimeGenerated, 5m)
| where FailedRequests > 100
| extend
    CredStuffingScore = iff(
        DistinctUsers > 50,
        "HIGH",
        "MEDIUM"
    )
| project
    TimeGenerated,
    clientIp_s,
    FailedRequests,
    DistinctEndpoints,
    DistinctUsers,
    CredStuffingScore
| sort by FailedRequests desc
""",
        "false_positives": [
            "Load testing from known IP",
            "Misconfigured integration"
        ],
        "validated": "2026-05-27",
        "abutech_model": "Identity Detector"
    },

    # --------------------------------------------------------
    # INITIAL ACCESS
    # --------------------------------------------------------
    {
        "query_id": "KQL-004",
        "title": "Impossible Travel Detection",
        "description": (
            "Detects when a user authenticates "
            "from two geographically distant "
            "locations in an impossibly short "
            "time — indicating account compromise."
        ),
        "technique_id": "T1078",
        "tactic": "Initial Access",
        "threat_actors": ["Multiple"],
        "data_sources": ["SigninLogs"],
        "severity": "HIGH",
        "kql": """
SigninLogs
| where TimeGenerated > ago(24h)
| where ResultType == "0"
| project
    TimeGenerated,
    UserPrincipalName,
    IPAddress,
    Location,
    AppDisplayName
| order by UserPrincipalName, TimeGenerated asc
| serialize
| extend
    PrevLocation = prev(Location, 1),
    PrevTime = prev(TimeGenerated, 1),
    PrevUser = prev(UserPrincipalName, 1)
| where UserPrincipalName == PrevUser
| where Location != PrevLocation
| extend
    TimeDiffMinutes = datetime_diff(
        'minute', TimeGenerated, PrevTime
    )
| where TimeDiffMinutes < 60
| where isnotempty(PrevLocation)
| project
    TimeGenerated,
    UserPrincipalName,
    PrevLocation,
    CurrentLocation = Location,
    TimeDiffMinutes,
    IPAddress,
    AppDisplayName
| sort by TimeDiffMinutes asc
""",
        "false_positives": [
            "VPN usage switching locations",
            "Proxy or anonymizer services"
        ],
        "validated": "2026-05-27",
        "abutech_model": "Identity Detector"
    },

    {
        "query_id": "KQL-005",
        "title": "New Admin Account Created",
        "description": (
            "Detects creation of new privileged "
            "accounts or assignment of admin roles. "
            "Common persistence technique after "
            "initial compromise."
        ),
        "technique_id": "T1136",
        "tactic": "Persistence",
        "threat_actors": ["Multiple"],
        "data_sources": ["AuditLogs"],
        "severity": "HIGH",
        "kql": """
AuditLogs
| where TimeGenerated > ago(24h)
| where OperationName in (
    "Add member to role",
    "Add user",
    "Create user"
)
| where Result == "success"
| extend
    TargetUser = tostring(
        TargetResources[0].userPrincipalName
    ),
    RoleAssigned = tostring(
        TargetResources[0].displayName
    ),
    InitiatedByUser = tostring(
        InitiatedBy.user.userPrincipalName
    )
| where RoleAssigned has_any (
    "Global Administrator",
    "Security Administrator",
    "Privileged Role Administrator",
    "Exchange Administrator"
)
| project
    TimeGenerated,
    OperationName,
    TargetUser,
    RoleAssigned,
    InitiatedByUser,
    Result
| sort by TimeGenerated desc
""",
        "false_positives": [
            "Planned admin account provisioning",
            "IT helpdesk role assignments"
        ],
        "validated": "2026-05-27",
        "abutech_model": "Identity Detector"
    },

    # --------------------------------------------------------
    # EXFILTRATION
    # --------------------------------------------------------
    {
        "query_id": "KQL-006",
        "title": "Large Data Transfer After Hours",
        "description": (
            "Detects unusually large data transfers "
            "occurring outside business hours. "
            "Key indicator of data exfiltration "
            "by insider threat or compromised account."
        ),
        "technique_id": "T1530",
        "tactic": "Exfiltration",
        "threat_actors": [
            "Insider Threat",
            "Multiple"
        ],
        "data_sources": ["CommonSecurityLog"],
        "severity": "HIGH",
        "kql": """
CommonSecurityLog
| where TimeGenerated > ago(24h)
| where toint(DestinationPort) in (443, 80, 22)
| where toint(SentBytes) > 104857600
| extend
    HourOfDay = hourofday(TimeGenerated),
    DayOfWeek = dayofweek(TimeGenerated),
    SentMB = round(
        todouble(SentBytes) / 1048576, 2
    )
| where HourOfDay < 6 or HourOfDay > 22
| summarize
    TotalSentMB = sum(SentMB),
    TransferCount = count(),
    DestinationIPs = make_set(DestinationIP, 5)
    by SourceIP, SourceUserName,
       bin(TimeGenerated, 1h)
| where TotalSentMB > 100
| project
    TimeGenerated,
    SourceIP,
    SourceUserName,
    TotalSentMB,
    TransferCount,
    DestinationIPs,
    HourOfDay
| sort by TotalSentMB desc
""",
        "false_positives": [
            "Scheduled backup jobs",
            "Approved data migration"
        ],
        "validated": "2026-05-27",
        "abutech_model": "Isolation Forest + LSTM"
    },

    {
        "query_id": "KQL-007",
        "title": "S3 Bucket Mass Download",
        "description": (
            "Detects mass download from S3 buckets "
            "which may indicate data exfiltration "
            "via compromised service account or "
            "misconfigured bucket."
        ),
        "technique_id": "T1530",
        "tactic": "Exfiltration",
        "threat_actors": ["Multiple"],
        "data_sources": ["AWSCloudTrail"],
        "severity": "CRITICAL",
        "kql": """
AWSCloudTrail
| where TimeGenerated > ago(24h)
| where EventName in (
    "GetObject",
    "ListObjects",
    "ListBucket"
)
| where isnotempty(RequestParameters)
| extend
    BucketName = tostring(
        parse_json(RequestParameters).bucketName
    ),
    ObjectKey = tostring(
        parse_json(RequestParameters).key
    )
| summarize
    DownloadCount = count(),
    UniqueBuckets = dcount(BucketName),
    UniqueObjects = dcount(ObjectKey),
    SourceIPs = make_set(SourceIPAddress, 5)
    by UserIdentityArn,
       bin(TimeGenerated, 1h)
| where DownloadCount > 1000
| extend
    HourOfDay = hourofday(TimeGenerated),
    IsAfterHours = HourOfDay < 6 or
                   HourOfDay > 22
| project
    TimeGenerated,
    UserIdentityArn,
    DownloadCount,
    UniqueBuckets,
    UniqueObjects,
    SourceIPs,
    IsAfterHours
| sort by DownloadCount desc
""",
        "false_positives": [
            "Scheduled ETL jobs",
            "Approved data pipeline"
        ],
        "validated": "2026-05-27",
        "abutech_model": "Isolation Forest"
    },

    # --------------------------------------------------------
    # COMMAND AND CONTROL
    # --------------------------------------------------------
    {
        "query_id": "KQL-008",
        "title": "DNS Tunneling Detection",
        "description": (
            "Detects DNS tunneling used for "
            "data exfiltration or C2 communication. "
            "Characterized by high query volume "
            "with many unique subdomains."
        ),
        "technique_id": "T1071.004",
        "tactic": "Command and Control",
        "threat_actors": ["Multiple"],
        "data_sources": ["DnsEvents"],
        "severity": "HIGH",
        "kql": """
DnsEvents
| where TimeGenerated > ago(1h)
| where Name has "."
| extend
    TopLevelDomain = tostring(
        split(Name, ".")[-1]
    ),
    SecondLevelDomain = strcat(
        tostring(split(Name, ".")[-2]),
        ".",
        tostring(split(Name, ".")[-1])
    )
| summarize
    QueryCount = count(),
    UniqueSubdomains = dcount(Name),
    AvgQueryLength = avg(strlen(Name)),
    MaxQueryLength = max(strlen(Name))
    by Computer, ClientIP,
       SecondLevelDomain,
       bin(TimeGenerated, 5m)
| where QueryCount > 100
| where UniqueSubdomains > 50
| where AvgQueryLength > 40
| extend
    TunnelingScore = iff(
        UniqueSubdomains > 200 and
        AvgQueryLength > 60,
        "HIGH",
        "MEDIUM"
    )
| project
    TimeGenerated,
    Computer,
    ClientIP,
    SecondLevelDomain,
    QueryCount,
    UniqueSubdomains,
    AvgQueryLength,
    TunnelingScore
| sort by QueryCount desc
""",
        "false_positives": [
            "CDN traffic with unique URLs",
            "Analytics platforms"
        ],
        "validated": "2026-05-27",
        "abutech_model": "DNS Classifier + LSTM"
    },

    {
        "query_id": "KQL-009",
        "title": "C2 Beaconing Pattern Detection",
        "description": (
            "Detects regular periodic outbound "
            "connections indicative of C2 "
            "beaconing behavior. Malware checks "
            "in with C2 server at regular intervals."
        ),
        "technique_id": "T1071",
        "tactic": "Command and Control",
        "threat_actors": ["Multiple"],
        "data_sources": ["CommonSecurityLog"],
        "severity": "HIGH",
        "kql": """
CommonSecurityLog
| where TimeGenerated > ago(6h)
| where DeviceAction == "allow"
| where toint(SentBytes) < 10240
| summarize
    ConnectionCount = count(),
    AvgIntervalSeconds = stdev(
        TimeGenerated
    ),
    DestinationPorts = make_set(
        DestinationPort, 5
    )
    by SourceIP, DestinationIP,
       bin(TimeGenerated, 1h)
| where ConnectionCount > 20
| extend
    BeaconingScore = iff(
        ConnectionCount > 50,
        "HIGH CONFIDENCE",
        "MEDIUM CONFIDENCE"
    )
| project
    TimeGenerated,
    SourceIP,
    DestinationIP,
    ConnectionCount,
    DestinationPorts,
    BeaconingScore
| sort by ConnectionCount desc
""",
        "false_positives": [
            "Monitoring agents",
            "Heartbeat services"
        ],
        "validated": "2026-05-27",
        "abutech_model": "LSTM"
    },

    # --------------------------------------------------------
    # DEFENSE EVASION
    # --------------------------------------------------------
    {
        "query_id": "KQL-010",
        "title": "CloudTrail Logging Disabled",
        "description": (
            "Detects when CloudTrail logging "
            "is stopped or deleted. Critical "
            "defense evasion technique to "
            "blind the defender before attack."
        ),
        "technique_id": "T1562.008",
        "tactic": "Defense Evasion",
        "threat_actors": ["Multiple"],
        "data_sources": ["AWSCloudTrail"],
        "severity": "CRITICAL",
        "kql": """
AWSCloudTrail
| where TimeGenerated > ago(24h)
| where EventName in (
    "StopLogging",
    "DeleteTrail",
    "UpdateTrail",
    "DeleteFlowLogs",
    "DeleteDetector",
    "StopConfigurationRecorder"
)
| project
    TimeGenerated,
    EventName,
    UserIdentityArn,
    SourceIPAddress,
    UserAgent,
    AWSRegion,
    RequestParameters
| sort by TimeGenerated desc
""",
        "false_positives": [
            "Planned infrastructure decommission",
            "Approved logging configuration change"
        ],
        "validated": "2026-05-27",
        "abutech_model": "Isolation Forest"
    },

    # --------------------------------------------------------
    # PRIVILEGE ESCALATION
    # --------------------------------------------------------
    {
        "query_id": "KQL-011",
        "title": "Privilege Escalation via IAM",
        "description": (
            "Detects privilege escalation attempts "
            "via IAM policy manipulation. Attacker "
            "modifies IAM policies to grant "
            "themselves elevated permissions."
        ),
        "technique_id": "T1078.004",
        "tactic": "Privilege Escalation",
        "threat_actors": ["Multiple"],
        "data_sources": ["AWSCloudTrail"],
        "severity": "CRITICAL",
        "kql": """
AWSCloudTrail
| where TimeGenerated > ago(24h)
| where EventName in (
    "AttachUserPolicy",
    "AttachRolePolicy",
    "PutUserPolicy",
    "PutRolePolicy",
    "CreatePolicyVersion",
    "SetDefaultPolicyVersion",
    "PassRole",
    "AssumeRole"
)
| extend
    PolicyArn = tostring(
        parse_json(RequestParameters).policyArn
    ),
    RoleName = tostring(
        parse_json(RequestParameters).roleName
    )
| where PolicyArn contains
    "AdministratorAccess"
    or PolicyArn contains "FullAccess"
| project
    TimeGenerated,
    EventName,
    UserIdentityArn,
    SourceIPAddress,
    PolicyArn,
    RoleName,
    AWSRegion
| sort by TimeGenerated desc
""",
        "false_positives": [
            "Approved IAM policy changes",
            "Infrastructure automation"
        ],
        "validated": "2026-05-27",
        "abutech_model": "CIEM Enricher + GNN"
    },

    # --------------------------------------------------------
    # LATERAL MOVEMENT
    # --------------------------------------------------------
    {
        "query_id": "KQL-012",
        "title": "Service Account Lateral Movement",
        "description": (
            "Detects service accounts accessing "
            "systems outside their normal scope. "
            "Indicates compromised service account "
            "used for lateral movement."
        ),
        "technique_id": "T1021",
        "tactic": "Lateral Movement",
        "threat_actors": ["Multiple"],
        "data_sources": [
            "SigninLogs", "AuditLogs"
        ],
        "severity": "HIGH",
        "kql": """
SigninLogs
| where TimeGenerated > ago(24h)
| where UserPrincipalName startswith "svc"
    or UserPrincipalName startswith "app"
    or UserPrincipalName startswith "api"
| where ResultType == "0"
| where IPAddress !startswith "10."
| where IPAddress !startswith "172."
| where IPAddress !startswith "192.168."
| summarize
    AppCount = dcount(AppDisplayName),
    LocationCount = dcount(Location),
    IPCount = dcount(IPAddress),
    AppList = make_set(AppDisplayName, 10),
    Locations = make_set(Location, 5)
    by UserPrincipalName,
       bin(TimeGenerated, 1h)
| where AppCount > 5
    or LocationCount > 2
    or IPCount > 3
| project
    TimeGenerated,
    UserPrincipalName,
    AppCount,
    LocationCount,
    IPCount,
    AppList,
    Locations
| sort by AppCount desc
""",
        "false_positives": [
            "Integration service accounts",
            "Monitoring service accounts"
        ],
        "validated": "2026-05-27",
        "abutech_model": "Identity Detector + GNN"
    },

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------
    {
        "query_id": "KQL-013",
        "title": "PowerShell Launched from Office",
        "description": (
            "Detects PowerShell execution spawned "
            "from Microsoft Office applications. "
            "Classic malware delivery via "
            "malicious macro in document."
        ),
        "technique_id": "T1059.001",
        "tactic": "Execution",
        "threat_actors": [
            "FIN7", "Lazarus Group", "Multiple"
        ],
        "data_sources": ["DeviceProcessEvents"],
        "severity": "CRITICAL",
        "kql": """
DeviceProcessEvents
| where TimeGenerated > ago(24h)
| where InitiatingProcessFileName in~ (
    "winword.exe",
    "excel.exe",
    "outlook.exe",
    "powerpnt.exe",
    "onenote.exe"
)
| where FileName =~ "powershell.exe"
    or FileName =~ "cmd.exe"
    or FileName =~ "wscript.exe"
    or FileName =~ "cscript.exe"
| extend
    CommandLength = strlen(ProcessCommandLine),
    IsEncoded = ProcessCommandLine
        contains "-enc"
        or ProcessCommandLine
        contains "-encodedcommand"
| project
    TimeGenerated,
    DeviceName,
    AccountName,
    InitiatingProcessFileName,
    FileName,
    ProcessCommandLine,
    CommandLength,
    IsEncoded
| sort by TimeGenerated desc
""",
        "false_positives": [
            "Legitimate macro-enabled documents",
            "IT automation via Office"
        ],
        "validated": "2026-05-27",
        "abutech_model": "Autoencoder"
    },

    # --------------------------------------------------------
    # COLLECTION
    # --------------------------------------------------------
    {
        "query_id": "KQL-014",
        "title": "Bulk Email Export Detection",
        "description": (
            "Detects bulk email export from "
            "Exchange or O365. Attacker exports "
            "mailbox contents for intelligence "
            "gathering or BEC preparation."
        ),
        "technique_id": "T1114",
        "tactic": "Collection",
        "threat_actors": [
            "Scattered Spider", "Multiple"
        ],
        "data_sources": ["OfficeActivity"],
        "severity": "HIGH",
        "kql": """
OfficeActivity
| where TimeGenerated > ago(24h)
| where Operation in (
    "New-MailboxExportRequest",
    "MailboxLogin",
    "FolderBind",
    "SendAs",
    "SendOnBehalf"
)
| summarize
    OperationCount = count(),
    UniqueMailboxes = dcount(UserId),
    Operations = make_set(Operation, 5)
    by ClientIP,
       bin(TimeGenerated, 1h)
| where OperationCount > 100
| project
    TimeGenerated,
    ClientIP,
    OperationCount,
    UniqueMailboxes,
    Operations
| sort by OperationCount desc
""",
        "false_positives": [
            "Approved mailbox migration",
            "Email archiving service"
        ],
        "validated": "2026-05-27",
        "abutech_model": "Isolation Forest"
    },

    # --------------------------------------------------------
    # IMPACT
    # --------------------------------------------------------
    {
        "query_id": "KQL-015",
        "title": "Mass Resource Deletion",
        "description": (
            "Detects mass deletion of cloud "
            "resources. Ransomware without "
            "encryption. Attacker destroys "
            "data for extortion."
        ),
        "technique_id": "T1485",
        "tactic": "Impact",
        "threat_actors": [
            "Ransomware Groups", "Multiple"
        ],
        "data_sources": ["AWSCloudTrail"],
        "severity": "CRITICAL",
        "kql": """
AWSCloudTrail
| where TimeGenerated > ago(1h)
| where EventName has "Delete"
    or EventName has "Terminate"
    or EventName has "Remove"
| where EventName in (
    "DeleteBucket",
    "DeleteObject",
    "DeleteDBInstance",
    "DeleteDBCluster",
    "TerminateInstances",
    "DeleteTable",
    "DeleteVolume",
    "DeleteSnapshot"
)
| summarize
    DeletionCount = count(),
    ResourceTypes = make_set(EventName, 10),
    SourceIPs = make_set(SourceIPAddress, 5)
    by UserIdentityArn,
       bin(TimeGenerated, 15m)
| where DeletionCount > 10
| extend
    Severity = iff(
        DeletionCount > 50,
        "CRITICAL",
        "HIGH"
    )
| project
    TimeGenerated,
    UserIdentityArn,
    DeletionCount,
    ResourceTypes,
    SourceIPs,
    Severity
| sort by DeletionCount desc
""",
        "false_positives": [
            "Approved infrastructure teardown",
            "Cost optimization cleanup"
        ],
        "validated": "2026-05-27",
        "abutech_model": "Isolation Forest"
    },
]


class KQLRepository:
    """
    Repository of KQL threat hunting queries.
    Organized by MITRE ATT&CK technique.

    Provides:
    - Query lookup by technique
    - Full text search
    - Financial sector query sets
    - Query validation
    """

    def __init__(self):
        self._queries = {
            q["query_id"]: q
            for q in KQL_QUERIES
        }
        logger.info(
            f"KQL Repository loaded: "
            f"{len(self._queries)} queries"
        )

    def get_query(
        self, query_id: str
    ) -> dict:
        """Get query by ID"""
        return self._queries.get(query_id, {})

    def get_by_technique(
        self, technique_id: str
    ) -> list:
        """
        Get all queries for a MITRE technique.

        Args:
            technique_id: e.g. "T1621"

        Returns:
            List of matching queries
        """
        return [
            q for q in self._queries.values()
            if q.get("technique_id", "")
            .startswith(technique_id)
        ]

    def get_by_tactic(
        self, tactic: str
    ) -> list:
        """Get all queries for a tactic"""
        tactic_lower = tactic.lower()
        return [
            q for q in self._queries.values()
            if tactic_lower in
            q.get("tactic", "").lower()
        ]

    def get_by_severity(
        self, severity: str
    ) -> list:
        """Get queries by severity level"""
        return [
            q for q in self._queries.values()
            if q.get("severity", "") == (
                severity.upper()
            )
        ]

    def get_by_data_source(
        self, data_source: str
    ) -> list:
        """Get queries requiring a data source"""
        ds_lower = data_source.lower()
        return [
            q for q in self._queries.values()
            if any(
                ds_lower in s.lower()
                for s in q.get("data_sources", [])
            )
        ]

    def get_by_threat_actor(
        self, actor: str
    ) -> list:
        """Get queries for a specific threat actor"""
        actor_lower = actor.lower()
        return [
            q for q in self._queries.values()
            if any(
                actor_lower in a.lower()
                for a in q.get("threat_actors", [])
            )
        ]

    def search(self, keyword: str) -> list:
        """
        Full text search across all queries.
        Searches title, description, technique.

        Args:
            keyword: Search term

        Returns:
            List of matching queries
        """
        keyword_lower = keyword.lower()
        results = []

        for q in self._queries.values():
            searchable = (
                q.get("title", "") + " " +
                q.get("description", "") + " " +
                q.get("technique_id", "") + " " +
                " ".join(
                    q.get("threat_actors", [])
                )
            ).lower()

            if keyword_lower in searchable:
                results.append(q)

        return results

    def get_financial_sector_queries(
        self
    ) -> list:
        """
        Return queries most relevant to
        financial services SOC teams.

        Prioritized for:
            Payment system protection
            Identity threat detection
            Data exfiltration prevention
            Compliance monitoring

        Returns:
            Prioritized list of queries
        """
        priority_ids = [
            "KQL-001",  # MFA Fatigue
            "KQL-004",  # Impossible Travel
            "KQL-006",  # Large Transfer
            "KQL-007",  # S3 Mass Download
            "KQL-010",  # CloudTrail Disabled
            "KQL-011",  # IAM Privilege Escalation
            "KQL-002",  # Brute Force
            "KQL-005",  # New Admin Account
            "KQL-012",  # Service Account Lateral
            "KQL-015",  # Mass Deletion
        ]

        return [
            self._queries[qid]
            for qid in priority_ids
            if qid in self._queries
        ]

    def get_all_queries(self) -> list:
        """Return all queries"""
        return list(self._queries.values())

    def get_statistics(self) -> dict:
        """Get repository statistics"""
        queries = list(self._queries.values())

        tactics = {}
        severities = {}
        data_sources = {}

        for q in queries:
            tactic = q.get("tactic", "Unknown")
            tactics[tactic] = (
                tactics.get(tactic, 0) + 1
            )

            severity = q.get("severity", "Unknown")
            severities[severity] = (
                severities.get(severity, 0) + 1
            )

            for ds in q.get("data_sources", []):
                data_sources[ds] = (
                    data_sources.get(ds, 0) + 1
                )

        return {
            "total_queries": len(queries),
            "by_tactic": tactics,
            "by_severity": severities,
            "by_data_source": data_sources,
            "last_updated": "2026-05-27"
        }