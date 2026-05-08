"""
Layer 1 — Data Ingestion
Data Security Schema

WHY THIS FILE EXISTS:
    Banks have 20+ data sources.
    S3, Oracle, DB2, Snowflake, SharePoint,
    Temenos core banking, ETL processes,
    service accounts, human users.

    Each produces different audit log formats.
    Your PII classifier needs ONE standard input.

    This is the THIRD schema in your platform:
    ECSNormalized  → endpoint/network events
    IamEvent       → identity events
    DataAccessEvent → data access events ← THIS FILE

    Together they cover the three pillars:
    Endpoint + Identity + Data

ACCESSOR TYPES IN BANKING:
    human           → banker, analyst, DBA
    service_account → svc_backup, svc_etl
    etl_process     → batch jobs, pipelines
    application     → Temenos, Salesforce, AML
    privileged      → DBA tools, admin consoles
    api_client      → REST API integrations
    replication     → database replication
    backup          → backup software agents

SENSITIVITY LABELS (regulatory coverage):
    PII  → GDPR, CCPA, PIPEDA
    PHI  → HIPAA
    PCI  → PCI-DSS
    PFI  → Financial data (GLBA, SOX)
    NONE → No sensitive data detected

ENVIRONMENTS:
    production → highest risk
    staging    → medium risk
    development → lower risk (should have masked data)
    test       → lowest risk (should have synthetic data)
"""

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Optional


# ============================================================
# SENSITIVITY LABELS
#
# WHY AN ENUM:
#   Prevents typos like "pii" vs "PII" vs "Pii"
#   Forces consistent labeling across platform
#   Easy to compare and filter in code
# ============================================================

class SensitivityLabel(Enum):
    """
    Data sensitivity classification labels.
    Maps to regulatory frameworks.
    """
    PII = "PII"
    # Personally Identifiable Information
    # Regulation: GDPR, CCPA, PIPEDA
    # Examples: name, email, SSN, DOB, address

    PHI = "PHI"
    # Protected Health Information
    # Regulation: HIPAA
    # Examples: medical records, diagnoses,
    #           prescriptions, insurance IDs

    PCI = "PCI"
    # Payment Card Industry data
    # Regulation: PCI-DSS
    # Examples: card numbers, CVV, expiry,
    #           cardholder name

    PFI = "PFI"
    # Protected Financial Information
    # Regulation: GLBA, SOX
    # Examples: account numbers, routing numbers,
    #           credit scores, loan details

    MIXED = "MIXED"
    # Multiple sensitivity types present
    # Requires review of all types found

    NONE = "NONE"
    # No sensitive data detected

    UNKNOWN = "UNKNOWN"
    # Not yet classified


class AccessorType(Enum):
    """
    Type of identity accessing the data.

    WHY THIS MATTERS:
    Same anomaly score means different things
    for different accessor types.

    anomaly_score=0.7 for HUMAN:
        Investigate immediately
        Possible insider threat

    anomaly_score=0.7 for ETL_PROCESS:
        Check if job was modified
        Review job scheduling

    anomaly_score=0.7 for SERVICE_ACCOUNT:
        Check if running off-schedule
        Verify destination unchanged
    """
    HUMAN = "human"
    # Person interactively logged in

    SERVICE_ACCOUNT = "service_account"
    # svc_backup, svc_etl, svc_reporting
    # Non-human, application-level identity

    ETL_PROCESS = "etl_process"
    # Scheduled batch jobs
    # Streaming pipelines (Kafka, Spark)
    # Data migration jobs

    APPLICATION = "application"
    # Core banking (Temenos, Finacle)
    # CRM (Salesforce), AML system
    # Fraud detection, loan origination

    PRIVILEGED = "privileged"
    # DBA tools, admin consoles
    # Direct database access tools
    # Highest risk category

    API_CLIENT = "api_client"
    # REST API integrations
    # Mobile banking apps
    # Partner integrations

    REPLICATION = "replication"
    # Database replication processes
    # DR/failover systems
    # CDC (Change Data Capture)

    BACKUP = "backup"
    # Backup software agents
    # Commvault, Veeam, AWS Backup
    # Often has broad read access

    UNKNOWN = "unknown"


class DataStoreType(Enum):
    """
    Type of data store being accessed.
    Used to determine which normalizer processed
    this event and what access patterns are normal.
    """
    # Cloud Storage
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"

    # Relational Databases
    RDS = "rds"
    SQL_SERVER = "sql_server"
    ORACLE = "oracle"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"

    # On-Premise Databases
    DB2 = "db2"
    SYBASE = "sybase"
    INFORMIX = "informix"

    # Data Warehouses
    SNOWFLAKE = "snowflake"
    REDSHIFT = "redshift"
    DATABRICKS = "databricks"
    BIGQUERY = "bigquery"
    SYNAPSE = "synapse"

    # File Systems
    SHAREPOINT = "sharepoint"
    ONEDRIVE = "onedrive"
    NFS = "nfs"
    SMB = "smb"

    # Banking Specific
    TEMENOS = "temenos"
    FINACLE = "finacle"
    FIS = "fis"
    JACK_HENRY = "jack_henry"

    # Email and Collaboration
    EXCHANGE = "exchange"
    TEAMS = "teams"
    SLACK = "slack"

    UNKNOWN = "unknown"


class DataOperation(Enum):
    """
    Type of operation performed on data.
    Higher risk operations scored more severely.
    """
    READ = "read"           # SELECT, GET
    WRITE = "write"         # INSERT, PUT
    UPDATE = "update"       # UPDATE
    DELETE = "delete"       # DELETE — always high risk
    LIST = "list"           # LIST, DESCRIBE — enumeration
    EXPORT = "export"       # EXPORT, COPY OUT
    IMPORT = "import"       # IMPORT, COPY IN
    BULK_READ = "bulk_read" # Reading many records
    SCHEMA = "schema"       # Reading table structure
    UNKNOWN = "unknown"


class Environment(Enum):
    """
    Environment where data store lives.
    Production data has highest risk.
    Development should never have real PII.
    """
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    TEST = "test"
    DISASTER_RECOVERY = "disaster_recovery"
    UNKNOWN = "unknown"


# ============================================================
# DATA FINDING
#
# WHY SEPARATE FROM EVENT:
# A DataAccessEvent tells us WHO accessed WHAT.
# A DataFinding tells us WHAT WAS FOUND inside.
# They are populated at different times:
#   Event: when access occurs (real-time)
#   Finding: when PII classifier runs (async)
# ============================================================

@dataclass
class DataFinding:
    """
    Result of PII classifier scanning data content.

    Populated by PII classifier after sampling
    the content of the accessed data store.

    WHY SEPARATE FROM EVENT:
    Access event is real-time.
    Content classification may be async
    (cannot always sample in real-time).
    Keeping them separate allows:
    - Real-time access monitoring
    - Async content classification
    - Both feeding the knowledge graph
    """
    sensitivity_label: SensitivityLabel = (
        SensitivityLabel.UNKNOWN
    )
    confidence: float = 0.0

    # What types of sensitive data were found
    data_types_found: list = field(
        default_factory=list
    )
    # Examples:
    # ["SSN", "CREDIT_CARD", "EMAIL"]
    # ["MEDICAL_RECORD_NUMBER", "DIAGNOSIS"]
    # ["PAN", "CVV", "CARDHOLDER_NAME"]

    # Sample count for each type
    # Used to assess volume of exposure
    data_type_counts: dict = field(
        default_factory=dict
    )
    # Example: {"SSN": 1500, "EMAIL": 3000}

    # Regulatory frameworks triggered
    regulations_triggered: list = field(
        default_factory=list
    )
    # Example: ["GDPR", "CCPA", "PCI-DSS"]

    # Detection method used
    detection_method: str = ""
    # "regex", "ml", "hybrid"

    # Sample of detected data (MASKED)
    # Never store actual sensitive values
    sample_context: str = ""
    # Example: "Found SSN pattern: XXX-XX-XXXX
    #           near text 'customer_id'"

    # Classification timestamp
    classified_at: str = ""

    def to_dict(self) -> dict:
        return {
            "sensitivity_label": (
                self.sensitivity_label.value
            ),
            "confidence": self.confidence,
            "data_types_found": self.data_types_found,
            "data_type_counts": self.data_type_counts,
            "regulations_triggered": (
                self.regulations_triggered
            ),
            "detection_method": self.detection_method,
            "classified_at": self.classified_at
        }


# ============================================================
# DATA ACCESS EVENT
#
# THE UNIVERSAL SCHEMA FOR ALL DATA SOURCES
# ============================================================

@dataclass
class DataAccessEvent:
    """
    Universal schema for ALL data access events.

    Works for every data source in a bank:
    Cloud: S3, Azure Blob, GCS
    Databases: Oracle, SQL Server, DB2, MySQL
    Warehouses: Snowflake, Redshift, Databricks
    Files: SharePoint, network drives
    Banking: Temenos, Finacle, FIS
    Email: Exchange, Teams

    This is the THIRD normalized schema:
    ECSNormalized   → endpoint/network events
    IamEvent        → identity events
    DataAccessEvent → data access events

    Together they give complete visibility:
    WHAT happened on the endpoint (EDR)
    WHO did it (IAM)
    WHAT DATA they touched (DSPM)

    ACCESSOR TYPE MATTERS FOR RISK:
    Human reading 10,000 records → investigate now
    svc_backup reading 10,000 records at 2am → normal
    svc_backup reading 10,000 records at 2pm → flag
    ETL accessing new table → governance violation
    DBA exporting entire database → critical alert
    """

    # ---- EVENT IDENTIFICATION ----
    event_id: str = ""
    event_time: str = ""
    source_system: str = ""
    # Which normalizer produced this event
    # "aws_s3", "oracle_audit", "snowflake"

    # ---- WHO ACCESSED ----
    accessor_identity: str = ""
    # The actual identifier:
    # Human: "jsmith@corp.com"
    # Service: "svc_backup"
    # ETL: "etl_daily_customers_job"
    # App: "temenos_core_banking"

    accessor_type: AccessorType = (
        AccessorType.UNKNOWN
    )
    accessor_domain: str = ""
    # Domain or namespace of accessor

    # ---- PROCESS CONTEXT ----
    # For ETL and service accounts
    process_name: str = ""
    # "daily_customer_extract.py"

    job_schedule: str = ""
    # When this accessor normally runs
    # "0 2 * * *" = 2am daily (cron format)

    is_scheduled_access: bool = False
    # Was this a scheduled/expected access?
    # Set by comparing event_time to job_schedule

    # ---- WHAT WAS ACCESSED ----
    data_store_type: DataStoreType = (
        DataStoreType.UNKNOWN
    )
    data_store_name: str = ""
    # "prod-customer-data-bucket"
    # "CUST_MASTER_TABLE"
    # "customers.pii_schema"

    data_path: str = ""
    # S3: "customers/pii/2024-q1.csv"
    # DB: "schema.table_name"
    # File: "/data/reports/customer_export.xlsx"

    database_name: str = ""
    schema_name: str = ""
    table_name: str = ""
    # Populated for database sources

    environment: Environment = (
        Environment.UNKNOWN
    )
    # production, staging, development, test

    # ---- HOW IT WAS ACCESSED ----
    operation: DataOperation = (
        DataOperation.UNKNOWN
    )
    query_text: str = ""
    # SQL query if available (masked for PII)

    # ---- VOLUME ----
    rows_accessed: int = 0
    bytes_accessed: int = 0
    files_accessed: int = 0
    # For file-based stores (S3, SharePoint)

    # ---- WHERE FROM ----
    source_ip: str = ""
    source_region: str = ""
    # "us-east-1", "eu-west-1"

    source_application: str = ""
    # Application that made the access

    # ---- TEMPORAL SIGNALS ----
    is_off_hours: bool = False
    # Access outside normal business hours

    is_off_schedule: bool = False
    # For ETL/service accounts:
    # access outside their normal schedule

    is_weekend: bool = False

    # ---- DATA CLASSIFICATION ----
    # Populated by PII classifier
    finding: Optional[DataFinding] = None

    # ---- RISK ASSESSMENT ----
    risk_score: float = 0.0
    risk_label: str = "UNKNOWN"
    risk_reasons: list = field(
        default_factory=list
    )

    # ---- RAW EVENT ----
    raw_event: dict = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_time": self.event_time,
            "source_system": self.source_system,
            "accessor_identity": (
                self.accessor_identity
            ),
            "accessor_type": (
                self.accessor_type.value
            ),
            "data_store_type": (
                self.data_store_type.value
            ),
            "data_store_name": self.data_store_name,
            "data_path": self.data_path,
            "operation": self.operation.value,
            "rows_accessed": self.rows_accessed,
            "bytes_accessed": self.bytes_accessed,
            "environment": self.environment.value,
            "is_off_hours": self.is_off_hours,
            "is_off_schedule": self.is_off_schedule,
            "risk_score": self.risk_score,
            "risk_label": self.risk_label,
            "risk_reasons": self.risk_reasons,
            "finding": (
                self.finding.to_dict()
                if self.finding else None
            )
        }


# ============================================================
# DATA STORE PROFILE
#
# WHY THIS EXISTS:
# Knowing WHAT data is IN a store
# is different from knowing who accessed it.
#
# This profile is built up over time
# as the PII classifier scans data stores.
# It tells Layer 3 what each store contains.
# ============================================================

@dataclass
class DataStoreProfile:
    """
    Profile of what a data store contains.

    Built by PII classifier scanning content.
    Stored in Layer 3 knowledge graph.
    Used to instantly know sensitivity
    when new access events arrive.

    Instead of scanning every time:
    1. Scan once → build profile
    2. New access event arrives
    3. Look up profile → instant sensitivity

    Profile refreshed on schedule or when
    data store changes.
    """
    store_id: str = ""
    store_name: str = ""
    store_type: DataStoreType = (
        DataStoreType.UNKNOWN
    )
    environment: Environment = (
        Environment.UNKNOWN
    )

    # What sensitive data this store contains
    sensitivity_labels: list = field(
        default_factory=list
    )
    # ["PII", "PCI"] means both present

    data_types_present: list = field(
        default_factory=list
    )
    # ["SSN", "CREDIT_CARD", "EMAIL", "DOB"]

    regulations_applicable: list = field(
        default_factory=list
    )
    # ["GDPR", "PCI-DSS", "CCPA"]

    # Estimated record counts by type
    estimated_pii_records: int = 0
    estimated_pci_records: int = 0
    estimated_phi_records: int = 0

    # Who should own this data store
    data_owner: str = ""
    # "fraud_team", "customer_service", "finance"

    # When last scanned
    last_scanned: str = ""
    last_scan_method: str = ""
    # "full_scan", "sample_scan", "manual"

    # Access control assessment
    is_publicly_accessible: bool = False
    is_encrypted: bool = True
    has_access_logging: bool = True

    # Overall risk
    overall_risk_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "store_id": self.store_id,
            "store_name": self.store_name,
            "store_type": self.store_type.value,
            "environment": self.environment.value,
            "sensitivity_labels": (
                self.sensitivity_labels
            ),
            "data_types_present": (
                self.data_types_present
            ),
            "regulations_applicable": (
                self.regulations_applicable
            ),
            "estimated_pii_records": (
                self.estimated_pii_records
            ),
            "is_publicly_accessible": (
                self.is_publicly_accessible
            ),
            "is_encrypted": self.is_encrypted,
            "last_scanned": self.last_scanned,
            "overall_risk_score": (
                self.overall_risk_score
            )
        }