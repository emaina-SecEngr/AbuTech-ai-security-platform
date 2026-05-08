"""
Layer 1 — Data Ingestion
AWS RDS Database Activity Normalizer

WHY THIS FILE EXISTS:
    RDS stores the most sensitive structured data
    in banking environments:
    Customer master tables, account records,
    transaction history, credit scores.

    S3 exfiltration is one big file download.
    RDS exfiltration is many small queries.
    Each query looks normal individually.
    The PATTERN across queries reveals the attack.

    Without this normalizer:
    Attacker runs SELECT * FROM customers
    15,000 times over 1 hour = 1.5M records
    Platform sees nothing.

    With this normalizer:
    Every query → DataAccessEvent
    Row accumulation tracked per accessor
    Pattern detected automatically.

TWO RDS LOG SOURCES:

SOURCE 1: RDS Database Activity Streams
    Row-level audit logging
    Every SQL query with row counts
    PII visible in WHERE clauses (masked before storage)
    Best for detecting data access patterns

    {
        "type": "DatabaseActivityMonitoringRecord",
        "databaseActivityEventList": [{
            "type": "record",
            "dbUserName": "svc_reporting",
            "remoteHost": "10.0.0.155",
            "command": "SELECT",
            "commandText": "SELECT * FROM customers
                            WHERE ssn = '...'",
            "objectName": "customers",
            "databaseName": "prod_db",
            "rowCount": 50000
        }]
    }

SOURCE 2: CloudTrail RDS Events
    API-level logging (connect, disconnect)
    Not query-level detail
    Used for connection pattern analysis

KEY DETECTIONS:
    - High row count queries (bulk extraction)
    - SELECT * patterns (full table reads)
    - Sensitive table access (/pii/, customers)
    - Off-hours database access
    - New accessor for database
    - PII in WHERE clause (masked automatically)
    - Accumulated rows across session
"""

import logging
import re
from datetime import datetime
from datetime import timezone
from typing import Optional

from layer1_ingestion.schema.data_schema import (
    DataAccessEvent,
    AccessorType,
    DataStoreType,
    DataOperation,
    Environment,
    SensitivityLabel
)

logger = logging.getLogger(__name__)


# ============================================================
# RDS OPERATION MAPPINGS
# ============================================================

READ_COMMANDS = {
    "SELECT", "SHOW", "DESCRIBE",
    "EXPLAIN", "DESC"
}

WRITE_COMMANDS = {
    "INSERT", "UPDATE", "MERGE",
    "REPLACE", "UPSERT"
}

DELETE_COMMANDS = {
    "DELETE", "TRUNCATE", "DROP"
}

SCHEMA_COMMANDS = {
    "CREATE", "ALTER", "RENAME",
    "COMMENT", "GRANT", "REVOKE"
}

HIGH_RISK_COMMANDS = {
    "SELECT *",     # Full table read
    "DROP",         # Destructive
    "TRUNCATE",     # Destructive
    "GRANT",        # Privilege escalation
    "REVOKE"        # Privilege removal
}

# ============================================================
# SENSITIVE TABLE/SCHEMA PATTERNS
#
# WHY TABLE-LEVEL SENSITIVITY:
# Database tables are named by developers.
# Table names reveal content:
# customers_pii → contains PII
# payment_cards → contains PCI
# medical_records → contains PHI
# ============================================================

HIGH_SENSITIVITY_TABLE_KEYWORDS = {
    "pii", "phi", "pci", "ssn", "sensitive",
    "confidential", "restricted", "secret",
    "private", "protected", "classified",
    "password", "credential", "secret_key"
}

MEDIUM_SENSITIVITY_TABLE_KEYWORDS = {
    "customer", "customers", "patient", "patients",
    "employee", "employees", "member", "members",
    "account", "accounts", "payment", "payments",
    "transaction", "transactions", "financial",
    "medical", "health", "insurance", "card",
    "cardholder", "credit", "loan", "mortgage",
    "personal", "identity", "profile", "contact"
}

# Row count thresholds
# Why rows not bytes: databases track exact records.
# 1 row = 1 customer = 1 GDPR data subject.
# Row count directly maps to breach notification scope.
NORMAL_QUERY_ROWS = 1000
ELEVATED_QUERY_ROWS = 10_000
HIGH_QUERY_ROWS = 100_000
CRITICAL_QUERY_ROWS = 1_000_000

# Business hours UTC
BUSINESS_HOURS_START = 13
BUSINESS_HOURS_END = 23

# SELECT * detection pattern
SELECT_ALL_PATTERN = re.compile(
    r"SELECT\s+\*\s+FROM",
    re.IGNORECASE
)


class RDSNormalizer:
    """
    Normalizes AWS RDS Database Activity Stream
    events to DataAccessEvent objects.

    KEY DIFFERENCE FROM S3 NORMALIZER:
    S3: one event per file access
    RDS: many events per session
         Pattern ACROSS events is the signal

    Tracks accumulated rows per accessor per session
    to detect incremental exfiltration.

    Usage:
        normalizer = RDSNormalizer()
        event = normalizer.normalize(
            raw_activity_stream_event
        )
    """

    def __init__(self):
        # Track accumulated rows per accessor
        # key: db_user
        # value: {"rows": int, "queries": int}
        self._accessor_row_counts = {}

        # Track known accessors per database
        self._db_accessors = {}

        # Statistics
        self.events_processed = 0
        self.high_risk_events = 0
        self.bulk_reads_detected = 0
        self.select_all_detected = 0
        self.off_hours_detected = 0

        logger.info("RDSNormalizer initialized")

    def normalize(
        self,
        raw_event: dict
    ) -> Optional[DataAccessEvent]:
        """
        Normalize RDS Database Activity Stream event.

        ETL PIPELINE:
        1. Extract query details
        2. Extract database user identity
        3. Map SQL command to DataOperation
        4. Score table name sensitivity
        5. Detect SELECT * pattern
        6. Track accumulated row counts
        7. Calculate risk score
        8. Return DataAccessEvent
        """
        if not raw_event:
            return None

        try:
            # Handle both Activity Stream format
            # and direct event format
            activity_list = raw_event.get(
                "databaseActivityEventList", []
            )

            if activity_list:
                # Process first event in list
                # In production process all events
                db_event = activity_list[0]
            else:
                db_event = raw_event

            # ---- EXTRACT CORE FIELDS ----
            event_time = (
                db_event.get("startTime", "") or
                db_event.get("eventTime", "") or
                raw_event.get("eventTime", "")
            )
            event_id = (
                db_event.get("logTime", "") or
                db_event.get("requestID", "")
            )

            # ---- EXTRACT IDENTITY ----
            db_user = db_event.get(
                "dbUserName", ""
            ) or db_event.get("userName", "")
            remote_host = db_event.get(
                "remoteHost", ""
            ) or db_event.get("sourceIPAddress", "")

            # ---- EXTRACT DATABASE CONTEXT ----
            database_name = db_event.get(
                "databaseName", ""
            ) or db_event.get("dbName", "")
            schema_name = db_event.get(
                "schemaName", ""
            ) or db_event.get("schema", "")
            table_name = db_event.get(
                "objectName", ""
            ) or db_event.get("tableName", "")
            db_instance = raw_event.get(
                "dbInstanceIdentifier", ""
            ) or db_event.get("dbInstance", "")

            # ---- EXTRACT QUERY ----
            command = db_event.get("command", "")
            command_text = db_event.get(
                "commandText", ""
            ) or db_event.get("queryText", "")

            # ---- EXTRACT ROW COUNT ----
            # WHY ROWS NOT BYTES:
            # 1 row = 1 customer record
            # Row count maps directly to
            # breach notification scope
            row_count = int(
                db_event.get("rowCount", 0) or
                db_event.get("rows", 0) or
                0
            )

            # ---- DETERMINE OPERATION ----
            operation = self._map_operation(command)

            # ---- DETERMINE ENVIRONMENT ----
            environment = self._detect_environment(
                db_instance, database_name
            )

            # ---- ACCESSOR TYPE ----
            accessor_type = self._map_accessor_type(
                db_user
            )

            # ---- BUILD DATA PATH ----
            # Normalize to schema.table format
            # for consistent path sensitivity scoring
            data_path = self._build_data_path(
                database_name, schema_name, table_name
            )

            # ---- PATH SENSITIVITY ----
            path_sensitivity, path_confidence = (
                self._score_path_sensitivity(
                    data_path, table_name
                )
            )

            # ---- SELECT * DETECTION ----
            is_select_all = self._detect_select_all(
                command_text
            )
            if is_select_all:
                self.select_all_detected += 1

            # ---- BULK READ DETECTION ----
            is_bulk_read = row_count >= ELEVATED_QUERY_ROWS
            if is_bulk_read:
                self.bulk_reads_detected += 1

            # ---- MASK QUERY TEXT ----
            # Must mask before storage
            # GDPR Article 25 + PCI-DSS Req 3.4
            # SQL WHERE clauses contain actual PII values
            masked_query = self._mask_query(
                command_text
            )

            # ---- OFF HOURS DETECTION ----
            is_off_hours = self._is_off_hours(
                event_time
            )
            if is_off_hours:
                self.off_hours_detected += 1

            # ---- NEW ACCESSOR DETECTION ----
            is_new_accessor = self._is_new_accessor(
                db_user, database_name
            )

            # ---- ACCUMULATE ROW COUNTS ----
            accumulated_rows = (
                self._accumulate_rows(
                    db_user, row_count
                )
            )

            # ---- BUILD DATA ACCESS EVENT ----
            event = DataAccessEvent(
                event_id=str(event_id),
                event_time=event_time,
                source_system="aws_rds",
                accessor_identity=db_user,
                accessor_type=accessor_type,
                data_store_type=DataStoreType.RDS,
                data_store_name=db_instance,
                data_path=data_path,
                database_name=database_name,
                schema_name=schema_name,
                table_name=table_name,
                operation=operation,
                query_text=masked_query,
                rows_accessed=row_count,
                environment=environment,
                source_ip=remote_host,
                is_off_hours=is_off_hours,
                raw_event=raw_event
            )

            # ---- RISK SCORING ----
            risk_score, risk_reasons = (
                self._calculate_risk(
                    event=event,
                    path_sensitivity=path_sensitivity,
                    path_confidence=path_confidence,
                    is_select_all=is_select_all,
                    is_bulk_read=is_bulk_read,
                    is_new_accessor=is_new_accessor,
                    row_count=row_count,
                    accumulated_rows=accumulated_rows,
                    command=command
                )
            )

            event.risk_score = risk_score
            event.risk_label = (
                self._score_to_label(risk_score)
            )
            event.risk_reasons = risk_reasons

            # ---- UPDATE HISTORY ----
            self._update_history(
                db_user, database_name
            )

            self.events_processed += 1
            if risk_score >= 0.7:
                self.high_risk_events += 1

            logger.info(
                f"RDS event normalized: "
                f"{command} "
                f"table={table_name} "
                f"user={db_user} "
                f"rows={row_count} "
                f"risk={risk_score:.2f}"
            )

            return event

        except Exception as e:
            logger.error(
                f"RDS normalization failed: {e}"
            )
            return None

    # ============================================================
    # FIELD EXTRACTORS
    # ============================================================

    def _map_operation(
        self,
        command: str
    ) -> DataOperation:
        """Map SQL command to DataOperation"""
        cmd_upper = command.upper().strip()

        if cmd_upper in READ_COMMANDS:
            return DataOperation.READ
        elif cmd_upper in WRITE_COMMANDS:
            return DataOperation.WRITE
        elif cmd_upper in DELETE_COMMANDS:
            return DataOperation.DELETE
        elif cmd_upper in SCHEMA_COMMANDS:
            return DataOperation.SCHEMA

        # Check for bulk operations
        if "SELECT" in cmd_upper:
            return DataOperation.READ
        elif "INSERT" in cmd_upper:
            return DataOperation.WRITE
        elif "DELETE" in cmd_upper:
            return DataOperation.DELETE

        return DataOperation.UNKNOWN

    def _map_accessor_type(
        self,
        db_user: str
    ) -> AccessorType:
        """
        Map database user to AccessorType.
        Database users follow naming conventions.
        """
        user_lower = db_user.lower()

        if any(
            p in user_lower
            for p in ["etl", "pipeline", "batch"]
        ):
            return AccessorType.ETL_PROCESS

        if any(
            p in user_lower
            for p in ["backup", "bkp"]
        ):
            return AccessorType.BACKUP

        if any(
            p in user_lower
            for p in ["repl", "replica", "sync"]
        ):
            return AccessorType.REPLICATION

        if any(
            p in user_lower
            for p in ["app", "api", "svc", "service"]
        ):
            return AccessorType.APPLICATION

        if any(
            p in user_lower
            for p in ["dba", "admin", "root", "sa"]
        ):
            return AccessorType.PRIVILEGED

        if any(
            p in user_lower
            for p in ["report", "bi", "tableau",
                      "power", "analytics"]
        ):
            return AccessorType.SERVICE_ACCOUNT

        return AccessorType.HUMAN

    def _detect_environment(
        self,
        db_instance: str,
        database_name: str
    ) -> Environment:
        """Detect environment from instance/db name"""
        combined = (
            f"{db_instance} {database_name}"
        ).lower()

        if any(
            p in combined
            for p in ["prod", "production", "prd"]
        ):
            return Environment.PRODUCTION

        if any(
            p in combined
            for p in ["stag", "staging", "uat"]
        ):
            return Environment.STAGING

        if any(
            p in combined
            for p in ["dev", "develop", "test"]
        ):
            return Environment.DEVELOPMENT

        return Environment.UNKNOWN

    def _build_data_path(
        self,
        database: str,
        schema: str,
        table: str
    ) -> str:
        """
        Build normalized data path.
        Format: database.schema.table
        Consistent across RDS and Snowflake.
        """
        parts = [
            p for p in [database, schema, table]
            if p
        ]
        return ".".join(parts)

    def _score_path_sensitivity(
        self,
        data_path: str,
        table_name: str
    ) -> tuple:
        """
        Score sensitivity from table/path name.

        Normalizes dots, slashes, underscores
        to spaces for consistent keyword matching
        across S3, RDS, and Snowflake paths.
        """
        # Normalize path for keyword matching
        normalized = data_path.lower()
        normalized = re.sub(r'[._/]', ' ', normalized)
        parts = normalized.split()

        # Check high sensitivity keywords
        high_hits = sum(
            1 for kw in HIGH_SENSITIVITY_TABLE_KEYWORDS
            if kw in parts or
            any(kw in p for p in parts)
        )

        # Check medium sensitivity
        medium_hits = sum(
            1 for kw in MEDIUM_SENSITIVITY_TABLE_KEYWORDS
            if kw in parts or
            any(kw in p for p in parts)
        )

        if high_hits >= 1:
            return SensitivityLabel.PII, 0.85

        if medium_hits >= 2:
            confidence = min(
                0.75, 0.4 + (medium_hits * 0.1)
            )
            return SensitivityLabel.PII, confidence

        if medium_hits == 1:
            return SensitivityLabel.PII, 0.4

        return SensitivityLabel.UNKNOWN, 0.1

    def _detect_select_all(
        self,
        query_text: str
    ) -> bool:
        """
        Detect SELECT * FROM pattern.

        SELECT * reads ALL columns including
        sensitive ones the accessor may not need.
        Principle of least privilege violation.
        Attackers use SELECT * for bulk extraction.
        """
        if not query_text:
            return False
        return bool(
            SELECT_ALL_PATTERN.search(query_text)
        )

    def _mask_query(
        self,
        query_text: str
    ) -> str:
        """
        Mask PII values in SQL query text.

        Must mask before storing in audit trail.
        Bank queries contain actual PII in WHERE:
        WHERE ssn = '123-45-6789'
        WHERE card_number = '4532...'

        GDPR Article 25 — privacy by design.
        PCI-DSS Requirement 3.4 — mask PAN.
        """
        if not query_text:
            return ""

        masked = query_text

        # Mask SSN patterns
        masked = re.sub(
            r"\b\d{3}-\d{2}-\d{4}\b",
            "XXX-XX-XXXX",
            masked
        )

        # Mask credit card patterns
        masked = re.sub(
            r"\b\d{4}[\s-]?\d{4}[\s-]?"
            r"\d{4}[\s-]?\d{4}\b",
            "XXXX-XXXX-XXXX-XXXX",
            masked
        )

        # Mask email addresses
        masked = re.sub(
            r"\b[a-zA-Z0-9._%+-]+"
            r"@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            "[EMAIL_MASKED]",
            masked
        )

        # Mask quoted string values in WHERE clauses
        # These often contain PII
        masked = re.sub(
            r"(WHERE\s+\w+\s*=\s*)'[^']*'",
            r"\1'[MASKED]'",
            masked,
            flags=re.IGNORECASE
        )

        return masked

    def _is_off_hours(
        self,
        event_time: str
    ) -> bool:
        """Check if access outside business hours"""
        try:
            if "T" in event_time:
                hour = int(
                    event_time.split("T")[1][:2]
                )
                return not (BUSINESS_HOURS_START <= hour < BUSINESS_HOURS_END)
    
        except Exception:
            pass
        return False

    def _is_new_accessor(
        self,
        db_user: str,
        database_name: str
    ) -> bool:
        """Check if first time user accessed database"""
        if not database_name:
            return False
        known = self._db_accessors.get(
            database_name, set()
        )
        return db_user not in known

    def _accumulate_rows(
        self,
        db_user: str,
        row_count: int
    ) -> int:
        """
        Accumulate row counts per accessor.

        WHY ACCUMULATE:
        One SELECT returning 1000 rows = normal.
        Same query 1500 times = 1.5M rows = exfiltration.

        Accumulation catches incremental exfiltration
        that individual query thresholds miss.
        """
        if db_user not in self._accessor_row_counts:
            self._accessor_row_counts[db_user] = {
                "rows": 0,
                "queries": 0
            }

        self._accessor_row_counts[db_user][
            "rows"
        ] += row_count
        self._accessor_row_counts[db_user][
            "queries"
        ] += 1

        return self._accessor_row_counts[
            db_user
        ]["rows"]

    # ============================================================
    # RISK SCORING
    # ============================================================

    def _calculate_risk(
        self,
        event: DataAccessEvent,
        path_sensitivity: SensitivityLabel,
        path_confidence: float,
        is_select_all: bool,
        is_bulk_read: bool,
        is_new_accessor: bool,
        row_count: int,
        accumulated_rows: int,
        command: str
    ) -> tuple:
        """Calculate risk for RDS access event"""
        score = 0.0
        reasons = []

        # Sensitive table detected
        if path_confidence >= 0.8:
            score += 0.4
            reasons.append(
                f"Sensitive table accessed: "
                f"{event.data_path} — "
                f"likely {path_sensitivity.value} data"
            )
        elif path_confidence >= 0.4:
            score += 0.2
            reasons.append(
                f"Potentially sensitive table: "
                f"{event.data_path}"
            )

        # SELECT * pattern
        if is_select_all:
            score += 0.3
            reasons.append(
                "SELECT * detected — full table read. "
                "Principle of least privilege violation. "
                "Common exfiltration technique."
            )

        # Bulk read by row count
        if row_count >= CRITICAL_QUERY_ROWS:
            score += 0.6
            reasons.append(
                f"Critical row count: "
                f"{row_count:,} rows in single query — "
                f"possible full table extraction"
            )
        elif row_count >= HIGH_QUERY_ROWS:
            score += 0.4
            reasons.append(
                f"High row count: "
                f"{row_count:,} rows accessed"
            )
        elif is_bulk_read:
            score += 0.2
            reasons.append(
                f"Elevated row count: "
                f"{row_count:,} rows accessed"
            )

        # Accumulated rows across session
        if accumulated_rows >= CRITICAL_QUERY_ROWS:
            score += 0.3
            reasons.append(
                f"Accumulated session rows: "
                f"{accumulated_rows:,} total rows — "
                f"possible incremental exfiltration"
            )

        # Destructive operations
        if command.upper() in ["DROP", "TRUNCATE"]:
            score += 0.5
            reasons.append(
                f"Destructive operation: {command} — "
                f"data loss risk"
            )

        # Off hours
        if event.is_off_hours:
            score += 0.2
            reasons.append(
                "Database access outside business hours"
            )

        # New accessor
        if is_new_accessor:
            score += 0.2
            reasons.append(
                f"First time {event.accessor_identity} "
                f"accessed {event.data_store_name}"
            )

        # Production environment
        if event.environment == Environment.PRODUCTION:
            score += 0.1
            reasons.append(
                "Production database access"
            )

        # Development with sensitive data
        if event.environment == Environment.DEVELOPMENT:
            if path_confidence >= 0.4:
                score += 0.3
                reasons.append(
                    "Sensitive table accessed in "
                    "development environment — "
                    "governance violation: dev should "
                    "use masked data"
                )

        # Privileged user
        if event.accessor_type == AccessorType.PRIVILEGED:
            score += 0.3
            reasons.append(
                f"Privileged database user: "
                f"{event.accessor_identity}"
            )

        return min(score, 1.0), reasons

    # ============================================================
    # HISTORY MANAGEMENT
    # ============================================================

    def _update_history(
        self,
        db_user: str,
        database_name: str
    ) -> None:
        """Update accessor and database history"""
        if database_name:
            if database_name not in self._db_accessors:
                self._db_accessors[database_name] = set()
            self._db_accessors[database_name].add(
                db_user
            )

    def _score_to_label(self, score: float) -> str:
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score > 0.0:
            return "LOW"
        return "UNKNOWN"

    def get_statistics(self) -> dict:
        return {
            "events_processed": self.events_processed,
            "high_risk_events": self.high_risk_events,
            "bulk_reads_detected": (
                self.bulk_reads_detected
            ),
            "select_all_detected": (
                self.select_all_detected
            ),
            "off_hours_detected": (
                self.off_hours_detected
            ),
            "databases_tracked": len(
                self._db_accessors
            ),
            "accessors_tracked": len(
                self._accessor_row_counts
            )
        }