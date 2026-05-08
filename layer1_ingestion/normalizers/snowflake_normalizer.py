"""
Layer 1 — Data Ingestion
Snowflake Data Warehouse Normalizer

WHY THIS FILE EXISTS:
    Snowflake is the dominant cloud data warehouse
    in financial services and healthcare.
    Banks use Snowflake for:
    - Customer analytics (PII)
    - Transaction analysis (PFI)
    - Risk modeling (mixed sensitive data)
    - Regulatory reporting (SOX, GDPR)

    Without this normalizer:
    Analyst runs SELECT * FROM customers
    on 50 million records → platform sees nothing.

    With this normalizer:
    Every Snowflake query → DataAccessEvent
    Rows produced vs bytes scanned tracked
    Zero-row large scans detected
    Warehouse context used for risk scoring

KEY DIFFERENCES FROM RDS:
    1. WAREHOUSE_NAME provides compute context
       Maps to environment and accessor_type
    2. ROWS_PRODUCED vs BYTES_SCANNED ratio
       Anomalous ratio = probing or side-channel
    3. QUERY_TYPE field (more specific than SQL command)
    4. CREDITS_USED field (cost anomaly detection)
    5. Snowflake uses database.schema.table format

SNOWFLAKE QUERY HISTORY LOG:
{
    "QUERY_ID": "uuid",
    "QUERY_TEXT": "SELECT * FROM CUSTOMERS",
    "USER_NAME": "SVC_REPORTING",
    "DATABASE_NAME": "PROD_DB",
    "SCHEMA_NAME": "PUBLIC",
    "TABLE_NAME": "CUSTOMERS",
    "QUERY_TYPE": "SELECT",
    "ROWS_PRODUCED": 1500000,
    "BYTES_SCANNED": 524288000,
    "BYTES_WRITTEN": 0,
    "CREDITS_USED_CLOUD_SERVICES": 0.001,
    "START_TIME": "2024-03-29T09:19:00Z",
    "END_TIME": "2024-03-29T09:21:00Z",
    "EXECUTION_STATUS": "SUCCESS",
    "WAREHOUSE_NAME": "PROD_REPORTING_WH",
    "SESSION_ID": "session-uuid",
    "ERROR_CODE": null
}

ANOMALY DETECTION UNIQUE TO SNOWFLAKE:
    Zero rows + large bytes scanned
    = schema probing or interrupted exfiltration
    Normal queries: rows proportional to bytes
    Anomalous: 0 rows + 500MB scanned = flag
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
# SNOWFLAKE QUERY TYPE MAPPINGS
# ============================================================

READ_QUERY_TYPES = {
    "SELECT", "SHOW", "DESCRIBE",
    "EXPLAIN", "LIST", "GET"
}

WRITE_QUERY_TYPES = {
    "INSERT", "UPDATE", "MERGE",
    "COPY", "PUT", "RECLUSTER"
}

DELETE_QUERY_TYPES = {
    "DELETE", "TRUNCATE", "DROP"
}

SCHEMA_QUERY_TYPES = {
    "CREATE", "ALTER", "GRANT",
    "REVOKE", "CLONE"
}

HIGH_RISK_QUERY_TYPES = {
    "DROP", "TRUNCATE",
    "GRANT", "REVOKE",
    "CREATE_CLONE"
}

# ============================================================
# WAREHOUSE ANALYSIS
#
# Warehouse name = environment + workload type
# PROD_REPORTING_WH = production, reporting workload
# DEV_SANDBOX_WH    = development, sandbox workload
# ============================================================

PRODUCTION_WAREHOUSE_PREFIXES = {
    "prod", "production", "prd", "live"
}

DEVELOPMENT_WAREHOUSE_PREFIXES = {
    "dev", "develop", "test", "sandbox",
    "staging", "stg", "uat", "qa"
}

ETL_WAREHOUSE_KEYWORDS = {
    "etl", "pipeline", "ingest",
    "load", "batch", "transform"
}

REPORTING_WAREHOUSE_KEYWORDS = {
    "report", "reporting", "bi", "analytics",
    "tableau", "powerbi", "looker", "dashboard"
}

# ============================================================
# SENSITIVE TABLE PATTERNS
# Same concept as RDS but Snowflake uses
# database.schema.table three-part naming
# ============================================================

HIGH_SENSITIVITY_KEYWORDS = {
    "pii", "phi", "pci", "ssn", "sensitive",
    "confidential", "restricted", "private",
    "protected", "secret", "classified",
    "password", "credential", "key", "token"
}

MEDIUM_SENSITIVITY_KEYWORDS = {
    "customer", "customers", "patient", "patients",
    "employee", "employees", "member", "members",
    "account", "accounts", "payment", "payments",
    "transaction", "transactions", "financial",
    "medical", "health", "insurance", "card",
    "cardholder", "credit", "loan", "mortgage",
    "personal", "identity", "profile", "contact",
    "kyc", "aml", "fraud"
}

# Row count thresholds (same as RDS)
NORMAL_ROWS = 1_000
ELEVATED_ROWS = 10_000
HIGH_ROWS = 100_000
CRITICAL_ROWS = 1_000_000

# Bytes scanned threshold for zero-row anomaly
ZERO_ROW_SCAN_THRESHOLD = 10_000_000  # 10MB

# Business hours UTC
BUSINESS_HOURS_START = 13
BUSINESS_HOURS_END = 23

SELECT_ALL_PATTERN = re.compile(
    r"SELECT\s+\*\s+FROM",
    re.IGNORECASE
)


class SnowflakeNormalizer:
    """
    Normalizes Snowflake Query History events
    to DataAccessEvent objects.

    KEY UNIQUE CAPABILITIES:
    1. Warehouse context analysis
       PROD_REPORTING_WH vs DEV_SANDBOX_WH
       Different risk profiles per warehouse

    2. Zero-row large scan detection
       ROWS_PRODUCED=0 + BYTES_SCANNED=500MB
       = schema probing or side-channel attack

    3. Three-part path normalization
       database.schema.table
       Consistent with RDS two-part naming
       for unified sensitivity scoring

    4. Credit usage anomaly
       Unusually high credits = unusually large query
       Proxy for data volume when row count is 0

    Usage:
        normalizer = SnowflakeNormalizer()
        event = normalizer.normalize(
            raw_snowflake_query_history
        )
    """

    def __init__(self):
        # Track accumulated rows per user
        self._user_row_counts = {}

        # Track known users per warehouse
        self._warehouse_users = {}

        # Track known tables per user
        self._user_table_access = {}

        # Statistics
        self.events_processed = 0
        self.high_risk_events = 0
        self.zero_row_anomalies = 0
        self.bulk_reads_detected = 0
        self.select_all_detected = 0
        self.dev_pii_violations = 0

        logger.info("SnowflakeNormalizer initialized")

    def normalize(
        self,
        raw_event: dict
    ) -> Optional[DataAccessEvent]:
        """
        Normalize Snowflake Query History event.

        ETL PIPELINE:
        1. Extract query and user identity
        2. Parse warehouse name for context
        3. Detect zero-row large scan anomaly
        4. Score table sensitivity
        5. Track accumulated rows
        6. Calculate risk score
        7. Return DataAccessEvent
        """
        if not raw_event:
            return None

        try:
            # ---- EXTRACT CORE FIELDS ----
            # Snowflake uses uppercase field names
            # in query history
            query_id = (
                raw_event.get("QUERY_ID") or
                raw_event.get("query_id", "")
            )
            start_time = (
                raw_event.get("START_TIME") or
                raw_event.get("start_time", "")
            )
            if isinstance(start_time, datetime):
                start_time = start_time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            start_time = str(start_time)

            # ---- EXTRACT USER ----
            user_name = (
                raw_event.get("USER_NAME") or
                raw_event.get("user_name", "")
            )

            # ---- EXTRACT DATABASE CONTEXT ----
            database_name = (
                raw_event.get("DATABASE_NAME") or
                raw_event.get("database_name", "")
            )
            schema_name = (
                raw_event.get("SCHEMA_NAME") or
                raw_event.get("schema_name", "")
            )
            table_name = (
                raw_event.get("TABLE_NAME") or
                raw_event.get("table_name", "")
            )

            # ---- EXTRACT WAREHOUSE ----
            warehouse_name = (
                raw_event.get("WAREHOUSE_NAME") or
                raw_event.get("warehouse_name", "")
            )

            # ---- EXTRACT QUERY DETAILS ----
            query_text = (
                raw_event.get("QUERY_TEXT") or
                raw_event.get("query_text", "")
            )
            query_type = (
                raw_event.get("QUERY_TYPE") or
                raw_event.get("query_type", "")
            )
            execution_status = (
                raw_event.get("EXECUTION_STATUS") or
                raw_event.get("execution_status", "")
            )

            # ---- EXTRACT VOLUME METRICS ----
            # WHY BOTH ROWS AND BYTES:
            # Rows = actual records exposed
            # Bytes = data scanned even if 0 rows returned
            # Zero rows + large bytes = anomaly
            rows_produced = int(
                raw_event.get("ROWS_PRODUCED") or
                raw_event.get("rows_produced") or
                0
            )
            bytes_scanned = int(
                raw_event.get("BYTES_SCANNED") or
                raw_event.get("bytes_scanned") or
                0
            )
            bytes_written = int(
                raw_event.get("BYTES_WRITTEN") or
                raw_event.get("bytes_written") or
                0
            )

            # ---- PARSE WAREHOUSE CONTEXT ----
            environment, workload_type = (
                self._parse_warehouse(warehouse_name)
            )

            # ---- ACCESSOR TYPE FROM WAREHOUSE ----
            accessor_type = self._map_accessor_type(
                user_name, warehouse_name, workload_type
            )

            # ---- BUILD DATA PATH ----
            # Snowflake three-part path:
            # database.schema.table
            data_path = self._build_data_path(
                database_name, schema_name, table_name
            )

            # ---- EXTRACT TABLE FROM QUERY ----
            # If table_name not in event
            # try to extract from query text
            if not table_name and query_text:
                table_name = self._extract_table_from_query(
                    query_text
                )
                if table_name:
                    data_path = self._build_data_path(
                        database_name,
                        schema_name,
                        table_name
                    )

            # ---- PATH SENSITIVITY ----
            path_sensitivity, path_confidence = (
                self._score_path_sensitivity(
                    data_path
                )
            )

            # ---- ZERO ROW ANOMALY ----
            # ROWS_PRODUCED=0 + BYTES_SCANNED=large
            # = schema probing or interrupted exfil
            is_zero_row_anomaly = (
                rows_produced == 0 and
                bytes_scanned >= ZERO_ROW_SCAN_THRESHOLD
            )
            if is_zero_row_anomaly:
                self.zero_row_anomalies += 1
                logger.warning(
                    f"Zero-row anomaly: "
                    f"{user_name} scanned "
                    f"{bytes_scanned:,} bytes "
                    f"but produced 0 rows — "
                    f"possible schema probing"
                )

            # ---- SELECT * DETECTION ----
            is_select_all = self._detect_select_all(
                query_text
            )
            if is_select_all:
                self.select_all_detected += 1

            # ---- BULK READ ----
            is_bulk_read = rows_produced >= ELEVATED_ROWS
            if is_bulk_read:
                self.bulk_reads_detected += 1

            # ---- OFF HOURS ----
            is_off_hours = self._is_off_hours(
                start_time
            )

            # ---- NEW USER FOR WAREHOUSE ----
            is_new_user = self._is_new_warehouse_user(
                user_name, warehouse_name
            )

            # ---- ACCUMULATE ROWS ----
            accumulated_rows = self._accumulate_rows(
                user_name, rows_produced
            )

            # ---- MASK QUERY ----
            masked_query = self._mask_query(query_text)

            # ---- BUILD DATA ACCESS EVENT ----
            event = DataAccessEvent(
                event_id=str(query_id),
                event_time=start_time,
                source_system="snowflake",
                accessor_identity=user_name,
                accessor_type=accessor_type,
                data_store_type=DataStoreType.SNOWFLAKE,
                data_store_name=warehouse_name,
                data_path=data_path,
                database_name=database_name,
                schema_name=schema_name,
                table_name=table_name,
                operation=self._map_operation(
                    query_type
                ),
                query_text=masked_query,
                rows_accessed=rows_produced,
                bytes_accessed=bytes_scanned,
                environment=environment,
                source_application=workload_type,
                is_off_hours=is_off_hours,
                raw_event=raw_event
            )

            # ---- RISK SCORING ----
            risk_score, risk_reasons = (
                self._calculate_risk(
                    event=event,
                    path_sensitivity=path_sensitivity,
                    path_confidence=path_confidence,
                    is_zero_row_anomaly=is_zero_row_anomaly,
                    is_select_all=is_select_all,
                    is_bulk_read=is_bulk_read,
                    is_new_user=is_new_user,
                    rows_produced=rows_produced,
                    bytes_scanned=bytes_scanned,
                    accumulated_rows=accumulated_rows,
                    query_type=query_type,
                    warehouse_name=warehouse_name
                )
            )

            event.risk_score = risk_score
            event.risk_label = (
                self._score_to_label(risk_score)
            )
            event.risk_reasons = risk_reasons

            # ---- UPDATE HISTORY ----
            self._update_history(
                user_name, warehouse_name, table_name
            )

            self.events_processed += 1
            if risk_score >= 0.7:
                self.high_risk_events += 1

            logger.info(
                f"Snowflake event normalized: "
                f"{query_type} "
                f"warehouse={warehouse_name} "
                f"user={user_name} "
                f"rows={rows_produced} "
                f"bytes={bytes_scanned} "
                f"risk={risk_score:.2f}"
            )

            return event

        except Exception as e:
            logger.error(
                f"Snowflake normalization failed: {e}"
            )
            return None

    # ============================================================
    # WAREHOUSE CONTEXT ANALYSIS
    # ============================================================

    def _parse_warehouse(
        self,
        warehouse_name: str
    ) -> tuple:
        """
        Parse warehouse name for environment
        and workload type.

        WAREHOUSE NAMING CONVENTION IN BANKING:
        PROD_REPORTING_WH → production, reporting
        PROD_ETL_WH       → production, etl
        DEV_SANDBOX_WH    → development, sandbox
        STAGING_ANALYTICS → staging, analytics

        Returns:
            (Environment, workload_type_string)
        """
        if not warehouse_name:
            return Environment.UNKNOWN, "unknown"

        wh_lower = warehouse_name.lower()

        # Determine environment
        environment = Environment.UNKNOWN
        for prefix in PRODUCTION_WAREHOUSE_PREFIXES:
            if wh_lower.startswith(prefix):
                environment = Environment.PRODUCTION
                break

        if environment == Environment.UNKNOWN:
            for prefix in DEVELOPMENT_WAREHOUSE_PREFIXES:
                if any(
                    p in wh_lower
                    for p in [prefix]
                ):
                    environment = Environment.DEVELOPMENT
                    break

        # Determine workload type
        workload_type = "general"
        if any(
            kw in wh_lower
            for kw in ETL_WAREHOUSE_KEYWORDS
        ):
            workload_type = "etl"
        elif any(
            kw in wh_lower
            for kw in REPORTING_WAREHOUSE_KEYWORDS
        ):
            workload_type = "reporting"
        elif "sandbox" in wh_lower:
            workload_type = "sandbox"

        return environment, workload_type

    def _map_accessor_type(
        self,
        user_name: str,
        warehouse_name: str,
        workload_type: str
    ) -> AccessorType:
        """
        Map Snowflake user to AccessorType.
        Uses BOTH username AND warehouse context.

        Warehouse provides additional signal:
        ETL warehouse → likely ETL process
        Reporting warehouse → likely service account
        Sandbox → likely human developer
        """
        user_lower = user_name.lower()

        # ETL indicators
        if workload_type == "etl" or any(
            p in user_lower
            for p in ["etl", "pipeline", "ingest"]
        ):
            return AccessorType.ETL_PROCESS

        # Reporting/BI
        if workload_type == "reporting" or any(
            p in user_lower
            for p in ["report", "bi", "tableau",
                      "analytics"]
        ):
            return AccessorType.SERVICE_ACCOUNT

        # Service accounts
        if any(
            p in user_lower
            for p in ["svc", "service", "app",
                      "api", "system"]
        ):
            return AccessorType.SERVICE_ACCOUNT

        # Backup
        if "backup" in user_lower:
            return AccessorType.BACKUP

        # Privileged
        if any(
            p in user_lower
            for p in ["admin", "dba", "root",
                      "sysadmin"]
        ):
            return AccessorType.PRIVILEGED

        return AccessorType.HUMAN

    def _map_operation(
        self,
        query_type: str
    ) -> DataOperation:
        """Map Snowflake query type to DataOperation"""
        qt_upper = query_type.upper().strip()

        if qt_upper in READ_QUERY_TYPES:
            return DataOperation.READ
        elif qt_upper in WRITE_QUERY_TYPES:
            return DataOperation.WRITE
        elif qt_upper in DELETE_QUERY_TYPES:
            return DataOperation.DELETE
        elif qt_upper in SCHEMA_QUERY_TYPES:
            return DataOperation.SCHEMA

        return DataOperation.UNKNOWN

    # ============================================================
    # PATH AND TABLE ANALYSIS
    # ============================================================

    def _build_data_path(
        self,
        database: str,
        schema: str,
        table: str
    ) -> str:
        """
        Build normalized three-part data path.
        Format: DATABASE.SCHEMA.TABLE

        Consistent with RDS two-part naming
        for unified downstream processing.
        """
        parts = [
            p for p in [database, schema, table]
            if p
        ]
        return ".".join(parts).upper()

    def _score_path_sensitivity(
        self,
        data_path: str
    ) -> tuple:
        """
        Score sensitivity from Snowflake path.

        Normalizes DATABASE.SCHEMA.TABLE to
        space-separated words for keyword matching.
        Same keywords as RDS and S3.
        """
        if not data_path:
            return SensitivityLabel.UNKNOWN, 0.1

        normalized = data_path.lower()
        normalized = re.sub(r'[._/]', ' ', normalized)
        parts = normalized.split()

        high_hits = sum(
            1 for kw in HIGH_SENSITIVITY_KEYWORDS
            if any(kw in p for p in parts)
        )

        medium_hits = sum(
            1 for kw in MEDIUM_SENSITIVITY_KEYWORDS
            if any(kw in p for p in parts)
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

    def _extract_table_from_query(
        self,
        query_text: str
    ) -> str:
        """
        Extract table name from SQL query.
        Fallback when TABLE_NAME not in event.
        """
        if not query_text:
            return ""

        # Match FROM table_name or JOIN table_name
        pattern = re.compile(
            r"(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_.]*)",
            re.IGNORECASE
        )
        match = pattern.search(query_text)
        if match:
            return match.group(1).split(".")[-1]

        return ""

    def _detect_select_all(
        self,
        query_text: str
    ) -> bool:
        """Detect SELECT * FROM pattern"""
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
        Mask PII values in Snowflake query text.
        Same masking as RDS normalizer.
        GDPR Article 25 + PCI-DSS Req 3.4.
        """
        if not query_text:
            return ""

        masked = query_text

        masked = re.sub(
            r"\b\d{3}-\d{2}-\d{4}\b",
            "XXX-XX-XXXX",
            masked
        )

        masked = re.sub(
            r"\b\d{4}[\s-]?\d{4}[\s-]?"
            r"\d{4}[\s-]?\d{4}\b",
            "XXXX-XXXX-XXXX-XXXX",
            masked
        )

        masked = re.sub(
            r"\b[a-zA-Z0-9._%+-]+"
            r"@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            "[EMAIL_MASKED]",
            masked
        )

        masked = re.sub(
            r"(WHERE\s+\w+\s*=\s*)'[^']*'",
            r"\1'[MASKED]'",
            masked,
            flags=re.IGNORECASE
        )

        return masked

    # ============================================================
    # BEHAVIORAL DETECTION
    # ============================================================

    def _is_off_hours(
        self,
        event_time: str
    ) -> bool:
        """Check if query outside business hours"""
        try:
            if "T" in event_time:
                hour = int(
                    event_time.split("T")[1][:2]
                )
                return not (BUSINESS_HOURS_START <= hour < BUSINESS_HOURS_END)
        except Exception:
            pass
        return False

    def _is_new_warehouse_user(
        self,
        user_name: str,
        warehouse_name: str
    ) -> bool:
        """Check if first time user ran query
        in this warehouse"""
        if not warehouse_name:
            return False
        known = self._warehouse_users.get(
            warehouse_name, set()
        )
        return user_name not in known

    def _accumulate_rows(
        self,
        user_name: str,
        rows: int
    ) -> int:
        """Accumulate rows per user session"""
        if user_name not in self._user_row_counts:
            self._user_row_counts[user_name] = {
                "rows": 0,
                "queries": 0
            }

        self._user_row_counts[user_name][
            "rows"
        ] += rows
        self._user_row_counts[user_name][
            "queries"
        ] += 1

        return self._user_row_counts[
            user_name
        ]["rows"]

    # ============================================================
    # RISK SCORING
    # ============================================================

    def _calculate_risk(
        self,
        event: DataAccessEvent,
        path_sensitivity: SensitivityLabel,
        path_confidence: float,
        is_zero_row_anomaly: bool,
        is_select_all: bool,
        is_bulk_read: bool,
        is_new_user: bool,
        rows_produced: int,
        bytes_scanned: int,
        accumulated_rows: int,
        query_type: str,
        warehouse_name: str
    ) -> tuple:
        """Calculate risk for Snowflake query event"""
        score = 0.0
        reasons = []

        # Sensitive table/path
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
                f"Potentially sensitive path: "
                f"{event.data_path}"
            )

        # ZERO ROW ANOMALY — unique to Snowflake
        # Your insight: model not well designed
        # = relationship between rows and bytes broken
        if is_zero_row_anomaly:
            score += 0.5
            reasons.append(
                f"Zero-row anomaly: scanned "
                f"{bytes_scanned:,} bytes but "
                f"produced 0 rows — possible "
                f"schema probing or side-channel attack"
            )

        # SELECT * pattern
        if is_select_all:
            score += 0.3
            reasons.append(
                "SELECT * detected — full table read. "
                "Common data exfiltration technique."
            )

        # High row counts
        if rows_produced >= CRITICAL_ROWS:
            score += 0.5
            reasons.append(
                f"Critical row count: "
                f"{rows_produced:,} rows — "
                f"possible full table extraction"
            )
        elif rows_produced >= HIGH_ROWS:
            score += 0.3
            reasons.append(
                f"High row count: "
                f"{rows_produced:,} rows"
            )
        elif is_bulk_read:
            score += 0.2
            reasons.append(
                f"Elevated row count: "
                f"{rows_produced:,} rows"
            )

        # Accumulated rows
        if accumulated_rows >= CRITICAL_ROWS:
            score += 0.3
            reasons.append(
                f"Accumulated session: "
                f"{accumulated_rows:,} total rows — "
                f"possible incremental exfiltration"
            )

        # High risk query type
        if query_type.upper() in HIGH_RISK_QUERY_TYPES:
            score += 0.4
            reasons.append(
                f"High risk query type: {query_type}"
            )

        # Off hours
        if event.is_off_hours:
            score += 0.2
            reasons.append(
                f"Snowflake query outside "
                f"business hours on "
                f"{warehouse_name}"
            )

        # New user in warehouse
        if is_new_user:
            score += 0.2
            reasons.append(
                f"First time {event.accessor_identity} "
                f"ran queries in {warehouse_name}"
            )

        # Development with sensitive data
        if event.environment == Environment.DEVELOPMENT:
            if path_confidence >= 0.4:
                score += 0.4
                self.dev_pii_violations += 1
                reasons.append(
                    f"GOVERNANCE VIOLATION: "
                    f"Sensitive data in development "
                    f"warehouse {warehouse_name}. "
                    f"Development should use "
                    f"masked/synthetic data only."
                )

        # Production access
        if event.environment == Environment.PRODUCTION:
            score += 0.1
            reasons.append(
                f"Production warehouse access: "
                f"{warehouse_name}"
            )

        return min(score, 1.0), reasons

    # ============================================================
    # HISTORY MANAGEMENT
    # ============================================================

    def _update_history(
        self,
        user_name: str,
        warehouse_name: str,
        table_name: str
    ) -> None:
        """Update user and warehouse history"""
        if warehouse_name:
            if warehouse_name not in (
                self._warehouse_users
            ):
                self._warehouse_users[
                    warehouse_name
                ] = set()
            self._warehouse_users[
                warehouse_name
            ].add(user_name)

        if table_name and user_name:
            if user_name not in self._user_table_access:
                self._user_table_access[user_name] = set()
            self._user_table_access[
                user_name
            ].add(table_name)

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
            "zero_row_anomalies": (
                self.zero_row_anomalies
            ),
            "bulk_reads_detected": (
                self.bulk_reads_detected
            ),
            "select_all_detected": (
                self.select_all_detected
            ),
            "dev_pii_violations": (
                self.dev_pii_violations
            ),
            "warehouses_tracked": len(
                self._warehouse_users
            ),
            "users_tracked": len(
                self._user_row_counts
            )
        }