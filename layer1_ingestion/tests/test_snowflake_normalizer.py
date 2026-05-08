"""
Snowflake Normalizer Tests

WHAT WE ARE PROVING:
    1. Warehouse context correctly parsed
    2. Zero-row anomaly detected
    3. Three-part path scoring works
    4. Accessor type uses warehouse context
    5. Development governance violation flagged
    6. Accumulated rows tracked
    7. SELECT * detection works
"""

import pytest
from layer1_ingestion.normalizers.snowflake_normalizer import (
    SnowflakeNormalizer
)
from layer1_ingestion.schema.data_schema import (
    DataAccessEvent,
    AccessorType,
    DataOperation,
    Environment
)


# ============================================================
# SAMPLE RAW SNOWFLAKE QUERY HISTORY EVENTS
# ============================================================

SF_NORMAL_SELECT = {
    "QUERY_ID": "sf-query-001",
    "START_TIME": "2024-03-29T15:00:00Z",
    "USER_NAME": "ANALYST_JOHN",
    "DATABASE_NAME": "PROD_DB",
    "SCHEMA_NAME": "PUBLIC",
    "TABLE_NAME": "CUSTOMERS",
    "QUERY_TEXT": (
        "SELECT id, name FROM CUSTOMERS LIMIT 100"
    ),
    "QUERY_TYPE": "SELECT",
    "ROWS_PRODUCED": 100,
    "BYTES_SCANNED": 1024000,
    "EXECUTION_STATUS": "SUCCESS",
    "WAREHOUSE_NAME": "PROD_REPORTING_WH"
}

SF_SELECT_ALL_BULK = {
    "QUERY_ID": "sf-query-002",
    "START_TIME": "2024-03-29T03:00:00Z",
    "USER_NAME": "SVC_REPORTING",
    "DATABASE_NAME": "PROD_DB",
    "SCHEMA_NAME": "PUBLIC",
    "TABLE_NAME": "CUSTOMER_PII",
    "QUERY_TEXT": (
        "SELECT * FROM PROD_DB.PUBLIC.CUSTOMER_PII"
    ),
    "QUERY_TYPE": "SELECT",
    "ROWS_PRODUCED": 1500000,
    "BYTES_SCANNED": 524288000,
    "EXECUTION_STATUS": "SUCCESS",
    "WAREHOUSE_NAME": "PROD_REPORTING_WH"
}

SF_ZERO_ROW_ANOMALY = {
    "QUERY_ID": "sf-query-003",
    "START_TIME": "2024-03-29T15:30:00Z",
    "USER_NAME": "UNKNOWN_USER",
    "DATABASE_NAME": "PROD_DB",
    "SCHEMA_NAME": "PUBLIC",
    "TABLE_NAME": "CUSTOMERS",
    "QUERY_TEXT": (
        "SELECT * FROM CUSTOMERS "
        "WHERE id = 99999999"
    ),
    "QUERY_TYPE": "SELECT",
    "ROWS_PRODUCED": 0,
    "BYTES_SCANNED": 524288000,
    "EXECUTION_STATUS": "SUCCESS",
    "WAREHOUSE_NAME": "PROD_REPORTING_WH"
}

SF_DEV_PII = {
    "QUERY_ID": "sf-query-004",
    "START_TIME": "2024-03-29T15:00:00Z",
    "USER_NAME": "DEV_JOHN",
    "DATABASE_NAME": "DEV_DB",
    "SCHEMA_NAME": "PUBLIC",
    "TABLE_NAME": "CUSTOMERS",
    "QUERY_TEXT": (
        "SELECT * FROM DEV_DB.PUBLIC.CUSTOMERS"
    ),
    "QUERY_TYPE": "SELECT",
    "ROWS_PRODUCED": 50000,
    "BYTES_SCANNED": 10240000,
    "EXECUTION_STATUS": "SUCCESS",
    "WAREHOUSE_NAME": "DEV_SANDBOX_WH"
}

SF_ETL_QUERY = {
    "QUERY_ID": "sf-query-005",
    "START_TIME": "2024-03-29T02:00:00Z",
    "USER_NAME": "ETL_PIPELINE_USER",
    "DATABASE_NAME": "PROD_DB",
    "SCHEMA_NAME": "FINANCE",
    "TABLE_NAME": "TRANSACTIONS",
    "QUERY_TEXT": (
        "SELECT account_id, amount "
        "FROM TRANSACTIONS "
        "WHERE date = '2024-03-28'"
    ),
    "QUERY_TYPE": "SELECT",
    "ROWS_PRODUCED": 250000,
    "BYTES_SCANNED": 52428800,
    "EXECUTION_STATUS": "SUCCESS",
    "WAREHOUSE_NAME": "PROD_ETL_WH"
}

SF_DROP_TABLE = {
    "QUERY_ID": "sf-query-006",
    "START_TIME": "2024-03-29T15:00:00Z",
    "USER_NAME": "SYSADMIN",
    "DATABASE_NAME": "PROD_DB",
    "SCHEMA_NAME": "PUBLIC",
    "TABLE_NAME": "OLD_CUSTOMERS",
    "QUERY_TEXT": "DROP TABLE OLD_CUSTOMERS",
    "QUERY_TYPE": "DROP",
    "ROWS_PRODUCED": 0,
    "BYTES_SCANNED": 0,
    "EXECUTION_STATUS": "SUCCESS",
    "WAREHOUSE_NAME": "PROD_REPORTING_WH"
}

SF_LOWERCASE = {
    "query_id": "sf-query-007",
    "start_time": "2024-03-29T15:00:00Z",
    "user_name": "analyst_jane",
    "database_name": "prod_db",
    "schema_name": "public",
    "table_name": "accounts",
    "query_text": "SELECT id FROM accounts LIMIT 10",
    "query_type": "SELECT",
    "rows_produced": 10,
    "bytes_scanned": 1024,
    "execution_status": "SUCCESS",
    "warehouse_name": "PROD_REPORTING_WH"
}


# ============================================================
# TEST CLASS — CORE NORMALIZATION
# ============================================================

class TestSnowflakeCoreNormalization:
    """Tests basic field extraction works"""

    def setup_method(self):
        self.normalizer = SnowflakeNormalizer()

    def test_event_normalized(self):
        """Snowflake event correctly normalized"""
        result = self.normalizer.normalize(
            SF_NORMAL_SELECT
        )
        assert result is not None
        assert isinstance(result, DataAccessEvent)
        assert result.source_system == "snowflake"

    def test_user_name_extracted(self):
        """User name correctly extracted"""
        result = self.normalizer.normalize(
            SF_NORMAL_SELECT
        )
        assert result.accessor_identity == "ANALYST_JOHN"

    def test_database_name_extracted(self):
        """Database name extracted"""
        result = self.normalizer.normalize(
            SF_NORMAL_SELECT
        )
        assert result.database_name == "PROD_DB"

    def test_rows_produced_extracted(self):
        """Rows produced correctly extracted"""
        result = self.normalizer.normalize(
            SF_NORMAL_SELECT
        )
        assert result.rows_accessed == 100

    def test_bytes_scanned_extracted(self):
        """Bytes scanned extracted"""
        result = self.normalizer.normalize(
            SF_NORMAL_SELECT
        )
        assert result.bytes_accessed == 1024000

    def test_three_part_path_built(self):
        """
        Three-part path DATABASE.SCHEMA.TABLE built.
        Consistent format for sensitivity scoring.
        """
        result = self.normalizer.normalize(
            SF_NORMAL_SELECT
        )
        assert "PROD_DB" in result.data_path
        assert "CUSTOMERS" in result.data_path

    def test_lowercase_fields_handled(self):
        """
        Lowercase field names handled.
        Snowflake sometimes returns lowercase.
        """
        result = self.normalizer.normalize(SF_LOWERCASE)
        assert result is not None
        assert result.rows_accessed == 10

    def test_none_returns_none(self):
        """None input handled gracefully"""
        assert self.normalizer.normalize(None) is None


# ============================================================
# TEST CLASS — WAREHOUSE CONTEXT
# ============================================================

class TestSnowflakeWarehouseContext:
    """
    Tests warehouse name parsing.

    Warehouse name = environment + workload.
    PROD_REPORTING_WH → production, reporting.
    DEV_SANDBOX_WH    → development, sandbox.
    """

    def setup_method(self):
        self.normalizer = SnowflakeNormalizer()

    def test_prod_warehouse_environment(self):
        """PROD_REPORTING_WH → PRODUCTION"""
        result = self.normalizer.normalize(
            SF_NORMAL_SELECT
        )
        assert result.environment == (
            Environment.PRODUCTION
        )

    def test_dev_warehouse_environment(self):
        """DEV_SANDBOX_WH → DEVELOPMENT"""
        result = self.normalizer.normalize(SF_DEV_PII)
        assert result.environment == (
            Environment.DEVELOPMENT
        )

    def test_etl_warehouse_accessor_type(self):
        """
        PROD_ETL_WH → ETL_PROCESS accessor type.
        Warehouse context adds to user name signal.
        """
        result = self.normalizer.normalize(SF_ETL_QUERY)
        assert result.accessor_type == (
            AccessorType.ETL_PROCESS
        )

    def test_warehouse_stored_as_data_store(self):
        """Warehouse name stored as data_store_name"""
        result = self.normalizer.normalize(
            SF_NORMAL_SELECT
        )
        assert result.data_store_name == (
            "PROD_REPORTING_WH"
        )


# ============================================================
# TEST CLASS — ZERO ROW ANOMALY
# ============================================================

class TestSnowflakeZeroRowAnomaly:
    """
    Tests for zero-row large scan detection.

    YOUR INSIGHT:
    rows=0 + bytes=500MB = relationship broken.
    Model not working as expected.
    Possible schema probing or side-channel attack.

    Normal queries: rows proportional to bytes.
    Anomalous:      0 rows + large bytes scanned.
    """

    def setup_method(self):
        self.normalizer = SnowflakeNormalizer()

    def test_zero_row_anomaly_detected(self):
        """
        0 rows + 500MB scanned flagged as anomaly.
        This is the unique Snowflake detection
        not possible in S3 or RDS.
        """
        result = self.normalizer.normalize(
            SF_ZERO_ROW_ANOMALY
        )
        reasons_text = " ".join(result.risk_reasons)
        assert (
            "zero" in reasons_text.lower() or
            "0 rows" in reasons_text.lower() or
            "anomaly" in reasons_text.lower()
        )

    def test_zero_row_anomaly_elevates_risk(self):
        """Zero-row anomaly significantly elevates risk"""
        result = self.normalizer.normalize(
            SF_ZERO_ROW_ANOMALY
        )
        assert result.risk_score >= 0.5

    def test_zero_row_counter_tracked(self):
        """Zero-row anomaly detections counted"""
        self.normalizer.normalize(SF_ZERO_ROW_ANOMALY)
        stats = self.normalizer.get_statistics()
        assert stats["zero_row_anomalies"] >= 1

    def test_normal_query_not_zero_row_anomaly(self):
        """
        Query returning 100 rows not flagged
        as zero-row anomaly.
        """
        result = self.normalizer.normalize(
            SF_NORMAL_SELECT
        )
        reasons_text = " ".join(result.risk_reasons)
        assert "zero-row" not in reasons_text.lower()


# ============================================================
# TEST CLASS — SELECT * DETECTION
# ============================================================

class TestSnowflakeSelectAll:
    """Tests for SELECT * detection"""

    def setup_method(self):
        self.normalizer = SnowflakeNormalizer()

    def test_select_all_detected(self):
        """SELECT * FROM detected"""
        result = self.normalizer.normalize(
            SF_SELECT_ALL_BULK
        )
        reasons_text = " ".join(result.risk_reasons)
        assert "SELECT *" in reasons_text

    def test_select_all_elevates_risk(self):
        """SELECT * elevates risk"""
        result = self.normalizer.normalize(
            SF_SELECT_ALL_BULK
        )
        assert result.risk_score >= 0.3

    def test_specific_select_not_flagged(self):
        """SELECT id, name does not trigger SELECT *"""
        result = self.normalizer.normalize(
            SF_NORMAL_SELECT
        )
        reasons_text = " ".join(result.risk_reasons)
        assert "SELECT *" not in reasons_text


# ============================================================
# TEST CLASS — GOVERNANCE AND ENVIRONMENT
# ============================================================

class TestSnowflakeGovernance:
    """
    Tests for governance violation detection.

    Development warehouse + sensitive table
    = governance violation always.
    Dev should use masked/synthetic data.
    """

    def setup_method(self):
        self.normalizer = SnowflakeNormalizer()

    def test_dev_pii_governance_flagged(self):
        """
        Sensitive table in DEV_SANDBOX_WH flagged.
        Development should never have real PII.
        This is a compliance finding regardless
        of time of day or who ran the query.
        """
        result = self.normalizer.normalize(SF_DEV_PII)
        reasons_text = " ".join(result.risk_reasons)
        assert (
            "governance" in reasons_text.lower() or
            "development" in reasons_text.lower() or
            "dev" in reasons_text.lower()
        )

    def test_dev_pii_elevates_risk(self):
        """Development PII access gets elevated risk"""
        result = self.normalizer.normalize(SF_DEV_PII)
        assert result.risk_score >= 0.4

    def test_dev_pii_violation_counter(self):
        """Development PII violations counted"""
        self.normalizer.normalize(SF_DEV_PII)
        stats = self.normalizer.get_statistics()
        assert stats["dev_pii_violations"] >= 1

    def test_drop_table_high_risk(self):
        """
        DROP TABLE gets high risk score.
        Destructive operation in production.
        Irreversible data loss risk.
        """
        result = self.normalizer.normalize(SF_DROP_TABLE)
        assert result.risk_score >= 0.3


# ============================================================
# TEST CLASS — ROW ACCUMULATION
# ============================================================

class TestSnowflakeRowAccumulation:
    """Tests accumulated row tracking"""

    def setup_method(self):
        self.normalizer = SnowflakeNormalizer()

    def test_rows_accumulate_per_user(self):
        """
        Rows accumulate per user across queries.
        Catches incremental exfiltration pattern.
        """
        self.normalizer.normalize(SF_NORMAL_SELECT)
        self.normalizer.normalize(SF_NORMAL_SELECT)

        accumulated = (
            self.normalizer._user_row_counts
            .get("ANALYST_JOHN", {})
            .get("rows", 0)
        )
        assert accumulated == 200

    def test_statistics_tracked(self):
        """Statistics correctly tracked"""
        self.normalizer.normalize(SF_NORMAL_SELECT)
        self.normalizer.normalize(SF_SELECT_ALL_BULK)
        stats = self.normalizer.get_statistics()
        assert stats["events_processed"] == 2