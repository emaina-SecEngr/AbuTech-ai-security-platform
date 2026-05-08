"""
RDS Normalizer Tests

WHAT WE ARE PROVING:
    1. Core field extraction works
    2. SQL command maps to DataOperation
    3. Row count thresholds trigger correctly
    4. SELECT * detection works
    5. Accumulated rows track across queries
    6. Development environment governance detection
    7. Table sensitivity scoring works
    8. Query masking removes PII before storage
"""

import pytest
from layer1_ingestion.normalizers.rds_normalizer import (
    RDSNormalizer
)
from layer1_ingestion.schema.data_schema import (
    DataAccessEvent,
    AccessorType,
    DataOperation,
    Environment
)


# ============================================================
# SAMPLE RAW RDS EVENTS
# ============================================================

RDS_SELECT_NORMAL = {
    "databaseActivityEventList": [{
        "type": "record",
        "startTime": "2024-03-29T15:00:00Z",
        "dbUserName": "analyst_john",
        "remoteHost": "10.0.0.155",
        "command": "SELECT",
        "commandText": (
            "SELECT id, name FROM customers "
            "WHERE region = 'NY' LIMIT 100"
        ),
        "objectName": "customers",
        "databaseName": "prod_db",
        "schemaName": "public",
        "rowCount": 100
    }],
    "dbInstanceIdentifier": "prod-customer-db"
}

RDS_SELECT_ALL = {
    "databaseActivityEventList": [{
        "type": "record",
        "startTime": "2024-03-29T03:00:00Z",
        "dbUserName": "svc_reporting",
        "remoteHost": "10.0.1.50",
        "command": "SELECT",
        "commandText": (
            "SELECT * FROM customers_pii"
        ),
        "objectName": "customers_pii",
        "databaseName": "prod_db",
        "schemaName": "public",
        "rowCount": 1500000
    }],
    "dbInstanceIdentifier": "prod-customer-db"
}

RDS_BULK_READ = {
    "databaseActivityEventList": [{
        "type": "record",
        "startTime": "2024-03-29T15:00:00Z",
        "dbUserName": "etl_daily_pipeline",
        "remoteHost": "10.0.1.100",
        "command": "SELECT",
        "commandText": (
            "SELECT * FROM transactions "
            "WHERE date = '2024-03-28'"
        ),
        "objectName": "transactions",
        "databaseName": "prod_db",
        "schemaName": "public",
        "rowCount": 500000
    }],
    "dbInstanceIdentifier": "prod-transaction-db"
}

RDS_DELETE = {
    "databaseActivityEventList": [{
        "type": "record",
        "startTime": "2024-03-29T15:00:00Z",
        "dbUserName": "dba_admin",
        "remoteHost": "10.0.0.10",
        "command": "DROP",
        "commandText": "DROP TABLE old_customers",
        "objectName": "old_customers",
        "databaseName": "prod_db",
        "schemaName": "public",
        "rowCount": 0
    }],
    "dbInstanceIdentifier": "prod-customer-db"
}

RDS_PII_IN_QUERY = {
    "databaseActivityEventList": [{
        "type": "record",
        "startTime": "2024-03-29T15:00:00Z",
        "dbUserName": "analyst_john",
        "remoteHost": "10.0.0.155",
        "command": "SELECT",
        "commandText": (
            "SELECT * FROM customers "
            "WHERE ssn = '123-45-6789' "
            "AND card_number = '4532015112830366'"
        ),
        "objectName": "customers",
        "databaseName": "prod_db",
        "schemaName": "public",
        "rowCount": 1
    }],
    "dbInstanceIdentifier": "prod-customer-db"
}

RDS_DEV_PII = {
    "databaseActivityEventList": [{
        "type": "record",
        "startTime": "2024-03-29T15:00:00Z",
        "dbUserName": "dev_john",
        "remoteHost": "10.0.0.200",
        "command": "SELECT",
        "commandText": (
            "SELECT * FROM customers"
        ),
        "objectName": "customers",
        "databaseName": "dev_db",
        "schemaName": "public",
        "rowCount": 50000
    }],
    "dbInstanceIdentifier": "dev-customer-db"
}

RDS_ETL_ACCESS = {
    "databaseActivityEventList": [{
        "type": "record",
        "startTime": "2024-03-29T02:00:00Z",
        "dbUserName": "etl_batch_job",
        "remoteHost": "10.0.1.100",
        "command": "SELECT",
        "commandText": (
            "SELECT account_id, balance "
            "FROM accounts "
            "WHERE date = '2024-03-28'"
        ),
        "objectName": "accounts",
        "databaseName": "prod_db",
        "schemaName": "finance",
        "rowCount": 250000
    }],
    "dbInstanceIdentifier": "prod-finance-db"
}


# ============================================================
# TEST CLASS — CORE NORMALIZATION
# ============================================================

class TestRDSCoreNormalization:
    """Tests that basic field extraction works"""

    def setup_method(self):
        self.normalizer = RDSNormalizer()

    def test_event_normalized(self):
        """RDS event correctly normalized"""
        result = self.normalizer.normalize(
            RDS_SELECT_NORMAL
        )
        assert result is not None
        assert isinstance(result, DataAccessEvent)
        assert result.source_system == "aws_rds"

    def test_database_name_extracted(self):
        """Database name correctly extracted"""
        result = self.normalizer.normalize(
            RDS_SELECT_NORMAL
        )
        assert result.database_name == "prod_db"

    def test_table_name_extracted(self):
        """Table name correctly extracted"""
        result = self.normalizer.normalize(
            RDS_SELECT_NORMAL
        )
        assert result.table_name == "customers"

    def test_user_name_extracted(self):
        """Database user correctly extracted"""
        result = self.normalizer.normalize(
            RDS_SELECT_NORMAL
        )
        assert result.accessor_identity == "analyst_john"

    def test_row_count_extracted(self):
        """Row count correctly extracted"""
        result = self.normalizer.normalize(
            RDS_SELECT_NORMAL
        )
        assert result.rows_accessed == 100

    def test_select_maps_to_read(self):
        """SELECT command maps to READ operation"""
        result = self.normalizer.normalize(
            RDS_SELECT_NORMAL
        )
        assert result.operation == DataOperation.READ

    def test_drop_maps_to_delete(self):
        """DROP command maps to DELETE operation"""
        result = self.normalizer.normalize(RDS_DELETE)
        assert result.operation == DataOperation.DELETE

    def test_data_path_built(self):
        """Data path built as database.schema.table"""
        result = self.normalizer.normalize(
            RDS_SELECT_NORMAL
        )
        assert "prod_db" in result.data_path
        assert "customers" in result.data_path

    def test_none_returns_none(self):
        """None input handled gracefully"""
        assert self.normalizer.normalize(None) is None


# ============================================================
# TEST CLASS — ACCESSOR TYPE
# ============================================================

class TestRDSAccessorType:
    """Tests for database user type mapping"""

    def setup_method(self):
        self.normalizer = RDSNormalizer()

    def test_etl_user_mapped(self):
        """
        etl_batch_job mapped to ETL_PROCESS.
        ETL processes have scheduled access patterns.
        Off-schedule = governance flag.
        """
        result = self.normalizer.normalize(
            RDS_ETL_ACCESS
        )
        assert result.accessor_type == (
            AccessorType.ETL_PROCESS
        )

    def test_dba_mapped_to_privileged(self):
        """dba_admin mapped to PRIVILEGED"""
        result = self.normalizer.normalize(RDS_DELETE)
        assert result.accessor_type == (
            AccessorType.PRIVILEGED
        )

    def test_human_analyst_mapped(self):
        """Regular analyst mapped to HUMAN"""
        result = self.normalizer.normalize(
            RDS_SELECT_NORMAL
        )
        assert result.accessor_type == (
            AccessorType.HUMAN
        )


# ============================================================
# TEST CLASS — ROW COUNT DETECTION
# ============================================================

class TestRDSRowCountDetection:
    """
    Tests for row-based exfiltration detection.

    WHY ROWS NOT BYTES:
    1 row = 1 customer record.
    Row count maps directly to breach scope.
    GDPR asks: how many data subjects affected?
    Row count answers that question directly.
    """

    def setup_method(self):
        self.normalizer = RDSNormalizer()

    def test_bulk_read_elevates_risk(self):
        """
        500K rows elevates risk score.
        Bulk reads are primary exfiltration signal.
        """
        result = self.normalizer.normalize(RDS_BULK_READ)
        assert result.risk_score >= 0.3

    def test_critical_row_count_critical_risk(self):
        """
        1.5M rows = critical risk.
        This is the complete customer table.
        GDPR breach notification required.
        """
        result = self.normalizer.normalize(RDS_SELECT_ALL)
        assert result.risk_score >= 0.6

    def test_normal_row_count_low_risk(self):
        """
        100 rows = low risk.
        Normal analyst query pattern.
        """
        result = self.normalizer.normalize(
            RDS_SELECT_NORMAL
        )
        assert result.risk_score < 0.6

    def test_accumulated_rows_tracked(self):
        """
        Rows accumulate across multiple queries.
        Incremental exfiltration detection.
        1000 queries × 1000 rows = 1M rows flagged.
        """
        small_event = {
            "databaseActivityEventList": [{
                "startTime": "2024-03-29T15:00:00Z",
                "dbUserName": "analyst_john",
                "remoteHost": "10.0.0.155",
                "command": "SELECT",
                "commandText": "SELECT * FROM customers",
                "objectName": "customers",
                "databaseName": "prod_db",
                "schemaName": "public",
                "rowCount": 500000
            }],
            "dbInstanceIdentifier": "prod-db"
        }

        self.normalizer.normalize(small_event)
        self.normalizer.normalize(small_event)

        stats = self.normalizer.get_statistics()
        accumulated = (
            self.normalizer._accessor_row_counts
            .get("analyst_john", {})
            .get("rows", 0)
        )
        assert accumulated == 1000000


# ============================================================
# TEST CLASS — SELECT * DETECTION
# ============================================================

class TestRDSSelectAllDetection:
    """
    Tests for SELECT * detection.

    SELECT * reads ALL columns.
    Attacker gets sensitive columns
    they did not specifically request.
    Principle of least privilege violation.
    """

    def setup_method(self):
        self.normalizer = RDSNormalizer()

    def test_select_all_detected(self):
        """SELECT * FROM detected in query"""
        result = self.normalizer.normalize(RDS_SELECT_ALL)
        reasons_text = " ".join(result.risk_reasons)
        assert "SELECT *" in reasons_text

    def test_select_all_elevates_risk(self):
        """SELECT * elevates risk score"""
        result = self.normalizer.normalize(RDS_SELECT_ALL)
        assert result.risk_score >= 0.3

    def test_specific_select_not_flagged(self):
        """SELECT id, name does not trigger SELECT *"""
        result = self.normalizer.normalize(
            RDS_SELECT_NORMAL
        )
        reasons_text = " ".join(result.risk_reasons)
        assert "SELECT *" not in reasons_text

    def test_select_all_counter_tracked(self):
        """SELECT * detections counted"""
        self.normalizer.normalize(RDS_SELECT_ALL)
        stats = self.normalizer.get_statistics()
        assert stats["select_all_detected"] >= 1


# ============================================================
# TEST CLASS — QUERY MASKING
# ============================================================

class TestRDSQueryMasking:
    """
    Tests for SQL query PII masking.

    Bank queries contain actual PII:
    WHERE ssn = '123-45-6789'
    Must mask before storing in audit trail.
    """

    def setup_method(self):
        self.normalizer = RDSNormalizer()

    def test_ssn_masked_in_stored_query(self):
        """
        SSN value masked in stored query text.
        Actual SSN never stored in audit trail.
        GDPR Article 25 compliance.
        """
        result = self.normalizer.normalize(
            RDS_PII_IN_QUERY
        )
        assert "123-45-6789" not in result.query_text

    def test_card_masked_in_stored_query(self):
        """Credit card masked in stored query"""
        result = self.normalizer.normalize(
            RDS_PII_IN_QUERY
        )
        assert "4532015112830366" not in result.query_text

    def test_masked_query_preserves_structure(self):
        """SQL structure preserved after masking"""
        result = self.normalizer.normalize(
            RDS_PII_IN_QUERY
        )
        assert "SELECT" in result.query_text
        assert "FROM" in result.query_text


# ============================================================
# TEST CLASS — ENVIRONMENT AND GOVERNANCE
# ============================================================

class TestRDSEnvironmentGovernance:
    """
    Tests for environment detection and governance.

    Finding PII in development = governance violation.
    Dev should NEVER have real customer data.
    """

    def setup_method(self):
        self.normalizer = RDSNormalizer()

    def test_prod_environment_detected(self):
        """prod-customer-db → PRODUCTION"""
        result = self.normalizer.normalize(
            RDS_SELECT_NORMAL
        )
        assert result.environment == (
            Environment.PRODUCTION
        )

    def test_dev_environment_detected(self):
        """dev-customer-db → DEVELOPMENT"""
        result = self.normalizer.normalize(RDS_DEV_PII)
        assert result.environment == (
            Environment.DEVELOPMENT
        )

    def test_dev_pii_governance_violation(self):
        """
        Sensitive table in dev environment flagged
        as governance violation.
        Dev should use masked data only.
        """
        result = self.normalizer.normalize(RDS_DEV_PII)
        reasons_text = " ".join(result.risk_reasons)
        assert (
            "governance" in reasons_text.lower() or
            "development" in reasons_text.lower()
        )

    def test_statistics_tracked(self):
        """Statistics correctly tracked"""
        self.normalizer.normalize(RDS_SELECT_NORMAL)
        self.normalizer.normalize(RDS_SELECT_ALL)
        stats = self.normalizer.get_statistics()
        assert stats["events_processed"] == 2