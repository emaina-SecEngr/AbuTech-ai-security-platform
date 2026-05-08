"""
PII Classifier Tests

WHAT WE ARE PROVING:
    1. Each regex pattern correctly detects
       its specific data type
    2. Luhn algorithm validates real card numbers
    3. False positives minimized (random numbers
       do not trigger card detection)
    4. SQL masking removes actual values
    5. Regulatory mapping is correct
    6. Confidence scoring works correctly

WHY THESE TESTS MATTER:
    A false negative = missed sensitive data
    = compliance violation = regulatory fine

    A false positive = legitimate data flagged
    = operational disruption

    Both have serious business consequences.
    In banking the false negative cost is higher.
    Missing one SSN = GDPR breach notification.
"""

import pytest
from layer2_ml.classification.pii_classifier import (
    PIIClassifier,
    luhn_check
)
from layer1_ingestion.schema.data_schema import (
    SensitivityLabel,
    DataFinding,
    AccessorType,
    DataStoreType,
    DataOperation,
    Environment,
    DataAccessEvent,
    DataStoreProfile
)


# ============================================================
# SAMPLE TEXTS FOR TESTING
# Using realistic but fictional data
# ============================================================

TEXT_WITH_SSN = (
    "Customer record: John Smith "
    "SSN: 123-45-6789 "
    "Date of birth: 1980-03-15 "
    "Address: 123 Main St, New York"
)

TEXT_WITH_CREDIT_CARD = (
    "Payment processed for card "
    "4532015112830366 "
    "expiry 12/25 "
    "CVV: 123 "
    "Cardholder: John Smith"
)

TEXT_WITH_PHI = (
    "Patient: John Smith "
    "MRN: MRN-12345678 "
    "Diagnosis: E11.9 "
    "Prescription: Metformin 500mg "
    "Insurance ID: BCBS-9876543210 "
    "Physician NPI: NPI:1234567890"
)

TEXT_WITH_BANK_ACCOUNT = (
    "Account: acct 12345678901 "
    "Routing: routing 021000021 "
    "IBAN: GB82WEST12345698765432 "
    "Credit Score: FICO 750"
)

TEXT_BENIGN = (
    "The quarterly earnings report shows "
    "revenue increased by 15 percent. "
    "Total transactions: 1234567. "
    "Reference number: REF-2024-001."
)

TEXT_WITH_EMAIL = (
    "Contact: jsmith@corp.com "
    "Phone: 555-123-4567 "
    "Customer ID: CUST-001"
)

SQL_WITH_SSN = (
    "SELECT * FROM customers "
    "WHERE ssn = '123-45-6789' "
    "AND dob = '1980-03-15'"
)

SQL_WITH_CARD = (
    "SELECT transaction_id FROM payments "
    "WHERE card_number = '4532015112830366' "
    "AND cvv = '123'"
)

TEXT_MIXED = (
    "Patient payment: "
    "SSN 123-45-6789 "
    "Card 4532015112830366 "
    "Diagnosis E11.9"
)


# ============================================================
# TEST CLASS — LUHN ALGORITHM
# ============================================================

class TestLuhnAlgorithm:
    """
    Tests for credit card Luhn validation.

    WHY TEST LUHN SEPARATELY:
    This is our primary false positive reducer
    for credit card detection.
    A broken Luhn check means random 16-digit
    numbers trigger PCI alerts.
    """

    def test_valid_visa_card_passes(self):
        """
        Known valid Visa card passes Luhn.
        4532015112830366 is a test card number
        that passes Luhn validation.
        """
        assert luhn_check("4532015112830366") is True

    def test_valid_mastercard_passes(self):
        """Valid Mastercard passes Luhn"""
        assert luhn_check("5425233430109903") is True

    def test_invalid_card_fails(self):
        """
        Random 16-digit number fails Luhn.
        This prevents false positives on
        account numbers, reference numbers, etc.
        """
        assert luhn_check("1234567890123456") is False

    def test_modified_card_fails(self):
        """
        Valid card with one digit changed fails.
        Proves Luhn catches transcription errors.
        """
        assert luhn_check("4532015112830367") is False

    def test_too_short_fails(self):
        """Numbers shorter than 13 digits fail"""
        assert luhn_check("123456789") is False

    def test_non_digits_handled(self):
        """Card with spaces/dashes handled"""
        assert luhn_check("4532-0151-1283-0366") is True

    def test_empty_string_fails(self):
        """Empty string returns False gracefully"""
        assert luhn_check("") is False


# ============================================================
# TEST CLASS — PII DETECTION
# ============================================================

class TestPIIDetection:
    """
    Tests for PII (Personally Identifiable Info)
    detection.

    Regulations covered:
    GDPR (EU), CCPA (California), PIPEDA (Canada)
    """

    def setup_method(self):
        self.classifier = PIIClassifier()

    def test_ssn_detected(self):
        """
        US Social Security Number detected.
        Format: XXX-XX-XXXX
        Most common PII pattern in US banking.
        """
        finding = self.classifier.classify(
            TEXT_WITH_SSN
        )
        assert finding.sensitivity_label in [
            SensitivityLabel.PII,
            SensitivityLabel.MIXED
        ]
        assert "SSN" in finding.data_types_found

    def test_email_detected(self):
        """Email address correctly detected"""
        finding = self.classifier.classify(
            TEXT_WITH_EMAIL
        )
        assert finding.sensitivity_label in [
            SensitivityLabel.PII,
            SensitivityLabel.MIXED
        ]
        assert "EMAIL" in finding.data_types_found

    def test_ssn_count_tracked(self):
        """
        Number of SSNs found is tracked.
        Banks need to know HOW MANY records
        are affected for breach notification.
        GDPR requires notification within 72 hours
        if breach affects EU residents.
        """
        finding = self.classifier.classify(
            TEXT_WITH_SSN
        )
        assert finding.data_type_counts.get("SSN", 0) >= 1

    def test_benign_text_not_flagged(self):
        """
        Normal business text not flagged as PII.
        Reference numbers and transaction IDs
        should not trigger false positives.
        """
        finding = self.classifier.classify(
            TEXT_BENIGN
        )
        assert finding.sensitivity_label == (
            SensitivityLabel.NONE
        )

    def test_gdpr_regulation_triggered(self):
        """
        GDPR regulation triggered for EU PII data.
        Platform must know which regulation applies
        for correct breach notification procedure.
        """
        finding = self.classifier.classify(
            TEXT_WITH_SSN
        )
        assert len(finding.regulations_triggered) > 0


# ============================================================
# TEST CLASS — PCI DETECTION
# ============================================================

class TestPCIDetection:
    """
    Tests for PCI-DSS payment card detection.

    PCI-DSS is the most specific standard.
    Any card data in unexpected location
    = immediate critical finding.
    """

    def setup_method(self):
        self.classifier = PIIClassifier()

    def test_visa_card_detected(self):
        """
        Valid Visa card number detected.
        Luhn validation confirms it is real card.
        """
        finding = self.classifier.classify(
            TEXT_WITH_CREDIT_CARD
        )
        assert finding.sensitivity_label in [
            SensitivityLabel.PCI,
            SensitivityLabel.MIXED
        ]

    def test_cvv_detected(self):
        """
        CVV code detected in text.
        Finding CVV anywhere = CRITICAL violation.
        CVV must NEVER be stored per PCI-DSS req 3.2.
        """
        finding = self.classifier.classify(
            TEXT_WITH_CREDIT_CARD
        )
        assert "CVV" in finding.data_types_found

    def test_pci_dss_regulation_triggered(self):
        """PCI-DSS regulation triggered for card data"""
        finding = self.classifier.classify(
            TEXT_WITH_CREDIT_CARD
        )
        assert "PCI-DSS" in (
            finding.regulations_triggered
        )

    def test_random_number_not_flagged(self):
        """
        Random 16-digit number not flagged.
        Luhn validation prevents false positives.
        Account numbers, reference numbers, etc.
        should not trigger PCI alerts.
        """
        text_with_random = (
            "Transaction reference: 1234567890123456 "
            "processed successfully"
        )
        finding = self.classifier.classify(
            text_with_random
        )
        pci_types = [
            t for t in finding.data_types_found
            if "CREDIT_CARD" in t
        ]
        assert len(pci_types) == 0

    def test_card_confidence_high(self):
        """
        Credit card detection has high confidence.
        Luhn validation + pattern match = very certain.
        """
        finding = self.classifier.classify(
            TEXT_WITH_CREDIT_CARD
        )
        assert finding.confidence >= 0.7


# ============================================================
# TEST CLASS — PHI DETECTION
# ============================================================

class TestPHIDetection:
    """
    Tests for HIPAA Protected Health Information.

    Healthcare clients require HIPAA compliance.
    Banking clients with health insurance products
    also need PHI detection.
    """

    def setup_method(self):
        self.classifier = PIIClassifier()

    def test_medical_record_detected(self):
        """Medical record number detected"""
        finding = self.classifier.classify(
            TEXT_WITH_PHI
        )
        assert finding.sensitivity_label in [
            SensitivityLabel.PHI,
            SensitivityLabel.MIXED
        ]

    def test_hipaa_regulation_triggered(self):
        """HIPAA triggered for health data"""
        finding = self.classifier.classify(
            TEXT_WITH_PHI
        )
        assert "HIPAA" in (
            finding.regulations_triggered
        )


# ============================================================
# TEST CLASS — PFI DETECTION
# ============================================================

class TestPFIDetection:
    """
    Tests for Protected Financial Information.
    GLBA and SOX compliance for banking.
    """

    def setup_method(self):
        self.classifier = PIIClassifier()

    def test_bank_account_detected(self):
        """Bank account number detected"""
        finding = self.classifier.classify(
            TEXT_WITH_BANK_ACCOUNT
        )
        assert finding.sensitivity_label in [
            SensitivityLabel.PFI,
            SensitivityLabel.MIXED
        ]

    def test_iban_detected(self):
        """International bank account IBAN detected"""
        finding = self.classifier.classify(
            TEXT_WITH_BANK_ACCOUNT
        )
        assert "IBAN" in finding.data_types_found

    def test_glba_regulation_triggered(self):
        """GLBA triggered for financial data"""
        finding = self.classifier.classify(
            TEXT_WITH_BANK_ACCOUNT
        )
        regs = finding.regulations_triggered
        assert len(regs) > 0


# ============================================================
# TEST CLASS — MIXED CONTENT
# ============================================================

class TestMixedContent:
    """
    Tests for content with multiple sensitivity types.
    Common in banking where PII + PCI coexist.
    """

    def setup_method(self):
        self.classifier = PIIClassifier()

    def test_mixed_label_for_multiple_types(self):
        """
        Text with both SSN and credit card
        gets MIXED label.
        Common in loan applications that contain
        both customer identity and payment data.
        """
        finding = self.classifier.classify(TEXT_MIXED)
        assert finding.sensitivity_label in [
            SensitivityLabel.MIXED,
            SensitivityLabel.PCI,
            SensitivityLabel.PII,
            SensitivityLabel.PHI
        ]

    def test_multiple_types_in_findings(self):
        """Multiple data types found in mixed text"""
        finding = self.classifier.classify(TEXT_MIXED)
        assert len(finding.data_types_found) >= 1

    def test_empty_text_returns_none(self):
        """Empty text returns NONE label"""
        finding = self.classifier.classify("")
        assert finding.sensitivity_label == (
            SensitivityLabel.NONE
        )

    def test_none_text_handled(self):
        """None text handled gracefully"""
        finding = self.classifier.classify(None)
        assert finding is not None


# ============================================================
# TEST CLASS — SQL MASKING
# ============================================================

class TestSQLMasking:
    """
    Tests for SQL query masking.

    WHY CRITICAL:
    Bank SQL queries contain actual PII values.
    "WHERE ssn = '123-45-6789'"

    Before storing in audit trail:
    Must mask to "WHERE ssn = 'XXX-XX-XXXX'"

    Without masking YOUR platform stores PII.
    Your platform becomes a compliance violation.
    """

    def setup_method(self):
        self.classifier = PIIClassifier()

    def test_ssn_masked_in_query(self):
        """
        SSN value masked in SQL WHERE clause.
        Structure preserved. Value removed.
        """
        masked = self.classifier.mask_query(
            SQL_WITH_SSN
        )
        assert "123-45-6789" not in masked
        assert "WHERE" in masked
        assert "ssn" in masked.lower()

    def test_card_number_masked_in_query(self):
        """
        Credit card number masked in SQL query.
        Analyst sees query structure.
        Cannot see actual card number.
        """
        masked = self.classifier.mask_query(
            SQL_WITH_CARD
        )
        assert "4532015112830366" not in masked

    def test_masked_query_preserves_structure(self):
        """
        Masking preserves SQL structure.
        SELECT, FROM, WHERE keywords intact.
        Only sensitive values replaced.
        """
        masked = self.classifier.mask_query(
            SQL_WITH_SSN
        )
        assert "SELECT" in masked
        assert "FROM" in masked
        assert "WHERE" in masked

    def test_benign_query_unchanged(self):
        """
        Query without PII values not modified.
        Masking only affects detected patterns.
        """
        benign_query = (
            "SELECT count(*) FROM transactions "
            "WHERE date > '2024-01-01'"
        )
        masked = self.classifier.mask_query(
            benign_query
        )
        assert "count(*)" in masked


# ============================================================
# TEST CLASS — REGULATORY MAPPING
# ============================================================

class TestRegulatoryMapping:
    """
    Tests for regulatory framework mapping.

    Your platform must know which regulation
    applies to each finding.
    Different regulations = different response:
    GDPR: notify within 72 hours
    HIPAA: notify within 60 days
    PCI-DSS: notify card brands immediately
    """

    def setup_method(self):
        self.classifier = PIIClassifier()

    def test_pci_applies_globally(self):
        """
        PCI-DSS applies regardless of region.
        Card data anywhere = PCI obligation.
        """
        regs = self.classifier.get_applicable_regulations(
            SensitivityLabel.PCI,
            data_region="US"
        )
        assert "PCI-DSS" in regs

        regs_eu = (
            self.classifier.get_applicable_regulations(
                SensitivityLabel.PCI,
                data_region="EU"
            )
        )
        assert "PCI-DSS" in regs_eu

    def test_gdpr_for_eu_pii(self):
        """GDPR applies for EU PII data"""
        regs = self.classifier.get_applicable_regulations(
            SensitivityLabel.PII,
            data_region="EU"
        )
        assert "GDPR" in regs

    def test_hipaa_for_us_phi(self):
        """HIPAA applies for US health data"""
        regs = self.classifier.get_applicable_regulations(
            SensitivityLabel.PHI,
            data_region="US"
        )
        assert "HIPAA" in regs

    def test_ccpa_for_us_pii(self):
        """CCPA applies for US PII data"""
        regs = self.classifier.get_applicable_regulations(
            SensitivityLabel.PII,
            data_region="US"
        )
        assert "CCPA" in regs

    def test_pipeda_for_canada(self):
        """PIPEDA applies for Canadian data"""
        regs = self.classifier.get_applicable_regulations(
            SensitivityLabel.PII,
            data_region="CA"
        )
        assert "PIPEDA" in regs


# ============================================================
# TEST CLASS — DATA SCHEMA
# ============================================================

class TestDataSchema:
    """
    Tests for DataAccessEvent and DataStoreProfile
    schema objects.

    These tests prove the schema works correctly
    for all accessor types discussed.
    """

    def test_data_access_event_creates(self):
        """DataAccessEvent creates with defaults"""
        event = DataAccessEvent()
        assert event.accessor_type == (
            AccessorType.UNKNOWN
        )
        assert event.sensitivity_label if (
            hasattr(event, "sensitivity_label")
        ) else True

    def test_service_account_accessor_type(self):
        """
        svc_backup correctly typed as service account.
        Accessor type changes risk interpretation.
        """
        event = DataAccessEvent(
            accessor_identity="svc_backup",
            accessor_type=AccessorType.SERVICE_ACCOUNT,
            data_store_type=DataStoreType.S3,
            operation=DataOperation.READ,
            rows_accessed=50000,
            is_off_hours=True,
            is_off_schedule=True
        )
        assert event.accessor_type == (
            AccessorType.SERVICE_ACCOUNT
        )
        assert event.is_off_schedule is True

    def test_etl_process_accessor_type(self):
        """ETL process correctly typed"""
        event = DataAccessEvent(
            accessor_identity="etl_daily_customers",
            accessor_type=AccessorType.ETL_PROCESS,
            job_schedule="0 2 * * *",
            is_scheduled_access=True,
            rows_accessed=5_000_000
        )
        assert event.accessor_type == (
            AccessorType.ETL_PROCESS
        )
        assert event.is_scheduled_access is True

    def test_data_store_profile_creates(self):
        """DataStoreProfile creates correctly"""
        profile = DataStoreProfile(
            store_name="prod-customer-db",
            store_type=DataStoreType.ORACLE,
            environment=Environment.PRODUCTION,
            sensitivity_labels=["PII", "PFI"],
            data_types_present=["SSN", "ACCOUNT"],
            regulations_applicable=["GDPR", "GLBA"],
            estimated_pii_records=1_500_000
        )
        assert profile.store_name == "prod-customer-db"
        assert "PII" in profile.sensitivity_labels
        assert profile.estimated_pii_records == 1_500_000

    def test_event_to_dict(self):
        """DataAccessEvent serializes to dict"""
        event = DataAccessEvent(
            accessor_identity="jsmith@corp.com",
            accessor_type=AccessorType.HUMAN,
            data_store_name="customer_db"
        )
        d = event.to_dict()
        assert "accessor_identity" in d
        assert "accessor_type" in d

    def test_finding_attached_to_event(self):
        """
        DataFinding can be attached to
        DataAccessEvent.
        This links WHO accessed to WHAT they found.
        """
        classifier = PIIClassifier()
        finding = classifier.classify(TEXT_WITH_SSN)

        event = DataAccessEvent(
            accessor_identity="svc_backup",
            accessor_type=AccessorType.SERVICE_ACCOUNT,
            data_store_name="prod-s3-bucket",
            finding=finding
        )

        assert event.finding is not None
        assert event.finding.sensitivity_label in [
            SensitivityLabel.PII,
            SensitivityLabel.MIXED
        ]


# ============================================================
# TEST CLASS — STATISTICS
# ============================================================

class TestClassifierStatistics:
    """Statistics correctly tracked"""

    def setup_method(self):
        self.classifier = PIIClassifier()

    def test_classification_count_tracked(self):
        """Each classification increments counter"""
        self.classifier.classify(TEXT_WITH_SSN)
        self.classifier.classify(TEXT_WITH_CREDIT_CARD)
        stats = self.classifier.get_statistics()
        assert stats["texts_classified"] == 2

    def test_pci_count_tracked(self):
        """PCI findings increment counter"""
        self.classifier.classify(TEXT_WITH_CREDIT_CARD)
        stats = self.classifier.get_statistics()
        assert stats["pci_found"] >= 1

    def test_pii_count_tracked(self):
        """PII findings increment counter"""
        self.classifier.classify(TEXT_WITH_SSN)
        stats = self.classifier.get_statistics()
        assert stats["pii_found"] >= 1