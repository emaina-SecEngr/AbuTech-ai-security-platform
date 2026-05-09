"""
SharePoint Normalizer Tests

WHAT WE ARE PROVING:
    1. Site name correctly extracted from URL
    2. Site risk profile lookup works
    3. Filename sensitivity scoring works
    4. Bulk download detection works
    5. Anonymous link detection works
    6. KYC site access tracked
    7. Unknown site flagged as shadow IT
"""

import pytest
from layer1_ingestion.normalizers.sharepoint_normalizer import (
    SharePointNormalizer
)
from layer1_ingestion.schema.data_schema import (
    DataAccessEvent,
    AccessorType,
    DataOperation,
    Environment
)


# ============================================================
# SAMPLE RAW SHAREPOINT AUDIT EVENTS
# ============================================================

SP_KYC_DOWNLOAD = {
    "CreationTime": "2024-03-29T15:00:00Z",
    "Id": "sp-event-001",
    "Operation": "FileDownloaded",
    "Workload": "SharePoint",
    "UserId": "jsmith@corp.com",
    "ClientIP": "10.0.0.155",
    "SiteUrl": (
        "https://corp.sharepoint.com/sites/KYC"
    ),
    "SourceFileName": "passport_john_smith.pdf",
    "SourceRelativeUrl": "KYC_Documents/2024/Pending",
    "ItemType": "File",
    "FileSizeBytes": 2048576
}

SP_FINANCE_ACCESS = {
    "CreationTime": "2024-03-29T15:00:00Z",
    "Id": "sp-event-002",
    "Operation": "FileAccessed",
    "Workload": "SharePoint",
    "UserId": "analyst@corp.com",
    "ClientIP": "10.0.0.200",
    "SiteUrl": (
        "https://corp.sharepoint.com/sites/Finance"
    ),
    "SourceFileName": "Q1_customer_report.xlsx",
    "SourceRelativeUrl": "Reports/2024",
    "ItemType": "File",
    "FileSizeBytes": 5120000
}

SP_ANONYMOUS_LINK = {
    "CreationTime": "2024-03-29T15:00:00Z",
    "Id": "sp-event-003",
    "Operation": "AnonymousLinkCreated",
    "Workload": "SharePoint",
    "UserId": "jsmith@corp.com",
    "ClientIP": "10.0.0.155",
    "SiteUrl": (
        "https://corp.sharepoint.com/sites/HR"
    ),
    "SourceFileName": "employee_list.xlsx",
    "SourceRelativeUrl": "HR/Lists",
    "ItemType": "File",
    "FileSizeBytes": 102400
}

SP_UNKNOWN_SITE = {
    "CreationTime": "2024-03-29T03:00:00Z",
    "Id": "sp-event-004",
    "Operation": "FileDownloaded",
    "Workload": "SharePoint",
    "UserId": "unknown@corp.com",
    "ClientIP": "185.220.101.45",
    "SiteUrl": (
        "https://corp.sharepoint.com/sites/ShadowIT"
    ),
    "SourceFileName": "customer_data.csv",
    "SourceRelativeUrl": "Data",
    "ItemType": "File",
    "FileSizeBytes": 10240000
}

SP_OFF_HOURS = {
    "CreationTime": "2024-03-29T03:00:00Z",
    "Id": "sp-event-005",
    "Operation": "FileDownloaded",
    "Workload": "SharePoint",
    "UserId": "jsmith@corp.com",
    "ClientIP": "10.0.0.155",
    "SiteUrl": (
        "https://corp.sharepoint.com/sites/KYC"
    ),
    "SourceFileName": "bank_statement_march.pdf",
    "SourceRelativeUrl": "KYC_Documents/2024",
    "ItemType": "File",
    "FileSizeBytes": 1024000
}

SP_GENERAL_SITE = {
    "CreationTime": "2024-03-29T15:00:00Z",
    "Id": "sp-event-006",
    "Operation": "FileAccessed",
    "Workload": "SharePoint",
    "UserId": "jsmith@corp.com",
    "ClientIP": "10.0.0.155",
    "SiteUrl": (
        "https://corp.sharepoint.com/sites/General"
    ),
    "SourceFileName": "company_policy.docx",
    "SourceRelativeUrl": "Policies",
    "ItemType": "File",
    "FileSizeBytes": 51200
}


# ============================================================
# TEST CLASS — CORE NORMALIZATION
# ============================================================

class TestSharePointCoreNormalization:
    """Tests basic field extraction"""

    def setup_method(self):
        self.normalizer = SharePointNormalizer()

    def test_event_normalized(self):
        """SharePoint event correctly normalized"""
        result = self.normalizer.normalize(
            SP_KYC_DOWNLOAD
        )
        assert result is not None
        assert isinstance(result, DataAccessEvent)
        assert result.source_system == "sharepoint"

    def test_user_extracted(self):
        """User ID correctly extracted"""
        result = self.normalizer.normalize(
            SP_KYC_DOWNLOAD
        )
        assert result.accessor_identity == (
            "jsmith@corp.com"
        )

    def test_filename_in_data_path(self):
        """Filename included in data path"""
        result = self.normalizer.normalize(
            SP_KYC_DOWNLOAD
        )
        assert "passport_john_smith.pdf" in (
            result.data_path
        )

    def test_download_operation_mapped(self):
        """FileDownloaded maps to READ operation"""
        result = self.normalizer.normalize(
            SP_KYC_DOWNLOAD
        )
        assert result.operation == DataOperation.READ

    def test_file_size_extracted(self):
        """File size correctly extracted"""
        result = self.normalizer.normalize(
            SP_KYC_DOWNLOAD
        )
        assert result.bytes_accessed == 2048576

    def test_none_returns_none(self):
        """None input handled gracefully"""
        assert self.normalizer.normalize(None) is None

    def test_production_environment_default(self):
        """
        Normal SharePoint URL defaults to production.
        No dev/test/staging in URL = production.
        """
        result = self.normalizer.normalize(
            SP_KYC_DOWNLOAD
        )
        assert result.environment == (
            Environment.PRODUCTION
        )


# ============================================================
# TEST CLASS — SITE CLASSIFICATION
# ============================================================

class TestSharePointSiteClassification:
    """
    Tests for site risk profile lookup.

    YOUR CONCEPT:
    Parse site name through SITE_RISK_PROFILES.
    Returns sensitivity and base risk.
    Unknown site = shadow IT = governance flag.
    """

    def setup_method(self):
        self.normalizer = SharePointNormalizer()

    def test_site_name_extracted_from_url(self):
        """
        Site name correctly extracted from URL.
        /sites/KYC → KYC
        """
        site_name = self.normalizer._extract_site_name(
            "https://corp.sharepoint.com/sites/KYC"
        )
        assert site_name == "KYC"

    def test_teams_url_handled(self):
        """
        /teams/ URL format handled.
        /teams/Finance → Finance
        """
        site_name = self.normalizer._extract_site_name(
            "https://corp.sharepoint.com/teams/Finance"
        )
        assert site_name == "Finance"

    def test_kyc_site_high_risk(self):
        """
        KYC site gets high base risk score.
        KYC documents contain passport, ID = PII.
        """
        result = self.normalizer.normalize(
            SP_KYC_DOWNLOAD
        )
        assert result.risk_score >= 0.6

    def test_finance_site_elevated_risk(self):
        """Finance site gets elevated risk"""
        result = self.normalizer.normalize(
            SP_FINANCE_ACCESS
        )
        assert result.risk_score >= 0.4

    def test_unknown_site_flagged(self):
        """
        Unknown site flagged as shadow IT.
        ShadowIT site not in SITE_RISK_PROFILES.
        Governance violation — unclassified site.
        """
        result = self.normalizer.normalize(
            SP_UNKNOWN_SITE
        )
        reasons_text = " ".join(result.risk_reasons)
        assert (
            "unknown" in reasons_text.lower() or
            "shadow" in reasons_text.lower() or
            "governance" in reasons_text.lower()
        )

    def test_unknown_site_counter(self):
        """Unknown site detections counted"""
        self.normalizer.normalize(SP_UNKNOWN_SITE)
        stats = self.normalizer.get_statistics()
        assert stats["unknown_sites_detected"] >= 1

    def test_general_site_low_risk(self):
        """
        General site gets low base risk.
        Company policies are not sensitive.
        """
        result = self.normalizer.normalize(
            SP_GENERAL_SITE
        )
        assert result.risk_score < 0.6


# ============================================================
# TEST CLASS — FILENAME SENSITIVITY
# ============================================================

class TestSharePointFilenameSensitivity:
    """
    Tests for filename-based pre-scoring.

    UNSTRUCTURED PII INSIGHT:
    Users name files for their content.
    passport_john_smith.pdf = PII obvious.
    Pre-scoring happens before content scanning.
    """

    def setup_method(self):
        self.normalizer = SharePointNormalizer()

    def test_passport_filename_high_confidence(self):
        """
        Passport in filename = high PII confidence.
        Most common KYC document type.
        """
        label, confidence = (
            self.normalizer._score_filename_sensitivity(
                "passport_john_smith.pdf", ""
            )
        )
        assert confidence >= 0.8

    def test_bank_statement_high_confidence(self):
        """Bank statement filename = high confidence"""
        label, confidence = (
            self.normalizer._score_filename_sensitivity(
                "bank_statement_march.pdf", ""
            )
        )
        assert confidence >= 0.7

    def test_customer_report_medium_confidence(self):
        """Customer report = medium confidence"""
        label, confidence = (
            self.normalizer._score_filename_sensitivity(
                "Q1_customer_report.xlsx", ""
            )
        )
        assert confidence >= 0.3

    def test_company_policy_low_confidence(self):
        """
        Company policy document = low confidence.
        Generic document with no PII signals.
        """
        label, confidence = (
            self.normalizer._score_filename_sensitivity(
                "company_policy.docx", ""
            )
        )
        assert confidence < 0.5

    def test_path_contributes_to_scoring(self):
        """
        KYC in path boosts filename confidence.
        Even generic filename becomes sensitive
        when in KYC_Documents folder.
        """
        label, confidence = (
            self.normalizer._score_filename_sensitivity(
                "document.pdf",
                "KYC_Documents/2024/Pending"
            )
        )
        assert confidence >= 0.3


# ============================================================
# TEST CLASS — BULK DOWNLOAD DETECTION
# ============================================================

class TestSharePointBulkDetection:
    """
    Tests for bulk file download detection.

    SHAREPOINT BULK SIGNAL = FILE COUNT.
    Unlike S3 (bytes) we count files.
    47 files in 30 seconds = bulk download.
    """

    def setup_method(self):
        self.normalizer = SharePointNormalizer()

    def test_bulk_download_detected(self):
        """
        Accessing 25+ files triggers bulk detection.
        Each event = one file = accumulates.
        """
        for i in range(25):
            event = dict(SP_KYC_DOWNLOAD)
            event["Id"] = f"sp-bulk-{i}"
            event["SourceFileName"] = (
                f"document_{i}.pdf"
            )
            self.normalizer.normalize(event)

        stats = self.normalizer.get_statistics()
        assert stats["bulk_downloads_detected"] >= 1

    def test_single_file_not_bulk(self):
        """Single file access not flagged as bulk"""
        self.normalizer.normalize(SP_KYC_DOWNLOAD)
        stats = self.normalizer.get_statistics()
        assert stats["bulk_downloads_detected"] == 0


# ============================================================
# TEST CLASS — ANONYMOUS LINK DETECTION
# ============================================================

class TestSharePointAnonymousLink:
    """
    Tests for anonymous sharing detection.

    AnonymousLinkCreated = document publicly accessible.
    Most common SharePoint data leak vector.
    Immediate review required.
    """

    def setup_method(self):
        self.normalizer = SharePointNormalizer()

    def test_anonymous_link_flagged(self):
        """
        AnonymousLinkCreated gets critical risk.
        File now publicly accessible to anyone.
        Most common accidental data exposure.
        """
        result = self.normalizer.normalize(
            SP_ANONYMOUS_LINK
        )
        reasons_text = " ".join(result.risk_reasons)
        assert (
            "anonymous" in reasons_text.lower() or
            "publicly" in reasons_text.lower()
        )

    def test_anonymous_link_critical_risk(self):
        """Anonymous link = high risk score"""
        result = self.normalizer.normalize(
            SP_ANONYMOUS_LINK
        )
        assert result.risk_score >= 0.6

    def test_anonymous_link_counter(self):
        """Anonymous link detections counted"""
        self.normalizer.normalize(SP_ANONYMOUS_LINK)
        stats = self.normalizer.get_statistics()
        assert stats["anonymous_links_detected"] == 1


# ============================================================
# TEST CLASS — KYC TRACKING
# ============================================================

class TestSharePointKYCTracking:
    """
    Tests for KYC-specific access tracking.

    KYC documents are the highest value target.
    Passport, bank statement, ID = identity theft.
    """

    def setup_method(self):
        self.normalizer = SharePointNormalizer()

    def test_kyc_access_tracked(self):
        """KYC site access events counted"""
        self.normalizer.normalize(SP_KYC_DOWNLOAD)
        stats = self.normalizer.get_statistics()
        assert stats["kyc_access_events"] >= 1

    def test_off_hours_kyc_elevated_risk(self):
        """
        KYC access at 3am = elevated risk.
        Banking KYC documents at off-hours
        = strong exfiltration signal.
        """
        result = self.normalizer.normalize(SP_OFF_HOURS)
        assert result.risk_score >= 0.6

    def test_statistics_tracked(self):
        """Statistics correctly tracked"""
        self.normalizer.normalize(SP_KYC_DOWNLOAD)
        self.normalizer.normalize(SP_FINANCE_ACCESS)
        stats = self.normalizer.get_statistics()
        assert stats["events_processed"] == 2