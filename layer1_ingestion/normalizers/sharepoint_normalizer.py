"""
Layer 1 — Data Ingestion
Microsoft SharePoint Access Normalizer

WHY THIS FILE EXISTS:
    SharePoint is the most common document
    management system in banking.
    It stores the documents that contain
    the most sensitive unstructured PII:

    KYC Documents:
        Passport scans (PII: name, DOB, nationality)
        Bank statements (PFI: accounts, transactions)
        Utility bills (PII: address, name)
        ID cards (PII: name, DOB, ID number)

    Contracts and Legal:
        Client agreements (PII + PFI)
        Loan documents (PII + PFI + credit score)
        Insurance policies (PII + PHI)

    Internal Reports:
        Customer lists (PII)
        Risk reports (PFI)
        Audit findings (mixed)

    Most DSPM tools scan S3 and databases.
    SharePoint is the BLIND SPOT.
    Attackers know this.
    KYC documents = goldmines for identity theft.

UNSTRUCTURED PII CHALLENGE:
    S3/RDS: PII in structured rows → easy to scan
    SharePoint: PII buried in document text
    
    Two-layer approach:
    Layer 1: Filename/path pre-scoring
             passport_john_smith.pdf → 0.95 PII
             Immediate risk estimate before content
    
    Layer 2: Content extraction + PII classifier
             PyMuPDF for PDFs (already in platform!)
             python-docx for Word documents
             openpyxl for Excel spreadsheets
             pytesseract for scanned images (OCR)

SITE CLASSIFICATION MAP:
    Every SharePoint site belongs to a department.
    HR_Site, Finance_Site, KYC_Site, Legal_Site.
    Site name reveals expected sensitivity.
    KYC_Site → PII by definition.
    Finance_Site → PFI by definition.
    Unknown site → shadow IT → governance flag.

BULK DOWNLOAD DETECTION:
    S3: bytes_transferred > threshold
    SharePoint: files_accessed_count > threshold
                in rolling time window
    47 files in 30 seconds = 94 files/minute
    Normal user: 1-5 files per minute
    19x normal rate = bulk download flag

SHAREPOINT AUDIT LOG STRUCTURE:
{
    "CreationTime": "2024-03-29T09:19:00Z",
    "Id": "event-uuid",
    "Operation": "FileDownloaded",
    "Workload": "SharePoint",
    "UserId": "jsmith@corp.com",
    "ClientIP": "10.0.0.155",
    "SiteUrl": "https://corp.sharepoint.com/sites/KYC",
    "SourceFileName": "passport_john_smith.pdf",
    "SourceRelativeUrl": "KYC_Documents/2024/Pending",
    "ItemType": "File",
    "FileSizeBytes": 2048576,
    "UserAgent": "Mozilla/5.0..."
}
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
# SHAREPOINT OPERATION MAPPINGS
# ============================================================

READ_OPERATIONS = {
    "FileAccessed",
    "FilePreviewed",
    "FileAccessed",
    "PageViewed",
    "SearchQueryPerformed"
}

DOWNLOAD_OPERATIONS = {
    "FileDownloaded",
    "FileSyncDownloadedFull",
    "FileSyncDownloadedPartial"
}

WRITE_OPERATIONS = {
    "FileUploaded",
    "FileModified",
    "FileVersionsAllDeleted",
    "FolderCreated"
}

DELETE_OPERATIONS = {
    "FileDeleted",
    "FileDeletedFirstStageRecycleBin",
    "FileDeletedSecondStageRecycleBin",
    "FolderDeleted"
}

SHARE_OPERATIONS = {
    "SharingSet",
    "SharingInvitationCreated",
    "AnonymousLinkCreated",
    "CompanyLinkCreated"
}

HIGH_RISK_OPERATIONS = {
    "AnonymousLinkCreated",
    "FileDeleted",
    "SharingSet",
    "FileSyncDownloadedFull"
}


# ============================================================
# SITE CLASSIFICATION MAP
#
# YOUR CONCEPT IMPLEMENTED:
# Each site name parsed through this object.
# Returns sensitivity and base risk.
# Unknown sites flagged as shadow IT.
# ============================================================

SITE_RISK_PROFILES = {
    # Critical sensitivity sites
    "kyc": {
        "sensitivity": SensitivityLabel.PII,
        "base_risk": 0.8,
        "description": "KYC documents contain "
                       "passport, ID, bank statement"
    },
    "aml": {
        "sensitivity": SensitivityLabel.PFI,
        "base_risk": 0.8,
        "description": "AML files contain "
                       "transaction analysis, PFI"
    },
    # High sensitivity sites
    "hr": {
        "sensitivity": SensitivityLabel.PII,
        "base_risk": 0.6,
        "description": "HR contains employee PII"
    },
    "legal": {
        "sensitivity": SensitivityLabel.PII,
        "base_risk": 0.6,
        "description": "Legal contains client PII"
    },
    "finance": {
        "sensitivity": SensitivityLabel.PFI,
        "base_risk": 0.7,
        "description": "Finance contains PFI data"
    },
    "compliance": {
        "sensitivity": SensitivityLabel.PII,
        "base_risk": 0.6,
        "description": "Compliance contains "
                       "audit and regulatory PII"
    },
    "riskmgmt": {
        "sensitivity": SensitivityLabel.PFI,
        "base_risk": 0.7,
        "description": "Risk management PFI data"
    },
    "payroll": {
        "sensitivity": SensitivityLabel.PFI,
        "base_risk": 0.8,
        "description": "Payroll contains "
                       "salary and banking PFI"
    },
    "medical": {
        "sensitivity": SensitivityLabel.PHI,
        "base_risk": 0.8,
        "description": "Medical records PHI"
    },
    # Medium sensitivity
    "it": {
        "sensitivity": SensitivityLabel.NONE,
        "base_risk": 0.2,
        "description": "IT documentation"
    },
    "marketing": {
        "sensitivity": SensitivityLabel.PII,
        "base_risk": 0.3,
        "description": "Marketing may contain PII"
    },
    "operations": {
        "sensitivity": SensitivityLabel.NONE,
        "base_risk": 0.2,
        "description": "Operations documentation"
    },
    # Low sensitivity
    "general": {
        "sensitivity": SensitivityLabel.NONE,
        "base_risk": 0.1,
        "description": "General information"
    },
    "public": {
        "sensitivity": SensitivityLabel.NONE,
        "base_risk": 0.0,
        "description": "Public content"
    }
}

# ============================================================
# FILENAME SENSITIVITY PATTERNS
#
# Filenames reveal content before opening.
# Developer/user named the file for its content.
# Much faster than content scanning.
# ============================================================

HIGH_SENSITIVITY_FILENAME_KEYWORDS = {
    "passport", "national_id", "id_card",
    "driving_licence", "drivers_license",
    "birth_certificate", "ssn", "tax_return",
    "w2", "1099", "bank_statement", "statement",
    "bank_stmt", "aml", "credit_report",
    "background_check", "medical_record",
    "health_record", "insurance_card",
    "prescription", "salary", "payslip", "payroll",
    "kyc"
}

MEDIUM_SENSITIVITY_FILENAME_KEYWORDS = {
    "customer", "client", "employee", "staff",
    "member", "account", "payment", "invoice",
    "contract", "agreement", "personal",
    "confidential", "restricted", "sensitive",
    "private", "internal", "financial",
    "report", "statement", "document"
}

SENSITIVE_FILE_EXTENSIONS = {
    ".pdf",   # Most KYC documents
    ".docx",  # Contracts, agreements
    ".xlsx",  # Customer lists, financial data
    ".doc",   # Legacy Word documents
    ".xls",   # Legacy Excel
    ".csv"    # Data exports
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tiff", ".tif"
}
# Images may be scanned ID documents
# Require OCR for content classification

# Business hours UTC
BUSINESS_HOURS_START = 13
BUSINESS_HOURS_END = 23

# Bulk download thresholds
BULK_FILE_THRESHOLD = 20   # files per minute
CRITICAL_FILE_THRESHOLD = 50  # files per minute


class SharePointNormalizer:
    """
    Normalizes Microsoft SharePoint audit logs
    to DataAccessEvent objects.

    HANDLES UNSTRUCTURED PII:
    SharePoint stores documents not rows.
    Two-layer sensitivity detection:
    1. Filename/path pre-scoring (immediate)
    2. Content extraction + PII classifier (confirmed)

    SITE CLASSIFICATION:
    Each site parsed through SITE_RISK_PROFILES.
    HR_Site → PII site → elevated base risk.
    Unknown site → shadow IT → governance flag.

    BULK DOWNLOAD DETECTION:
    Accumulates files_accessed per accessor.
    47 files in 30 seconds = bulk download flag.
    Unlike S3 (bytes) SharePoint uses file count.

    Usage:
        normalizer = SharePointNormalizer()
        event = normalizer.normalize(
            raw_sharepoint_audit_log
        )
    """

    def __init__(self):
        # Accumulate file access per accessor
        # key: user_id
        # value: {"files": int, "last_reset": str}
        self._accessor_file_counts = {}

        # Track known users per site
        self._site_users = {}

        # Track known file paths per user
        self._user_file_access = {}

        # Statistics
        self.events_processed = 0
        self.high_risk_events = 0
        self.bulk_downloads_detected = 0
        self.anonymous_links_detected = 0
        self.unknown_sites_detected = 0
        self.kyc_access_events = 0

        logger.info("SharePointNormalizer initialized")

    def normalize(
        self,
        raw_event: dict
    ) -> Optional[DataAccessEvent]:
        """
        Normalize SharePoint audit log event.

        ETL PIPELINE:
        1. Extract user identity and operation
        2. Parse site URL for site name
        3. Look up site risk profile
        4. Score filename sensitivity
        5. Detect bulk download
        6. Detect anonymous sharing
        7. Calculate risk score
        8. Return DataAccessEvent
        """
        if not raw_event:
            return None

        # Validate SharePoint event
        workload = raw_event.get("Workload", "")
        operation = raw_event.get("Operation", "")

        if not operation:
            return None

        # Accept SharePoint and OneDrive events
        valid_workloads = {
            "SharePoint", "OneDrive", ""
        }
        if workload and workload not in valid_workloads:
            return None

        try:
            # ---- EXTRACT CORE FIELDS ----
            event_time = raw_event.get(
                "CreationTime", ""
            )
            event_id = raw_event.get("Id", "")
            user_id = raw_event.get("UserId", "")
            client_ip = raw_event.get("ClientIP", "")

            # ---- EXTRACT FILE CONTEXT ----
            site_url = raw_event.get("SiteUrl", "")
            source_filename = raw_event.get(
                "SourceFileName", ""
            )
            source_relative_url = raw_event.get(
                "SourceRelativeUrl", ""
            )
            item_type = raw_event.get(
                "ItemType", "File"
            )
            file_size = int(
                raw_event.get("FileSizeBytes", 0)
                or 0
            )

            # ---- PARSE SITE NAME ----
            site_name = self._extract_site_name(
                site_url
            )

            # ---- LOOK UP SITE RISK PROFILE ----
            site_profile = self._get_site_profile(
                site_name
            )
            is_unknown_site = site_profile is None
            if is_unknown_site:
                self.unknown_sites_detected += 1

            # ---- BUILD DATA PATH ----
            data_path = self._build_data_path(
                site_name,
                source_relative_url,
                source_filename
            )

            # ---- FILENAME SENSITIVITY ----
            filename_sensitivity, filename_confidence = (
                self._score_filename_sensitivity(
                    source_filename,
                    source_relative_url
                )
            )

            # ---- DETERMINE OPERATION ----
            data_operation = self._map_operation(
                operation
            )

            # ---- DETERMINE ENVIRONMENT ----
            environment = self._detect_environment(
                site_url, site_name
            )

            # ---- ACCESSOR TYPE ----
            accessor_type = self._map_accessor_type(
                user_id
            )

            # ---- ANONYMOUS LINK DETECTION ----
            is_anonymous_share = (
                operation == "AnonymousLinkCreated"
            )
            if is_anonymous_share:
                self.anonymous_links_detected += 1
                logger.warning(
                    f"Anonymous link created: "
                    f"{user_id} shared "
                    f"{source_filename}"
                )

            # ---- BULK DOWNLOAD DETECTION ----
            is_bulk = self._detect_bulk_download(
                user_id, operation, event_time
            )
            if is_bulk:
                self.bulk_downloads_detected += 1

            # ---- KYC ACCESS TRACKING ----
            if site_name and "kyc" in site_name.lower():
                self.kyc_access_events += 1

            # ---- OFF HOURS ----
            is_off_hours = self._is_off_hours(
                event_time
            )

            # ---- NEW USER FOR SITE ----
            is_new_user = self._is_new_site_user(
                user_id, site_name
            )

            # ---- BUILD DATA ACCESS EVENT ----
            event = DataAccessEvent(
                event_id=event_id,
                event_time=event_time,
                source_system="sharepoint",
                accessor_identity=user_id,
                accessor_type=accessor_type,
                data_store_type=DataStoreType.SHAREPOINT,
                data_store_name=site_name,
                data_path=data_path,
                operation=data_operation,
                bytes_accessed=file_size,
                files_accessed=1,
                environment=environment,
                source_ip=client_ip,
                is_off_hours=is_off_hours,
                raw_event=raw_event
            )

            # ---- RISK SCORING ----
            risk_score, risk_reasons = (
                self._calculate_risk(
                    event=event,
                    operation=operation,
                    site_profile=site_profile,
                    site_name=site_name,
                    filename_sensitivity=(
                        filename_sensitivity
                    ),
                    filename_confidence=(
                        filename_confidence
                    ),
                    is_anonymous_share=(
                        is_anonymous_share
                    ),
                    is_bulk=is_bulk,
                    is_unknown_site=is_unknown_site,
                    is_new_user=is_new_user,
                    source_filename=source_filename
                )
            )

            event.risk_score = risk_score
            event.risk_label = (
                self._score_to_label(risk_score)
            )
            event.risk_reasons = risk_reasons

            # ---- UPDATE HISTORY ----
            self._update_history(
                user_id, site_name, source_filename
            )

            self.events_processed += 1
            if risk_score >= 0.7:
                self.high_risk_events += 1

            logger.info(
                f"SharePoint event normalized: "
                f"{operation} "
                f"site={site_name} "
                f"file={source_filename} "
                f"user={user_id} "
                f"risk={risk_score:.2f}"
            )

            return event

        except Exception as e:
            logger.error(
                f"SharePoint normalization "
                f"failed: {e}"
            )
            return None

    # ============================================================
    # SITE ANALYSIS
    # ============================================================

    def _extract_site_name(
        self,
        site_url: str
    ) -> str:
        """
        Extract site name from SharePoint URL.

        URL formats:
        https://corp.sharepoint.com/sites/KYC
        https://corp.sharepoint.com/sites/HR_Site
        https://corp.sharepoint.com/teams/Finance

        Extract: KYC, HR_Site, Finance
        """
        if not site_url:
            return ""

        # Match /sites/NAME or /teams/NAME
        pattern = re.compile(
            r"/(?:sites|teams)/([^/]+)",
            re.IGNORECASE
        )
        match = pattern.search(site_url)
        if match:
            return match.group(1)

        return ""

    def _get_site_profile(
         self,
         site_name: str   

    )-> Optional [dict]:
        """
        Look up site risk profile.
        Splits site name into words for exact matching.
        ShadowIT → ["shadow", "it"] → no match → None
        Finance_Site → ["finance", "site"] → matches "finance"
        """
        if not site_name:
            return None
        site_lower = site_name.lower()
        site_words = re.split(r'[_\-\s]', site_lower)

        for key, profile in SITE_RISK_PROFILES.items():
         if key in site_words:
            return profile

        return None 

    def _build_data_path(
        self,
        site_name: str,
        relative_url: str,
        filename: str
    ) -> str:
        """
        Build consistent data path.
        Format: SiteName/RelativePath/Filename
        """
        parts = [
            p for p in [
                site_name, relative_url, filename
            ]
            if p
        ]
        return "/".join(parts)

    # ============================================================
    # FILENAME SENSITIVITY SCORING
    # ============================================================

    def _score_filename_sensitivity(
        self,
        filename: str,
        relative_url: str
    ) -> tuple:
        """
        Score sensitivity from filename and path.

        UNSTRUCTURED DOCUMENT INSIGHT:
        Users name files for their content:
        passport_john_smith.pdf → PII obvious
        Q1_customer_export.xlsx → PII likely
        marketing_presentation.pptx → low risk

        This pre-scoring happens BEFORE
        content extraction.
        Immediate risk estimate.
        High confidence files prioritized
        for content scanning.
        """
        if not filename and not relative_url:
            return SensitivityLabel.UNKNOWN, 0.1

        combined = (
            f"{filename} {relative_url}"
        ).lower()

        # Remove extension for keyword matching
        combined_no_ext = re.sub(
            r'\.\w+$', '', combined
        )
        combined_no_ext = re.sub(
            r'[_\-/\.]', ' ', combined_no_ext
        )

        # Check high sensitivity keywords
        high_hits = sum(
            1 for kw in
            HIGH_SENSITIVITY_FILENAME_KEYWORDS
            if kw in combined_no_ext
        )

        # Check medium sensitivity
        medium_hits = sum(
            1 for kw in
            MEDIUM_SENSITIVITY_FILENAME_KEYWORDS
            if kw in combined_no_ext
        )

        # File extension signal
        ext = ""
        if "." in filename:
            ext = "." + filename.rsplit(".", 1)[-1].lower()

        is_sensitive_ext = ext in SENSITIVE_FILE_EXTENSIONS
        is_image = ext in IMAGE_EXTENSIONS

        # Score calculation
        if high_hits >= 1:
            confidence = 0.90
            if is_image:
                confidence = 0.85
                # Could be scanned ID document
            return SensitivityLabel.PII, confidence

        if medium_hits >= 2:
            confidence = min(
                0.75,
                0.4 + (medium_hits * 0.1)
            )
            return SensitivityLabel.PII, confidence

        if medium_hits == 1 and is_sensitive_ext:
            return SensitivityLabel.PII, 0.45

        if is_image:
            # Images could be scanned documents
            return SensitivityLabel.UNKNOWN, 0.3

        return SensitivityLabel.UNKNOWN, 0.1

    # ============================================================
    # BEHAVIORAL DETECTION
    # ============================================================

    def _detect_bulk_download(
        self,
        user_id: str,
        operation: str,
        event_time: str
    ) -> bool:
        """
        Detect bulk file download.

        SHAREPOINT BULK SIGNAL:
        files_accessed_count > threshold
        in rolling time window.

        Unlike S3 (bytes) we count FILES.
        47 files in 30 seconds = bulk download.
        Normal user: 1-5 files per minute.

        Download and sync operations counted.
        View/preview operations not counted.
        """
        download_ops = (
            DOWNLOAD_OPERATIONS | READ_OPERATIONS
        )
        if operation not in download_ops:
            return False

        if user_id not in self._accessor_file_counts:
            self._accessor_file_counts[user_id] = {
                "files": 0,
                "queries": 0
            }

        self._accessor_file_counts[user_id][
            "files"
        ] += 1
        self._accessor_file_counts[user_id][
            "queries"
        ] += 1

        file_count = self._accessor_file_counts[
            user_id
        ]["files"]

        if file_count >= BULK_FILE_THRESHOLD:
            logger.warning(
                f"Bulk download: {user_id} "
                f"accessed {file_count} files"
            )
            return True

        return False

    def _map_operation(
        self,
        operation: str
    ) -> DataOperation:
        """Map SharePoint operation to DataOperation"""
        if operation in READ_OPERATIONS:
            return DataOperation.READ
        elif operation in DOWNLOAD_OPERATIONS:
            return DataOperation.READ
        elif operation in WRITE_OPERATIONS:
            return DataOperation.WRITE
        elif operation in DELETE_OPERATIONS:
            return DataOperation.DELETE
        elif operation in SHARE_OPERATIONS:
            return DataOperation.WRITE
        return DataOperation.UNKNOWN

    def _map_accessor_type(
        self,
        user_id: str
    ) -> AccessorType:
        """Map SharePoint user to AccessorType"""
        user_lower = user_id.lower()

        if any(
            p in user_lower
            for p in ["svc", "service", "system",
                      "app", "api"]
        ):
            return AccessorType.SERVICE_ACCOUNT

        if any(
            p in user_lower
            for p in ["admin", "sharepoint"]
        ):
            return AccessorType.PRIVILEGED

        return AccessorType.HUMAN

    def _detect_environment(
        self,
        site_url: str,
        site_name: str
    ) -> Environment:
        """Detect environment from URL and site name"""
        combined = (
            f"{site_url} {site_name}"
        ).lower()

        if any(
            p in combined
            for p in ["dev", "test", "sandbox", "uat"]
        ):
            return Environment.DEVELOPMENT

        if any(
            p in combined
            for p in ["staging", "stg"]
        ):
            return Environment.STAGING

        return Environment.PRODUCTION

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

    def _is_new_site_user(
        self,
        user_id: str,
        site_name: str
    ) -> bool:
        """Check if first time user accessed site"""
        if not site_name:
            return False
        known = self._site_users.get(
            site_name, set()
        )
        return user_id not in known

    # ============================================================
    # RISK SCORING
    # ============================================================

    def _calculate_risk(
        self,
        event: DataAccessEvent,
        operation: str,
        site_profile: Optional[dict],
        site_name: str,
        filename_sensitivity: SensitivityLabel,
        filename_confidence: float,
        is_anonymous_share: bool,
        is_bulk: bool,
        is_unknown_site: bool,
        is_new_user: bool,
        source_filename: str
    ) -> tuple:
        """Calculate risk for SharePoint event"""
        score = 0.0
        reasons = []

        # Site risk profile base score
        if site_profile:
            base = site_profile["base_risk"]
            score += base
            if base >= 0.6:
                reasons.append(
                    f"High sensitivity site: "
                    f"{site_name} — "
                    f"{site_profile['description']}"
                )
            elif base >= 0.3:
                reasons.append(
                    f"Medium sensitivity site: "
                    f"{site_name}"
                )

        # Unknown site = shadow IT
        if is_unknown_site:
            score += 0.3
            reasons.append(
                f"Unknown SharePoint site: "
                f"{site_name} — "
                f"possible shadow IT. "
                f"Site not in governance registry."
            )

        # Filename sensitivity
        if filename_confidence >= 0.8:
            score += 0.4
            reasons.append(
                f"High sensitivity filename: "
                f"{source_filename} — "
                f"likely {filename_sensitivity.value} "
                f"document"
            )
        elif filename_confidence >= 0.4:
            score += 0.2
            reasons.append(
                f"Potentially sensitive file: "
                f"{source_filename}"
            )

        # Anonymous link created
        if is_anonymous_share:
            score += 0.6
            reasons.append(
                f"Anonymous sharing link created "
                f"for {source_filename} — "
                f"file now publicly accessible. "
                f"Immediate review required."
            )

        # Bulk download
        if is_bulk:
            file_count = (
                self._accessor_file_counts
                .get(event.accessor_identity, {})
                .get("files", 0)
            )
            score += 0.4
            reasons.append(
                f"Bulk download detected: "
                f"{file_count} files accessed — "
                f"possible data exfiltration"
            )

        # Download vs preview
        if operation in DOWNLOAD_OPERATIONS:
            score += 0.1
            reasons.append(
                f"File downloaded (not just viewed): "
                f"{source_filename}"
            )

        # Off hours
        if event.is_off_hours:
            score += 0.2
            reasons.append(
                f"SharePoint access outside "
                f"business hours"
            )

        # New user for this site
        if is_new_user:
            score += 0.2
            reasons.append(
                f"First time {event.accessor_identity} "
                f"accessed site {site_name}"
            )

        # High risk operation
        if operation in HIGH_RISK_OPERATIONS:
            score += 0.3
            reasons.append(
                f"High risk operation: {operation}"
            )

        return min(score, 1.0), reasons

    # ============================================================
    # HISTORY MANAGEMENT
    # ============================================================

    def _update_history(
        self,
        user_id: str,
        site_name: str,
        filename: str
    ) -> None:
        """Update user and site history"""
        if site_name:
            if site_name not in self._site_users:
                self._site_users[site_name] = set()
            self._site_users[site_name].add(user_id)

        if filename and user_id:
            if user_id not in self._user_file_access:
                self._user_file_access[user_id] = set()
            self._user_file_access[user_id].add(
                filename
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
            "bulk_downloads_detected": (
                self.bulk_downloads_detected
            ),
            "anonymous_links_detected": (
                self.anonymous_links_detected
            ),
            "unknown_sites_detected": (
                self.unknown_sites_detected
            ),
            "kyc_access_events": (
                self.kyc_access_events
            ),
            "sites_tracked": len(self._site_users),
            "users_tracked": len(
                self._accessor_file_counts
            )
        }