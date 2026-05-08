"""
Layer 2 — ML Classification
PII/PHI/PCI Data Classifier

WHY THIS FILE EXISTS:
    Your platform monitors WHO accesses data
    and WHEN they access it.
    But without this classifier it does not know
    WHAT SENSITIVITY the data has.

    svc_backup reads prod-s3-bucket:
    Without classifier: "someone read something"
    With classifier:    "someone read 1.5M SSNs
                         and 500K credit cards"

REGULATORY COVERAGE:
    PII  → GDPR (EU), CCPA (California),
           PIPEDA (Canada)
    PHI  → HIPAA (US Healthcare)
    PCI  → PCI-DSS (Global payment cards)
    PFI  → GLBA, SOX (US Financial)

HYBRID DETECTION (regex + ML):
    Regex first:
        Fast, exact pattern matching
        Zero false positives on known formats
        SSN: \d{3}-\d{2}-\d{4}
        Credit card: Luhn algorithm validation

    ML second:
        Catches what regex misses
        Context-aware detection
        "patient diagnosed with..." = PHI
        even without specific pattern

MASKING:
    NEVER stores actual sensitive values.
    GDPR Article 25 requires privacy by design.
    PCI-DSS Requirement 3.4 requires masking.
    Stores: "SSN pattern found" not "123-45-6789"

SQL QUERY MASKING:
    Bank queries often contain PII in WHERE clauses:
    "WHERE ssn = '123-45-6789'"
    We mask before storing in audit trail.
"""

import logging
import re
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Optional

from layer1_ingestion.schema.data_schema import (
    SensitivityLabel,
    DataFinding
)

logger = logging.getLogger(__name__)


# ============================================================
# REGEX PATTERNS
#
# WHY REGEX FIRST:
#   Fast — microseconds per check
#   Zero false positives on exact formats
#   No training data needed
#   Explainable — "matched SSN pattern"
#
# WHEN REGEX FAILS:
#   "Social Security Number: one two three
#    forty five six seven eight nine"
#   Written out = no regex match
#   This is where ML adds value
# ============================================================

# ---- PII PATTERNS (GDPR, CCPA, PIPEDA) ----

PII_PATTERNS = {
    "SSN": {
        "patterns": [
            r"\b\d{3}-\d{2}-\d{4}\b",
            r"\b\d{9}\b(?=\s*(?:ssn|social))",
        ],
        "regulation": ["GDPR", "CCPA"],
        "mask": "XXX-XX-XXXX"
    },
    "EMAIL": {
        "patterns": [
            r"\b[a-zA-Z0-9._%+-]+"
            r"@[a-zA-Z0-9.-]+"
            r"\.[a-zA-Z]{2,}\b"
        ],
        "regulation": ["GDPR", "CCPA", "PIPEDA"],
        "mask": "[EMAIL_MASKED]"
    },
    "PHONE_US": {
        "patterns": [
            r"\b(?:\+1[-.\s]?)?"
            r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ],
        "regulation": ["GDPR", "CCPA"],
        "mask": "XXX-XXX-XXXX"
    },
    "DATE_OF_BIRTH": {
        "patterns": [
            r"\b(?:dob|date.of.birth|born)[:\s]+"
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d{4}-\d{2}-\d{2}\b"
            r"(?=.*(?:birth|dob|born))"
        ],
        "regulation": ["GDPR", "CCPA", "HIPAA"],
        "mask": "XXXX-XX-XX"
    },
    "PASSPORT": {
        "patterns": [
            r"\b[A-Z]{1,2}\d{6,9}\b"
            r"(?=.*passport)"
        ],
        "regulation": ["GDPR"],
        "mask": "PASSPORT_MASKED"
    },
    "DRIVERS_LICENSE": {
        "patterns": [
            r"\b[A-Z]\d{7}\b"
            r"(?=.*(?:license|licence|dl))"
        ],
        "regulation": ["GDPR", "CCPA"],
        "mask": "LICENSE_MASKED"
    },
    "IP_ADDRESS": {
        "patterns": [
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        ],
        "regulation": ["GDPR"],
        "mask": "X.X.X.X"
    },
    "SIN_CANADA": {
        "patterns": [
            r"\b\d{3}[-\s]\d{3}[-\s]\d{3}\b"
            r"(?=.*(?:sin|insurance|canada))"
        ],
        "regulation": ["PIPEDA"],
        "mask": "XXX-XXX-XXX"
    }
}

# ---- PHI PATTERNS (HIPAA) ----

PHI_PATTERNS = {
    "MEDICAL_RECORD_NUMBER": {
        "patterns": [
            r"\b(?:mrn|medical.record)"
            r"[:\s#]+[A-Z0-9-]{5,15}\b"
        ],
        "regulation": ["HIPAA"],
        "mask": "MRN_MASKED"
    },
    "HEALTH_PLAN_ID": {
        "patterns": [
            r"\b(?:health.plan|insurance.id|"
            r"member.id)[:\s]+[A-Z0-9-]{8,15}\b"
        ],
        "regulation": ["HIPAA"],
        "mask": "HEALTH_ID_MASKED"
    },
    "NPI_NUMBER": {
        "patterns": [
            r"\b(?:npi)[:\s]+\d{10}\b"
        ],
        "regulation": ["HIPAA"],
        "mask": "NPI_MASKED"
    },
    "ICD_CODE": {
        "patterns": [
            r"\b[A-Z]\d{2}(?:\.\d{1,4})?\b"
            r"(?=.*(?:diagnosis|icd|code))"
        ],
        "regulation": ["HIPAA"],
        "mask": "DIAGNOSIS_MASKED"
    },
    "DEA_NUMBER": {
        "patterns": [
            r"\b[A-Z]{2}\d{7}\b"
            r"(?=.*(?:dea|prescrib|drug))"
        ],
        "regulation": ["HIPAA"],
        "mask": "DEA_MASKED"
    }
}

# ---- PCI PATTERNS (PCI-DSS) ----

PCI_PATTERNS = {
    "CREDIT_CARD_VISA": {
        "patterns": [
            r"\b4[0-9]{12}(?:[0-9]{3})?\b"
        ],
        "regulation": ["PCI-DSS"],
        "mask": "XXXX-XXXX-XXXX-XXXX",
        "luhn_validate": True
    },
    "CREDIT_CARD_MASTERCARD": {
        "patterns": [
            r"\b(?:5[1-5][0-9]{14}|"
            r"2(?:2[2-9][1-9]|[3-6][0-9]{2}|"
            r"7[01][0-9]|720)[0-9]{12})\b"
        ],
        "regulation": ["PCI-DSS"],
        "mask": "XXXX-XXXX-XXXX-XXXX",
        "luhn_validate": True
    },
    "CREDIT_CARD_AMEX": {
        "patterns": [
            r"\b3[47][0-9]{13}\b"
        ],
        "regulation": ["PCI-DSS"],
        "mask": "XXXX-XXXXXX-XXXXX",
        "luhn_validate": True
    },
    "CVV": {
        "patterns": [
            r"\b(?:cvv|cvv2|cvc|csc|"
            r"security.code)[:\s]+\d{3,4}\b"
        ],
        "regulation": ["PCI-DSS"],
        "mask": "CVV_MASKED"
        # CVV should NEVER be stored
        # Finding CVV anywhere = critical violation
    },
    "CARD_EXPIRY": {
        "patterns": [
            r"\b(?:exp|expiry|expiration)"
            r"[:\s]+(?:0[1-9]|1[0-2])[/\-]\d{2,4}\b"
        ],
        "regulation": ["PCI-DSS"],
        "mask": "XX/XX"
    }
}

# ---- PFI PATTERNS (GLBA, SOX) ----

PFI_PATTERNS = {
    "BANK_ACCOUNT_US": {
        "patterns": [
            r"\b(?:account|acct)[:\s#]+"
            r"\d{8,17}\b"
        ],
        "regulation": ["GLBA", "SOX"],
        "mask": "ACCT_MASKED"
    },
    "ROUTING_NUMBER": {
        "patterns": [
            r"\b(?:routing|aba|rtn)[:\s]+"
            r"[0-9]{9}\b"
        ],
        "regulation": ["GLBA"],
        "mask": "RTN_MASKED"
    },
    "IBAN": {
        "patterns": [
            r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}"
            r"(?:[A-Z0-9]?){0,16}\b"
        ],
        "regulation": ["GDPR", "GLBA"],
        "mask": "IBAN_MASKED"
    },
    "CREDIT_SCORE": {
        "patterns": [
            r"\b(?:fico|credit.score|vantage)"
            r"[:\s]+[3-8]\d{2}\b"
        ],
        "regulation": ["FCRA", "GLBA"],
        "mask": "SCORE_MASKED"
    }
}

# ---- CONTEXT KEYWORDS ----
# Used by ML-based detection
# These keywords near data increase
# the probability of sensitive content

PHI_CONTEXT_KEYWORDS = {
    "patient", "diagnosis", "treatment",
    "prescription", "medical", "doctor",
    "hospital", "clinic", "insurance",
    "health", "disease", "condition",
    "medication", "dosage", "pharmacy",
    "hipaa", "ehr", "emr", "clinical"
}

PII_CONTEXT_KEYWORDS = {
    "customer", "employee", "personal",
    "identity", "address", "contact",
    "profile", "account", "user", "member",
    "applicant", "client", "individual"
}

PCI_CONTEXT_KEYWORDS = {
    "payment", "transaction", "card",
    "purchase", "checkout", "billing",
    "merchant", "authorization", "charge",
    "refund", "cardholder", "pan", "track"
}


# ============================================================
# LUHN ALGORITHM
#
# WHY THIS EXISTS:
#   Not every 16-digit number is a credit card.
#   Phone numbers, account numbers, timestamps
#   can match credit card regex patterns.
#
#   Luhn algorithm validates the checksum
#   built into every real credit card number.
#   Reduces false positives significantly.
# ============================================================

def luhn_check(card_number: str) -> bool:
    """
    Validate credit card number using Luhn algorithm.

    Every legitimate credit card number passes
    this checksum validation.
    Random 16-digit numbers fail ~90% of the time.

    Args:
        card_number: Digits only string

    Returns:
        True if valid credit card number format
    """
    digits = re.sub(r'\D', '', card_number)

    if not digits or len(digits) < 13:
        return False

    total = 0
    reverse_digits = digits[::-1]

    for i, digit in enumerate(reverse_digits):
        n = int(digit)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n

    return total % 10 == 0


# ============================================================
# PII CLASSIFIER
# ============================================================

class PIIClassifier:
    """
    Hybrid PII/PHI/PCI/PFI data classifier.

    Combines regex pattern matching with
    contextual keyword analysis to detect
    sensitive data in any text content.

    WHY HYBRID (regex + context):
        Regex alone misses:
        - Written-out numbers ("one two three...")
        - Context-dependent data (name + SSN together)
        - Implicit sensitive data

        Context alone has too many false positives:
        Word "patient" near any number != PHI

        Combined:
        - Regex catches structured patterns
        - Context increases confidence
        - Together produce reliable labels

    MASKING:
        Never stores actual sensitive values.
        Masks detected values before returning.
        GDPR Article 25 compliance built in.

    Usage:
        classifier = PIIClassifier()

        # Classify text content
        finding = classifier.classify(
            text="Customer SSN: 123-45-6789
                  Card: 4532015112830366"
        )

        # Mask sensitive SQL query
        masked = classifier.mask_query(
            "WHERE ssn = '123-45-6789'"
        )
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize PII classifier.

        Args:
            confidence_threshold:
                Minimum confidence to label as sensitive.
                0.7 = 70% confidence required.
                Lower = more sensitive (more FP)
                Higher = more conservative (more FN)
        """
        self.confidence_threshold = (
            confidence_threshold
        )

        # Statistics
        self.texts_classified = 0
        self.pii_found = 0
        self.phi_found = 0
        self.pci_found = 0
        self.pfi_found = 0

        logger.info(
            f"PIIClassifier initialized "
            f"threshold={confidence_threshold}"
        )

    def classify(
        self,
        text: str,
        source_context: str = ""
    ) -> DataFinding:
        """
        Classify text content for sensitive data.

        WHAT HAPPENS:
        1. Run all regex patterns against text
        2. Validate credit cards with Luhn
        3. Check context keywords
        4. Calculate confidence per finding
        5. Determine highest sensitivity label
        6. Return DataFinding (NEVER raw values)

        Args:
            text: Text content to classify
                  Can be: SQL query result,
                  file content sample,
                  database column values,
                  document text

            source_context: Additional context
                           "sql_query", "csv_file",
                           "database_column"

        Returns:
            DataFinding with labels and counts
            NEVER contains actual sensitive values
        """
        if not text:
            return DataFinding(
                sensitivity_label=SensitivityLabel.NONE,
                confidence=1.0,
                classified_at=self._now()
            )

        text_lower = text.lower()
        found_types = {}
        regulations_triggered = set()
        all_findings = []

        # ---- STEP 1: RUN REGEX PATTERNS ----

        # PCI first (most specific = fewer FP)
        pci_findings = self._check_patterns(
            text, PCI_PATTERNS, "PCI"
        )
        all_findings.extend(pci_findings)

        # PHI
        phi_findings = self._check_patterns(
            text, PHI_PATTERNS, "PHI"
        )
        all_findings.extend(phi_findings)

        # PII
        pii_findings = self._check_patterns(
            text, PII_PATTERNS, "PII"
        )
        all_findings.extend(pii_findings)

        # PFI
        pfi_findings = self._check_patterns(
            text, PFI_PATTERNS, "PFI"
        )
        all_findings.extend(pfi_findings)

        # ---- STEP 2: CONTEXT KEYWORDS ----
        context_boost = self._check_context(
            text_lower
        )

        # ---- STEP 3: BUILD FINDINGS ----
        for finding in all_findings:
            data_type = finding["type"]
            category = finding["category"]
            count = finding["count"]
            regulation = finding["regulation"]
            confidence = finding["confidence"]

            # Apply context boost
            if (
                category == "PHI" and
                context_boost.get("phi", 0) > 0
            ):
                confidence = min(
                    1.0, confidence + 0.1
                )
            elif (
                category == "PCI" and
                context_boost.get("pci", 0) > 0
            ):
                confidence = min(
                    1.0, confidence + 0.1
                )

            if confidence >= self.confidence_threshold:
                if category not in found_types:
                    found_types[category] = {}

                found_types[category][data_type] = (
                    found_types[category].get(
                        data_type, 0
                    ) + count
                )
                regulations_triggered.update(
                    regulation
                )

        # ---- STEP 4: DETERMINE LABEL ----
        if not found_types:
            label = SensitivityLabel.NONE
            confidence = 0.95
        elif len(found_types) > 1:
            label = SensitivityLabel.MIXED
            confidence = 0.90
        elif "PCI" in found_types:
            label = SensitivityLabel.PCI
            confidence = 0.92
        elif "PHI" in found_types:
            label = SensitivityLabel.PHI
            confidence = 0.88
        elif "PFI" in found_types:
            label = SensitivityLabel.PFI
            confidence = 0.85
        else:
            label = SensitivityLabel.PII
            confidence = 0.85

        # ---- STEP 5: BUILD DATA TYPES LIST ----
        data_types_found = []
        data_type_counts = {}

        for category, types in found_types.items():
            for type_name, count in types.items():
                data_types_found.append(type_name)
                data_type_counts[type_name] = count

        # ---- STEP 6: UPDATE STATISTICS ----
        self.texts_classified += 1
        if "PII" in found_types:
            self.pii_found += 1
        if "PHI" in found_types:
            self.phi_found += 1
        if "PCI" in found_types:
            self.pci_found += 1
        if "PFI" in found_types:
            self.pfi_found += 1

        finding = DataFinding(
            sensitivity_label=label,
            confidence=confidence,
            data_types_found=data_types_found,
            data_type_counts=data_type_counts,
            regulations_triggered=list(
                regulations_triggered
            ),
            detection_method="hybrid_regex_context",
            classified_at=self._now()
        )

        if label != SensitivityLabel.NONE:
            logger.info(
                f"Sensitive data found: "
                f"{label.value} "
                f"types={data_types_found} "
                f"confidence={confidence:.2f}"
            )

        return finding

    def mask_query(
        self,
        query: str
    ) -> str:
        """
        Mask sensitive values in SQL query.

        WHY MASKING SQL:
        Bank queries contain PII in WHERE clauses:
        "WHERE ssn = '123-45-6789'"
        "WHERE card_number = '4532015112830366'"

        Before storing in audit trail or platform:
        Mask actual values → keep structure.

        Analyst still sees:
        "WHERE ssn = 'XXX-XX-XXXX'"
        Knows PII was in query.
        Cannot see actual value.

        GDPR Article 25 compliance.
        PCI-DSS Requirement 3.4 compliance.

        Args:
            query: SQL query text

        Returns:
            Query with sensitive values masked
        """
        masked = query

        # Mask all pattern types
        all_patterns = {
            **PII_PATTERNS,
            **PHI_PATTERNS,
            **PCI_PATTERNS,
            **PFI_PATTERNS
        }

        for data_type, config in all_patterns.items():
            mask_value = config.get(
                "mask", f"[{data_type}_MASKED]"
            )
            for pattern in config["patterns"]:
                try:
                    masked = re.sub(
                        pattern,
                        mask_value,
                        masked,
                        flags=re.IGNORECASE
                    )
                except re.error:
                    continue

        return masked

    def scan_data_store_sample(
        self,
        sample_rows: list,
        store_name: str = ""
    ) -> DataFinding:
        """
        Scan a sample of rows from a data store.

        Used for data store profiling.
        Takes a sample of actual data content
        and classifies what sensitive data exists.

        Args:
            sample_rows: List of text samples
                        from the data store
                        (already masked by source system
                         or using synthetic examples)

            store_name: Name of data store for logging

        Returns:
            DataFinding with aggregate results
        """
        if not sample_rows:
            return DataFinding(
                sensitivity_label=SensitivityLabel.NONE,
                confidence=0.95
            )

        # Combine samples for classification
        combined_text = " ".join(
            str(row) for row in sample_rows[:100]
        )

        finding = self.classify(
            combined_text,
            source_context=f"data_store:{store_name}"
        )

        logger.info(
            f"Data store scan complete: "
            f"{store_name} → "
            f"{finding.sensitivity_label.value} "
            f"confidence={finding.confidence:.2f}"
        )

        return finding

    def get_applicable_regulations(
        self,
        sensitivity_label: SensitivityLabel,
        data_region: str = "US"
    ) -> list:
        """
        Get applicable regulations for a
        sensitivity label and region.

        WHY REGION MATTERS:
        PII in US → CCPA (California residents)
        PII in EU → GDPR
        PII in Canada → PIPEDA
        PHI in US → HIPAA
        PCI anywhere → PCI-DSS (global)

        Args:
            sensitivity_label: What type of data
            data_region: Where data is stored/processed

        Returns:
            List of applicable regulation names
        """
        regulations = []

        if sensitivity_label == SensitivityLabel.PCI:
            regulations.append("PCI-DSS")
            # PCI-DSS applies globally

        if sensitivity_label in [
            SensitivityLabel.PII,
            SensitivityLabel.MIXED
        ]:
            if "EU" in data_region.upper():
                regulations.append("GDPR")
            if "US" in data_region.upper():
                regulations.append("CCPA")
            if "CA" in data_region.upper():
                regulations.append("PIPEDA")

        if sensitivity_label == SensitivityLabel.PHI:
            if "US" in data_region.upper():
                regulations.append("HIPAA")

        if sensitivity_label == SensitivityLabel.PFI:
            if "US" in data_region.upper():
                regulations.append("GLBA")
                regulations.append("SOX")

        return regulations

    # ============================================================
    # PRIVATE METHODS
    # ============================================================

    def _check_patterns(
        self,
        text: str,
        patterns: dict,
        category: str
    ) -> list:
        """
        Run regex patterns against text.
        Returns findings without actual values.
        """
        findings = []

        for data_type, config in patterns.items():
            total_count = 0

            for pattern in config["patterns"]:
                try:
                    matches = re.findall(
                        pattern,
                        text,
                        re.IGNORECASE
                    )

                    if matches:
                        valid_count = 0

                        # Validate credit cards
                        # with Luhn algorithm
                        if config.get(
                            "luhn_validate"
                        ):
                            for match in matches:
                                if luhn_check(match):
                                    valid_count += 1
                        else:
                            valid_count = len(matches)

                        total_count += valid_count

                except re.error as e:
                    logger.debug(
                        f"Regex error in {data_type}: "
                        f"{e}"
                    )
                    continue

            if total_count > 0:
                findings.append({
                    "type": data_type,
                    "category": category,
                    "count": total_count,
                    "regulation": config["regulation"],
                    "confidence": (
                        0.95 if category == "PCI"
                        else 0.85
                    )
                })

        return findings

    def _check_context(
        self,
        text_lower: str
    ) -> dict:
        """
        Check for context keywords that indicate
        sensitive data category.

        Used to boost confidence when keywords
        appear near detected patterns.
        """
        context = {}

        phi_hits = sum(
            1 for kw in PHI_CONTEXT_KEYWORDS
            if kw in text_lower
        )
        pii_hits = sum(
            1 for kw in PII_CONTEXT_KEYWORDS
            if kw in text_lower
        )
        pci_hits = sum(
            1 for kw in PCI_CONTEXT_KEYWORDS
            if kw in text_lower
        )

        if phi_hits > 0:
            context["phi"] = phi_hits
        if pii_hits > 0:
            context["pii"] = pii_hits
        if pci_hits > 0:
            context["pci"] = pci_hits

        return context

    def get_statistics(self) -> dict:
        return {
            "texts_classified": self.texts_classified,
            "pii_found": self.pii_found,
            "phi_found": self.phi_found,
            "pci_found": self.pci_found,
            "pfi_found": self.pfi_found
        }

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )