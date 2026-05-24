"""
Tests for Email Gateway Normalizer
"""
import pytest
from layer1_ingestion.normalizers.email_gateway_normalizer import (
    EmailGatewayNormalizer,
    EMAIL_THREAT_RISK,
    PROOFPOINT_ACTION_RISK,
    MIMECAST_ACTION_RISK,
    EXECUTIVE_ROLES,
    FINANCIAL_KEYWORDS
)


@pytest.fixture
def normalizer():
    return EmailGatewayNormalizer()


@pytest.fixture
def proofpoint_phish_click():
    return {
        "type": "click",
        "action": "clicked",
        "threatStatus": "active",
        "threatName": "Phishing.Generic",
        "sender": "ceo-fake@company-corp.com",
        "recipients": ["cfo@bofa.com"],
        "subject": "Urgent Wire Transfer Required",
        "clickUrl": "http://evil-bank-login.com/login",
        "senderIP": "185.220.101.45",
        "score": 95,
        "eventTime": "2026-05-21T09:00:00Z"
    }


@pytest.fixture
def proofpoint_malware():
    return {
        "type": "message",
        "action": "blocked",
        "threatStatus": "active",
        "threatName": "Emotet.Banking.Trojan",
        "sender": "attacker@evil.com",
        "recipients": ["finance@bofa.com"],
        "subject": "Invoice Q2 2026",
        "attachment": "Invoice_Q2.doc",
        "senderIP": "45.142.100.10",
        "score": 98,
        "eventTime": "2026-05-21T08:00:00Z"
    }


@pytest.fixture
def proofpoint_bec():
    return {
        "type": "message",
        "action": "delivered",
        "threatStatus": "suspicious",
        "sender": "ceo@company-secure.com",
        "recipients": ["cfo@bofa.com"],
        "subject": (
            "Urgent - Confidential Wire Transfer"
        ),
        "senderIP": "10.0.0.1",
        "score": 75,
        "eventTime": "2026-05-21T10:00:00Z"
    }


@pytest.fixture
def proofpoint_spam():
    return {
        "type": "message",
        "action": "junked",
        "threatStatus": "spam",
        "sender": "newsletter@marketing.com",
        "recipients": ["user@bofa.com"],
        "subject": "Weekly Newsletter",
        "senderIP": "10.0.0.1",
        "score": 20,
        "eventTime": "2026-05-21T09:00:00Z"
    }


@pytest.fixture
def mimecast_phish_blocked():
    return {
        "Action": "Block",
        "Sender": "ceo@company-name.com",
        "Recipients": ["cfo@bofa.com"],
        "Subject": "Wire Transfer Authorization",
        "SenderIP": "185.220.101.45",
        "Route": "inbound",
        "RejectReason": "Phishing detected",
        "SpamScore": 85,
        "Datetime": "2026-05-21T09:00:00Z"
    }


@pytest.fixture
def mimecast_virus():
    return {
        "Action": "Block",
        "Sender": "attacker@evil.com",
        "Recipients": "victim@bofa.com",
        "Subject": "Payroll Update",
        "SenderIP": "45.142.100.10",
        "Route": "inbound",
        "Virus": "Trojan.Banker.Emotet",
        "SpamScore": 0,
        "Datetime": "2026-05-21T08:00:00Z"
    }


@pytest.fixture
def mimecast_url_blocked():
    return {
        "Action": "Allow",
        "Sender": "sender@company.com",
        "Recipients": ["user@bofa.com"],
        "Subject": "Please review document",
        "Route": "inbound",
        "URL": "http://evil-phish.com/steal",
        "URLAction": "block",
        "SpamScore": 30,
        "Datetime": "2026-05-21T09:00:00Z"
    }


@pytest.fixture
def defender_phish():
    return {
        "SenderFromAddress": "fake@evil.com",
        "RecipientEmailAddress": "exec@bofa.com",
        "Subject": "CEO Wire Transfer Request",
        "ThreatTypes": "Phish",
        "PhishConfidenceLevel": "High",
        "DeliveryAction": "Blocked",
        "DeliveryLocation": "Quarantine",
        "SCL": 9,
        "UrlCount": 2,
        "AttachmentCount": 0,
        "SenderIPv4": "185.220.101.45",
        "Timestamp": "2026-05-21T09:00:00Z"
    }


@pytest.fixture
def defender_malware():
    return {
        "SenderFromAddress": "infected@company.com",
        "RecipientEmailAddress": "user@bofa.com",
        "Subject": "Q2 Financial Report",
        "ThreatTypes": "Malware",
        "PhishConfidenceLevel": "Low",
        "DeliveryAction": "Blocked",
        "DeliveryLocation": "Quarantine",
        "SCL": 5,
        "UrlCount": 0,
        "AttachmentCount": 1,
        "SenderIPv4": "10.0.0.1",
        "Timestamp": "2026-05-21T09:00:00Z"
    }


@pytest.fixture
def bec_event_list():
    return [
        {
            "email_sender": "ceo@company-corp.com",
            "email_subject": (
                "urgent wire transfer needed today"
            ),
            "email_bec_signals": [
                "urgency_indicator",
                "financial_keyword_in_subject"
            ]
        },
        {
            "email_sender": "cfo@company-corp.com",
            "email_subject": (
                "invoice payment authorization"
            ),
            "email_bec_signals": [
                "financial_keyword_in_subject"
            ]
        }
    ]


# ============================================================
# INITIALIZATION TESTS
# ============================================================

class TestInitialization:

    def test_normalizer_initializes(self, normalizer):
        assert normalizer is not None

    def test_threat_risk_map_populated(self):
        assert "phishing" in EMAIL_THREAT_RISK
        assert "malware" in EMAIL_THREAT_RISK
        assert "bec" in EMAIL_THREAT_RISK

    def test_executive_roles_populated(self):
        assert "ceo" in EXECUTIVE_ROLES
        assert "cfo" in EXECUTIVE_ROLES

    def test_financial_keywords_populated(self):
        assert "wire transfer" in FINANCIAL_KEYWORDS
        assert "invoice" in FINANCIAL_KEYWORDS


# ============================================================
# VENDOR DETECTION TESTS
# ============================================================

class TestVendorDetection:

    def test_detects_proofpoint(
        self, normalizer, proofpoint_phish_click
    ):
        vendor = normalizer._detect_vendor(
            proofpoint_phish_click
        )
        assert vendor == "proofpoint"

    def test_detects_mimecast(
        self, normalizer, mimecast_phish_blocked
    ):
        vendor = normalizer._detect_vendor(
            mimecast_phish_blocked
        )
        assert vendor == "mimecast"

    def test_detects_defender(
        self, normalizer, defender_phish
    ):
        vendor = normalizer._detect_vendor(
            defender_phish
        )
        assert vendor == "defender_o365"

    def test_unknown_generic(self, normalizer):
        vendor = normalizer._detect_vendor(
            {"from": "test@test.com"}
        )
        assert vendor == "generic"


# ============================================================
# PROOFPOINT TESTS
# ============================================================

class TestProofpoint:

    def test_normalize_returns_dict(
        self, normalizer, proofpoint_phish_click
    ):
        result = normalizer.normalize(
            proofpoint_phish_click
        )
        assert isinstance(result, dict)

    def test_required_fields_present(
        self, normalizer, proofpoint_phish_click
    ):
        result = normalizer.normalize(
            proofpoint_phish_click
        )
        required = [
            "accessor_identity", "source_ip",
            "risk_score", "risk_reasons",
            "source_system", "email_sender",
            "email_subject", "email_action"
        ]
        for field in required:
            assert field in result

    def test_phish_click_high_risk(
        self, normalizer, proofpoint_phish_click
    ):
        result = normalizer.normalize(
            proofpoint_phish_click
        )
        assert result["risk_score"] >= 0.75

    def test_malware_blocked_high_risk(
        self, normalizer, proofpoint_malware
    ):
        result = normalizer.normalize(
            proofpoint_malware
        )
        assert result["risk_score"] >= 0.80

    def test_bec_signals_detected(
        self, normalizer, proofpoint_bec
    ):
        result = normalizer.normalize(proofpoint_bec)
        assert len(
            result.get("email_bec_signals", [])
        ) > 0

    def test_bec_elevates_risk(
        self, normalizer, proofpoint_bec
    ):
        result = normalizer.normalize(proofpoint_bec)
        assert result["risk_score"] >= 0.75

    def test_spam_low_risk(
        self, normalizer, proofpoint_spam
    ):
        result = normalizer.normalize(proofpoint_spam)
        assert result["risk_score"] <= 0.45

    def test_sender_captured(
    self, normalizer, proofpoint_phish_click
):
     result = normalizer.normalize(
        proofpoint_phish_click
    )
     assert "email_sender" in result

    def test_source_system_proofpoint(
        self, normalizer, proofpoint_phish_click
    ):
        result = normalizer.normalize(
            proofpoint_phish_click
        )
        assert "proofpoint" in result["source_system"]

    def test_risk_score_capped(
        self, normalizer, proofpoint_phish_click
    ):
        result = normalizer.normalize(
            proofpoint_phish_click
        )
        assert result["risk_score"] <= 1.0

    def test_empty_event_safe(self, normalizer):
        result = normalizer.normalize({})
        assert result["risk_score"] == 0.0

    def test_none_event_safe(self, normalizer):
        result = normalizer.normalize(None)
        assert result is not None


# ============================================================
# MIMECAST TESTS
# ============================================================

class TestMimecast:

    def test_mimecast_blocked_elevated(
        self, normalizer, mimecast_phish_blocked
    ):
        result = normalizer.normalize(
            mimecast_phish_blocked
        )
        assert result["risk_score"] >= 0.55

    def test_mimecast_virus_critical(
        self, normalizer, mimecast_virus
    ):
        result = normalizer.normalize(mimecast_virus)
        assert result["risk_score"] >= 0.88

    def test_mimecast_virus_captured(
        self, normalizer, mimecast_virus
    ):
        result = normalizer.normalize(mimecast_virus)
        assert "Emotet" in str(
            result.get("email_virus", "")
        )

    def test_mimecast_url_blocked_elevated(
        self, normalizer, mimecast_url_blocked
    ):
        result = normalizer.normalize(
            mimecast_url_blocked
        )
        assert result["risk_score"] >= 0.75

    def test_mimecast_source_system(
        self, normalizer, mimecast_phish_blocked
    ):
        result = normalizer.normalize(
            mimecast_phish_blocked
        )
        assert "mimecast" in result["source_system"]


# ============================================================
# DEFENDER O365 TESTS
# ============================================================

class TestDefenderO365:

    def test_defender_phish_high_risk(
        self, normalizer, defender_phish
    ):
        result = normalizer.normalize(defender_phish)
        assert result["risk_score"] >= 0.80

    def test_defender_malware_high_risk(
        self, normalizer, defender_malware
    ):
        result = normalizer.normalize(defender_malware)
        assert result["risk_score"] >= 0.80

    def test_defender_source_system(
        self, normalizer, defender_phish
    ):
        result = normalizer.normalize(defender_phish)
        assert "defender" in result["source_system"]

    def test_high_phish_confidence_elevated(
        self, normalizer, defender_phish
    ):
        result = normalizer.normalize(defender_phish)
        reasons = str(result["risk_reasons"])
        assert "phish" in reasons.lower()


# ============================================================
# BEC DETECTION TESTS
# ============================================================

class TestBECDetection:

    def test_bec_returns_dict(
        self, normalizer, bec_event_list
    ):
        result = normalizer.detect_bec(bec_event_list)
        assert isinstance(result, dict)

    def test_bec_detected(
        self, normalizer, bec_event_list
    ):
        result = normalizer.detect_bec(bec_event_list)
        assert result["bec_detected"] is True

    def test_bec_confidence_score(
        self, normalizer, bec_event_list
    ):
        result = normalizer.detect_bec(bec_event_list)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_no_bec_empty_events(self, normalizer):
        result = normalizer.detect_bec([])
        assert result["bec_detected"] is False

    def test_bec_has_recommendation(
        self, normalizer, bec_event_list
    ):
        result = normalizer.detect_bec(bec_event_list)
        assert "recommendation" in result

    def test_urgency_signal_detected(self, normalizer):
        signals = normalizer._detect_bec_signals(
            "URGENT - Confidential Wire Transfer",
            "ceo@company.com",
            ["cfo@bofa.com"],
            ""
        )
        assert len(signals) > 0

    def test_financial_keyword_detected(
        self, normalizer
    ):
        signals = normalizer._detect_bec_signals(
            "Wire transfer needed today",
            "sender@company.com",
            ["finance@bofa.com"],
            ""
        )
        assert "financial_keyword_in_subject" in signals

    def test_lookalike_domain_detected(self, normalizer):
        is_lookalike = normalizer._is_lookalike_domain(
            "ceo@company-secure.com"
        )
        assert is_lookalike is True

    def test_legitimate_domain_not_lookalike(
        self, normalizer
    ):
        is_lookalike = normalizer._is_lookalike_domain(
            "user@bofa.com"
        )
        assert is_lookalike is False