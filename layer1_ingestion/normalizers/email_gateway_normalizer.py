"""
Layer 1 — Data Ingestion
Email Gateway Security Normalizer

Converts email security events from multiple
vendors into DataAccessEvent format.

VENDORS SUPPORTED:
    Proofpoint TAP (Targeted Attack Protection):
        Click events (URL clicks)
        Message delivered events
        Message blocked events
        Attachment analysis
        VAP (Very Attacked People) data
    
    Mimecast:
        Rejection events
        Hold events
        URL protect clicks
        Attachment protect events
        Impersonation protect events
    
    Microsoft Defender for O365:
        Anti-phishing alerts
        Anti-malware alerts
        Safe Links click events
        Safe Attachments events
        (Also flows through Sentinel)

WHY EMAIL SECURITY MATTERS FOR BANKS:
    91% of cyberattacks start with phishing.
    BEC costs US businesses $2.7B annually.
    
    ATTACK SCENARIOS YOUR PLATFORM CATCHES:
    
    SPEAR PHISHING:
    Targeted email to CFO with fake invoice.
    Attacker impersonates CEO.
    YOUR PLATFORM: BEC pattern + wire transfer request.
    
    CREDENTIAL HARVESTING:
    Email with link to fake bank login page.
    User clicks. Enters credentials.
    YOUR PLATFORM: Click event + subsequent
    login from new location = account takeover.
    
    MALWARE DELIVERY:
    Email with malicious macro-enabled attachment.
    User opens. Macro executes PowerShell.
    YOUR PLATFORM: Email event + endpoint event
    connected in knowledge graph.
    
    BUSINESS EMAIL COMPROMISE:
    Attacker compromises executive email.
    Sends wire transfer request.
    YOUR PLATFORM: Behavioral anomaly in
    email patterns + financial transaction.

USAGE:
    normalizer = EmailGatewayNormalizer()
    event = normalizer.normalize(raw_event)
    event = normalizer.normalize_proofpoint(pp_event)
    event = normalizer.normalize_mimecast(mc_event)
    bec = normalizer.detect_bec(events)
"""

import logging
import re
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)

# Email threat types and risk scores
EMAIL_THREAT_RISK = {
    # Phishing
    "phish":            0.90,
    "phishing":         0.90,
    "spear_phish":      0.95,
    "spearphish":       0.95,
    "targeted":         0.90,

    # Malware
    "malware":          0.92,
    "virus":            0.88,
    "trojan":           0.90,
    "ransomware":       0.98,
    "macro":            0.85,

    # BEC
    "bec":              0.95,
    "impersonation":    0.88,
    "fraud":            0.85,
    "executive":        0.80,

    # Spam
    "spam":             0.30,
    "bulk":             0.20,
    "newsletter":       0.05,

    # URL threats
    "malicious_url":    0.85,
    "credential_harvest": 0.92,

    # Attachments
    "malicious_attachment": 0.90,
    "suspicious_attachment": 0.70,
}

# Proofpoint action types
PROOFPOINT_ACTION_RISK = {
    "delivered":    0.20,
    "blocked":      0.65,
    "quarantined":  0.55,
    "junked":       0.30,
    "discarded":    0.60,
    "clicked":      0.75,
    "allowed":      0.20,
}

# Mimecast action types
MIMECAST_ACTION_RISK = {
    "Allow":        0.15,
    "Block":        0.65,
    "Bounce":       0.30,
    "Hold":         0.55,
    "Reject":       0.65,
    "Delete":       0.60,
    "Tag":          0.35,
}

# Executive roles for BEC detection
EXECUTIVE_ROLES = [
    "ceo", "cfo", "coo", "ciso", "cto",
    "president", "vp", "vice president",
    "director", "managing director",
    "head of", "chief"
]

# Financial keywords for BEC detection
FINANCIAL_KEYWORDS = [
    "wire transfer", "bank transfer", "payment",
    "invoice", "urgent", "confidential",
    "account number", "routing number",
    "swift", "iban", "authorize",
    "approve", "immediately", "today only"
]


class EmailGatewayNormalizer:
    """
    Normalizes email security gateway events
    from multiple vendors into DataAccessEvent.

    Handles: Proofpoint TAP, Mimecast,
    Microsoft Defender for O365.
    """

    def __init__(self):
        self.source_system = "email_gateway"

    def normalize(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize email security event.
        Auto-detects vendor.

        Args:
            raw_event: Email security event dict

        Returns:
            DataAccessEvent compatible dict
        """
        if not raw_event:
            return self._empty_event()

        vendor = self._detect_vendor(raw_event)

        if vendor == "proofpoint":
            return self.normalize_proofpoint(raw_event)
        elif vendor == "mimecast":
            return self.normalize_mimecast(raw_event)
        elif vendor == "defender_o365":
            return self.normalize_defender_o365(
                raw_event
            )
        else:
            return self._normalize_generic(raw_event)

    def normalize_proofpoint(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize Proofpoint TAP event.
        Proofpoint is most common at banks.
        """
        event_type = raw_event.get(
            "type",
            raw_event.get("event_type", "")
        ).lower()

        msg = raw_event.get(
            "message",
            raw_event.get("msg", {})
        )
        if isinstance(msg, str):
            msg = {}

        sender = (
            raw_event.get("sender", "") or
            msg.get("fromAddress", [""])[0]
            if isinstance(
                msg.get("fromAddress"), list
            ) else msg.get("fromAddress", "")
        )
        recipients = (
            raw_event.get("recipients", []) or
            msg.get("toAddresses", [])
        )
        subject = (
            raw_event.get("subject", "") or
            msg.get("subject", "")
        )
        action = raw_event.get(
            "action",
            raw_event.get("disposition", "delivered")
        ).lower()
        threat_status = raw_event.get(
            "threatStatus",
            raw_event.get("threat_status", "")
        ).lower()
        threat_name = raw_event.get(
            "threatName",
            raw_event.get("malware_name", "")
        )
        url = raw_event.get(
            "url",
            raw_event.get("clickUrl", "")
        )
        attachment = raw_event.get(
            "attachment",
            raw_event.get("fileName", "")
        )
        sender_ip = raw_event.get(
            "senderIP",
            raw_event.get("sender_ip", "")
        )
        score = float(
            raw_event.get("score", 0) or 0
        )

        base_risk = PROOFPOINT_ACTION_RISK.get(
            action, 0.30
        )

        threat_risk = 0.0
        for threat_kw, risk in (
            EMAIL_THREAT_RISK.items()
        ):
            if threat_kw in threat_status.lower() or (
                threat_kw in threat_name.lower()
            ):
                threat_risk = max(threat_risk, risk)

        risk_score = max(base_risk, threat_risk)

        if score >= 90:
            risk_score = max(risk_score, 0.90)
        elif score >= 70:
            risk_score = max(risk_score, 0.75)

        risk_reasons = [
            f"proofpoint_action:{action}",
            f"threat_status:{threat_status}"
        ]

        bec_signals = self._detect_bec_signals(
            subject, sender, recipients,
            raw_event.get("body", "")
        )
        if bec_signals:
            risk_score = max(risk_score, 0.88)
            risk_reasons.extend(bec_signals)

        if event_type == "click":
            risk_score = max(risk_score, 0.75)
            risk_reasons.append(
                "user_clicked_malicious_url"
            )

        recipient_str = (
            recipients[0] if recipients else ""
        )

        return {
            "accessor_identity": sender,
            "accessor_type": "human",
            "data_store_name": recipient_str,
            "data_path": subject[:300],
            "data_classification": "PII",
            "bytes_accessed": 0,
            "event_time": raw_event.get(
                "eventTime",
                raw_event.get("timestamp", _now())
            ),
            "source_ip": sender_ip,
            "risk_score": min(risk_score, 1.0),
            "risk_reasons": risk_reasons,
            "source_system": "email_proofpoint",
            "raw_event": raw_event,
            "email_vendor": "proofpoint",
            "email_sender": sender,
            "email_recipients": recipients,
            "email_subject": subject[:300],
            "email_action": action,
            "email_threat_status": threat_status,
            "email_threat_name": threat_name,
            "email_url": url[:500] if url else "",
            "email_attachment": attachment,
            "email_score": score,
            "email_event_type": event_type,
            "email_bec_signals": bec_signals
        }

    def normalize_mimecast(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize Mimecast email security event.
        """
        action = raw_event.get(
            "Action",
            raw_event.get("action", "Allow")
        )
        sender = raw_event.get(
            "Sender",
            raw_event.get("sender", "")
        )
        recipients = raw_event.get(
            "Recipients",
            raw_event.get("recipients", [])
        )
        if isinstance(recipients, str):
            recipients = [recipients]
        subject = raw_event.get(
            "Subject",
            raw_event.get("subject", "")
        )
        sender_ip = raw_event.get(
            "SenderIP",
            raw_event.get("sender_ip", "")
        )
        route = raw_event.get("Route", "inbound")
        reject_reason = raw_event.get(
            "RejectReason",
            raw_event.get("reject_reason", "")
        )
        virus = raw_event.get(
            "Virus",
            raw_event.get("virus", "")
        )
        url = raw_event.get("URL", "")
        url_action = raw_event.get(
            "URLAction", ""
        )
        event_type = raw_event.get(
            "EventType",
            raw_event.get("type", "email")
        )
        spam_score = float(
            raw_event.get(
                "SpamScore",
                raw_event.get("spam_score", 0)
            ) or 0
        )

        base_risk = MIMECAST_ACTION_RISK.get(
            action, 0.30
        )

        risk_score = base_risk
        risk_reasons = [
            f"mimecast_action:{action}",
            f"mimecast_route:{route}"
        ]

        if virus:
            risk_score = max(risk_score, 0.92)
            risk_reasons.append(
                f"virus_detected:{virus}"
            )

        if reject_reason:
            risk_score = max(risk_score, 0.65)
            risk_reasons.append(
                f"reject_reason:{reject_reason}"
            )

        if url and url_action == "block":
            risk_score = max(risk_score, 0.80)
            risk_reasons.append(
                "malicious_url_blocked"
            )

        if spam_score >= 70:
            risk_score = max(risk_score, 0.60)
            risk_reasons.append(
                f"high_spam_score:{spam_score}"
            )

        bec_signals = self._detect_bec_signals(
            subject, sender,
            recipients if isinstance(
                recipients, list
            ) else [recipients],
            ""
        )
        if bec_signals:
            risk_score = max(risk_score, 0.85)
            risk_reasons.extend(bec_signals)

        recipient_str = (
            recipients[0]
            if isinstance(recipients, list)
            and recipients
            else str(recipients)
        )

        return {
            "accessor_identity": sender,
            "accessor_type": "human",
            "data_store_name": recipient_str,
            "data_path": subject[:300],
            "data_classification": "PII",
            "bytes_accessed": 0,
            "event_time": raw_event.get(
                "Datetime",
                raw_event.get("timestamp", _now())
            ),
            "source_ip": sender_ip,
            "risk_score": min(risk_score, 1.0),
            "risk_reasons": risk_reasons,
            "source_system": "email_mimecast",
            "raw_event": raw_event,
            "email_vendor": "mimecast",
            "email_sender": sender,
            "email_recipients": (
                recipients if isinstance(
                    recipients, list
                ) else [recipients]
            ),
            "email_subject": subject[:300],
            "email_action": action,
            "email_virus": virus,
            "email_url": url[:500] if url else "",
            "email_spam_score": spam_score,
            "email_event_type": event_type,
            "email_bec_signals": bec_signals
        }

    def normalize_defender_o365(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize Microsoft Defender for O365 event.
        These also flow through Sentinel
        via EmailEvents table.
        """
        action = raw_event.get(
            "Action",
            raw_event.get("DeliveryAction", "Delivered")
        )
        sender = raw_event.get(
            "SenderFromAddress",
            raw_event.get("sender", "")
        )
        recipient = raw_event.get(
            "RecipientEmailAddress",
            raw_event.get("recipient", "")
        )
        subject = raw_event.get(
            "Subject",
            raw_event.get("subject", "")
        )
        threat_types = raw_event.get(
            "ThreatTypes",
            raw_event.get("threat_types", "")
        )
        detection_methods = raw_event.get(
            "DetectionMethods", ""
        )
        phish_confidence = raw_event.get(
            "PhishConfidenceLevel", ""
        )
        spam_confidence = int(
            raw_event.get("SCL", 0) or 0
        )
        delivery_location = raw_event.get(
            "DeliveryLocation", ""
        )
        sender_ip = raw_event.get(
            "SenderIPv4",
            raw_event.get("SenderIPv6", "")
        )
        url_count = int(
            raw_event.get("UrlCount", 0) or 0
        )
        attachment_count = int(
            raw_event.get("AttachmentCount", 0) or 0
        )

        risk_score = 0.20
        risk_reasons = [
            f"defender_action:{action}"
        ]

        if "Phish" in str(threat_types):
            risk_score = max(risk_score, 0.88)
            risk_reasons.append("phishing_detected")

        if "Malware" in str(threat_types):
            risk_score = max(risk_score, 0.92)
            risk_reasons.append("malware_detected")

        if "Spam" in str(threat_types):
            risk_score = max(risk_score, 0.35)
            risk_reasons.append("spam_detected")

        if phish_confidence == "High":
            risk_score = max(risk_score, 0.90)
            risk_reasons.append(
                "high_phish_confidence"
            )

        if spam_confidence >= 8:
            risk_score = max(risk_score, 0.70)
            risk_reasons.append(
                f"high_spam_confidence:{spam_confidence}"
            )

        if delivery_location == "Inbox":
            if risk_score >= 0.70:
                risk_score = min(
                    risk_score + 0.10, 1.0
                )
                risk_reasons.append(
                    "threat_delivered_to_inbox"
                )

        bec_signals = self._detect_bec_signals(
            subject, sender, [recipient], ""
        )
        if bec_signals:
            risk_score = max(risk_score, 0.85)
            risk_reasons.extend(bec_signals)

        return {
            "accessor_identity": sender,
            "accessor_type": "human",
            "data_store_name": recipient,
            "data_path": subject[:300],
            "data_classification": "PII",
            "bytes_accessed": 0,
            "event_time": raw_event.get(
                "Timestamp",
                raw_event.get(
                    "TimeGenerated", _now()
                )
            ),
            "source_ip": sender_ip,
            "risk_score": min(risk_score, 1.0),
            "risk_reasons": risk_reasons,
            "source_system": "email_defender_o365",
            "raw_event": raw_event,
            "email_vendor": "defender_o365",
            "email_sender": sender,
            "email_recipients": [recipient],
            "email_subject": subject[:300],
            "email_action": action,
            "email_threat_types": threat_types,
            "email_phish_confidence": phish_confidence,
            "email_url_count": url_count,
            "email_attachment_count": attachment_count,
            "email_bec_signals": bec_signals
        }

    def detect_bec(
        self,
        events: list,
        window_hours: int = 24
    ) -> dict:
        """
        Detect Business Email Compromise pattern.
        BEC = attacker impersonates executive
        to request fraudulent wire transfer.

        Args:
            events: List of normalized email events
            window_hours: Analysis time window

        Returns:
            BEC analysis dict
        """
        if not events:
            return {
                "bec_detected": False,
                "confidence": 0.0
            }

        bec_signals = []
        executive_targets = []
        financial_requests = []

        for event in events:
            signals = event.get(
                "email_bec_signals", []
            )
            if signals:
                bec_signals.extend(signals)

            subject = event.get(
                "email_subject", ""
            ).lower()
            sender = event.get(
                "email_sender", ""
            ).lower()

            if any(
                role in subject or role in sender
                for role in EXECUTIVE_ROLES
            ):
                executive_targets.append(event)

            if any(
                kw in subject
                for kw in FINANCIAL_KEYWORDS
            ):
                financial_requests.append(event)

        bec_detected = (
            len(bec_signals) >= 2 or
            (executive_targets and financial_requests)
        )

        confidence = min(
            (len(bec_signals) * 0.20) +
            (len(executive_targets) * 0.25) +
            (len(financial_requests) * 0.25),
            1.0
        )

        return {
            "bec_detected": bec_detected,
            "confidence": confidence,
            "bec_signals": list(set(bec_signals)),
            "executive_events": len(
                executive_targets
            ),
            "financial_events": len(
                financial_requests
            ),
            "total_events_analyzed": len(events),
            "recommendation": (
                "BLOCK and investigate immediately. "
                "Contact targeted executive via phone."
                if bec_detected
                else "Normal email activity"
            )
        }

    def _detect_bec_signals(
        self,
        subject: str,
        sender: str,
        recipients: list,
        body: str
    ) -> list:
        """Detect BEC signals in email"""
        signals = []
        subject_lower = subject.lower()
        sender_lower = sender.lower()
        body_lower = body.lower()

        if any(
            role in subject_lower
            for role in EXECUTIVE_ROLES
        ):
            signals.append(
                "executive_impersonation_subject"
            )

        if any(
            kw in subject_lower
            for kw in FINANCIAL_KEYWORDS
        ):
            signals.append(
                "financial_keyword_in_subject"
            )

        if any(
            kw in body_lower
            for kw in FINANCIAL_KEYWORDS
        ):
            signals.append(
                "financial_keyword_in_body"
            )

        if "urgent" in subject_lower or (
            "confidential" in subject_lower
        ):
            signals.append("urgency_indicator")

        if "noreply" in sender_lower or (
            "no-reply" in sender_lower
        ):
            signals.append(
                "noreply_sender_suspicious"
            )

        if self._is_lookalike_domain(sender):
            signals.append(
                "lookalike_domain_detected"
            )

        return signals

    def _is_lookalike_domain(
        self, email: str
    ) -> bool:
        """Detect lookalike domain in sender"""
        if not email or "@" not in email:
            return False

        domain = email.split("@")[-1].lower()

        lookalike_patterns = [
            r".*-corp\.com$",
            r".*corp[0-9]+\.com$",
            r".*\.(net|org)$",
            r".*support.*\.com$",
            r".*secure.*\.com$",
            r".*banking.*\.com$",
        ]

        for pattern in lookalike_patterns:
            if re.match(pattern, domain):
                return True

        return False

    def _detect_vendor(
        self, raw_event: dict
    ) -> str:
        """Auto-detect email security vendor"""
        if "threatStatus" in raw_event:
            return "proofpoint"
        if "threatName" in raw_event and (
            "senderIP" in raw_event or
            "clickUrl" in raw_event
        ):
            return "proofpoint"
        if "disposition" in raw_event and (
            "score" in raw_event
        ):
            return "proofpoint"

        if "SpamScore" in raw_event:
            return "mimecast"
        if "RejectReason" in raw_event:
            return "mimecast"
        if (
            "Sender" in raw_event and
            "Action" in raw_event and
            "Route" in raw_event
        ):
            return "mimecast"

        if "PhishConfidenceLevel" in raw_event:
            return "defender_o365"
        if "ThreatTypes" in raw_event:
            return "defender_o365"
        if "DetectionMethods" in raw_event:
            return "defender_o365"
        if "SCL" in raw_event:
            return "defender_o365"
        if "DeliveryAction" in raw_event:
            return "defender_o365"

        return "generic"

    def _normalize_generic(
        self, raw_event: dict
    ) -> dict:
        """Generic email normalization"""
        sender = (
            raw_event.get("sender", "") or
            raw_event.get("from", "") or
            raw_event.get("From", "")
        )
        subject = (
            raw_event.get("subject", "") or
            raw_event.get("Subject", "")
        )
        action = (
            raw_event.get("action", "") or
            raw_event.get("Action", "delivered")
        ).lower()
        risk_score = PROOFPOINT_ACTION_RISK.get(
            action, 0.25
        )

        return {
            "accessor_identity": sender,
            "accessor_type": "human",
            "data_store_name": raw_event.get(
                "recipient", ""
            ),
            "data_path": subject[:300],
            "data_classification": "PII",
            "bytes_accessed": 0,
            "event_time": raw_event.get(
                "timestamp", _now()
            ),
            "source_ip": raw_event.get(
                "sender_ip", ""
            ),
            "risk_score": risk_score,
            "risk_reasons": [
                f"email_action:{action}"
            ],
            "source_system": "email_generic",
            "raw_event": raw_event,
            "email_vendor": "unknown",
            "email_sender": sender,
            "email_subject": subject[:300],
            "email_action": action,
            "email_bec_signals": []
        }

    def _empty_event(self) -> dict:
        return {
            "accessor_identity": "unknown",
            "accessor_type": "human",
            "data_store_name": "unknown",
            "data_path": "",
            "data_classification": "PII",
            "bytes_accessed": 0,
            "event_time": _now(),
            "source_ip": "",
            "risk_score": 0.0,
            "risk_reasons": [],
            "source_system": "email_gateway",
            "raw_event": {}
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")