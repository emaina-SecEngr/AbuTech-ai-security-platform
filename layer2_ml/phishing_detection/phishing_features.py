"""
Layer 2 — ML Detection
Phishing Feature Extraction

Turns a normalized email event into a numerical
feature vector that a phishing classifier can score.

This is the feature-engineering layer — the same
pattern as the DNS/DGA detector, where raw input is
converted to the signals that actually distinguish
phishing from legitimate mail:

    URL signals     - IP-based URLs, lookalike domains,
                      URL shorteners, @ in URL, many dots
    Sender signals  - display-name vs domain mismatch,
                      reply-to mismatch, free-mail sender
    Content signals - urgency words, credential requests,
                      suspicious phrasing
    Attachment      - dangerous file extensions

Separating feature extraction from classification means
the same features can feed a weighted-scoring fallback
today and a trained model later.
"""

import re
import math

# Known URL shortener domains
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co",
    "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
    "cutt.ly", "shorturl.at",
}

# Free email providers (sender from these for a
# "corporate" request is a mild signal)
FREE_MAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com",
    "outlook.com", "aol.com", "protonmail.com",
    "mail.com", "gmx.com",
}

# Dangerous attachment extensions
DANGEROUS_EXTENSIONS = {
    ".exe", ".scr", ".vbs", ".js", ".jar", ".bat",
    ".cmd", ".com", ".pif", ".hta", ".wsf", ".ps1",
    ".docm", ".xlsm", ".pptm",  # macro-enabled
    ".iso", ".img",             # container evasion
}

# Urgency / social-engineering keywords
URGENCY_KEYWORDS = {
    "urgent", "immediately", "verify", "suspend",
    "suspended", "expire", "expired", "act now",
    "confirm your", "update your", "click here",
    "password", "account locked", "unusual activity",
    "verify your account", "confirm your identity",
    "limited time", "final notice", "your payment",
}

# Brands commonly impersonated (for lookalike checks)
IMPERSONATED_BRANDS = {
    "paypal", "microsoft", "apple", "amazon",
    "google", "netflix", "bank", "chase", "wellsfargo",
    "americanexpress", "amex", "docusign", "office365",
}


def shannon_entropy(s: str) -> float:
    """Compute Shannon entropy of a string (high
    entropy can indicate random/generated domains)."""
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    entropy = 0.0
    length = len(s)
    for c in counts.values():
        p = c / length
        entropy -= p * math.log2(p)
    return round(entropy, 4)


class PhishingFeatureExtractor:
    """
    Extracts phishing-relevant features from a
    normalized email event.
    """

    def extract(self, email_event: dict) -> dict:
        """
        Args:
            email_event: dict with sender, subject,
                         url, attachment, recipients
                         (from the email normalizers)

        Returns:
            A dict of named numerical/boolean features.
        """
        if not email_event or not isinstance(
            email_event, dict
        ):
            return self._empty_features()

        sender = str(
            email_event.get("sender", "")
        ).lower()
        subject = str(
            email_event.get("subject", "")
        ).lower()
        url = str(
            email_event.get("url", "")
            or email_event.get("clickUrl", "")
        ).lower()
        attachment = str(
            email_event.get("attachment", "")
        ).lower()
        reply_to = str(
            email_event.get("reply_to", "")
        ).lower()
        display_name = str(
            email_event.get("display_name", "")
        ).lower()

        features = {}

        # --- URL features ---
        features["has_url"] = 1 if url else 0
        features["url_is_ip"] = (
            1 if self._url_is_ip(url) else 0
        )
        features["url_is_shortener"] = (
            1 if self._is_shortener(url) else 0
        )
        features["url_has_at"] = (
            1 if "@" in url else 0
        )
        features["url_dot_count"] = url.count(".")
        features["url_has_https"] = (
            1 if url.startswith("https") else 0
        )
        features["url_lookalike_brand"] = (
            1 if self._has_lookalike_brand(url) else 0
        )
        features["url_entropy"] = shannon_entropy(
            self._url_domain(url)
        )

        # --- Sender features ---
        features["sender_free_mail"] = (
            1 if self._is_free_mail(sender) else 0
        )
        features["display_name_mismatch"] = (
            1 if self._display_mismatch(
                display_name, sender
            ) else 0
        )
        features["reply_to_mismatch"] = (
            1 if self._reply_to_mismatch(
                sender, reply_to
            ) else 0
        )
        features["sender_brand_impersonation"] = (
            1 if self._brand_in_display_not_domain(
                display_name, sender
            ) else 0
        )

        # --- Content features ---
        text = f"{subject}"
        features["urgency_keyword_count"] = (
            self._count_urgency(text)
        )
        features["has_credential_request"] = (
            1 if self._credential_request(text) else 0
        )

        # --- Attachment features ---
        features["has_attachment"] = (
            1 if attachment else 0
        )
        features["dangerous_attachment"] = (
            1 if self._dangerous_attachment(attachment)
            else 0
        )

        return features

    # --------------------------------------------------------
    # URL helpers
    # --------------------------------------------------------

    def _url_domain(self, url: str) -> str:
        """Extract the domain portion of a URL."""
        if not url:
            return ""
        # strip scheme
        u = re.sub(r"^https?://", "", url)
        # take up to first slash
        u = u.split("/")[0]
        # strip port and userinfo
        u = u.split("@")[-1].split(":")[0]
        return u

    def _url_is_ip(self, url: str) -> bool:
        """URL host is a raw IP address."""
        domain = self._url_domain(url)
        return bool(
            re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain)
        )

    def _is_shortener(self, url: str) -> bool:
        domain = self._url_domain(url)
        return domain in URL_SHORTENERS

    def _has_lookalike_brand(self, url: str) -> bool:
        """
        Domain contains a brand with a substitution
        (e.g. paypa1, micr0soft) or a brand as a
        subdomain of an unrelated domain.
        """
        domain = self._url_domain(url)
        if not domain:
            return False

        # homoglyph-style substitutions
        substituted = domain
        for a, b in [
            ("0", "o"), ("1", "l"), ("3", "e"),
            ("5", "s"), ("$", "s")
        ]:
            substituted = substituted.replace(a, b)

        for brand in IMPERSONATED_BRANDS:
            # exact brand present via substitution but
            # NOT the legitimate domain
            if brand in substituted and (
                not domain.endswith(f"{brand}.com")
            ):
                return True
        return False

    # --------------------------------------------------------
    # Sender helpers
    # --------------------------------------------------------

    def _sender_domain(self, sender: str) -> str:
        if "@" in sender:
            return sender.split("@")[-1].strip()
        return ""

    def _is_free_mail(self, sender: str) -> bool:
        return self._sender_domain(sender) in (
            FREE_MAIL_DOMAINS
        )

    def _display_mismatch(
        self, display_name: str, sender: str
    ) -> bool:
        """
        Display name mentions a domain/brand different
        from the actual sender domain.
        """
        if not display_name:
            return False
        sender_domain = self._sender_domain(sender)
        # display name claims a brand the sender domain
        # doesn't match
        for brand in IMPERSONATED_BRANDS:
            if brand in display_name and (
                brand not in sender_domain
            ):
                return True
        return False

    def _reply_to_mismatch(
        self, sender: str, reply_to: str
    ) -> bool:
        if not reply_to or not sender:
            return False
        return self._sender_domain(sender) != (
            self._sender_domain(reply_to)
        )

    def _brand_in_display_not_domain(
        self, display_name: str, sender: str
    ) -> bool:
        return self._display_mismatch(
            display_name, sender
        )

    # --------------------------------------------------------
    # Content / attachment helpers
    # --------------------------------------------------------

    def _count_urgency(self, text: str) -> int:
        count = 0
        for kw in URGENCY_KEYWORDS:
            if kw in text:
                count += 1
        return count

    def _credential_request(self, text: str) -> bool:
        cred_signals = [
            "password", "verify your account",
            "confirm your identity", "update your",
            "login", "sign in", "credentials",
        ]
        return any(s in text for s in cred_signals)

    def _dangerous_attachment(
        self, attachment: str
    ) -> bool:
        if not attachment:
            return False
        return any(
            attachment.endswith(ext)
            for ext in DANGEROUS_EXTENSIONS
        )

    def _empty_features(self) -> dict:
        return {
            "has_url": 0, "url_is_ip": 0,
            "url_is_shortener": 0, "url_has_at": 0,
            "url_dot_count": 0, "url_has_https": 0,
            "url_lookalike_brand": 0, "url_entropy": 0.0,
            "sender_free_mail": 0,
            "display_name_mismatch": 0,
            "reply_to_mismatch": 0,
            "sender_brand_impersonation": 0,
            "urgency_keyword_count": 0,
            "has_credential_request": 0,
            "has_attachment": 0,
            "dangerous_attachment": 0,
        }