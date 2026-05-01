"""
Layer 2 — ML Processing Engine
DNS NLP Model — Domain Feature Extraction

This module extracts features from domain name strings
for DGA detection and C2 beaconing identification.

Unlike previous models that used numerical features,
this module analyzes TEXT — the domain name string itself
contains rich security intelligence.

Three Feature Categories:

1. ENTROPY AND RANDOMNESS FEATURES
   How random is the domain name?
   DGA domains have high character entropy.
   Legitimate domains have low entropy —
   they follow human language patterns.

2. STRUCTURAL FEATURES
   How is the domain structured?
   How many subdomains?
   What TLD is used?
   How long are the labels?

3. LEXICAL FEATURES
   What characters appear in the domain?
   What is the ratio of consonants to vowels?
   Are there numbers mixed with letters?
   Does it contain known brand names?

4. BEHAVIORAL CONTEXT FEATURES
   What process made this DNS request?
   Is the TLD suspicious?
   Is it a dynamic DNS provider?

ATT&CK Coverage:
    T1568.002  Dynamic Resolution: DGA
    T1071.001  Application Layer Protocol: Web
    T1071.004  Application Layer Protocol: DNS
    T1583.001  Acquire Infrastructure: Domains
"""

import math
import re
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# DOMAIN INTELLIGENCE LISTS
# Security domain knowledge encoded as lookup sets
# ============================================================

# Dynamic DNS providers
# Frequently abused by malware for C2 infrastructure
# Legitimate use exists but warrants higher scrutiny
DYNAMIC_DNS_PROVIDERS = {
    "duckdns.org",
    "no-ip.com",
    "no-ip.org",
    "ddns.net",
    "hopto.org",
    "zapto.org",
    "myftp.biz",
    "myftp.org",
    "myvnc.com",
    "redirectme.net",
    "serveblog.net",
    "serveftp.com",
    "servegame.com",
    "servehttp.com",
    "serveminecraft.net",
    "sytes.net",
    "webhop.me",
    "chickenkiller.com",
    "ignorelist.com",
    "bounceme.net",
    "freedynamicdns.net"
}

# Suspicious TLDs
# Cheap or easily abused top-level domains
# Often used for malware infrastructure
SUSPICIOUS_TLDS = {
    ".xyz",
    ".top",
    ".club",
    ".online",
    ".site",
    ".tech",
    ".space",
    ".fun",
    ".pw",
    ".cc",
    ".tk",
    ".ml",
    ".ga",
    ".cf",
    ".gq",
    ".click",
    ".link",
    ".download",
    ".zip",
    ".review",
    ".country",
    ".kim",
    ".cricket",
    ".science",
    ".work",
    ".party",
    ".loan",
    ".racing",
    ".win",
    ".bid",
    ".trade",
    ".accountant",
    ".webcam"
}

# Trusted TLDs
# These are expensive or regulated
# Much less commonly abused
TRUSTED_TLDS = {
    ".gov",
    ".edu",
    ".mil",
    ".int",
    ".ac.uk",
    ".gov.uk"
}

# Known legitimate domains
# Used to reduce false positives
KNOWN_LEGITIMATE = {
    "google.com", "googleapis.com",
    "microsoft.com", "windows.com",
    "windowsupdate.com", "azure.com",
    "amazon.com", "amazonaws.com",
    "cloudflare.com", "akamai.com",
    "apple.com", "icloud.com",
    "facebook.com", "fbcdn.net",
    "twitter.com", "twimg.com",
    "github.com", "githubusercontent.com"
}

# C2 action words commonly used in malicious domains
# Attackers craft domains that look like
# legitimate update or CDN services
C2_ACTION_WORDS = {
    "update", "upgrade", "download", "install",
    "service", "support", "secure", "safety",
    "check", "verify", "confirm", "validate",
    "cdn", "content", "delivery", "static",
    "api", "gateway", "proxy", "relay",
    "bot", "agent", "client", "connect"
}


class DNSFeatureExtractor:
    """
    Extracts ML features from DNS domain names.

    Takes a domain name string and returns a feature
    vector capturing entropy, structure, and lexical
    properties that distinguish DGA domains from
    legitimate ones.

    Usage:
        extractor = DNSFeatureExtractor()
        features = extractor.extract("xjf8k2mp.duckdns.org")
        # Returns numpy array of features
    """

    def __init__(self):
        self.feature_names = self._define_feature_names()

    def extract(
        self,
        domain: str,
        requesting_process: str = ""
    ) -> Optional[np.ndarray]:
        """
        Extract feature vector from domain name.

        Args:
            domain: Full domain name string
                   e.g. "xjf8k2mp.duckdns.org"
            requesting_process: Process that made
                               the DNS request
                               e.g. "svchost.exe"

        Returns:
            numpy array of features or None if invalid
        """
        if not domain or not isinstance(domain, str):
            return None

        domain = domain.lower().strip()

        # Remove trailing dot if present
        if domain.endswith("."):
            domain = domain[:-1]

        try:
            features = {}

            features.update(
                self._entropy_features(domain)
            )
            features.update(
                self._structural_features(domain)
            )
            features.update(
                self._lexical_features(domain)
            )
            features.update(
                self._tld_features(domain)
            )
            features.update(
                self._behavioral_features(
                    domain, requesting_process
                )
            )

            # Build ordered vector
            vector = np.array(
                [features.get(name, 0)
                 for name in self.feature_names],
                dtype=np.float32
            )

            return vector

        except Exception as e:
            logger.error(
                f"DNS feature extraction failed "
                f"for {domain}: {e}"
            )
            return None

    def extract_from_ecs(
        self,
        ecs_event
    ) -> Optional[np.ndarray]:
        """
        Extract features from ECSNormalized DNS event.

        Args:
            ecs_event: ECSNormalized with DNS data
                      event.category = "dns"
                      destination.domain = queried domain

        Returns:
            numpy array of features or None
        """
        if ecs_event is None:
            return None

        if not hasattr(ecs_event, "destination"):
            return None

        if ecs_event.destination is None:
            return None

        domain = ecs_event.destination.domain or ""

        process_name = ""
        if hasattr(ecs_event, "process"):
            if ecs_event.process:
                process_name = (
                    ecs_event.process.name or ""
                )

        return self.extract(domain, process_name)

    # ============================================================
    # FEATURE GROUP EXTRACTORS
    # ============================================================

    def _entropy_features(self, domain: str) -> dict:
        """
        Shannon entropy features.

        High entropy = random characters = DGA
        Low entropy = English-like = legitimate

        Shannon entropy formula:
        H = -Σ p(x) × log2(p(x))
        where p(x) is the probability of character x

        Examples:
            google.com → entropy ≈ 2.25 (low)
            xjf8k2mp   → entropy ≈ 3.0  (high)
        """
        # Get the second level domain (before TLD)
        # e.g. xjf8k2mp from xjf8k2mp.duckdns.org
        parts = domain.split(".")
        sld = parts[0] if parts else domain

        # Full domain entropy
        full_entropy = self._shannon_entropy(domain)

        # SLD entropy (most informative)
        sld_entropy = self._shannon_entropy(sld)

        # Digit entropy — random domains mix digits
        digits_only = re.sub(r"[^0-9]", "", sld)
        digit_entropy = (
            self._shannon_entropy(digits_only)
            if digits_only else 0
        )

        return {
            "full_domain_entropy": full_entropy,
            "sld_entropy": sld_entropy,
            "digit_entropy": digit_entropy,
            "is_high_entropy": int(sld_entropy > 2.8),
            "is_very_high_entropy": int(
                sld_entropy > 4.0
            )
        }

    def _structural_features(self, domain: str) -> dict:
        """
        Domain structure features.

        DGA domains often have:
        - Unusual number of subdomains
        - Unusually long or short labels
        - Random label lengths
        """
        parts = domain.split(".")

        # Label analysis
        labels = parts[:-1] if len(parts) > 1 else parts
        label_lengths = [len(l) for l in labels]

        return {
            "domain_length": len(domain),
            "num_labels": len(parts),
            "num_subdomains": len(parts) - 2
                if len(parts) > 2 else 0,
            "sld_length": len(parts[0]),
            "max_label_length": (
                max(label_lengths) if label_lengths else 0
            ),
            "mean_label_length": (
                sum(label_lengths) / len(label_lengths)
                if label_lengths else 0
            ),
            "is_long_domain": int(len(domain) > 50),
            "is_very_long_domain": int(len(domain) > 75),
            "is_short_sld": int(len(parts[0]) < 5),
            "has_many_subdomains": int(len(parts) > 4)
        }

    def _lexical_features(self, domain: str) -> dict:
        """
        Character-level lexical features.

        DGA domains have distinctive character patterns:
        - High consonant to vowel ratio
          (random strings have few vowels)
        - Mixed letters and numbers
        - No recognizable words
        - Unusual character n-grams
        """
        # Remove TLD for analysis
        parts = domain.split(".")
        sld = parts[0] if parts else domain

        total_chars = len(sld) if sld else 1

        # Character type counts
        vowels = sum(1 for c in sld if c in "aeiou")
        consonants = sum(
            1 for c in sld
            if c.isalpha() and c not in "aeiou"
        )
        digits = sum(1 for c in sld if c.isdigit())
        special = sum(
            1 for c in sld
            if not c.isalnum() and c != "-"
        )

        # Ratios
        vowel_ratio = vowels / total_chars
        digit_ratio = digits / total_chars
        consonant_ratio = consonants / total_chars

        # Unique character ratio
        # DGA domains often have many unique chars
        unique_chars = len(set(sld))
        unique_ratio = unique_chars / total_chars

        # Consecutive consonants
        # Human words rarely have 4+ consecutive consonants
        max_consonant_run = 0
        current_run = 0
        for c in sld:
            if c.isalpha() and c not in "aeiou":
                current_run += 1
                max_consonant_run = max(
                    max_consonant_run, current_run
                )
            else:
                current_run = 0

        # Check for C2 action words
        has_action_word = int(
            any(word in domain for word in C2_ACTION_WORDS)
        )

        # Check for known legitimate domain
        is_known_legitimate = int(
            any(legit in domain
                for legit in KNOWN_LEGITIMATE)
        )

        # Hex-like appearance
        # DGA domains sometimes use hex characters
        hex_chars = set("0123456789abcdef")
        hex_ratio = (
            sum(1 for c in sld if c in hex_chars)
            / total_chars
        )
        looks_like_hex = int(
            hex_ratio > 0.8 and len(sld) >= 8
        )

        return {
            "vowel_ratio": vowel_ratio,
            "consonant_ratio": consonant_ratio,
            "digit_ratio": digit_ratio,
            "special_char_ratio": special / total_chars,
            "unique_char_ratio": unique_ratio,
            "max_consonant_run": max_consonant_run,
            "has_digits_in_sld": int(digits > 0),
            "has_action_word": has_action_word,
            "is_known_legitimate": is_known_legitimate,
            "looks_like_hex": looks_like_hex
        }

    def _tld_features(self, domain: str) -> dict:
        """
        Top-level domain features.

        TLD choice is a strong DGA indicator:
        .xyz, .top, .click → cheap, often malicious
        .gov, .edu         → regulated, rarely malicious
        .duckdns.org       → dynamic DNS, higher risk
        """
        # Extract TLD
        parts = domain.split(".")
        tld = f".{parts[-1]}" if parts else ""

        # Check dynamic DNS
        is_dynamic_dns = int(
            any(dyn in domain
                for dyn in DYNAMIC_DNS_PROVIDERS)
        )

        # Check suspicious TLD
        is_suspicious_tld = int(
            tld in SUSPICIOUS_TLDS
        )

        # Check trusted TLD
        is_trusted_tld = int(
            any(trusted in domain
                for trusted in TRUSTED_TLDS)
        )

        # Common legitimate TLDs
        common_tlds = {".com", ".net", ".org", ".io"}
        is_common_tld = int(tld in common_tlds)

        # Two-character TLD (country code)
        is_country_code = int(
            len(tld) == 3  # .xx format
        )

        return {
            "is_dynamic_dns": is_dynamic_dns,
            "is_suspicious_tld": is_suspicious_tld,
            "is_trusted_tld": is_trusted_tld,
            "is_common_tld": is_common_tld,
            "is_country_code_tld": is_country_code,
            "tld_length": len(tld)
        }

    def _behavioral_features(
        self,
        domain: str,
        requesting_process: str
    ) -> dict:
        """
        Behavioral context features.

        The process making the DNS request
        provides critical context.

        Your observation from the pipeline:
        svchost.exe querying xjf8k2mp.duckdns.org
        is far more suspicious than
        chrome.exe querying google.com

        ATT&CK T1055 Process Injection:
        Legitimate processes injected with malware
        make DNS requests they never should make.
        """
        process = requesting_process.lower()

        # System processes that should not
        # query random external domains
        suspicious_dns_processes = {
            "svchost.exe",     # Service host
            "lsass.exe",       # Auth process
            "csrss.exe",       # Client/server runtime
            "winlogon.exe",    # Login process
            "services.exe",    # Service control
            "wininit.exe"      # Windows init
        }

        is_suspicious_process = int(
            process in suspicious_dns_processes
        )

        # Browsers make lots of DNS requests — normal
        browser_processes = {
            "chrome.exe", "firefox.exe",
            "msedge.exe", "iexplore.exe",
            "safari.exe", "opera.exe"
        }
        is_browser = int(
            process in browser_processes
        )

        # Combined risk signal
        # System process + suspicious domain = very high risk
        is_suspicious_combination = int(
            is_suspicious_process == 1 and
            (
                self._is_dga_like(domain) or
                any(dyn in domain
                    for dyn in DYNAMIC_DNS_PROVIDERS)
            )
        )

        return {
            "requesting_process_suspicious": (
                is_suspicious_process
            ),
            "requesting_process_browser": is_browser,
            "is_suspicious_combination": (
                is_suspicious_combination
            )
        }

    # ============================================================
    # HELPER METHODS
    # ============================================================

    def _shannon_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of a string.

        H = -Σ p(x) × log2(p(x))

        Higher entropy = more random = more DGA-like
        """
        if not text:
            return 0.0

        # Character frequency
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1

        # Calculate entropy
        length = len(text)
        entropy = 0.0
        for count in freq.values():
            prob = count / length
            if prob > 0:
                entropy -= prob * math.log2(prob)

        return round(entropy, 4)

    def _is_dga_like(self, domain: str) -> bool:
        """
        Quick heuristic check for DGA-like domain.
        Used internally for combination features.
        """
        parts = domain.split(".")
        sld = parts[0] if parts else domain

        # High entropy
        if self._shannon_entropy(sld) > 3.5:
            return True

        # Short with mixed chars
        if len(sld) < 12:
            vowels = sum(1 for c in sld if c in "aeiou")
            if len(sld) > 0 and vowels / len(sld) < 0.2:
                return True

        return False

    def _define_feature_names(self) -> list:
        """Define complete ordered feature name list"""
        return [
            # Entropy features
            "full_domain_entropy",
            "sld_entropy",
            "digit_entropy",
            "is_high_entropy",
            "is_very_high_entropy",

            # Structural features
            "domain_length",
            "num_labels",
            "num_subdomains",
            "sld_length",
            "max_label_length",
            "mean_label_length",
            "is_long_domain",
            "is_very_long_domain",
            "is_short_sld",
            "has_many_subdomains",

            # Lexical features
            "vowel_ratio",
            "consonant_ratio",
            "digit_ratio",
            "special_char_ratio",
            "unique_char_ratio",
            "max_consonant_run",
            "has_digits_in_sld",
            "has_action_word",
            "is_known_legitimate",
            "looks_like_hex",

            # TLD features
            "is_dynamic_dns",
            "is_suspicious_tld",
            "is_trusted_tld",
            "is_common_tld",
            "is_country_code_tld",
            "tld_length",

            # Behavioral features
            "requesting_process_suspicious",
            "requesting_process_browser",
            "is_suspicious_combination"
        ]

    def get_feature_names(self) -> list:
        """Return feature names for explainability"""
        return self.feature_names

    def get_feature_count(self) -> int:
        """Return total feature count"""
        return len(self.feature_names)

    def get_security_context(self) -> dict:
        """
        Return security context for each feature group.
        Used by SHAP explainability layer.
        """
        return {
            "entropy": (
                "How random is the domain name? "
                "DGA domains have high entropy."
            ),
            "structural": (
                "How is the domain structured? "
                "DGA domains often have unusual lengths."
            ),
            "lexical": (
                "What characters appear? "
                "DGA domains have few vowels and "
                "mixed digits."
            ),
            "tld": (
                "What TLD is used? "
                "Dynamic DNS and cheap TLDs are "
                "higher risk."
            ),
            "behavioral": (
                "What process made the request? "
                "System processes querying random "
                "domains indicates injection."
            )
        }