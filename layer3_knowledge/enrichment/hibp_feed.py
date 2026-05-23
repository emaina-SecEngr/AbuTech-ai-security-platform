"""
Layer 3 — Knowledge Graph
HaveIBeenPwned Feed Integration

Checks if credentials or domains appear
in known data breaches.

WHY THIS MATTERS FOR BANKS:
    When a customer's email appears in a breach
    their BofA/Amex credentials may be at risk
    even if BofA was not breached directly.
    
    Attackers use CREDENTIAL STUFFING:
    1. Download breach database (free on dark web)
    2. Try every email/password combination
    3. 2-5% success rate = thousands of accounts
    
    YOUR PLATFORM WITH HIBP:
    Customer jsmith@gmail.com logs in.
    HIBP says: jsmith@gmail.com in 3 breaches.
    Risk score elevated automatically.
    MFA enforcement triggered.
    
    BEFORE the credential stuffing attack lands.
    Not after.

USAGE:
    feed = HIBPFeed()
    
    # Check single email
    result = feed.check_email("jsmith@gmail.com")
    
    # Check domain (enterprise - all emails)
    breaches = feed.check_domain("company.com")
    
    # Get breach context
    info = feed.get_breach_info("LinkedIn")

API: haveibeenpwned.com
COST: Free for single checks
      $3.50/month for API key (required)
      Enterprise: contact HIBP
"""

import logging
import os
import time
from datetime import datetime
from datetime import timezone
from typing import Optional

logger = logging.getLogger(__name__)

HIBP_API_BASE = "https://haveibeenpwned.com/api/v3"
HIBP_PASTE_BASE = "https://haveibeenpwned.com/api/v3/pasteaccount"


class HIBPFeed:
    """
    HaveIBeenPwned integration for credential
    breach detection.

    Checks if email addresses appear in known
    data breaches to enable proactive credential
    stuffing prevention.
    """

    def __init__(
        self,
        api_key: str = None,
        rate_limit_seconds: float = 1.5
    ):
        self.api_key = (
            api_key or
            os.getenv("HIBP_API_KEY", "")
        )
        self.rate_limit = rate_limit_seconds
        self._last_call = 0.0

    def check_email(
        self,
        email: str,
        truncate: bool = True
    ) -> dict:
        """
        Check if email appears in any breach.

        Args:
            email: Email address to check
            truncate: Return summary only (faster)

        Returns:
            dict with breach count and details
        """
        if not email or "@" not in email:
            return self._empty_result(email)

        if not self.api_key:
            logger.warning(
                "HIBP API key not configured. "
                "Using simulated result."
            )
            return self._simulated_result(email)

        try:
            import requests
            self._rate_limit()

            url = (
                f"{HIBP_API_BASE}/breachedaccount/"
                f"{email}"
            )
            params = {
                "truncateResponse": str(truncate).lower()
            }
            headers = {
                "hibp-api-key": self.api_key,
                "user-agent": "AbuTech-Security-Platform"
            }

            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                breaches = response.json()
                return self._format_result(
                    email, breaches
                )
            elif response.status_code == 404:
                return {
                    "email": email,
                    "breach_count": 0,
                    "breaches": [],
                    "risk_score": 0.0,
                    "risk_label": "CLEAN",
                    "recommendation": "No action needed"
                }
            else:
                logger.warning(
                    f"HIBP API error: "
                    f"{response.status_code}"
                )
                return self._empty_result(email)

        except Exception as e:
            logger.error(f"HIBP check failed: {e}")
            return self._empty_result(email)

    def check_domain(
        self,
        domain: str
    ) -> dict:
        """
        Check all breaches affecting a domain.
        Enterprise feature — all emails at domain.

        Args:
            domain: Domain to check (company.com)

        Returns:
            dict with breach summary for domain
        """
        if not self.api_key:
            return self._simulated_domain_result(domain)

        try:
            import requests
            self._rate_limit()

            url = (
                f"{HIBP_API_BASE}/breaches"
                f"?domain={domain}"
            )
            headers = {
                "hibp-api-key": self.api_key,
                "user-agent": "AbuTech-Security-Platform"
            }

            response = requests.get(
                url, headers=headers, timeout=10
            )

            if response.status_code == 200:
                breaches = response.json()
                return {
                    "domain": domain,
                    "breach_count": len(breaches),
                    "breaches": [
                        b.get("Name", "") for b in breaches
                    ],
                    "latest_breach": max(
                        [b.get("BreachDate", "")
                         for b in breaches],
                        default=""
                    ) if breaches else "",
                    "total_accounts_exposed": sum(
                        b.get("PwnCount", 0)
                        for b in breaches
                    ),
                    "risk_score": min(
                        len(breaches) * 0.1, 1.0
                    )
                }
            return {
                "domain": domain,
                "breach_count": 0,
                "risk_score": 0.0
            }

        except Exception as e:
            logger.error(
                f"HIBP domain check failed: {e}"
            )
            return {"domain": domain, "error": str(e)}

    def get_risk_score_for_email(
        self,
        email: str
    ) -> float:
        """
        Get normalized risk score for email.
        Used by ML ensemble for credential risk.

        Returns:
            float 0.0-1.0 breach risk score
        """
        result = self.check_email(email)
        return result.get("risk_score", 0.0)

    def _format_result(
        self,
        email: str,
        breaches: list
    ) -> dict:
        """Format HIBP response into platform format"""
        count = len(breaches)

        if count == 0:
            risk_score = 0.0
            risk_label = "CLEAN"
            recommendation = "No action needed"
        elif count <= 2:
            risk_score = 0.4
            risk_label = "LOW"
            recommendation = (
                "Monitor for credential stuffing"
            )
        elif count <= 5:
            risk_score = 0.65
            risk_label = "MEDIUM"
            recommendation = (
                "Enforce MFA. Prompt password reset."
            )
        else:
            risk_score = 0.85
            risk_label = "HIGH"
            recommendation = (
                "Force password reset immediately. "
                "Flag account for enhanced monitoring."
            )

        breach_names = []
        if isinstance(breaches, list):
            for b in breaches:
                if isinstance(b, dict):
                    breach_names.append(
                        b.get("Name", "Unknown")
                    )
                elif isinstance(b, str):
                    breach_names.append(b)

        return {
            "email": email,
            "breach_count": count,
            "breaches": breach_names,
            "risk_score": risk_score,
            "risk_label": risk_label,
            "recommendation": recommendation,
            "checked_at": _now()
        }

    def _simulated_result(self, email: str) -> dict:
        """
        Simulated result when no API key.
        Used in development and testing.
        """
        email_lower = email.lower()

        # Simulate high-risk known test emails
        if "test.breach" in email_lower:
            return self._format_result(
                email, ["LinkedIn", "Adobe",
                        "Dropbox", "MySpace",
                        "Yahoo", "Tumblr"]
            )
        if "moderate" in email_lower:
            return self._format_result(
                email, ["Adobe", "Dropbox"]
            )

        return self._format_result(email, [])

    def _simulated_domain_result(
        self, domain: str
    ) -> dict:
        """Simulated domain result for testing"""
        return {
            "domain": domain,
            "breach_count": 2,
            "breaches": ["LinkedIn", "Adobe"],
            "risk_score": 0.2,
            "simulated": True
        }

    def _empty_result(self, email: str) -> dict:
        """Empty result for invalid input"""
        return {
            "email": email,
            "breach_count": 0,
            "breaches": [],
            "risk_score": 0.0,
            "risk_label": "UNKNOWN",
            "recommendation": "Check skipped"
        }

    def _rate_limit(self):
        """Enforce HIBP rate limiting (1.5s between calls)"""
        elapsed = time.time() - self._last_call
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_call = time.time()


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")