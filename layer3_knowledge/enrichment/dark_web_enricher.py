"""
Layer 3 — Knowledge Graph
Dark Web Enricher

Orchestrates all threat intelligence feeds
into a unified enrichment pipeline.

This is the MAIN ENTRY POINT for threat intel.
Coordinates:
    HaveIBeenPwned    (credential breaches)
    AlienVault OTX    (community IOCs)
    Abuse.ch          (malware, URLs, C2)
    Recorded Future   (dark web, criminal intel)

USAGE:
    enricher = DarkWebEnricher()
    
    # Enrich a security event
    result = enricher.enrich_event(data_event)
    
    # Get organization-wide dark web alerts
    alerts = enricher.get_org_alerts("bofa.com")
    
    # Elevate risk score with TI context
    elevated_score = enricher.elevate_risk_score(
        base_score=0.30,
        entity="jsmith@bofa.com",
        source_ip="185.220.101.45"
    )
"""

import logging
from datetime import datetime
from datetime import timezone
from typing import Optional

from layer3_knowledge.enrichment.hibp_feed import (
    HIBPFeed
)
from layer3_knowledge.enrichment.otx_feed import (
    OTXFeed
)
from layer3_knowledge.enrichment.abusech_feed import (
    AbusechFeed
)
from layer3_knowledge.enrichment.recorded_future_feed import (
    RecordedFutureFeed
)

logger = logging.getLogger(__name__)


class DarkWebEnricher:
    """
    Unified dark web and threat intelligence
    enrichment pipeline.

    Combines all feed sources into one
    coherent risk elevation framework.
    """

    def __init__(self):
        self.hibp = HIBPFeed()
        self.otx = OTXFeed()
        self.abusech = AbusechFeed()
        self.rf = RecordedFutureFeed()

    def enrich_event(
        self,
        data_event: dict
    ) -> dict:
        """
        Enrich a DataAccessEvent with threat
        intelligence from all sources.

        Args:
            data_event: DataAccessEvent dict

        Returns:
            Enriched event with TI context
            and elevated risk score
        """
        enrichment = {
            "ti_sources_checked": [],
            "ti_findings": [],
            "ti_risk_elevation": 0.0,
            "ti_risk_reasons": [],
            "dark_web_context": []
        }

        accessor = data_event.get(
            "accessor_identity", ""
        )
        source_ip = data_event.get("source_ip", "")
        base_score = float(
            data_event.get("risk_score", 0.0) or 0.0
        )

        # 1. Check IP reputation across all sources
        if source_ip:
            ip_enrichment = self._enrich_ip(source_ip)
            if ip_enrichment["max_risk"] > 0.3:
                enrichment["ti_findings"].append(
                    ip_enrichment
                )
                enrichment["ti_risk_elevation"] = max(
                    enrichment["ti_risk_elevation"],
                    ip_enrichment["max_risk"] * 0.4
                )
                enrichment["ti_risk_reasons"].append(
                    f"ip_threat_intel:"
                    f"{ip_enrichment['summary']}"
                )
            enrichment["ti_sources_checked"].append(
                "ip_reputation"
            )

        # 2. Check credential exposure for accessor
        if accessor and "@" in accessor:
            cred_result = self.hibp.check_email(
                accessor
            )
            if cred_result["breach_count"] > 0:
                enrichment["ti_findings"].append({
                    "type": "credential_breach",
                    "email": accessor,
                    "breach_count": (
                        cred_result["breach_count"]
                    ),
                    "risk_score": (
                        cred_result["risk_score"]
                    )
                })
                enrichment["ti_risk_elevation"] = max(
                    enrichment["ti_risk_elevation"],
                    cred_result["risk_score"] * 0.3
                )
                enrichment["ti_risk_reasons"].append(
                    f"credential_breach:"
                    f"{cred_result['breach_count']}"
                    f"_known_breaches"
                )
            enrichment["ti_sources_checked"].append(
                "hibp_credentials"
            )

        # 3. Check domain of accessor organization
        if accessor and "@" in accessor:
            domain = accessor.split("@")[-1]
            if domain and not domain.startswith(
                ("gmail", "yahoo", "hotmail", "outlook")
            ):
                dark_web = (
                    self.rf.get_dark_web_alerts(domain)
                )
                if dark_web:
                    enrichment["dark_web_context"] = (
                        dark_web
                    )
                    enrichment["ti_risk_elevation"] = max(
                        enrichment["ti_risk_elevation"],
                        0.20
                    )
                    enrichment["ti_risk_reasons"].append(
                        f"dark_web_alerts:"
                        f"{len(dark_web)}_active"
                    )
                enrichment["ti_sources_checked"].append(
                    "dark_web_monitoring"
                )

        # 4. Calculate final elevated score
        elevated_score = min(
            base_score + enrichment["ti_risk_elevation"],
            1.0
        )

        enrichment["original_risk_score"] = base_score
        enrichment["elevated_risk_score"] = elevated_score
        enrichment["elevation_applied"] = (
            elevated_score - base_score
        )

        return {
            **data_event,
            "risk_score": elevated_score,
            "risk_reasons": (
                data_event.get("risk_reasons", []) +
                enrichment["ti_risk_reasons"]
            ),
            "threat_intelligence": enrichment
        }

    def elevate_risk_score(
        self,
        base_score: float,
        entity: str = "",
        source_ip: str = "",
        domain: str = ""
    ) -> dict:
        """
        Elevate a risk score based on
        threat intelligence context.

        Used by ML ensemble for TI-aware scoring.

        Returns:
            dict with original, elevated scores
            and reasons for elevation
        """
        elevation = 0.0
        reasons = []

        # IP threat intelligence
        if source_ip:
            otx_result = self.otx.get_ip_reputation(
                source_ip
            )
            rf_result = self.rf.get_ip_intelligence(
                source_ip
            )
            c2_result = self.abusech.check_ip_c2(
                source_ip
            )

            ip_max = max(
                otx_result.get("risk_score", 0.0),
                rf_result.get("risk_score", 0.0),
                0.95 if c2_result.get("is_c2") else 0.0
            )

            if ip_max > 0.5:
                elevation = max(
                    elevation, ip_max * 0.35
                )
                reasons.append(
                    f"ip_threat_intel:{source_ip}"
                    f"_score_{ip_max:.2f}"
                )

        # Credential breach intelligence
        if entity and "@" in entity:
            hibp_result = self.hibp.check_email(
                entity
            )
            if hibp_result["breach_count"] > 0:
                elevation = max(
                    elevation,
                    hibp_result["risk_score"] * 0.25
                )
                reasons.append(
                    f"credential_exposed_in_"
                    f"{hibp_result['breach_count']}"
                    f"_breaches"
                )

        elevated = min(base_score + elevation, 1.0)

        return {
            "original_score": base_score,
            "elevated_score": elevated,
            "elevation": elevation,
            "reasons": reasons,
            "ti_enriched": len(reasons) > 0
        }

    def get_org_alerts(
        self, domain: str
    ) -> dict:
        """
        Get all threat intelligence alerts
        for an organization.

        IBM stakeholders use this to see
        what dark web activity targets their org.

        Args:
            domain: Organization domain

        Returns:
            Comprehensive threat picture
        """
        return {
            "domain": domain,
            "report_generated": _now(),
            "dark_web_alerts": (
                self.rf.get_dark_web_alerts(domain)
            ),
            "credential_alerts": (
                self.rf.get_credential_alerts(domain)
            ),
            "domain_reputation": (
                self.otx.get_domain_reputation(domain)
            ),
            "active_banking_trojans": (
                self.abusech.get_banking_trojan_c2s()
            ),
            "financial_intel": (
                self.rf.get_financial_sector_intelligence()
            )
        }

    def _enrich_ip(self, ip: str) -> dict:
        """Get unified IP risk from all sources"""
        otx = self.otx.get_ip_reputation(ip)
        rf = self.rf.get_ip_intelligence(ip)
        c2 = self.abusech.check_ip_c2(ip)

        max_risk = max(
            otx.get("risk_score", 0.0),
            rf.get("risk_score", 0.0),
            0.95 if c2.get("is_c2") else 0.0
        )

        sources = []
        if otx.get("risk_score", 0) > 0.3:
            sources.append(
                f"OTX:{otx['pulse_count']}pulses"
            )
        if rf.get("dark_web_mentions", 0) > 0:
            sources.append(
                f"RF:{rf['dark_web_mentions']}"
                f"dark_web_mentions"
            )
        if c2.get("is_c2"):
            sources.append(
                f"C2:{c2.get('malware_families','')}"
            )

        return {
            "ip": ip,
            "max_risk": max_risk,
            "otx_risk": otx.get("risk_score", 0.0),
            "rf_risk": rf.get("risk_score", 0.0),
            "is_c2": c2.get("is_c2", False),
            "sources": sources,
            "summary": (
                f"IP risk {max_risk:.2f} "
                f"from {len(sources)} sources"
            )
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")