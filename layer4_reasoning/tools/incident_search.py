"""
Layer 4 — Agent Tools
Tool 4: Past Incident Search

Searches past investigations for similar
incidents using pattern matching.

In production: uses ChromaDB vector search.
In development: uses JSON file storage.

Used by IntelAgent to find related incidents
and detect patterns across time.

USAGE BY AGENTS:
    results = search_past_incidents(
        accessor="svc_backup",
        pattern="data_exfiltration"
    )
    for incident in results:
        print(incident["summary"])
"""

import json
import logging
import os
from datetime import datetime
from datetime import timezone
from typing import List

logger = logging.getLogger(__name__)

# Storage file for past incidents
INCIDENTS_FILE = "data/past_incidents.json"


def search_past_incidents(
    accessor: str = None,
    pattern: str = None,
    source_ip: str = None,
    max_results: int = 5
) -> List[dict]:
    """
    Search past investigations for similar
    incidents.

    Args:
        accessor: Username or service account
        pattern: Attack pattern to search for
        source_ip: Source IP to search for
        max_results: Maximum results to return

    Returns:
        List of matching past incidents
    """
    incidents = _load_incidents()

    if not incidents:
        return []

    matches = []
    for incident in incidents:
        score = _match_score(
            incident, accessor, pattern, source_ip
        )
        if score > 0:
            matches.append({
                **incident,
                "match_score": score
            })

    # Sort by match score then date
    matches.sort(
        key=lambda x: (
            x["match_score"],
            x.get("timestamp", "")
        ),
        reverse=True
    )

    return matches[:max_results]


def store_incident(incident_data: dict) -> bool:
    """
    Store a completed investigation
    for future reference.

    Args:
        incident_data: Investigation result dict

    Returns:
        True if stored successfully
    """
    try:
        incidents = _load_incidents()

        # Build storage record
        record = {
            "id": incident_data.get(
                "event_id", f"inc_{len(incidents)}"
            ),
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "accessor": incident_data.get(
                "event_user", ""
            ),
            "host": incident_data.get(
                "event_host", ""
            ),
            "verdict": incident_data.get(
                "overall_verdict", "UNKNOWN"
            ),
            "severity": incident_data.get(
                "severity_rating", "UNKNOWN"
            ),
            "risk_score": incident_data.get(
                "overall_risk_score", 0.0
            ),
            "attack_type": incident_data.get(
                "triage_verdict", "UNKNOWN"
            ),
            "techniques": incident_data.get(
                "confirmed_techniques", []
            ),
            "summary": incident_data.get(
                "executive_summary", ""
            ),
            "source_ip": _extract_source_ip(
                incident_data
            ),
            "response_actions": incident_data.get(
                "response_actions", []
            )
        }

        incidents.append(record)

        # Keep last 1000 incidents only
        if len(incidents) > 1000:
            incidents = incidents[-1000:]

        _save_incidents(incidents)
        logger.info(
            f"Stored incident: {record['id']}"
        )
        return True

    except Exception as e:
        logger.error(
            f"Failed to store incident: {e}"
        )
        return False


def get_incident_stats(
    accessor: str = None
) -> dict:
    """
    Get statistics about past incidents.

    Args:
        accessor: Filter by accessor (optional)

    Returns:
        dict with incident statistics
    """
    incidents = _load_incidents()

    if accessor:
        incidents = [
            i for i in incidents
            if i.get("accessor") == accessor
        ]

    if not incidents:
        return {
            "total": 0,
            "critical": 0,
            "high": 0,
            "by_verdict": {},
            "repeat_offender": False,
            "summary": "No past incidents found."
        }

    critical = sum(
        1 for i in incidents
        if i.get("severity") == "CRITICAL"
    )
    high = sum(
        1 for i in incidents
        if i.get("severity") == "HIGH"
    )

    verdicts = {}
    for incident in incidents:
        v = incident.get("verdict", "UNKNOWN")
        verdicts[v] = verdicts.get(v, 0) + 1

    repeat_offender = (
        accessor is not None and
        len(incidents) >= 3
    )

    summary = (
        f"Found {len(incidents)} past incidents"
        f"{f' for {accessor}' if accessor else ''}. "
        f"Critical: {critical}, High: {high}. "
    )
    if repeat_offender:
        summary += (
            f"⚠️  REPEAT OFFENDER: "
            f"{accessor} has {len(incidents)} "
            f"incidents on record."
        )

    return {
        "total": len(incidents),
        "critical": critical,
        "high": high,
        "by_verdict": verdicts,
        "repeat_offender": repeat_offender,
        "summary": summary
    }


def _match_score(
    incident: dict,
    accessor: str = None,
    pattern: str = None,
    source_ip: str = None
) -> float:
    """Calculate how well an incident matches search criteria"""
    score = 0.0

    if accessor and incident.get("accessor") == accessor:
        score += 1.0

    if pattern:
        pattern_lower = pattern.lower()
        if pattern_lower in str(
            incident.get("verdict", "")
        ).lower():
            score += 0.8
        if pattern_lower in str(
            incident.get("attack_type", "")
        ).lower():
            score += 0.6
        if pattern_lower in str(
            incident.get("summary", "")
        ).lower():
            score += 0.3

    if source_ip and (
        incident.get("source_ip") == source_ip
    ):
        score += 0.9

    return score


def _extract_source_ip(
    incident_data: dict
) -> str:
    """Extract source IP from investigation data"""
    blast_radius = incident_data.get(
        "blast_radius", []
    )
    for entity in blast_radius:
        if entity.get("type") == "ip_address":
            return entity.get("entity", "")
    return ""


def _load_incidents() -> list:
    """Load incidents from storage file"""
    try:
        os.makedirs("data", exist_ok=True)
        if os.path.exists(INCIDENTS_FILE):
            with open(INCIDENTS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.debug(
            f"Could not load incidents: {e}"
        )
    return []


def _save_incidents(incidents: list) -> None:
    """Save incidents to storage file"""
    try:
        os.makedirs("data", exist_ok=True)
        with open(INCIDENTS_FILE, "w") as f:
            json.dump(incidents, f, indent=2)
    except Exception as e:
        logger.error(
            f"Could not save incidents: {e}"
        )