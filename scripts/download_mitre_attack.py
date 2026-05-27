"""
Download MITRE ATT&CK Enterprise Matrix
and store locally in AbuTech platform.

USAGE:
    python scripts/download_mitre_attack.py

OUTPUT:
    layer3_knowledge/data/mitre_attack_db.json

WHAT IT DOWNLOADS:
    All ATT&CK Tactics (14 tactics)
    All ATT&CK Techniques (700+ techniques)
    All Sub-techniques (400+ sub-techniques)
    All Mitigations (40+ mitigations)
    All Data Sources
    Technique to Mitigation mappings
    Technique to Data Source mappings

SOURCE:
    MITRE ATT&CK STIX data
    https://github.com/mitre/cti
    Free. Public domain.
    Updated quarterly.

RUN THIS SCRIPT:
    When first setting up the platform.
    When MITRE releases a new version.
    Current version: ATT&CK v14
"""

import json
import os
import sys
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MITRE ATT&CK STIX bundle URL
MITRE_ATTACK_URL = (
    "https://raw.githubusercontent.com/"
    "mitre/cti/master/enterprise-attack/"
    "enterprise-attack.json"
)

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )),
    "layer3_knowledge", "data",
    "mitre_attack_db.json"
)


def download_mitre_attack():
    """
    Download MITRE ATT&CK STIX bundle
    and extract into structured format.
    """
    logger.info("Downloading MITRE ATT&CK data...")
    logger.info(f"Source: {MITRE_ATTACK_URL}")

    try:
        with urllib.request.urlopen(
            MITRE_ATTACK_URL, timeout=60
        ) as response:
            raw_data = json.loads(
                response.read().decode("utf-8")
            )
        logger.info("Download complete.")
    except Exception as e:
        logger.warning(
            f"Download failed: {e}\n"
            f"Using embedded fallback database."
        )
        return build_fallback_database()

    return parse_attack_bundle(raw_data)


def parse_attack_bundle(bundle: dict) -> dict:
    """Parse STIX bundle into structured format."""

    tactics = {}
    techniques = {}
    mitigations = {}
    relationships = []

    objects = bundle.get("objects", [])
    logger.info(
        f"Processing {len(objects)} STIX objects..."
    )

    # First pass — collect all objects
    for obj in objects:
        obj_type = obj.get("type", "")
        revoked = obj.get("revoked", False)
        deprecated = obj.get(
            "x_mitre_deprecated", False
        )

        if revoked or deprecated:
            continue

        # Extract tactics
        if obj_type == "x-mitre-tactic":
            tactic_id = _get_external_id(obj)
            if tactic_id:
                tactics[tactic_id] = {
                    "id": tactic_id,
                    "name": obj.get("name", ""),
                    "shortname": obj.get(
                        "x_mitre_shortname", ""
                    ),
                    "description": obj.get(
                        "description", ""
                    )[:500]
                }

        # Extract techniques
        elif obj_type == "attack-pattern":
            tech_id = _get_external_id(obj)
            if not tech_id:
                continue

            is_subtechnique = obj.get(
                "x_mitre_is_subtechnique", False
            )

            tactic_refs = []
            for phase in obj.get(
                "kill_chain_phases", []
            ):
                if phase.get(
                    "kill_chain_name"
                ) == "mitre-attack":
                    tactic_refs.append(
                        phase.get("phase_name", "")
                    )

            platforms = obj.get(
                "x_mitre_platforms", []
            )
            data_sources = obj.get(
                "x_mitre_data_sources", []
            )
            detection = obj.get(
                "x_mitre_detection", ""
            )[:500]

            techniques[tech_id] = {
                "id": tech_id,
                "name": obj.get("name", ""),
                "description": obj.get(
                    "description", ""
                )[:300],
                "is_subtechnique": is_subtechnique,
                "tactic_refs": tactic_refs,
                "platforms": platforms,
                "data_sources": data_sources,
                "detection": detection,
                "mitigations": [],
                "stix_id": obj.get("id", "")
            }

        # Extract mitigations
        elif obj_type == "course-of-action":
            mit_id = _get_external_id(obj)
            if mit_id and (
                mit_id.startswith("M")
            ):
                mitigations[mit_id] = {
                    "id": mit_id,
                    "name": obj.get("name", ""),
                    "description": obj.get(
                        "description", ""
                    )[:300],
                    "stix_id": obj.get("id", "")
                }

        # Collect relationships
        elif obj_type == "relationship":
            relationships.append(obj)

    # Second pass — link mitigations to techniques
    logger.info(
        f"Processing {len(relationships)} "
        f"relationships..."
    )

    mit_stix_map = {
        v["stix_id"]: v
        for v in mitigations.values()
    }
    tech_stix_map = {
        v["stix_id"]: k
        for k, v in techniques.items()
    }

    for rel in relationships:
        rel_type = rel.get(
            "relationship_type", ""
        )
        source = rel.get("source_ref", "")
        target = rel.get("target_ref", "")

        if rel_type == "mitigates":
            if (
                source in mit_stix_map and
                target in tech_stix_map
            ):
                tech_id = tech_stix_map[target]
                mit = mit_stix_map[source]
                if tech_id in techniques:
                    techniques[tech_id][
                        "mitigations"
                    ].append({
                        "id": mit["id"],
                        "name": mit["name"],
                        "description": mit[
                            "description"
                        ]
                    })

    db = {
        "version": "ATT&CK v14",
        "source": "https://github.com/mitre/cti",
        "tactics_count": len(tactics),
        "techniques_count": len(techniques),
        "mitigations_count": len(mitigations),
        "tactics": tactics,
        "techniques": techniques,
        "mitigations": mitigations
    }

    logger.info(
        f"Database built:\n"
        f"  Tactics:    {len(tactics)}\n"
        f"  Techniques: {len(techniques)}\n"
        f"  Mitigations:{len(mitigations)}"
    )

    return db


def build_fallback_database() -> dict:
    """
    Fallback database with most important
    financial sector ATT&CK techniques.
    Used when download fails.
    """
    logger.info(
        "Building fallback MITRE database..."
    )

    return {
        "version": "ATT&CK v14 (fallback)",
        "source": "embedded",
        "tactics": {
            "TA0001": {
                "id": "TA0001",
                "name": "Initial Access",
                "shortname": "initial-access",
                "description": (
                    "Techniques to gain initial "
                    "foothold in a network."
                )
            },
            "TA0003": {
                "id": "TA0003",
                "name": "Persistence",
                "shortname": "persistence",
                "description": (
                    "Techniques to maintain "
                    "access to systems."
                )
            },
            "TA0004": {
                "id": "TA0004",
                "name": "Privilege Escalation",
                "shortname": "privilege-escalation",
                "description": (
                    "Techniques to gain "
                    "higher-level permissions."
                )
            },
            "TA0006": {
                "id": "TA0006",
                "name": "Credential Access",
                "shortname": "credential-access",
                "description": (
                    "Techniques to steal "
                    "credentials."
                )
            },
            "TA0007": {
                "id": "TA0007",
                "name": "Discovery",
                "shortname": "discovery",
                "description": (
                    "Techniques to explore "
                    "the environment."
                )
            },
            "TA0008": {
                "id": "TA0008",
                "name": "Lateral Movement",
                "shortname": "lateral-movement",
                "description": (
                    "Techniques to move through "
                    "the environment."
                )
            },
            "TA0009": {
                "id": "TA0009",
                "name": "Collection",
                "shortname": "collection",
                "description": (
                    "Techniques to gather "
                    "data of interest."
                )
            },
            "TA0010": {
                "id": "TA0010",
                "name": "Exfiltration",
                "shortname": "exfiltration",
                "description": (
                    "Techniques to steal data "
                    "from the network."
                )
            },
            "TA0011": {
                "id": "TA0011",
                "name": "Command and Control",
                "shortname": "command-and-control",
                "description": (
                    "Techniques to communicate "
                    "with compromised systems."
                )
            },
            "TA0040": {
                "id": "TA0040",
                "name": "Impact",
                "shortname": "impact",
                "description": (
                    "Techniques to disrupt "
                    "availability or integrity."
                )
            }
        },
        "techniques": {
            "T1530": {
                "id": "T1530",
                "name": "Data from Cloud Storage",
                "description": (
                    "Adversaries may access data "
                    "from cloud storage. Many cloud "
                    "providers offer online storage "
                    "solutions such as S3, Azure Blob, "
                    "and GCS."
                ),
                "tactic_refs": ["collection"],
                "platforms": ["IaaS"],
                "data_sources": [
                    "Cloud Storage",
                    "Cloud Service"
                ],
                "detection": (
                    "Monitor cloud storage access "
                    "for unusual volumes or "
                    "after-hours access."
                ),
                "mitigations": [
                    {
                        "id": "M1022",
                        "name": (
                            "Restrict File and "
                            "Directory Permissions"
                        ),
                        "description": (
                            "Restrict S3 bucket "
                            "access using IAM "
                            "policies."
                        )
                    },
                    {
                        "id": "M1037",
                        "name": "Filter Network Traffic",
                        "description": (
                            "Block unauthorized "
                            "IP ranges at network "
                            "perimeter."
                        )
                    }
                ]
            },
            "T1078": {
                "id": "T1078",
                "name": "Valid Accounts",
                "description": (
                    "Adversaries may obtain and "
                    "abuse credentials of existing "
                    "accounts as a means of gaining "
                    "initial access, persistence, "
                    "privilege escalation, or "
                    "defense evasion."
                ),
                "tactic_refs": [
                    "initial-access",
                    "persistence",
                    "privilege-escalation",
                    "defense-evasion"
                ],
                "platforms": [
                    "Windows", "Linux",
                    "macOS", "IaaS",
                    "SaaS", "Azure AD"
                ],
                "data_sources": [
                    "Logon Session",
                    "User Account"
                ],
                "detection": (
                    "Monitor for unusual account "
                    "activity including impossible "
                    "travel and after-hours access."
                ),
                "mitigations": [
                    {
                        "id": "M1032",
                        "name": (
                            "Multi-factor Authentication"
                        ),
                        "description": (
                            "Enforce MFA for all "
                            "accounts especially "
                            "privileged accounts."
                        )
                    },
                    {
                        "id": "M1027",
                        "name": "Password Policies",
                        "description": (
                            "Enforce strong password "
                            "policies and rotation."
                        )
                    }
                ]
            },
            "T1110": {
                "id": "T1110",
                "name": "Brute Force",
                "description": (
                    "Adversaries may use brute "
                    "force techniques to gain "
                    "access to accounts when "
                    "passwords are unknown or "
                    "when password hashes are "
                    "obtained."
                ),
                "tactic_refs": ["credential-access"],
                "platforms": [
                    "Windows", "Linux",
                    "macOS", "Azure AD",
                    "SaaS", "IaaS"
                ],
                "data_sources": [
                    "User Account",
                    "Application Log"
                ],
                "detection": (
                    "Monitor authentication logs "
                    "for multiple failed attempts "
                    "from single source."
                ),
                "mitigations": [
                    {
                        "id": "M1032",
                        "name": (
                            "Multi-factor Authentication"
                        ),
                        "description": (
                            "MFA significantly reduces "
                            "effectiveness of brute "
                            "force attacks."
                        )
                    },
                    {
                        "id": "M1036",
                        "name": "Account Use Policies",
                        "description": (
                            "Implement account lockout "
                            "policies after failed "
                            "login attempts."
                        )
                    }
                ]
            },
            "T1621": {
                "id": "T1621",
                "name": (
                    "Multi-Factor Authentication "
                    "Request Generation"
                ),
                "description": (
                    "Adversaries may attempt to "
                    "bypass multi-factor "
                    "authentication by generating "
                    "numerous MFA requests to "
                    "cause MFA fatigue."
                ),
                "tactic_refs": ["credential-access"],
                "platforms": [
                    "Windows", "macOS",
                    "Linux", "Azure AD",
                    "SaaS"
                ],
                "data_sources": [
                    "Logon Session",
                    "User Account"
                ],
                "detection": (
                    "Monitor for unusual spikes "
                    "in MFA push notifications "
                    "for a single user."
                ),
                "mitigations": [
                    {
                        "id": "M1032",
                        "name": (
                            "Multi-factor Authentication"
                        ),
                        "description": (
                            "Use number matching MFA "
                            "to prevent fatigue attacks."
                        )
                    }
                ]
            },
            "T1071": {
                "id": "T1071",
                "name": (
                    "Application Layer Protocol"
                ),
                "description": (
                    "Adversaries may communicate "
                    "using application layer "
                    "protocols to avoid detection "
                    "and network filtering."
                ),
                "tactic_refs": [
                    "command-and-control"
                ],
                "platforms": [
                    "Windows", "Linux",
                    "macOS", "Network"
                ],
                "data_sources": [
                    "Network Traffic",
                    "Application Log"
                ],
                "detection": (
                    "Analyze network traffic for "
                    "unusual patterns in common "
                    "protocols like DNS and HTTP."
                ),
                "mitigations": [
                    {
                        "id": "M1031",
                        "name": (
                            "Network Intrusion Prevention"
                        ),
                        "description": (
                            "Use network IPS to "
                            "detect C2 patterns."
                        )
                    }
                ]
            },
            "T1048": {
                "id": "T1048",
                "name": (
                    "Exfiltration Over "
                    "Alternative Protocol"
                ),
                "description": (
                    "Adversaries may steal data "
                    "by exfiltrating it over a "
                    "different protocol than the "
                    "existing command and control "
                    "channel."
                ),
                "tactic_refs": ["exfiltration"],
                "platforms": [
                    "Linux", "macOS",
                    "Windows", "Network"
                ],
                "data_sources": [
                    "Network Traffic",
                    "Command"
                ],
                "detection": (
                    "Monitor for large outbound "
                    "transfers over unusual "
                    "protocols or ports."
                ),
                "mitigations": [
                    {
                        "id": "M1037",
                        "name": "Filter Network Traffic",
                        "description": (
                            "Block unusual outbound "
                            "protocols at perimeter."
                        )
                    }
                ]
            },
            "T1562": {
                "id": "T1562",
                "name": "Impair Defenses",
                "description": (
                    "Adversaries may maliciously "
                    "modify components of a victim "
                    "environment in order to hinder "
                    "or disable defensive mechanisms."
                ),
                "tactic_refs": ["defense-evasion"],
                "platforms": [
                    "Windows", "Linux",
                    "macOS", "IaaS"
                ],
                "data_sources": [
                    "Cloud Service",
                    "Windows Registry"
                ],
                "detection": (
                    "Monitor for changes to "
                    "security tool configurations "
                    "and audit logging settings."
                ),
                "mitigations": [
                    {
                        "id": "M1047",
                        "name": "Audit",
                        "description": (
                            "Regularly audit security "
                            "tool configurations and "
                            "logging settings."
                        )
                    }
                ]
            },
            "T1190": {
                "id": "T1190",
                "name": "Exploit Public-Facing Application",
                "description": (
                    "Adversaries may attempt to "
                    "take advantage of a weakness "
                    "in an Internet-facing computer "
                    "or program using software, "
                    "data, or commands."
                ),
                "tactic_refs": ["initial-access"],
                "platforms": [
                    "Windows", "Linux",
                    "macOS", "Network",
                    "IaaS"
                ],
                "data_sources": [
                    "Application Log",
                    "Network Traffic"
                ],
                "detection": (
                    "Monitor application logs "
                    "for exploitation indicators "
                    "such as SQLi and XSS patterns."
                ),
                "mitigations": [
                    {
                        "id": "M1048",
                        "name": (
                            "Application Isolation "
                            "and Sandboxing"
                        ),
                        "description": (
                            "Use WAF and application "
                            "sandboxing to limit "
                            "exploitation impact."
                        )
                    }
                ]
            },
            "T1136": {
                "id": "T1136",
                "name": "Create Account",
                "description": (
                    "Adversaries may create an "
                    "account to maintain access "
                    "to victim systems."
                ),
                "tactic_refs": ["persistence"],
                "platforms": [
                    "Windows", "Linux",
                    "macOS", "Azure AD",
                    "IaaS", "SaaS"
                ],
                "data_sources": [
                    "User Account"
                ],
                "detection": (
                    "Monitor for new account "
                    "creation especially outside "
                    "normal provisioning processes."
                ),
                "mitigations": [
                    {
                        "id": "M1032",
                        "name": (
                            "Multi-factor Authentication"
                        ),
                        "description": (
                            "Require MFA for account "
                            "creation processes."
                        )
                    }
                ]
            },
            "T1485": {
                "id": "T1485",
                "name": "Data Destruction",
                "description": (
                    "Adversaries may destroy data "
                    "and files on specific systems "
                    "or in large numbers on a network "
                    "to interrupt availability."
                ),
                "tactic_refs": ["impact"],
                "platforms": [
                    "Windows", "Linux",
                    "macOS", "IaaS"
                ],
                "data_sources": [
                    "Cloud Storage",
                    "File"
                ],
                "detection": (
                    "Monitor for bulk deletion "
                    "of files or cloud storage "
                    "objects especially by "
                    "service accounts."
                ),
                "mitigations": [
                    {
                        "id": "M1053",
                        "name": "Data Backup",
                        "description": (
                            "Ensure critical data "
                            "is backed up with "
                            "immutable backups."
                        )
                    }
                ]
            }
        },
        "mitigations": {
            "M1022": {
                "id": "M1022",
                "name": (
                    "Restrict File and "
                    "Directory Permissions"
                ),
                "description": (
                    "Restrict access by setting "
                    "directory and file permissions "
                    "to deny access to critical "
                    "systems."
                )
            },
            "M1032": {
                "id": "M1032",
                "name": "Multi-factor Authentication",
                "description": (
                    "Use two or more pieces of "
                    "evidence to authenticate users."
                )
            },
            "M1037": {
                "id": "M1037",
                "name": "Filter Network Traffic",
                "description": (
                    "Use network appliances to "
                    "filter ingress or egress "
                    "traffic."
                )
            }
        }
    }


def _get_external_id(obj: dict) -> str:
    """Extract MITRE external ID from STIX object"""
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id", "")
    return ""


def main():
    logger.info(
        "AbuTech MITRE ATT&CK Database Builder\n"
        "======================================"
    )

    db = download_mitre_attack()

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(db, f, indent=2)

    logger.info(
        f"\nDatabase saved to:\n{OUTPUT_PATH}\n"
        f"Tactics:    {db.get('tactics_count', len(db.get('tactics', {})))}\n"
        f"Techniques: {db.get('techniques_count', len(db.get('techniques', {})))}\n"
        f"Mitigations:{db.get('mitigations_count', len(db.get('mitigations', {})))}\n"
        f"\nRun this script quarterly to update."
    )


if __name__ == "__main__":
    main()