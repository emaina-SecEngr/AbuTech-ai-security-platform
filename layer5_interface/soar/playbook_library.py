"""
Layer 5 — Interface
SOAR Playbook Library

Production-ready security response playbooks.

Each playbook defines:
    name:               Playbook identifier
    description:        What it does
    trigger:            When it should run
    requires_approval:  HITL approval needed
    stop_on_failure:    Halt if step fails
    steps:              Sequence of actions

PLAYBOOK TYPES:
    Identity containment
    Network containment
    Data containment
    Notification

PARAMETER RESOLUTION:
    {event.field_name} pulls from the event.
    Literal values used as-is.

USAGE:
    from layer5_interface.soar.playbook_library\
        import (
            CONTAIN_COMPROMISED_ACCOUNT,
            BLOCK_MALICIOUS_IP,
            get_playbook,
            get_playbook_for_technique
        )

    engine.execute_playbook(
        playbook=CONTAIN_COMPROMISED_ACCOUNT,
        event=critical_event,
        approved_by="analyst@bank.com"
    )
"""


# ============================================================
# IDENTITY CONTAINMENT PLAYBOOKS
# ============================================================

CONTAIN_COMPROMISED_ACCOUNT = {
    "name": "contain_compromised_account",
    "description": (
        "Full containment of a compromised "
        "user or service account. Disables "
        "identity, revokes sessions, blocks "
        "source IP, preserves evidence."
    ),
    "trigger": {
        "min_risk_score": 0.80,
    },
    "requires_approval": True,
    "stop_on_failure": True,
    "mitre_techniques": ["T1078", "T1078.004"],
    "steps": [
        {
            "action": "disable_identity",
            "params": {
                "identity": "{event.accessor_identity}",
                "platform": "aws_iam"
            }
        },
        {
            "action": "revoke_sessions",
            "params": {
                "identity": "{event.accessor_identity}",
                "platform": "aws_sts"
            }
        },
        {
            "action": "block_ip",
            "params": {
                "ip_address": "{event.source_ip}",
                "platform": "palo_alto"
            }
        },
        {
            "action": "preserve_logs",
            "params": {
                "log_source": "aws_cloudtrail",
                "time_range_hours": 24
            }
        },
        {
            "action": "create_incident",
            "params": {
                "title": "Compromised account contained",
                "platform": "sentinel"
            }
        },
        {
            "action": "page_oncall",
            "params": {
                "message": "Compromised account contained automatically",
                "severity": "HIGH"
            }
        }
    ]
}


RESPOND_MFA_FATIGUE = {
    "name": "respond_mfa_fatigue",
    "description": (
        "Respond to MFA fatigue attack. "
        "Revokes sessions, forces MFA "
        "re-registration, blocks source IP."
    ),
    "trigger": {
        "min_risk_score": 0.70,
        "mitre_technique": "T1621"
    },
    "requires_approval": True,
    "stop_on_failure": False,
    "mitre_techniques": ["T1621"],
    "steps": [
        {
            "action": "revoke_sessions",
            "params": {
                "identity": "{event.accessor_identity}",
                "platform": "entra_id"
            }
        },
        {
            "action": "force_mfa_reregistration",
            "params": {
                "identity": "{event.accessor_identity}"
            }
        },
        {
            "action": "block_ip",
            "params": {
                "ip_address": "{event.source_ip}",
                "platform": "aws_waf"
            }
        },
        {
            "action": "notify_team",
            "params": {
                "team": "SOC",
                "message": "MFA fatigue attack contained"
            }
        }
    ]
}


ROTATE_EXPOSED_CREDENTIALS = {
    "name": "rotate_exposed_credentials",
    "description": (
        "Rotate credentials that may have "
        "been exposed. Rotates in CyberArk, "
        "revokes old sessions."
    ),
    "trigger": {
        "min_risk_score": 0.60,
    },
    "requires_approval": True,
    "stop_on_failure": True,
    "mitre_techniques": ["T1552"],
    "steps": [
        {
            "action": "rotate_credentials",
            "params": {
                "identity": "{event.accessor_identity}",
                "vault": "cyberark"
            }
        },
        {
            "action": "revoke_sessions",
            "params": {
                "identity": "{event.accessor_identity}",
                "platform": "aws_sts"
            }
        },
        {
            "action": "notify_team",
            "params": {
                "team": "IAM",
                "message": "Credentials rotated due to exposure"
            }
        }
    ]
}


# ============================================================
# NETWORK CONTAINMENT PLAYBOOKS
# ============================================================

BLOCK_MALICIOUS_IP = {
    "name": "block_malicious_ip",
    "description": (
        "Block a confirmed malicious IP "
        "across all network controls. "
        "Safe reversible action - "
        "auto-executes."
    ),
    "trigger": {
        "min_risk_score": 0.50,
    },
    "requires_approval": False,
    "stop_on_failure": False,
    "mitre_techniques": ["T1090"],
    "steps": [
        {
            "action": "block_ip",
            "params": {
                "ip_address": "{event.source_ip}",
                "platform": "palo_alto"
            }
        },
        {
            "action": "block_ip",
            "params": {
                "ip_address": "{event.source_ip}",
                "platform": "aws_waf"
            }
        },
        {
            "action": "block_ip",
            "params": {
                "ip_address": "{event.source_ip}",
                "platform": "cloudflare"
            }
        }
    ]
}


ISOLATE_COMPROMISED_ENDPOINT = {
    "name": "isolate_compromised_endpoint",
    "description": (
        "Isolate a compromised endpoint from "
        "the network. Preserves for forensics."
    ),
    "trigger": {
        "min_risk_score": 0.80,
    },
    "requires_approval": True,
    "stop_on_failure": True,
    "mitre_techniques": ["T1059", "T1204"],
    "steps": [
        {
            "action": "isolate_endpoint",
            "params": {
                "device_id": "{event.data_store_name}",
                "platform": "crowdstrike"
            }
        },
        {
            "action": "preserve_logs",
            "params": {
                "log_source": "endpoint_telemetry",
                "time_range_hours": 48
            }
        },
        {
            "action": "page_oncall",
            "params": {
                "message": "Endpoint isolated - investigate",
                "severity": "CRITICAL"
            }
        }
    ]
}


# ============================================================
# DATA CONTAINMENT PLAYBOOKS
# ============================================================

CONTAIN_DATA_EXFILTRATION = {
    "name": "contain_data_exfiltration",
    "description": (
        "Contain active data exfiltration. "
        "Disables identity, restricts data "
        "store, blocks IP, snapshots for "
        "forensics, starts compliance timer."
    ),
    "trigger": {
        "min_risk_score": 0.80,
        "mitre_technique": "T1530"
    },
    "requires_approval": True,
    "stop_on_failure": True,
    "mitre_techniques": ["T1530", "T1048"],
    "steps": [
        {
            "action": "disable_identity",
            "params": {
                "identity": "{event.accessor_identity}",
                "platform": "aws_iam"
            }
        },
        {
            "action": "restrict_data_store",
            "params": {
                "data_store": "{event.data_store_name}",
                "platform": "aws_s3"
            }
        },
        {
            "action": "block_ip",
            "params": {
                "ip_address": "{event.source_ip}",
                "platform": "palo_alto"
            }
        },
        {
            "action": "snapshot_for_forensics",
            "params": {
                "resource": "{event.data_store_name}",
                "resource_type": "s3"
            }
        },
        {
            "action": "preserve_logs",
            "params": {
                "log_source": "aws_cloudtrail",
                "time_range_hours": 72
            }
        },
        {
            "action": "start_compliance_timer",
            "params": {
                "regulation": "PCI-DSS"
            }
        },
        {
            "action": "notify_team",
            "params": {
                "team": "PCI Compliance",
                "message": "Data exfiltration contained - PCI review required"
            }
        },
        {
            "action": "create_incident",
            "params": {
                "title": "Data exfiltration containment",
                "platform": "sentinel"
            }
        }
    ]
}


RESPOND_MASS_DELETION = {
    "name": "respond_mass_deletion",
    "description": (
        "Respond to mass resource deletion "
        "(ransomware/destruction). Disables "
        "identity, snapshots survivors, "
        "escalates immediately."
    ),
    "trigger": {
        "min_risk_score": 0.80,
        "mitre_technique": "T1485"
    },
    "requires_approval": True,
    "stop_on_failure": False,
    "mitre_techniques": ["T1485", "T1486"],
    "steps": [
        {
            "action": "disable_identity",
            "params": {
                "identity": "{event.accessor_identity}",
                "platform": "aws_iam"
            }
        },
        {
            "action": "revoke_sessions",
            "params": {
                "identity": "{event.accessor_identity}",
                "platform": "aws_sts"
            }
        },
        {
            "action": "snapshot_for_forensics",
            "params": {
                "resource": "{event.data_store_name}",
                "resource_type": "rds"
            }
        },
        {
            "action": "page_oncall",
            "params": {
                "message": "MASS DELETION DETECTED - possible ransomware",
                "severity": "CRITICAL"
            }
        },
        {
            "action": "notify_team",
            "params": {
                "team": "Incident Response",
                "message": "Mass deletion event - activate IR plan"
            }
        }
    ]
}


# ============================================================
# DEFENSE EVASION PLAYBOOK
# ============================================================

RESPOND_LOG_TAMPERING = {
    "name": "respond_log_tampering",
    "description": (
        "Respond to audit log tampering. "
        "Critical defense evasion. Disables "
        "identity, re-enables logging, "
        "escalates."
    ),
    "trigger": {
        "min_risk_score": 0.70,
        "mitre_technique": "T1562.008"
    },
    "requires_approval": True,
    "stop_on_failure": True,
    "mitre_techniques": ["T1562.008"],
    "steps": [
        {
            "action": "disable_identity",
            "params": {
                "identity": "{event.accessor_identity}",
                "platform": "aws_iam"
            }
        },
        {
            "action": "preserve_logs",
            "params": {
                "log_source": "aws_cloudtrail",
                "time_range_hours": 168
            }
        },
        {
            "action": "page_oncall",
            "params": {
                "message": "Audit logging tampered - defender blinding attempt",
                "severity": "CRITICAL"
            }
        }
    ]
}


# ============================================================
# PLAYBOOK REGISTRY
# ============================================================

ALL_PLAYBOOKS = {
    "contain_compromised_account":
        CONTAIN_COMPROMISED_ACCOUNT,
    "respond_mfa_fatigue":
        RESPOND_MFA_FATIGUE,
    "rotate_exposed_credentials":
        ROTATE_EXPOSED_CREDENTIALS,
    "block_malicious_ip":
        BLOCK_MALICIOUS_IP,
    "isolate_compromised_endpoint":
        ISOLATE_COMPROMISED_ENDPOINT,
    "contain_data_exfiltration":
        CONTAIN_DATA_EXFILTRATION,
    "respond_mass_deletion":
        RESPOND_MASS_DELETION,
    "respond_log_tampering":
        RESPOND_LOG_TAMPERING,
}

# Maps MITRE technique to recommended playbook
TECHNIQUE_TO_PLAYBOOK = {
    "T1078": "contain_compromised_account",
    "T1078.004": "contain_compromised_account",
    "T1621": "respond_mfa_fatigue",
    "T1552": "rotate_exposed_credentials",
    "T1090": "block_malicious_ip",
    "T1059": "isolate_compromised_endpoint",
    "T1204": "isolate_compromised_endpoint",
    "T1530": "contain_data_exfiltration",
    "T1048": "contain_data_exfiltration",
    "T1485": "respond_mass_deletion",
    "T1486": "respond_mass_deletion",
    "T1562.008": "respond_log_tampering",
}


def get_playbook(name: str) -> dict:
    """Get playbook by name"""
    return ALL_PLAYBOOKS.get(name, {})


def get_playbook_for_technique(
    technique_id: str
) -> dict:
    """Get recommended playbook for technique"""
    playbook_name = TECHNIQUE_TO_PLAYBOOK.get(
        technique_id
    )
    if playbook_name:
        return ALL_PLAYBOOKS.get(playbook_name, {})
    return {}


def get_all_playbook_names() -> list:
    """Get list of all playbook names"""
    return list(ALL_PLAYBOOKS.keys())


def get_auto_execute_playbooks() -> list:
    """Get playbooks that auto-execute"""
    return [
        p for p in ALL_PLAYBOOKS.values()
        if not p.get("requires_approval", True)
    ]