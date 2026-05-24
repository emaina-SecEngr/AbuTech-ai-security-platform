"""
Layer 1 — Data Ingestion
Firewall Normalizer

Converts firewall logs from multiple vendors
into DataAccessEvent format.

VENDORS SUPPORTED:
    Palo Alto Networks:
        CEF (Common Event Format) via syslog
        Traffic logs, Threat logs, URL logs
        Wildfire malware logs
        
    Fortinet FortiGate:
        CEF format via syslog
        Traffic, UTM, IPS events
        
    Cisco ASA / Firepower:
        Syslog format (proprietary)
        ASA-6-302013, ASA-4-106023 etc.
        Firepower IPS alerts
        
    Check Point:
        CEF format via SmartLog
        Firewall, IPS, URL filtering

WHY FIREWALL LOGS MATTER FOR BANKS:
    Firewall is the FIRST line of defense.
    Every connection in or out passes through it.
    
    WHAT YOUR PLATFORM DETECTS:
    
    DATA EXFILTRATION:
    Large outbound transfers at unusual hours.
    Transfer to unusual destinations.
    IsolationForest: 524MB to Romania at 3am.
    
    C2 COMMUNICATION:
    Regular beaconing to suspicious IPs.
    LSTM: connection every 60 seconds.
    DNSClassifier: DGA domain resolved.
    
    LATERAL MOVEMENT:
    Internal hosts connecting to each other
    in unusual patterns.
    GNN: unusual east-west traffic graph.
    
    PORT SCANNING:
    Sequential port connections.
    IsolationForest: anomalous scan pattern.
    
    BLOCKED ATTACKS:
    Repeated blocked connection attempts.
    IPS signature matches.
    Geographic anomalies.

CEF FORMAT:
    CEF:Version|Device Vendor|Device Product|
    Device Version|Signature ID|Name|Severity|
    Extension fields
    
    Example Palo Alto CEF:
    CEF:0|Palo Alto Networks|PAN-OS|10.2|
    TRAFFIC|Traffic Log|3|
    src=10.0.1.105 dst=198.51.100.42
    spt=54321 dpt=443 proto=TCP
    act=allow bytes=524288000

USAGE:
    normalizer = FirewallNormalizer()
    event = normalizer.normalize(raw_log)
    
    # Palo Alto specific
    event = normalizer.normalize_palo_alto(log)
    
    # Fortinet specific
    event = normalizer.normalize_fortinet(log)
    
    # Cisco ASA specific
    event = normalizer.normalize_cisco_asa(log)
"""

import logging
import re
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)

# Firewall actions mapped to risk scores
FIREWALL_ACTION_RISK = {
    # Allowed traffic (may still be suspicious)
    "allow":    0.10,
    "allowed":  0.10,
    "permit":   0.10,
    "accept":   0.10,
    "pass":     0.10,

    # Blocked traffic
    "deny":     0.55,
    "denied":   0.55,
    "block":    0.55,
    "blocked":  0.55,
    "drop":     0.50,
    "dropped":  0.50,
    "reject":   0.50,
    "reset":    0.45,

    # Threat detected
    "threat":   0.85,
    "alert":    0.75,
    "sinkhole": 0.90,
}

# Palo Alto threat categories
PALO_ALTO_THREAT_RISK = {
    "virus":            0.90,
    "spyware":          0.85,
    "vulnerability":    0.80,
    "wildfire-virus":   0.95,
    "file":             0.60,
    "data":             0.70,
    "url":              0.50,
    "dos":              0.80,
}

# High risk destination ports
HIGH_RISK_PORTS = {
    22:    "SSH",
    23:    "TELNET",
    3389:  "RDP",
    445:   "SMB",
    1433:  "MSSQL",
    3306:  "MYSQL",
    5432:  "POSTGRESQL",
    6379:  "REDIS",
    27017: "MONGODB",
    4444:  "METASPLOIT",
    31337: "BACKDOOR",
    9001:  "TOR",
    9050:  "TOR_PROXY",
}

# Geographic high risk countries
HIGH_RISK_COUNTRIES = [
    "RU", "CN", "KP", "IR", "SY",
    "BY", "CU", "MM", "SD", "VE"
]


class FirewallNormalizer:
    """
    Normalizes firewall logs from multiple vendors
    into DataAccessEvent format.

    Handles: Palo Alto, Fortinet, Cisco ASA,
    Check Point.
    """

    def __init__(self):
        self.source_system = "firewall"

    def normalize(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize firewall log.
        Auto-detects vendor.

        Args:
            raw_event: Firewall log dict

        Returns:
            DataAccessEvent compatible dict
        """
        if not raw_event:
            return self._empty_event()

        vendor = self._detect_vendor(raw_event)

        if vendor == "palo_alto":
            return self.normalize_palo_alto(raw_event)
        elif vendor == "fortinet":
            return self.normalize_fortinet(raw_event)
        elif vendor == "cisco_asa":
            return self.normalize_cisco_asa(raw_event)
        elif vendor == "checkpoint":
            return self.normalize_checkpoint(raw_event)
        else:
            return self._normalize_generic(raw_event)

    def normalize_palo_alto(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize Palo Alto Networks firewall log.
        Handles Traffic, Threat, and URL log types.
        """
        log_type = raw_event.get(
            "log_type",
            raw_event.get("type", "TRAFFIC")
        ).upper()

        src_ip = raw_event.get(
            "src",
            raw_event.get("src_ip", "")
        )
        dst_ip = raw_event.get(
            "dst",
            raw_event.get("dst_ip", "")
        )
        src_port = int(
            raw_event.get("spt", raw_event.get(
                "src_port", 0
            ) or 0)
        )
        dst_port = int(
            raw_event.get("dpt", raw_event.get(
                "dst_port", 0
            ) or 0)
        )
        action = raw_event.get(
            "act",
            raw_event.get("action", "allow")
        ).lower()
        bytes_out = int(
            raw_event.get("out", raw_event.get(
                "bytes_sent", 0
            ) or 0)
        )
        bytes_in = int(
            raw_event.get("in", raw_event.get(
                "bytes_received", 0
            ) or 0)
        )
        total_bytes = bytes_out + bytes_in
        proto = raw_event.get(
            "proto",
            raw_event.get("protocol", "TCP")
        )
        app = raw_event.get(
            "app",
            raw_event.get("application", "")
        )
        src_user = raw_event.get(
            "suser",
            raw_event.get("src_user", "unknown")
        )
        dst_zone = raw_event.get(
            "cs3",
            raw_event.get("dst_zone", "")
        )
        src_zone = raw_event.get(
            "cs1",
            raw_event.get("src_zone", "")
        )
        threat_name = raw_event.get(
            "threatid",
            raw_event.get("threat_name", "")
        )
        severity = raw_event.get(
            "severity", ""
        )
        url = raw_event.get("request", "")
        src_country = raw_event.get(
            "src_country", ""
        )

        accessor = src_user if (
            src_user and src_user != "unknown"
        ) else src_ip

        risk_score, risk_reasons = (
            self._calculate_firewall_risk(
                action, src_ip, dst_ip,
                dst_port, total_bytes,
                log_type, threat_name,
                src_country, raw_event
            )
        )

        return {
            "accessor_identity": accessor,
            "accessor_type": (
                "human" if (
                    src_user and
                    "@" in str(src_user)
                ) else "service_account"
            ),
            "data_store_name": dst_ip,
            "data_path": (
                f"{proto}:{dst_port} {app}"
            ).strip(),
            "data_classification": "UNKNOWN",
            "bytes_accessed": total_bytes,
            "event_time": self._extract_time(
                raw_event
            ),
            "source_ip": src_ip,
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "firewall_palo_alto",
            "raw_event": raw_event,
            "fw_vendor": "palo_alto",
            "fw_action": action,
            "fw_log_type": log_type,
            "fw_src_ip": src_ip,
            "fw_dst_ip": dst_ip,
            "fw_src_port": src_port,
            "fw_dst_port": dst_port,
            "fw_protocol": proto,
            "fw_application": app,
            "fw_bytes_out": bytes_out,
            "fw_bytes_in": bytes_in,
            "fw_src_zone": src_zone,
            "fw_dst_zone": dst_zone,
            "fw_threat_name": threat_name,
            "fw_severity": severity,
            "fw_url": url[:500] if url else "",
            "fw_src_country": src_country
        }

    def normalize_fortinet(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize Fortinet FortiGate log.
        FortiGate uses key=value CEF format.
        """
        action = raw_event.get(
            "action",
            raw_event.get("act", "accept")
        ).lower()
        src_ip = raw_event.get(
            "srcip",
            raw_event.get("src", "")
        )
        dst_ip = raw_event.get(
            "dstip",
            raw_event.get("dst", "")
        )
        dst_port = int(
            raw_event.get("dstport", raw_event.get(
                "dpt", 0
            ) or 0)
        )
        src_port = int(
            raw_event.get("srcport", raw_event.get(
                "spt", 0
            ) or 0)
        )
        bytes_sent = int(
            raw_event.get("sentbyte", raw_event.get(
                "out", 0
            ) or 0)
        )
        bytes_recv = int(
            raw_event.get("rcvdbyte", raw_event.get(
                "in", 0
            ) or 0)
        )
        total_bytes = bytes_sent + bytes_recv
        proto = raw_event.get(
            "proto", raw_event.get("protocol", "TCP")
        )
        service = raw_event.get("service", "")
        user = raw_event.get(
            "user",
            raw_event.get("srcuser", "")
        )
        threat_name = raw_event.get(
            "attack",
            raw_event.get("virus", "")
        )
        src_country = raw_event.get(
            "srccountry", ""
        )
        policy_id = raw_event.get("policyid", "")
        log_type = raw_event.get(
            "type", "traffic"
        ).upper()

        accessor = user if user else src_ip

        risk_score, risk_reasons = (
            self._calculate_firewall_risk(
                action, src_ip, dst_ip,
                dst_port, total_bytes,
                log_type, threat_name,
                src_country, raw_event
            )
        )

        return {
            "accessor_identity": accessor,
            "accessor_type": (
                "human" if (
                    user and "@" in str(user)
                ) else "service_account"
            ),
            "data_store_name": dst_ip,
            "data_path": (
                f"{proto}:{dst_port} {service}"
            ).strip(),
            "data_classification": "UNKNOWN",
            "bytes_accessed": total_bytes,
            "event_time": self._extract_time(
                raw_event
            ),
            "source_ip": src_ip,
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "firewall_fortinet",
            "raw_event": raw_event,
            "fw_vendor": "fortinet",
            "fw_action": action,
            "fw_log_type": log_type,
            "fw_src_ip": src_ip,
            "fw_dst_ip": dst_ip,
            "fw_dst_port": dst_port,
            "fw_protocol": proto,
            "fw_bytes_out": bytes_sent,
            "fw_bytes_in": bytes_recv,
            "fw_threat_name": threat_name,
            "fw_src_country": src_country,
            "fw_policy_id": policy_id
        }

    def normalize_cisco_asa(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize Cisco ASA/Firepower syslog.
        ASA uses proprietary syslog message IDs.
        Common: ASA-6-302013, ASA-4-106023
        """
        message_id = raw_event.get(
            "message_id",
            raw_event.get("msg_id", "")
        )
        action = raw_event.get(
            "action", "permit"
        ).lower()
        src_ip = raw_event.get(
            "src_ip",
            raw_event.get("source_address", "")
        )
        dst_ip = raw_event.get(
            "dst_ip",
            raw_event.get("dest_address", "")
        )
        src_port = int(
            raw_event.get(
                "src_port",
                raw_event.get("source_port", 0)
            ) or 0
        )
        dst_port = int(
            raw_event.get(
                "dst_port",
                raw_event.get("dest_port", 0)
            ) or 0
        )
        proto = raw_event.get("protocol", "TCP")
        bytes_val = int(
            raw_event.get(
                "bytes",
                raw_event.get("byte_count", 0)
            ) or 0
        )
        interface = raw_event.get("interface", "")
        user = raw_event.get("user", "")
        reason = raw_event.get("reason", "")

        is_deny = any(
            kw in message_id
            for kw in ["106", "106023", "106001"]
        )
        if is_deny:
            action = "deny"

        accessor = user if user else src_ip

        risk_score, risk_reasons = (
            self._calculate_firewall_risk(
                action, src_ip, dst_ip,
                dst_port, bytes_val,
                "TRAFFIC", "",
                "", raw_event
            )
        )

        if message_id:
            risk_reasons.append(
                f"cisco_asa_msg:{message_id}"
            )

        return {
            "accessor_identity": accessor,
            "accessor_type": (
                "human" if (
                    user and "@" in str(user)
                ) else "service_account"
            ),
            "data_store_name": dst_ip,
            "data_path": (
                f"{proto}:{dst_port}"
            ),
            "data_classification": "UNKNOWN",
            "bytes_accessed": bytes_val,
            "event_time": self._extract_time(
                raw_event
            ),
            "source_ip": src_ip,
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "firewall_cisco_asa",
            "raw_event": raw_event,
            "fw_vendor": "cisco_asa",
            "fw_action": action,
            "fw_message_id": message_id,
            "fw_src_ip": src_ip,
            "fw_dst_ip": dst_ip,
            "fw_src_port": src_port,
            "fw_dst_port": dst_port,
            "fw_protocol": proto,
            "fw_bytes_out": bytes_val,
            "fw_interface": interface,
            "fw_reason": reason
        }

    def normalize_checkpoint(
        self, raw_event: dict
    ) -> dict:
        """
        Normalize Check Point firewall log.
        Check Point uses CEF via SmartLog.
        """
        action = raw_event.get(
            "action",
            raw_event.get("act", "accept")
        ).lower()
        src_ip = raw_event.get(
            "src",
            raw_event.get("src_ip", "")
        )
        dst_ip = raw_event.get(
            "dst",
            raw_event.get("dst_ip", "")
        )
        dst_port = int(
            raw_event.get(
                "dpt",
                raw_event.get("dst_port", 0)
            ) or 0
        )
        bytes_val = int(
            raw_event.get(
                "bytes",
                raw_event.get("out", 0)
            ) or 0
        )
        proto = raw_event.get("proto", "TCP")
        product = raw_event.get(
            "product", "Firewall"
        )
        blade = raw_event.get("blade", "")
        rule_name = raw_event.get("rule_name", "")
        threat_name = raw_event.get(
            "malware_name",
            raw_event.get("attack_info", "")
        )
        src_country = raw_event.get(
            "src_country", ""
        )

        risk_score, risk_reasons = (
            self._calculate_firewall_risk(
                action, src_ip, dst_ip,
                dst_port, bytes_val,
                "TRAFFIC", threat_name,
                src_country, raw_event
            )
        )

        return {
            "accessor_identity": src_ip,
            "accessor_type": "service_account",
            "data_store_name": dst_ip,
            "data_path": f"{proto}:{dst_port}",
            "data_classification": "UNKNOWN",
            "bytes_accessed": bytes_val,
            "event_time": self._extract_time(
                raw_event
            ),
            "source_ip": src_ip,
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "firewall_checkpoint",
            "raw_event": raw_event,
            "fw_vendor": "checkpoint",
            "fw_action": action,
            "fw_product": product,
            "fw_blade": blade,
            "fw_rule_name": rule_name,
            "fw_src_ip": src_ip,
            "fw_dst_ip": dst_ip,
            "fw_dst_port": dst_port,
            "fw_protocol": proto,
            "fw_bytes_out": bytes_val,
            "fw_threat_name": threat_name,
            "fw_src_country": src_country
        }

    def normalize_cef(
        self, cef_string: str
    ) -> dict:
        """
        Parse raw CEF string into dict
        then normalize.

        CEF format:
        CEF:0|Vendor|Product|Version|ID|Name|Sev|ext

        Args:
            cef_string: Raw CEF log line

        Returns:
            DataAccessEvent compatible dict
        """
        if not cef_string:
            return self._empty_event()

        try:
            parsed = self._parse_cef(cef_string)
            return self.normalize(parsed)
        except Exception as e:
            logger.warning(
                f"CEF parse failed: {e}"
            )
            return self._empty_event()

    def _parse_cef(
        self, cef_string: str
    ) -> dict:
        """Parse CEF string into dict"""
        result = {}

        if not cef_string.startswith("CEF:"):
            return result

        parts = cef_string.split("|", 8)
        if len(parts) >= 7:
            result["cef_version"] = parts[0].replace(
                "CEF:", ""
            )
            result["device_vendor"] = parts[1]
            result["device_product"] = parts[2]
            result["device_version"] = parts[3]
            result["signature_id"] = parts[4]
            result["name"] = parts[5]
            result["severity"] = parts[6]

            if "palo" in parts[1].lower():
                result["_vendor_hint"] = "palo_alto"
            elif "fortinet" in parts[1].lower():
                result["_vendor_hint"] = "fortinet"
            elif "cisco" in parts[1].lower():
                result["_vendor_hint"] = "cisco_asa"
            elif "check point" in parts[1].lower():
                result["_vendor_hint"] = "checkpoint"

        if len(parts) >= 8:
            ext = parts[7]
            pairs = re.findall(
                r'(\w+)=((?:[^\s\\]|\\[\s\\])*)',
                ext
            )
            for key, val in pairs:
                result[key] = val.replace(
                    "\\=", "="
                )

        return result

    def _calculate_firewall_risk(
        self,
        action: str,
        src_ip: str,
        dst_ip: str,
        dst_port: int,
        bytes_val: int,
        log_type: str,
        threat_name: str,
        src_country: str,
        raw_event: dict
    ) -> tuple:
        """Calculate risk score for firewall event"""
        risk = FIREWALL_ACTION_RISK.get(
            action.lower(), 0.20
        )
        reasons = [f"fw_action:{action}"]

        if threat_name:
            risk = max(risk, 0.85)
            reasons.append(
                f"threat_detected:{threat_name}"
            )

        if log_type == "THREAT":
            risk = max(risk, 0.80)
            reasons.append("threat_log_type")

        if src_ip.startswith("185.220"):
            risk = min(risk + 0.30, 1.0)
            reasons.append("tor_exit_node_src")

        if dst_ip.startswith("185.220"):
            risk = min(risk + 0.30, 1.0)
            reasons.append("tor_exit_node_dst")

        bytes_mb = bytes_val / (1024 * 1024)
        if bytes_mb > 500:
            risk = min(risk + 0.25, 1.0)
            reasons.append("large_transfer_500mb+")
        elif bytes_mb > 100:
            risk = min(risk + 0.15, 1.0)
            reasons.append("large_transfer_100mb+")

        if dst_port in HIGH_RISK_PORTS:
            service = HIGH_RISK_PORTS[dst_port]
            risk = min(risk + 0.15, 1.0)
            reasons.append(
                f"high_risk_port:{dst_port}_{service}"
            )

        event_time = self._extract_time(raw_event)
        try:
            hour = int(event_time[11:13])
            if hour < 6 or hour > 22:
                risk = min(risk + 0.10, 1.0)
                reasons.append("after_hours_traffic")
        except (ValueError, IndexError):
            pass

        if src_country in HIGH_RISK_COUNTRIES:
            risk = min(risk + 0.10, 1.0)
            reasons.append(
                f"high_risk_country:{src_country}"
            )

        return min(risk, 1.0), reasons

    def _detect_vendor(
        self, raw_event: dict
    ) -> str:
        """Auto-detect firewall vendor"""
        vendor_hint = raw_event.get(
            "_vendor_hint", ""
        )
        if vendor_hint:
            return vendor_hint

        device_vendor = raw_event.get(
            "device_vendor", ""
        ).lower()
        if "palo" in device_vendor:
            return "palo_alto"
        if "fortinet" in device_vendor:
            return "fortinet"
        if "cisco" in device_vendor:
            return "cisco_asa"
        if "check point" in device_vendor:
            return "checkpoint"

        if "threatid" in raw_event:
            return "palo_alto"
        if "srcip" in raw_event and (
            "sentbyte" in raw_event
        ):
            return "fortinet"
        if "message_id" in raw_event and str(
            raw_event.get("message_id", "")
        ).startswith("ASA"):
            return "cisco_asa"
        if "blade" in raw_event:
            return "checkpoint"
        if "policyid" in raw_event:
            return "fortinet"

        return "generic"

    def _extract_time(
        self, raw_event: dict
    ) -> str:
        """Extract timestamp from firewall log"""
        for field in [
            "rt", "start", "end",
            "timestamp", "time",
            "TimeReceived", "date"
        ]:
            val = raw_event.get(field, "")
            if val:
                return str(val)
        return _now()

    def _normalize_generic(
        self, raw_event: dict
    ) -> dict:
        """Generic firewall normalization"""
        action = (
            raw_event.get("action", "") or
            raw_event.get("act", "") or
            raw_event.get("Action", "allow")
        ).lower()

        src_ip = (
            raw_event.get("src", "") or
            raw_event.get("src_ip", "") or
            raw_event.get("srcip", "")
        )
        dst_ip = (
            raw_event.get("dst", "") or
            raw_event.get("dst_ip", "") or
            raw_event.get("dstip", "")
        )
        dst_port = int(
            raw_event.get("dpt", "") or
            raw_event.get("dst_port", "") or
            raw_event.get("dstport", 0) or 0
        )
        bytes_val = int(
            raw_event.get("bytes", "") or
            raw_event.get("out", 0) or 0
        )

        risk_score, risk_reasons = (
            self._calculate_firewall_risk(
                action, src_ip, dst_ip,
                dst_port, bytes_val,
                "TRAFFIC", "", "", raw_event
            )
        )

        return {
            "accessor_identity": src_ip or "unknown",
            "accessor_type": "service_account",
            "data_store_name": dst_ip or "unknown",
            "data_path": f"port:{dst_port}",
            "data_classification": "UNKNOWN",
            "bytes_accessed": bytes_val,
            "event_time": self._extract_time(
                raw_event
            ),
            "source_ip": src_ip,
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": "firewall_generic",
            "raw_event": raw_event,
            "fw_vendor": "unknown",
            "fw_action": action,
            "fw_src_ip": src_ip,
            "fw_dst_ip": dst_ip,
            "fw_dst_port": dst_port
        }

    def _empty_event(self) -> dict:
        return {
            "accessor_identity": "unknown",
            "accessor_type": "service_account",
            "data_store_name": "unknown",
            "data_path": "",
            "data_classification": "UNKNOWN",
            "bytes_accessed": 0,
            "event_time": _now(),
            "source_ip": "",
            "risk_score": 0.0,
            "risk_reasons": [],
            "source_system": "firewall",
            "raw_event": {}
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")