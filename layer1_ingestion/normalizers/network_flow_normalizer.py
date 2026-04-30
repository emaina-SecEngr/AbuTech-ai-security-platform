"""
Layer 1 — Data Ingestion
Network Flow Normalizer — Zeek/Bro conn.log

Transforms Zeek network flow logs into
Elastic Common Schema (ECS) format.

Why Zeek Instead of CrowdStrike for Network Data:
    CrowdStrike captures: who connected to what
    Zeek captures: everything about the conversation

    CrowdStrike NetworkConnectIP4:
        source IP, dest IP, port, process name
        No flow statistics

    Zeek conn.log:
        Full flow statistics — duration, bytes,
        packets, flags, inter-arrival times
        These are exactly the 20 features your
        intrusion detection model needs

Zeek conn.log Format:
    Tab-separated values with these key fields:
    ts, uid, orig_h, orig_p, resp_h, resp_p,
    proto, service, duration, orig_bytes,
    resp_bytes, conn_state, missed_bytes,
    history, orig_pkts, orig_ip_bytes,
    resp_pkts, resp_ip_bytes

Connection to Layer 2:
    Every normalized Zeek flow feeds directly into
    NetworkIntrusionDetector via the ECS bridge
    in detector.py._extract_ecs_features()

    The 20 model features map exactly to
    ECSNetwork flow statistics fields.
"""

import logging
from datetime import datetime
from datetime import timezone
from typing import Optional

from layer1_ingestion.schema.ecs_schema import (
    ECSDestination,
    ECSEvent,
    ECSHost,
    ECSNormalized,
    ECSNetwork,
    ECSSource,
    ECSUser
)
from layer1_ingestion.normalizers.base_normalizer import (
    BaseNormalizer,
    NormalizationError
)

logger = logging.getLogger(__name__)


# ============================================================
# ZEEK CONN.LOG FIELD INDICES
#
# Zeek conn.log is tab-separated with these columns
# in this exact order. Field indices used for parsing.
# ============================================================

ZEEK_FIELDS = {
    "ts": 0,              # Unix timestamp
    "uid": 1,             # Unique connection ID
    "orig_h": 2,          # Originator IP (source)
    "orig_p": 3,          # Originator port
    "resp_h": 4,          # Responder IP (destination)
    "resp_p": 5,          # Responder port
    "proto": 6,           # Protocol tcp/udp/icmp
    "service": 7,         # Detected service
    "duration": 8,        # Flow duration in seconds
    "orig_bytes": 9,      # Bytes from originator
    "resp_bytes": 10,     # Bytes from responder
    "conn_state": 11,     # Connection state
    "local_orig": 12,     # Local originator flag
    "local_resp": 13,     # Local responder flag
    "missed_bytes": 14,   # Missed bytes
    "history": 15,        # Connection history flags
    "orig_pkts": 16,      # Packets from originator
    "orig_ip_bytes": 17,  # IP bytes from originator
    "resp_pkts": 18,      # Packets from responder
    "resp_ip_bytes": 19,  # IP bytes from responder
}

# Zeek connection states and their meanings
# Used for security context enrichment
ZEEK_CONN_STATES = {
    "S0": "Connection attempt, no reply",
    "S1": "Connection established, not terminated",
    "SF": "Normal establishment and termination",
    "REJ": "Connection attempt rejected",
    "S2": "Connection established, originator closing",
    "S3": "Connection established, responder closing",
    "RSTO": "Connection reset by originator",
    "RSTR": "Connection reset by responder",
    "RSTOS0": "Originator reset, no reply",
    "RSTRH": "Responder reset, no handshake",
    "SH": "Originator sent SYN, responder SYN-ACK",
    "SHR": "Responder sent SYN-ACK, no ACK",
    "OTH": "No SYN, mid-stream traffic"
}

# Suspicious connection states
# These indicate scanning, refused connections,
# or abnormal termination
SUSPICIOUS_CONN_STATES = {
    "S0",     # SYN sent, no response — scanning
    "REJ",    # Connection rejected — scanning
    "RSTOS0", # Reset with no response — scanning
    "RSTRH",  # Reset no handshake — scanning
    "SH",     # Incomplete handshake — scanning
    "SHR"     # Incomplete handshake — scanning
}


class NetworkFlowNormalizer(BaseNormalizer):
    """
    Normalizes Zeek network flow logs to ECS format.

    Handles two input formats:
    1. Zeek conn.log line (tab-separated string)
    2. Dictionary with Zeek field names

    Both produce identical ECSNormalized output
    that feeds directly into NetworkIntrusionDetector.

    Usage:
        normalizer = NetworkFlowNormalizer()

        # From Zeek conn.log line
        result = normalizer.normalize_line(log_line)

        # From dictionary
        result = normalizer.normalize(zeek_dict)
    """

    def __init__(self):
        super().__init__(
            source_name="zeek_network_flow"
        )

    def normalize_event(
        self,
        raw_event: dict
    ) -> ECSNormalized:
        """
        Normalize a Zeek flow dictionary to ECS.

        Args:
            raw_event: Dictionary with Zeek field names
                      Keys match ZEEK_FIELDS names

        Returns:
            ECSNormalized with full flow statistics
        """
        # ---- TIMESTAMP ----
        ts = raw_event.get("ts", 0)
        try:
            ts_float = float(ts)
            timestamp = self.convert_unix_seconds_to_iso(
                int(ts_float)
            )
        except (ValueError, TypeError):
            raise NormalizationError(
                f"Invalid timestamp: {ts}"
            )

        # ---- SOURCE AND DESTINATION ----
        orig_h = raw_event.get("orig_h", "")
        orig_p = self._safe_int(
            raw_event.get("orig_p", 0)
        )
        resp_h = raw_event.get("resp_h", "")
        resp_p = self._safe_int(
            raw_event.get("resp_p", 0)
        )

        source = ECSSource(
            ip=orig_h,
            port=orig_p,
            bytes=self._safe_int(
                raw_event.get("orig_bytes", 0)
            )
        )

        destination = ECSDestination(
            ip=resp_h,
            port=resp_p,
            bytes=self._safe_int(
                raw_event.get("resp_bytes", 0)
            )
        )

        # ---- NETWORK FLOW STATISTICS ----
        # These are the 20 features your intrusion
        # detector was trained on
        duration_sec = self._safe_float(
            raw_event.get("duration", 0)
        )
        orig_bytes = self._safe_float(
            raw_event.get("orig_bytes", 0)
        )
        resp_bytes = self._safe_float(
            raw_event.get("resp_bytes", 0)
        )
        orig_pkts = self._safe_float(
            raw_event.get("orig_pkts", 0)
        )
        resp_pkts = self._safe_float(
            raw_event.get("resp_pkts", 0)
        )
        orig_ip_bytes = self._safe_float(
            raw_event.get("orig_ip_bytes", 0)
        )
        resp_ip_bytes = self._safe_float(
            raw_event.get("resp_ip_bytes", 0)
        )

        # Convert duration to milliseconds
        duration_ms = duration_sec * 1000

        # Calculate derived features
        epsilon = 1e-10
        total_bytes = orig_bytes + resp_bytes
        total_pkts = orig_pkts + resp_pkts

        flow_bytes_per_sec = (
            total_bytes / (duration_sec + epsilon)
        )
        flow_packets_per_sec = (
            total_pkts / (duration_sec + epsilon)
        )

        # Packet length means
        fwd_packet_len_mean = (
            orig_bytes / (orig_pkts + epsilon)
        )
        bwd_packet_len_mean = (
            resp_bytes / (resp_pkts + epsilon)
        )

        # Packet length max estimated from IP bytes
        fwd_packet_len_max = (
            orig_ip_bytes / (orig_pkts + epsilon)
            if orig_pkts > 0 else 0
        )
        bwd_packet_len_max = (
            resp_ip_bytes / (resp_pkts + epsilon)
            if resp_pkts > 0 else 0
        )

        # Parse TCP flags from Zeek history string
        history = raw_event.get("history", "")
        syn_flags, rst_flags, fin_flags = (
            self._parse_history_flags(history)
        )

        # Parse connection state for security context
        conn_state = raw_event.get("conn_state", "")
        is_suspicious_state = int(
            conn_state in SUSPICIOUS_CONN_STATES
        )

        # Build ECSNetwork with full flow statistics
        network = ECSNetwork(
            protocol="ipv4",
            transport=raw_event.get("proto", "tcp"),
            direction="outbound",
            bytes=int(total_bytes),
            packets=int(total_pkts),

            # Flow statistics for intrusion detector
            duration_ms=duration_ms,
            fwd_packets=int(orig_pkts),
            bwd_packets=int(resp_pkts),
            fwd_bytes=int(orig_bytes),
            bwd_bytes=int(resp_bytes),
            fwd_packet_len_max=fwd_packet_len_max,
            fwd_packet_len_mean=fwd_packet_len_mean,
            bwd_packet_len_max=bwd_packet_len_max,
            bwd_packet_len_mean=bwd_packet_len_mean,
            flow_bytes_per_sec=flow_bytes_per_sec,
            flow_packets_per_sec=flow_packets_per_sec,

            # IAT fields default to 0
            # Requires Zeek packets.log for accuracy
            iat_mean=0.0,
            iat_std=0.0,
            fwd_iat_mean=0.0,
            bwd_iat_mean=0.0,

            # TCP flags from history string
            syn_flags=syn_flags,
            rst_flags=rst_flags,
            psh_flags=0,
            ack_flags=0,

            # Window size not in conn.log
            init_win_bytes_fwd=0
        )

        # ---- HOST ----
        host = ECSHost(
            ip=orig_h,
            os_platform="unknown"
        )

        # ---- USER ----
        user = ECSUser()

        # ---- SECURITY SEVERITY ASSESSMENT ----
        severity = self._assess_flow_severity(
            conn_state=conn_state,
            flow_bytes_per_sec=flow_bytes_per_sec,
            flow_packets_per_sec=flow_packets_per_sec,
            orig_pkts=orig_pkts,
            resp_pkts=resp_pkts,
            syn_flags=syn_flags,
            rst_flags=rst_flags,
            resp_p=resp_p
        )

        # ---- ECS EVENT ----
        uid = raw_event.get("uid", self.generate_event_id())
        ecs_event = ECSEvent(
            id=uid,
            category="network",
            type="connection",
            dataset="zeek.conn",
            provider="zeek",
            created=timestamp,
            severity=severity
        )

        return ECSNormalized(
            timestamp=timestamp,
            event=ecs_event,
            user=user,
            host=host,
            source=source,
            destination=destination,
            network=network,
            data_source="zeek_network_flow"
        )

    def normalize_line(
        self,
        log_line: str
    ) -> Optional[ECSNormalized]:
        """
        Normalize a raw Zeek conn.log line.

        Zeek log lines are tab-separated strings.
        Lines starting with # are comments/headers.

        Args:
            log_line: Single line from Zeek conn.log

        Returns:
            ECSNormalized or None if line is header/invalid
        """
        line = log_line.strip()

        # Skip Zeek header lines
        if line.startswith("#") or not line:
            return None

        # Parse tab-separated fields
        fields = line.split("\t")

        if len(fields) < 15:
            logger.warning(
                f"Zeek line has too few fields: "
                f"{len(fields)}"
            )
            return None

        # Build dictionary from field names
        zeek_dict = {}
        for field_name, idx in ZEEK_FIELDS.items():
            if idx < len(fields):
                value = fields[idx]
                # Zeek uses - for missing values
                zeek_dict[field_name] = (
                    "" if value == "-" else value
                )

        return self.normalize(zeek_dict)

    def normalize_file(
        self,
        file_path: str
    ) -> list:
        """
        Normalize an entire Zeek conn.log file.

        Used for batch processing historical flows
        in Databricks notebooks.

        Args:
            file_path: Path to Zeek conn.log file

        Returns:
            List of ECSNormalized events
        """
        results = []

        try:
            with open(file_path, "r") as f:
                for line in f:
                    result = self.normalize_line(line)
                    if result:
                        results.append(result)

            logger.info(
                f"Normalized {len(results)} flows "
                f"from {file_path}"
            )

        except FileNotFoundError:
            logger.error(
                f"Zeek log file not found: {file_path}"
            )

        return results

    # ============================================================
    # SECURITY ASSESSMENT
    # ============================================================

    def _assess_flow_severity(
        self,
        conn_state: str,
        flow_bytes_per_sec: float,
        flow_packets_per_sec: float,
        orig_pkts: float,
        resp_pkts: float,
        syn_flags: int,
        rst_flags: int,
        resp_p: int
    ) -> int:
        """
        Assess initial severity of network flow.

        Fast rule-based pre-filter before Layer 2
        ML scoring. Flags obviously suspicious flows
        immediately without waiting for ML model.
        """
        severity = 0

        # Suspicious connection state
        # REJ, S0 indicate scanning activity
        if conn_state in SUSPICIOUS_CONN_STATES:
            severity += 30

        # Very high packet rate — DoS indicator
        if flow_packets_per_sec > 10000:
            severity += 40

        # Very high byte rate — DDoS indicator
        if flow_bytes_per_sec > 1_000_000:
            severity += 35

        # One directional — scanning or DoS
        if orig_pkts > 10 and resp_pkts == 0:
            severity += 25

        # SYN without ACK — incomplete handshake
        if syn_flags > 0 and rst_flags > 0:
            severity += 20

        # Suspicious destination ports
        suspicious_ports = {
            4444,   # Metasploit default
            1337,   # Common backdoor
            31337,  # Elite backdoor
            8080,   # Common C2
            9001,   # Tor
            9030,   # Tor
        }
        if resp_p in suspicious_ports:
            severity += 25

        return min(severity, 100)

    # ============================================================
    # PRIVATE UTILITY METHODS
    # ============================================================

    def _parse_history_flags(
        self,
        history: str
    ) -> tuple:
        """
        Parse Zeek history string for TCP flags.

        Zeek history field encodes connection events
        as a string of characters:
            S = SYN sent by originator
            s = SYN sent by responder
            A = ACK
            F = FIN
            R = RST
            D = data packet
            C = bad checksum

        Uppercase = originator
        Lowercase = responder

        Returns:
            Tuple of (syn_flags, rst_flags, fin_flags)
        """
        if not history:
            return 0, 0, 0

        syn_flags = int("S" in history)
        rst_flags = int(
            "R" in history or "r" in history
        )
        fin_flags = int(
            "F" in history or "f" in history
        )

        return syn_flags, rst_flags, fin_flags

    def _safe_int(self, value) -> int:
        """Convert value to int safely"""
        try:
            if value == "-" or value == "":
                return 0
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    def _safe_float(self, value) -> float:
        """Convert value to float safely"""
        try:
            if value == "-" or value == "":
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0