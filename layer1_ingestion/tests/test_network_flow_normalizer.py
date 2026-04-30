"""
Layer 1 — Data Ingestion
Network Flow Normalizer Tests

Tests verify three things:

1. FIELD MAPPING
   Do Zeek fields correctly map to ECS fields?
   Is duration converted correctly?
   Are bytes and packets mapped to right directions?

2. DERIVED FEATURES
   Are flow rates calculated correctly?
   Are packet length means correct?
   Are TCP flags parsed from history string?

3. SECURITY ASSESSMENT
   Does severity correctly flag DoS patterns?
   Does scanning behavior get flagged?
   Do suspicious ports increase severity?

4. EDGE CASES
   Are missing Zeek fields handled gracefully?
   Are Zeek header lines correctly skipped?
   Are dash values for missing fields handled?
"""

import pytest
import numpy as np
from layer1_ingestion.normalizers.network_flow_normalizer import (
    NetworkFlowNormalizer,
    SUSPICIOUS_CONN_STATES
)


# ============================================================
# SAMPLE ZEEK FLOW DICTIONARIES
# Realistic Zeek conn.log entries for testing
# ============================================================

def make_normal_flow(**kwargs) -> dict:
    """Normal HTTPS browsing flow"""
    flow = {
        "ts": "1711678723.0",
        "uid": "CHhAvVtest123",
        "orig_h": "10.0.0.155",
        "orig_p": "54832",
        "resp_h": "142.250.80.46",
        "resp_p": "443",
        "proto": "tcp",
        "service": "ssl",
        "duration": "0.5",
        "orig_bytes": "1248",
        "resp_bytes": "15000",
        "conn_state": "SF",
        "local_orig": "T",
        "local_resp": "F",
        "missed_bytes": "0",
        "history": "ShADadFf",
        "orig_pkts": "10",
        "orig_ip_bytes": "1728",
        "resp_pkts": "15",
        "resp_ip_bytes": "15480"
    }
    flow.update(kwargs)
    return flow


def make_dos_flow(**kwargs) -> dict:
    """DoS flooding attack flow"""
    flow = {
        "ts": "1711678800.0",
        "uid": "CDos123attack",
        "orig_h": "192.168.1.100",
        "orig_p": "45000",
        "resp_h": "10.0.0.1",
        "resp_p": "80",
        "proto": "tcp",
        "service": "-",
        "duration": "0.001",
        "orig_bytes": "640000",
        "resp_bytes": "0",
        "conn_state": "S0",
        "local_orig": "T",
        "local_resp": "T",
        "missed_bytes": "0",
        "history": "S",
        "orig_pkts": "10000",
        "orig_ip_bytes": "660000",
        "resp_pkts": "0",
        "resp_ip_bytes": "0"
    }
    flow.update(kwargs)
    return flow


def make_portscan_flow(**kwargs) -> dict:
    """Port scanning flow"""
    flow = {
        "ts": "1711678900.0",
        "uid": "CScan456probe",
        "orig_h": "10.0.0.50",
        "orig_p": "45001",
        "resp_h": "192.168.1.1",
        "resp_p": "22",
        "proto": "tcp",
        "service": "-",
        "duration": "0.05",
        "orig_bytes": "40",
        "resp_bytes": "40",
        "conn_state": "REJ",
        "local_orig": "T",
        "local_resp": "T",
        "missed_bytes": "0",
        "history": "Sr",
        "orig_pkts": "1",
        "orig_ip_bytes": "60",
        "resp_pkts": "1",
        "resp_ip_bytes": "60"
    }
    flow.update(kwargs)
    return flow


def make_c2_beacon_flow(**kwargs) -> dict:
    """C2 beaconing flow to known malicious IP"""
    flow = {
        "ts": "1711678723.0",
        "uid": "CC2beacon789",
        "orig_h": "10.0.0.155",
        "orig_p": "54832",
        "resp_h": "185.220.101.45",
        "resp_p": "443",
        "proto": "tcp",
        "service": "ssl",
        "duration": "2819.4",
        "orig_bytes": "1248",
        "resp_bytes": "48291",
        "conn_state": "SF",
        "local_orig": "T",
        "local_resp": "F",
        "missed_bytes": "0",
        "history": "ShADadFf",
        "orig_pkts": "42",
        "orig_ip_bytes": "3612",
        "resp_pkts": "891",
        "resp_ip_bytes": "203848"
    }
    flow.update(kwargs)
    return flow


# Sample raw Zeek conn.log lines
SAMPLE_ZEEK_LINE = (
    "1711678723.0\tCHhAvVtest123\t10.0.0.155\t54832\t"
    "142.250.80.46\t443\ttcp\tssl\t0.5\t1248\t15000\t"
    "SF\tT\tF\t0\tShADadFf\t10\t1728\t15\t15480"
)

SAMPLE_ZEEK_HEADER = (
    "#separator \\x09\n"
    "#fields\tts\tuid\torig_h\torig_p\tresp_h"
)


# ============================================================
# TEST CLASS — FIELD MAPPING
# ============================================================

class TestFieldMapping:
    """Tests that Zeek fields map correctly to ECS"""

    def setup_method(self):
        self.normalizer = NetworkFlowNormalizer()

    def test_normalizes_successfully(self):
        """Normal flow produces ECSNormalized object"""
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None

    def test_source_ip_mapped_correctly(self):
        """Zeek orig_h maps to ECS source.ip"""
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        assert result.source.ip == "10.0.0.155"

    def test_destination_ip_mapped_correctly(self):
        """Zeek resp_h maps to ECS destination.ip"""
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        assert result.destination.ip == "142.250.80.46"

    def test_source_port_mapped_correctly(self):
        """Zeek orig_p maps to ECS source.port"""
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        assert result.source.port == 54832

    def test_destination_port_mapped_correctly(self):
        """Zeek resp_p maps to ECS destination.port"""
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        assert result.destination.port == 443

    def test_protocol_mapped_correctly(self):
        """Zeek proto maps to ECS network.transport"""
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        assert result.network.transport == "tcp"

    def test_fwd_bytes_from_orig_bytes(self):
        """
        Zeek orig_bytes maps to ECS network.fwd_bytes.
        orig = originator = forward direction.
        """
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        assert result.network.fwd_bytes == 1248

    def test_bwd_bytes_from_resp_bytes(self):
        """
        Zeek resp_bytes maps to ECS network.bwd_bytes.
        resp = responder = backward direction.
        """
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        assert result.network.bwd_bytes == 15000

    def test_fwd_packets_from_orig_pkts(self):
        """Zeek orig_pkts maps to network.fwd_packets"""
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        assert result.network.fwd_packets == 10

    def test_bwd_packets_from_resp_pkts(self):
        """Zeek resp_pkts maps to network.bwd_packets"""
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        assert result.network.bwd_packets == 15

    def test_duration_converted_to_milliseconds(self):
        """
        Zeek duration is in seconds.
        ECS duration_ms is in milliseconds.
        0.5 seconds = 500 milliseconds.
        """
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        assert result.network.duration_ms == 500.0

    def test_dataset_is_zeek(self):
        """Dataset correctly identifies Zeek as source"""
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        assert result.event.dataset == "zeek.conn"
        assert result.event.provider == "zeek"

    def test_data_source_label(self):
        """Data source label correctly set"""
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        assert result.data_source == "zeek_network_flow"

    def test_uid_used_as_event_id(self):
        """Zeek UID preserved as event ID"""
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        assert result.event.id == "CHhAvVtest123"

    def test_timestamp_converted_to_iso(self):
        """Unix timestamp converted to ISO 8601"""
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        assert "T" in result.timestamp
        assert "Z" in result.timestamp


# ============================================================
# TEST CLASS — DERIVED FEATURES
# ============================================================

class TestDerivedFeatures:
    """Tests for features calculated from raw Zeek fields"""

    def setup_method(self):
        self.normalizer = NetworkFlowNormalizer()

    def test_flow_bytes_per_sec_calculated(self):
        """
        flow_bytes_per_sec = total_bytes / duration
        (1248 + 15000) / 0.5 = 32496 bytes/sec
        """
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        expected = (1248 + 15000) / 0.5
        assert abs(
            result.network.flow_bytes_per_sec - expected
        ) < 1.0

    def test_flow_packets_per_sec_calculated(self):
        """
        flow_packets_per_sec = total_pkts / duration
        (10 + 15) / 0.5 = 50 packets/sec
        """
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        expected = (10 + 15) / 0.5
        assert abs(
            result.network.flow_packets_per_sec - expected
        ) < 1.0

    def test_fwd_packet_len_mean_calculated(self):
        """
        fwd_packet_len_mean = orig_bytes / orig_pkts
        1248 / 10 = 124.8
        """
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        expected = 1248 / 10
        assert abs(
            result.network.fwd_packet_len_mean - expected
        ) < 1.0

    def test_bwd_packet_len_mean_calculated(self):
        """
        bwd_packet_len_mean = resp_bytes / resp_pkts
        15000 / 15 = 1000.0
        """
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        expected = 15000 / 15
        assert abs(
            result.network.bwd_packet_len_mean - expected
        ) < 1.0

    def test_dos_flow_high_packets_per_sec(self):
        """
        DoS flow has very high packets per second.
        10000 packets in 0.001 seconds.
        """
        result = self.normalizer.normalize(
            make_dos_flow()
        )
        assert result is not None
        assert result.network.flow_packets_per_sec > 1000

    def test_zero_duration_handled_safely(self):
        """
        Zero duration does not cause division by zero.
        Epsilon prevents this edge case.
        """
        result = self.normalizer.normalize(
            make_normal_flow(duration="0")
        )
        assert result is not None
        assert not (
            result.network.flow_bytes_per_sec == float("inf")
        )


# ============================================================
# TEST CLASS — TCP FLAG PARSING
# ============================================================

class TestTCPFlagParsing:
    """Tests for Zeek history string flag parsing"""

    def setup_method(self):
        self.normalizer = NetworkFlowNormalizer()

    def test_syn_flag_detected_in_history(self):
        """
        S in history string indicates SYN flag.
        Normal connections always have SYN.
        """
        result = self.normalizer.normalize(
            make_normal_flow(history="ShADadFf")
        )
        assert result is not None
        assert result.network.syn_flags == 1

    def test_rst_flag_detected_in_history(self):
        """
        R in history string indicates RST flag.
        Port scans often produce RST responses.
        """
        result = self.normalizer.normalize(
            make_portscan_flow(history="Sr")
        )
        assert result is not None
        assert result.network.rst_flags == 1

    def test_no_flags_in_empty_history(self):
        """Empty history string produces zero flags"""
        result = self.normalizer.normalize(
            make_normal_flow(history="")
        )
        assert result is not None
        assert result.network.syn_flags == 0
        assert result.network.rst_flags == 0

    def test_syn_only_no_rst(self):
        """
        Normal SYN without RST.
        Clean connection establishment.
        """
        result = self.normalizer.normalize(
            make_normal_flow(history="ShADadFf")
        )
        assert result is not None
        assert result.network.syn_flags == 1
        assert result.network.rst_flags == 0


# ============================================================
# TEST CLASS — SECURITY SEVERITY ASSESSMENT
# ============================================================

class TestSecuritySeverity:
    """Tests for network flow security assessment"""

    def setup_method(self):
        self.normalizer = NetworkFlowNormalizer()

    def test_dos_flow_scores_high_severity(self):
        """
        DoS flooding scores high severity.
        High packet rate and S0 state are clear signals.
        """
        result = self.normalizer.normalize(
            make_dos_flow()
        )
        assert result is not None
        assert result.event.severity > 50

    def test_portscan_scores_elevated_severity(self):
        """
        Port scan with REJ state scores elevated severity.
        REJ is in SUSPICIOUS_CONN_STATES.
        """
        result = self.normalizer.normalize(
            make_portscan_flow()
        )
        assert result is not None
        assert result.event.severity > 25

    def test_normal_flow_scores_low_severity(self):
        """Normal HTTPS browsing scores low severity"""
        result = self.normalizer.normalize(
            make_normal_flow()
        )
        assert result is not None
        assert result.event.severity < 30

    def test_suspicious_port_increases_severity(self):
        """
        Known backdoor ports increase severity.
        Port 4444 is Metasploit default.
        """
        result = self.normalizer.normalize(
            make_normal_flow(resp_p="4444")
        )
        assert result is not None
        assert result.event.severity > 20

    def test_severity_capped_at_100(self):
        """Severity never exceeds 100"""
        result = self.normalizer.normalize(
            make_dos_flow()
        )
        assert result is not None
        assert result.event.severity <= 100


# ============================================================
# TEST CLASS — LINE AND FILE PARSING
# ============================================================

class TestLineParsing:
    """Tests for raw Zeek log line parsing"""

    def setup_method(self):
        self.normalizer = NetworkFlowNormalizer()

    def test_normalize_line_produces_ecs_event(self):
        """Raw Zeek conn.log line correctly parsed"""
        result = self.normalizer.normalize_line(
            SAMPLE_ZEEK_LINE
        )
        assert result is not None
        assert result.source.ip == "10.0.0.155"
        assert result.destination.ip == "142.250.80.46"

    def test_header_line_returns_none(self):
        """
        Zeek header lines starting with # are skipped.
        Headers are metadata not flow data.
        """
        result = self.normalizer.normalize_line(
            "#separator \\x09"
        )
        assert result is None

    def test_empty_line_returns_none(self):
        """Empty lines handled gracefully"""
        result = self.normalizer.normalize_line("")
        assert result is None

    def test_dash_values_treated_as_zero(self):
        """
        Zeek uses - for missing values.
        These convert to 0 safely.
        """
        result = self.normalizer.normalize(
            make_normal_flow(
                orig_bytes="-",
                resp_bytes="-"
            )
        )
        assert result is not None
        assert result.network.fwd_bytes == 0
        assert result.network.bwd_bytes == 0


# ============================================================
# TEST CLASS — ERROR HANDLING
# ============================================================

class TestErrorHandling:
    """Tests for graceful error handling"""

    def setup_method(self):
        self.normalizer = NetworkFlowNormalizer()

    def test_none_input_returns_none(self):
        """None input handled gracefully"""
        result = self.normalizer.normalize(None)
        assert result is None

    def test_empty_dict_returns_none(self):
        """Empty dict handled gracefully"""
        result = self.normalizer.normalize({})
        assert result is None

    def test_missing_fields_use_defaults(self):
        """
        Missing Zeek fields default to zero.
        Partial flows still produce valid ECS events.
        """
        minimal_flow = {
            "ts": "1711678723.0",
            "uid": "Cminimal123",
            "orig_h": "10.0.0.1",
            "resp_h": "8.8.8.8",
            "orig_p": "12345",
            "resp_p": "53",
            "proto": "udp"
        }
        result = self.normalizer.normalize(minimal_flow)
        assert result is not None
        assert result.network.fwd_bytes == 0
        assert result.network.bwd_bytes == 0

    def test_statistics_tracked(self):
        """
        Normalizer tracks success statistics.
        Used by MLOps monitoring in production.
        """
        self.normalizer.normalize(make_normal_flow())
        self.normalizer.normalize(make_dos_flow())
        self.normalizer.normalize(None)

        stats = self.normalizer.get_statistics()

        assert stats["events_processed"] == 3
        assert stats["events_succeeded"] == 2
        assert stats["events_failed"] == 1
    