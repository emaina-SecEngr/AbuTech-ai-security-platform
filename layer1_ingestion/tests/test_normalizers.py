"""
Layer 1 — Data Ingestion
Normalizer Tests

Tests verify three things:

1. CODE CORRECTNESS
   Does the normalizer correctly transform
   each field from raw CrowdStrike format
   to ECS format?

2. SECURITY LOGIC
   Does the severity assessment correctly
   identify suspicious behaviors?
   Encoded PowerShell should score high.
   Normal notepad.exe should score low.

3. EDGE CASES
   Does the normalizer handle missing fields,
   malformed data, and unknown event types
   gracefully without crashing?

Run with: pytest layer1_ingestion/tests/ -v
"""

import pytest
from layer1_ingestion.normalizers.crowdstrike_normalizer import (
    CrowdStrikeNormalizer
)
from layer1_ingestion.normalizers.base_normalizer import (
    BaseNormalizer,
    NormalizationError
)


# ============================================================
# SAMPLE RAW EVENTS
# Realistic CrowdStrike event samples used across tests
# ============================================================

SAMPLE_PROCESS_EVENT = {
    "metadata": {
        "eventType": "ProcessRollup2",
        "eventCreationTime": 1711678663000,
        "customerIDString": "test_customer_123"
    },
    "event": {
        "ProcessId": 4892,
        "ParentProcessId": 3241,
        "ComputerName": "WKSTN-JSMITH-01",
        "UserName": "CORP\\jsmith",
        "ImageFileName": (
            "\\Device\\HarddiskVolume3\\"
            "Windows\\System32\\WindowsPowerShell\\"
            "v1.0\\powershell.exe"
        ),
        "CommandLine": "powershell.exe -enc JABjAGwAaQBlAG4AdA==",
        "MD5HashData": "eb84f6e4376d1b9a50f7d3a4a48d9f2c",
        "SHA256HashData": "a3f8d2c1e5b7a9f0d4c2e6b8",
        "ParentImageFileName": (
            "\\Device\\HarddiskVolume3\\"
            "Windows\\Microsoft.NET\\"
            "Framework64\\v4.0.30319\\MSBuild.exe"
        ),
        "ParentCommandLine": "MSBuild.exe /nologo suspicious.proj",
        "IntegrityLevel": "Medium"
    }
}

SAMPLE_NETWORK_EVENT = {
    "metadata": {
        "eventType": "NetworkConnectIP4",
        "eventCreationTime": 1711678723000,
        "customerIDString": "test_customer_123"
    },
    "event": {
        "ProcessId": 4892,
        "ComputerName": "WKSTN-JSMITH-01",
        "UserName": "CORP\\jsmith",
        "ImageFileName": (
            "\\Device\\HarddiskVolume3\\"
            "Windows\\System32\\svchost.exe"
        ),
        "LocalAddress": "10.0.0.155",
        "LocalPort": 54832,
        "RemoteAddress": "185.220.101.45",
        "RemotePort": 443,
        "Protocol": "TCP"
    }
}

SAMPLE_DNS_EVENT = {
    "metadata": {
        "eventType": "DnsRequest",
        "eventCreationTime": 1711678743000,
        "customerIDString": "test_customer_123"
    },
    "event": {
        "ProcessId": 4892,
        "ComputerName": "WKSTN-JSMITH-01",
        "UserName": "CORP\\jsmith",
        "ImageFileName": (
            "\\Device\\HarddiskVolume3\\"
            "Windows\\System32\\svchost.exe"
        ),
        "DomainName": "malicious-c2-domain.xyz"
    }
}

SAMPLE_BENIGN_PROCESS_EVENT = {
    "metadata": {
        "eventType": "ProcessRollup2",
        "eventCreationTime": 1711678663000,
        "customerIDString": "test_customer_123"
    },
    "event": {
        "ProcessId": 1234,
        "ParentProcessId": 5678,
        "ComputerName": "WKSTN-JSMITH-01",
        "UserName": "CORP\\jsmith",
        "ImageFileName": (
            "\\Device\\HarddiskVolume3\\"
            "Windows\\System32\\notepad.exe"
        ),
        "CommandLine": "notepad.exe document.txt",
        "ParentImageFileName": (
            "\\Device\\HarddiskVolume3\\"
            "Windows\\explorer.exe"
        ),
        "ParentCommandLine": "explorer.exe",
        "IntegrityLevel": "Medium"
    }
}


# ============================================================
# TEST CLASS — BASE NORMALIZER UTILITIES
# Tests shared utility methods inherited by all normalizers
# ============================================================

class TestBaseNormalizerUtilities:
    """
    Tests for shared utility methods in BaseNormalizer.

    These utilities are used by every normalizer
    so correctness here affects the entire platform.
    """

    def setup_method(self):
        """Create normalizer instance before each test"""
        self.normalizer = CrowdStrikeNormalizer()

    def test_convert_unix_ms_to_iso(self):
        """Timestamp conversion produces valid ISO format"""
        result = self.normalizer.convert_unix_ms_to_iso(
            1711678663000
        )
        assert "2024" in result or "2025" in result
        assert "T" in result
        assert "Z" in result

    def test_split_domain_username_backslash(self):
        """CORP\\jsmith splits correctly"""
        domain, username = self.normalizer.split_domain_username(
            "CORP\\jsmith"
        )
        assert domain == "CORP"
        assert username == "jsmith"

    def test_split_domain_username_no_domain(self):
        """Username without domain returns None domain"""
        domain, username = self.normalizer.split_domain_username(
            "jsmith"
        )
        assert domain is None
        assert username == "jsmith"

    def test_split_domain_username_email_format(self):
        """jsmith@corp.com splits correctly"""
        domain, username = self.normalizer.split_domain_username(
            "jsmith@corp.com"
        )
        assert domain == "corp.com"
        assert username == "jsmith"

    def test_split_domain_username_empty(self):
        """Empty username returns None for both"""
        domain, username = self.normalizer.split_domain_username(
            ""
        )
        assert domain is None
        assert username is None

    def test_extract_filename_from_windows_path(self):
        """Extracts filename from Windows device path"""
        result = self.normalizer.extract_filename_from_path(
            "\\Device\\HarddiskVolume3\\Windows\\"
            "System32\\powershell.exe"
        )
        assert result == "powershell.exe"

    def test_extract_filename_from_linux_path(self):
        """Extracts filename from Linux path"""
        result = self.normalizer.extract_filename_from_path(
            "/usr/bin/python3"
        )
        assert result == "python3"

    def test_clean_windows_device_path(self):
        """Converts device path to readable format"""
        result = self.normalizer.clean_windows_device_path(
            "\\Device\\HarddiskVolume3\\Windows\\"
            "System32\\powershell.exe"
        )
        assert "C:" in result
        assert "powershell.exe" in result
        assert "HarddiskVolume" not in result


# ============================================================
# TEST CLASS — PROCESS EVENT NORMALIZATION
# ============================================================

class TestProcessEventNormalization:
    """
    Tests for CrowdStrike ProcessRollup2 event normalization.

    These are the most important tests because process events
    are the primary data source for malware detection.
    """

    def setup_method(self):
        self.normalizer = CrowdStrikeNormalizer()

    def test_process_event_normalizes_successfully(self):
        """Process event returns ECSNormalized object"""
        result = self.normalizer.normalize(SAMPLE_PROCESS_EVENT)
        assert result is not None

    def test_process_event_timestamp_converted(self):
        """Unix millisecond timestamp converted to ISO"""
        result = self.normalizer.normalize(SAMPLE_PROCESS_EVENT)
        assert result is not None
        assert "T" in result.timestamp
        assert "Z" in result.timestamp

    def test_process_event_username_split(self):
        """CORP\\jsmith correctly split into domain and name"""
        result = self.normalizer.normalize(SAMPLE_PROCESS_EVENT)
        assert result is not None
        assert result.user.name == "jsmith"
        assert result.user.domain == "CORP"

    def test_process_event_hostname_extracted(self):
        """Hostname correctly extracted"""
        result = self.normalizer.normalize(SAMPLE_PROCESS_EVENT)
        assert result is not None
        assert result.host.hostname == "WKSTN-JSMITH-01"

    def test_process_event_process_name_extracted(self):
        """Process name extracted from device path"""
        result = self.normalizer.normalize(SAMPLE_PROCESS_EVENT)
        assert result is not None
        assert result.process.name == "powershell.exe"

    def test_process_event_parent_name_extracted(self):
        """Parent process name extracted from device path"""
        result = self.normalizer.normalize(SAMPLE_PROCESS_EVENT)
        assert result is not None
        assert result.process.parent is not None
        assert result.process.parent.name == "MSBuild.exe"

    def test_process_event_command_line_preserved(self):
        """Full command line preserved including encoded command"""
        result = self.normalizer.normalize(SAMPLE_PROCESS_EVENT)
        assert result is not None
        assert "-enc" in result.process.command_line
        assert "JABjAGwAaQBlAG4AdA==" in result.process.command_line

    def test_process_event_category_is_process(self):
        """ECS event category is process"""
        result = self.normalizer.normalize(SAMPLE_PROCESS_EVENT)
        assert result is not None
        assert result.event.category == "process"

    def test_process_event_dataset_is_crowdstrike(self):
        """Dataset correctly identifies CrowdStrike as source"""
        result = self.normalizer.normalize(SAMPLE_PROCESS_EVENT)
        assert result is not None
        assert "crowdstrike" in result.event.dataset

    def test_process_event_data_source_label(self):
        """Data source label correctly set"""
        result = self.normalizer.normalize(SAMPLE_PROCESS_EVENT)
        assert result is not None
        assert result.data_source == "crowdstrike_edr"

    def test_process_event_normalized_flag(self):
        """Normalized flag set to True"""
        result = self.normalizer.normalize(SAMPLE_PROCESS_EVENT)
        assert result is not None
        assert result.normalized is True


# ============================================================
# TEST CLASS — SECURITY SEVERITY ASSESSMENT
# These tests verify your security domain knowledge
# is correctly encoded in the normalizer
# ============================================================

class TestSecuritySeverityAssessment:
    """
    Tests for the _assess_process_severity method.

    This is where security domain knowledge becomes
    verifiable code. These tests ensure that:
    - Malicious patterns score HIGH
    - Benign patterns score LOW
    - The severity logic cannot be accidentally broken
    """

    def setup_method(self):
        self.normalizer = CrowdStrikeNormalizer()

    def test_encoded_powershell_scores_high(self):
        """
        PowerShell with -enc flag must score high.
        This is a clear malicious signal.
        Reference: MITRE ATT&CK T1059.001
        """
        result = self.normalizer.normalize(SAMPLE_PROCESS_EVENT)
        assert result is not None
        assert result.event.severity > 50

    def test_benign_process_scores_low(self):
        """
        notepad.exe from explorer.exe scores low.
        This is a normal user action.
        """
        result = self.normalizer.normalize(
            SAMPLE_BENIGN_PROCESS_EVENT
        )
        assert result is not None
        assert result.event.severity < 30

    def test_msbuild_parent_increases_severity(self):
        """
        MSBuild.exe as parent process increases severity.
        Living off the land technique.
        Reference: MITRE ATT&CK T1127
        """
        severity = self.normalizer._assess_process_severity(
            process_name="powershell.exe",
            parent_name="msbuild.exe",
            command_line="powershell.exe"
        )
        assert severity > 30

    def test_winword_parent_increases_severity(self):
        """
        Word spawning PowerShell is a phishing indicator.
        Reference: MITRE ATT&CK T1566.001
        """
        severity = self.normalizer._assess_process_severity(
            process_name="powershell.exe",
            parent_name="winword.exe",
            command_line="powershell.exe"
        )
        assert severity > 30

    def test_bypass_execution_policy_scores_high(self):
        """
        PowerShell -Bypass flag scores high.
        Attackers bypass execution policy
        to run malicious scripts.
        """
        severity = self.normalizer._assess_process_severity(
            process_name="powershell.exe",
            parent_name="explorer.exe",
            command_line="powershell -ExecutionPolicy Bypass -File evil.ps1"
        )
        assert severity > 25

    def test_severity_capped_at_100(self):
        """
        Severity never exceeds 100 regardless of
        how many suspicious signals are present.
        """
        severity = self.normalizer._assess_process_severity(
            process_name="powershell.exe",
            parent_name="winword.exe",
            command_line="powershell -enc JABj -ExecutionPolicy Bypass"
        )
        assert severity <= 100


# ============================================================
# TEST CLASS — NETWORK EVENT NORMALIZATION
# ============================================================

class TestNetworkEventNormalization:
    """Tests for CrowdStrike NetworkConnectIP4 normalization"""

    def setup_method(self):
        self.normalizer = CrowdStrikeNormalizer()

    def test_network_event_normalizes_successfully(self):
        """Network event returns ECSNormalized object"""
        result = self.normalizer.normalize(SAMPLE_NETWORK_EVENT)
        assert result is not None

    def test_network_event_source_ip_extracted(self):
        """Source IP correctly extracted"""
        result = self.normalizer.normalize(SAMPLE_NETWORK_EVENT)
        assert result is not None
        assert result.source.ip == "10.0.0.155"

    def test_network_event_destination_ip_extracted(self):
        """Destination IP correctly extracted"""
        result = self.normalizer.normalize(SAMPLE_NETWORK_EVENT)
        assert result is not None
        assert result.destination.ip == "185.220.101.45"

    def test_network_event_destination_port_extracted(self):
        """Destination port correctly extracted"""
        result = self.normalizer.normalize(SAMPLE_NETWORK_EVENT)
        assert result is not None
        assert result.destination.port == 443

    def test_network_event_category_is_network(self):
        """ECS category is network"""
        result = self.normalizer.normalize(SAMPLE_NETWORK_EVENT)
        assert result is not None
        assert result.event.category == "network"


# ============================================================
# TEST CLASS — DNS EVENT NORMALIZATION
# ============================================================

class TestDNSEventNormalization:
    """Tests for CrowdStrike DnsRequest normalization"""

    def setup_method(self):
        self.normalizer = CrowdStrikeNormalizer()

    def test_dns_event_normalizes_successfully(self):
        """DNS event returns ECSNormalized object"""
        result = self.normalizer.normalize(SAMPLE_DNS_EVENT)
        assert result is not None

    def test_dns_event_domain_extracted(self):
        """Queried domain correctly extracted"""
        result = self.normalizer.normalize(SAMPLE_DNS_EVENT)
        assert result is not None
        assert result.destination.domain == "malicious-c2-domain.xyz"

    def test_dns_event_category_is_dns(self):
        """ECS category is dns"""
        result = self.normalizer.normalize(SAMPLE_DNS_EVENT)
        assert result is not None
        assert result.event.category == "dns"


# ============================================================
# TEST CLASS — ERROR HANDLING AND EDGE CASES
# ============================================================

class TestErrorHandling:
    """
    Tests for edge cases and error handling.

    A production normalizer must never crash
    regardless of what input it receives.
    Bad data should be rejected gracefully
    not crash the pipeline.
    """

    def setup_method(self):
        self.normalizer = CrowdStrikeNormalizer()

    def test_none_input_returns_none(self):
        """None input handled gracefully"""
        result = self.normalizer.normalize(None)
        assert result is None

    def test_empty_dict_returns_none(self):
        """Empty dict handled gracefully"""
        result = self.normalizer.normalize({})
        assert result is None

    def test_unknown_event_type_returns_none(self):
        """Unknown event type handled gracefully"""
        unknown_event = {
            "metadata": {
                "eventType": "UnknownEventType123",
                "eventCreationTime": 1711678663000
            },
            "event": {
                "ComputerName": "WKSTN-001"
            }
        }
        result = self.normalizer.normalize(unknown_event)
        assert result is None

    def test_missing_metadata_returns_none(self):
        """Event missing metadata section returns None"""
        incomplete_event = {
            "event": {
                "ComputerName": "WKSTN-001"
            }
        }
        result = self.normalizer.normalize(incomplete_event)
        assert result is None

    def test_statistics_tracked_correctly(self):
        """
        Normalizer tracks success and failure counts.
        Used by MLOps monitoring in production.
        """
        # Process two valid events
        self.normalizer.normalize(SAMPLE_PROCESS_EVENT)
        self.normalizer.normalize(SAMPLE_NETWORK_EVENT)

        # Process one invalid event
        self.normalizer.normalize(None)

        stats = self.normalizer.get_statistics()

        assert stats["events_processed"] == 3
        assert stats["events_succeeded"] == 2
        assert stats["events_failed"] == 1
        assert stats["success_rate_pct"] > 60