"""
Layer 1 — Data Ingestion
CrowdStrike Falcon EDR Normalizer

Transforms raw CrowdStrike Falcon EDR events into
Elastic Common Schema (ECS) format.

CrowdStrike generates several event types. This normalizer
handles the most security-critical ones:

    ProcessRollup2      - Process creation events
                          Most important for malware detection
                          Contains parent-child relationships

    NetworkConnectIP4   - Network connection events
                          Maps process to network activity
                          Critical for C2 detection

    DnsRequest          - DNS lookup events
                          Early C2 detection signal
                          Malware beaconing patterns

Raw CrowdStrike Event Structure:
    {
        "metadata": {
            "eventType": "ProcessRollup2",
            "eventCreationTime": 1711678663000,
            "customerIDString": "abc123"
        },
        "event": {
            "ComputerName": "WKSTN-JSMITH-01",
            "UserName": "CORP\\\\jsmith",
            "ImageFileName": "\\\\Device\\\\HarddiskVolume3\\\\...",
            "CommandLine": "powershell.exe -enc JABj...",
            "ParentImageFileName": "\\\\Device\\\\...\\\\MSBuild.exe",
            "MD5HashData": "eb84f6e4...",
            "SHA256HashData": "a3f8d2c1..."
        }
    }

Security Note:
    This normalizer preserves the full command line including
    encoded commands. Never truncate command lines in security
    data - the full command is evidence and may be needed for
    forensic investigation.
"""

from layer1_ingestion.schema.ecs_schema import ECSEvent
from layer1_ingestion.schema.ecs_schema import ECSHost
from layer1_ingestion.schema.ecs_schema import ECSNormalized
from layer1_ingestion.schema.ecs_schema import ECSParentProcess
from layer1_ingestion.schema.ecs_schema import ECSProcess
from layer1_ingestion.schema.ecs_schema import ECSSource
from layer1_ingestion.schema.ecs_schema import ECSDestination
from layer1_ingestion.schema.ecs_schema import ECSNetwork
from layer1_ingestion.schema.ecs_schema import ECSUser
from layer1_ingestion.normalizers.base_normalizer import BaseNormalizer
from layer1_ingestion.normalizers.base_normalizer import NormalizationError


# ============================================================
# SUSPICIOUS PARENT PROCESS INDICATORS
#
# These parent processes spawning PowerShell, cmd, or
# scripting engines are high-confidence malicious signals.
#
# This list is used to enrich the normalized event with
# an initial suspicion flag before Layer 2 ML scoring.
#
# Reference: MITRE ATT&CK T1059 - Command and Scripting
# Reference: MITRE ATT&CK T1566 - Phishing (document macros)
# ============================================================

SUSPICIOUS_PARENT_PROCESSES = {
    "winword.exe",       # Word document macros - T1566.001
    "excel.exe",         # Excel macros - T1566.001
    "powerpnt.exe",      # PowerPoint macros - T1566.001
    "outlook.exe",       # Email attachments - T1566.001
    "msbuild.exe",       # Living off the land - T1127
    "regsvr32.exe",      # Script execution - T1218.010
    "rundll32.exe",      # DLL execution - T1218.011
    "wscript.exe",       # Script host - T1059.005
    "cscript.exe",       # Script host - T1059.005
    "mshta.exe",         # HTML application - T1218.005
}

# Processes that commonly spawn PowerShell legitimately
# Used to reduce false positives in suspicious parent detection
LEGITIMATE_POWERSHELL_PARENTS = {
    "services.exe",
    "svchost.exe",
    "taskhost.exe",
    "taskhostw.exe",
    "explorer.exe",
}


class CrowdStrikeNormalizer(BaseNormalizer):
    """
    Normalizes CrowdStrike Falcon EDR events to ECS format.

    Inherits from BaseNormalizer which provides:
        - Input validation
        - Error handling
        - Statistics tracking
        - Utility methods for timestamp conversion,
          username splitting, path cleaning

    This class only implements the CrowdStrike-specific
    transformation logic in normalize_event().
    """

    def __init__(self):
        super().__init__(source_name="crowdstrike_edr")

    def normalize_event(
        self,
        raw_event: dict
    ) -> ECSNormalized:
        """
        Transform raw CrowdStrike event to ECS format.

        Routes to the appropriate handler based on
        the eventType field in the metadata.

        Args:
            raw_event: Raw CrowdStrike Falcon event dict

        Returns:
            ECSNormalized: Standardized ECS event

        Raises:
            NormalizationError: If required fields missing
                               or event type unknown
        """

        # Extract top level sections
        metadata = raw_event.get("metadata", {})
        event_data = raw_event.get("event", {})

        if not metadata or not event_data:
            raise NormalizationError(
                "CrowdStrike event missing metadata "
                "or event sections"
            )

        # Get event type to route to correct handler
        event_type = metadata.get("eventType", "")

        if event_type == "ProcessRollup2":
            return self._normalize_process_event(
                metadata,
                event_data
            )

        elif event_type == "NetworkConnectIP4":
            return self._normalize_network_event(
                metadata,
                event_data
            )

        elif event_type == "DnsRequest":
            return self._normalize_dns_event(
                metadata,
                event_data
            )

        else:
            # Unknown event type
            # Log and raise so pipeline can track
            # which CrowdStrike event types we see
            # but have not yet implemented handlers for
            raise NormalizationError(
                f"Unknown CrowdStrike event type: {event_type}"
            )

    # ============================================================
    # EVENT TYPE HANDLERS
    # Each method handles one CrowdStrike event type
    # ============================================================

    def _normalize_process_event(
        self,
        metadata: dict,
        event_data: dict
    ) -> ECSNormalized:
        """
        Normalize a ProcessRollup2 event.

        ProcessRollup2 is the most valuable CrowdStrike
        event type for security detection. It captures
        every process creation with full context:
        - What ran (ImageFileName)
        - Who ran it (UserName)
        - What ran it (ParentImageFileName)
        - How it was invoked (CommandLine)
        - Cryptographic identity (MD5, SHA256)

        This is the primary data source for your
        Layer 2 malware classifier.
        """

        # ---- TIMESTAMP ----
        # CrowdStrike uses Unix milliseconds
        # BaseNormalizer.convert_unix_ms_to_iso() handles this
        raw_timestamp = metadata.get("eventCreationTime", 0)
        timestamp = self.convert_unix_ms_to_iso(raw_timestamp)

        # ---- USER ----
        # CrowdStrike format: "CORP\\jsmith"
        # BaseNormalizer.split_domain_username() handles this
        raw_username = event_data.get("UserName", "")
        domain, username = self.split_domain_username(
            raw_username
        )

        user = ECSUser(
            name=username,
            domain=domain
        )

        # ---- HOST ----
        host = ECSHost(
            hostname=event_data.get("ComputerName", ""),
            os_platform="windows"
        )

        # ---- PROCESS ----
        # Extract and clean the process file path
        raw_image_path = event_data.get("ImageFileName", "")
        clean_path = self.clean_windows_device_path(
            raw_image_path
        )
        process_name = self.extract_filename_from_path(
            clean_path
        )

        # Extract and clean parent process path
        raw_parent_path = event_data.get(
            "ParentImageFileName", ""
        )
        clean_parent_path = self.clean_windows_device_path(
            raw_parent_path
        )
        parent_name = self.extract_filename_from_path(
            clean_parent_path
        )

        # Build parent process object
        parent_process = ECSParentProcess(
            name=parent_name,
            pid=event_data.get("ParentProcessId"),
            executable=clean_parent_path,
            command_line=event_data.get("ParentCommandLine", "")
        )

        # Count command line arguments
        command_line = event_data.get("CommandLine", "")
        args_count = len(command_line.split()) if command_line else 0

        # Build process object
        process = ECSProcess(
            name=process_name,
            pid=event_data.get("ProcessId"),
            executable=clean_path,
            command_line=command_line,
            args_count=args_count,
            integrity_level=event_data.get("IntegrityLevel", ""),
            parent=parent_process
        )

        # ---- SECURITY ENRICHMENT ----
        # Check for suspicious parent-child relationships
        # This is an initial signal before Layer 2 ML scoring
        # Sets severity based on domain knowledge
        severity = self._assess_process_severity(
            process_name=process_name,
            parent_name=parent_name,
            command_line=command_line
        )

        # ---- ECS EVENT ----
        ecs_event = ECSEvent(
            id=self.generate_event_id(),
            category="process",
            type="start",
            dataset="crowdstrike.process",
            provider="crowdstrike",
            created=timestamp,
            severity=severity
        )

        # ---- ASSEMBLE NORMALIZED EVENT ----
        return ECSNormalized(
            timestamp=timestamp,
            event=ecs_event,
            user=user,
            host=host,
            process=process,
            data_source="crowdstrike_edr"
        )

    def _normalize_network_event(
        self,
        metadata: dict,
        event_data: dict
    ) -> ECSNormalized:
        """
        Normalize a NetworkConnectIP4 event.

        CrowdStrike network events are unique because
        they attribute network connections to specific
        processes. A network flow from a SIEM only tells
        you which machine made a connection. A CrowdStrike
        network event tells you WHICH PROCESS on that
        machine made the connection.

        This attribution is critical for C2 detection.
        svchost.exe connecting to a known malicious IP
        is high confidence malicious.
        """

        raw_timestamp = metadata.get("eventCreationTime", 0)
        timestamp = self.convert_unix_ms_to_iso(raw_timestamp)

        raw_username = event_data.get("UserName", "")
        domain, username = self.split_domain_username(
            raw_username
        )

        user = ECSUser(name=username, domain=domain)

        host = ECSHost(
            hostname=event_data.get("ComputerName", ""),
            os_platform="windows"
        )

        # Network source and destination
        source = ECSSource(
            ip=event_data.get("LocalAddress", ""),
            port=event_data.get("LocalPort")
        )

        destination = ECSDestination(
            ip=event_data.get("RemoteAddress", ""),
            port=event_data.get("RemotePort")
        )

        network = ECSNetwork(
            protocol="ipv4",
            transport=event_data.get("Protocol", "tcp"),
            direction="outbound"
        )

        # Process that made the connection
        raw_image_path = event_data.get("ImageFileName", "")
        clean_path = self.clean_windows_device_path(
            raw_image_path
        )
        process_name = self.extract_filename_from_path(
            clean_path
        )

        process = ECSProcess(
            name=process_name,
            pid=event_data.get("ProcessId"),
            executable=clean_path
        )

        ecs_event = ECSEvent(
            id=self.generate_event_id(),
            category="network",
            type="connection",
            dataset="crowdstrike.network",
            provider="crowdstrike",
            created=timestamp
        )

        return ECSNormalized(
            timestamp=timestamp,
            event=ecs_event,
            user=user,
            host=host,
            source=source,
            destination=destination,
            network=network,
            process=process,
            data_source="crowdstrike_edr"
        )

    def _normalize_dns_event(
        self,
        metadata: dict,
        event_data: dict
    ) -> ECSNormalized:
        """
        Normalize a DnsRequest event.

        DNS requests are one of the earliest signals
        of malicious activity. Malware must resolve
        its C2 domain before it can communicate.

        Patterns that indicate malicious DNS:
        - High entropy domain names (DGA domains)
        - Newly registered domains
        - Domains matching known C2 patterns
        - Unusually high DNS request frequency

        Your Layer 2 NLP and GNN models use
        DNS events to build domain relationship
        graphs and detect DGA patterns.
        """

        raw_timestamp = metadata.get("eventCreationTime", 0)
        timestamp = self.convert_unix_ms_to_iso(raw_timestamp)

        raw_username = event_data.get("UserName", "")
        domain, username = self.split_domain_username(
            raw_username
        )

        user = ECSUser(name=username, domain=domain)

        host = ECSHost(
            hostname=event_data.get("ComputerName", ""),
            os_platform="windows"
        )

        # DNS specific - destination is the queried domain
        destination = ECSDestination(
            domain=event_data.get("DomainName", "")
        )

        # Process that made the DNS request
        raw_image_path = event_data.get("ImageFileName", "")
        clean_path = self.clean_windows_device_path(
            raw_image_path
        )
        process_name = self.extract_filename_from_path(
            clean_path
        )

        process = ECSProcess(
            name=process_name,
            pid=event_data.get("ProcessId"),
            executable=clean_path
        )

        ecs_event = ECSEvent(
            id=self.generate_event_id(),
            category="dns",
            type="info",
            dataset="crowdstrike.dns",
            provider="crowdstrike",
            created=timestamp
        )

        return ECSNormalized(
            timestamp=timestamp,
            event=ecs_event,
            user=user,
            host=host,
            destination=destination,
            process=process,
            data_source="crowdstrike_edr"
        )

    # ============================================================
    # SECURITY ASSESSMENT HELPERS
    # Domain knowledge encoded as functions
    # These produce initial signals before Layer 2 ML scoring
    # ============================================================

    def _assess_process_severity(
        self,
        process_name: str,
        parent_name: str,
        command_line: str
    ) -> int:
        """
        Assess initial severity of a process creation event.

        This encodes security domain knowledge as rules
        that produce an initial severity score BEFORE
        Layer 2 ML models run.

        Think of it as a fast pre-filter:
        - Obviously suspicious events get elevated severity
        - Layer 2 ML models then refine this score

        Severity scale:
            0-25:   Low     - likely benign
            26-50:  Medium  - worth monitoring
            51-75:  High    - investigate
            76-100: Critical - immediate action

        Args:
            process_name: Name of the spawned process
            parent_name: Name of the parent process
            command_line: Full command line string

        Returns:
            Integer severity score 0-100
        """
        severity = 0
        process_lower = process_name.lower()
        parent_lower = parent_name.lower()
        cmd_lower = command_line.lower()

        # ---- CHECK 1: Suspicious parent process ----
        # Office apps, script hosts spawning shells
        # is a classic malware pattern
        if parent_lower in SUSPICIOUS_PARENT_PROCESSES:
            severity += 40

        # ---- CHECK 2: Encoded PowerShell ----
        # -enc or -encodedcommand flag indicates
        # obfuscated PowerShell - strong malicious signal
        # Reference: MITRE ATT&CK T1059.001
        if (
            "powershell" in process_lower and
            ("-enc" in cmd_lower or
             "-encodedcommand" in cmd_lower)
        ):
            severity += 35

        # ---- CHECK 3: PowerShell execution policy bypass ----
        # Attackers bypass execution policy to run scripts
        # Reference: MITRE ATT&CK T1059.001
        if (
            "powershell" in process_lower and
            "bypass" in cmd_lower
        ):
            severity += 30

        # ---- CHECK 4: Living off the land binaries ----
        # Legitimate Windows tools used maliciously
        # Reference: MITRE ATT&CK T1218
        lolbins = {
            "certutil.exe",    # Download tool misuse
            "bitsadmin.exe",   # Download tool misuse
            "regsvr32.exe",    # Script execution
            "rundll32.exe",    # DLL execution
            "mshta.exe",       # HTML application
            "wmic.exe",        # WMI execution
        }
        if process_lower in lolbins:
            severity += 25

        # ---- CHECK 5: Temp directory execution ----
        # Malware commonly runs from temp directories
        if any(
            temp_dir in cmd_lower
            for temp_dir in [
                "\\temp\\",
                "\\tmp\\",
                "\\appdata\\local\\temp\\"
            ]
        ):
            severity += 20

        # Cap at 100
        return min(severity, 100)