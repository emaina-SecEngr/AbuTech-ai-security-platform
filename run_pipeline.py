"""
AbuTech AI Security Platform
End-to-End Pipeline Demonstration

This script demonstrates the complete flow:
    Raw CrowdStrike EDR event
        ↓
    Layer 1: Normalization
        ↓
    Layer 2: ML Scoring
        ↓
    Detection Result

This is the moment everything connects.
Layer 1 and Layer 2 working together
as a unified security intelligence system.

Run with:
    python run_pipeline.py
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from layer1_ingestion.normalizers.crowdstrike_normalizer import (
    CrowdStrikeNormalizer
)
from layer2_ml.intrusion_detection.detector import (
    NetworkIntrusionDetector
)


# ============================================================
# SAMPLE RAW CROWDSTRIKE EVENTS
# These simulate real events your platform would receive
# ============================================================

# Event 1: Suspicious PowerShell from MSBuild
# High severity process event
SUSPICIOUS_PROCESS_EVENT = {
    "metadata": {
        "eventType": "ProcessRollup2",
        "eventCreationTime": 1711678663000,
        "customerIDString": "abutech_demo"
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
        "CommandLine": (
            "powershell.exe -enc JABjAGwAaQBlAG4AdA=="
        ),
        "ParentImageFileName": (
            "\\Device\\HarddiskVolume3\\"
            "Windows\\Microsoft.NET\\"
            "Framework64\\v4.0.30319\\MSBuild.exe"
        ),
        "ParentCommandLine": (
            "MSBuild.exe /nologo suspicious.proj"
        ),
        "IntegrityLevel": "Medium"
    }
}

# Event 2: Suspicious outbound network connection
# C2 communication attempt
SUSPICIOUS_NETWORK_EVENT = {
    "metadata": {
        "eventType": "NetworkConnectIP4",
        "eventCreationTime": 1711678723000,
        "customerIDString": "abutech_demo"
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

# Event 3: DNS request to suspicious domain
DNS_EVENT = {
    "metadata": {
        "eventType": "DnsRequest",
        "eventCreationTime": 1711678743000,
        "customerIDString": "abutech_demo"
    },
    "event": {
        "ProcessId": 4892,
        "ComputerName": "WKSTN-JSMITH-01",
        "UserName": "CORP\\jsmith",
        "ImageFileName": (
            "\\Device\\HarddiskVolume3\\"
            "Windows\\System32\\svchost.exe"
        ),
        "DomainName": "xjf8k2mp.duckdns.org"
    }
}

# Event 4: Normal benign process
BENIGN_PROCESS_EVENT = {
    "metadata": {
        "eventType": "ProcessRollup2",
        "eventCreationTime": 1711678800000,
        "customerIDString": "abutech_demo"
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


class AbuTechPipeline:
    """
    End-to-end security intelligence pipeline.

    Connects Layer 1 normalization with
    Layer 2 ML scoring into a unified system.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize pipeline components.

        Args:
            model_path: Path to trained model
                       If None runs without ML scoring
        """
        print("Initializing AbuTech Pipeline...")

        # Layer 1 — Normalizer
        self.normalizer = CrowdStrikeNormalizer()
        print("  ✅ Layer 1: CrowdStrike Normalizer ready")

        # Layer 2 — Detector
        self.detector = NetworkIntrusionDetector(
            threshold=0.5,
            model_version="1.0.0"
        )

        # Load model if available
        if model_path and Path(model_path).exists():
            scaler_path = str(
                Path(model_path).parent / "scaler.pkl"
            )
            self.detector.load_model(
                model_path, scaler_path
            )
            print("  ✅ Layer 2: ML Model loaded")
            self.model_loaded = True
        else:
            print(
                "  ⚠️  Layer 2: No model loaded "
                "(normalization only)"
            )
            self.model_loaded = False

        print()

    def process_event(
        self,
        raw_event: dict,
        event_label: str = "Unknown"
    ) -> dict:
        """
        Process a single raw CrowdStrike event
        through the complete pipeline.

        Args:
            raw_event: Raw CrowdStrike JSON event
            event_label: Human readable label for display

        Returns:
            Pipeline result dictionary
        """
        result = {
            "event_label": event_label,
            "raw_event_type": raw_event.get(
                "metadata", {}
            ).get("eventType", "Unknown"),
            "layer1_success": False,
            "layer2_scored": False,
            "normalized_event": None,
            "detection_result": None,
            "pipeline_notes": []
        }

        # ---- LAYER 1: NORMALIZATION ----
        normalized = self.normalizer.normalize(raw_event)

        if normalized is None:
            result["pipeline_notes"].append(
                "Layer 1: Normalization failed"
            )
            return result

        result["layer1_success"] = True
        result["normalized_event"] = normalized
        result["pipeline_notes"].append(
            f"Layer 1: Normalized successfully "
            f"(severity={normalized.event.severity})"
        )

        # ---- LAYER 2: ML SCORING ----
        if not self.model_loaded:
            result["pipeline_notes"].append(
                "Layer 2: Skipped - no model loaded"
            )
            return result

        detection = self.detector.score_ecs_event(normalized)

        if detection is None:
            result["pipeline_notes"].append(
                f"Layer 2: Event type "
                f"'{normalized.event.category}' "
                f"not scoreable by intrusion detector "
                f"(no network features). "
                f"Route to appropriate model."
            )
        else:
            result["layer2_scored"] = True
            result["detection_result"] = detection
            result["pipeline_notes"].append(
                f"Layer 2: Scored successfully "
                f"(risk={detection.risk_score}, "
                f"confidence={detection.confidence})"
            )

        return result

    def run_demonstration(self, events: list) -> None:
        """
        Run multiple events through pipeline
        and display formatted results.
        """
        print("=" * 65)
        print("ABUTECH AI SECURITY PLATFORM")
        print("End-to-End Pipeline Demonstration")
        print("=" * 65)

        for label, raw_event in events:
            print(f"\n{'─' * 65}")
            print(f"EVENT: {label}")
            print(f"Type:  {raw_event['metadata']['eventType']}")
            print(f"{'─' * 65}")

            result = self.process_event(raw_event, label)

            # Display Layer 1 results
            print("\n📥 LAYER 1 — NORMALIZATION")
            if result["layer1_success"]:
                norm = result["normalized_event"]
                print(f"   Status:    ✅ Success")
                print(
                    f"   Timestamp: {norm.timestamp}"
                )
                print(
                    f"   Host:      "
                    f"{norm.host.hostname}"
                )
                print(
                    f"   User:      "
                    f"{norm.user.domain}\\"
                    f"{norm.user.name}"
                )
                print(
                    f"   Category:  {norm.event.category}"
                )
                print(
                    f"   Dataset:   {norm.event.dataset}"
                )

                # Show process details if available
                if norm.process:
                    print(
                        f"   Process:   {norm.process.name}"
                    )
                    if norm.process.parent:
                        print(
                            f"   Parent:    "
                            f"{norm.process.parent.name}"
                        )
                    if norm.process.command_line:
                        cmd = norm.process.command_line
                        if len(cmd) > 50:
                            cmd = cmd[:47] + "..."
                        print(f"   Command:   {cmd}")

                # Show network details if available
                if norm.source and norm.destination:
                    print(
                        f"   Source:    "
                        f"{norm.source.ip}:"
                        f"{norm.source.port}"
                    )
                    print(
                        f"   Dest:      "
                        f"{norm.destination.ip}:"
                        f"{norm.destination.port}"
                    )

                # Show DNS details if available
                if (norm.destination and
                        norm.destination.domain):
                    print(
                        f"   Domain:    "
                        f"{norm.destination.domain}"
                    )

                # Show initial severity
                severity = norm.event.severity or 0
                severity_label = (
                    "🔴 CRITICAL" if severity >= 75
                    else "🟠 HIGH" if severity >= 50
                    else "🟡 MEDIUM" if severity >= 25
                    else "🟢 LOW"
                )
                print(
                    f"   Severity:  {severity_label} "
                    f"({severity}/100)"
                )
            else:
                print("   Status: ❌ Failed")

            # Display Layer 2 results
            print("\n🤖 LAYER 2 — ML SCORING")
            if result["layer2_scored"]:
                det = result["detection_result"]
                verdict = (
                    "🚨 ATTACK DETECTED"
                    if det.is_attack
                    else "✅ BENIGN"
                )
                print(f"   Verdict:    {verdict}")
                print(
                    f"   Risk Score: {det.risk_score}"
                )
                print(
                    f"   Confidence: {det.confidence}"
                )
                print(
                    f"   Latency:    "
                    f"{det.inference_time_ms}ms"
                )
                print(f"   Explanation:")

                # Wrap explanation text
                explanation = det.explanation
                words = explanation.split()
                line = "              "
                for word in words:
                    if len(line) + len(word) > 60:
                        print(line)
                        line = "              " + word
                    else:
                        line += " " + word
                if line.strip():
                    print(line)

            else:
                note = next(
                    (n for n in result["pipeline_notes"]
                     if "Layer 2" in n),
                    "Layer 2: Not scored"
                )
                print(f"   Status: ⚠️  {note}")

            # Pipeline notes
            print("\n📋 PIPELINE NOTES")
            for note in result["pipeline_notes"]:
                print(f"   → {note}")

        # Final statistics
        print(f"\n{'=' * 65}")
        print("PIPELINE STATISTICS")
        print(f"{'=' * 65}")

        norm_stats = self.normalizer.get_statistics()
        print(f"\nLayer 1 — Normalizer:")
        print(
            f"   Events processed: "
            f"{norm_stats['events_processed']}"
        )
        print(
            f"   Success rate:     "
            f"{norm_stats['success_rate_pct']}%"
        )

        if self.model_loaded:
            det_stats = self.detector.get_performance_stats()
            print(f"\nLayer 2 — Detector:")
            print(
                f"   Events scored:    "
                f"{det_stats['total_scored']}"
            )
            print(
                f"   Attacks detected: "
                f"{det_stats['total_attacks_detected']}"
            )
            print(
                f"   Avg latency:      "
                f"{det_stats['avg_inference_ms']}ms"
            )

        print(f"\n{'=' * 65}\n")


def main():

    # Initialize pipeline with trained model
    model_path = "models/intrusion_detection/best_model.pkl"

    pipeline = AbuTechPipeline(model_path=model_path)

    # Define events to process
    events = [
        (
            "Suspicious PowerShell from MSBuild",
            SUSPICIOUS_PROCESS_EVENT
        ),
        (
            "Outbound C2 Network Connection",
            SUSPICIOUS_NETWORK_EVENT
        ),
        (
            "DNS Request to Suspicious Domain",
            DNS_EVENT
        ),
        (
            "Normal Notepad Process",
            BENIGN_PROCESS_EVENT
        )
    ]

    # Run demonstration
    pipeline.run_demonstration(events)


if __name__ == "__main__":
    main()
    