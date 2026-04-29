"""
AbuTech AI Security Platform
End-to-End Pipeline Demonstration — With Router

This script demonstrates the complete automated flow:
    Raw CrowdStrike EDR event
        ↓
    Layer 1: CrowdStrike Normalization
        ↓
    Layer 2: Automated Routing + ML Scoring
        NetworkIntrusionDetector  ← network, dns
        MalwareClassifier         ← process
        ↓
    RoutingResult with all model scores

Run with:
    python run_pipeline.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

sys.path.insert(0, str(Path(__file__).parent))

from layer1_ingestion.normalizers.crowdstrike_normalizer import (
    CrowdStrikeNormalizer
)
from layer2_ml.router import Layer2Router


# ============================================================
# SAMPLE RAW CROWDSTRIKE EVENTS
# ============================================================

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


def main():

    print("\n" + "=" * 65)
    print("ABUTECH AI SECURITY PLATFORM")
    print("End-to-End Pipeline — Layer 1 + Layer 2 Router")
    print("=" * 65)

    # ---- INITIALIZE LAYER 1 ----
    print("\nInitializing platform components...")
    normalizer = CrowdStrikeNormalizer()
    print("  ✅ Layer 1: CrowdStrike Normalizer")

    # ---- INITIALIZE LAYER 2 ROUTER ----
    router = Layer2Router()

    # Load models
    loaded = router.load_models(
        intrusion_model_path=(
            "models/intrusion_detection/best_model.pkl"
        ),
        intrusion_scaler_path=(
            "models/intrusion_detection/scaler.pkl"
        ),
        malware_model_path=(
            "models/malware_classifier/best_model.pkl"
        ),
        malware_scaler_path=(
            "models/malware_classifier/scaler.pkl"
        )
    )

    # Show model status
    status = router.get_model_status()
    for model_name, info in status.items():
        icon = "✅" if info["loaded"] else "⚠️ "
        categories = ", ".join(info["categories"])
        print(
            f"  {icon} Layer 2: {model_name} "
            f"[{categories}]"
        )

    # ---- DEFINE TEST EVENTS ----
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
            "DNS to Suspicious DGA Domain",
            DNS_EVENT
        ),
        (
            "Normal Notepad Process",
            BENIGN_PROCESS_EVENT
        )
    ]

    # ---- PROCESS EACH EVENT ----
    print(f"\n{'─' * 65}")

    for label, raw_event in events:
        event_type = raw_event["metadata"]["eventType"]

        print(f"\nEVENT: {label}")
        print(f"Type:  {event_type}")
        print(f"{'─' * 65}")

        # Layer 1: Normalize
        normalized = normalizer.normalize(raw_event)

        if normalized is None:
            print("❌ Layer 1: Normalization failed")
            continue

        # Layer 1 output
        severity = normalized.event.severity or 0
        severity_label = (
            "🔴 CRITICAL" if severity >= 75
            else "🟠 HIGH" if severity >= 50
            else "🟡 MEDIUM" if severity >= 25
            else "🟢 LOW"
        )

        print(f"\n📥 LAYER 1 — NORMALIZATION")
        print(f"   Status:   ✅ Success")
        print(f"   Host:     {normalized.host.hostname}")
        print(
            f"   User:     "
            f"{normalized.user.domain}\\"
            f"{normalized.user.name}"
        )
        print(
            f"   Category: {normalized.event.category}"
        )

        if normalized.process:
            print(
                f"   Process:  {normalized.process.name}"
            )
            if normalized.process.parent:
                print(
                    f"   Parent:   "
                    f"{normalized.process.parent.name}"
                )

        if normalized.source and normalized.destination:
            print(
                f"   Source:   "
                f"{normalized.source.ip}:"
                f"{normalized.source.port}"
            )
            print(
                f"   Dest:     "
                f"{normalized.destination.ip}:"
                f"{normalized.destination.port}"
            )

        if (normalized.destination and
                normalized.destination.domain):
            print(
                f"   Domain:   "
                f"{normalized.destination.domain}"
            )

        print(
            f"   Severity: {severity_label} "
            f"({severity}/100)"
        )

        # Layer 2: Route and score
        routing_result = router.route(normalized)

        print(f"\n🤖 LAYER 2 — AUTOMATED ROUTING")
        print(
            f"   Routed to: "
            f"{', '.join(routing_result.routed_to)}"
        )

        # Show intrusion detection result
        if routing_result.intrusion_result:
            det = routing_result.intrusion_result
            verdict = (
                "🚨 ATTACK" if det.is_attack
                else "✅ BENIGN"
            )
            print(f"\n   NetworkIntrusionDetector:")
            print(f"   Verdict:    {verdict}")
            print(f"   Risk Score: {det.risk_score}")
            print(f"   Confidence: {det.confidence}")
            print(f"   Latency:    {det.inference_time_ms}ms")

        # Show malware detection result
        if routing_result.malware_result:
            mal = routing_result.malware_result
            verdict = (
                "🚨 MALWARE" if mal.is_malicious
                else "✅ BENIGN"
            )
            print(f"\n   MalwareClassifier:")
            print(f"   Verdict:    {verdict}")
            print(f"   Risk Score: {mal.risk_score}")
            print(f"   Confidence: {mal.confidence}")
            print(f"   Latency:    {mal.inference_time_ms}ms")

            if mal.malware_indicators:
                print(f"   Indicators:")
                for indicator in mal.malware_indicators[:3]:
                    print(f"     → {indicator}")

            if mal.attack_techniques:
                print(
                    f"   ATT&CK: "
                    f"{', '.join(mal.attack_techniques[:2])}"
                )

        # Overall verdict
        verdict_icon = (
            "🚨" if routing_result.is_threat()
            else "✅"
        )
        print(f"\n   {'─' * 40}")
        print(
            f"   {verdict_icon} OVERALL: "
            f"{routing_result.overall_verdict} "
            f"(risk={routing_result.overall_risk_score})"
        )
        print(
            f"   Primary model: "
            f"{routing_result.primary_model}"
        )
        print(
            f"   Routing time: "
            f"{routing_result.routing_time_ms}ms"
        )

    # ---- FINAL STATISTICS ----
    print(f"\n{'=' * 65}")
    print("PLATFORM STATISTICS")
    print(f"{'=' * 65}")

    stats = router.get_statistics()
    norm_stats = normalizer.get_statistics()

    print(f"\nLayer 1 — Normalizer:")
    print(
        f"   Events processed: "
        f"{norm_stats['events_processed']}"
    )
    print(
        f"   Success rate:     "
        f"{norm_stats['success_rate_pct']}%"
    )

    print(f"\nLayer 2 — Router:")
    print(
        f"   Total routed:   {stats['total_routed']}"
    )
    print(
        f"   Threats found:  {stats['threat_count']}"
    )
    print(
        f"   Threat rate:    "
        f"{stats['threat_rate']:.1%}"
    )
    print(f"   By category:")
    for cat, count in stats["routing_counts"].items():
        print(f"     {cat}: {count}")

    print(f"\n{'=' * 65}\n")


if __name__ == "__main__":
    main()