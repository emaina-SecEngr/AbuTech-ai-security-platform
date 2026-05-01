"""
AbuTech AI Security Platform
End-to-End Pipeline — Layer 1 + Layer 2 + Layer 3

Complete flow:
    Raw CrowdStrike EDR event
        ↓
    Layer 1: Normalization
        ↓
    Layer 2: ML Scoring + Routing
        ↓
    Layer 3: Knowledge Graph Building + Enrichment
        ↓
    Graph Summary with threat intelligence context
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
from layer3_knowledge.graph.security_graph import (
    SecurityKnowledgeGraph
)
from layer3_knowledge.graph.graph_builder import (
    GraphBuilder
)
from layer3_knowledge.enrichment.threat_enricher import (
    ThreatEnricher
)


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
    print("End-to-End Pipeline — Layers 1 + 2 + 3")
    print("=" * 65)

    # ---- INITIALIZE LAYER 1 ----
    print("\nInitializing platform components...")
    normalizer = CrowdStrikeNormalizer()
    print("  ✅ Layer 1: CrowdStrike Normalizer")

    # ---- INITIALIZE LAYER 2 ROUTER ----
    router = Layer2Router()
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

    status = router.get_model_status()
    for model_name, info in status.items():
        icon = "✅" if info["loaded"] else "⚠️ "
        categories = ", ".join(info["categories"])
        print(
            f"  {icon} Layer 2: {model_name} "
            f"[{categories}]"
        )

    # ---- INITIALIZE LAYER 3 ----
    graph = SecurityKnowledgeGraph()
    builder = GraphBuilder(graph)
    enricher = ThreatEnricher(graph)
    print("  ✅ Layer 3: Knowledge Graph ready")

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

        print(f"\n🤖 LAYER 2 — ML SCORING")
        print(
            f"   Routed to: "
            f"{', '.join(routing_result.routed_to)}"
        )

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
            if mal.malware_indicators:
                print(f"   Indicators:")
                for ind in mal.malware_indicators[:2]:
                    print(f"     → {ind}")
            if mal.attack_techniques:
                print(
                    f"   ATT&CK: "
                    f"{', '.join(mal.attack_techniques[:2])}"
                )

        if routing_result.intrusion_result:
            intr = routing_result.intrusion_result
            verdict = (
                "🚨 ATTACK" if intr.is_attack
                else "✅ BENIGN"
            )
            print(f"\n   NetworkIntrusionDetector:")
            print(f"   Verdict:    {verdict}")
            print(f"   Risk Score: {intr.risk_score}")

        if routing_result.dns_result:
            dns = routing_result.dns_result
            verdict = (
                "🚨 DGA DETECTED" if dns.is_dga
                else "✅ LEGITIMATE"
            )
            print(f"\n   DNSClassifier:")
            print(f"   Verdict:    {verdict}")
            print(f"   Risk Score: {dns.risk_score}")
            print(f"   DGA Family: {dns.dga_family}")
            if dns.dga_indicators:
                for ind in dns.dga_indicators[:2]:
                    print(f"     → {ind}")

        verdict_icon = (
            "🚨" if routing_result.is_threat()
            else "✅"
        )
        print(
            f"\n   {verdict_icon} OVERALL: "
            f"{routing_result.overall_verdict} "
            f"(risk={routing_result.overall_risk_score})"
        )

        # Layer 3: Build knowledge graph
        builder.process_routing_result(
            normalized, routing_result
        )
        print(f"\n🕸️  LAYER 3 — KNOWLEDGE GRAPH")
        stats = graph.get_statistics()
        print(
            f"   Nodes: {stats['total_nodes']}  "
            f"Edges: {stats['total_edges']}"
        )
        print(f"   Node types: {stats['node_types']}")

    # ---- LAYER 3 ENRICHMENT ----
    print(f"\n{'=' * 65}")
    print("LAYER 3 — THREAT INTELLIGENCE ENRICHMENT")
    print(f"{'=' * 65}")

    enrichment_results = enricher.enrich_all()
    print(f"\n  IPs enriched:      "
          f"{enrichment_results['ips_enriched']}")
    print(f"  Domains enriched:  "
          f"{enrichment_results['domains_enriched']}")
    print(f"  Campaigns found:   "
          f"{enrichment_results['campaigns_identified']}")

    # ---- HOST ENRICHMENT ----
    host_enrichment = enricher.enrich_host_from_graph(
        "WKSTN-JSMITH-01"
    )

    if host_enrichment:
        print(f"\n  Host: WKSTN-JSMITH-01")
        print(
            f"  Graph-enriched risk: "
            f"{host_enrichment['host_risk_score']:.2f} "
            f"{host_enrichment['host_risk_label']}"
        )
        print(
            f"  Threat connections: "
            f"{host_enrichment['threat_connections']}"
        )
        if host_enrichment["malicious_connections"]:
            print(f"  Connected threats:")
            for conn in (
                host_enrichment["malicious_connections"]
            ):
                print(
                    f"    → {conn['entity']} "
                    f"({conn['type']}) "
                    f"risk={conn['risk']:.2f}"
                )

    # ---- GRAPH SUMMARY ----
    print(f"\n{'=' * 65}")
    print("KNOWLEDGE GRAPH SUMMARY")
    print(f"{'=' * 65}")
    print(graph.get_summary())

    # ---- THREAT SUMMARY ----
    threat_summary = enricher.get_threat_summary()
    print(f"Active Alerts:    "
          f"{threat_summary['active_alerts']}")
    print(f"High Risk Entities: "
          f"{threat_summary['high_risk_entities']}")

    if threat_summary["alert_details"]:
        print(f"\nAlert Details:")
        for alert in threat_summary["alert_details"]:
            print(
                f"  🚨 {alert['type']} "
                f"(risk={alert['risk']:.2f})"
            )

    print(f"\n{'=' * 65}\n")


if __name__ == "__main__":
    main()