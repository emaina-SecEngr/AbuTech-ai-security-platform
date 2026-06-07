"""
Tests for the Layer 1 Ingestion Pipeline:
SourceDetector, IngestionRouter, and
IngestionPipeline.
"""

import pytest
from layer1_ingestion.pipeline.source_detector\
    import SourceDetector, KNOWN_SOURCES
from layer1_ingestion.pipeline.ingestion_router\
    import IngestionRouter
from layer1_ingestion.pipeline.ingestion_pipeline\
    import IngestionPipeline


# ============================================================
# FIXTURES — realistic raw events per source
# ============================================================

@pytest.fixture
def detector():
    return SourceDetector()


@pytest.fixture
def router():
    return IngestionRouter()


@pytest.fixture
def pipeline():
    return IngestionPipeline()


@pytest.fixture
def s3_event():
    return {
        "eventName": "GetObject",
        "eventSource": "s3.amazonaws.com",
        "requestParameters": {
            "bucketName": "prod-pci-data"
        },
        "sourceIPAddress": "185.220.101.45",
        "userIdentity": {"userName": "svc_backup"}
    }


@pytest.fixture
def k8s_audit_event():
    return {
        "objectRef": {
            "resource": "secrets",
            "namespace": "kube-system",
            "name": "db-creds"
        },
        "verb": "get",
        "user": {"username": "system:serviceaccount:x"},
        "sourceIPs": ["10.0.0.5"],
        "requestReceivedTimestamp": "2026-06-01T03:00:00Z"
    }


@pytest.fixture
def falco_event():
    return {
        "rule": "Terminal Shell in Container",
        "priority": "CRITICAL",
        "output": "shell spawned in container",
        "output_fields": {
            "container.name": "payment-api",
            "proc.name": "bash"
        }
    }


@pytest.fixture
def defender_event():
    return {
        "properties": {
            "alertDisplayName": "Crypto mining detected",
            "description": "crypto mining activity",
            "severity": "High",
            "intent": "Execution",
            "compromisedEntity": "prod-vm-01"
        }
    }


@pytest.fixture
def purview_event():
    return {
        "UserId": "john@company.com",
        "Workload": "Exchange",
        "DLPAction": "Block",
        "SensitiveInfoTypeData": [
            {"SensitiveType": "Credit Card Number"}
        ]
    }


@pytest.fixture
def cwpp_prisma_event():
    return {
        "severity": "CRITICAL",
        "rule": "Container escape detected",
        "hostname": "prod-node-01",
        "containerName": "payment-api"
    }

@pytest.fixture
def guardduty_event():
    return {
        "type": "CryptoCurrency:EC2/BitcoinTool.B!DNS",
        "severity": 8.0,
        "title": "Crypto mining detected",
        "resource": {
            "resourceType": "Instance",
            "instanceDetails": {"instanceId": "i-0abc"}
        },
        "service": {"count": 1}
    }



@pytest.fixture
def cwpp_aqua_event():
    return {
        "severity": "CRITICAL",
        "control": "Reverse shell established",
        "container": "api-gw",
        "image": "internal/api:latest"
    }


# ============================================================
# SOURCE DETECTOR
# ============================================================

class TestSourceDetector:

    def test_known_sources_populated(self):
        assert "s3" in KNOWN_SOURCES
        assert "defender_cloud" in KNOWN_SOURCES
        assert len(KNOWN_SOURCES) >= 26

    def test_explicit_hint_honored(
        self, detector, s3_event
    ):
        result = detector.detect(
            s3_event, hint="crowdstrike"
        )
        assert result == "crowdstrike"

    def test_hint_case_insensitive(
        self, detector, s3_event
    ):
        result = detector.detect(
            s3_event, hint="S3"
        )
        assert result == "s3"

    def test_detect_s3(self, detector, s3_event):
        assert detector.detect(s3_event) == "s3"

    def test_detect_kubernetes_audit(
        self, detector, k8s_audit_event
    ):
        assert detector.detect(
            k8s_audit_event
        ) == "kubernetes"

    def test_detect_falco_as_kubernetes(
        self, detector, falco_event
    ):
        assert detector.detect(
            falco_event
        ) == "kubernetes"

    def test_detect_defender(
        self, detector, defender_event
    ):
        assert detector.detect(
            defender_event
        ) == "defender_cloud"

    def test_detect_purview(
        self, detector, purview_event
    ):
        assert detector.detect(
            purview_event
        ) == "purview_dlp"

    def test_detect_cwpp_prisma(
        self, detector, cwpp_prisma_event
    ):
        assert detector.detect(
            cwpp_prisma_event
        ) == "cwpp"

    def test_detect_cwpp_aqua(
        self, detector, cwpp_aqua_event
    ):
        assert detector.detect(
            cwpp_aqua_event
        ) == "cwpp"

    def test_empty_event_unknown(self, detector):
        assert detector.detect({}) == "unknown"

    def test_none_event_unknown(self, detector):
        assert detector.detect(None) == "unknown"

    def test_unrecognizable_unknown(self, detector):
        assert detector.detect(
            {"random_field": "value"}
        ) == "unknown"

    def test_statistics_tracked(
        self, detector, s3_event
    ):
        detector.detect(s3_event)
        detector.detect({})
        stats = detector.get_statistics()
        assert stats["total"] == 2
        assert stats["detected"] == 1
        assert stats["unknown"] == 1


# ============================================================
# INGESTION ROUTER
# ============================================================

class TestIngestionRouter:

    def test_supported_sources(self, router):
        sources = router.supported_sources()
        assert "s3" in sources
        assert "defender_cloud" in sources
        assert "purview_dlp" in sources

    def test_route_s3(self, router, s3_event):
        result = router.route(s3_event, "s3")
        assert result is not None
        assert isinstance(result, dict)

    def test_route_defender(
        self, router, defender_event
    ):
        result = router.route(
            defender_event, "defender_cloud"
        )
        assert result is not None
        assert result["source_system"] == (
            "defender_for_cloud"
        )

    def test_route_purview(
        self, router, purview_event
    ):
        result = router.route(
            purview_event, "purview_dlp"
        )
        assert result is not None
        assert result["source_system"] == (
            "purview_dlp"
        )

    def test_route_cwpp_prisma(
        self, router, cwpp_prisma_event
    ):
        result = router.route(
            cwpp_prisma_event, "cwpp"
        )
        assert result is not None
        assert result["cwpp_vendor"] == (
            "prisma_cloud_compute"
        )

    def test_route_cwpp_aqua_method(
        self, router, cwpp_aqua_event
    ):
        result = router.route(
            cwpp_aqua_event, "cwpp"
        )
        assert result is not None
        assert result["cwpp_vendor"] == (
            "aqua_security"
        )

    def test_route_falco_method(
        self, router, falco_event
    ):
        result = router.route(
            falco_event, "kubernetes"
        )
        assert result is not None
        assert result["source_system"] == (
            "falco_runtime"
        )

    def test_unknown_source_returns_none(
        self, router, s3_event
    ):
        result = router.route(s3_event, "unknown")
        assert result is None

    def test_unregistered_source_returns_none(
        self, router, s3_event
    ):
        result = router.route(
            s3_event, "nonexistent_source"
        )
        assert result is None

    def test_statistics_tracked(
        self, router, s3_event
    ):
        router.route(s3_event, "s3")
        router.route(s3_event, "unknown")
        stats = router.get_statistics()
        assert stats["routed"] == 1
        assert stats["unknown"] == 1
        assert stats["by_source"]["s3"] == 1


# ============================================================
# INGESTION PIPELINE (end to end)
# ============================================================

class TestIngestionPipeline:

    def test_pipeline_initializes(self, pipeline):
        assert pipeline is not None
        assert pipeline.detector is not None
        assert pipeline.router is not None

    def test_ingest_with_hint(
        self, pipeline, s3_event
    ):
        result = pipeline.ingest(
            s3_event, source="s3"
        )
        assert result is not None
        assert result["ingestion_source"] == "s3"

    def test_ingest_inferred(
        self, pipeline, defender_event
    ):
        # No hint - pipeline must detect it
        result = pipeline.ingest(defender_event)
        assert result is not None
        assert result["ingestion_source"] == (
            "defender_cloud"
        )

    def test_ingest_purview_inferred(
        self, pipeline, purview_event
    ):
        result = pipeline.ingest(purview_event)
        assert result is not None
        assert result["ingestion_source"] == (
            "purview_dlp"
        )

    def test_ingest_falco_inferred(
        self, pipeline, falco_event
    ):
        result = pipeline.ingest(falco_event)
        assert result is not None
        assert result["ingestion_source"] == (
            "kubernetes"
        )

    def test_ingest_guardduty_inferred(
        self, pipeline, guardduty_event
    ):
        result = pipeline.ingest(guardduty_event)
        assert result is not None
        assert result["ingestion_source"] == (
            "guardduty"
        )
        assert result["mitre_technique"] == "T1496"
        
    def test_ingest_invalid_returns_none(
        self, pipeline
    ):
        assert pipeline.ingest(None) is None
        assert pipeline.ingest({}) is None

    def test_ingest_unknown_returns_none(
        self, pipeline
    ):
        result = pipeline.ingest(
            {"mystery": "data"}
        )
        assert result is None

    def test_ingest_batch(
        self, pipeline, s3_event,
        defender_event, purview_event
    ):
        batch = [
            s3_event, defender_event, purview_event
        ]
        results = pipeline.ingest_batch(batch)
        assert len(results) == 3

    def test_ingest_batch_skips_failures(
        self, pipeline, s3_event
    ):
        batch = [
            s3_event,
            {},                    # invalid
            {"mystery": "data"},   # unknown
            s3_event
        ]
        results = pipeline.ingest_batch(batch)
        # Only the 2 valid s3 events normalize
        assert len(results) == 2

    def test_ingest_empty_batch(self, pipeline):
        assert pipeline.ingest_batch([]) == []

    def test_batch_with_shared_hint(
        self, pipeline, s3_event
    ):
        batch = [s3_event, s3_event, s3_event]
        results = pipeline.ingest_batch(
            batch, source="s3"
        )
        assert len(results) == 3

    def test_statistics_complete(
        self, pipeline, s3_event
    ):
        pipeline.ingest(s3_event, source="s3")
        pipeline.ingest({})
        stats = pipeline.get_statistics()
        assert stats["pipeline"]["received"] == 2
        assert stats["pipeline"]["normalized"] == 1
        assert stats["pipeline"]["dropped"] == 1
        assert "detector" in stats
        assert "router" in stats

    def test_supported_sources(self, pipeline):
        sources = pipeline.supported_sources()
        assert isinstance(sources, list)
        assert "s3" in sources

    def test_end_to_end_defender_full(
        self, pipeline, defender_event
    ):
        # Full path: detect -> route -> tag
        result = pipeline.ingest(defender_event)
        assert result["ingestion_source"] == (
            "defender_cloud"
        )
        assert result["mitre_technique"] == "T1496"
        assert result["risk_score"] >= 0.80