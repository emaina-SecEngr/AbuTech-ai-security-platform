"""
Layer 1 — Data Ingestion
Ingestion Router

The dispatcher that connects all 27 normalizers.

Takes a raw event plus its identified source, looks
up the correct normalizer, calls it, and returns the
normalized event. This is the connective tissue that
turns 27 isolated normalizers into one pipeline.

DESIGN:
    A registry maps each source name to a callable
    that invokes the right normalizer method. Some
    normalizers have one normalize() method; some
    (like kubernetes, cwpp) have several. The
    registry encodes which method to call.

    Normalizers are lazily instantiated and cached
    so we do not rebuild them on every event.

ERROR HANDLING:
    An unknown source returns None (the pipeline
    skips it, logs it, continues). A normalizer
    that throws is caught so one bad event never
    crashes the pipeline.

USAGE:
    router = IngestionRouter()
    normalized = router.route(raw_event, "s3")
"""

import logging

from layer1_ingestion.normalizers.s3_normalizer\
    import S3Normalizer
from layer1_ingestion.normalizers.rds_normalizer\
    import RDSNormalizer
from layer1_ingestion.normalizers.kubernetes_normalizer\
    import KubernetesNormalizer
from layer1_ingestion.normalizers.cwpp_normalizer\
    import CWPPNormalizer
from layer1_ingestion.normalizers.defender_cloud_normalizer\
    import DefenderForCloudNormalizer
from layer1_ingestion.normalizers.purview_dlp_normalizer\
    import PurviewDLPNormalizer
from layer1_ingestion.normalizers.cspm_normalizer\
    import CSPMNormalizer

logger = logging.getLogger(__name__)


class IngestionRouter:
    """
    Routes raw events to the correct normalizer
    based on the identified source.

    The router lazily instantiates normalizers and
    caches them. It maintains a registry mapping
    each source to the function that normalizes it.
    """

    def __init__(self):
        # Normalizer instance cache
        self._cache = {}

        # Routing statistics
        self.routed = 0
        self.failed = 0
        self.unknown = 0
        self.by_source = {}

        # Build the routing registry.
        # Each entry maps a source name to a lambda
        # that takes (raw_event) and returns the
        # normalized dict.
        self._registry = self._build_registry()

    def _build_registry(self) -> dict:
        """
        Build the source -> normalize-function map.

        Lazily resolves normalizer instances via
        _get(). Some sources route to a specific
        method on a multi-method normalizer.
        """
        return {
            "s3": lambda e: self._get(
                "s3", S3Normalizer
            ).normalize(e),

            "rds": lambda e: self._get(
                "rds", RDSNormalizer
            ).normalize(e),

            "kubernetes": lambda e: self._route_k8s(e),

            "cwpp": lambda e: self._route_cwpp(e),

            "defender_cloud": lambda e: self._get(
                "defender_cloud",
                DefenderForCloudNormalizer
            ).normalize(e),

            "purview_dlp": lambda e: self._get(
                "purview_dlp", PurviewDLPNormalizer
            ).normalize(e),

            "cspm": lambda e: self._get(
                "cspm", CSPMNormalizer
            ).normalize(e),
        }

    def _get(self, key: str, cls):
        """Lazily instantiate and cache a normalizer"""
        if key not in self._cache:
            self._cache[key] = cls()
        return self._cache[key]

    def _route_k8s(self, e: dict) -> dict:
        """
        Kubernetes normalizer has multiple methods.
        Pick the right one based on event shape.
        """
        k8s = self._get(
            "kubernetes", KubernetesNormalizer
        )
        # Falco runtime alert
        if "rule" in e and "priority" in e:
            return k8s.normalize_falco_alert(e)
        # Vulnerability scan
        if "VulnerabilityID" in e or (
            "cve_id" in e
        ):
            return k8s.normalize_vulnerability(e)
        # Default: audit log
        return k8s.normalize(e)

    def _route_cwpp(self, e: dict) -> dict:
        """
        CWPP normalizer has vendor-specific methods.
        Pick based on distinctive fields.
        """
        cwpp = self._get("cwpp", CWPPNormalizer)
        # Aqua uses 'control'
        if "control" in e:
            return cwpp.normalize_aqua(e)
        # Falcon uses 'detect_name'
        if "detect_name" in e or (
            "instance_id" in e
        ):
            return cwpp.normalize_falcon_cwp(e)
        # Default: Prisma Cloud Compute
        return cwpp.normalize(e)

    def route(
        self,
        raw_event: dict,
        source: str
    ):
        """
        Route a raw event to its normalizer.

        Args:
            raw_event: The raw event dict
            source: Identified source name

        Returns:
            Normalized event dict, or None if the
            source is unknown or normalization fails
        """
        if source == "unknown" or (
            source not in self._registry
        ):
            self.unknown += 1
            logger.warning(
                f"No normalizer registered for "
                f"source: {source}"
            )
            return None

        try:
            normalizer_fn = self._registry[source]
            result = normalizer_fn(raw_event)

            # Normalizers are not uniform: some return
            # a plain dict (Pattern B), some return a
            # DataAccessEvent object. Normalize to dict
            # so downstream layers get one consistent type.
            if result is not None and not isinstance(
                result, dict
            ):
                if hasattr(result, "to_dict"):
                    result = result.to_dict()
                elif hasattr(result, "__dict__"):
                    result = dict(result.__dict__)

            self.routed += 1
            self.by_source[source] = (
                self.by_source.get(source, 0) + 1
            )
            return result

        except Exception as e:
            self.failed += 1
            logger.error(
                f"Normalization failed for source "
                f"{source}: {str(e)}",
                exc_info=True
            )
            return None

    def supported_sources(self) -> list:
        """Return list of sources the router handles"""
        return sorted(self._registry.keys())

    def get_statistics(self) -> dict:
        """Return routing statistics"""
        total = self.routed + self.failed + (
            self.unknown
        )
        success_rate = (
            self.routed / total * 100
            if total > 0 else 0
        )
        return {
            "total_events": total,
            "routed": self.routed,
            "failed": self.failed,
            "unknown": self.unknown,
            "success_rate_pct": round(
                success_rate, 2
            ),
            "by_source": dict(self.by_source)
        }