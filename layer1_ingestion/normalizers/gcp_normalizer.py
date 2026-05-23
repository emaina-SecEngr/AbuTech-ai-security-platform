"""
Layer 1 — Data Ingestion
Google Cloud Platform Normalizer

Converts GCP audit logs and security events
into DataAccessEvent format.

GCP DATA SOURCES HANDLED:
    Cloud Audit Logs:
        Admin Activity  — IAM changes, resource creation
        Data Access     — BigQuery, GCS, CloudSQL reads
        System Events   — Google maintenance events
    
    BigQuery:
        Job completion logs (queries run)
        Table data access events
        Dataset permission changes
    
    Cloud Storage (GCS):
        Object read/write/delete events
        Bucket permission changes
        Public access changes
    
    GCP IAM:
        Service account key creation
        Role binding changes
        Policy modifications
    
    VPC Flow Logs:
        Network traffic metadata
        Inter-project communication
        Egress to internet
    
    Cloud SQL:
        Database access events
        Query execution logs
        Connection events

WHY GCP MATTERS FOR BANKS:
    Citi, Deutsche Bank, HSBC all use GCP.
    BigQuery is the dominant cloud data warehouse.
    "SELECT * FROM customer_data" at 3am
    by a service account = your svc_backup scenario
    but in GCP.
    
    GCP IAM is notoriously over-permissive.
    Service accounts often have Owner role.
    Your CIEM enricher catches this.

GCP EVENT FORMAT:
    Google Cloud Audit Logs use a consistent
    protoPayload format:
    {
      "protoPayload": {
        "authenticationInfo": {
          "principalEmail": "svc@project.iam.gserviceaccount.com"
        },
        "requestMetadata": {
          "callerIp": "185.220.101.45"
        },
        "resourceName": "projects/prod/datasets/customer_pii",
        "methodName": "google.bigquery.v2.TableDataService.list"
      }
    }

USAGE:
    normalizer = GCPNormalizer()
    data_event = normalizer.normalize(gcp_audit_log)
"""

import logging
import os
from datetime import datetime
from datetime import timezone

logger = logging.getLogger(__name__)

# GCP method names mapped to readable operations
GCP_METHOD_MAP = {
    # BigQuery
    "google.bigquery.v2.TableDataService.list":
        "bigquery_table_read",
    "google.bigquery.v2.TableDataService.insertAll":
        "bigquery_table_write",
    "google.bigquery.v2.JobService.insert":
        "bigquery_job_create",
    "google.bigquery.v2.DatasetService.get":
        "bigquery_dataset_access",
    "google.bigquery.v2.DatasetService.patch":
        "bigquery_dataset_modify",

    # Cloud Storage
    "storage.objects.get":
        "gcs_object_read",
    "storage.objects.create":
        "gcs_object_write",
    "storage.objects.delete":
        "gcs_object_delete",
    "storage.buckets.getIamPolicy":
        "gcs_bucket_iam_read",
    "storage.buckets.setIamPolicy":
        "gcs_bucket_iam_modify",

    # IAM
    "google.iam.admin.v1.CreateServiceAccountKey":
        "iam_key_create",
    "google.iam.admin.v1.DeleteServiceAccountKey":
        "iam_key_delete",
    "SetIamPolicy":
        "iam_policy_modify",
    "google.iam.admin.v1.CreateServiceAccount":
        "iam_service_account_create",

    # Cloud SQL
    "cloudsql.instances.login":
        "cloudsql_login",
    "cloudsql.instances.query":
        "cloudsql_query",

    # Compute
    "v1.compute.instances.delete":
        "compute_instance_delete",
    "v1.compute.instances.insert":
        "compute_instance_create",

    # Secret Manager
    "google.cloud.secretmanager.v1.SecretManagerService.AccessSecretVersion":
        "secret_access",
}

# GCP service names for data store identification
GCP_SERVICE_MAP = {
    "bigquery.googleapis.com": "gcp_bigquery",
    "storage.googleapis.com": "gcp_cloud_storage",
    "cloudsql.googleapis.com": "gcp_cloud_sql",
    "iam.googleapis.com": "gcp_iam",
    "secretmanager.googleapis.com": "gcp_secrets",
    "compute.googleapis.com": "gcp_compute",
    "container.googleapis.com": "gcp_gke",
}


class GCPNormalizer:
    """
    Normalizes Google Cloud Platform audit logs
    into DataAccessEvent format.

    Handles all GCP service types through the
    unified Cloud Audit Log format.
    """

    def __init__(self):
        self.source_system = "gcp"

    def normalize(self, raw_event: dict) -> dict:
        """
        Convert GCP audit log to DataAccessEvent.

        Args:
            raw_event: GCP Cloud Audit Log dict

        Returns:
            DataAccessEvent compatible dict
        """
        if not raw_event:
            return self._empty_event()

        proto = raw_event.get(
            "protoPayload", {}
        )

        accessor = self._extract_accessor(proto)
        source_ip = self._extract_source_ip(proto)
        data_store = self._extract_data_store(
            proto, raw_event
        )
        data_path = self._extract_data_path(proto)
        bytes_accessed = self._extract_bytes(
            proto, raw_event
        )
        event_time = self._extract_time(raw_event)
        method = proto.get("methodName", "")
        operation = GCP_METHOD_MAP.get(
            method, method
        )
        service = raw_event.get(
            "resource", {}
        ).get("type", "")
        source_system = GCP_SERVICE_MAP.get(
            proto.get("serviceName", ""),
            "gcp_unknown"
        )
        accessor_type = self._detect_accessor_type(
            accessor
        )
        classification = self._detect_classification(
            data_store, data_path
        )
        risk_score, risk_reasons = (
            self._calculate_risk(
                raw_event, proto,
                accessor, source_ip,
                method, classification
            )
        )

        return {
            "accessor_identity": accessor,
            "accessor_type": accessor_type,
            "data_store_name": data_store,
            "data_path": data_path,
            "data_classification": classification,
            "bytes_accessed": bytes_accessed,
            "event_time": event_time,
            "source_ip": source_ip,
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "source_system": source_system,
            "raw_event": raw_event,
            "gcp_method": method,
            "gcp_operation": operation,
            "gcp_service": service,
            "gcp_project": self._extract_project(
                raw_event
            )
        }

    def normalize_bigquery_event(
        self, raw_event: dict
    ) -> dict:
        """
        Specialized normalization for BigQuery events.
        BigQuery is highest risk for data exfiltration.
        Large queries = potential data theft.
        """
        base = self.normalize(raw_event)

        proto = raw_event.get("protoPayload", {})
        stats = proto.get(
            "serviceData", {}
        ).get(
            "jobCompletedEvent", {}
        ).get("job", {}).get(
            "jobStatistics", {}
        )

        billed_bytes = stats.get(
            "totalBilledBytes", 0
        )
        processed_bytes = stats.get(
            "totalProcessedBytes", 0
        )

        if billed_bytes > 0:
            base["bytes_accessed"] = int(billed_bytes)

        query = (
            proto.get("serviceData", {})
            .get("jobCompletedEvent", {})
            .get("job", {})
            .get("jobConfiguration", {})
            .get("query", {})
            .get("query", "")
        )

        if query:
            base["data_path"] = query[:500]
            base["gcp_query"] = query[:2000]

            if self._is_suspicious_query(query):
                base["risk_score"] = min(
                    base["risk_score"] + 0.20, 1.0
                )
                base["risk_reasons"].append(
                    "suspicious_bigquery_pattern"
                )

        return base

    def _extract_accessor(
        self, proto: dict
    ) -> str:
        """Extract principal email from GCP proto"""
        auth_info = proto.get(
            "authenticationInfo", {}
        )

        email = auth_info.get(
            "principalEmail", ""
        )
        if email:
            return email

        service_account = auth_info.get(
            "serviceAccountDelegationInfo", []
        )
        if service_account:
            return service_account[0].get(
                "principalEmail", "unknown"
            )

        return "unknown"

    def _extract_source_ip(
        self, proto: dict
    ) -> str:
        """Extract caller IP from GCP request metadata"""
        metadata = proto.get(
            "requestMetadata", {}
        )

        ip = metadata.get("callerIp", "")
        if ip and ip not in [
            "private", "unknown", ""
        ]:
            return ip

        return ""

    def _extract_data_store(
        self,
        proto: dict,
        raw_event: dict
    ) -> str:
        """Extract resource name as data store"""
        resource_name = proto.get(
            "resourceName", ""
        )

        if resource_name:
            parts = resource_name.split("/")
            if len(parts) >= 4:
                return "/".join(parts[:4])
            return resource_name

        resource = raw_event.get("resource", {})
        labels = resource.get("labels", {})

        for key in [
            "dataset_id", "bucket_name",
            "instance_id", "database_id"
        ]:
            val = labels.get(key, "")
            if val:
                return val

        return "gcp-unknown-resource"

    def _extract_data_path(
        self, proto: dict
    ) -> str:
        """Extract specific resource path"""
        resource_name = proto.get(
            "resourceName", ""
        )
        if resource_name:
            return resource_name

        request = proto.get("request", {})
        for field in [
            "name", "resource", "object",
            "tableId", "datasetId"
        ]:
            val = request.get(field, "")
            if val:
                return str(val)[:500]

        return ""

    def _extract_bytes(
        self,
        proto: dict,
        raw_event: dict
    ) -> int:
        """Extract bytes processed/accessed"""
        for path in [
            ["serviceData", "jobCompletedEvent",
             "job", "jobStatistics",
             "totalBilledBytes"],
            ["response", "totalBytesProcessed"],
            ["request", "maxResults"]
        ]:
            val = proto
            for key in path:
                if isinstance(val, dict):
                    val = val.get(key, {})
                else:
                    val = None
                    break
            if val and isinstance(val, (int, str)):
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass

        return 0

    def _extract_time(
        self, raw_event: dict
    ) -> str:
        """Extract and normalize timestamp"""
        ts = raw_event.get(
            "timestamp",
            raw_event.get(
                "receiveTimestamp", ""
            )
        )
        if ts:
            if "T" in ts and (
                "Z" in ts or "+" in ts
            ):
                return ts.replace("+00:00", "Z")
        return _now()

    def _extract_project(
        self, raw_event: dict
    ) -> str:
        """Extract GCP project ID"""
        resource = raw_event.get("resource", {})
        labels = resource.get("labels", {})
        return labels.get(
            "project_id",
            raw_event.get("logName", "")
            .split("/")[1]
            if "/" in raw_event.get("logName", "")
            else ""
        )

    def _detect_accessor_type(
        self, accessor: str
    ) -> str:
        """Detect if GCP principal is service account"""
        if not accessor:
            return "unknown"

        if "iam.gserviceaccount.com" in accessor:
            return "service_account"

        if "@" not in accessor:
            return "service_account"

        if any(
            kw in accessor.lower()
            for kw in [
                "svc-", "svc_", "-sa@",
                "service", "func-",
                "robot", "terraform"
            ]
        ):
            return "service_account"

        return "human"

    def _detect_classification(
        self,
        data_store: str,
        data_path: str
    ) -> str:
        """Detect data sensitivity classification"""
        combined = (
            data_store.lower() + " " +
            data_path.lower()
        )

        if any(
            kw in combined
            for kw in [
                "pci", "card", "payment",
                "credit", "cardholder"
            ]
        ):
            return "PCI"

        if any(
            kw in combined
            for kw in [
                "phi", "health", "medical",
                "patient", "hipaa"
            ]
        ):
            return "PHI"

        if any(
            kw in combined
            for kw in [
                "pii", "personal", "ssn",
                "customer", "member", "user_data"
            ]
        ):
            return "PII"

        if any(
            kw in combined
            for kw in [
                "internal", "confidential",
                "restricted", "sensitive"
            ]
        ):
            return "INTERNAL"

        return "UNKNOWN"

    def _calculate_risk(
        self,
        raw_event: dict,
        proto: dict,
        accessor: str,
        source_ip: str,
        method: str,
        classification: str
    ) -> tuple:
        """Calculate initial risk score"""
        risk = 0.0
        reasons = []

        # After hours
        event_time = self._extract_time(raw_event)
        try:
            hour = int(event_time[11:13])
            if hour < 6 or hour > 22:
                risk += 0.15
                reasons.append("after_hours")
        except (ValueError, IndexError):
            pass

        # Tor or suspicious IP
        if source_ip:
            if source_ip.startswith("185.220"):
                risk += 0.40
                reasons.append("tor_exit_node")
            elif not any(
                source_ip.startswith(p)
                for p in ["10.", "192.168.", "172."]
            ):
                if source_ip:
                    risk += 0.05
                    reasons.append(
                        "external_ip_access"
                    )

        # High risk GCP operations
        high_risk_methods = [
            "SetIamPolicy",
            "google.iam.admin.v1.CreateServiceAccountKey",
            "storage.buckets.setIamPolicy",
            "google.bigquery.v2.TableDataService.list"
        ]
        if method in high_risk_methods:
            risk += 0.20
            reasons.append(
                f"high_risk_gcp_method:{method}"
            )

        # Service account doing sensitive ops
        if (
            "iam.gserviceaccount.com" in accessor and
            "iam" in method.lower()
        ):
            risk += 0.15
            reasons.append(
                "service_account_iam_modification"
            )

        # Data classification
        if classification == "PCI":
            risk += 0.20
            reasons.append("pci_data_access")
        elif classification == "PHI":
            risk += 0.18
            reasons.append("phi_data_access")
        elif classification == "PII":
            risk += 0.15
            reasons.append("pii_data_access")

        # IAM key creation — high risk
        if "CreateServiceAccountKey" in method:
            risk += 0.25
            reasons.append(
                "new_service_account_key_created"
            )

        return min(risk, 1.0), reasons

    def _is_suspicious_query(
        self, query: str
    ) -> bool:
        """Detect suspicious BigQuery patterns"""
        query_lower = query.lower()

        suspicious = [
            "select *",
            "limit 1000000",
            "where 1=1",
            "information_schema",
            "union select",
            "drop table",
            "delete from",
        ]

        return any(
            s in query_lower for s in suspicious
        )

    def _empty_event(self) -> dict:
        """Return empty event for invalid input"""
        return {
            "accessor_identity": "unknown",
            "accessor_type": "unknown",
            "data_store_name": "unknown",
            "data_path": "",
            "data_classification": "UNKNOWN",
            "bytes_accessed": 0,
            "event_time": _now(),
            "source_ip": "",
            "risk_score": 0.0,
            "risk_reasons": [],
            "source_system": "gcp",
            "raw_event": {}
        }


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")