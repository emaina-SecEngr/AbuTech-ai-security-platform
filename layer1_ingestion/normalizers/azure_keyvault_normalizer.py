"""
Layer 1 — Data Ingestion
Azure Key Vault Normalizer

This module transforms Azure Key Vault diagnostic
logs into IamSecretEvent objects.

Azure Key Vault Diagnostic Log Structure:
    Azure logs every operation against Key Vault
    through Azure Monitor diagnostic settings.
    
    The logs are structured JSON with this format:
    {
        "time": "2024-03-29T09:19:00Z",
        "resourceId": "/subscriptions/sub-id/
                       resourceGroups/prod-rg/
                       providers/Microsoft.KeyVault/
                       vaults/prod-vault",
        "operationName": "SecretGet",
        "operationVersion": "7.0",
        "category": "AuditEvent",
        "resultType": "Success",
        "resultSignature": "OK",
        "callerIpAddress": "10.0.0.155",
        "correlationId": "abc-123",
        "identity": {
            "claim": {
                "oid": "user-object-id",
                "upn": "jsmith@corp.com",
                "appid": "app-client-id"
            }
        },
        "properties": {
            "id": "https://prod-vault.vault.azure.net/
                   secrets/db-password/version",
            "clientInfo": "Python/3.9",
            "httpStatusCode": 200,
            "requestUri": "https://prod-vault...",
            "isAddressAuthorized": true
        }
    }

Key Differences From AWS:
    Azure uses "SecretGet" not "GetSecretValue"
    Identity in "identity.claim.upn" not
    "userIdentity.userName"
    Secret path in "properties.id" as full URL
    not just path string

Azure Specific Risk Signals:
    isAddressAuthorized = false
        → Access from unauthorized network
        → High risk regardless of success

    Service Principal access (appid present):
        → Application/service account
        → Higher scrutiny needed

    Access from outside tenant:
        → External identity accessing secrets
        → Potential data exfiltration
"""

import logging
from typing import Optional

from layer1_ingestion.schema.iam_schema import (
    IamSecretEvent,
    IamEvent
)

logger = logging.getLogger(__name__)


# ============================================================
# AZURE KEY VAULT OPERATION MAPPINGS
# ============================================================

# Read operations
READ_OPERATIONS = {
    "SecretGet",
    "SecretList",
    "SecretGetVersions",
    "KeyGet",
    "KeyList",
    "CertificateGet",
    "CertificateList"
}

# Write operations
WRITE_OPERATIONS = {
    "SecretSet",
    "SecretBackup",
    "SecretRestore",
    "KeyCreate",
    "KeyImport",
    "CertificateCreate",
    "CertificateImport"
}

# Delete operations
DELETE_OPERATIONS = {
    "SecretDelete",
    "SecretPurge",
    "KeyDelete",
    "KeyPurge",
    "CertificateDelete",
    "CertificatePurge"
}

# High risk operations
HIGH_RISK_OPERATIONS = {
    "SecretDelete",
    "SecretPurge",
    "SecretList",   # enumeration
    "KeyDelete",
    "CertificateDelete"
}

# Business hours (UTC)
BUSINESS_HOURS_START = 13
BUSINESS_HOURS_END = 23

# Adaptive thresholds
BULK_THRESHOLD_BUSINESS = 10
BULK_THRESHOLD_OFFHOURS = 3


class AzureKeyVaultNormalizer:
    """
    Normalizes Azure Key Vault diagnostic logs
    to IamSecretEvent objects.

    Handles three Azure identity types:
    - User (upn field) — human identity
    - Service Principal (appid field) — application
    - Managed Identity — Azure-managed service account

    Usage:
        normalizer = AzureKeyVaultNormalizer()
        iam_event = normalizer.normalize(
            raw_azure_diagnostic_log
        )
    """

    def __init__(self):
        self._access_history = {}
        self._known_paths = {}

        self.events_processed = 0
        self.bulk_access_detected = 0
        self.high_risk_events = 0
        self.unauthorized_access_attempts = 0

        logger.info(
            "AzureKeyVaultNormalizer initialized"
        )

    def normalize(
        self,
        raw_event: dict
    ) -> Optional[IamEvent]:
        """
        Normalize Azure Key Vault diagnostic log event.

        ETL Pipeline:
        1. Validate this is a Key Vault event
        2. Extract identity from claim fields
        3. Extract secret path from URL format
        4. Detect unauthorized network access
        5. Apply bulk access detection
        6. Calculate risk score
        7. Return IamEvent container

        Args:
            raw_event: Raw Azure diagnostic log dict

        Returns:
            IamEvent or None if not Key Vault event
        """
        if not raw_event:
            return None

        # Validate Key Vault event
        resource_id = raw_event.get("resourceId", "")
        category = raw_event.get("category", "")

        if (
            "KeyVault" not in resource_id and
            "AuditEvent" not in category
        ):
            if not raw_event.get("operationName", ""):
                return None

        try:
            # ---- EXTRACT CORE FIELDS ----
            operation_name = raw_event.get(
                "operationName", ""
            )
            event_time = raw_event.get("time", "")
            result_type = raw_event.get(
                "resultType", ""
            )
            caller_ip = raw_event.get(
                "callerIpAddress", ""
            )

            # ---- EXTRACT IDENTITY ----
            identity = raw_event.get("identity", {})
            claim = identity.get("claim", {})

            accessor_name, accessor_type = (
                self._extract_identity(claim)
            )

            # ---- EXTRACT SECRET PATH ----
            properties = raw_event.get(
                "properties", {}
            )
            secret_url = properties.get("id", "")
            secret_path = self._parse_secret_path(
                secret_url
            )
            vault_name = self._extract_vault_name(
                resource_id
            )

            # ---- DETECT UNAUTHORIZED ACCESS ----
            is_unauthorized = not properties.get(
                "isAddressAuthorized", True
            )
            if is_unauthorized:
                self.unauthorized_access_attempts += 1
                logger.warning(
                    f"Unauthorized network access "
                    f"to Key Vault: "
                    f"{accessor_name} from {caller_ip}"
                )

            # ---- DETERMINE OPERATION ----
            operation = self._map_operation(
                operation_name
            )

            # ---- BULK ACCESS DETECTION ----
            is_bulk = self._detect_bulk_access(
                accessor_name, event_time
            )

            # ---- NEW PATH DETECTION ----
            is_new_path = self._is_new_path(
                accessor_name, secret_path
            )

            # ---- BUILD SECRET EVENT ----
            secret_event = IamSecretEvent(
                event_id=raw_event.get(
                    "correlationId", ""
                ),
                event_type=operation_name,
                event_time=event_time,
                accessor_name=accessor_name,
                accessor_type=accessor_type,
                secret_path=secret_path,
                secret_mount=vault_name,
                operation=operation,
                is_root_token=False,
                is_bulk_access=is_bulk,
                secrets_accessed_count=(
                    self._get_access_count(
                        accessor_name
                    )
                ),
                is_new_secret_path=is_new_path
            )

            # ---- RISK SCORING ----
            risk_score, risk_reasons = (
                self._calculate_risk(
                    secret_event,
                    operation_name,
                    event_time,
                    is_unauthorized,
                    result_type,
                    accessor_type
                )
            )

            secret_event.risk_score = risk_score
            secret_event.risk_reasons = risk_reasons

            # ---- UPDATE HISTORY ----
            self._update_history(
                accessor_name,
                secret_path,
                event_time
            )

            # ---- BUILD IAM EVENT ----
            iam_event = IamEvent(
                event_type="secret",
                source_system="azure_key_vault",
                timestamp=event_time,
                host=caller_ip,
                user=accessor_name,
                secret_event=secret_event,
                overall_risk_score=risk_score,
                overall_risk_label=(
                    self._score_to_label(risk_score)
                ),
                risk_reasons=risk_reasons
            )

            self.events_processed += 1
            if risk_score >= 0.7:
                self.high_risk_events += 1

            logger.info(
                f"Azure KV event normalized: "
                f"{operation_name} "
                f"path={secret_path} "
                f"accessor={accessor_name} "
                f"risk={risk_score:.2f}"
            )

            return iam_event

        except Exception as e:
            logger.error(
                f"Azure normalization failed: {e}"
            )
            return None

    # ============================================================
    # FIELD EXTRACTORS
    # ============================================================

    def _extract_identity(
        self,
        claim: dict
    ) -> tuple:
        """
        Extract accessor name and type from
        Azure identity claim.

        Azure identity types:
        - upn present     → human user
        - appid + no upn  → service principal
        - oid only        → managed identity
        """
        upn = claim.get("upn", "")
        app_id = claim.get("appid", "")
        oid = claim.get("oid", "")

        if upn:
            return upn, "human"
        elif app_id:
            app_name = claim.get(
                "app_displayname",
                f"app_{app_id[:8]}"
            )
            return app_name, "service_account"
        elif oid:
            return f"managed_identity_{oid[:8]}", (
                "service_account"
            )

        return "unknown", "unknown"

    def _parse_secret_path(
        self,
        secret_url: str
    ) -> str:
        """
        Parse secret path from Azure URL format.

        Azure secret URLs:
        https://vault-name.vault.azure.net/
        secrets/secret-name/version-id

        We extract: "secrets/secret-name"
        Removing the vault hostname and version.
        """
        if not secret_url:
            return ""

        try:
            # Remove https://vault.azure.net prefix
            if ".vault.azure.net/" in secret_url:
                path = secret_url.split(
                    ".vault.azure.net/"
                )[1]
                # Remove version suffix
                parts = path.split("/")
                if len(parts) >= 2:
                    return f"{parts[0]}/{parts[1]}"
                return path

            return secret_url

        except Exception:
            return secret_url

    def _extract_vault_name(
        self,
        resource_id: str
    ) -> str:
        """
        Extract vault name from Azure resource ID.

        Resource ID format:
        /subscriptions/{sub}/resourceGroups/{rg}/
        providers/Microsoft.KeyVault/vaults/{name}
        """
        if not resource_id:
            return ""

        try:
            parts = resource_id.split("/")
            if "vaults" in parts:
                vault_idx = parts.index("vaults")
                if vault_idx + 1 < len(parts):
                    return parts[vault_idx + 1]
        except Exception:
            pass

        return ""

    def _map_operation(
        self,
        operation_name: str
    ) -> str:
        """Map Azure operation name to type"""
        if operation_name in READ_OPERATIONS:
            return "read"
        elif operation_name in WRITE_OPERATIONS:
            return "write"
        elif operation_name in DELETE_OPERATIONS:
            return "delete"
        return "unknown"

    # ============================================================
    # BULK ACCESS DETECTION
    # ============================================================

    def _detect_bulk_access(
        self,
        accessor_name: str,
        event_time: str
    ) -> bool:
        """Detect bulk access with adaptive threshold"""
        if accessor_name not in self._access_history:
            return False

        recent_count = len(
            self._access_history.get(
                accessor_name, []
            )[-20:]
        )

        threshold = self._get_threshold(event_time)

        if recent_count >= threshold:
            self.bulk_access_detected += 1
            logger.warning(
                f"Azure KV bulk access: "
                f"{accessor_name} "
                f"count={recent_count} "
                f"threshold={threshold}"
            )
            return True

        return False

    def _get_threshold(
        self,
        event_time: str
    ) -> int:
        """Get adaptive threshold by time of day"""
        try:
            if "T" in event_time:
                hour = int(
                    event_time.split("T")[1][:2]
                )
                if BUSINESS_HOURS_START <= hour < BUSINESS_HOURS_END:
                    return BULK_THRESHOLD_BUSINESS
                return BULK_THRESHOLD_OFFHOURS
        except Exception:
            pass
        return BULK_THRESHOLD_BUSINESS

    def _is_new_path(
        self,
        accessor_name: str,
        secret_path: str
    ) -> bool:
        """Check if this is a new secret path"""
        if not secret_path:
            return False
        known = self._known_paths.get(
            accessor_name, set()
        )
        return secret_path not in known

    def _get_access_count(
        self,
        accessor_name: str
    ) -> int:
        """Get total access count"""
        return len(
            self._access_history.get(
                accessor_name, []
            )
        )

    # ============================================================
    # RISK SCORING
    # ============================================================

    def _calculate_risk(
        self,
        secret_event: IamSecretEvent,
        operation_name: str,
        event_time: str,
        is_unauthorized: bool,
        result_type: str,
        accessor_type: str
    ) -> tuple:
        """Calculate risk score for Azure KV event"""
        score = 0.0
        reasons = []

        # Unauthorized network access
        if is_unauthorized:
            score += 0.5
            reasons.append(
                "Access from unauthorized network "
                "to Azure Key Vault"
            )

        # High risk operation
        if operation_name in HIGH_RISK_OPERATIONS:
            score += 0.3
            reasons.append(
                f"High risk operation: {operation_name}"
            )

        # Bulk access
        if secret_event.is_bulk_access:
            score += 0.4
            reasons.append(
                f"Bulk secret access: "
                f"{secret_event.secrets_accessed_count}"
                f" operations"
            )

        # Secret deletion
        if operation_name in DELETE_OPERATIONS:
            score += 0.4
            reasons.append(
                f"Secret deleted: "
                f"{secret_event.secret_path}"
            )

        # New path access
        if secret_event.is_new_secret_path:
            score += 0.2
            reasons.append(
                f"New secret accessed: "
                f"{secret_event.secret_path}"
            )

        # Off hours
        try:
            if "T" in event_time:
                hour = int(
                    event_time.split("T")[1][:2]
                )
                if not (
                    BUSINESS_HOURS_START <= hour < BUSINESS_HOURS_END
                ):
                    score += 0.2
                    reasons.append(
                        "Secret accessed outside "
                        "business hours"
                    )
        except Exception:
            pass

        return min(score, 1.0), reasons

    # ============================================================
    # HISTORY MANAGEMENT
    # ============================================================

    def _update_history(
        self,
        accessor_name: str,
        secret_path: str,
        event_time: str
    ) -> None:
        """Update access history"""
        if accessor_name not in self._access_history:
            self._access_history[accessor_name] = []

        self._access_history[accessor_name].append(
            event_time
        )
        self._access_history[accessor_name] = (
            self._access_history[accessor_name][-100:]
        )

        if accessor_name not in self._known_paths:
            self._known_paths[accessor_name] = set()
        if secret_path:
            self._known_paths[accessor_name].add(
                secret_path
            )

    def _score_to_label(self, score: float) -> str:
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score > 0.0:
            return "LOW"
        return "UNKNOWN"

    def get_statistics(self) -> dict:
        return {
            "events_processed": self.events_processed,
            "bulk_access_detected": (
                self.bulk_access_detected
            ),
            "high_risk_events": self.high_risk_events,
            "unauthorized_attempts": (
                self.unauthorized_access_attempts
            ),
            "accessors_tracked": len(
                self._access_history
            )
        }