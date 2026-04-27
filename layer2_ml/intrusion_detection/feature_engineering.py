"""
Layer 2 — ML Processing Engine
Network Intrusion Detection — Feature Engineering

This module creates new features from existing CICIDS2017
features to give the model stronger detection signals.

Why Feature Engineering Matters:
    Raw network flow features capture what happened.
    Engineered features capture what it MEANS.

    Example:
    Raw:        Total Fwd Packets = 1000
                Flow Duration = 0.001 seconds
    Engineered: Packets per second = 1,000,000
                This immediately signals a DoS attack

    The model could theoretically learn to divide
    these features itself but engineered features
    make the pattern explicit and speed up learning.

Security Domain Knowledge Applied:
    Every engineered feature in this module is
    grounded in how specific attack types behave.
    This is where your CTI knowledge becomes
    a direct advantage over generic data scientists.
"""

import logging
import numpy as np
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)


class NetworkFlowFeatureEngineer:
    """
    Engineers new features from raw CICIDS2017 network flows.

    Each feature group targets a specific attack category:
    - Rate features     → DoS and DDoS detection
    - Ratio features    → Scanning and reconnaissance
    - Entropy features  → Encrypted C2 traffic
    - Behavioral flags  → Known attack signatures
    """

    def __init__(self):
        self.engineered_feature_names = []

    def engineer_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.

        Args:
            df: DataFrame with raw CICIDS2017 features

        Returns:
            DataFrame with additional engineered features
        """
        logger.info("Starting feature engineering")
        original_cols = len(df.columns)

        # Apply each feature group
        df = self._engineer_rate_features(df)
        df = self._engineer_ratio_features(df)
        df = self._engineer_behavioral_flags(df)
        df = self._engineer_statistical_features(df)

        new_cols = len(df.columns) - original_cols
        logger.info(
            f"Engineered {new_cols} new features. "
            f"Total features: {len(df.columns)}"
        )

        return df

    def _engineer_rate_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Rate-based features for DoS and DDoS detection.

        DoS attacks generate traffic at rates that are
        physically impossible for legitimate users.
        A single connection at 1 million packets per second
        is always an attack.

        These features make DoS detection trivially easy
        for your Random Forest model.

        ATT&CK Reference: T1498 Network Denial of Service
                          T1499 Endpoint Denial of Service
        """

        duration_col = " Flow Duration"
        fwd_packets_col = " Total Fwd Packets"
        bwd_packets_col = " Total Backward Packets"
        fwd_bytes_col = " Total Length of Fwd Packets"
        bwd_bytes_col = " Total Length of Bwd Packets"

        # Avoid division by zero
        # Add small epsilon to duration
        epsilon = 1e-10

        if duration_col in df.columns:
            duration = df[duration_col] + epsilon

            # Packets per second
            # DoS attacks have extremely high pps
            if fwd_packets_col in df.columns:
                df["engineered_fwd_pps"] = (
                    df[fwd_packets_col] / duration
                )

            if bwd_packets_col in df.columns:
                df["engineered_bwd_pps"] = (
                    df[bwd_packets_col] / duration
                )

            # Bytes per second
            # Data exfiltration has high outbound bps
            if fwd_bytes_col in df.columns:
                df["engineered_fwd_bps"] = (
                    df[fwd_bytes_col] / duration
                )

            if bwd_bytes_col in df.columns:
                df["engineered_bwd_bps"] = (
                    df[bwd_bytes_col] / duration
                )

        self.engineered_feature_names.extend([
            "engineered_fwd_pps",
            "engineered_bwd_pps",
            "engineered_fwd_bps",
            "engineered_bwd_bps"
        ])

        return df

    def _engineer_ratio_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Ratio features for scanning and reconnaissance.

        Network scanning has a distinctive asymmetry:
        the scanner sends many small packets but receives
        very few responses because most ports are closed.

        Forward/backward packet ratio:
            Normal traffic: roughly balanced (1:1 to 3:1)
            Port scan:      very high (many sent, few received)
            DoS:            extremely high (attacker sends only)

        ATT&CK Reference: T1046 Network Service Scanning
                          T1595 Active Scanning
        """

        fwd_packets_col = " Total Fwd Packets"
        bwd_packets_col = " Total Backward Packets"
        fwd_bytes_col = " Total Length of Fwd Packets"
        bwd_bytes_col = " Total Length of Bwd Packets"

        epsilon = 1e-10

        # Packet ratio
        if (fwd_packets_col in df.columns and
                bwd_packets_col in df.columns):
            df["engineered_packet_ratio"] = (
                df[fwd_packets_col] /
                (df[bwd_packets_col] + epsilon)
            )

        # Byte ratio
        if (fwd_bytes_col in df.columns and
                bwd_bytes_col in df.columns):
            df["engineered_byte_ratio"] = (
                df[fwd_bytes_col] /
                (df[bwd_bytes_col] + epsilon)
            )

        # Total packets
        # Helps distinguish scan patterns from normal
        if (fwd_packets_col in df.columns and
                bwd_packets_col in df.columns):
            df["engineered_total_packets"] = (
                df[fwd_packets_col] + df[bwd_packets_col]
            )

        self.engineered_feature_names.extend([
            "engineered_packet_ratio",
            "engineered_byte_ratio",
            "engineered_total_packets"
        ])

        return df

    def _engineer_behavioral_flags(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Binary behavioral flags encoding known attack signatures.

        These flags directly encode security domain knowledge
        as binary features the model can use immediately.

        Instead of the model learning that SYN_flag_count > 0
        and RST_flag_count > 0 together is suspicious,
        we explicitly create an is_scan_like feature.

        This is where YOUR security knowledge becomes a
        competitive advantage — you know which combinations
        of TCP flags indicate which attack types.

        ATT&CK Reference: T1046 Network Service Scanning
                          T1498 Network DoS
                          T1071 Application Layer Protocol
        """

        syn_col = " SYN Flag Count"
        rst_col = " RST Flag Count"
        ack_col = " ACK Flag Count"
        psh_col = " PSH Flag Count"
        fwd_packets_col = " Total Fwd Packets"
        bwd_packets_col = " Total Backward Packets"
        duration_col = " Flow Duration"

        # Scan-like behavior
        # High SYN count with high RST count
        # Many connection attempts that were refused
        if (syn_col in df.columns and
                rst_col in df.columns):
            df["engineered_is_scan_like"] = (
                (df[syn_col] > 0) &
                (df[rst_col] > 0)
            ).astype(int)

        # One-directional flow
        # Attacker sends but never receives response
        # Strong DoS and scanning indicator
        if (fwd_packets_col in df.columns and
                bwd_packets_col in df.columns):
            df["engineered_is_one_directional"] = (
                df[bwd_packets_col] == 0
            ).astype(int)

        # Very short duration with many packets
        # Characteristic of DoS flooding attacks
        if (duration_col in df.columns and
                fwd_packets_col in df.columns):
            df["engineered_is_flood_like"] = (
                (df[duration_col] < 1000) &
                (df[fwd_packets_col] > 100)
            ).astype(int)

        # Data push without acknowledgment
        # Can indicate protocol abuse
        if (psh_col in df.columns and
                ack_col in df.columns):
            df["engineered_psh_without_ack"] = (
                (df[psh_col] > 0) &
                (df[ack_col] == 0)
            ).astype(int)

        self.engineered_feature_names.extend([
            "engineered_is_scan_like",
            "engineered_is_one_directional",
            "engineered_is_flood_like",
            "engineered_psh_without_ack"
        ])

        return df

    def _engineer_statistical_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Statistical features for C2 beaconing detection.

        Command and Control beaconing has very regular
        timing patterns. A compromised host checking in
        with its C2 server every 90 seconds produces
        a very low standard deviation in inter-arrival times.

        Normal user traffic has irregular timing because
        humans don't interact with computers at perfectly
        regular intervals.

        Low IAT standard deviation = potential beaconing
        High IAT standard deviation = normal user behavior

        ATT&CK Reference: T1071.001 Web Protocols C2
                          T1573 Encrypted Channel
        """

        iat_mean_col = " Flow IAT Mean"
        iat_std_col = " Flow IAT Std"
        fwd_iat_col = " Fwd IAT Mean"
        bwd_iat_col = " Bwd IAT Mean"

        epsilon = 1e-10

        # Coefficient of variation for IAT
        # Low value = regular beaconing pattern
        # This is more informative than raw std alone
        if (iat_mean_col in df.columns and
                iat_std_col in df.columns):
            df["engineered_iat_cv"] = (
                df[iat_std_col] /
                (df[iat_mean_col].abs() + epsilon)
            )

        # Forward/backward IAT ratio
        # Asymmetric timing can indicate C2 patterns
        if (fwd_iat_col in df.columns and
                bwd_iat_col in df.columns):
            df["engineered_iat_asymmetry"] = (
                df[fwd_iat_col] /
                (df[bwd_iat_col] + epsilon)
            )

        self.engineered_feature_names.extend([
            "engineered_iat_cv",
            "engineered_iat_asymmetry"
        ])

        return df

    def get_engineered_feature_names(self) -> List[str]:
        """Return names of all engineered features"""
        return self.engineered_feature_names

    def get_feature_importance_context(self) -> dict:
        """
        Return security context for each engineered feature.

        Used by SHAP explainability layer to provide
        analyst-friendly explanations.

        When SHAP says engineered_is_scan_like=1 contributed
        most to a detection, the analyst sees:
        'High SYN+RST flag combination indicating port scan'
        """
        return {
            "engineered_fwd_pps": (
                "Forward packets per second - "
                "high values indicate DoS attack"
            ),
            "engineered_bwd_pps": (
                "Backward packets per second - "
                "asymmetry indicates one-way attack"
            ),
            "engineered_packet_ratio": (
                "Forward/backward packet ratio - "
                "high values indicate scanning"
            ),
            "engineered_is_scan_like": (
                "SYN+RST flag pattern - "
                "indicates port scanning behavior"
            ),
            "engineered_is_one_directional": (
                "No return traffic - "
                "indicates DoS or blind scanning"
            ),
            "engineered_is_flood_like": (
                "Short duration high packet count - "
                "indicates flooding attack"
            ),
            "engineered_iat_cv": (
                "Regular timing pattern - "
                "low values indicate C2 beaconing"
            )
        }