"""
Layer 2 — ML Processing Engine
Network Intrusion Detection — Data Preparation

This module handles all data preparation for the
network intrusion detection model.

It loads CICIDS2017 CSV files, cleans the data,
handles class imbalance, and produces train/test
splits ready for model training.

CICIDS2017 Dataset:
    Canadian Institute for Cybersecurity
    Intrusion Detection System 2017
    Contains labeled network flows across 5 attack types:
    - DoS, DDoS, PortScan, Brute Force, Web Attacks,
      Botnet, Infiltration

Key Challenges:
    1. Severely imbalanced classes
       Benign traffic vastly outnumbers attacks
    2. Missing and infinite values
       Flow calculations can produce inf/nan values
    3. Temporal integrity
       Must split by time not randomly
    4. Feature selection
       78 features available, not all are useful
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


# ============================================================
# FEATURE CONFIGURATION
#
# These 20 features were selected based on:
# 1. Research literature on CICIDS2017
# 2. Security domain knowledge
# 3. Availability across all CSV files
#
# We do not use all 78 features because:
# - Some are identifiers (IP, port) that cause data leakage
# - Some have very low variance and add noise
# - Fewer features = faster training and inference
# ============================================================

SELECTED_FEATURES = [ # Flow duration
    " Flow Duration",

    # Packet counts
    " Total Fwd Packets",
    " Total Backward Packets",

    # Byte counts
    "Total Length of Fwd Packets",
    " Total Length of Bwd Packets",

    # Packet size statistics
    " Fwd Packet Length Max",
    " Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    " Bwd Packet Length Mean",

    # Flow rates
    "Flow Bytes/s",
    " Flow Packets/s",

    # Inter-arrival times
    " Flow IAT Mean",
    " Flow IAT Std",
    " Fwd IAT Mean",
    " Bwd IAT Mean",

    # TCP flags
    " SYN Flag Count",
    " RST Flag Count",
    " PSH Flag Count",
    " ACK Flag Count",

    # Window size
    "Init_Win_bytes_forward",
]

# Label column name in CICIDS2017
LABEL_COLUMN = " Label"

# Binary classification mapping
# We start with binary (attack vs benign)
# Multi-class comes later
BINARY_LABEL_MAP = {
    "BENIGN": 0,
    # Everything else maps to 1 (attack)
}


class CICIDSDataPreparation:
    """
    Handles all data preparation for CICIDS2017 dataset.

    Usage:
        prep = CICIDSDataPreparation(data_dir="data/cicids2017")
        X_train, X_test, y_train, y_test = prep.prepare()
    """

    def __init__(
        self,
        data_dir: str,
        test_size: float = 0.2,
        random_state: int = 42,
        apply_smote: bool = True,
        max_samples: int = None
    ):
        """
        Initialize data preparation.

        Args:
            data_dir: Path to CICIDS2017 CSV files
            test_size: Fraction of data for testing (0.2 = 20%)
            random_state: Seed for reproducibility
            apply_smote: Whether to apply SMOTE oversampling
            max_samples: Limit samples for faster development
                        Set to None for full dataset
        """
        self.data_dir = Path(data_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.apply_smote = apply_smote
        self.max_samples = max_samples
        self.scaler = StandardScaler()

        # Statistics tracked during preparation
        self.stats = {}

    def prepare(
        self,
        csv_files: list = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full data preparation pipeline.

        Runs all preparation steps in the correct order:
        1. Load CSV files
        2. Clean data
        3. Engineer binary labels
        4. Select features
        5. Handle class imbalance
        6. Split train/test
        7. Scale features

        Args:
            csv_files: List of CSV filenames to load
                      If None loads all CSVs in data_dir

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            All as numpy arrays ready for model training
        """
        logger.info("Starting CICIDS2017 data preparation")

        # Step 1: Load data
        df = self._load_csv_files(csv_files)
        logger.info(f"Loaded {len(df)} total samples")

        # Step 2: Clean data
        df = self._clean_data(df)
        logger.info(f"After cleaning: {len(df)} samples")

        # Step 3: Create binary labels
        df = self._create_binary_labels(df)

        # Step 4: Log class distribution
        self._log_class_distribution(df)

        # Step 5: Select features
        X, y = self._select_features(df)

        # Step 6: Optional sampling for development
        if self.max_samples:
            X, y = self._sample_data(X, y)
            logger.info(f"Sampled to {len(y)} samples")

        # Step 7: Split BEFORE oversampling
        # Critical: never apply SMOTE before splitting
        # Applying SMOTE before split causes data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # Maintain class ratio in both splits
        )

        logger.info(
            f"Split: {len(y_train)} train, {len(y_test)} test"
        )

        # Step 8: Apply SMOTE to training data ONLY
        # Never apply SMOTE to test data
        # Test data must reflect real class distribution
        if self.apply_smote:
            X_train, y_train = self._apply_smote(
                X_train, y_train
            )

        # Step 9: Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        logger.info("Data preparation complete")
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def _load_csv_files(
        self,
        csv_files: list = None
    ) -> pd.DataFrame:
        """
        Load one or more CICIDS2017 CSV files.

        CICIDS2017 has known encoding issues.
        Some files use latin-1 encoding not UTF-8.
        We handle this automatically.

        Args:
            csv_files: Specific filenames to load
                      If None loads all CSVs in directory

        Returns:
            Combined DataFrame of all loaded files
        """
        if csv_files is None:
            # Load all CSV files in directory
            csv_paths = list(self.data_dir.glob("*.csv"))
            if not csv_paths:
                # Also try without extension
                csv_paths = list(self.data_dir.glob("*ISCX*"))
        else:
            csv_paths = [self.data_dir / f for f in csv_files]

        if not csv_paths:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir}"
            )

        logger.info(f"Loading {len(csv_paths)} CSV files")

        dfs = []
        for path in csv_paths:
            logger.info(f"Loading {path.name}")
            try:
                df = pd.read_csv(
                    path,
                    encoding="latin-1",
                    low_memory=False
                )
                dfs.append(df)
                logger.info(
                    f"  Loaded {len(df)} rows from {path.name}"
                )
            except Exception as e:
                logger.error(f"Failed to load {path.name}: {e}")

        if not dfs:
            raise ValueError("No files were loaded successfully")

        combined = pd.concat(dfs, ignore_index=True)
        self.stats["total_loaded"] = len(combined)
        return combined

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw CICIDS2017 data.

        CICIDS2017 has several known data quality issues:

        1. Infinite values
           Flow rate calculations can produce infinity
           when flow duration is zero. These must be
           removed as ML models cannot handle infinity.

        2. NaN values
           Some features have missing values.
           We drop rows with any NaN in selected features.

        3. Duplicate rows
           Some CSV files have duplicate entries.
           We remove exact duplicates.

        4. Whitespace in column names
           CICIDS2017 column names have leading spaces.
           We keep them as-is since SELECTED_FEATURES
           already accounts for this.

        Args:
            df: Raw loaded DataFrame

        Returns:
            Cleaned DataFrame
        """
        initial_count = len(df)

        # Remove infinite values
        # Replace inf with NaN first then drop
        df = df.replace([np.inf, -np.inf], np.nan)

        # Drop rows with NaN in our selected features
        # Only drop if the column exists in this file
        available_features = [
            f for f in SELECTED_FEATURES
            if f in df.columns
        ]

        df = df.dropna(subset=available_features)

        # Remove exact duplicates
        df = df.drop_duplicates()

        removed = initial_count - len(df)
        self.stats["rows_removed_cleaning"] = removed
        logger.info(
            f"Cleaning removed {removed} rows "
            f"({removed/initial_count*100:.1f}%)"
        )

        return df

    def _create_binary_labels(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert CICIDS2017 labels to binary classification.

        CICIDS2017 has many specific attack labels:
            BENIGN
            DDoS
            DoS Hulk
            DoS GoldenEye
            PortScan
            Bot
            FTP-Patator
            SSH-Patator
            Web Attack - Brute Force
            etc.

        For our first model we use binary classification:
            BENIGN = 0
            Any attack = 1

        This simplifies the problem and gives us a strong
        baseline. We add multi-class classification later.

        Why binary first:
        - Easier to evaluate and interpret
        - Gives you a strong working baseline quickly
        - Binary models are faster to train
        - You understand what is working before adding complexity

        Args:
            df: DataFrame with original Label column

        Returns:
            DataFrame with added binary_label column
        """
        # Strip whitespace from labels
        df[LABEL_COLUMN] = df[LABEL_COLUMN].str.strip()

        # Create binary label
        # 0 = BENIGN, 1 = any attack
        df["binary_label"] = df[LABEL_COLUMN].apply(
            lambda x: 0 if x == "BENIGN" else 1
        )

        return df

    def _log_class_distribution(
        self,
        df: pd.DataFrame
    ) -> None:
        """
        Log class distribution for monitoring.

        This is critical information for understanding
        the imbalance problem before applying SMOTE.
        """
        label_counts = df["binary_label"].value_counts()
        total = len(df)

        benign_count = label_counts.get(0, 0)
        attack_count = label_counts.get(1, 0)

        benign_pct = benign_count / total * 100
        attack_pct = attack_count / total * 100

        logger.info("Class Distribution:")
        logger.info(
            f"  Benign:  {benign_count:>8,} ({benign_pct:.1f}%)"
        )
        logger.info(
            f"  Attack:  {attack_count:>8,} ({attack_pct:.1f}%)"
        )
        logger.info(
            f"  Ratio:   {benign_count/max(attack_count,1):.0f}:1"
        )

        self.stats["benign_count"] = int(benign_count)
        self.stats["attack_count"] = int(attack_count)
        self.stats["imbalance_ratio"] = (
            benign_count / max(attack_count, 1)
        )

    def _select_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select features and extract label.

        Returns X (features) and y (labels) as numpy arrays.

        Only uses SELECTED_FEATURES that exist in the file.
        This handles cases where different CSV files have
        slightly different column sets.
        """
        available_features = [
            f for f in SELECTED_FEATURES
            if f in df.columns
        ]

        if len(available_features) < 10:
            raise ValueError(
                f"Too few features available: "
                f"{len(available_features)}. "
                f"Check column names in CSV file."
            )

        logger.info(
            f"Using {len(available_features)} features"
        )

        X = df[available_features].values.astype(np.float32)
        y = df["binary_label"].values.astype(np.int32)

        return X, y

    def _sample_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample data for faster development iteration.

        When max_samples is set we take a stratified
        sample that maintains the class distribution.
        This lets you iterate quickly during development
        without waiting for full dataset training.

        In production set max_samples=None to use
        the full dataset.
        """
        if self.max_samples >= len(y):
            return X, y

        # Stratified sampling maintains class ratio
        _, X_sample, _, y_sample = train_test_split(
            X, y,
            test_size=self.max_samples / len(y),
            random_state=self.random_state,
            stratify=y
        )

        return X_sample, y_sample

    def _apply_smote(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to balance training data.

        SMOTE — Synthetic Minority Oversampling Technique
        Creates synthetic attack samples by interpolating
        between existing attack samples.

        Why SMOTE instead of simple duplication:
        Simple duplication just copies existing samples.
        The model memorizes them and does not generalize.

        SMOTE creates NEW synthetic samples that are
        similar but not identical to existing ones.
        The model learns the general pattern of attacks
        not just specific examples.

        CRITICAL: Only apply to training data.
        Test data must reflect real-world distribution
        to give you honest performance metrics.

        Args:
            X_train: Training features
            y_train: Training labels (imbalanced)

        Returns:
            X_train_balanced, y_train_balanced
        """
        attack_count = np.sum(y_train == 1)
        benign_count = np.sum(y_train == 0)

        logger.info(
            f"Before SMOTE: "
            f"{benign_count} benign, {attack_count} attacks"
        )

        smote = SMOTE(
            random_state=self.random_state,
            k_neighbors=min(5, attack_count - 1)
        )

        X_balanced, y_balanced = smote.fit_resample(
            X_train, y_train
        )

        new_attack_count = np.sum(y_balanced == 1)
        logger.info(
            f"After SMOTE: "
            f"{np.sum(y_balanced==0)} benign, "
            f"{new_attack_count} attacks"
        )

        return X_balanced, y_balanced

    def get_feature_names(self) -> list:
        """
        Return list of feature names used by the model.
        Used by SHAP explainability in MLOps layer.
        """
        return SELECTED_FEATURES

    def get_statistics(self) -> dict:
        """Return preparation statistics for monitoring"""
        return self.stats