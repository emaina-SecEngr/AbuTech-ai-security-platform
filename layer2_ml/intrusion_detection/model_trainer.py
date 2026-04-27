"""
Layer 2 — ML Processing Engine
Network Intrusion Detection — Model Trainer

This module trains and compares two models:
    1. Random Forest  — fast, interpretable baseline
    2. XGBoost        — higher accuracy, sequential learning

Both models are tracked with MLflow so you can:
    - Compare performance across runs
    - Reproduce any previous training run exactly
    - Deploy the best performing model
    - Track how models improve over time

MLflow Experiment Structure:
    Experiment: "network_intrusion_detection"
    Run 1: Random Forest baseline
    Run 2: XGBoost optimized
    Run 3: Random Forest with more trees
    ...compare all runs in MLflow UI

This is your MLOps experiment tracking in practice.
Every training decision is recorded and reproducible.
"""

import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ============================================================
# MODEL CONFIGURATION
#
# These are the default hyperparameters for each model.
# MLflow tracks every run so you can experiment with
# different values and compare results side by side.
#
# Security-specific tuning decisions documented inline.
# ============================================================

RANDOM_FOREST_CONFIG = {
    # Number of trees
    # More trees = better accuracy but slower training
    # 100 is a good starting point for security data
    "n_estimators": 100,

    # Maximum tree depth
    # None = unlimited depth (can overfit)
    # 10-20 is good for security data
    "max_depth": 15,

    # CRITICAL for imbalanced security data
    # Automatically adjusts weights so malicious samples
    # count proportionally more than benign samples
    # Without this the model ignores the minority class
    "class_weight": "balanced",

    # Minimum samples required to split a node
    # Higher values prevent overfitting
    "min_samples_split": 10,

    # Use all CPU cores for parallel tree building
    "n_jobs": -1,

    # For reproducibility
    "random_state": 42
}

XGBOOST_CONFIG = {
    # Number of boosting rounds
    # XGBoost builds trees sequentially
    # More rounds = more correction iterations
    "n_estimators": 200,

    # Maximum tree depth per round
    # Lower than Random Forest because XGBoost
    # uses many shallow trees rather than few deep ones
    "max_depth": 6,

    # Learning rate — how much each tree contributes
    # Lower = more conservative learning
    # Lower rate needs more estimators
    "learning_rate": 0.1,

    # Subsample ratio of training data per tree
    # Prevents overfitting through randomization
    "subsample": 0.8,

    # Subsample ratio of features per tree
    "colsample_bytree": 0.8,

    # CRITICAL for class imbalance
    # Set to ratio of negative/positive samples
    # e.g. 1000 benign / 10 attack = scale_pos_weight=100
    # We calculate this dynamically from training data
    "scale_pos_weight": None,  # Set dynamically

    # Use all CPU cores
    "n_jobs": -1,

    # Binary classification objective
    "objective": "binary:logistic",

    # Evaluation metric
    "eval_metric": "auc",

    "random_state": 42,

    # Suppress XGBoost verbose output
    "verbosity": 0
}


class IntrusionDetectionTrainer:
    """
    Trains network intrusion detection models with
    full MLflow experiment tracking.

    Usage:
        trainer = IntrusionDetectionTrainer()
        rf_model = trainer.train_random_forest(
            X_train, y_train, X_test, y_test
        )
        xgb_model = trainer.train_xgboost(
            X_train, y_train, X_test, y_test
        )
        best = trainer.get_best_model()
    """

    def __init__(
        self,
        experiment_name: str = "network_intrusion_detection",
        model_save_dir: str = "models/intrusion_detection"
    ):
        """
        Initialize the trainer.

        Args:
            experiment_name: MLflow experiment name
                            All runs grouped under this name
            model_save_dir: Where to save trained models
        """
        self.experiment_name = experiment_name
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # Track best model across all runs
        self.best_model = None
        self.best_f1_score = 0.0
        self.best_model_name = ""

        # Setup MLflow
        self._setup_mlflow()

    def _setup_mlflow(self):
        """
        Configure MLflow experiment.

        MLflow creates a local tracking server by default.
        All runs stored in ./mlruns directory.
        View in browser: mlflow ui
        """
        mlflow.set_experiment(self.experiment_name)
        logger.info(
            f"MLflow experiment: {self.experiment_name}"
        )

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list = None,
        config: dict = None
    ) -> RandomForestClassifier:
        """
        Train Random Forest with MLflow tracking.

        Random Forest is your baseline model.
        It is fast, reliable, and highly interpretable.
        Feature importance scores show analysts exactly
        which network features drove each detection.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_names: Names of features for logging
            config: Override default hyperparameters

        Returns:
            Trained RandomForestClassifier
        """
        model_config = config or RANDOM_FOREST_CONFIG

        logger.info("Training Random Forest model")
        logger.info(f"Config: {model_config}")

        with mlflow.start_run(run_name="random_forest"):

            # Log all hyperparameters
            # Every parameter is recorded for reproducibility
            mlflow.log_params(model_config)
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param(
                "training_samples", len(y_train)
            )
            mlflow.log_param(
                "test_samples", len(y_test)
            )
            mlflow.log_param(
                "attack_ratio_train",
                round(np.mean(y_train), 4)
            )

            # Train the model
            model = RandomForestClassifier(**model_config)
            model.fit(X_train, y_train)

            # Evaluate and log metrics
            metrics = self._evaluate_model(
                model, X_test, y_test,
                model_name="Random Forest"
            )
            mlflow.log_metrics(metrics)

            # Log feature importance
            if feature_names:
                self._log_feature_importance(
                    model.feature_importances_,
                    feature_names
                )

            # Save model to MLflow
            mlflow.sklearn.log_model(
                model,
                "random_forest_model"
            )

            # Track best model
            if metrics["f1_score"] > self.best_f1_score:
                self.best_f1_score = metrics["f1_score"]
                self.best_model = model
                self.best_model_name = "Random Forest"

            logger.info(
                f"Random Forest F1: {metrics['f1_score']:.4f}"
            )

        return model

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list = None,
        config: dict = None
    ) -> XGBClassifier:
        """
        Train XGBoost with dynamic class weight and
        MLflow tracking.

        XGBoost is your high-accuracy model.
        It learns sequentially — each tree corrects
        mistakes of the previous ones.

        Key advantage over Random Forest:
        scale_pos_weight handles class imbalance
        more aggressively than class_weight="balanced"
        in Random Forest.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_names: Names of features for logging
            config: Override default hyperparameters

        Returns:
            Trained XGBClassifier
        """
        model_config = dict(config or XGBOOST_CONFIG)

        # Calculate scale_pos_weight dynamically
        # This is the ratio of benign to attack samples
        # Tells XGBoost each attack sample counts this
        # many times more than a benign sample
        benign_count = np.sum(y_train == 0)
        attack_count = np.sum(y_train == 1)
        scale_pos_weight = benign_count / max(attack_count, 1)
        model_config["scale_pos_weight"] = scale_pos_weight

        logger.info("Training XGBoost model")
        logger.info(
            f"scale_pos_weight: {scale_pos_weight:.2f}"
        )

        with mlflow.start_run(run_name="xgboost"):

            # Log all hyperparameters
            mlflow.log_params(model_config)
            mlflow.log_param("model_type", "XGBoost")
            mlflow.log_param(
                "training_samples", len(y_train)
            )
            mlflow.log_param(
                "test_samples", len(y_test)
            )
            mlflow.log_param(
                "scale_pos_weight",
                round(scale_pos_weight, 2)
            )

            # Train the model
            model = XGBClassifier(**model_config)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

            # Evaluate and log metrics
            metrics = self._evaluate_model(
                model, X_test, y_test,
                model_name="XGBoost"
            )
            mlflow.log_metrics(metrics)

            # Log feature importance
            if feature_names:
                self._log_feature_importance(
                    model.feature_importances_,
                    feature_names
                )

            # Save model to MLflow
            mlflow.xgboost.log_model(
                model,
                "xgboost_model"
            )

            # Track best model
            if metrics["f1_score"] > self.best_f1_score:
                self.best_f1_score = metrics["f1_score"]
                self.best_model = model
                self.best_model_name = "XGBoost"

            logger.info(
                f"XGBoost F1: {metrics['f1_score']:.4f}"
            )

        return model

    def _evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> dict:
        """
        Evaluate model using security-appropriate metrics.

        WHY THESE METRICS:

        F1 Score - harmonic mean of precision and recall
            Balances catching attacks and avoiding false alarms
            Primary metric for security ML models

        Recall (Detection Rate) - of all real attacks
            what fraction did we catch?
            CRITICAL in security - missed attacks = breaches
            Target: > 95%

        Precision - of everything we flagged as attack
            what fraction was actually an attack?
            Important for analyst workload
            Low precision = SOC fatigue from false positives

        ROC-AUC - model discrimination ability
            How well does the model separate attack from benign
            across all possible thresholds?
            Useful for comparing models independent of threshold

        NOT USING ACCURACY because:
            With 99% benign traffic a model that predicts
            everything as benign gets 99% accuracy
            while missing every single attack.
            Accuracy is meaningless for imbalanced security data.
        """
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate all metrics
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        auc = roc_auc_score(y_test, y_prob)

        # False positive rate
        # How many benign flows did we incorrectly flag?
        tn = np.sum((y_test == 0) & (y_pred == 0))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fpr = fp / max((tn + fp), 1)

        # False negative rate
        # How many attacks did we miss?
        fn = np.sum((y_test == 1) & (y_pred == 0))
        tp = np.sum((y_test == 1) & (y_pred == 1))
        fnr = fn / max((tp + fn), 1)

        metrics = {
            "f1_score": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "roc_auc": round(auc, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn)
        }

        # Log detailed report
        logger.info(f"\n{model_name} Evaluation:")
        logger.info(f"  F1 Score:    {f1:.4f}")
        logger.info(f"  Recall:      {recall:.4f} "
                    f"(Detection Rate)")
        logger.info(f"  Precision:   {precision:.4f}")
        logger.info(f"  ROC-AUC:     {auc:.4f}")
        logger.info(f"  FPR:         {fpr:.4f} "
                    f"(False Alarm Rate)")
        logger.info(f"  FNR:         {fnr:.4f} "
                    f"(Miss Rate)")

        # Security assessment
        if recall < 0.90:
            logger.warning(
                f"  WARNING: Detection rate {recall:.1%} "
                f"below 90% threshold. "
                f"Model missing too many attacks."
            )

        if fpr > 0.05:
            logger.warning(
                f"  WARNING: False positive rate {fpr:.1%} "
                f"above 5% threshold. "
                f"Model generating too many false alarms."
            )

        if recall >= 0.95 and fpr <= 0.01:
            logger.info(
                f"  EXCELLENT: Model meets production "
                f"deployment criteria."
            )

        return metrics

    def _log_feature_importance(
        self,
        importances: np.ndarray,
        feature_names: list
    ) -> None:
        """
        Log feature importance scores to MLflow.

        Feature importance tells you which network
        features most influence attack detection.

        In security this is critical for:
        1. Analyst trust - they can see why the model
           flagged something
        2. Feature validation - security domain knowledge
           should match what the model found important
        3. Adversarial robustness - if attackers know
           your top features they can try to evade them
        """
        # Sort features by importance
        feature_importance_pairs = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )

        # Log top 10 most important features
        logger.info("Top 10 Most Important Features:")
        for i, (name, importance) in enumerate(
            feature_importance_pairs[:10]
        ):
            logger.info(f"  {i+1:2d}. {name}: {importance:.4f}")
            mlflow.log_metric(
                f"feature_importance_{i+1}",
                round(importance, 4)
            )

    def get_best_model(self):
        """
        Return the best performing model across all runs.

        Called after training both models to get
        the one to deploy to production.
        """
        if self.best_model is None:
            raise ValueError(
                "No models trained yet. "
                "Call train_random_forest() or "
                "train_xgboost() first."
            )

        logger.info(
            f"Best model: {self.best_model_name} "
            f"with F1={self.best_f1_score:.4f}"
        )

        return self.best_model, self.best_model_name

    def compare_models_summary(self) -> str:
        """
        Generate a summary comparing all trained models.

        Used in Layer 5 dashboard and client reports.
        """
        return (
            f"Model Comparison Summary\n"
            f"Experiment: {self.experiment_name}\n"
            f"Best Model: {self.best_model_name}\n"
            f"Best F1 Score: {self.best_f1_score:.4f}\n"
            f"View full comparison: mlflow ui"
        )