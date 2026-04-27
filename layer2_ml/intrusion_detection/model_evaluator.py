"""
Layer 2 — ML Processing Engine
Network Intrusion Detection — Model Evaluator

This module provides deep evaluation of trained models
beyond the basic metrics logged during training.

It generates:
    1. Confusion matrix visualization
    2. Feature importance charts
    3. Precision-recall curves
    4. Threshold analysis
    5. Attack-type breakdown
    6. Analyst-friendly report

Why Deep Evaluation Matters:
    A model with 95% F1 score sounds great.
    But which attacks is it missing?
    Is it missing the most dangerous ones?
    Which features drive its decisions?
    Can analysts trust its explanations?

    Deep evaluation answers these questions
    before you deploy to production.

Security Operations Context:
    Your evaluation report becomes the evidence
    that convinces a SOC manager to trust the model.
    Technical metrics alone are not enough.
    You need visualizations and explanations.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score
)

logger = logging.getLogger(__name__)


class IntrusionDetectionEvaluator:
    """
    Deep evaluation and visualization for trained
    intrusion detection models.

    Generates reports and charts suitable for:
    - SOC analyst briefings
    - Client security assessments
    - MLOps model monitoring dashboards
    - Academic portfolio demonstrations
    """

    def __init__(
        self,
        output_dir: str = "reports/intrusion_detection"
    ):
        """
        Initialize evaluator.

        Args:
            output_dir: Where to save reports and charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def full_evaluation(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        feature_names: list = None,
        label_names: list = None
    ) -> dict:
        """
        Run complete model evaluation.

        Generates all charts and reports then
        returns a summary dictionary.

        Args:
            model: Trained sklearn or XGBoost model
            X_test: Test features
            y_test: True labels
            model_name: Name for charts and reports
            feature_names: Feature names for importance chart
            label_names: Class names for confusion matrix

        Returns:
            Dictionary of all evaluation metrics
        """
        logger.info(
            f"Running full evaluation for {model_name}"
        )

        label_names = label_names or ["Benign", "Attack"]

        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Generate all outputs
        metrics = self._calculate_metrics(
            y_test, y_pred, y_prob
        )

        self._plot_confusion_matrix(
            y_test, y_pred,
            label_names, model_name
        )

        self._plot_precision_recall_curve(
            y_test, y_prob, model_name
        )

        self._plot_roc_curve(
            y_test, y_prob, model_name
        )

        if feature_names is not None:
            self._plot_feature_importance(
                model, feature_names, model_name
            )

        self._plot_threshold_analysis(
            y_test, y_prob, model_name
        )

        report = self._generate_text_report(
            metrics, model_name
        )

        logger.info(
            f"Evaluation complete. "
            f"Reports saved to {self.output_dir}"
        )

        return metrics

    def _calculate_metrics(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> dict:
        """Calculate comprehensive security metrics"""

        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(
            y_test, y_pred
        ).ravel()

        # Core security metrics
        recall = tp / max((tp + fn), 1)
        precision = tp / max((tp + fp), 1)
        f1 = (
            2 * precision * recall /
            max((precision + recall), 1e-10)
        )
        fpr = fp / max((tn + fp), 1)
        fnr = fn / max((tp + fn), 1)

        # Curve metrics
        auc = roc_auc_score(y_test, y_prob)
        avg_precision = average_precision_score(
            y_test, y_prob
        )

        return {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "roc_auc": round(auc, 4),
            "average_precision": round(avg_precision, 4),
            "attacks_detected": int(tp),
            "attacks_missed": int(fn),
            "false_alarms": int(fp)
        }

    def _plot_confusion_matrix(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        label_names: list,
        model_name: str
    ) -> None:
        """
        Plot confusion matrix with security context.

        The confusion matrix is the most important
        visualization for security ML models because
        it shows exactly what the model gets wrong.

        For security specifically:
        - False Negatives (bottom left) = missed attacks
          These are BREACHES. Minimize at all costs.

        - False Positives (top right) = false alarms
          These cause SOC fatigue. Keep manageable.

        - True Positives (bottom right) = caught attacks
          This is what we want. Maximize this.
        """
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
            ax=ax
        )

        ax.set_title(
            f"{model_name} — Confusion Matrix\n"
            f"Bottom-left = Missed Attacks (Breaches)\n"
            f"Top-right = False Alarms (SOC Fatigue)",
            fontsize=11
        )
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_xlabel("Predicted", fontsize=12)

        # Highlight the false negative cell
        # This is the most dangerous cell in security
        ax.add_patch(plt.Rectangle(
            (0, 1), 1, 1,
            fill=False,
            edgecolor="red",
            lw=3,
            label="Missed Attacks"
        ))

        plt.tight_layout()

        save_path = (
            self.output_dir /
            f"{model_name.lower().replace(' ', '_')}"
            f"_confusion_matrix.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion matrix saved: {save_path}")

    def _plot_precision_recall_curve(
        self,
        y_test: np.ndarray,
        y_prob: np.ndarray,
        model_name: str
    ) -> None:
        """
        Plot precision-recall curve.

        More informative than ROC curve for
        imbalanced security datasets.

        Shows the tradeoff between:
        - Precision: how many alerts are real attacks
        - Recall: how many attacks are caught

        SOC managers use this to pick the operating
        threshold that matches their capacity.
        High recall = catch more attacks but more alerts.
        High precision = fewer alerts but miss more attacks.
        """
        precision, recall, thresholds = (
            precision_recall_curve(y_test, y_prob)
        )
        avg_precision = average_precision_score(
            y_test, y_prob
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            recall, precision,
            color="blue",
            lw=2,
            label=f"AP = {avg_precision:.3f}"
        )

        # Mark the operating point at 0.5 threshold
        # Find closest threshold to 0.5
        if len(thresholds) > 0:
            idx = np.argmin(np.abs(thresholds - 0.5))
            ax.scatter(
                recall[idx], precision[idx],
                color="red",
                s=100,
                zorder=5,
                label=f"Threshold=0.5"
            )

        ax.set_xlabel("Recall (Detection Rate)", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(
            f"{model_name} — Precision-Recall Curve\n"
            f"Higher area = better model",
            fontsize=11
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()

        save_path = (
            self.output_dir /
            f"{model_name.lower().replace(' ', '_')}"
            f"_precision_recall.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"PR curve saved: {save_path}")

    def _plot_roc_curve(
        self,
        y_test: np.ndarray,
        y_prob: np.ndarray,
        model_name: str
    ) -> None:
        """
        Plot ROC curve with AUC score.

        Shows model discrimination ability across
        all possible classification thresholds.

        AUC = 1.0: Perfect model
        AUC = 0.5: Random guessing
        AUC > 0.95: Excellent for security use
        """
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            fpr, tpr,
            color="blue",
            lw=2,
            label=f"ROC AUC = {auc:.3f}"
        )

        # Random classifier baseline
        ax.plot(
            [0, 1], [0, 1],
            color="gray",
            lw=1,
            linestyle="--",
            label="Random Classifier"
        )

        ax.set_xlabel(
            "False Positive Rate (False Alarm Rate)",
            fontsize=12
        )
        ax.set_ylabel(
            "True Positive Rate (Detection Rate)",
            fontsize=12
        )
        ax.set_title(
            f"{model_name} — ROC Curve\n"
            f"Higher curve = better model",
            fontsize=11
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = (
            self.output_dir /
            f"{model_name.lower().replace(' ', '_')}"
            f"_roc_curve.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"ROC curve saved: {save_path}")

    def _plot_feature_importance(
        self,
        model,
        feature_names: list,
        model_name: str,
        top_n: int = 15
    ) -> None:
        """
        Plot top N most important features.

        This is critical for analyst trust and
        model explainability.

        When a SOC analyst asks why the model flagged
        a flow as malicious you point to this chart.
        The top features drove the decision.

        Security validation:
        If flow_bytes_per_sec and syn_flag_count are
        top features for DoS detection that makes
        sense domain-wise. If something unexpected
        appears at the top it warrants investigation.
        """
        importances = model.feature_importances_

        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = [
            "#e74c3c" if "engineered" in f else "#3498db"
            for f in top_features
        ]

        bars = ax.barh(
            range(len(top_features)),
            top_importances,
            color=colors
        )

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(
            [f.strip() for f in top_features],
            fontsize=9
        )
        ax.invert_yaxis()

        ax.set_xlabel("Feature Importance Score", fontsize=12)
        ax.set_title(
            f"{model_name} — Top {top_n} Features\n"
            f"Red = Engineered features  "
            f"Blue = Raw CICIDS features",
            fontsize=11
        )

        # Add value labels
        for i, (bar, imp) in enumerate(
            zip(bars, top_importances)
        ):
            ax.text(
                imp + 0.001, i,
                f"{imp:.3f}",
                va="center",
                fontsize=8
            )

        plt.tight_layout()

        save_path = (
            self.output_dir /
            f"{model_name.lower().replace(' ', '_')}"
            f"_feature_importance.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(
            f"Feature importance chart saved: {save_path}"
        )

    def _plot_threshold_analysis(
        self,
        y_test: np.ndarray,
        y_prob: np.ndarray,
        model_name: str
    ) -> None:
        """
        Plot precision, recall, and F1 across thresholds.

        This is one of the most operationally useful
        charts for SOC deployment decisions.

        The default threshold is 0.5 but for security
        you often want to adjust it:

        Lower threshold (e.g. 0.3):
            Catch more attacks (higher recall)
            More false alarms (lower precision)
            Use when: missing an attack is catastrophic

        Higher threshold (e.g. 0.7):
            Fewer false alarms (higher precision)
            Miss more attacks (lower recall)
            Use when: SOC is overwhelmed with alerts

        This chart lets the SOC manager make that
        decision with full visibility of the tradeoff.
        """
        thresholds = np.arange(0.1, 0.95, 0.05)
        precisions = []
        recalls = []
        f1_scores = []

        for threshold in thresholds:
            y_pred_thresh = (y_prob >= threshold).astype(int)

            tp = np.sum((y_test == 1) & (y_pred_thresh == 1))
            fp = np.sum((y_test == 0) & (y_pred_thresh == 1))
            fn = np.sum((y_test == 1) & (y_pred_thresh == 0))

            p = tp / max((tp + fp), 1)
            r = tp / max((tp + fn), 1)
            f1 = 2 * p * r / max((p + r), 1e-10)

            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            thresholds, precisions,
            "b-", lw=2, label="Precision"
        )
        ax.plot(
            thresholds, recalls,
            "r-", lw=2, label="Recall (Detection Rate)"
        )
        ax.plot(
            thresholds, f1_scores,
            "g-", lw=2, label="F1 Score"
        )

        # Mark default threshold
        ax.axvline(
            x=0.5, color="gray",
            linestyle="--", alpha=0.7,
            label="Default threshold (0.5)"
        )

        # Shade the recommended operating zone
        ax.axvspan(
            0.3, 0.5,
            alpha=0.1, color="green",
            label="Recommended range"
        )

        ax.set_xlabel(
            "Classification Threshold", fontsize=12
        )
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(
            f"{model_name} — Threshold Analysis\n"
            f"Lower threshold = more detections "
            f"but more false alarms",
            fontsize=11
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.1, 0.95])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()

        save_path = (
            self.output_dir /
            f"{model_name.lower().replace(' ', '_')}"
            f"_threshold_analysis.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(
            f"Threshold analysis saved: {save_path}"
        )

    def _generate_text_report(
        self,
        metrics: dict,
        model_name: str
    ) -> str:
        """
        Generate analyst-friendly text report.

        This report is what goes into your client
        security assessment documentation and your
        MLOps model registry entry.
        """
        attacks_detected = metrics["attacks_detected"]
        attacks_missed = metrics["attacks_missed"]
        false_alarms = metrics["false_alarms"]
        total_attacks = attacks_detected + attacks_missed

        report = f"""
=====================================
AbuTech AI Security Platform
Network Intrusion Detection Report
=====================================
Model: {model_name}

DETECTION PERFORMANCE
---------------------
Detection Rate:     {metrics['recall']:.1%}
  Attacks Detected: {attacks_detected:,}
  Attacks Missed:   {attacks_missed:,}
  Total Attacks:    {total_attacks:,}

FALSE ALARM RATE
----------------
False Positive Rate: {metrics['false_positive_rate']:.2%}
False Alarms:        {false_alarms:,}

OVERALL SCORES
--------------
F1 Score:    {metrics['f1_score']:.4f}
Precision:   {metrics['precision']:.4f}
ROC-AUC:     {metrics['roc_auc']:.4f}

DEPLOYMENT ASSESSMENT
---------------------"""

        if metrics["recall"] >= 0.95:
            report += "\n✅ Detection rate meets threshold (>95%)"
        else:
            report += (
                f"\n❌ Detection rate below threshold "
                f"({metrics['recall']:.1%} < 95%)"
            )

        if metrics["false_positive_rate"] <= 0.01:
            report += "\n✅ False alarm rate meets threshold (<1%)"
        else:
            report += (
                f"\n⚠️  False alarm rate above threshold "
                f"({metrics['false_positive_rate']:.2%} > 1%)"
            )

        if (metrics["recall"] >= 0.95 and
                metrics["false_positive_rate"] <= 0.01):
            report += (
                "\n\n🚀 RECOMMENDATION: Model meets "
                "production deployment criteria"
            )
        else:
            report += (
                "\n\n⚠️  RECOMMENDATION: Additional tuning "
                "required before production deployment"
            )

        report += f"""

Reports saved to: {self.output_dir}
=====================================
"""

        # Save report to file
        report_path = (
            self.output_dir /
            f"{model_name.lower().replace(' ', '_')}_report.txt"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(report)
        return report