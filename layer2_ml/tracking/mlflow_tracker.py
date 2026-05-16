"""
Layer 2 — ML Processing
MLflow Model Tracker

Provides MLflow integration for tracking
model performance, parameters, and artifacts.

SR 11-7 COMPLIANCE:
    Every model evaluation logged.
    Performance history available for audit.
    Model versions tracked and compared.
    Degradation detected automatically.

USAGE:
    tracker = MLflowTracker()
    
    with tracker.start_run("isolation_forest_eval"):
        tracker.log_params({
            "contamination": 0.1,
            "n_estimators": 100
        })
        tracker.log_metrics({
            "auc": 0.91,
            "precision": 0.88,
            "recall": 0.85
        })
        tracker.log_model_validation(
            model_name="isolation_forest",
            passed=True,
            threshold=0.75,
            actual=0.91
        )
"""

import logging
import os
from datetime import datetime
from datetime import timezone
from typing import Optional

logger = logging.getLogger(__name__)

# MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "mlruns"
)

# Experiment names
EXPERIMENTS = {
    "isolation_forest": "AbuTech-IsolationForest",
    "autoencoder": "AbuTech-Autoencoder",
    "random_forest": "AbuTech-RandomForest",
    "dns_classifier": "AbuTech-DNSClassifier",
    "identity_detector": "AbuTech-IdentityDetector",
    "pii_classifier": "AbuTech-PIIClassifier",
    "lstm_detector": "AbuTech-LSTMDetector",
    "gnn_detector": "AbuTech-GNNDetector",
    "ensemble": "AbuTech-Ensemble",
    "platform": "AbuTech-Platform"
}


class MLflowTracker:
    """
    MLflow integration for AbuTech platform.

    Tracks model performance across all 8 models
    for SR 11-7 compliance and model governance.

    Works with or without MLflow installed.
    Falls back to local JSON logging if unavailable.
    """

    def __init__(
        self,
        tracking_uri: str = MLFLOW_TRACKING_URI,
        experiment_name: str = "AbuTech-Platform"
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.use_mlflow = False
        self.current_run = None
        self._run_data = {}
        self._initialize()

    def _initialize(self):
        """Initialize MLflow connection"""
        try:
            import mlflow
            mlflow.set_tracking_uri(
                self.tracking_uri
            )
            mlflow.set_experiment(
                self.experiment_name
            )
            self.use_mlflow = True
            logger.info(
                f"MLflow initialized: "
                f"{self.tracking_uri}"
            )
        except ImportError:
            logger.info(
                "MLflow not available. "
                "Using local JSON logging."
            )
        except Exception as e:
            logger.warning(
                f"MLflow init failed: {e}. "
                f"Using local JSON logging."
            )

    def start_run(
        self,
        run_name: str = None,
        tags: dict = None
    ):
        """
        Start an MLflow tracking run.
        Use as context manager:

            with tracker.start_run("eval"):
                tracker.log_metrics(...)
        """
        self._run_data = {
            "run_name": run_name or "unnamed",
            "started_at": _now(),
            "params": {},
            "metrics": {},
            "tags": tags or {}
        }

        if self.use_mlflow:
            try:
                import mlflow
                self.current_run = mlflow.start_run(
                    run_name=run_name,
                    tags=tags
                )
            except Exception as e:
                logger.warning(
                    f"MLflow run start failed: {e}"
                )

        return self

    def end_run(self):
        """End the current MLflow run"""
        if self.use_mlflow and self.current_run:
            try:
                import mlflow
                mlflow.end_run()
                self.current_run = None
            except Exception as e:
                logger.warning(
                    f"MLflow run end failed: {e}"
                )

        self._save_local_run()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end_run()

    def log_params(self, params: dict) -> None:
        """Log model parameters"""
        self._run_data["params"].update(params)

        if self.use_mlflow:
            try:
                import mlflow
                mlflow.log_params(params)
            except Exception as e:
                logger.debug(
                    f"MLflow log_params failed: {e}"
                )

    def log_metrics(
        self,
        metrics: dict,
        step: int = None
    ) -> None:
        """Log model performance metrics"""
        self._run_data["metrics"].update(metrics)

        if self.use_mlflow:
            try:
                import mlflow
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.debug(
                    f"MLflow log_metrics failed: {e}"
                )

    def log_metric(
        self,
        key: str,
        value: float,
        step: int = None
    ) -> None:
        """Log a single metric"""
        self.log_metrics({key: value}, step=step)

    def log_model_validation(
        self,
        model_name: str,
        passed: bool,
        threshold: float,
        actual: float,
        metric_name: str = "detection_rate"
    ) -> None:
        """
        Log SR 11-7 model validation result.

        Args:
            model_name: Name of the model
            passed: Whether validation passed
            threshold: Required minimum value
            actual: Actual measured value
            metric_name: What metric was measured
        """
        validation_data = {
            f"{model_name}_passed": int(passed),
            f"{model_name}_{metric_name}": actual,
            f"{model_name}_threshold": threshold,
            f"{model_name}_margin": actual - threshold
        }
        self.log_metrics(validation_data)

        status = "PASS" if passed else "FAIL"
        logger.info(
            f"Model validation {model_name}: "
            f"{status} "
            f"({metric_name}={actual:.3f} "
            f"threshold={threshold:.3f})"
        )

    def log_drift_detection(
        self,
        feature: str,
        ks_statistic: float,
        p_value: float,
        psi: float,
        severity: str
    ) -> None:
        """
        Log data drift detection results.

        Used by check_drift.py for SR 11-7
        ongoing monitoring requirements.
        """
        drift_data = {
            f"drift_{feature}_ks": ks_statistic,
            f"drift_{feature}_pvalue": p_value,
            f"drift_{feature}_psi": psi,
            f"drift_{feature}_critical": int(
                severity == "CRITICAL"
            )
        }
        self.log_metrics(drift_data)

    def log_ensemble_result(
        self,
        final_score: float,
        model_scores: dict,
        pii_sensitivity: str,
        verdict: str
    ) -> None:
        """Log ensemble scoring result"""
        metrics = {
            "ensemble_final_score": final_score,
            "ensemble_pii_elevated": int(
                pii_sensitivity != "NONE"
            )
        }
        for model, score in model_scores.items():
            metrics[f"ensemble_{model}_score"] = score

        self.log_metrics(metrics)
        self.log_params({
            "ensemble_verdict": verdict,
            "ensemble_pii_sensitivity": pii_sensitivity
        })

    def log_investigation(
        self,
        event_id: str,
        severity: str,
        agents_run: int,
        duration_seconds: float,
        hitl_required: bool
    ) -> None:
        """Log investigation metadata"""
        self.log_metrics({
            "investigation_duration": duration_seconds,
            "investigation_agents": agents_run,
            "investigation_hitl": int(hitl_required)
        })
        self.log_params({
            "event_id": event_id,
            "severity": severity
        })

    def get_experiment_summary(
        self,
        experiment_name: str = None
    ) -> dict:
        """
        Get summary of recent runs for dashboard.

        Returns latest metrics for SR 11-7 reporting.
        """
        exp_name = (
            experiment_name or self.experiment_name
        )

        if self.use_mlflow:
            try:
                return self._get_mlflow_summary(
                    exp_name
                )
            except Exception as e:
                logger.warning(
                    f"MLflow summary failed: {e}"
                )

        return self._get_local_summary()

    def _get_mlflow_summary(
        self,
        experiment_name: str
    ) -> dict:
        """Get summary from MLflow"""
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        experiment = client.get_experiment_by_name(
            experiment_name
        )

        if not experiment:
            return {"error": "Experiment not found"}

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=10,
            order_by=["start_time DESC"]
        )

        if not runs:
            return {"total_runs": 0}

        latest = runs[0]
        return {
            "total_runs": len(runs),
            "latest_run_id": latest.info.run_id,
            "latest_metrics": latest.data.metrics,
            "latest_params": latest.data.params,
            "latest_status": latest.info.status,
            "experiment_id": (
                experiment.experiment_id
            )
        }

    def _get_local_summary(self) -> dict:
        """Get summary from local JSON logs"""
        try:
            import json
            log_file = "data/mlflow/runs.json"
            if os.path.exists(log_file):
                with open(log_file) as f:
                    runs = json.load(f)
                    if runs:
                        return {
                            "total_runs": len(runs),
                            "latest_run": runs[-1],
                            "backend": "local_json"
                        }
        except Exception:
            pass
        return {
            "total_runs": 0,
            "backend": "local_json"
        }

    def _save_local_run(self) -> None:
        """Save run data to local JSON"""
        try:
            import json
            os.makedirs("data/mlflow", exist_ok=True)
            log_file = "data/mlflow/runs.json"

            runs = []
            if os.path.exists(log_file):
                with open(log_file) as f:
                    runs = json.load(f)

            self._run_data["ended_at"] = _now()
            runs.append(self._run_data)

            if len(runs) > 1000:
                runs = runs[-1000:]

            with open(log_file, "w") as f:
                json.dump(runs, f, indent=2)

        except Exception as e:
            logger.debug(
                f"Local run save failed: {e}"
            )


def log_model_performance(
    model_name: str,
    metrics: dict,
    params: dict = None,
    experiment: str = None
) -> None:
    """
    Convenience function for logging
    model performance from anywhere.

    Used by validate_models.py and
    check_drift.py in CI/CD pipeline.
    """
    exp = (
        experiment or
        EXPERIMENTS.get(model_name, "AbuTech-Platform")
    )
    tracker = MLflowTracker(
        experiment_name=exp
    )

    with tracker.start_run(
        run_name=f"{model_name}_{_short_ts()}"
    ):
        if params:
            tracker.log_params(params)
        tracker.log_metrics(metrics)

    logger.info(
        f"Logged performance for {model_name}: "
        f"{metrics}"
    )


def _now() -> str:
    return datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


def _short_ts() -> str:
    return datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
