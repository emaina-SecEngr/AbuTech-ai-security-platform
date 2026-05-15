"""
Tests for MLflow Tracker
"""

import pytest
import os
import tempfile


@pytest.fixture
def tracker(tmp_path):
    from layer2_ml.tracking.mlflow_tracker import (
        MLflowTracker
    )
    return MLflowTracker(
        tracking_uri=str(tmp_path / "mlruns"),
        experiment_name="test-experiment"
    )


class TestMLflowTrackerInit:

    def test_tracker_initializes(self, tracker):
        assert tracker is not None

    def test_tracker_has_backend_flag(self, tracker):
        assert hasattr(tracker, "use_mlflow")

    def test_tracking_uri_set(self, tracker):
        assert tracker.tracking_uri is not None

    def test_experiment_name_set(self, tracker):
        assert tracker.experiment_name == (
            "test-experiment"
        )


class TestRunManagement:

    def test_start_run_returns_self(self, tracker):
        result = tracker.start_run("test-run")
        tracker.end_run()
        assert result is tracker

    def test_context_manager(self, tracker):
        with tracker.start_run("ctx-run"):
            tracker.log_metric("test", 0.5)

    def test_run_name_stored(self, tracker):
        tracker.start_run("my-run")
        assert tracker._run_data["run_name"] == "my-run"
        tracker.end_run()

    def test_multiple_runs(self, tracker):
        with tracker.start_run("run-1"):
            tracker.log_metric("m1", 0.8)
        with tracker.start_run("run-2"):
            tracker.log_metric("m2", 0.9)


class TestLogging:

    def test_log_params(self, tracker):
        with tracker.start_run("params-test"):
            tracker.log_params({
                "n_estimators": 100,
                "contamination": 0.1
            })
        assert tracker._run_data["params"][
            "n_estimators"
        ] == 100

    def test_log_metrics(self, tracker):
        with tracker.start_run("metrics-test"):
            tracker.log_metrics({
                "auc": 0.91,
                "precision": 0.88
            })
        assert tracker._run_data["metrics"][
            "auc"
        ] == 0.91

    def test_log_metric_single(self, tracker):
        with tracker.start_run("single-metric"):
            tracker.log_metric("recall", 0.85)
        assert tracker._run_data["metrics"][
            "recall"
        ] == 0.85

    def test_log_model_validation_pass(
        self, tracker
    ):
        with tracker.start_run("validation-pass"):
            tracker.log_model_validation(
                model_name="isolation_forest",
                passed=True,
                threshold=0.75,
                actual=0.91
            )
        metrics = tracker._run_data["metrics"]
        assert metrics["isolation_forest_passed"] == 1
        assert metrics[
            "isolation_forest_detection_rate"
        ] == 0.91

    def test_log_model_validation_fail(
        self, tracker
    ):
        with tracker.start_run("validation-fail"):
            tracker.log_model_validation(
                model_name="lstm_detector",
                passed=False,
                threshold=0.70,
                actual=0.65
            )
        metrics = tracker._run_data["metrics"]
        assert metrics["lstm_detector_passed"] == 0

    def test_log_drift_detection(self, tracker):
        with tracker.start_run("drift-test"):
            tracker.log_drift_detection(
                feature="timestamp_norm",
                ks_statistic=0.05,
                p_value=0.82,
                psi=0.03,
                severity="OK"
            )
        metrics = tracker._run_data["metrics"]
        assert "drift_timestamp_norm_ks" in metrics
        assert "drift_timestamp_norm_psi" in metrics

    def test_log_drift_critical(self, tracker):
        with tracker.start_run("drift-critical"):
            tracker.log_drift_detection(
                feature="volume_norm",
                ks_statistic=0.45,
                p_value=0.01,
                psi=0.35,
                severity="CRITICAL"
            )
        metrics = tracker._run_data["metrics"]
        assert metrics[
            "drift_volume_norm_critical"
        ] == 1

    def test_log_ensemble_result(self, tracker):
        with tracker.start_run("ensemble-test"):
            tracker.log_ensemble_result(
                final_score=0.95,
                model_scores={
                    "isolation_forest": 0.90,
                    "pii_classifier": 0.95
                },
                pii_sensitivity="PCI",
                verdict="DATA_EXFILTRATION"
            )
        metrics = tracker._run_data["metrics"]
        assert metrics["ensemble_final_score"] == 0.95
        assert metrics[
            "ensemble_isolation_forest_score"
        ] == 0.90

    def test_log_investigation(self, tracker):
        with tracker.start_run("inv-test"):
            tracker.log_investigation(
                event_id="evt-001",
                severity="CRITICAL",
                agents_run=4,
                duration_seconds=44.1,
                hitl_required=True
            )
        metrics = tracker._run_data["metrics"]
        assert metrics["investigation_agents"] == 4
        assert metrics["investigation_hitl"] == 1


class TestExperimentSummary:

    def test_get_summary_returns_dict(self, tracker):
        summary = tracker.get_experiment_summary()
        assert isinstance(summary, dict)

    def test_summary_has_total_runs(self, tracker):
        with tracker.start_run("summary-test"):
            tracker.log_metric("test", 0.5)
        summary = tracker.get_experiment_summary()
        assert "total_runs" in summary


class TestConvenienceFunction:

    def test_log_model_performance(self, tmp_path):
        from layer2_ml.tracking.mlflow_tracker import (
            log_model_performance
        )
        log_model_performance(
            model_name="isolation_forest",
            metrics={"auc": 0.91, "precision": 0.88},
            params={"n_estimators": 100}
        )

    def test_log_model_performance_no_params(
        self, tmp_path
    ):
        from layer2_ml.tracking.mlflow_tracker import (
            log_model_performance
        )
        log_model_performance(
            model_name="pii_classifier",
            metrics={"precision": 0.95}
        )