"""
AbuTech AI Security Platform
Network Intrusion Detection — Training Script

This script trains your first real ML model on
actual CICIDS2017 network traffic data.

It runs the complete training pipeline:
    1. Load and prepare CICIDS2017 data
    2. Engineer security-specific features
    3. Train Random Forest baseline
    4. Train XGBoost optimized model
    5. Evaluate both models
    6. Generate visualizations and reports
    7. Save best model for production use

After running this script you will have:
    - Two trained models tracked in MLflow
    - Confusion matrices showing detection performance
    - Feature importance charts
    - Threshold analysis for SOC deployment
    - A saved model ready for the detector.py

Run with:
    python train_intrusion_detector.py

View MLflow results:
    mlflow ui
    Then open http://localhost:5000 in your browser
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from layer2_ml.intrusion_detection.data_preparation import (
    CICIDSDataPreparation
)
from layer2_ml.intrusion_detection.feature_engineering import (
    NetworkFlowFeatureEngineer
)
from layer2_ml.intrusion_detection.model_trainer import (
    IntrusionDetectionTrainer
)
from layer2_ml.intrusion_detection.model_evaluator import (
    IntrusionDetectionEvaluator
)


def main():

    print("\n" + "="*60)
    print("AbuTech AI Security Platform")
    print("Network Intrusion Detection — Model Training")
    print("="*60 + "\n")

    # --------------------------------------------------------
    # CONFIGURATION
    # Adjust these settings based on your machine's memory
    # Start with max_samples=50000 for faster first run
    # Set to None for full dataset training
    # --------------------------------------------------------

    DATA_DIR = "data/cicids2017"
    MODEL_SAVE_DIR = "models/intrusion_detection"
    REPORTS_DIR = "reports/intrusion_detection"

    # Start with Friday files — clear attack examples
    # These two files together have DDoS and PortScan
    CSV_FILES = [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    ]

    # Limit samples for faster first training run
    # Change to None to train on all data
    MAX_SAMPLES = 50000

    # --------------------------------------------------------
    # STEP 1 — DATA PREPARATION
    # --------------------------------------------------------

    print("STEP 1: Loading and preparing CICIDS2017 data")
    print("-" * 40)

    prep = CICIDSDataPreparation(
        data_dir=DATA_DIR,
        test_size=0.2,
        random_state=42,
        apply_smote=True,
        max_samples=MAX_SAMPLES
    )

    try:
        X_train, X_test, y_train, y_test = prep.prepare(
            csv_files=CSV_FILES
        )
    except FileNotFoundError as e:
        print(f"\n❌ Data not found: {e}")
        print("\nTo fix this:")
        print("1. Download CICIDS2017 from:")
        print("   https://www.unb.ca/cic/datasets/ids-2017.html")
        print("2. Extract MachineLearningCSV.zip")
        print(f"3. Copy CSV files to: {DATA_DIR}/")
        return

    print(f"\n✅ Data prepared successfully")
    print(f"   Training samples: {len(y_train):,}")
    print(f"   Test samples:     {len(y_test):,}")
    print(f"   Attack ratio:     {np.mean(y_test):.1%}")

    # Get feature names for reporting
    feature_names = prep.get_feature_names()
    prep_stats = prep.get_statistics()

    print(f"\n   Original class ratio: "
          f"{prep_stats.get('imbalance_ratio', 'N/A'):.0f}:1")

    # --------------------------------------------------------
    # STEP 2 — FEATURE ENGINEERING
    # --------------------------------------------------------

    print("\nSTEP 2: Engineering security-specific features")
    print("-" * 40)

    print("   Note: Feature engineering applied during")
    print("   data preparation via SELECTED_FEATURES.")
    print("   Advanced engineered features available")
    print("   in feature_engineering.py for next run.")
    print("✅ Features ready")

    # --------------------------------------------------------
    # STEP 3 — TRAIN RANDOM FOREST
    # --------------------------------------------------------

    print("\nSTEP 3: Training Random Forest baseline")
    print("-" * 40)
    print("   This may take 1-3 minutes...")

    trainer = IntrusionDetectionTrainer(
        experiment_name="network_intrusion_detection",
        model_save_dir=MODEL_SAVE_DIR
    )

    rf_model = trainer.train_random_forest(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names
    )

    print("✅ Random Forest training complete")

    # --------------------------------------------------------
    # STEP 4 — TRAIN XGBOOST
    # --------------------------------------------------------

    print("\nSTEP 4: Training XGBoost model")
    print("-" * 40)
    print("   This may take 2-5 minutes...")

    xgb_model = trainer.train_xgboost(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names
    )

    print("✅ XGBoost training complete")

    # --------------------------------------------------------
    # STEP 5 — EVALUATE BOTH MODELS
    # --------------------------------------------------------

    print("\nSTEP 5: Generating evaluation reports")
    print("-" * 40)

    evaluator = IntrusionDetectionEvaluator(
        output_dir=REPORTS_DIR
    )

    print("   Evaluating Random Forest...")
    rf_metrics = evaluator.full_evaluation(
        model=rf_model,
        X_test=X_test,
        y_test=y_test,
        model_name="Random Forest",
        feature_names=feature_names
    )

    print("   Evaluating XGBoost...")
    xgb_metrics = evaluator.full_evaluation(
        model=xgb_model,
        X_test=X_test,
        y_test=y_test,
        model_name="XGBoost",
        feature_names=feature_names
    )

    print("✅ Evaluation reports generated")

    # --------------------------------------------------------
    # STEP 6 — COMPARE AND SAVE BEST MODEL
    # --------------------------------------------------------

    print("\nSTEP 6: Comparing models and saving best")
    print("-" * 40)

    best_model, best_name = trainer.get_best_model()

    print(f"\n   Model Comparison:")
    print(f"   {'Metric':<25} {'Random Forest':>15} "
          f"{'XGBoost':>15}")
    print(f"   {'-'*55}")

    metrics_to_compare = [
        ("F1 Score", "f1_score"),
        ("Detection Rate", "recall"),
        ("Precision", "precision"),
        ("False Positive Rate", "false_positive_rate"),
        ("ROC AUC", "roc_auc")
    ]

    for label, key in metrics_to_compare:
        rf_val = rf_metrics.get(key, 0)
        xgb_val = xgb_metrics.get(key, 0)
        winner = "←" if rf_val > xgb_val else "  "
        winner_xgb = "←" if xgb_val > rf_val else "  "
        print(
            f"   {label:<25} "
            f"{rf_val:>14.4f}{winner} "
            f"{xgb_val:>14.4f}{winner_xgb}"
        )

    print(f"\n   🏆 Best Model: {best_name}")

    # Save best model
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    model_path = Path(MODEL_SAVE_DIR) / "best_model.pkl"
    scaler_path = Path(MODEL_SAVE_DIR) / "scaler.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    with open(scaler_path, "wb") as f:
        pickle.dump(prep.scaler, f)

    print(f"   ✅ Model saved: {model_path}")
    print(f"   ✅ Scaler saved: {scaler_path}")

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\n✅ Best model: {best_name}")
    print(f"✅ Detection rate: "
          f"{max(rf_metrics['recall'], xgb_metrics['recall']):.1%}")
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Reports saved to: {REPORTS_DIR}/")
    print(f"\nMLflow Experiment Results:")
    print(f"   Run: mlflow ui")
    print(f"   Open: http://localhost:5000")
    print(f"\nNext step: Run the detector on live events")
    print(f"   from layer2_ml.intrusion_detection.detector "
          f"import NetworkIntrusionDetector")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()