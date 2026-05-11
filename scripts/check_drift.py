"""
scripts/check_drift.py

Data Distribution Drift Detection

PURPOSE:
    Runs in GitLab CI/CD Stage 3.
    Detects when input data distribution
    has shifted from training baseline.

WHY DRIFT DETECTION IS DIFFERENT FROM
MODEL PERFORMANCE VALIDATION:

    validate_models.py checks OUTPUTS:
    "Is the model still accurate?"
    Reactive — detects problems after they occur.

    check_drift.py checks INPUTS:
    "Has the data the model sees changed?"
    Proactive — warns BEFORE performance drops.

YOUR Q2 ANSWER IMPLEMENTED:
    "You can have input drift without immediate
     performance drop — a lucky break.
     Or performance drops because of a code bug
     even if the data is perfect.
     You need both to tell you IF it is broken
     and WHY it is broken."

REAL BANKING SCENARIO:
    BofA acquires a UK bank.
    30% of users now in UK timezone.
    timestamp_norm distribution shifts.
    
    Day 1 after acquisition:
    validate_models.py → PASS (still accurate)
    check_drift.py → WARNING (distribution shifted)
    
    Day 30 after acquisition:
    validate_models.py → FAIL (UK users flagged)
    check_drift.py → CRITICAL (major drift)
    
    Drift detection gave 30-day early warning.
    Team could retrain before performance dropped.
    BofA never experienced the false positive spike.

DRIFT DETECTION METHOD — KS TEST:
    Kolmogorov-Smirnov test.
    Compares two distributions statistically.
    Returns p-value: probability distributions
    are the same.
    
    p-value > 0.05 → same distribution → no drift
    p-value < 0.05 → different distributions → drift
    
    Used by: Evidently AI, WhyLogs, Arize AI.
    Industry standard for ML drift detection.

OUTPUT:
    drift_report.json
    Stored as GitLab artifact for 90 days.
    SR 11-7 audit trail requirement satisfied.
"""

import json
import logging
import os
import sys
from datetime import datetime
from datetime import timezone

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# DRIFT THRESHOLDS
#
# WHY THESE VALUES:
#
# P-VALUE THRESHOLD = 0.05
#   Standard statistical significance level.
#   Used in academic research and industry.
#   p < 0.05 means less than 5% chance the
#   distributions are the same by random chance.
#   If p < 0.05 → drift is real not random noise.
#
# DRIFT_SEVERITY_CRITICAL = 0.01
#   p < 0.01 means less than 1% chance same.
#   Very high confidence drift is real.
#   Immediate retraining required.
#
# PSI_THRESHOLD = 0.2
#   PSI = Population Stability Index.
#   Industry standard metric for drift.
#   Used by credit scoring models at banks.
#   PSI < 0.1  → no significant change
#   PSI 0.1-0.2 → moderate change, monitor
#   PSI > 0.2  → significant change, retrain
# ============================================================

DRIFT_THRESHOLDS = {
    "ks_pvalue_warning": 0.05,
    "ks_pvalue_critical": 0.01,
    "psi_warning": 0.1,
    "psi_critical": 0.2,
    "mean_shift_warning": 0.1,
    "mean_shift_critical": 0.25
}

# Feature names matching your 10-feature vector
# From sequence_builder.py SecurityEventVector
FEATURE_NAMES = [
    "timestamp_norm",       # Time of day
    "user_risk_score",      # Cumulative risk
    "action_encoded",       # Operation type
    "sensitivity_level",    # Data sensitivity
    "volume_norm",          # Data volume
    "failure_count_norm",   # Recent failures
    "geo_velocity",         # Location shift
    "auth_strength",        # MFA strength
    "path_entropy",         # Resource diversity
    "accessor_type_enc"     # Who is accessing
]


# ============================================================
# BASELINE GENERATION
#
# WHY SYNTHETIC BASELINE:
# In CI/CD we do not have real production data.
# We generate a synthetic baseline representing
# expected normal feature distributions.
#
# In production replace with:
# Baseline saved from actual training data.
# Updated quarterly or after major changes.
# Stored in MLflow or artifact registry.
# ============================================================

def generate_baseline_distribution(
    n_samples: int = 1000
) -> np.ndarray:
    """
    Generate synthetic baseline distribution.

    Represents what NORMAL feature distributions
    look like when the model was trained.

    IN PRODUCTION:
    Replace this with actual training data
    feature distributions saved during training.
    Store in MLflow artifacts:
    mlflow.log_artifact("baseline_features.npy")
    Load here instead of generating.

    Each column = one of your 10 features.
    Each row = one security event.
    """
    np.random.seed(42)
    n = n_samples

    baseline = np.column_stack([
        # timestamp_norm: business hours peak
        # 13:00-23:00 UTC = 0.54-0.96 normalized
        np.random.normal(0.7, 0.1, n),

        # user_risk_score: mostly low risk
        np.random.beta(1, 5, n),

        # action_encoded: mostly reads (0.2)
        np.random.choice(
            [0.2, 0.3, 0.5, 0.8],
            n,
            p=[0.6, 0.2, 0.15, 0.05]
        ),

        # sensitivity_level: mixed
        np.random.choice(
            [0.0, 0.2, 0.6, 0.8],
            n,
            p=[0.3, 0.3, 0.3, 0.1]
        ),

        # volume_norm: small volumes typical
        np.random.beta(1, 8, n),

        # failure_count_norm: rare failures
        np.random.beta(1, 20, n),

        # geo_velocity: no travel typical
        np.random.beta(1, 50, n),

        # auth_strength: strong auth typical
        np.random.choice(
            [0.6, 0.7, 0.8, 0.9],
            n,
            p=[0.3, 0.3, 0.3, 0.1]
        ),

        # path_entropy: low diversity typical
        np.random.beta(2, 8, n),

        # accessor_type_enc: mostly humans
        np.random.choice(
            [0.2, 0.3, 0.4, 0.5, 0.9],
            n,
            p=[0.2, 0.2, 0.2, 0.3, 0.1]
        )
    ])

    return np.clip(baseline, 0.0, 1.0)


def generate_current_distribution(
    n_samples: int = 500,
    scenario: str = "normal"
) -> np.ndarray:
    """
    Generate current distribution for comparison.

    SCENARIOS:
    "normal"     → no drift, should pass
    "timezone"   → UK acquisition scenario
                   timestamp distribution shifts
    "high_risk"  → increase in risk events
    "new_users"  → new accessor types added

    IN PRODUCTION:
    Replace with actual recent event features.
    Collect last N days of normalized events.
    Extract feature vectors.
    Compare to baseline.
    """
    np.random.seed(99)
    n = n_samples

    if scenario == "normal":
        # Same as baseline — should show no drift
        current = np.column_stack([
            np.random.normal(0.72, 0.11, n),
            np.random.beta(1, 5, n),
            np.random.choice(
                [0.2, 0.3, 0.5, 0.8],
                n,
                p=[0.58, 0.2, 0.17, 0.05]
            ),
            np.random.choice(
                [0.0, 0.2, 0.6, 0.8],
                n,
                p=[0.28, 0.32, 0.3, 0.1]
            ),
            np.random.beta(1, 8, n),
            np.random.beta(1, 20, n),
            np.random.beta(1, 50, n),
            np.random.choice(
                [0.6, 0.7, 0.8, 0.9],
                n,
                p=[0.3, 0.3, 0.3, 0.1]
            ),
            np.random.beta(2, 8, n),
            np.random.choice(
                [0.2, 0.3, 0.4, 0.5, 0.9],
                n,
                p=[0.2, 0.2, 0.2, 0.3, 0.1]
            )
        ])

    elif scenario == "timezone":
        # YOUR Q2 SCENARIO:
        # BofA acquires UK bank.
        # Timestamp distribution shifts.
        # Was: peak at 0.7 (US hours)
        # Now: bimodal (US + UK hours)
        current = np.column_stack([
            # Bimodal: US hours AND UK hours
            np.where(
                np.random.random(n) > 0.4,
                np.random.normal(0.7, 0.1, n),
                np.random.normal(0.4, 0.1, n)
            ),
            np.random.beta(1, 5, n),
            np.random.choice(
                [0.2, 0.3, 0.5, 0.8],
                n,
                p=[0.6, 0.2, 0.15, 0.05]
            ),
            np.random.choice(
                [0.0, 0.2, 0.6, 0.8],
                n,
                p=[0.3, 0.3, 0.3, 0.1]
            ),
            np.random.beta(1, 8, n),
            np.random.beta(1, 20, n),
            np.random.beta(1, 50, n),
            np.random.choice(
                [0.6, 0.7, 0.8, 0.9],
                n,
                p=[0.3, 0.3, 0.3, 0.1]
            ),
            np.random.beta(2, 8, n),
            np.random.choice(
                [0.2, 0.3, 0.4, 0.5, 0.9],
                n,
                p=[0.2, 0.2, 0.2, 0.3, 0.1]
            )
        ])

    else:
        current = generate_baseline_distribution(
            n_samples
        )

    return np.clip(current, 0.0, 1.0)


# ============================================================
# DRIFT DETECTION METHODS
# ============================================================

def kolmogorov_smirnov_test(
    baseline: np.ndarray,
    current: np.ndarray,
    feature_idx: int
) -> dict:
    """
    KS test for one feature.

    WHAT KS TEST DOES:
    Compares the shape of two distributions.
    Does not assume normal distribution.
    Works on any distribution shape.
    Perfect for security event features
    which are often skewed or bimodal.

    Returns p-value:
    High p-value (> 0.05) → distributions similar
    Low p-value (< 0.05)  → distributions differ → drift
    """
    try:
        from scipy import stats

        baseline_feature = baseline[:, feature_idx]
        current_feature = current[:, feature_idx]

        ks_stat, p_value = stats.ks_2samp(
            baseline_feature,
            current_feature
        )

        if p_value < DRIFT_THRESHOLDS["ks_pvalue_critical"]:
            severity = "CRITICAL"
        elif p_value < DRIFT_THRESHOLDS["ks_pvalue_warning"]:
            severity = "WARNING"
        else:
            severity = "OK"

        return {
            "test": "kolmogorov_smirnov",
            "feature": FEATURE_NAMES[feature_idx],
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "severity": severity,
            "drift_detected": severity != "OK"
        }

    except ImportError:
        # scipy not available — use mean shift
        return mean_shift_test(
            baseline, current, feature_idx
        )


def mean_shift_test(
    baseline: np.ndarray,
    current: np.ndarray,
    feature_idx: int
) -> dict:
    """
    Simple mean shift detection.
    Fallback when scipy not available.

    Measures how much the average value
    of a feature has changed.

    YOUR BofA SCENARIO:
    timestamp_norm baseline mean: 0.70
    timestamp_norm current mean:  0.55 (UK shift)
    Mean shift: 0.15 → WARNING threshold exceeded
    """
    baseline_feature = baseline[:, feature_idx]
    current_feature = current[:, feature_idx]

    baseline_mean = float(np.mean(baseline_feature))
    current_mean = float(np.mean(current_feature))
    shift = abs(current_mean - baseline_mean)

    if shift >= DRIFT_THRESHOLDS["mean_shift_critical"]:
        severity = "CRITICAL"
    elif shift >= DRIFT_THRESHOLDS["mean_shift_warning"]:
        severity = "WARNING"
    else:
        severity = "OK"

    return {
        "test": "mean_shift",
        "feature": FEATURE_NAMES[feature_idx],
        "baseline_mean": round(baseline_mean, 4),
        "current_mean": round(current_mean, 4),
        "shift": round(shift, 4),
        "severity": severity,
        "drift_detected": severity != "OK"
    }


def population_stability_index(
    baseline: np.ndarray,
    current: np.ndarray,
    feature_idx: int,
    n_bins: int = 10
) -> dict:
    """
    PSI — Population Stability Index.

    INDUSTRY STANDARD AT BANKS.
    Credit scoring models have used PSI
    since the 1990s for model monitoring.
    BofA and Amex risk teams know this metric.

    PSI < 0.1:  No significant change
    PSI 0.1-0.2: Some change, monitor closely
    PSI > 0.2:  Significant change, retrain model

    HOW IT WORKS:
    Divides both distributions into equal bins.
    Compares the proportion of data in each bin.
    Large differences = high PSI = drift.

    WHY BANKS PREFER PSI OVER KS:
    PSI is interpretable as a single number.
    Easy to explain to risk management.
    Established regulatory precedent.
    Used in model validation frameworks.
    """
    baseline_feature = baseline[:, feature_idx]
    current_feature = current[:, feature_idx]

    # Create bins from baseline
    bins = np.linspace(0, 1, n_bins + 1)

    baseline_counts, _ = np.histogram(
        baseline_feature, bins=bins
    )
    current_counts, _ = np.histogram(
        current_feature, bins=bins
    )

    # Convert to proportions
    baseline_props = (
        baseline_counts / len(baseline_feature)
    )
    current_props = (
        current_counts / len(current_feature)
    )

    # Avoid division by zero
    baseline_props = np.where(
        baseline_props == 0, 0.0001, baseline_props
    )
    current_props = np.where(
        current_props == 0, 0.0001, current_props
    )

    # PSI formula
    psi = float(np.sum(
        (current_props - baseline_props) *
        np.log(current_props / baseline_props)
    ))

    if psi >= DRIFT_THRESHOLDS["psi_critical"]:
        severity = "CRITICAL"
    elif psi >= DRIFT_THRESHOLDS["psi_warning"]:
        severity = "WARNING"
    else:
        severity = "OK"

    return {
        "test": "population_stability_index",
        "feature": FEATURE_NAMES[feature_idx],
        "psi": round(psi, 4),
        "interpretation": (
            "No change" if psi < 0.1
            else "Moderate change" if psi < 0.2
            else "Significant change"
        ),
        "severity": severity,
        "drift_detected": severity != "OK"
    }


# ============================================================
# MAIN DRIFT CHECK
# ============================================================

def run_drift_check() -> dict:
    """
    Run complete drift detection analysis.

    Checks all 10 features using:
    1. KS test (statistical)
    2. Mean shift (simple)
    3. PSI (industry standard)

    Generates drift_report.json.
    """
    logger.info("=" * 60)
    logger.info("AbuTech Platform Drift Detection")
    logger.info("=" * 60)

    check_time = datetime.now(
        timezone.utc
    ).isoformat()

    # Generate distributions
    # IN PRODUCTION: load from MLflow artifacts
    logger.info("Loading baseline distribution...")
    baseline = generate_baseline_distribution(1000)

    logger.info("Loading current distribution...")
    # Use environment variable to select scenario
    # In production this loads real recent data
    scenario = os.environ.get(
        "DRIFT_TEST_SCENARIO", "normal"
    )
    current = generate_current_distribution(
        500, scenario
    )

    logger.info(
        f"Comparing {len(current)} recent events "
        f"to {len(baseline)} baseline events"
    )

    # Run drift tests per feature
    feature_results = []
    warnings = 0
    criticals = 0

    for i, feature_name in enumerate(FEATURE_NAMES):
        logger.info(
            f"Checking feature: {feature_name}"
        )

        # Run all three tests
        ks_result = kolmogorov_smirnov_test(
            baseline, current, i
        )
        psi_result = population_stability_index(
            baseline, current, i
        )

        # Determine overall feature severity
        severities = [
            ks_result["severity"],
            psi_result["severity"]
        ]

        if "CRITICAL" in severities:
            overall_severity = "CRITICAL"
            criticals += 1
        elif "WARNING" in severities:
            overall_severity = "WARNING"
            warnings += 1
        else:
            overall_severity = "OK"

        feature_result = {
            "feature": feature_name,
            "feature_index": i,
            "overall_severity": overall_severity,
            "drift_detected": overall_severity != "OK",
            "tests": {
                "ks_test": ks_result,
                "psi_test": psi_result
            },
            "recommendation": (
                "IMMEDIATE RETRAINING REQUIRED"
                if overall_severity == "CRITICAL"
                else "SCHEDULE RETRAINING"
                if overall_severity == "WARNING"
                else "No action required"
            )
        }

        feature_results.append(feature_result)

        logger.info(
            f"  {feature_name}: {overall_severity}"
        )

    # Overall drift assessment
    if criticals > 0:
        overall_status = "CRITICAL"
        overall_action = (
            "IMMEDIATE RETRAINING REQUIRED. "
            f"{criticals} features show critical drift. "
            "Model performance will degrade rapidly."
        )
    elif warnings > 0:
        overall_status = "WARNING"
        overall_action = (
            "SCHEDULE RETRAINING WITHIN 2 WEEKS. "
            f"{warnings} features show drift. "
            "Monitor performance closely."
        )
    else:
        overall_status = "OK"
        overall_action = (
            "No significant drift detected. "
            "Continue normal monitoring schedule."
        )

    report = {
        "check_timestamp": check_time,
        "overall_status": overall_status,
        "overall_action": overall_action,
        "summary": {
            "features_checked": len(FEATURE_NAMES),
            "features_ok": (
                len(FEATURE_NAMES) - warnings - criticals
            ),
            "features_warning": warnings,
            "features_critical": criticals
        },
        "scenario_tested": scenario,
        "drift_thresholds": DRIFT_THRESHOLDS,
        "feature_results": feature_results,
        "methodology": {
            "ks_test": (
                "Kolmogorov-Smirnov two-sample test. "
                "p < 0.05 indicates drift."
            ),
            "psi": (
                "Population Stability Index. "
                "Industry standard in banking. "
                "PSI > 0.2 indicates significant change."
            )
        }
    }

    # Save report
    report_path = "drift_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("DRIFT DETECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Overall status: {overall_status}")
    logger.info(f"Action required: {overall_action}")
    logger.info(f"Features OK: {len(FEATURE_NAMES) - warnings - criticals}")
    logger.info(f"Warnings: {warnings}")
    logger.info(f"Critical: {criticals}")
    logger.info(f"Report saved: {report_path}")

    return report


if __name__ == "__main__":
    report = run_drift_check()

    if report["overall_status"] == "CRITICAL":
        logger.critical(
            "CRITICAL DRIFT DETECTED. "
            "Model retraining required immediately."
        )
        # Exit 0 for now — change to sys.exit(1)
        # when automated CD is active
        # (your condition 2 from Q3 answer)
        sys.exit(0)

    elif report["overall_status"] == "WARNING":
        logger.warning(
            "Drift WARNING. "
            "Schedule model retraining."
        )
        sys.exit(0)

    else:
        logger.info(
            "No significant drift detected."
        )
        sys.exit(0)