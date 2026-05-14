"""
Tests for Unified Ensemble Scorer

Tests cover:
    Weight validation
    Basic scoring
    PII multiplier
    Verdict determination
    Edge cases
    SR 11-7 explainability
"""

import pytest
from layer2_ml.ensemble.ensemble_scorer import (
    EnsembleScorer,
    ModelScore,
    EnsembleResult,
    MODEL_WEIGHTS,
    PII_MULTIPLIERS,
    RISK_THRESHOLDS
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def scorer():
    return EnsembleScorer()


@pytest.fixture
def critical_scores():
    """Scores representing a CRITICAL attack event"""
    return {
        "isolation_forest": ModelScore(
            model_name="isolation_forest",
            score=0.92, confidence=0.95
        ),
        "autoencoder": ModelScore(
            model_name="autoencoder",
            score=0.88, confidence=0.90
        ),
        "random_forest": ModelScore(
            model_name="random_forest",
            score=0.85, confidence=0.92
        ),
        "dns_classifier": ModelScore(
            model_name="dns_classifier",
            score=0.30, confidence=0.80
        ),
        "identity_detector": ModelScore(
            model_name="identity_detector",
            score=0.91, confidence=0.95
        ),
        "pii_classifier": ModelScore(
            model_name="pii_classifier",
            score=0.95, confidence=1.0
        ),
        "lstm_detector": ModelScore(
            model_name="lstm_detector",
            score=0.82, confidence=0.85
        ),
        "gnn_detector": ModelScore(
            model_name="gnn_detector",
            score=0.91, confidence=0.88
        ),
    }


@pytest.fixture
def normal_scores():
    """Scores representing normal activity"""
    return {
        "isolation_forest": ModelScore(
            model_name="isolation_forest",
            score=0.15, confidence=0.95
        ),
        "autoencoder": ModelScore(
            model_name="autoencoder",
            score=0.12, confidence=0.90
        ),
        "random_forest": ModelScore(
            model_name="random_forest",
            score=0.08, confidence=0.92
        ),
        "identity_detector": ModelScore(
            model_name="identity_detector",
            score=0.10, confidence=0.95
        ),
        "pii_classifier": ModelScore(
            model_name="pii_classifier",
            score=0.05, confidence=1.0
        ),
    }


# ============================================================
# WEIGHT VALIDATION TESTS
# ============================================================

class TestWeightValidation:

    def test_default_weights_sum_to_one(self):
        total = sum(MODEL_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_all_models_have_weights(self):
        expected = {
            "isolation_forest", "autoencoder",
            "random_forest", "dns_classifier",
            "identity_detector", "pii_classifier",
            "lstm_detector", "gnn_detector"
        }
        assert set(MODEL_WEIGHTS.keys()) == expected

    def test_invalid_weights_raise_error(self):
        with pytest.raises(ValueError):
            EnsembleScorer(weights={
                "isolation_forest": 0.5,
                "autoencoder": 0.8
            })

    def test_scorer_initializes_with_default_weights(
        self, scorer
    ):
        assert scorer.weights == MODEL_WEIGHTS

    def test_custom_weights_accepted(self):
        custom = {
            "isolation_forest": 0.50,
            "autoencoder": 0.20,
            "random_forest": 0.10,
            "dns_classifier": 0.05,
            "identity_detector": 0.05,
            "pii_classifier": 0.05,
            "lstm_detector": 0.025,
            "gnn_detector": 0.025
        }
        scorer = EnsembleScorer(weights=custom)
        assert scorer.weights == custom


# ============================================================
# BASIC SCORING TESTS
# ============================================================

class TestBasicScoring:

    def test_critical_scores_produce_critical_label(
        self, scorer, critical_scores
    ):
        result = scorer.score(critical_scores)
        assert result.risk_label == "CRITICAL"
        assert result.final_score >= 0.8

    def test_normal_scores_produce_low_label(
        self, scorer, normal_scores
    ):
        result = scorer.score(normal_scores)
        assert result.risk_label in ["LOW", "MEDIUM", "UNKNOWN"]
        assert result.final_score < 0.6

    def test_empty_scores_return_unknown(self, scorer):
        result = scorer.score({})
        assert result.risk_label == "UNKNOWN"
        assert result.final_score == 0.0

    def test_single_model_score(self, scorer):
        scores = {
            "isolation_forest": ModelScore(
                model_name="isolation_forest",
                score=0.9, confidence=1.0
            )
        }
        result = scorer.score(scores)
        assert result.final_score > 0.0
        assert result.models_scored == 1

    def test_score_never_exceeds_one(
        self, scorer, critical_scores
    ):
        result = scorer.score(
            critical_scores, pii_sensitivity="PCI"
        )
        assert result.final_score <= 1.0

    def test_score_never_below_zero(
        self, scorer, normal_scores
    ):
        result = scorer.score(normal_scores)
        assert result.final_score >= 0.0

    def test_models_scored_count(
        self, scorer, critical_scores
    ):
        result = scorer.score(critical_scores)
        assert result.models_scored == len(critical_scores)

    def test_models_available_count(self, scorer):
        result = scorer.score({})
        assert result.models_available == len(MODEL_WEIGHTS)


# ============================================================
# PII MULTIPLIER TESTS
# ============================================================

class TestPIIMultiplier:

    def test_pci_multiplier_elevates_score(
        self, scorer, normal_scores
    ):
        result_no_pii = scorer.score(normal_scores)
        result_pci = scorer.score(
            normal_scores, pii_sensitivity="PCI"
        )
        assert result_pci.final_score > result_no_pii.final_score

    def test_pci_multiplier_is_1_3(
        self, scorer, normal_scores
    ):
        result = scorer.score(
            normal_scores, pii_sensitivity="PCI"
        )
        assert result.pii_multiplier == 1.30

    def test_phi_multiplier_is_1_25(
        self, scorer, normal_scores
    ):
        result = scorer.score(
            normal_scores, pii_sensitivity="PHI"
        )
        assert result.pii_multiplier == 1.25

    def test_pii_multiplier_is_1_2(
        self, scorer, normal_scores
    ):
        result = scorer.score(
            normal_scores, pii_sensitivity="PII"
        )
        assert result.pii_multiplier == 1.20

    def test_none_multiplier_is_1_0(
        self, scorer, normal_scores
    ):
        result = scorer.score(
            normal_scores, pii_sensitivity="NONE"
        )
        assert result.pii_multiplier == 1.0

    def test_pii_elevates_medium_to_critical(
        self, scorer
    ):
        medium_scores = {
            "isolation_forest": ModelScore(
                model_name="isolation_forest",
                score=0.65, confidence=1.0
            ),
            "identity_detector": ModelScore(
                model_name="identity_detector",
                score=0.60, confidence=1.0
            )
        }
        result = scorer.score(
            medium_scores, pii_sensitivity="PCI"
        )
        assert result.final_score > 0.7

    def test_pre_pii_score_recorded(
        self, scorer, normal_scores
    ):
        result = scorer.score(
            normal_scores, pii_sensitivity="PCI"
        )
        assert result.pre_pii_score > 0.0
        assert result.pre_pii_score < result.final_score


# ============================================================
# VERDICT DETERMINATION TESTS
# ============================================================

class TestVerdictDetermination:

    def test_data_exfiltration_verdict(self, scorer):
        scores = {
            "isolation_forest": ModelScore(
                model_name="isolation_forest",
                score=0.85, confidence=1.0
            ),
            "pii_classifier": ModelScore(
                model_name="pii_classifier",
                score=0.95, confidence=1.0
            ),
            "identity_detector": ModelScore(
                model_name="identity_detector",
                score=0.80, confidence=1.0
            )
        }
        result = scorer.score(scores, pii_sensitivity="PCI")
        assert result.verdict == "DATA_EXFILTRATION"

    def test_identity_compromise_verdict(self, scorer):
        scores = {
            "identity_detector": ModelScore(
                model_name="identity_detector",
                score=0.92, confidence=1.0
            ),
            "isolation_forest": ModelScore(
                model_name="isolation_forest",
                score=0.75, confidence=1.0
            )
        }
        result = scorer.score(scores)
        assert result.verdict == "IDENTITY_COMPROMISE"

    def test_dga_verdict(self, scorer):
        scores = {
            "dns_classifier": ModelScore(
                model_name="dns_classifier",
                score=0.95, confidence=1.0
            )
        }
        result = scorer.score(scores)
        assert result.verdict == "DGA_DOMAIN"

    def test_normal_verdict_for_low_scores(
        self, scorer, normal_scores
    ):
        result = scorer.score(normal_scores)
        assert result.verdict in [
            "NORMAL", "ANOMALY_MEDIUM", "ANOMALY_HIGH"
        ]


# ============================================================
# SCORE FROM DICT TESTS
# ============================================================

class TestScoreFromDict:

    def test_score_from_dict_basic(self, scorer):
        result = scorer.score_from_dict({
            "isolation_forest": 0.85,
            "pii_classifier": 0.90
        })
        assert result.final_score > 0.0

    def test_score_from_dict_with_pii(self, scorer):
        result = scorer.score_from_dict(
            {"isolation_forest": 0.70},
            pii_sensitivity="PCI"
        )
        assert result.pii_multiplier == 1.30

    def test_score_from_dict_ignores_none(self, scorer):
        result = scorer.score_from_dict({
            "isolation_forest": 0.85,
            "pii_classifier": None
        })
        assert result.models_scored == 1

    def test_score_from_dict_matches_score_method(
        self, scorer
    ):
        dict_scores = {
            "isolation_forest": 0.80,
            "random_forest": 0.75
        }
        model_scores = {
            name: ModelScore(
                model_name=name,
                score=score,
                confidence=1.0
            )
            for name, score in dict_scores.items()
        }
        result1 = scorer.score_from_dict(dict_scores)
        result2 = scorer.score(model_scores)
        assert abs(
            result1.final_score - result2.final_score
        ) < 0.001


# ============================================================
# EXPLAINABILITY TESTS (SR 11-7)
# ============================================================

class TestExplainability:

    def test_explanation_not_empty(
        self, scorer, critical_scores
    ):
        result = scorer.score(critical_scores)
        assert len(result.explanation) > 0

    def test_explanation_contains_label(
        self, scorer, critical_scores
    ):
        result = scorer.score(critical_scores)
        assert result.risk_label in result.explanation

    def test_highest_model_identified(
        self, scorer, critical_scores
    ):
        result = scorer.score(critical_scores)
        assert result.highest_model != ""
        assert result.highest_score > 0.0

    def test_model_contributions_sum_reasonable(
        self, scorer, critical_scores
    ):
        result = scorer.score(critical_scores)
        total = sum(
            result.model_contributions.values()
        )
        assert total > 0.0

    def test_to_dict_contains_all_fields(
        self, scorer, critical_scores
    ):
        result = scorer.score(critical_scores)
        d = result.to_dict()
        required_fields = [
            "final_score", "risk_label", "verdict",
            "model_scores", "model_contributions",
            "pii_sensitivity", "pii_multiplier",
            "highest_model", "models_scored",
            "explanation"
        ]
        for field_name in required_fields:
            assert field_name in d

    def test_model_scores_in_result(
        self, scorer, critical_scores
    ):
        result = scorer.score(critical_scores)
        assert len(result.model_scores) > 0
        for score in result.model_scores.values():
            assert 0.0 <= score <= 1.0


# ============================================================
# WEIGHT UPDATE TESTS
# ============================================================

class TestWeightUpdate:

    def test_update_weight_rebalances(self, scorer):
        scorer.update_weight("isolation_forest", 0.30)
        total = sum(scorer.weights.values())
        assert abs(total - 1.0) < 0.001

    def test_update_invalid_model_raises(self, scorer):
        with pytest.raises(ValueError):
            scorer.update_weight(
                "nonexistent_model", 0.5
            )

    def test_get_weight(self, scorer):
        weight = scorer.get_weight("isolation_forest")
        assert weight == MODEL_WEIGHTS["isolation_forest"]

    def test_get_weight_unknown_model(self, scorer):
        weight = scorer.get_weight("unknown_model")
        assert weight == 0.0


# ============================================================
# INTEGRATION TEST — SVCBACKUP SCENARIO
# ============================================================

class TestSvcBackupScenario:
    """
    End-to-end test using the svc_backup
    data exfiltration scenario from the demo.
    
    This is the exact scenario presented to BofA.
    """

    def test_svcbackup_attack_detected(self, scorer):
        """
        svc_backup accessed prod-customer-data
        from Tor exit node at 3am.
        Should be CRITICAL DATA_EXFILTRATION.
        """
        scores = {
            "isolation_forest": ModelScore(
                model_name="isolation_forest",
                score=0.90,
                confidence=0.95,
                reason="524MB at 3am from Tor node"
            ),
            "pii_classifier": ModelScore(
                model_name="pii_classifier",
                score=0.95,
                confidence=1.0,
                reason="PCI card data detected"
            ),
            "identity_detector": ModelScore(
                model_name="identity_detector",
                score=0.85,
                confidence=0.90,
                reason="Service account anomaly"
            ),
            "lstm_detector": ModelScore(
                model_name="lstm_detector",
                score=0.82,
                confidence=0.85,
                reason="10-day exfil pattern"
            ),
            "gnn_detector": ModelScore(
                model_name="gnn_detector",
                score=0.91,
                confidence=0.88,
                reason="Tor exit node proximity"
            )
        }

        result = scorer.score(
            scores, pii_sensitivity="PCI"
        )

        assert result.risk_label == "CRITICAL"
        assert result.final_score >= 0.8
        assert result.verdict == "DATA_EXFILTRATION"
        assert result.pii_multiplier == 1.30
        assert result.models_scored == 5

    def test_normal_svcbackup_not_flagged(self, scorer):
        """
        Normal svc_backup backup activity.
        Should be LOW or MEDIUM, not CRITICAL.
        """
        scores = {
            "isolation_forest": ModelScore(
                model_name="isolation_forest",
                score=0.15,
                confidence=0.95,
                reason="Normal 5MB backup"
            ),
            "pii_classifier": ModelScore(
                model_name="pii_classifier",
                score=0.05,
                confidence=1.0,
                reason="Backup data no PII"
            ),
            "identity_detector": ModelScore(
                model_name="identity_detector",
                score=0.10,
                confidence=0.90,
                reason="Normal service account"
            )
        }

        result = scorer.score(
            scores, pii_sensitivity="NONE"
        )

        assert result.risk_label in ["LOW", "UNKNOWN"]
        assert result.final_score < 0.4
        assert result.verdict == "NORMAL"