"""
Layer 2 — ML Processing Engine
DNS NLP Model — DGA Classifier

This module trains and deploys a DGA detection model
that scores DNS events for malicious domain patterns.

Why This Model Uses Different Techniques:
    Previous models used numerical features directly.
    This model analyzes TEXT — domain name strings.

    The feature extraction in dns_features.py converts
    text to numerical features that XGBoost can learn.

    This is a simplified NLP pipeline:
        Text → Feature Extraction → ML Classifier

    Full BERT-based NLP comes in the next iteration
    when you add email content analysis for phishing.

Training Data:
    Synthetic data encoding known DGA patterns:
    - Random character DGA domains
    - Known malware family patterns
    - Dynamic DNS abuse patterns
    - Legitimate domain patterns for contrast

Production Path:
    Augment with real DGA feeds:
    - DGArchive (largest DGA repository)
    - Bambenek Consulting DGA feeds
    - Your own collected SOC data

ATT&CK Coverage:
    T1568.002  Dynamic Resolution: DGA
    T1071.001  Application Layer Protocol: Web
    T1071.004  Application Layer Protocol: DNS
"""

import logging
import pickle
import numpy as np
import mlflow
import mlflow.sklearn
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from xgboost import XGBClassifier

from layer2_ml.nlp.dns_features import (
    DNSFeatureExtractor,
    DYNAMIC_DNS_PROVIDERS,
    SUSPICIOUS_TLDS,
    KNOWN_LEGITIMATE
)

logger = logging.getLogger(__name__)


@dataclass
class DNSDetectionResult:
    """
    Result of scoring a single DNS event.

    Consumed by Layer2Router which adds dns
    category routing to NetworkIntrusionDetector
    AND this DNSClassifier.

    Fields consumed by each layer:
        is_dga          → Layer 3 knowledge graph
        risk_score      → Layer 3 threat weighting
        dga_family      → Layer 3 campaign clustering
        explanation     → Layer 5 analyst display
        indicators      → Layer 5 analyst display
    """
    # Core result
    is_dga: bool
    risk_score: float
    confidence: str

    # DGA context
    dga_indicators: list
    dga_family: str        # likely DGA family if known
    explanation: str

    # Domain preserved for investigation
    domain: str
    requesting_process: str

    # MLOps metadata
    model_name: str
    model_version: str
    scored_at: str
    inference_time_ms: float

    def to_dict(self) -> dict:
        return {
            "is_dga": self.is_dga,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "dga_indicators": self.dga_indicators,
            "dga_family": self.dga_family,
            "explanation": self.explanation,
            "domain": self.domain,
            "requesting_process": (
                self.requesting_process
            ),
            "model_name": self.model_name,
            "scored_at": self.scored_at
        }


# ============================================================
# TRAINING DATA GENERATION
# ============================================================

class DGATrainingDataGenerator:
    """
    Generates synthetic training data for DGA detection.

    Creates realistic DGA and legitimate domain samples
    encoding known malware family patterns.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.extractor = DNSFeatureExtractor()
        np.random.seed(random_state)

    def generate(
        self,
        n_legitimate: int = 5000,
        n_dga: int = 2000
    ):
        """
        Generate balanced training dataset.

        Returns:
            X: Feature matrix
            y: Labels (0=legitimate, 1=DGA)
            domains: Domain strings for analysis
        """
        logger.info(
            f"Generating {n_legitimate} legitimate "
            f"and {n_dga} DGA samples"
        )

        legit_domains = self._generate_legitimate(
            n_legitimate
        )
        dga_domains = self._generate_dga(n_dga)

        all_domains = legit_domains + dga_domains
        labels = (
            [0] * n_legitimate + [1] * n_dga
        )

        # Extract features
        X_list = []
        y_list = []
        domain_list = []

        for domain, label in zip(all_domains, labels):
            process = self._get_process_for_domain(
                domain, label
            )
            features = self.extractor.extract(
                domain, process
            )
            if features is not None:
                X_list.append(features)
                y_list.append(label)
                domain_list.append(domain)

        X = np.array(X_list)
        y = np.array(y_list, dtype=np.int32)

        # Shuffle
        indices = np.random.permutation(len(y))
        return X[indices], y[indices], [
            domain_list[i] for i in indices
        ]

    def _generate_legitimate(
        self,
        n_samples: int
    ) -> list:
        """Generate realistic legitimate domain names"""
        domains = []

        # Known legitimate domains
        known = list(KNOWN_LEGITIMATE)
        for _ in range(n_samples // 5):
            domains.append(
                np.random.choice(known)
            )

        # Realistic corporate domains
        words = [
            "mail", "vpn", "remote", "portal",
            "intranet", "files", "share", "apps",
            "login", "auth", "sso", "api",
            "docs", "help", "support", "hr",
            "finance", "sales", "marketing", "it"
        ]
        companies = [
            "acme", "globex", "initech", "umbrella",
            "wayne", "stark", "oscorp", "cyberdyne"
        ]
        tlds = [".com", ".net", ".org", ".io"]

        for _ in range(n_samples - len(domains)):
            word = np.random.choice(words)
            company = np.random.choice(companies)
            tld = np.random.choice(tlds)
            domain = f"{word}.{company}{tld}"
            domains.append(domain)

        return domains[:n_samples]

    def _generate_dga(self, n_samples: int) -> list:
        """
        Generate realistic DGA domain samples.

        Each pattern mimics a real malware family's
        DGA algorithm characteristics.
        """
        domains = []
        per_pattern = n_samples // 6

        # Pattern 1: Pure random — Conficker-like
        # High entropy, no vowels, random length
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        for _ in range(per_pattern):
            length = np.random.randint(8, 16)
            sld = "".join(
                np.random.choice(list(chars))
                for _ in range(length)
            )
            tld = np.random.choice([
                ".com", ".net", ".xyz", ".top"
            ])
            domains.append(f"{sld}{tld}")

        # Pattern 2: Dynamic DNS abuse — most common
        # Your test domain xjf8k2mp.duckdns.org
        dyn_providers = list(DYNAMIC_DNS_PROVIDERS)
        consonants = "bcdfghjklmnpqrstvwxyz"
        digits = "0123456789"
        for _ in range(per_pattern):
            length = np.random.randint(6, 12)
            sld_chars = list(consonants) + list(digits)
            sld = "".join(
                np.random.choice(sld_chars)
                for _ in range(length)
            )
            provider = np.random.choice(dyn_providers)
            domains.append(f"{sld}.{provider}")

        # Pattern 3: Suspicious TLD with random SLD
        # Low cost TLDs used for malware campaigns
        susp_tlds = [
            ".xyz", ".top", ".club",
            ".online", ".site", ".pw"
        ]
        for _ in range(per_pattern):
            length = np.random.randint(6, 14)
            sld = "".join(
                np.random.choice(list(chars))
                for _ in range(length)
            )
            tld = np.random.choice(susp_tlds)
            domains.append(f"{sld}{tld}")

        # Pattern 4: Hex-like DGA — Cryptolocker-like
        # Domains that look like hex strings
        hex_chars = "0123456789abcdef"
        for _ in range(per_pattern):
            length = np.random.randint(12, 20)
            sld = "".join(
                np.random.choice(list(hex_chars))
                for _ in range(length)
            )
            tld = np.random.choice([".com", ".net"])
            domains.append(f"{sld}{tld}")

        # Pattern 5: Typosquatting — brand impersonation
        # Looks legitimate but slightly wrong
        brands = [
            "micros0ft", "g00gle", "paypa1",
            "arnazon", "facebok", "appie",
            "windovvs", "0utlook", "linkedln"
        ]
        for _ in range(per_pattern):
            brand = np.random.choice(brands)
            suffix = np.random.choice([
                "-update", "-secure", "-login",
                "-verify", "-cdn", "-service"
            ])
            tld = np.random.choice([
                ".com", ".net", ".xyz"
            ])
            domains.append(f"{brand}{suffix}{tld}")

        # Pattern 6: Long subdomain DGA
        # Multiple random subdomains
        remaining = n_samples - len(domains)
        for _ in range(remaining):
            sub1_len = np.random.randint(6, 10)
            sub2_len = np.random.randint(4, 8)
            sub1 = "".join(
                np.random.choice(list(chars))
                for _ in range(sub1_len)
            )
            sub2 = "".join(
                np.random.choice(list(chars))
                for _ in range(sub2_len)
            )
            tld = np.random.choice([
                ".com", ".net", ".xyz"
            ])
            domains.append(f"{sub1}.{sub2}{tld}")

        return domains[:n_samples]

    def _get_process_for_domain(
        self,
        domain: str,
        label: int
    ) -> str:
        """
        Assign realistic requesting process.
        DGA requests often come from system processes.
        """
        if label == 1:
            # DGA — more likely system processes
            processes = [
                "svchost.exe", "powershell.exe",
                "cmd.exe", "explorer.exe",
                "chrome.exe"
            ]
            weights = [0.4, 0.2, 0.1, 0.1, 0.2]
        else:
            # Legitimate — mostly browsers
            processes = [
                "chrome.exe", "firefox.exe",
                "msedge.exe", "outlook.exe",
                "explorer.exe"
            ]
            weights = [0.4, 0.2, 0.2, 0.1, 0.1]

        return np.random.choice(processes, p=weights)


# ============================================================
# CLASSIFIER
# ============================================================

class DNSClassifier:
    """
    Trains and deploys DGA detection model.

    Combines feature extraction from DNSFeatureExtractor
    with XGBoost classification and MLflow tracking.
    """

    def __init__(
        self,
        experiment_name: str = "dns_dga_detection",
        model_save_dir: str = "models/dns_classifier"
    ):
        self.experiment_name = experiment_name
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(
            parents=True, exist_ok=True
        )
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = DNSFeatureExtractor()
        self.best_f1 = 0.0
        self.model_name = "unknown"
        self.model_version = "1.0.0"

        # Performance tracking
        self.total_scored = 0
        self.total_dga_detected = 0
        self.total_inference_ms = 0.0

        mlflow.set_experiment(experiment_name)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """
        Train DGA classifier with MLflow tracking.
        Trains both RF and XGBoost, returns best.
        """
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        rf_metrics = self._train_rf(
            X_train_s, y_train,
            X_test_s, y_test
        )
        xgb_metrics = self._train_xgb(
            X_train_s, y_train,
            X_test_s, y_test
        )

        best = (
            "XGBoost"
            if xgb_metrics["f1"] > rf_metrics["f1"]
            else "Random Forest"
        )

        logger.info(f"Best DNS model: {best}")
        return {
            "random_forest": rf_metrics,
            "xgboost": xgb_metrics,
            "best": best
        }

    def _train_rf(
        self, X_train, y_train, X_test, y_test
    ) -> dict:
        """Train Random Forest with MLflow"""
        config = {
            "n_estimators": 100,
            "max_depth": 15,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42
        }

        with mlflow.start_run(run_name="dns_rf"):
            mlflow.log_params(config)
            mlflow.log_param("model_type", "RandomForest")

            model = RandomForestClassifier(**config)
            model.fit(X_train, y_train)

            metrics = self._evaluate(
                model, X_test, y_test, "RF"
            )
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "dns_rf")

            if metrics["f1"] > self.best_f1:
                self.best_f1 = metrics["f1"]
                self.model = model
                self.model_name = "dns_random_forest"

        return metrics

    def _train_xgb(
        self, X_train, y_train, X_test, y_test
    ) -> dict:
        """Train XGBoost with MLflow"""
        benign = np.sum(y_train == 0)
        dga = np.sum(y_train == 1)
        spw = benign / max(dga, 1)

        config = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": spw,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": 0
        }

        with mlflow.start_run(run_name="dns_xgboost"):
            mlflow.log_params(config)
            mlflow.log_param("model_type", "XGBoost")

            model = XGBClassifier(**config)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

            metrics = self._evaluate(
                model, X_test, y_test, "XGBoost"
            )
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(
                model, "dns_xgboost"
            )

            if metrics["f1"] > self.best_f1:
                self.best_f1 = metrics["f1"]
                self.model = model
                self.model_name = "dns_xgboost"

        return metrics

    def _evaluate(
        self, model, X_test, y_test, name
    ) -> dict:
        """Evaluate with security metrics"""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        auc = roc_auc_score(y_test, y_prob)

        tn = np.sum((y_test == 0) & (y_pred == 0))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fpr = fp / max((tn + fp), 1)

        metrics = {
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "roc_auc": round(auc, 4),
            "false_positive_rate": round(fpr, 4)
        }

        logger.info(f"\nDNS {name} Results:")
        logger.info(f"  F1:      {f1:.4f}")
        logger.info(f"  Recall:  {recall:.4f}")
        logger.info(f"  FPR:     {fpr:.4f}")

        return metrics

    def save_model(self) -> str:
        """Save model and scaler to disk"""
        if self.model is None:
            raise ValueError("No model trained yet")

        model_path = (
            self.model_save_dir / "best_model.pkl"
        )
        scaler_path = (
            self.model_save_dir / "scaler.pkl"
        )

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        logger.info(f"DNS model saved: {model_path}")
        return str(model_path)

    def load_model(
        self,
        model_path: str,
        scaler_path: str = None
    ) -> None:
        """Load trained model from disk"""
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"DNS model not found: {model_path}"
            )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.model_name = model_path.stem

        if scaler_path:
            scaler_path = Path(scaler_path)
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)

        logger.info(
            f"DNS model loaded: {model_path}"
        )

    def score_domain(
        self,
        domain: str,
        requesting_process: str = ""
    ) -> Optional[DNSDetectionResult]:
        """
        Score a domain name string directly.

        Args:
            domain: Domain name to score
            requesting_process: Process that queried it

        Returns:
            DNSDetectionResult or None
        """
        if not domain:
            return None

        start_time = datetime.now(timezone.utc)

        features = self.feature_extractor.extract(
            domain, requesting_process
        )

        if features is None:
            return None

        try:
            if self.model is not None:
                scaled = self.scaler.transform(
                    features.reshape(1, -1)
                )
                prob = self.model.predict_proba(
                    scaled
                )[0][1]
            else:
                # Rule-based fallback
                prob = self._rule_based_score(
                    domain, requesting_process
                )

            is_dga = bool(prob >= 0.5)

            end_time = datetime.now(timezone.utc)
            inference_ms = (
                (end_time - start_time).microseconds
                / 1000
            )

            result = self._build_result(
                domain=domain,
                requesting_process=requesting_process,
                prob=prob,
                is_dga=is_dga,
                features=features,
                inference_ms=inference_ms
            )

            self.total_scored += 1
            if is_dga:
                self.total_dga_detected += 1
            self.total_inference_ms += inference_ms

            return result

        except Exception as e:
            logger.error(
                f"DNS scoring failed for {domain}: {e}"
            )
            return None

    def score_ecs_event(
        self,
        ecs_event
    ) -> Optional[DNSDetectionResult]:
        """
        Score a normalized ECS DNS event.

        Called by Layer2Router for dns category events.
        """
        if ecs_event is None:
            return None

        if not hasattr(ecs_event, "event"):
            return None

        if ecs_event.event.category != "dns":
            return None

        domain = ""
        if (ecs_event.destination and
                ecs_event.destination.domain):
            domain = ecs_event.destination.domain

        if not domain:
            return None

        process_name = ""
        if ecs_event.process:
            process_name = ecs_event.process.name or ""

        return self.score_domain(domain, process_name)

    def get_performance_stats(self) -> dict:
        """Return performance statistics for MLOps"""
        detection_rate = (
            self.total_dga_detected /
            max(self.total_scored, 1)
        )
        avg_latency = (
            self.total_inference_ms /
            max(self.total_scored, 1)
        )

        return {
            "total_scored": self.total_scored,
            "total_dga_detected": self.total_dga_detected,
            "detection_rate": round(detection_rate, 4),
            "avg_inference_ms": round(avg_latency, 2),
            "model_name": self.model_name
        }

    # ============================================================
    # PRIVATE METHODS
    # ============================================================

    def _rule_based_score(
        self,
        domain: str,
        requesting_process: str
    ) -> float:
        """
        Rule-based DGA scoring fallback.
        Used when no model is loaded.
        """
        score = 0.0
        domain_lower = domain.lower()

        # Dynamic DNS is high risk
        if any(dyn in domain_lower
               for dyn in DYNAMIC_DNS_PROVIDERS):
            score += 0.4

        # Suspicious TLD
        for tld in SUSPICIOUS_TLDS:
            if domain_lower.endswith(tld):
                score += 0.2
                break

        # Known legitimate
        if any(legit in domain_lower
               for legit in KNOWN_LEGITIMATE):
            score = max(0, score - 0.5)

        # High entropy SLD
        parts = domain_lower.split(".")
        sld = parts[0] if parts else domain_lower
        vowels = sum(1 for c in sld if c in "aeiou")
        if len(sld) > 0:
            if vowels / len(sld) < 0.15:
                score += 0.3

        # Suspicious requesting process
        suspicious_procs = {
            "svchost.exe", "lsass.exe",
            "csrss.exe", "winlogon.exe"
        }
        if requesting_process.lower() in suspicious_procs:
            score += 0.2

        return min(score, 1.0)

    def _build_result(
        self,
        domain: str,
        requesting_process: str,
        prob: float,
        is_dga: bool,
        features: np.ndarray,
        inference_ms: float
    ) -> DNSDetectionResult:
        """Build DNSDetectionResult from scored domain"""

        indicators = self._identify_indicators(
            domain, requesting_process, features
        )

        dga_family = self._identify_dga_family(
            domain, features
        )

        explanation = self._generate_explanation(
            domain, prob, is_dga,
            indicators, dga_family
        )

        return DNSDetectionResult(
            is_dga=is_dga,
            risk_score=round(float(prob), 4),
            confidence=self._score_to_confidence(prob),
            dga_indicators=indicators,
            dga_family=dga_family,
            explanation=explanation,
            domain=domain,
            requesting_process=requesting_process,
            model_name=self.model_name,
            model_version=self.model_version,
            scored_at=datetime.now(
                timezone.utc
            ).isoformat(),
            inference_time_ms=round(inference_ms, 2)
        )

    def _identify_indicators(
        self,
        domain: str,
        process: str,
        features: np.ndarray
    ) -> list:
        """Identify which DGA indicators fired"""
        indicators = []
        feature_names = (
            self.feature_extractor.get_feature_names()
        )

        feature_dict = dict(
            zip(feature_names, features)
        )

        if feature_dict.get("is_dynamic_dns", 0):
            indicators.append(
                "Dynamic DNS provider detected"
            )

        if feature_dict.get("sld_entropy", 0) > 3.5:
            entropy = feature_dict.get(
                "sld_entropy", 0
            )
            indicators.append(
                f"High domain entropy: {entropy:.2f}"
            )

        if feature_dict.get("is_suspicious_tld", 0):
            indicators.append(
                "Suspicious TLD detected"
            )

        if feature_dict.get(
            "requesting_process_suspicious", 0
        ):
            indicators.append(
                f"System process making DNS request: "
                f"{process}"
            )

        if feature_dict.get(
            "is_suspicious_combination", 0
        ):
            indicators.append(
                "High-risk combination: system process "
                "+ DGA-like domain"
            )

        vowel_ratio = feature_dict.get(
            "vowel_ratio", 0.5
        )
        if vowel_ratio < 0.15:
            indicators.append(
                f"Very low vowel ratio: {vowel_ratio:.2f}"
                f" (random characters)"
            )

        return indicators

    def _identify_dga_family(
        self,
        domain: str,
        features: np.ndarray
    ) -> str:
        """
        Attempt to identify likely DGA family.
        Based on domain characteristics.
        """
        feature_names = (
            self.feature_extractor.get_feature_names()
        )
        feature_dict = dict(
            zip(feature_names, features)
        )

        if feature_dict.get("is_dynamic_dns", 0):
            return "dynamic_dns_abuse"

        if feature_dict.get("looks_like_hex", 0):
            return "hex_dga"

        sld_len = feature_dict.get("sld_length", 0)
        entropy = feature_dict.get("sld_entropy", 0)

        if entropy > 4.0 and sld_len > 12:
            return "high_entropy_dga"

        if entropy > 3.5:
            return "random_dga"

        if feature_dict.get("has_action_word", 0):
            return "typosquatting_c2"

        return "unknown_dga"

    def _generate_explanation(
        self,
        domain: str,
        prob: float,
        is_dga: bool,
        indicators: list,
        dga_family: str
    ) -> str:
        """Generate analyst-friendly explanation"""

        if not is_dga:
            return (
                f"Domain {domain} appears legitimate "
                f"(risk: {prob:.2f}). "
                f"No significant DGA indicators."
            )

        explanation = (
            f"DGA DOMAIN DETECTED: {domain} "
            f"(risk: {prob:.2f}). "
            f"Family: {dga_family}. "
        )

        if indicators:
            explanation += (
                f"Indicators: "
                f"{'; '.join(indicators[:3])}."
            )

        return explanation

    def _score_to_confidence(self, prob: float) -> str:
        """Convert probability to confidence label"""
        if prob >= 0.80:
            return "HIGH"
        elif prob >= 0.50:
            return "MEDIUM"
        else:
            return "LOW"