"""
Layer 2 — ML Detection
Model Catalog

The single source of truth describing every detection
model in Layer 2: its algorithm, tech stack, learning
type (supervised/unsupervised), what it detects, how it
scores, what it catches that rules miss, MITRE
relevance, and honest build status.

Drives both the /api/models endpoint and the live
Layer 2 dashboard.

BUILD STATUS values are deliberately honest:
    "trained"     - a real trained ML model
    "deep"        - a deep-learning model (PyTorch)
    "built"       - working detection logic, tested
    "model-ready" - feature layer + explainable scoring,
                    pluggable trained model
"""

# What Layer 2 does, for the dashboard header.
LAYER_INFO = {
    "layer": 2,
    "name": "ML Detection & Scoring",
    "tagline": (
        "Specialized models score every event for "
        "threat — supervised models catch known attack "
        "patterns, unsupervised models catch novel ones "
        "that signature rules miss."
    ),
    "what_it_does": (
        "Layer 2 takes a normalized event from the "
        "ingestion layer and scores it for threat. "
        "Each model is a specialist for a different "
        "attack pattern. An ensemble combines their "
        "scores into one explainable risk score."
    ),
    "why_it_matters": (
        "Rules catch known threats. Machine learning "
        "catches novel ones — behavior that has no "
        "signature yet. We use both supervised and "
        "unsupervised models so we cover known attacks "
        "and unknown unknowns."
    ),
    "how_scores_combine": (
        "Each model outputs an independent score. The "
        "ensemble scorer weights and combines them into "
        "a single 0-1 risk score, so one model's blind "
        "spot is covered by the others. Every score is "
        "explainable and maps to MITRE ATT&CK."
    ),
}


# Status metadata for UI badges
STATUS_META = {
    "trained": {
        "label": "Trained model",
        "color": "#1D9E75",
    },
    "deep": {
        "label": "Deep learning",
        "color": "#7C5CDB",
    },
    "built": {
        "label": "Built + tested",
        "color": "#378ADD",
    },
    "model-ready": {
        "label": "Model-ready",
        "color": "#C28A2B",
    },
}

# Learning-type metadata for UI
LEARNING_META = {
    "supervised": {
        "label": "Supervised",
        "color": "#3DA5D9",
    },
    "unsupervised": {
        "label": "Unsupervised",
        "color": "#E0A458",
    },
    "meta": {
        "label": "Meta / ensemble",
        "color": "#9A8CDB",
    },
}


# The model catalog — one entry per detection model.
MODEL_CATALOG = [
    {
        "id": "isolation_forest",
        "name": "Isolation Forest",
        "category": "Anomaly detection",
        "algorithm": "Isolation Forest (scikit-learn)",
        "tech_stack": ["scikit-learn", "NumPy", "MLflow"],
        "learning_type": "unsupervised",
        "icon": "ti-tree",
        "detects": (
            "Statistical outliers — events that don't "
            "fit normal patterns of behavior."
        ),
        "how_it_scores": (
            "Isolates each event by randomly "
            "partitioning features. Anomalies are "
            "isolated in fewer splits, so they score "
            "higher. No labeled attacks needed."
        ),
        "rules_miss": (
            "Catches never-before-seen anomalies that "
            "no rule was written for — unknown unknowns."
        ),
        "mitre": "TA0010 (Exfiltration), TA0007 (Discovery)",
        "status": "trained",
    },
    {
        "id": "autoencoder",
        "name": "Autoencoder",
        "category": "Anomaly detection",
        "algorithm": "Neural Autoencoder (deep learning)",
        "tech_stack": ["PyTorch", "NumPy"],
        "learning_type": "unsupervised",
        "icon": "ti-binary-tree",
        "detects": (
            "Subtle anomalies in high-dimensional "
            "behavior that simpler models miss."
        ),
        "how_it_scores": (
            "Learns to reconstruct normal events. When "
            "an event can't be reconstructed well, the "
            "reconstruction error is high — flagging it "
            "as anomalous."
        ),
        "rules_miss": (
            "Detects complex, multi-feature deviations "
            "from normal that have no single rule."
        ),
        "mitre": "TA0010 (Exfiltration)",
        "status": "deep",
    },
    {
        "id": "intrusion_detection",
        "name": "Intrusion Detection",
        "category": "Network",
        "algorithm": "Random Forest + XGBoost",
        "tech_stack": ["XGBoost", "scikit-learn", "pandas", "MLflow"],
        "learning_type": "supervised",
        "icon": "ti-network",
        "detects": (
            "Network-based attacks in flow data — "
            "port scans, DoS, brute force, infiltration."
        ),
        "how_it_scores": (
            "Trains two models on labeled network flows "
            "and keeps the higher-F1 performer (usually "
            "XGBoost). Classifies each flow as benign or "
            "an attack class."
        ),
        "rules_miss": (
            "Learns attack patterns from data rather "
            "than fixed thresholds, catching variants "
            "that evade static signatures."
        ),
        "mitre": "T1046 (Network Discovery), T1498 (DoS)",
        "status": "trained",
    },
    {
        "id": "dns_dga",
        "name": "DNS / DGA Classifier",
        "category": "NLP",
        "algorithm": "Random Forest + XGBoost on NLP features",
        "tech_stack": ["XGBoost", "scikit-learn", "MLflow"],
        "learning_type": "supervised",
        "icon": "ti-world-search",
        "detects": (
            "Malware command-and-control domains "
            "generated by algorithms (DGA)."
        ),
        "how_it_scores": (
            "Treats domain names as text — extracts "
            "character entropy, n-gram patterns, length "
            "and consonant ratios — then classifies. "
            "Random-looking domains score high."
        ),
        "rules_miss": (
            "Blocklists can't keep up with thousands of "
            "new algorithmically-generated domains a "
            "day; this generalizes to unseen ones."
        ),
        "mitre": "T1568.002 (Domain Generation Algorithms)",
        "status": "trained",
    },
    {
        "id": "lstm_attention",
        "name": "LSTM + Attention",
        "category": "Sequence",
        "algorithm": "LSTM with attention (PyTorch)",
        "tech_stack": ["PyTorch", "NumPy"],
        "learning_type": "supervised",
        "icon": "ti-timeline",
        "detects": (
            "Slow, multi-step attacks that unfold over "
            "time — low-and-slow data exfiltration."
        ),
        "how_it_scores": (
            "Reads a sequence of events in order and "
            "learns temporal patterns. Attention "
            "highlights which steps in the sequence "
            "drive the risk."
        ),
        "rules_miss": (
            "Each step looks benign alone; only the "
            "sequence reveals the attack. Point-in-time "
            "rules can't see across time."
        ),
        "mitre": "T1030 (Data Transfer Size Limits)",
        "status": "deep",
    },
    {
        "id": "gnn",
        "name": "GNN Detector",
        "category": "Graph",
        "algorithm": "Graph Neural Network",
        "tech_stack": ["PyTorch", "NetworkX", "NumPy"],
        "learning_type": "supervised",
        "icon": "ti-affiliate",
        "detects": (
            "Relationship-based attacks — lateral "
            "movement, privilege chains, unusual "
            "access paths."
        ),
        "how_it_scores": (
            "Models entities (users, hosts, resources) "
            "as a graph and learns from how they "
            "connect. Flags suspicious relationship "
            "patterns a flat model can't see."
        ),
        "rules_miss": (
            "Captures the connections between events — "
            "the attack path — not just individual "
            "events in isolation."
        ),
        "mitre": "T1021 (Lateral Movement), T1078 (Valid Accounts)",
        "status": "built",
    },
    {
        "id": "identity_threat",
        "name": "Identity Threat Detector",
        "category": "Identity",
        "algorithm": "Behavioral analytics",
        "tech_stack": ["scikit-learn", "pandas"],
        "learning_type": "unsupervised",
        "icon": "ti-user-shield",
        "detects": (
            "Compromised accounts and insider threats — "
            "behavior that deviates from a user's norm."
        ),
        "how_it_scores": (
            "Builds a behavioral baseline per identity "
            "(times, locations, resources) and scores "
            "deviations — impossible travel, off-hours "
            "privileged access, access spikes."
        ),
        "rules_miss": (
            "What's normal for one user is anomalous for "
            "another; per-identity baselines beat global "
            "thresholds."
        ),
        "mitre": "T1078 (Valid Accounts), T1098 (Account Manipulation)",
        "status": "built",
    },
    {
        "id": "pii_classifier",
        "name": "PII Classifier",
        "category": "Classification",
        "algorithm": "Supervised classifier",
        "tech_stack": ["scikit-learn"],
        "learning_type": "supervised",
        "icon": "ti-file-shield",
        "detects": (
            "Sensitive data — PII, PCI, PHI — in events "
            "and data flows."
        ),
        "how_it_scores": (
            "Classifies content to identify regulated "
            "data, raising the risk of events that move "
            "or expose it."
        ),
        "rules_miss": (
            "Generalizes beyond fixed regex patterns to "
            "catch sensitive data in varied formats."
        ),
        "mitre": "T1530 (Data from Cloud Storage)",
        "status": "built",
    },
    {
        "id": "malware_classifier",
        "name": "Malware Classifier",
        "category": "Endpoint",
        "algorithm": "Supervised classifier on process features",
        "tech_stack": ["scikit-learn", "pickle"],
        "learning_type": "supervised",
        "icon": "ti-bug",
        "detects": (
            "Malicious processes and binaries by their "
            "behavioral features."
        ),
        "how_it_scores": (
            "Extracts process features and classifies "
            "them as benign or malicious based on "
            "learned patterns."
        ),
        "rules_miss": (
            "Detects malicious behavior even when the "
            "file hash is unknown to signature databases."
        ),
        "mitre": "T1204 (User Execution), T1059 (Command Execution)",
        "status": "built",
    },
    {
        "id": "phishing",
        "name": "Phishing Detector",
        "category": "Email",
        "algorithm": "Feature scoring + pluggable model",
        "tech_stack": ["Python", "scikit-learn (pluggable)"],
        "learning_type": "supervised",
        "icon": "ti-mail-x",
        "detects": (
            "Phishing emails — lookalike domains, sender "
            "spoofing, urgency, dangerous attachments."
        ),
        "how_it_scores": (
            "Extracts URL, sender, content and "
            "attachment features (homoglyph domains, "
            "display-name mismatch, urgency keywords) "
            "and scores them explainably; a trained "
            "model can plug in on top."
        ),
        "rules_miss": (
            "Weighs many weak signals together to catch "
            "novel phishing that no single keyword rule "
            "would flag."
        ),
        "mitre": "T1566 (Phishing)",
        "status": "model-ready",
    },
    {
        "id": "ensemble",
        "name": "Ensemble Scorer",
        "category": "Ensemble",
        "algorithm": "Weighted score combination",
        "tech_stack": ["NumPy", "Python"],
        "learning_type": "meta",
        "icon": "ti-stack-2",
        "detects": (
            "The final verdict — combines every model "
            "into one explainable risk score."
        ),
        "how_it_scores": (
            "Weights and combines the individual model "
            "scores so one model's blind spot is covered "
            "by the others, producing a single 0-1 "
            "score with the reasons attached."
        ),
        "rules_miss": (
            "No single model is right every time; the "
            "ensemble is more robust than any one alone."
        ),
        "mitre": "Aggregates all techniques",
        "status": "built",
    },
]


def get_models() -> list:
    """Return the full model catalog."""
    return MODEL_CATALOG


def get_model(model_id: str) -> dict:
    """Return a single model entry by id."""
    for m in MODEL_CATALOG:
        if m["id"] == model_id:
            return m
    return {}


def get_categories() -> list:
    """Return models grouped by category."""
    cats = {}
    for m in MODEL_CATALOG:
        cats.setdefault(m["category"], []).append(m)
    return [
        {"category": c, "models": ms}
        for c, ms in cats.items()
    ]


def catalog_stats() -> dict:
    """Summary counts for the catalog."""
    by_status = {}
    by_learning = {}
    for m in MODEL_CATALOG:
        by_status[m["status"]] = (
            by_status.get(m["status"], 0) + 1
        )
        lt = m.get("learning_type", "")
        if lt:
            by_learning[lt] = by_learning.get(lt, 0) + 1
    detection_models = [
        m for m in MODEL_CATALOG
        if m["id"] != "ensemble"
    ]
    return {
        "total_models": len(detection_models),
        "by_status": by_status,
        "by_learning": by_learning,
    }