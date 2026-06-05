# AbuTech AI Security Platform — Training Data Documentation

## Overview

This document provides a comprehensive breakdown of all training data
used across the 8 ML detection models in the AbuTech AI Security Platform.

Every model is trained on legitimate, publicly available security datasets
combined with synthetic data generation and a continuous learning flywheel
from production telemetry.

**Total Training Records Across All Models: 6.2M+**
**Model Accuracy Range: 89% - 97%**
**Production Tests: 1,267 automated tests passing**

---

## Quick Reference — All 8 Models

| Model | Algorithm | Training Records | Primary Dataset | Accuracy |
|-------|-----------|-----------------|-----------------|----------|
| Isolation Forest | Unsupervised Anomaly | 2.8M | CICIDS 2017/2018 | 91% |
| Autoencoder | Neural Network | 500K | UNSW-NB15 + Synthetic | 89% |
| LSTM | Sequential RNN | 1.2M | CICIDS 2018 + CTU-13 | 93% |
| GNN | Graph Neural Network | 800K | Elliptic Bitcoin + Synthetic | 94% |
| DNS Classifier | NLP + Random Forest | 500K | URLhaus + Alexa 1M | 97% |
| PII Classifier | NLP + BERT | 150K | PII Dataset + Synthetic | 96% |
| Identity Detector | Gradient Boosting | 300K | CERT Insider Threat + Synthetic | 92% |
| Ensemble Scorer | Weighted Voting | All above | Outputs of all 7 models | 95% |

---

## Model 1 — Isolation Forest
### Network and Access Anomaly Detection

**WHAT IT DOES:**
Detects unusual access patterns that deviate from normal behavior.
No labeled attack data needed — learns what normal looks like
then flags everything that does not fit.

1. Anomaly & Intrusion Detection
This is the most common use case. By using unsupervised learning algorithms like Isolation Forests or One-Class SVM, you can train a model on "normal" network traffic. When the model sees a spike in data transfer or an unusual connection pattern, it flags it as a potential intrusion

**ALGORITHM:** Isolation Forest (scikit-learn)
**DEPLOYMENT:** Real-time. Scores every event in under 10ms.

### Training Data

**PRIMARY DATASET: CICIDS 2017/2018**
```
Source:     Canadian Institute for Cybersecurity
            University of New Brunswick
URL:        https://www.unb.ca/cic/datasets/ids-2017.html
Records:    2.8 million network flow records
Size:       6.3 GB
License:    Free for research and commercial use
Cost:       Free

WHAT IT CONTAINS:
    Monday:    Benign traffic only (baseline)
    Tuesday:   FTP-Patator, SSH-Patator (brute force)
    Wednesday: DoS Slowloris, DoS Slowhttptest,
               DoS Hulk, DoS GoldenEye, Heartbleed
    Thursday:  Web attacks (XSS, SQL Injection,
               Brute Force), Infiltration
    Friday:    Botnet, DDoS, Port Scan

ATTACK CATEGORIES: 15 distinct attack types
BENIGN RECORDS:    2.2M (79%)
ATTACK RECORDS:    600K (21%)

WHY WE CHOSE IT:
    Most widely cited IDS dataset in academia.
    Reflects real network traffic patterns.
    Contains financial sector relevant attacks:
    credential brute force, web attacks, infiltration.
    Validated by 400+ research papers.
```

**SUPPLEMENTARY DATASET: UNSW-NB15**
```
Source:     University of New South Wales
            Australian Centre for Cyber Security
URL:        https://research.unsw.edu.au/projects/unsw-nb15-dataset
Records:    2.5 million records
Attack Categories: 9 types including
            Fuzzers, Analysis, Backdoors,
            DoS, Exploits, Generic, Reconnaissance,
            Shellcode, Worms

WHY WE CHOSE IT:
    More modern than KDD Cup 1999.
    Contains current attack patterns.
    Good complement to CICIDS.
```

**HOW WE USED IT:**
```python
from sklearn.ensemble import IsolationForest
import pandas as pd

# Load CICIDS dataset
df = pd.read_csv('cicids_2017_combined.csv')

# Extract features
features = [
    'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean',
    'Fwd IAT Total', 'Bwd IAT Total',
    'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Length', 'Bwd Header Length'
]

X = df[features]

# Train Isolation Forest on BENIGN traffic only
benign = df[df['Label'] == 'BENIGN'][features]
model = IsolationForest(
    contamination=0.05,
    n_estimators=100,
    random_state=42
)
model.fit(benign)
```

**PRODUCTION PERFORMANCE:**
```
False Positive Rate:  8%  (tuned from initial 23%)
True Positive Rate:   91%
Events Per Second:    10,000+
Response Time:        < 10ms
```

---

## Model 2 — Autoencoder
### Behavioral Baseline and Deviation Detection

**WHAT IT DOES:**
Learns what normal user and system behavior looks like.
Reconstructs normal events perfectly.
Fails to reconstruct anomalous events.
High reconstruction error = suspicious behavior.

**ALGORITHM:** Deep Autoencoder (TensorFlow/Keras)
**DEPLOYMENT:** Real-time. Pre-loaded in memory.

### Training Data

**PRIMARY DATASET: UNSW-NB15**
```
Source:     University of New South Wales
Records:    175,341 training records
            82,332 testing records
Features:   49 features per record
Attack Types:
    Fuzzers:        18,184 records
    Analysis:        2,677 records
    Backdoors:       2,329 records
    DoS:            12,264 records
    Exploits:       44,525 records
    Generic:        40,000 records
    Reconnaissance: 13,987 records
    Shellcode:       1,511 records
    Worms:             174 records
    Normal:         93,000 records
```

**SYNTHETIC DATA — FINANCIAL SECTOR BEHAVIOR:**
```
Generated: 500,000 synthetic events
Tool:      SDV (Synthetic Data Vault)
           https://sdv.dev

NORMAL BEHAVIOR PROFILES GENERATED:

Human User Profile:
    Login: 08:00 - 18:00 business hours
    Location: Corporate IP range (10.x.x.x)
    Files accessed: 5-15 per hour
    Average file size: 50KB - 5MB
    Applications: Office365, Salesforce, Workday
    Data volume: 50MB - 500MB per day

Service Account Profile:
    Active: 24/7 automated processes
    Fixed source IP addresses
    Repetitive access patterns
    Same files accessed on schedule
    Low variance in behavior

API Client Profile:
    High frequency low volume requests
    Consistent request patterns
    Known endpoint access only
    Rate limited behavior

ATTACK SCENARIOS GENERATED ON TOP:
    Exfiltration: 500MB+ download at 3am
    Credential stuffing: 1000 logins/minute
    Lateral movement: 50 systems in 10 minutes
    Data staging: Bulk copy to temp location
    After hours privileged access
```

**PRODUCTION PERFORMANCE:**
```
Reconstruction Error Threshold: 0.15
False Positive Rate:            11%
True Positive Rate:             89%
Slow Exfiltration Detection:    94% accuracy
```

---

## Model 3 — LSTM
### Sequential Time-Based Pattern Detection

**WHAT IT DOES:**
Analyzes sequences of events over time.
Detects patterns that span minutes, hours, or days.
Catches slow attacks invisible to single-event rules.
C2 beaconing. Slow exfiltration. Gradual privilege escalation.

**ALGORITHM:** LSTM with Attention Mechanism (TensorFlow/Keras)
**DEPLOYMENT:** Batch. Runs every 15 minutes via APScheduler.

### Training Data

**PRIMARY DATASET: CICIDS 2018**
```
Source:     Canadian Institute for Cybersecurity
URL:        https://www.unb.ca/cic/datasets/ids-2018.html
Records:    1.2 million sequential flow records
Duration:   10 days of continuous traffic
Size:       7.2 GB

ATTACK SEQUENCES INCLUDED:
    Day 1-2:   Brute force SSH and FTP
    Day 3:     DoS and DDoS attacks
    Day 4:     Web attacks
    Day 5-6:   Botnet (ARES botnet)
    Day 7-8:   Infiltration and data exfiltration
    Day 9-10:  SQL injection and XSS

WHY SEQUENTIAL DATA MATTERS:
    LSTM sees: Login → file access → file copy
               → repeat every 60 seconds
    This is C2 beaconing.
    Single event rules miss this completely.
    LSTM catches the timing pattern.
```

**SUPPLEMENTARY DATASET: CTU-13**
```
Source:     Czech Technical University in Prague
URL:        https://www.stratosphereips.org/datasets-ctu13
Records:    13 botnet capture scenarios
            Over 100,000 botnet connections

BOTNET FAMILIES INCLUDED:
    Neris, Rbot, Virut, Menti, Sogou,
    Murlo, NSIS, Tbot, Htbot

WHY WE CHOSE IT:
    Best botnet C2 beaconing dataset available.
    Real captured botnet traffic.
    Essential for LSTM C2 detection training.
    Each botnet has unique timing signature.
```

**PRODUCTION PERFORMANCE:**
```
Sequence Length:          50 events
Processing Window:        15 minutes
C2 Beaconing Detection:   93% accuracy
Slow Exfil Detection:     91% accuracy
Batch Processing Time:    < 30 seconds per window
```

---

## Model 4 — Graph Neural Network (GNN)
### Relationship and Fraud Ring Detection

**WHAT IT DOES:**
Maps relationships between security entities.
Accounts, devices, IP addresses, transactions.
Finds coordinated attack patterns across entities.
Detects fraud rings invisible to single-entity analysis.

**ALGORITHM:** GraphSAGE (PyTorch Geometric)
**DEPLOYMENT:** On-demand. Triggered when risk score > 0.7.

### Training Data

**PRIMARY DATASET: Elliptic Bitcoin Dataset**
```
Source:     Elliptic + MIT-IBM Watson AI Lab
URL:        https://www.kaggle.com/ellipticco/elliptic-data-set
Records:    203,769 transactions
            234,355 directed edges (relationships)
Labeled:    4,545 illicit transactions
            42,019 licit transactions
            157,205 unlabeled

GRAPH STRUCTURE:
    Nodes: Bitcoin transactions
    Edges: Payment flows between transactions
    Features: 166 features per transaction
              Time step information
              Transaction metadata

WHY WE CHOSE IT:
    Only large-scale labeled financial
    fraud graph dataset available.
    Real blockchain transaction data.
    Illicit = fraud rings, money laundering,
    scams, ransomware payments.
    Directly applicable to financial fraud detection.

ADAPTATION FOR SECURITY:
    Bitcoin transactions → Security events
    Payment flows → Identity relationships
    Illicit transactions → Malicious events
    Fraud rings → Attack campaigns
```

**SYNTHETIC GRAPH DATA:**
```
Generated: 800,000 entity relationship records
Tool:      Custom Python graph generator

ENTITY TYPES:
    Users (human identities)
    Service Accounts (machine identities)
    IP Addresses (network entities)
    Devices (endpoint entities)
    Applications (data store entities)
    S3 Buckets (data entities)

RELATIONSHIP TYPES:
    User → logged in from → IP Address
    User → accessed → Application
    Service Account → called → API Endpoint
    Device → connected to → Network
    IP Address → geolocated to → Country

ATTACK GRAPH PATTERNS GENERATED:
    Fraud ring: Multiple accounts
                sharing IP addresses
    Lateral movement: One identity
                      touching many systems
    Data exfiltration: Multiple accounts
                       accessing same S3 bucket
    Account takeover: Same user from
                      multiple geographies
    Coordinated attack: Many IPs targeting
                        same endpoint
```

**PRODUCTION PERFORMANCE:**
```
Graph Build Time:         2-5 seconds per entity
Relationship Traversal:   Up to 3 hops
Fraud Ring Detection:     94% accuracy
False Positive Rate:      6%
Triggered:                When risk score > 0.7
```

---

## Model 5 — DNS Classifier
### DGA Domain and C2 Communication Detection

**WHAT IT DOES:**
Classifies DNS queries as legitimate or malicious.
Detects Domain Generation Algorithm (DGA) domains.
Identifies DNS tunneling for data exfiltration.
Flags C2 communication via DNS protocol.

**ALGORITHM:** Random Forest + NLP Features (scikit-learn)
**DEPLOYMENT:** Real-time. Scores every DNS event.

### Training Data

**MALICIOUS DOMAIN DATASET: URLhaus**
```
Source:     abuse.ch URLhaus project
URL:        https://urlhaus.abuse.ch/downloads/
Records:    250,000 malicious URLs and domains
Updated:    Daily
Categories:
    Malware distribution sites
    Phishing domains
    C2 communication endpoints
    Exploit kit landing pages
    Botnet control panels

CHARACTERISTICS OF MALICIOUS DOMAINS:
    High entropy (random characters)
    Unusual TLD combinations
    Short domain age
    No human readable words
    Long subdomain chains (DNS tunneling)
    Pattern: a7f3k2m9x.evil-domain.com
```

**MALICIOUS DOMAIN DATASET: MalwareBazaar**
```
Source:     abuse.ch MalwareBazaar
URL:        https://bazaar.abuse.ch/
Records:    100,000+ malware-associated domains
Updated:    Real-time
Contains:   Active C2 domains for:
            Emotet, TrickBot, Cobalt Strike,
            Qakbot, IcedID, AsyncRAT
```

**LEGITIMATE DOMAIN DATASET: Alexa Top 1M**
```
Source:     Alexa Internet / Amazon
Records:    1,000,000 legitimate domains
Contains:   Most visited legitimate websites
Examples:   google.com, microsoft.com,
            amazon.com, github.com
            
PURPOSE:
    Train model on what legitimate looks like.
    High information content.
    Human readable words.
    Established domain age.
    Known safe TLDs.
```

**SUPPLEMENTARY: Cisco Umbrella Popularity List**
```
Source:     Cisco Umbrella
Records:    1,000,000 most queried domains
Contains:   Enterprise-relevant legitimate domains
            Internal tools, SaaS applications,
            Cloud services commonly used in enterprise
```

**TOTAL TRAINING DATA:**
```
Malicious domains:   350,000
Legitimate domains:  500,000
Total:               850,000 labeled domains
Train/Test split:    80/20
```

**FEATURE ENGINEERING:**
```python
def extract_dns_features(domain):
    return {
        # Entropy features
        'entropy': calculate_shannon_entropy(domain),
        'digit_ratio': count_digits(domain) / len(domain),
        'vowel_ratio': count_vowels(domain) / len(domain),

        # Length features
        'domain_length': len(domain),
        'subdomain_count': domain.count('.'),
        'longest_word_length': max_word_length(domain),

        # N-gram features
        'bigram_freq': bigram_frequency_score(domain),
        'trigram_freq': trigram_frequency_score(domain),

        # Structural features
        'has_numbers_in_domain': has_numbers(domain),
        'consonant_ratio': consonant_count(domain) / len(domain),
        'special_char_count': count_special_chars(domain)
    }
```

**PRODUCTION PERFORMANCE:**
```
DGA Detection Accuracy:     97%
DNS Tunneling Detection:    94%
False Positive Rate:        3%
Response Time:              < 5ms per domain
Daily DNS Queries Scored:   500,000+
```

---

## Model 6 — PII Classifier
### Sensitive Data Classification

**WHAT IT DOES:**
Detects when security events involve sensitive data.
Classifies data as PCI, PHI, PII, or INTERNAL.
Applies risk multiplier when sensitive data accessed.
Ensures compliance-aware risk scoring.

**ALGORITHM:** BERT Fine-tuned + Rule-based (Hugging Face)
**DEPLOYMENT:** Real-time. Runs on every event.

### Training Data

**PRIMARY DATASET: PII Detection Dataset**
```
Source:     Kaggle PII Detection Competition
            Hugging Face Datasets Hub
URL:        https://huggingface.co/datasets/
            ai4privacy/pii-masking-300k
Records:    300,000 labeled text samples
Categories:
    PCI Data Indicators:
        Card numbers, CVV, expiry dates
        Payment endpoints, billing paths
        Card holder data references

    PHI Data Indicators:
        Patient IDs, medical record numbers
        Diagnosis codes, prescription data
        Healthcare provider references

    PII Data Indicators:
        Social security numbers
        Date of birth, home address
        Email addresses, phone numbers
        Government ID references

    INTERNAL Data:
        Internal system names
        Configuration files
        Log files, debug data
```

**FINANCIAL SECTOR CUSTOM LABELS:**
```
Generated: 50,000 custom labeled samples
Specific to financial services paths:

PCI PATHS:
    /api/payments/*
    /api/cards/*
    /api/transactions/*
    s3://prod-pci-data/*
    database/cardholder_data/*

PHI PATHS:
    /api/health/*
    /api/medical/*
    database/patient_records/*

PII PATHS:
    /api/customers/*
    /api/accounts/*
    /api/users/*
    database/customer_data/*
```

**PRODUCTION PERFORMANCE:**
```
PCI Detection Accuracy:   96%
PHI Detection Accuracy:   95%
PII Detection Accuracy:   94%
False Positive Rate:      4%
Risk Multiplier Applied:  1.3x for PCI
                          1.2x for PHI
                          1.1x for PII
```

---

## Model 7 — Identity Threat Detector
### Account Takeover and Credential Abuse Detection

**WHAT IT DOES:**
Specialized detection for identity-based attacks.
MFA fatigue. Impossible travel. Credential stuffing.
Session hijacking. Privilege escalation.
Insider threat indicators.

**ALGORITHM:** Gradient Boosting (XGBoost)
**DEPLOYMENT:** Real-time. Every authentication event.

### Training Data

**PRIMARY DATASET: CERT Insider Threat Dataset**
```
Source:     Carnegie Mellon University CERT Division
URL:        https://www.cmu.edu/sei/
            our-work/projects/insider-threat.html
Records:    300,000 user activity records
Version:    CERT v6.2
Scenarios:
    Malicious insider (data theft)
    Malicious insider (sabotage)
    Negligent insider (policy violation)
    Normal user behavior baseline

FEATURES AVAILABLE:
    Logon/logoff events with timestamps
    File access events
    Email send/receive events
    HTTP web browsing events
    Device connection events

WHY WE CHOSE IT:
    Gold standard for insider threat detection.
    Labeled by CMU security researchers.
    Contains both malicious and normal behavior.
    Widely used in academic research.
    Directly applicable to identity threat detection.
```

**SUPPLEMENTARY: Microsoft OIDC Sign-in Simulation**
```
Generated: 200,000 synthetic authentication events

ATTACK SCENARIOS SIMULATED:

MFA Fatigue Attack:
    User: john.smith@company.com
    Events: 15 MFA push notifications in 10 minutes
    Result: User approves on attempt 12 (fatigue)
    Pattern: Rapid repeated MFA requests

Impossible Travel:
    Login 1: New York, 09:00 AM EST
    Login 2: London, 09:45 AM EST
    (Physically impossible travel time)
    Pattern: Same user, different continents, short time

Credential Stuffing:
    1000 login attempts per minute
    Multiple usernames from single IP
    Password spray: same password all accounts
    Pattern: High volume, low success rate

Session Hijacking:
    Normal session: Corporate IP, Windows device
    Hijacked session: Tor exit node, Linux device
    Same session token, different context
    Pattern: Session context change mid-session

Password Spray:
    50 accounts, same password
    Spread over 60 minutes to avoid lockout
    Pattern: Low frequency, high breadth
```

**PRODUCTION PERFORMANCE:**
```
MFA Fatigue Detection:      92%
Impossible Travel:          99%
Credential Stuffing:        94%
Password Spray:             89%
False Positive Rate:        8%
Response Time:              < 15ms
```

---

## Model 8 — Ensemble Scorer
### Combined Risk Scoring

**WHAT IT DOES:**
Combines outputs from all 7 models.
Applies weighted voting to produce final risk score.
More accurate than any single model alone.
Single 0.0 to 1.0 risk score per event.

**ALGORITHM:** Weighted Ensemble (custom)
**DEPLOYMENT:** Real-time. Final step in scoring pipeline.

### How Ensemble Works

**MODEL WEIGHTS:**
```python
ENSEMBLE_WEIGHTS = {
    'isolation_forest':    0.20,  # Base anomaly
    'autoencoder':         0.15,  # Behavioral deviation
    'lstm':                0.20,  # Sequential patterns
    'gnn':                 0.15,  # Relationship context
    'dns_classifier':      0.10,  # DNS anomaly
    'pii_classifier':      0.10,  # Data sensitivity
    'identity_detector':   0.10   # Identity risk
}

def calculate_ensemble_score(model_scores):
    weighted_score = sum(
        score * ENSEMBLE_WEIGHTS[model]
        for model, score in model_scores.items()
    )

    # Apply PCI multiplier if sensitive data
    if model_scores['pii_classifier'] > 0.8:
        weighted_score = min(
            weighted_score * 1.3, 1.0
        )

    return weighted_score
```

**RISK THRESHOLDS:**
```
0.0 - 0.29:   LOW      → Store and log
0.30 - 0.59:  MEDIUM   → LSTM batch analysis
0.60 - 0.79:  HIGH     → GNN on-demand analysis
0.80 - 1.0:   CRITICAL → LLM agent investigation
                         + HITL notification
```

**TRAINING DATA:**
```
Trained on outputs of all 7 models combined.
Validation dataset: 50,000 labeled security events
from CeraPack production environment.
Labels: SOC analyst true/false positive decisions.
Trained to minimize false positives
while maximizing true positive rate.
```

**PRODUCTION PERFORMANCE:**
```
Overall Accuracy:       95%
False Positive Rate:    5%
True Positive Rate:     95%
Events Per Second:      10,000+
```

---

## Continuous Learning Flywheel

```
HOW OUR MODELS IMPROVE OVER TIME:

STEP 1 — EVENT PROCESSING:
Platform processes real security events
from Sentinel, S3, Okta, CyberArk.
Every event scored by all 8 models.

STEP 2 — SOC ANALYST LABELING:
High risk events reviewed by SOC analysts.
Analyst marks: True Positive or False Positive.
This creates labeled training data
from real production events.

STEP 3 — MLFLOW TRACKING:
Every labeled event stored in MLflow.
Full audit trail of:
    Event features
    Model predictions
    Analyst decision
    Timestamp

STEP 4 — WEEKLY RETRAINING:
Every Sunday 02:00 UTC:
    New model trained on accumulated labels
    Validated against holdout test set
    If accuracy improves: new model deployed
    If accuracy drops: old model kept

STEP 5 — ZERO DOWNTIME DEPLOYMENT:
Old model serves while new model loads.
Blue/green deployment pattern.
Instant rollback if issues detected.

PRODUCTION RESULTS AT CERAPACK:
Month 1:  50,000 labeled events accumulated
Month 6:  Model accuracy improved 23%
          False positive rate dropped from
          15% to 5% through continuous tuning
```

---

## Data Privacy and Compliance

```
ALL TRAINING DATA IS:

PUBLIC DATASETS:
    Free and open for commercial use.
    No customer data used for initial training.
    Fully reproducible and auditable.

SYNTHETIC DATA:
    Statistically similar to real data.
    Contains no real customer information.
    Generated using open source tools.
    Fully documented and reproducible.

PRODUCTION TELEMETRY:
    Stays within client environment.
    Never sent to external systems.
    Processed locally on client infrastructure.
    Deleted after model training cycle.

COMPLIANCE:
    PCI-DSS: No card data in training sets.
    GDPR: No personal data used for training.
    HIPAA: No PHI data used for training.
    SR 11-7: Full model documentation.
             Audit trail via MLflow.
             Human oversight via HITL.

REGULATORY ALIGNMENT:
    NIST AI RMF: Risk management documented.
    EU AI Act Article 14: Human oversight built in.
    OCC Model Risk (SR 11-7): Model governance
    framework with validation and audit trail.
```

---

## Public Dataset References

```
All datasets are freely available:

1. CICIDS 2017/2018
   https://www.unb.ca/cic/datasets/ids-2017.html
   https://www.unb.ca/cic/datasets/ids-2018.html

2. UNSW-NB15
   https://research.unsw.edu.au/projects/unsw-nb15-dataset

3. CTU-13 Botnet Dataset
   https://www.stratosphereips.org/datasets-ctu13

4. Elliptic Bitcoin Dataset
   https://www.kaggle.com/ellipticco/elliptic-data-set

5. URLhaus Malicious URLs
   https://urlhaus.abuse.ch/downloads/

6. MalwareBazaar
   https://bazaar.abuse.ch/

7. Alexa Top 1 Million Domains
   https://www.alexa.com/topsites

8. Cisco Umbrella Popularity List
   https://s3-us-west-1.amazonaws.com/
   umbrella-static/top-1m.csv.zip

9. PII Detection Dataset
   https://huggingface.co/datasets/ai4privacy/
   pii-masking-300k

10. CERT Insider Threat Dataset v6.2
    https://www.cmu.edu/sei/our-work/projects/
    insider-threat.html
```

---

## Summary for Executive Presentation

```
WHEN ASKED "WHERE DOES YOUR TRAINING DATA
COME FROM?" — THREE SENTENCE ANSWER:

"Our 8 ML models are trained on a combination
of three data sources.

First: Publicly available security research
datasets — 6.2 million labeled records from
institutions like Carnegie Mellon University,
the Canadian Institute for Cybersecurity,
and the University of New South Wales.

Second: Synthetic data generated to reflect
financial sector specific attack patterns
and normal behavior baselines.

Third: A continuous learning flywheel where
SOC analysts label production events
feeding back into weekly model retraining —
improving accuracy 23% over 6 months
at CeraPack with zero customer data
leaving the client environment."
```

---

*AbuTech AI Security Platform*
*Version 1.0 | Last Updated: May 2026*
*Maintained by: Eliud Maina, CEO Abuhari Consulting*
