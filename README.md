# 🧬 SepsiArc — From Sepsis Detection to Recovery Risk

> **Explainable dual-stage machine learning for early ICU sepsis detection and post-sepsis risk stratification.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mutexsepsiarc.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Stage%201-XGBoost-orange?logo=xgboost)
![XGBoost](https://img.shields.io/badge/Stage%202-XGBoost-red?logo=xgboost)
![License](https://img.shields.io/badge/License-Academic%20Research-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## ⚠️ Clinical Disclaimer

> **SepsiArc is a clinical decision-support and research tool — NOT an autonomous diagnostic system.**
> All predictions and risk scores must be interpreted by an appropriately qualified healthcare professional.
> Results must never be used as a substitute for clinical judgement, and must not constitute treatment orders or confirmed diagnoses.
> Stage 2 post-sepsis risk is a project-defined proxy classification — not a validated clinical outcome.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dual-Stage Architecture](#dual-stage-architecture)
- [Stage 1 — Sepsis Detection](#stage-1--sepsis-detection)
- [Stage 2 — Post-Sepsis Risk Stratification](#stage-2--post-sepsis-risk-stratification)
- [AI Clinical Assistant](#ai-clinical-assistant)
- [Explainability — SHAP & LIME](#explainability--shap--lime)
- [PDF Patient Reports](#pdf-patient-reports)
- [Input Format](#input-format)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Installation and Local Run](#installation-and-local-run)
- [Deployment](#deployment)
- [Team](#team)
- [References](#references)

---

## Overview

**SepsiArc** is a full-stack interactive clinical decision-support dashboard built with Streamlit, implementing a **dual-stage explainable machine learning pipeline**:

1. **Stage 1 — Early Sepsis Detection** — Analyses a patient's hourly ICU timeline using 135 engineered clinical features and a cost-sensitive XGBoost model to screen for sepsis risk at every observed hour.
2. **Stage 2 — Post-Sepsis Risk Stratification** — For patients who screen positive or borderline, assesses post-sepsis physiological trajectory using 65 patient-level aggregate features and a second XGBoost model, classifying patients as Higher Risk or Lower Risk.
3. **Explainable AI (SHAP & LIME)** — Every prediction is explained with SHAP (global/local Shapley attributions) and LIME (local surrogate model) visualisations.
4. **Conversational AI Assistant** — An LLM-powered medical education assistant grounded in verified project context and clinical guidelines.
5. **PDF Clinical Reports** — Downloadable Stage 1, Stage 2, and combined patient reports generated without any external binary dependencies.

Sepsis affects nearly **49 million people** per year globally with a **20–30% ICU mortality rate**. Every hour of delay in treatment increases mortality risk by approximately 7–10%. SepsiArc provides interpretable, continuous ICU screening to support earlier clinical decisions.

SepsiArc was developed as a final-year engineering project at **Mangalore Institute of Technology and Engineering** under the guidance of **Dr. Dinesh L**, Department of Computer Science and Engineering.

---

## Key Features

| Feature | Description |
|---|---|
| 🔬 Stage 1 — Sepsis Detection | Regularized Cost-Sensitive XGBoost (Model A) · 135 features · Threshold 0.143 |
| 🧪 Stage 2 — Risk Stratification | XGBoost patient-level model · 65 aggregate features · Threshold 0.29 |
| 📊 SHAP Explanations | TreeExplainer — top-10 feature attribution bar charts for every prediction |
| 🔍 LIME Explanations | Local surrogate model — patient-specific feature contribution charts |
| 📋 PDF Reports | Downloadable Stage 1, Stage 2, and combined dual-stage clinical reports |
| 💬 AI Clinical Assistant | LLM-powered conversational assistant (AIMLAPI / OpenRouter) with safety guardrails |
| 📈 Research Dashboard | Verified model metrics, feature engineering transparency, dual-stage pipeline overview |
| 👥 About & Team | Dataset provenance, methodology, team, and clinical disclaimer |
| 🎨 Light Clinical Theme | Clean blue-white clinical UI (`#F0F9FF` background) |

---

## Dual-Stage Architecture

```
Patient ICU Data (PSV or CSV — one patient, hourly rows)
        |
        v
+--------------------------------------------+
|  STAGE 1 — Sepsis Detection                |
|  Algorithm : XGBoost (Model A)             |
|  Features  : 135 engineered clinical feats |
|  Threshold : 0.143 (14.3%)                 |
|  Output    : Sepsis probability per hour   |
+-------------------+------------------------+
                    |
       Peak probability across all hours
                    |
     +--------------+--------------+
     |              |              |
     v              v              v
 >= 0.143        >= 0.08       < 0.08
SEPSIS ALERT   BORDERLINE     LOW RISK
(Stage 2 on)  (Stage 2 on)  (Stage 2 off)
     |              |
     +------+-------+
            |
            v
+--------------------------------------------+
|  STAGE 2 — Post-Sepsis Risk Stratification |
|  Algorithm : XGBoost (patient-level)       |
|  Features  : 65 patient-level aggregates   |
|  Threshold : 0.29 (29%)                    |
|  Output    : Higher Risk / Lower Risk      |
+--------------------------------------------+
            |
            v
    SHAP + LIME Explanations
            |
            v
    Downloadable PDF Reports
```

### Stage 2 Eligibility Gates

| Stage 1 Result | Peak Probability | Stage 2 Access |
|---|---|---|
| **Sepsis Alert** | ≥ 14.3% | Unlocked — full access |
| **Borderline** | ≥ 8.0% and < 14.3% | Unlocked — proceed with caution |
| **Low Risk** | < 8.0% | Locked — Stage 2 not applicable |

---

## Stage 1 — Sepsis Detection

### Model
- **Algorithm:** XGBoost — Regularized Cost-Sensitive Model A
- **Artifact:** `models/Stage1_Final_Model_A.json` (XGBoost native JSON)
- **Features:** 135 engineered clinical features (from ~20 raw ICU variables)
- **Decision Threshold:** **0.143** (14.3%) — selected to achieve ≥ 70% recall on the validation set
- **Output:** Per-hour sepsis probability (0–1); patient-level result = peak probability across all hours

### Feature Engineering (10-step pipeline)
Starting from raw PSV/CSV columns, 135 features are generated:
1. Drop sparse/non-contract raw columns
2. Chronological sort by `ICULOS`
3. **18 missingness indicators** (`{var}_was_missing` — absence of measurements is a clinical signal)
4. **18 recency features** (`{var}_hours_since_last` — time since last valid observation)
5. Causal forward fill (max 4 h for vitals/labs, max 3 h for lactate — no future leakage)
6. **24 lag & change features** (1-hour and 3-hour lags + deltas for 6 vital signs)
7. **48 rolling window statistics** (3-hour and 6-hour rolling mean/min/max/std for 6 vitals)
8. **5 clinical derived features** — Shock Index (HR÷SBP), Pulse Pressure (SBP−DBP), MAP Ratio, partial SIRS score, Low MAP flag
9. Column-median imputation for any residual NaNs
10. Replace infinities with NaN → 0

### Risk Classification
| Class | Condition | Stage 2 |
|---|---|---|
| **SEPSIS ALERT** | Peak probability ≥ 0.143 | Available |
| **BORDERLINE** | Peak probability ≥ 0.080 | Available (with caution note) |
| **LOW RISK** | Peak probability < 0.080 | Locked |

---

## Stage 2 — Post-Sepsis Risk Stratification

### Model
- **Algorithm:** XGBoost (patient-level classifier, max_depth = 2)
- **Artifacts:** `models/stage2_xgb_model.joblib` (model) + `models/stage2_xgb_imputer.joblib` (SimpleImputer)
- **Features:** 65 patient-level aggregate features
- **Decision Threshold:** **0.29** (29%)
- **Target:** `Risk_50pct` — a rule-derived physiological risk proxy (see Disclaimer)
- **Access:** Gated — requires Stage 1 ALERT or BORDERLINE result in the same session

### How Stage 2 Works
Stage 2 splits the patient's ICU timeline at the **reference hour** (the first hour Stage 1 flagged an alert) into two windows:
- **Pre-label window** — all hours before the reference hour
- **Early post-label window** — the reference hour and all hours after it

For 6 vital signs (HR, SBP, MAP, Resp, O2Sat, Temp), it computes 10 aggregate statistics per window:
count, mean, min, max, range, SD, slope (linear regression), first-to-last change, and cross-window mean difference.
Plus 3 patient scalars: Age, HospAdmTime, ICULOS at the reference label.
Total: **65 features** aligned to the saved feature contract in `stage2_xgb_metadata.json`.

Residual NaN values (e.g., if a window has no observations) are imputed using the pre-fitted `stage2_xgb_imputer.joblib` — never re-fitted at inference time.

### Risk Classification
| Class | Condition | Meaning |
|---|---|---|
| **HIGHER RISK** | Probability ≥ 0.29 | Physiological deterioration pattern detected — high clinical vigilance recommended |
| **LOWER RISK** | Probability < 0.29 | Signs of physiological stabilisation |

> **Important:** `Risk_50pct` is a project-defined proxy severity score (based on Lactate, pH, Creatinine, Platelets, MAP, and qSOFA in post-label hours). It is **not** a validated clinical outcome such as 30-day mortality.

---

## AI Clinical Assistant

- **Location:** `pages/3_AI_Assistant.py`
- **Type:** LLM-powered conversational assistant via OpenAI-compatible API
- **Providers supported:** [AIMLAPI](https://aimlapi.com) (key prefix `sk-`) or [OpenRouter](https://openrouter.ai) (key prefix `sk-or-`)
- **Fallback:** If the primary model returns empty, automatically cascades through a list of free OpenRouter models
- **Configuration:** Set `AIMLAPI_API_KEY` and `AIMLAPI_MODEL` in `.streamlit/secrets.toml` (local) or Streamlit Cloud Secrets (deployed)
- **Scope:** Sepsis education, clinical scoring systems (SIRS, qSOFA, SOFA), SepsiArc project facts, general critical care concepts
- **Safety guardrails:** Built-in safety prompt — never diagnoses real patients, always directs emergencies to call emergency services (999/911/112), never invents research citations, discloses educational-only status
- **Conversation memory:** Last 12 exchanges (24 messages) retained per session

---

## Explainability — SHAP & LIME

Both Stage 1 and Stage 2 provide two complementary local explanations for every prediction:

### SHAP (SHapley Additive exPlanations)
- **Method:** `shap.TreeExplainer` (exact Shapley values for tree models)
- **Target:** Peak-risk hour for Stage 1; patient-level feature vector for Stage 2
- **Positive SHAP value:** Feature pushes prediction toward Sepsis Alert / Higher Risk
- **Negative SHAP value:** Feature reduces predicted risk
- **Display:** Top-10 feature attribution bar chart, colour-coded

### LIME (Local Interpretable Model-agnostic Explanations)
- **Method:** `lime.lime_tabular.LimeTabularExplainer` with a Gaussian perturbation background
- **Target:** Same peak-hour row (Stage 1) or patient vector (Stage 2)
- **Display:** Bar chart of local feature contributions with threshold rules (e.g., `HR > 105`)

> SHAP and LIME values describe **model behaviour** — they do not prove clinical causation.

---

## PDF Patient Reports

All reports are generated client-side using `fpdf2` (pure Python — no binary dependencies).

| Report | Contents | Download |
|---|---|---|
| **Stage 1 Report** | Patient summary, peak probability, first alert hour, vital trend table, SHAP drivers, Stage 2 eligibility, disclaimer | `SepsiArc_Stage1_Report_{ID}_{timestamp}.pdf` |
| **Stage 2 Report** | Reference hour, window observation counts, risk probability vs threshold, vital trend table, SHAP drivers, disclaimer | `SepsiArc_Stage2_Report_{ID}_{timestamp}.pdf` |
| **Combined Report** | Full dual-stage analysis — Stage 1 + transition metrics + Stage 2 in one document | `SepsiArc_Complete_Report_{ID}_{timestamp}.pdf` |

---

## Input Format

SepsiArc accepts **PSV (pipe-separated) or CSV (comma-separated)** files containing one patient's hourly ICU observations.

- **One row = one ICU hour** — variable number of rows supported (no minimum/maximum)
- **Columns matched by name** — column order in the file does not matter
- **Separator auto-detected** from the first line
- **SepsisLabel column** — displayed for reference if present; never used as a model input feature

**Expected columns (any subset is acceptable):**
```
ICULOS | HR | O2Sat | Temp | SBP | MAP | DBP | Resp |
BUN | Calcium | Creatinine | Glucose | Lactate | Magnesium |
Potassium | Hct | Hgb | WBC | Platelets | Age | Gender | HospAdmTime
```

> Note: Manual single-value entry is **not supported**. A time-series file is required to compute lag, rolling, and recency features.

---

## Dataset

- **Source:** PhysioNet / Computing in Cardiology Challenge 2019 — Reyna et al., 2019
- **Patients:** 40,336 ICU patients across multiple US hospital systems (Training Sets A and B)
- **Observations:** ~1.55 million hourly ICU time-points
- **Sepsis prevalence:** ~1.8% of rows (severe class imbalance)
- **Stage 1 target:** `SepsisLabel` — binary (1 = sepsis onset at that hour)
- **Stage 2 target:** `Risk_50pct` — engineered proxy label (score ≥ 3 from Lactate, pH, Creatinine, Platelets, MAP, qSOFA thresholds in post-label hours)
- **Class imbalance handling (Stage 1 training):** `scale_pos_weight = 3` in XGBoost; `RandomUnderSampler` to 5:1 ratio in baseline benchmark comparison
- **Train/test split:** Patient-level stratified 80/20 — zero patient overlap between splits

> The dataset is **not included** in this repository (excluded by `.gitignore`).
> Download from: https://physionet.org/content/challenge-2019/

---

## Preprocessing

The live application applies this pipeline to each uploaded patient file:

1. Auto-detect separator (`|` or `,`) and parse all columns as numeric
2. Sort rows chronologically by `ICULOS`
3. Drop 20 sparse/non-contract columns (`EtCO2`, `SaO2`, `AST`, `Alkalinephos`, `Bilirubin_direct`, `TroponinI`, `Fibrinogen`, `PTT`, `SaO2`, `Phosphate`, `HCO3`, `Chloride`, `EtCO2`, `BaseExcess`, `PaCO2`, `pH`, `FiO2`, `Unit1`, `Unit2`, etc.)
4. Generate 18 missingness indicators and 18 recency features
5. Causal forward fill (restricted window — no future-data leakage)
6. Generate 24 lag/delta features and 48 rolling window statistics
7. Compute 5 clinical derived features (Shock Index, Pulse Pressure, MAP Ratio, SIRS partial, Low MAP)
8. Fill residual NaNs with column medians; replace infinities with 0
9. Align columns to the 135-feature contract from `Stage1_Final_Model_A.json`

For Stage 2 training, the offline pipeline (`preprocess.py`, not deployed) additionally applied MICE imputation (`IterativeImputer` with `ExtraTreesRegressor`) to lab columns on a 50,000-row sample before generating Stage 2 patient-level features.

---

## Model Performance

> All metrics are from the **held-out patient-level test set** (80/20 stratified split).
> Patient-level splitting ensures zero patient data leakage between train and test.

### Stage 1 — Sepsis Detection

**Model:** Regularized Cost-Sensitive XGBoost — Model A (`Stage1_Final_Model_A.json`)

| Metric | Value |
|---|---|
| AUROC | 0.7587 |
| PR-AUC | 0.1728 (validation) |
| Recall | 70.97% |
| Precision | 11.20% |
| F1 Score | 19.35% |
| **Decision Threshold** | **0.143** |
| Features | 135 |

> Low precision reflects the severe class imbalance (~1.8% sepsis rate). The threshold is deliberately low to maximise recall — capturing more sepsis cases at the cost of more false alerts.

**Benchmark comparison** (`model_results.json` — all models evaluated on the same test set):

| Model | AUROC | Recall | Threshold |
|---|---|---|---|
| RandomForest | 0.8180 | 27.42% | 0.58 |
| XGBoost (baseline) | 0.8093 | 30.33% | 0.67 |
| LightGBM | 0.8087 | 27.53% | 0.63 |
| LightGBM-SMOTE | 0.8192 | 50.24% | 0.20 |
| LightGBM-Calibrated | 0.8229 | 50.38% | 0.33 |
| **XGBoost Model A (active)** | **0.7587** | **70.97%** | **0.143** |

> XGBoost Model A was selected for **highest recall** (≥70%) rather than highest AUROC, as missing a sepsis case is clinically more costly than a false alert.

---

### Stage 2 — Post-Sepsis Risk Stratification

**Model:** XGBoost (patient-level, max_depth = 2) — `stage2_xgb_model.joblib`
**Test patients:** 575 | **Target prevalence:** 43.1% Higher Risk

| Metric | Value |
|---|---|
| ROC-AUC | 0.8294 |
| PR-AUC | 0.8001 |
| Recall | 85.48% |
| Precision | 62.35% |
| F1 Score | 0.7211 |
| Accuracy | 71.48% |
| Brier Score | 0.1659 |
| **Decision Threshold** | **0.29** |
| Features | 65 |

**Confusion matrix (575 test patients):**
```
                     Predicted Lower Risk    Predicted Higher Risk
Actual Lower Risk           199                      128
Actual Higher Risk           36                      212
```

> Stage 2 target `Risk_50pct` is a project-defined physiological proxy, not a clinically validated outcome.

---

## Project Structure

```
SepsiArc/
│
├── app.py                              # Home dashboard — epidemiology stats, navigation
│
├── pages/
│   ├── 1_Stage_1_Detection.py         # Stage 1: XGBoost screening, SHAP, LIME, PDF download
│   ├── 2_Stage_2_Risk.py              # Stage 2: XGBoost risk, SHAP, LIME, PDF download
│   ├── 3_AI_Assistant.py              # LLM conversational assistant (AIMLAPI / OpenRouter)
│   ├── 4_Model_Information.py         # Research dashboard: metrics, features, pipeline
│   └── 5_About.py                     # Methodology, dataset, team, clinical disclaimer
│
├── utils/
│   ├── ui_components.py               # Shared header, sidebar, footer, CSS, design tokens
│   └── pdf_report.py                  # fpdf2 PDF generator (Stage 1, Stage 2, combined reports)
│
├── models/
│   ├── Stage1_Final_Model_A.json       # ★ Active Stage 1 XGBoost model (135 features)
│   ├── Stage1_Final_Model_A_Metadata.json  # Hyperparameters + threshold 0.143
│   ├── Stage1_FeatureImportance.csv    # Stage 1 feature importances
│   ├── Stage1_Threshold.txt            # Stage 1 threshold record (0.143)
│   ├── Stage1_XGBoost.pkl              # Pickle backup of Stage 1 model
│   │
│   ├── stage2_xgb_model.joblib         # ★ Active Stage 2 XGBoost model (65 features)
│   ├── stage2_xgb_imputer.joblib       # SimpleImputer fitted on Stage 2 training cohort
│   ├── stage2_xgb_metadata.json        # Stage 2 feature names + threshold 0.29
│   ├── stage2_xgb_summary.json         # Stage 2 test metrics (ROC-AUC 0.8294)
│   │
│   ├── Stage2_Final_Risk50pct_Exports/ # Stage 2 evaluation archive (confusion matrix, predictions)
│   │
│   ├── feature_names.json              # Legacy 79-feature list (v1 training; not used by active app)
│   ├── model_results.json              # Stage 1 benchmark comparison (all 6 models)
│   ├── stage2_results.json             # Legacy Stage 2 LightGBM metrics (superseded)
│   ├── stage2_model.pkl                # Legacy Stage 2 LightGBM model (superseded)
│   ├── stage2_feature_names.json       # Legacy Stage 2 feature list (superseded)
│   ├── sepsis_best_model.pkl           # Baseline LightGBM-Calibrated model
│   ├── lgb_explainer_model.pkl         # SHAP LightGBM explainer (benchmark)
│   ├── iso_calibrator.pkl              # Isotonic calibrator (baseline model)
│   ├── clinical_gate_params.json       # Legacy clinical gate config (not active)
│   ├── median_imputer.pkl              # Legacy median imputer
│   └── sepsis_threshold.pkl            # Baseline threshold record
│
├── graphs/                             # Pre-generated evaluation plots (ROC, confusion matrix, etc.)
├── public/images/                      # Team and mentor photos (used in About page)
│
├── .streamlit/
│   ├── config.toml                     # Streamlit theme and server config
│   └── secrets.toml.example            # Template for API key configuration
│
└── requirements.txt                    # Pinned Python dependencies
```

> `preprocess.py` and `train_models.py` exist locally but are excluded from the repository by `.gitignore` — they are offline training utilities, not part of the deployed application.

---

## Installation and Local Run

### Prerequisites
- Python 3.10+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/MutexTeam/Sepsiarc.git
cd Sepsiarc

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

### AI Assistant Configuration (optional)
To enable the AI Clinical Assistant, create `.streamlit/secrets.toml`:
```toml
AIMLAPI_API_KEY = "your-api-key-here"   # AIMLAPI or OpenRouter key
AIMLAPI_MODEL   = "gpt-4o-mini"         # Any model ID supported by your provider
```
If the key is not configured, the assistant displays a setup prompt — all other pages work without it.

---

## Deployment

The application is deployed on **Streamlit Community Cloud**:

**Live App:** https://mutexsepsiarc.streamlit.app

### Deploy Your Own
1. Fork this repository on GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub account
4. Select this repo, `main` branch, `app.py` as the entry point
5. Add `AIMLAPI_API_KEY` and `AIMLAPI_MODEL` in the **Secrets** tab (for AI Assistant)
6. Click **Deploy**

Streamlit Cloud auto-deploys on every push to `main`.

---

## Team

| Role | Name |
|---|---|
| **Project Mentor / Guide** | Dr. Dinesh L — Dept. of CSE, MITE |
| ML / Data Science | Affan Zama |
| Backend / Data Pipeline | Karthik Hegde |
| Frontend / UX | Manish Kumar |
| Research / Documentation | Jagadish |

**Institution:** Mangalore Institute of Technology and Engineering (MITE)  
**Department:** Computer Science and Engineering  
**Academic Year:** 2024–2025

---

## References

1. Reyna, M. A., et al. (2019). *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019.* Critical Care Medicine.
2. Singer, M., et al. (2016). *The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).* JAMA.
3. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS.
4. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* ACM KDD.
5. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
6. Ribeiro, M. T., et al. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME).* ACM KDD.
7. Surviving Sepsis Campaign. (2018). *Hour-1 Bundle.* Society of Critical Care Medicine.

---

**SepsiArc** — From Sepsis Detection to Recovery Risk  
Built for clinical research and education. Not for autonomous diagnosis.  
© 2026 SepsiArc · Mangalore Institute of Technology and Engineering
