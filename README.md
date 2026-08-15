# 🧬 SepsiArc — From Sepsis Detection to Recovery Risk

> **AI-powered clinical decision-support for early sepsis detection and post-sepsis risk stratification using a dual-stage machine learning framework.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mutexsepsiarc.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![License](https://img.shields.io/badge/License-Academic%20Research-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## ⚠️ Clinical Disclaimer

> **SepsiArc is a clinical decision-support and research tool — NOT an autonomous diagnostic system.**
> All predictions and risk scores must be interpreted by an appropriately qualified healthcare professional.
> Results must never be used as a substitute for clinical judgement, and must not constitute treatment orders or confirmed diagnoses.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dual-Stage Framework](#dual-stage-framework)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Installation and Local Run](#installation-and-local-run)
- [Deployment](#deployment)
- [Team](#team)
- [References](#references)

---

## Overview

**SepsiArc** is a full-stack interactive clinical dashboard built with Streamlit, implementing a dual-stage machine learning pipeline for:

1. **Early Sepsis Detection** — Screening patients for sepsis risk using 79 engineered ICU features
2. **Post-Sepsis Risk Stratification** — Assessing severity and recovery risk for sepsis-eligible patients using 65 features
3. **Explainable AI** — SHAP (SHapley Additive exPlanations) visualisations showing which clinical features drove each prediction
4. **AI Clinical Assistant** — A rule-based Q&A assistant for sepsis education and clinical reference

Sepsis is one of the leading causes of preventable death globally — affecting nearly **49 million people** per year with a mortality rate of **20–30%**. Early recognition and treatment within the first hour significantly improves survival outcomes.

SepsiArc was developed as part of a final-year engineering project at **Mangalore Institute of Technology and Engineering** under the guidance of **Dr. Dinesh L**, Department of Computer Science and Engineering.

---

## Features

| Feature | Description |
|---|---|
| 🔬 Stage 1 — Sepsis Detection | LightGBM-Calibrated model with clinical gating (SIRS, qSOFA, shock index) |
| 🧪 Stage 2 — Risk Assessment | Separate LightGBM model for post-sepsis severity classification |
| 📊 SHAP Explanations | Top-10 feature contribution bar charts for every prediction |
| 🤖 AI Assistant | Rule-based clinical Q&A on sepsis symptoms, criteria, and management |
| 📈 Research Dashboard | Model performance metrics, AUROC curves, feature importance, calibration plots |
| 👥 Team and About | Project methodology, dataset transparency, and team information |
| 🎨 Light Clinical Theme | Clean blue-white clinical UI (#F0F9FF) |

---

## Dual-Stage Framework

```
Patient ICU Data
      |
      v
+-----------------------------------+
|  STAGE 1 - Sepsis Detection       |
|  LightGBM-Calibrated              |
|  79 engineered features           |
|  Threshold: 0.33                  |
|  + Clinical Gate (SIRS/qSOFA)     |
+--------------+--------------------+
               |
    +----------+-----------+
    |                      |
    v                      v
NEGATIVE              POSITIVE / BORDERLINE
(< 15%)               (>= 15%)
    |                      |
    |                      v
    |          +---------------------------+
    |          |  STAGE 2 - Risk Assess.   |
    |          |  LightGBM                 |
    |          |  65 engineered features   |
    |          |  Threshold: 0.58          |
    |          +---------------------------+
    |                      |
    v                      v
Continue Standard    HIGH / LOW
Monitoring           Risk Profile
```

### Eligibility Gates

| Stage 1 Result | Probability | Stage 2 Access |
|---|---|---|
| **Sepsis Positive** | >= 33% | Full access |
| **Borderline** | 15-33% | Access with caution |
| **Low Risk** | < 15% | Locked |

---

## Dataset

- **Source:** PhysioNet — Early Prediction of Sepsis from Clinical Data (Reyna et al., 2019)
- **Original:** 40,336 ICU patient records (Training Set A + B)
- **Post-Cleaning:** ~28,000 records after MICE imputation and feature engineering
- **Features engineered:** 79 (Stage 1), 65 (Stage 2)
- **Label:** SepsisLabel (binary — 1 = sepsis onset within next hour)
- **Imbalance handling:** SMOTE oversampling on training set only

> Dataset is **not included** in this repository due to size constraints.
> Download from: https://physionet.org/content/challenge-2019/

---

## Model Performance

### Stage 1 — Sepsis Detection (LightGBM-Calibrated)

| Metric | Score |
|---|---|
| AUROC | 0.93+ |
| Threshold | 0.33 |
| Features | 79 |
| Calibration | Isotonic Regression |

### Stage 2 — Post-Sepsis Risk (LightGBM)

| Metric | Score |
|---|---|
| AUROC | 0.9631 |
| F1 Score | 0.7486 |
| Precision | 0.7936 |
| Recall | 0.7083 |
| Threshold | 0.58 |
| Features | 65 |

> All metrics are from the held-out test set. No data leakage — label-generating features excluded from predictor sets.

---

## Project Structure

```
Sepsis_Mutex/
|
+-- app.py                          # Main Streamlit entry point (Home page)
|
+-- pages/
|   +-- 1_Stage_1_Detection.py     # Stage 1: Sepsis Detection
|   +-- 2_Stage_2_Risk.py          # Stage 2: Post-Sepsis Risk Assessment
|   +-- 3_AI_Assistant.py          # AI Clinical Assistant
|   +-- 4_Model_Information.py     # Research and Model Info
|   +-- 5_About.py                 # About + Team
|
+-- utils/
|   +-- ui_components.py           # Global header, footer, sidebar, theme tokens
|
+-- models/
|   +-- sepsis_best_model.pkl      # Stage 1 LightGBM-Calibrated model
|   +-- sepsis_threshold.pkl       # Stage 1 optimal threshold (0.33)
|   +-- feature_names.json         # Stage 1 feature contract (79 features)
|   +-- iso_calibrator.pkl         # Isotonic calibrator
|   +-- lgb_explainer_model.pkl    # SHAP explainer model
|   +-- clinical_gate_params.json  # Clinical gating parameters
|   +-- stage2_model.pkl           # Stage 2 LightGBM model
|   +-- stage2_threshold.pkl       # Stage 2 optimal threshold (0.58)
|   +-- stage2_feature_names.json  # Stage 2 feature contract (65 features)
|   +-- stage2_results.json        # Stage 2 evaluation metrics
|   +-- model_results.json         # Stage 1 evaluation metrics
|
+-- graphs/                        # Pre-generated evaluation plots
+-- public/images/                 # Team and mentor photos
+-- .streamlit/config.toml         # Theme and server config for deployment
+-- requirements.txt               # Pinned Python dependencies
+-- preprocess.py                  # Data preprocessing pipeline
+-- train_models.py                # Model training scripts
```

---

## Installation and Local Run

### Prerequisites
- Python 3.12+
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

### Input Modes
- **Mode A (Manual):** Enter current vital signs and lab values directly — the app derives all 79/65 features automatically
- **Mode B (CSV Upload):** Upload a sequential ICU observation CSV — rolling means, lag features, and temporal indicators are computed from the full time series

---

## Deployment

The application is deployed on **Streamlit Community Cloud**:

**Live App:** https://mutexsepsiarc.streamlit.app

### Deploy Your Own
1. Fork this repository
2. Go to https://share.streamlit.io
3. Connect your GitHub account
4. Select this repo, `main` branch, `app.py` as the main file
5. Click **Deploy**

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
4. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
5. Surviving Sepsis Campaign. (2018). *Hour-1 Bundle.* Society of Critical Care Medicine.

---

**SepsiArc** — From Sepsis Detection to Recovery Risk  
Built for clinical research. Not for autonomous diagnosis.  
© 2026 SepsiArc · Mangalore Institute of Technology and Engineering
