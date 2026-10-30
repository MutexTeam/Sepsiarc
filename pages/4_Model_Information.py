"""
pages/4_Model_Information.py  --  SepsiArc Research & Model Information
Updated to reflect:
  Stage 1: Regularized Cost-Sensitive XGBoost Model A (135 features, threshold 0.143)
  Stage 2: XGBoost post-sepsis risk model (65 patient-level aggregate features, threshold 0.29)
Input: PSV/CSV file upload for both stages
"""
import streamlit as st
import json, joblib, os, sys, io, base64
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.ui_components import (
    render_header, render_footer, render_sidebar,
    render_disclaimer_banner,
    NAVY, TEAL, GREEN, AMBER, RED, SLATE_200, SLATE_400, SLATE_700,
)

st.set_page_config(
    page_title="SepsiArc – Research & Model Information",
    page_icon="📊",
    layout="wide",
)

render_header(
    page_title="Research & Model Information",
    page_subtitle="Model architecture, evaluation metrics, feature engineering, and dual-stage pipeline transparency.",
    icon="📊",
)
render_sidebar("models")
render_disclaimer_banner()

st.info(
    "All evaluation metrics are from the **project held-out test set** using "
    "**patient-level stratified splitting** (no patient-level data leakage). "
    "Results are for research transparency and are not live clinical performance measurements.",
    icon="ℹ️"
)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_all_metadata():
    mdir = os.path.join(_root, "models")
    # Stage 1 — XGBoost model
    with open(os.path.join(mdir, "Stage1_Final_Model_A_Metadata.json")) as f:
        s1_meta = json.load(f)
    s1_feat_imp = pd.read_csv(os.path.join(mdir, "Stage1_FeatureImportance.csv"))
    # Stage 2 — new XGBoost model
    with open(os.path.join(mdir, "stage2_xgb_metadata.json")) as f:
        s2_meta = json.load(f)
    with open(os.path.join(mdir, "stage2_xgb_summary.json")) as f:
        s2_summary = json.load(f)
    s2_results = s2_summary.get("metrics", {})
    s2_thresh   = float(s2_meta.get("decision_threshold", 0.29))
    s2_feats    = s2_meta.get("features", [])
    return s1_meta, s1_feat_imp, s2_results, s2_thresh, s2_feats, s2_meta

try:
    s1_meta, s1_feat_imp, s2_results, s2_thresh, s2_feats, s2_meta = load_all_metadata()
    meta_ok = True
except Exception as e:
    st.error(f"Could not load model metadata: {str(e)[:160]}")
    meta_ok = False

if not meta_ok:
    render_footer()
    st.stop()

s1_thresh = float(s1_meta.get("threshold", 0.143))

# ══════════════════════════════════════════════════════════════════════════════
# DUAL-STAGE PIPELINE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🔄 Dual-Stage Pipeline Overview")

st.markdown(f"""
<div style="background:#fff;border-radius:14px;padding:1.6rem 2rem;
            box-shadow:0 2px 12px rgba(0,0,0,0.08);font-size:0.83rem;
            color:{SLATE_700};line-height:2;">

  <b style="color:{NAVY};font-size:0.95rem;">Input</b><br>
  PSV or CSV file → one patient → multiple hourly ICU observation rows
  <br><br>

  <b style="color:{TEAL};font-size:0.95rem;">Stage 1 — Sepsis Detection</b><br>
  Raw file → drop sparse columns → chronological sort →
  18 missingness indicators + 18 recency features → causal forward fill →
  24 lag/change features → 48 rolling statistics → 5 clinical derived features →
  <b>135 total features</b> → XGBoost Model A →
  probability per hour → peak probability vs threshold (0.143) →
  SEPSIS ALERT / BORDERLINE / LOW RISK + SHAP explanation
  <br><br>

  <b style="color:{RED};font-size:0.95rem;">Stage 2 — Post-Sepsis Risk Stratification</b><br>
  (Accessible only after Stage 1 detects sepsis) →
  Same file → identify reference hour (first positive prediction) →
  compute patient-level aggregate features over pre-label and early post-label windows:
  mean, slope, SD, range, observed count, first-to-last change per vital →
  3 patient scalars (Age, HospAdmTime, ICULOS_at_label) →
  <b>65 total features</b> → XGBoost model (max_depth=2) →
  patient-level probability vs threshold (0.29) →
  HIGHER RISK / LOWER RISK + SHAP + LIME explanation
  <br><br>

  <b style="color:{NAVY};">Key Design Principles</b><br>
  • Patient-level data splitting — zero patient overlap between train/validation/test<br>
  • Causal forward fill only — no future-information leakage in time-series features<br>
  • Threshold selected on validation set only — test set untouched during selection<br>
  • Missingness encoded as features — absence of measurements carries clinical signal<br>
  • SHAP explanations use actual model internals — not approximations

</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🔬 Stage 1 — Sepsis Detection")

s1c1, s1c2, s1c3 = st.columns(3)

with s1c1:
    st.markdown(f"""
<div style="background:#fff;border-radius:14px;padding:1.4rem 1.6rem;
            box-shadow:0 2px 12px rgba(0,0,0,0.08);border-top:5px solid {TEAL};height:100%;">
  <div style="font-size:0.66rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1.5px;color:{SLATE_400};margin-bottom:0.4rem;">Algorithm</div>
  <div style="font-size:1.1rem;font-weight:800;color:{NAVY};">XGBoost</div>
  <div style="font-size:0.72rem;color:{SLATE_400};margin-top:2px;">
    Regularized Cost-Sensitive — Model A
  </div>

  <div style="margin-top:1.1rem;font-size:0.66rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1.5px;color:{SLATE_400};margin-bottom:0.3rem;">Decision Threshold</div>
  <div style="font-size:2rem;font-weight:800;color:{TEAL};">{s1_thresh:.3f}</div>
  <div style="font-size:0.72rem;color:{SLATE_400};">
    Probability ≥ {s1_thresh:.3f} → SEPSIS ALERT<br>
    Probability ≥ 0.08 → BORDERLINE
  </div>

  <div style="margin-top:1rem;font-size:0.66rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1.5px;color:{SLATE_400};margin-bottom:0.3rem;">Features</div>
  <div style="font-size:1.3rem;font-weight:700;color:{NAVY};">135 engineered features</div>

  <div style="margin-top:1rem;font-size:0.66rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1.5px;color:{SLATE_400};margin-bottom:0.3rem;">Hyperparameters</div>
  <div style="font-size:0.75rem;color:{SLATE_700};line-height:1.9;">
    max_depth = {s1_meta.get('max_depth', 4)}<br>
    min_child_weight = {s1_meta.get('min_child_weight', 5)}<br>
    gamma = {s1_meta.get('gamma', 0.1)}<br>
    reg_alpha = {s1_meta.get('reg_alpha', 0.1)}<br>
    reg_lambda = {s1_meta.get('reg_lambda', 2.0)}<br>
    scale_pos_weight = {s1_meta.get('scale_pos_weight', 3)}<br>
    learning_rate = {s1_meta.get('learning_rate', 0.03)}<br>
    subsample = {s1_meta.get('subsample', 0.8)}<br>
    n_estimators (max) = {s1_meta.get('n_estimators', 1200)}<br>
    early_stopping = {s1_meta.get('early_stopping_rounds', 60)} rounds
  </div>
</div>
""", unsafe_allow_html=True)

with s1c2:
    st.markdown(f"""
<div style="background:#fff;border-radius:14px;padding:1.4rem 1.6rem;
            box-shadow:0 2px 12px rgba(0,0,0,0.08);border-top:5px solid {TEAL};">
  <div style="font-size:0.66rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1.5px;color:{SLATE_400};margin-bottom:0.8rem;">
    Test Set Metrics (Held-Out)
  </div>
""", unsafe_allow_html=True)
    for metric, val, note in [
        ("AUROC",     0.7587, "Threshold-independent ranking"),
        ("PR-AUC",    0.1605, "Positive-class ranking (imbalanced)"),
        ("Recall",    0.7097, "70.97% of sepsis hours detected"),
        ("Precision", 0.1120, "Low due to high-recall strategy"),
        ("F1 Score",  0.1935, "Harmonic mean at threshold 0.143"),
        ("Accuracy",  0.6805, "Reduced by low-threshold strategy"),
    ]:
        bar_pct   = min(int(val * 100), 100)
        bar_color = GREEN if val >= 0.7 else AMBER if val >= 0.5 else RED
        st.markdown(f"""
<div style="margin-bottom:0.6rem;">
  <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
    <span style="font-size:0.78rem;color:{SLATE_700};font-weight:500;">{metric}</span>
    <span style="font-size:0.78rem;color:{NAVY};font-weight:700;">{val:.4f}</span>
  </div>
  <div style="background:{SLATE_200};border-radius:999px;height:6px;overflow:hidden;">
    <div style="background:{bar_color};width:{bar_pct}%;height:100%;border-radius:999px;"></div>
  </div>
  <div style="font-size:0.67rem;color:{SLATE_400};margin-top:1px;">{note}</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(f"""
<div style="background:#E0F7FA;border-radius:10px;padding:0.8rem 1rem;
            margin-top:0.8rem;font-size:0.75rem;color:#0D2137;line-height:1.7;">
  <b>Why is accuracy 68%?</b><br>
  The threshold was deliberately set to 0.143 (vs. standard 0.5) to maximise recall.
  At threshold 0.5, the model detects only ~2% of sepsis cases despite 94% accuracy.
  The project prioritises detecting more sepsis cases over overall classification accuracy.
</div>
""", unsafe_allow_html=True)

with s1c3:
    st.markdown(f"""
<div style="background:#fff;border-radius:14px;padding:1.4rem 1.6rem;
            box-shadow:0 2px 12px rgba(0,0,0,0.08);border-top:5px solid {TEAL};">
  <div style="font-size:0.66rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1.5px;color:{SLATE_400};margin-bottom:0.6rem;">Training Dataset</div>
  <div style="font-size:0.78rem;color:{SLATE_700};line-height:1.9;">
    <b>Source:</b> PhysioNet CinC Challenge 2019<br>
    <b>Patients (original):</b> 40,336<br>
    <b>Patients (after selection):</b> 11,932<br>
    <b>Train:</b> 7,636 patients / 320,296 rows<br>
    <b>Validation:</b> 1,909 patients / 80,593 rows<br>
    <b>Test:</b> 2,387 patients / 102,564 rows<br>
    <b>Sepsis rate (rows):</b> ~5.5%<br>
    <b>Split:</b> Patient-level stratified (64/16/20)
  </div>

  <div style="margin-top:1rem;font-size:0.66rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1.5px;color:{SLATE_400};margin-bottom:0.4rem;">
    Class Imbalance Handling
  </div>
  <div style="font-size:0.78rem;color:{SLATE_700};line-height:1.9;">
    <b>scale_pos_weight = 3</b> (cost-sensitive learning)<br>
    No SMOTE — class weighting applied at training<br>
    Threshold lowered to 0.143 for ≥70% recall target
  </div>

  <div style="margin-top:1rem;font-size:0.66rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1.5px;color:{SLATE_400};margin-bottom:0.4rem;">
    Validation PR-AUC
  </div>
  <div style="font-size:1.4rem;font-weight:800;color:{TEAL};">
    {s1_meta.get('validation_pr_auc', 0.1728):.4f}
  </div>
  <div style="font-size:0.72rem;color:{SLATE_400};">
    Best configuration (Model A) selected on this metric
  </div>
</div>
""", unsafe_allow_html=True)

# ── Stage 1 Confusion Matrix ───────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📊 Stage 1 — Confusion Matrix (Test Set, Threshold = 0.143)", expanded=False):
    st.markdown(f"""
<div style="max-width:520px;">
  <div style="font-size:0.78rem;color:{SLATE_700};margin-bottom:0.7rem;line-height:1.7;">
    <b>Test set: 102,564 hourly observations</b> · 5,539 sepsis-positive (5.4%) · 97,025 sepsis-negative (94.6%)<br>
    At threshold 0.143: <b>Sensitivity (Recall) = 70.97%</b> · <b>Specificity = 67.88%</b> · FDR = 88.80%
  </div>
  <table style="border-collapse:collapse;width:100%;font-size:0.82rem;text-align:center;">
    <thead>
      <tr>
        <th style="padding:6px 10px;"></th>
        <th style="padding:6px 10px;background:#E0F7FA;color:{NAVY};border-radius:6px 0 0 0;">Predicted Negative</th>
        <th style="padding:6px 10px;background:#FFF1F2;color:{RED};border-radius:0 6px 0 0;">Predicted Positive</th>
        <th style="padding:6px 10px;color:{SLATE_400};">Row Total</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:6px 10px;font-weight:700;color:{TEAL};text-align:left;">Actual Negative</td>
        <td style="padding:8px 10px;background:#E0F7FA;color:{NAVY};font-weight:800;font-size:1rem;">65,863<br><span style="font-size:0.68rem;font-weight:400;">TN</span></td>
        <td style="padding:8px 10px;background:#FEE2E2;color:{RED};font-weight:700;">31,162<br><span style="font-size:0.68rem;font-weight:400;">FP</span></td>
        <td style="padding:6px 10px;color:{SLATE_400};">97,025</td>
      </tr>
      <tr>
        <td style="padding:6px 10px;font-weight:700;color:{RED};text-align:left;">Actual Positive</td>
        <td style="padding:8px 10px;background:#FEE2E2;color:{RED};font-weight:700;">1,608<br><span style="font-size:0.68rem;font-weight:400;">FN</span></td>
        <td style="padding:8px 10px;background:#D1FAE5;color:#065F46;font-weight:800;font-size:1rem;">3,931<br><span style="font-size:0.68rem;font-weight:400;">TP</span></td>
        <td style="padding:6px 10px;color:{SLATE_400};">5,539</td>
      </tr>
    </tbody>
  </table>
  <div style="margin-top:0.7rem;font-size:0.72rem;color:{SLATE_400};line-height:1.6;">
    High FP count is expected: threshold 0.143 was deliberately chosen to achieve &ge;70% recall
    on the minority sepsis class. The application prioritises <b>not missing sepsis</b> over minimising false alarms.
  </div>
</div>
""", unsafe_allow_html=True)

# PR-AUC clarification note
st.markdown(f"""
<div style="background:#EFF6FF;border-radius:10px;padding:0.75rem 1rem;margin-top:0.5rem;
            font-size:0.76rem;color:#1E3A5F;line-height:1.7;">
  <b>PR-AUC note:</b> 0.1605 is the held-out <b>test set</b> PR-AUC.
  0.1728 is the <b>validation set</b> PR-AUC used during model selection.
  Both values describe Model A — the difference reflects the split (validation vs. test).
  Low PR-AUC is expected given the severe class imbalance (~5.4% positive rows).
</div>
""", unsafe_allow_html=True)

# ── Stage 1 Feature Engineering Breakdown ─────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🔧 Stage 1 — Feature Engineering Breakdown (135 features)", expanded=False):
    fe_cols = st.columns(4)
    groups = [
        ("Raw Clinical Inputs", "22 features",
         ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","BUN","Calcium","Creatinine",
          "Glucose","Lactate","Hct","Hgb","WBC","Platelets","Age","Gender",
          "HospAdmTime","ICULOS","Potassium","Magnesium"],
         TEAL, "#E0F7FA"),
        ("Missingness Indicators\n(×18)", "18 features",
         ["<var>_was_missing (binary)\nfor each of 18 clinical variables"],
         "#4338CA","#EEF2FF"),
        ("Time-Since-Last\nMeasurement (×18)", "18 features",
         ["<var>_hours_since_last\nhours since last observed value\nfor each of 18 variables"],
         "#7C3AED","#F5F3FF"),
        ("Lag & Change\nFeatures (×6 vars)", "24 features",
         ["HR_lag1, HR_lag3","HR_change1h, HR_change3h",
          "same for O2Sat,Temp,MAP,Resp,SBP"],
         AMBER,"#FEF3C7"),
    ]
    for col, (title, count, items, color, bg) in zip(fe_cols, groups):
        col.markdown(f"""
<div style="background:{bg};border-radius:10px;padding:0.9rem;border-top:3px solid {color};
            height:100%;font-size:0.75rem;">
  <div style="font-weight:700;color:{NAVY};margin-bottom:0.2rem;">{title}</div>
  <div style="font-size:0.7rem;color:{color};font-weight:700;margin-bottom:0.5rem;">{count}</div>
  {''.join(f'<div style="color:{SLATE_700};margin-bottom:1px;">• {i}</div>' for i in items)}
</div>""", unsafe_allow_html=True)

    fe_cols2 = st.columns(3)
    groups2 = [
        ("Rolling Window Stats\n(3h & 6h × 4 stats × 6 vars)", "48 features",
         ["mean, min, max, std over 3h and 6h",
          "for HR, O2Sat, Temp, MAP, Resp, SBP"],
         "#0891B2","#ECFEFF"),
        ("Clinical Derived", "5 features",
         ["Shock_Index = HR÷SBP",
          "Pulse_Pressure = SBP−DBP",
          "MAP_Ratio = MAP÷SBP",
          "SIRS_partial (0–3 score)",
          "Low_MAP (MAP<65 flag)"],
         RED,"#FFF1F2"),
        ("Dropped Sparse Columns", "Excluded from model",
         ["PTT, SaO2, Phosphate, HCO3","Chloride, EtCO2, BaseExcess",
          "PaCO2, pH, FiO2, Unit1, Unit2",
          "Bilirubin_direct, Fibrinogen","TroponinI, AST, Alkalinephos"],
         SLATE_400,"#F8FAFC"),
    ]
    for col, (title, count, items, color, bg) in zip(fe_cols2, groups2):
        col.markdown(f"""
<div style="background:{bg};border-radius:10px;padding:0.9rem;border-top:3px solid {color};
            height:100%;font-size:0.75rem;margin-top:0.6rem;">
  <div style="font-weight:700;color:{NAVY};margin-bottom:0.2rem;">{title}</div>
  <div style="font-size:0.7rem;color:{color};font-weight:700;margin-bottom:0.5rem;">{count}</div>
  {''.join(f'<div style="color:{SLATE_700};margin-bottom:1px;">• {i}</div>' for i in items)}
</div>""", unsafe_allow_html=True)

# ── Stage 1 Feature Importance ─────────────────────────────────────────────
with st.expander("📊 Stage 1 — Top Feature Importances", expanded=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = s1_feat_imp.head(20).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    ax.barh(top["Feature"], top["Importance"],
            color="#0E7490", alpha=0.82, edgecolor="white", height=0.65)
    ax.set_xlabel("Feature Importance (XGBoost gain)", fontsize=9)
    ax.set_title("Stage 1 — Top 20 Feature Importances", fontweight="bold",
                 fontsize=10, color=NAVY)
    ax.set_facecolor("#FAFAFA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    _buf_fi = __import__('io').BytesIO()
    fig.savefig(_buf_fi, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    _buf_fi.seek(0)
    st.image(_buf_fi, use_container_width=True)
    st.caption(
        "ICULOS (ICU length of stay) is the most influential feature — consistent with clinical "
        "evidence that sepsis risk increases over ICU stay. Lactate, Temp, and Shock_Index "
        "are the strongest physiological predictors."
    )

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🧪 Stage 2 — Post-Sepsis Risk Assessment")

st.info(
    "Stage 2 is a completely separate model from Stage 1 — different algorithm (XGBoost, "
    "patient-level), different feature contract (65 vs 135), different threshold (0.29 vs 0.143), "
    "and different training objective (risk stratification vs sepsis detection at hour level).",
    icon="🔬"
)

s2c1, s2c2, s2c3 = st.columns(3)

with s2c1:
    st.markdown(f"""
<div style="background:#fff;border-radius:14px;padding:1.4rem 1.6rem;
            box-shadow:0 2px 12px rgba(0,0,0,0.08);border-top:5px solid {RED};height:100%;">
  <div style="font-size:0.66rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1.5px;color:{SLATE_400};margin-bottom:0.4rem;">Algorithm</div>
  <div style="font-size:1.1rem;font-weight:800;color:{NAVY};">XGBoost</div>
  <div style="font-size:0.72rem;color:{SLATE_400};margin-top:2px;">
    Patient-Level Risk Stratification Model
  </div>

  <div style="margin-top:1.1rem;font-size:0.66rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1.5px;color:{SLATE_400};margin-bottom:0.3rem;">Decision Threshold</div>
  <div style="font-size:2rem;font-weight:800;color:{RED};">{s2_thresh:.2f}</div>
  <div style="font-size:0.72rem;color:{SLATE_400};">
    Probability ≥ {s2_thresh:.2f} → HIGHER RISK<br>
    Probability &lt; {s2_thresh:.2f} → LOWER RISK
  </div>

  <div style="margin-top:1rem;font-size:0.66rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1.5px;color:{SLATE_400};margin-bottom:0.3rem;">Features</div>
  <div style="font-size:1.3rem;font-weight:700;color:{NAVY};">65 engineered features</div>

  <div style="margin-top:1rem;background:#FEF3C7;border-radius:8px;padding:0.7rem;
              font-size:0.74rem;color:#92400E;line-height:1.7;">
    <b>Access Gate:</b> Stage 2 is accessible only after Stage 1 returns
    SEPSIS ALERT or BORDERLINE. Session-state enforces this workflow.
  </div>
</div>
""", unsafe_allow_html=True)

with s2c2:
    st.markdown(f"""
<div style="background:#fff;border-radius:14px;padding:1.4rem 1.6rem;
            box-shadow:0 2px 12px rgba(0,0,0,0.08);border-top:5px solid {RED};">
  <div style="font-size:0.66rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1.5px;color:{SLATE_400};margin-bottom:0.8rem;">
    Test Set Metrics (Held-Out)
  </div>
""", unsafe_allow_html=True)
    for metric, val, note in [
        ("ROC-AUC",     s2_results.get("ROC_AUC",   0.8294), "Strong patient-level discrimination"),
        ("PR-AUC",      s2_results.get("PR_AUC",    0.8001), "High-risk class precision-recall"),
        ("Recall",      s2_results.get("Recall",    0.8548), "85.5% of higher-risk patients detected"),
        ("Precision",   s2_results.get("Precision", 0.6235), "62.4% of flagged patients are true higher risk"),
        ("F1 Score",    s2_results.get("F1",        0.7211), "Balanced precision-recall at threshold 0.29"),
        ("Accuracy",    s2_results.get("Accuracy",  0.7148), "Overall patient-level accuracy"),
        ("Brier Score", s2_results.get("Brier_Score", 0.1659), "Calibration quality (lower = better; 0 = perfect)"),
        ("Log Loss",    s2_results.get("Log_Loss",    0.5021), "Probability calibration loss (lower = better)"),
    ]:
        bar_pct   = min(int(val * 100), 100)
        bar_color = GREEN if val >= 0.7 else AMBER if val >= 0.5 else RED
        st.markdown(f"""
<div style="margin-bottom:0.7rem;">
  <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
    <span style="font-size:0.78rem;color:{SLATE_700};font-weight:500;">{metric}</span>
    <span style="font-size:0.78rem;color:{NAVY};font-weight:700;">{val:.4f}</span>
  </div>
  <div style="background:{SLATE_200};border-radius:999px;height:6px;overflow:hidden;">
    <div style="background:{bar_color};width:{bar_pct}%;height:100%;border-radius:999px;"></div>
  </div>
  <div style="font-size:0.67rem;color:{SLATE_400};margin-top:1px;">{note}</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(f"""
<div style="background:#FFF1F2;border-radius:10px;padding:0.8rem 1rem;
            margin-top:0.8rem;font-size:0.75rem;color:#9F1239;line-height:1.7;">
  <b>Stage 2 label note:</b> Risk classification is a project-defined severity proxy
  derived from available clinical indicators in the PhysioNet dataset.
  It is not a validated long-term mortality or discharge outcome label.
</div>
""", unsafe_allow_html=True)

with s2c3:
    st.markdown(f"""
<div style="background:#fff;border-radius:14px;padding:1.4rem 1.6rem;
            box-shadow:0 2px 12px rgba(0,0,0,0.08);border-top:5px solid {RED};">
  <div style="font-size:0.66rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1.5px;color:{SLATE_400};margin-bottom:0.6rem;">
    Stage 2 vs Stage 1 — Key Differences
  </div>
  <table style="font-size:0.75rem;color:{SLATE_700};width:100%;border-collapse:collapse;">
    <tr style="border-bottom:1px solid #E2E8F0;">
      <td style="padding:5px 4px;font-weight:600;">Aspect</td>
      <td style="padding:5px 4px;color:{TEAL};font-weight:600;">Stage 1</td>
      <td style="padding:5px 4px;color:{RED};font-weight:600;">Stage 2</td>
    </tr>
    <tr style="border-bottom:1px solid #F1F5F9;">
      <td style="padding:5px 4px;">Algorithm</td>
      <td style="padding:5px 4px;">XGBoost (hourly)</td>
      <td style="padding:5px 4px;">XGBoost (patient)</td>
    </tr>
    <tr style="border-bottom:1px solid #F1F5F9;">
      <td style="padding:5px 4px;">Features</td>
      <td style="padding:5px 4px;">135</td>
      <td style="padding:5px 4px;">65</td>
    </tr>
    <tr style="border-bottom:1px solid #F1F5F9;">
      <td style="padding:5px 4px;">Threshold</td>
      <td style="padding:5px 4px;">0.143</td>
      <td style="padding:5px 4px;">0.29</td>
    </tr>
    <tr style="border-bottom:1px solid #F1F5F9;">
      <td style="padding:5px 4px;">ROC-AUC</td>
      <td style="padding:5px 4px;">0.7587</td>
      <td style="padding:5px 4px;">0.8294</td>
    </tr>
    <tr style="border-bottom:1px solid #F1F5F9;">
      <td style="padding:5px 4px;">Recall</td>
      <td style="padding:5px 4px;">70.97%</td>
      <td style="padding:5px 4px;">85.48%</td>
    </tr>
    <tr style="border-bottom:1px solid #F1F5F9;">
      <td style="padding:5px 4px;">Precision</td>
      <td style="padding:5px 4px;">11.2%</td>
      <td style="padding:5px 4px;">62.35%</td>
    </tr>
    <tr style="border-bottom:1px solid #F1F5F9;">
      <td style="padding:5px 4px;">Training scope</td>
      <td style="padding:5px 4px;">Row-level (hourly)</td>
      <td style="padding:5px 4px;">Patient-level</td>
    </tr>
    <tr>
      <td style="padding:5px 4px;">Access</td>
      <td style="padding:5px 4px;">Always</td>
      <td style="padding:5px 4px;">After S1 positive</td>
    </tr>
  </table>

  <div style="margin-top:1rem;font-size:0.66rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1.5px;color:{SLATE_400};margin-bottom:0.4rem;">
    Why is Stage 2 AUROC much higher?
  </div>
  <div style="font-size:0.75rem;color:{SLATE_700};line-height:1.7;">
    Stage 2 operates on the sepsis-positive subpopulation only —
    a more clinically homogeneous group than the full ICU cohort.
    The risk stratification task (high vs. low severity) is more separable
    than the detection task (sepsis vs. no sepsis in mixed population).
  </div>
</div>
""", unsafe_allow_html=True)

# ── Stage 2 Feature Breakdown ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

# Stage 2 Confusion Matrix
with st.expander("📊 Stage 2 — Confusion Matrix (Test Set, Threshold = 0.29)", expanded=False):
    st.markdown(f"""
<div style="max-width:520px;">
  <div style="font-size:0.78rem;color:{SLATE_700};margin-bottom:0.7rem;line-height:1.7;">
    <b>Test set: 575 patients</b> (sepsis-positive sub-cohort eligible for Stage 2)<br>
    248 Higher-Risk (43.1%) &middot; 327 Lower-Risk (56.9%)<br>
    At threshold 0.29: <b>Sensitivity (Recall) = 85.48%</b> &middot; <b>Specificity = 60.86%</b>
  </div>
  <table style="border-collapse:collapse;width:100%;font-size:0.82rem;text-align:center;">
    <thead>
      <tr>
        <th style="padding:6px 10px;"></th>
        <th style="padding:6px 10px;background:#E0F7FA;color:{NAVY};border-radius:6px 0 0 0;">Predicted Lower Risk</th>
        <th style="padding:6px 10px;background:#FFF1F2;color:{RED};border-radius:0 6px 0 0;">Predicted Higher Risk</th>
        <th style="padding:6px 10px;color:{SLATE_400};">Row Total</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:6px 10px;font-weight:700;color:{TEAL};text-align:left;">Actual Lower Risk</td>
        <td style="padding:8px 10px;background:#E0F7FA;color:{NAVY};font-weight:800;font-size:1rem;">199<br><span style="font-size:0.68rem;font-weight:400;">TN</span></td>
        <td style="padding:8px 10px;background:#FEE2E2;color:{RED};font-weight:700;">128<br><span style="font-size:0.68rem;font-weight:400;">FP</span></td>
        <td style="padding:6px 10px;color:{SLATE_400};">327</td>
      </tr>
      <tr>
        <td style="padding:6px 10px;font-weight:700;color:{RED};text-align:left;">Actual Higher Risk</td>
        <td style="padding:8px 10px;background:#FEE2E2;color:{RED};font-weight:700;">36<br><span style="font-size:0.68rem;font-weight:400;">FN</span></td>
        <td style="padding:8px 10px;background:#D1FAE5;color:#065F46;font-weight:800;font-size:1rem;">212<br><span style="font-size:0.68rem;font-weight:400;">TP</span></td>
        <td style="padding:6px 10px;color:{SLATE_400};">248</td>
      </tr>
    </tbody>
  </table>
  <div style="margin-top:0.7rem;font-size:0.72rem;color:{SLATE_400};line-height:1.6;">
    Threshold 0.29 was selected to maximise recall of the Higher-Risk class (85.5%).
    Stage 2 correctly identifies 212 of 248 Higher-Risk patients.
    FP rate of 39.1% reflects the conservative threshold favouring sensitivity over specificity.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🔧 Stage 2 — Feature Engineering Breakdown (65 features)", expanded=False):
    s2fe_cols = st.columns(3)
    s2_groups = [
        ("Patient Scalars", "3 features",
         ["Age",
          "HospAdmTime (hours before ICU)",
          "ICULOS_at_First_Sepsis_Label (reference hour)"],
         RED, "#FFF1F2"),
        ("Pre-Label Window Aggregates", "~31 features",
         ["Window: all hours before the Stage 1 reference hour",
          "Per vital (HR, SBP, MAP, Resp, O2Sat, Temp):",
          "  • Mean / Slope (linear trend)",
          "  • Observed Count / Mean / Min / Max",
          "  • Observed SD / Range",
          "  • FirstToLast_Change",
          "Cross-window: Early_vs_Pre_Change"],
         AMBER, "#FEF3C7"),
        ("Early Post-Label Window Aggregates", "~31 features",
         ["Window: hours at and after the reference hour",
          "Per vital (HR, SBP, MAP, Resp, O2Sat, Temp):",
          "  • Mean / Slope (linear trend)",
          "  • Observed Count / Mean / Min / Max",
          "  • Observed SD / Range",
          "  • FirstToLast_Change",
          "Imputed with training-set column means (SimpleImputer)"],
         "#7C3AED", "#F5F3FF"),
    ]
    for col, (title, count, items, color, bg) in zip(s2fe_cols, s2_groups):
        col.markdown(f"""
<div style="background:{bg};border-radius:10px;padding:0.9rem;border-top:3px solid {color};
            font-size:0.75rem;">
  <div style="font-weight:700;color:{NAVY};margin-bottom:0.2rem;">{title}</div>
  <div style="font-size:0.7rem;color:{color};font-weight:700;margin-bottom:0.5rem;">{count}</div>
  {''.join(f'<div style="color:{SLATE_700};margin-bottom:2px;">• {i}</div>' for i in items)}
</div>""", unsafe_allow_html=True)

    st.markdown(
        f"<div style='font-size:0.78rem;color:{SLATE_700};margin-top:0.6rem;line-height:1.7;'"
        f">Stage 2 operates at the <b>patient level</b> — one feature row per patient, "
        f"aggregated from all available ICU hours relative to the Stage 1 reference point. "
        f"This contrasts with Stage 1 which operates at the <b>hourly row level</b>.</div>",
        unsafe_allow_html=True,
    )
    with st.expander("📋 All 65 Stage 2 Feature Names", expanded=False):
        for f in s2_feats:
            st.markdown(
                f"<span style='font-size:0.8rem;font-family:monospace;background:#FFF1F2;"
                f"padding:2px 6px;border-radius:4px;margin:1px;display:inline-block;'>"
                f"{f}</span>",
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# REFERENCES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📚 References")
st.markdown(f"""
<div style="background:#fff;border-radius:14px;padding:1.4rem 1.8rem;
            box-shadow:0 2px 8px rgba(0,0,0,0.06);font-size:0.80rem;
            color:{SLATE_700};line-height:2;">

1. Reyna, M. A., et al. (2019). <i>Early Prediction of Sepsis from Clinical Data —
   PhysioNet/CinC Challenge 2019.</i> Critical Care Medicine.

2. Singer, M., et al. (2016). <i>Third International Consensus Definitions for
   Sepsis and Septic Shock (Sepsis-3).</i> JAMA, 315(8), 801–810.

3. Lundberg, S. M., & Lee, S.-I. (2017). <i>A Unified Approach to Interpreting
   Model Predictions (SHAP).</i> NeurIPS.

4. Chen, T., & Guestrin, C. (2016). <i>XGBoost: A Scalable Tree Boosting System.</i>
   KDD 2016.

5. Ke, G., et al. (2017). <i>LightGBM: A Highly Efficient Gradient Boosting
   Decision Tree.</i> NeurIPS.

6. Surviving Sepsis Campaign. (2018). <i>Hour-1 Bundle of Care.</i>
   Society of Critical Care Medicine.

</div>
""", unsafe_allow_html=True)

render_footer()
