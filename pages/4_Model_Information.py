"""
pages/4_Model_Information.py  --  SepsiArc Model Information
Transparency page: model names, versions, thresholds, metrics, feature contracts.
"""
import streamlit as st
import json, joblib, os, sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from utils.ui_components import (
    render_header, render_footer, render_sidebar,
    render_disclaimer_banner,
    NAVY, TEAL, GREEN, AMBER, RED, SLATE_200, SLATE_400, SLATE_700,
)

st.set_page_config(
    page_title="SepsiArc – Model Information",
    page_icon="📊",
    layout="wide",
)

render_header(
    page_title="Model Information",
    page_subtitle="Research transparency: model names, thresholds, evaluation metrics, and feature contracts.",
    icon="📊",
)
render_sidebar("models")
render_disclaimer_banner()

st.info(
    "Model information is provided for research and transparency purposes. "
    "Evaluation metrics are from the **project test set** and are not live clinical performance measurements.",
    icon="ℹ️"
)

# ── Load model metadata ────────────────────────────────────────────────────────
@st.cache_data
def load_model_metadata():
    mdir = "models"
    s1_results = json.load(open(os.path.join(mdir, "model_results.json")))
    s2_results = json.load(open(os.path.join(mdir, "stage2_results.json")))
    best_name  = open(os.path.join(mdir, "best_model_name.txt")).read().strip()
    s1_thresh  = float(joblib.load(os.path.join(mdir, "sepsis_threshold.pkl")))
    s2_thresh  = float(joblib.load(os.path.join(mdir, "stage2_threshold.pkl")))
    s1_feats   = json.load(open(os.path.join(mdir, "feature_names.json")))
    s2_feats   = json.load(open(os.path.join(mdir, "stage2_feature_names.json")))
    gate       = json.load(open(os.path.join(mdir, "clinical_gate_params.json")))
    return s1_results, s2_results, best_name, s1_thresh, s2_thresh, s1_feats, s2_feats, gate

try:
    s1_results, s2_results, best_name, s1_thresh, s2_thresh, s1_feats, s2_feats, gate = load_model_metadata()
    meta_ok = True
except Exception as e:
    st.error(f"Could not load model metadata: {str(e)[:120]}")
    meta_ok = False

if not meta_ok:
    render_footer()
    st.stop()

s1_best = s1_results.get(best_name, next(iter(s1_results.values())))

# ── STAGE 1 MODEL ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"## Stage 1 — Sepsis Detection")

s1c1, s1c2, s1c3 = st.columns(3)

with s1c1:
    st.markdown(f"""
<div style="background:#fff; border-radius:14px; padding:1.4rem 1.6rem;
            box-shadow:0 2px 12px rgba(0,0,0,0.08); border-top:5px solid {TEAL};">
  <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:{SLATE_400}; margin-bottom:0.6rem;">Model</div>
  <div style="font-size:1.1rem; font-weight:700; color:{NAVY};">{best_name}</div>
  <div style="font-size:0.72rem; color:{SLATE_400}; margin-top:4px;">Deployed Stage 1 artifact</div>

  <div style="margin-top:1rem; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:{SLATE_400}; margin-bottom:0.4rem;">Purpose</div>
  <div style="font-size:0.85rem; color:{SLATE_700};">Early sepsis detection from ICU clinical data</div>

  <div style="margin-top:1rem; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:{SLATE_400}; margin-bottom:0.4rem;">Decision Threshold</div>
  <div style="font-size:1.6rem; font-weight:800; color:{TEAL};">{s1_thresh:.2f}</div>
  <div style="font-size:0.72rem; color:{SLATE_400};">Probability ≥ {s1_thresh:.2f} → SEPSIS POSITIVE</div>

  <div style="margin-top:1rem; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:{SLATE_400}; margin-bottom:0.4rem;">Feature Count</div>
  <div style="font-size:1.3rem; font-weight:700; color:{NAVY};">{len(s1_feats)} features</div>
</div>
""", unsafe_allow_html=True)

with s1c2:
    st.markdown(f"""
<div style="background:#fff; border-radius:14px; padding:1.4rem 1.6rem;
            box-shadow:0 2px 12px rgba(0,0,0,0.08); border-top:5px solid {TEAL};">
  <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:{SLATE_400}; margin-bottom:0.8rem;">
    Evaluation Metrics (Project Test Set)
  </div>
""", unsafe_allow_html=True)
    for metric, val in [
        ("AUROC",     s1_best.get("AUROC", 0)),
        ("AUPRC",     s1_best.get("AUPRC", 0)),
        ("F1 Score",  s1_best.get("F1", 0)),
        ("Precision", s1_best.get("Precision", 0)),
        ("Recall",    s1_best.get("Recall", 0)),
    ]:
        bar_pct = min(int(val * 100), 100)
        bar_color = GREEN if val >= 0.7 else AMBER if val >= 0.5 else RED
        st.markdown(f"""
<div style="margin-bottom:0.7rem;">
  <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
    <span style="font-size:0.78rem; color:{SLATE_700}; font-weight:500;">{metric}</span>
    <span style="font-size:0.78rem; color:{NAVY}; font-weight:700;">{val:.4f}</span>
  </div>
  <div style="background:{SLATE_200}; border-radius:999px; height:6px; overflow:hidden;">
    <div style="background:{bar_color}; width:{bar_pct}%; height:100%; border-radius:999px;"></div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with s1c3:
    st.markdown(f"""
<div style="background:#fff; border-radius:14px; padding:1.4rem 1.6rem;
            box-shadow:0 2px 12px rgba(0,0,0,0.08); border-top:5px solid {TEAL};">
  <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:{SLATE_400}; margin-bottom:0.6rem;">Clinical Gating Parameters</div>
  <div style="font-size:0.78rem; color:{SLATE_700}; line-height:1.8;">
    <b>Purpose:</b> Reduce false alarms when no clinical sepsis criteria are met.<br>
    <b>Condition:</b> SIRS = 0 AND qSOFA = 0 AND Lactate risk = 0 AND Shock Index &lt; {gate.get('shock_index_max_for_gate', 0.75)}<br>
    <b>Action:</b> Cap probability at {gate.get('cap_probability', 0.10):.0%} when all gate conditions met<br><br>
    <b>Gate loaded from:</b> <code>clinical_gate_params.json</code>
  </div>
  <div style="margin-top:1rem; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:{SLATE_400}; margin-bottom:0.4rem;">Training Dataset</div>
  <div style="font-size:0.78rem; color:{SLATE_700}; line-height:1.8;">
    PhysioNet CinC Challenge 2019<br>
    40,336 ICU patients<br>
    1.55M hourly observations<br>
    Sepsis rate: ~1.8%
  </div>
</div>
""", unsafe_allow_html=True)

# ── Stage 1 model comparison ──────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📈 Stage 1 — All Model Comparison (Project Evaluation)", expanded=False):
    st.caption("All models evaluated on the same held-out test set. The deployed model is highlighted.")
    rows = []
    for name, metrics in s1_results.items():
        rows.append({
            "Model":      name,
            "AUROC":      f"{metrics.get('AUROC', 0):.4f}",
            "AUPRC":      f"{metrics.get('AUPRC', 0):.4f}",
            "F1":         f"{metrics.get('F1', 0):.4f}",
            "Precision":  f"{metrics.get('Precision', 0):.4f}",
            "Recall":     f"{metrics.get('Recall', 0):.4f}",
            "Threshold":  f"{metrics.get('Threshold', 0):.4f}",
            "Deployed":   "✅ Deployed" if name == best_name else "",
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Note: These metrics include SMOTE, calibration, and other pipeline variants.")

# ── STAGE 1 FEATURE CONTRACT ─────────────────────────────────────────────────
with st.expander("🔍 Stage 1 — Feature Contract (79 features)", expanded=False):
    # Split into clinical inputs vs derived
    raw_inputs = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
        "Creatinine", "WBC", "Platelets", "Glucose", "Lactate",
        "Age", "Gender", "ICULOS",
        "Unit1", "Unit2", "HospAdmTime",
        "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2",
        "BUN", "Calcium", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "Hct", "Hgb", "PTT",
    ]
    derived = [f for f in s1_feats if f not in raw_inputs]

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown(f"**Clinical Inputs ({len(raw_inputs)} fields)**")
        st.caption("Entered by user or populated with clinical defaults")
        for i, f in enumerate(raw_inputs):
            st.markdown(f"<span style='font-size:0.8rem; font-family:monospace; background:#F1F5F9; padding:2px 6px; border-radius:4px; margin:1px; display:inline-block;'>{f}</span>", unsafe_allow_html=True)
    with fc2:
        st.markdown(f"**Derived Features ({len(derived)} features)**")
        st.caption("Automatically calculated by SepsiArc — not editable by user")
        for f in derived:
            st.markdown(f"<span style='font-size:0.8rem; font-family:monospace; background:#ECFDF5; padding:2px 6px; border-radius:4px; margin:1px; display:inline-block;'>{f}</span>", unsafe_allow_html=True)


# ── STAGE 2 MODEL ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"## Stage 2 — Post-Sepsis Risk Assessment")

st.info(
    "Stage 2 is a completely separate model from Stage 1. "
    "It uses a different feature contract, different threshold, and different model artifact.",
    icon="🔬"
)

s2c1, s2c2 = st.columns([2, 3])

with s2c1:
    st.markdown(f"""
<div style="background:#fff; border-radius:14px; padding:1.4rem 1.6rem;
            box-shadow:0 2px 12px rgba(0,0,0,0.08); border-top:5px solid {RED};">
  <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:{SLATE_400}; margin-bottom:0.4rem;">Model</div>
  <div style="font-size:1.1rem; font-weight:700; color:{NAVY};">LightGBM</div>
  <div style="font-size:0.72rem; color:{SLATE_400}; margin-top:4px;">Deployed Stage 2 artifact (stage2_model.pkl)</div>

  <div style="margin-top:1rem; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:{SLATE_400}; margin-bottom:0.3rem;">Purpose</div>
  <div style="font-size:0.85rem; color:{SLATE_700};">Post-sepsis risk stratification for eligible patients</div>

  <div style="margin-top:1rem; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:{SLATE_400}; margin-bottom:0.3rem;">Decision Threshold</div>
  <div style="font-size:1.6rem; font-weight:800; color:{RED};">{s2_thresh:.2f}</div>
  <div style="font-size:0.72rem; color:{SLATE_400};">Probability ≥ {s2_thresh:.2f} → HIGH RISK</div>

  <div style="margin-top:1rem; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:{SLATE_400}; margin-bottom:0.3rem;">Feature Count</div>
  <div style="font-size:1.3rem; font-weight:700; color:{NAVY};">{len(s2_feats)} features</div>

  <div style="margin-top:1rem; background:#FEF3C7; border-radius:8px; padding:0.7rem; font-size:0.74rem; color:#92400E; line-height:1.6;">
    <b>Eligibility:</b> Stage 2 is accessible only after a Stage 1 POSITIVE result.
    The session-state gate enforces this workflow.
  </div>
</div>
""", unsafe_allow_html=True)

with s2c2:
    st.markdown(f"""
<div style="background:#fff; border-radius:14px; padding:1.4rem 1.6rem;
            box-shadow:0 2px 12px rgba(0,0,0,0.08); border-top:5px solid {RED};">
  <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:{SLATE_400}; margin-bottom:0.8rem;">
    Evaluation Metrics (Project Test Set)
  </div>
""", unsafe_allow_html=True)
    s2_metric_pairs = [
        ("AUROC",     s2_results.get("AUROC", 0)),
        ("F1 Score",  s2_results.get("F1", 0)),
        ("Precision", s2_results.get("Precision", 0)),
        ("Recall",    s2_results.get("Recall", 0)),
    ]
    m1, m2 = st.columns(2)
    for i, (metric, val) in enumerate(s2_metric_pairs):
        col = m1 if i % 2 == 0 else m2
        col.metric(metric, f"{val:.4f}")

    st.markdown(f"""
<div style="margin-top:0.8rem; background:#FFF1F2; border-radius:8px; padding:0.8rem 1rem; font-size:0.76rem; color:#9F1239; line-height:1.7;">
  <b>Stage 2 Label Disclaimer:</b><br>
  Stage 2 risk is a project-defined proxy severity/risk classification derived from available
  clinical indicators. It is <b>not</b> a validated long-term mortality or survivorship prediction.
  The PhysioNet dataset does not provide a validated long-term survivor outcome label.
</div>
""", unsafe_allow_html=True)

with st.expander("🔍 Stage 2 — Feature Contract (65 features)", expanded=False):
    for f in s2_feats:
        st.markdown(f"<span style='font-size:0.8rem; font-family:monospace; background:#FFF1F2; padding:2px 6px; border-radius:4px; margin:1px; display:inline-block;'>{f}</span>", unsafe_allow_html=True)

# ── PIPELINE ARCHITECTURE ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Model Pipeline Architecture")

st.markdown(f"""
<div style="background:#fff; border-radius:14px; padding:1.6rem 1.8rem;
            box-shadow:0 2px 12px rgba(0,0,0,0.08); font-size:0.82rem; color:{SLATE_700}; line-height:1.9;">
  <b style="color:{NAVY};">Stage 1 Pipeline:</b><br>
  Raw clinical inputs (15) → Feature engineering (79 features derived) →
  Clinical gating check → LightGBM-Calibrated prediction → Probability → Threshold (0.33) →
  SEPSIS POSITIVE / NEGATIVE → SHAP explanation<br><br>

  <b style="color:{NAVY};">Stage 2 Pipeline:</b><br>
  Stage 1 POSITIVE eligibility gate → Raw inputs → Feature engineering (65 features) →
  LightGBM (Stage 2) prediction → Probability → Threshold (0.58) →
  HIGH RISK / LOW RISK → SHAP explanation<br><br>

  <b style="color:{NAVY};">Key principles:</b><br>
  • Models are loaded once at application startup and cached<br>
  • Feature contract is verified before every prediction (count + names + order)<br>
  • Stage 1 and Stage 2 are completely separate model artifacts<br>
  • Clinical gating is loaded from configuration, not hardcoded<br>
  • SHAP explanations use the actual model, not simulated values
</div>
""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
render_footer()
