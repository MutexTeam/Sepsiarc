"""
pages/1_Stage_1_Detection.py
SepsiArc — Stage 1: Sepsis Detection
Model: Regularized Cost-Sensitive XGBoost (Model A)
Threshold: 0.143 | Features: 135 | AUROC: 0.7587 | Recall: 70.97%
Input: PSV / CSV file upload (hourly patient timeline)
"""

import os, io, json, warnings, base64
import numpy as np
import pandas as pd
import os, sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp')  # ensure matplotlib cache is writable on Cloud
import matplotlib
matplotlib.use('Agg')  # must be before any other matplotlib/pyplot import
import matplotlib.pyplot as plt
import streamlit as st
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Project root & page config ────────────────────────────────────────────────
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(
    page_title="Stage 1 — Sepsis Detection | SepsiArc",
    page_icon="🔬",
    layout="wide",
)

from utils.ui_components import (
    render_header, render_sidebar, render_footer,
    render_disclaimer_banner,
    NAVY, TEAL, SLATE_400, RED, GREEN, AMBER,
    risk_badge_html,
)

render_header(
    page_title="Stage 1 — Sepsis Detection",
    page_subtitle="Upload a patient's hourly ICU timeline (PSV/CSV) for sepsis screening.",
    icon="🔬",
    active="stage1",
)
render_sidebar("stage1")
render_disclaimer_banner()

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH    = os.path.join(_root, "models", "Stage1_Final_Model_A.json")
META_PATH     = os.path.join(_root, "models", "Stage1_Final_Model_A_Metadata.json")
FEAT_IMP_PATH = os.path.join(_root, "models", "Stage1_FeatureImportance.csv")

@st.cache_resource(show_spinner="Loading Stage 1 model…")
def load_stage1():
    bst = xgb.Booster()
    bst.load_model(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    feature_names = bst.feature_names          # exact 135-feature list
    threshold     = float(meta.get("threshold", 0.143))
    return bst, feature_names, threshold, meta

try:
    s1_model, S1_FEATURES, S1_THRESH, S1_META = load_stage1()
    model_ok = True
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    model_ok = False

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING PIPELINE  (replicates training notebook exactly)
# ══════════════════════════════════════════════════════════════════════════════

# 18 variables used for missingness indicators, recency, forward fill
_FE_VARS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    "Glucose", "Creatinine", "WBC", "Platelets", "BUN",
    "Calcium", "Potassium", "Magnesium", "Hgb", "Hct", "Lactate",
]

# 6 variables used for lag/change and rolling features
_TEMPORAL_VARS = ["HR", "O2Sat", "Temp", "MAP", "Resp", "SBP"]

# Columns removed in training (sparse / non-clinical)
_DROP_RAW = [
    "Unnamed: 0",
    "Bilirubin_direct", "Fibrinogen", "TroponinI",
    "Bilirubin_total", "Alkalinephos", "AST",
    "PTT", "SaO2", "Phosphate", "HCO3", "Chloride",
    "EtCO2", "BaseExcess", "PaCO2", "pH", "FiO2",
    "Unit1", "Unit2",
    # columns not in the 135-feature contract
    "Maxglucose", "Minglucose",
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates the full feature-engineering pipeline from the training notebook.
    Input:  raw patient PSV/CSV as a DataFrame (one row = one ICU hour)
    Output: DataFrame ready for XGBoost inference (columns = S1_FEATURES order)
    """
    df = df.copy()

    # ── Ensure ICULOS exists (use row index as fallback) ───────────────────
    if "ICULOS" not in df.columns:
        df["ICULOS"] = range(1, len(df) + 1)
    df = df.sort_values("ICULOS").reset_index(drop=True)

    # ── Drop unneeded raw columns ──────────────────────────────────────────
    for c in _DROP_RAW:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # ── Step 3: Missingness indicators ────────────────────────────────────
    for v in _FE_VARS:
        if v in df.columns:
            df[f"{v}_was_missing"] = df[v].isna().astype(int)
        else:
            df[f"{v}_was_missing"] = 1          # column absent → always missing

    # ── Step 4: Time since last measurement ───────────────────────────────
    for v in _FE_VARS:
        col = f"{v}_hours_since_last"
        if v in df.columns:
            last = df[v].notna().cumsum()
            # hours_since = current_ICULOS - ICULOS of last valid obs
            last_hour = df["ICULOS"].where(df[v].notna()).ffill()
            df[col] = df["ICULOS"] - last_hour
            df[col] = df[col].where(df[v].notna() | (last > 0))
        else:
            df[col] = np.nan

    # ── Step 5: Causal forward fill ───────────────────────────────────────
    vital_cols   = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
    slow_labs    = ["Creatinine", "WBC", "Platelets", "BUN",
                    "Calcium", "Potassium", "Magnesium", "Hgb", "Hct"]
    glucose_cols = ["Glucose"]
    lactate_cols = ["Lactate"]

    for col_group, limit in [
        (vital_cols,   4),
        (glucose_cols, 4),
        (slow_labs,    4),
        (lactate_cols, 3),
    ]:
        for c in col_group:
            if c in df.columns:
                df[c] = df[c].ffill(limit=limit)

    # ── Step 6: Lag and change features ───────────────────────────────────
    for v in _TEMPORAL_VARS:
        if v in df.columns:
            df[f"{v}_lag1"]     = df[v].shift(1)
            df[f"{v}_lag3"]     = df[v].shift(3)
            df[f"{v}_change1h"] = df[v] - df[f"{v}_lag1"]
            df[f"{v}_change3h"] = df[v] - df[f"{v}_lag3"]
        else:
            for suf in ["_lag1","_lag3","_change1h","_change3h"]:
                df[f"{v}{suf}"] = np.nan

    # ── Step 7: Rolling temporal features ─────────────────────────────────
    for v in _TEMPORAL_VARS:
        for w in [3, 6]:
            tag = f"{v}_rolling{w}h"
            if v in df.columns:
                r = df[v].rolling(w, min_periods=1)
                df[f"{tag}_mean"] = r.mean()
                df[f"{tag}_min"]  = r.min()
                df[f"{tag}_max"]  = r.max()
                df[f"{tag}_std"]  = r.std(ddof=0)
            else:
                for s in ["_mean","_min","_max","_std"]:
                    df[f"{tag}{s}"] = np.nan

    # ── Step 8: Clinical derived features ─────────────────────────────────
    hr  = df.get("HR",  pd.Series(np.nan, index=df.index))
    sbp = df.get("SBP", pd.Series(np.nan, index=df.index))
    dbp = df.get("DBP", pd.Series(np.nan, index=df.index))
    map_ = df.get("MAP", pd.Series(np.nan, index=df.index))
    temp = df.get("Temp", pd.Series(np.nan, index=df.index))
    resp = df.get("Resp", pd.Series(np.nan, index=df.index))

    df["Shock_Index"]    = hr / sbp.replace(0, np.nan)
    df["Pulse_Pressure"] = sbp - dbp
    df["MAP_Ratio"]      = map_ / sbp.replace(0, np.nan)
    df["SIRS_partial"]   = (
        (hr   > 90).astype(float) +
        ((temp < 36) | (temp > 38)).astype(float) +
        (resp  > 20).astype(float)
    ).fillna(0)
    df["Low_MAP"] = (map_ < 65).astype(float)

    # Clip infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df

def build_model_input(df_engineered: pd.DataFrame) -> pd.DataFrame:
    """
    Select and order exactly the 135 model features.
    Missing columns are filled with NaN (handled by median imputation below).
    Column order matches what the model was trained on.
    """
    for feat in S1_FEATURES:
        if feat not in df_engineered.columns:
            df_engineered[feat] = np.nan

    X = df_engineered[S1_FEATURES].copy()

    # ── Mean imputation: use column median from patient's own data ─────────
    # (training-set imputer not saved separately; patient-wise median is
    #  the correct inference-time fallback for a single-patient timeline)
    col_medians = X.median()
    X = X.fillna(col_medians)

    # Fallback: any remaining NaN → 0
    X = X.fillna(0)
    return X

# ══════════════════════════════════════════════════════════════════════════════
# FILE PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_patient_file(uploaded_file) -> pd.DataFrame:
    """
    Parse a PSV (|) or CSV (,) patient file into a DataFrame.
    Detects separator automatically. Column order in the file does NOT matter
    — columns are matched by name.
    """
    content = uploaded_file.read().decode("utf-8", errors="replace")
    uploaded_file.seek(0)

    # Detect separator: if first line has more | than ,
    first_line = content.split("\n")[0]
    sep = "|" if first_line.count("|") >= first_line.count(",") else ","

    import io
    df = pd.read_csv(io.StringIO(content), sep=sep, low_memory=False)
    df.columns = df.columns.str.strip()

    # Convert all numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ══════════════════════════════════════════════════════════════════════════════
# UI — Info banner
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div style="background:#E0F7FA;border-radius:10px;padding:0.9rem 1.4rem;'
    'border-left:4px solid #0E7490;margin-bottom:1.2rem;">'
    '<b style="color:#0D2137;">📋 How to use Stage 1</b><br>'
    '<span style="font-size:0.82rem;color:#334155;">'
    'Upload a <b>PSV</b> (pipe-separated) or <b>CSV</b> (comma-separated) file '
    'containing one patient\'s hourly ICU observations. Each row = one hour. '
    'The model analyses the full timeline and screens for sepsis risk at each hour.'
    '</span></div>',
    unsafe_allow_html=True,
)

# ── Model info card ────────────────────────────────────────────────────────
with st.expander("ℹ️ Stage 1 Model Information", expanded=False):
    mi1, mi2, mi3, mi4 = st.columns(4)
    mi1.metric("Algorithm",  "XGBoost")
    mi2.metric("Threshold",  f"{S1_THRESH:.3f}" if model_ok else "–")
    mi3.metric("Features",   "135")
    mi4.metric("Test Recall", "70.97%")
    st.caption(
        "Regularized Cost-Sensitive XGBoost — Model A. "
        "Trained on PhysioNet 2019 ICU data with patient-level stratified splitting. "
        f"Hyperparameters: max_depth=4, scale_pos_weight=3, learning_rate=0.03. "
        "Threshold selected to achieve ≥70% recall on validation set. "
        "Test AUROC 0.7587 · PR-AUC 0.1605 · Accuracy 68.05% · F1 19.35%."
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📂 Upload Patient Data File")
uploaded = st.file_uploader(
    "Upload a PSV or CSV file — one patient, hourly rows",
    type=["psv", "csv", "txt"],
    help="Each row must be one hourly ICU observation. "
         "Columns are matched by name — column order does not matter. "
         "Required columns: ICULOS (ICU hour), plus any available vital signs / labs.",
)

if not model_ok:
    st.warning("Model is not loaded. Please check that `models/Stage1_Final_Model_A.json` exists.")
    render_footer()
    st.stop()

if uploaded is None:
    st.info("⬆️ Upload a patient file above to begin the sepsis screening analysis.")
    # Show expected columns
    with st.expander("📋 Expected column names (any subset is acceptable)", expanded=False):
        expected_cols = [
            "ICULOS","HR","O2Sat","Temp","SBP","MAP","DBP","Resp",
            "BUN","Calcium","Creatinine","Glucose","Lactate","Magnesium",
            "Potassium","Hct","Hgb","WBC","Platelets","Age","Gender",
            "HospAdmTime","SepsisLabel",
        ]
        st.code("|".join(expected_cols), language=None)
        st.caption(
            "PSV files use `|` as separator. CSV files use `,`. "
            "Columns can be in any order — the model finds them by name. "
            "Columns not recognised are safely ignored."
        )
    render_footer()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PARSE → ENGINEER → PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Parsing file…"):
    try:
        raw_df = parse_patient_file(uploaded)
    except Exception as e:
        st.error(f"Could not parse file: {e}")
        render_footer()
        st.stop()

n_rows = len(raw_df)
if n_rows == 0:
    st.error("The uploaded file appears to be empty.")
    render_footer()
    st.stop()

# Show raw file preview
with st.expander(f"📄 Raw file preview ({n_rows} hourly observations, {len(raw_df.columns)} columns)", expanded=False):
    st.dataframe(raw_df.head(10), use_container_width=True)
    missing_pct = raw_df.isnull().mean() * 100
    miss_df = missing_pct[missing_pct > 0].sort_values(ascending=False).reset_index()
    miss_df.columns = ["Column", "Missing %"]
    if len(miss_df):
        st.caption("Columns with missing values:")
        st.dataframe(miss_df, use_container_width=True)

# ── Feature engineering ────────────────────────────────────────────────────
with st.spinner("Engineering 135 clinical features…"):
    try:
        eng_df  = engineer_features(raw_df)
        model_X = build_model_input(eng_df)
    except Exception as e:
        st.error(f"Feature engineering failed: {e}")
        st.exception(e)
        render_footer()
        st.stop()

# ── Prediction ────────────────────────────────────────────────────────────
with st.spinner("Running XGBoost sepsis screening…"):
    try:
        dmat   = xgb.DMatrix(model_X, feature_names=S1_FEATURES)
        probs  = s1_model.predict(dmat)          # probability array (n_hours,)
        labels = (probs >= S1_THRESH).astype(int)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)
        render_footer()
        st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# BUILD RESULTS TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
hours = eng_df["ICULOS"].values if "ICULOS" in eng_df.columns else np.arange(1, n_rows + 1)

result_df = pd.DataFrame({
    "Hour (ICULOS)":   hours.astype(int),
    "Sepsis Prob (%)": (probs * 100).round(2),
    "Prediction":      ["⚠️ ALERT" if l else "✅ LOW RISK" for l in labels],
})

# True labels if present
if "SepsisLabel" in raw_df.columns:
    result_df["True Label"] = raw_df["SepsisLabel"].values

# Overall patient-level assessment
peak_prob     = float(probs.max())
peak_hour_idx = int(probs.argmax())
peak_hour     = int(hours[peak_hour_idx])
n_alerts      = int(labels.sum())
first_alert_h = int(hours[labels == 1][0]) if n_alerts > 0 else None
patient_high  = peak_prob >= S1_THRESH

# Store for Stage 2
st.session_state["s1_probability"]   = peak_prob
st.session_state["s1_positive"]      = patient_high
st.session_state["s1_borderline"]    = (not patient_high) and (peak_prob >= 0.08)
st.session_state["s1_result_df"]     = result_df
st.session_state["s1_peak_hour"]     = peak_hour
st.session_state["s1_n_hours"]       = n_rows
st.session_state["s1_first_alert_h"] = first_alert_h
st.session_state["s1_n_alerts"]      = n_alerts
# Store the raw dataframe for Stage 2 auto-inheritance (avoids re-upload)
_s1_file_name = uploaded.name
st.session_state["s1_raw_df"]        = raw_df.copy()
st.session_state["s1_patient_id"]    = _s1_file_name.replace(".psv","").replace(".csv","").replace(".txt","")

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🩺 Sepsis Screening Results")

# ── Overall risk card ─────────────────────────────────────────────────────
res_color = RED if patient_high else (AMBER if st.session_state["s1_borderline"] else GREEN)
res_label = "SEPSIS ALERT" if patient_high else ("BORDERLINE" if st.session_state["s1_borderline"] else "LOW RISK")

card_col, summary_col = st.columns([2, 3])
with card_col:
    st.markdown(f"""
<div style="background:#fff;border-radius:14px;padding:1.4rem 1.8rem;
            box-shadow:0 3px 16px rgba(0,0,0,0.09);border-left:6px solid {res_color};">
  <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1.5px;color:{SLATE_400};margin-bottom:0.4rem;">
    Peak Sepsis Probability
  </div>
  <div style="font-size:3.5rem;font-weight:800;color:{res_color};line-height:1;">
    {peak_prob * 100:.1f}%
  </div>
  <div style="margin:0.7rem 0 0.5rem;">
    {risk_badge_html(res_label, 'high' if patient_high else ('medium' if st.session_state['s1_borderline'] else 'low'))}
  </div>
  <div style="font-size:0.75rem;color:{SLATE_400};margin-top:0.5rem;">
    Threshold: {S1_THRESH:.3f} &nbsp;·&nbsp; Hours analysed: {n_rows}
    {'&nbsp;·&nbsp; First alert: Hour ' + str(first_alert_h) if first_alert_h else ''}
  </div>
</div>
""", unsafe_allow_html=True)
    st.progress(min(peak_prob, 1.0))

    # Hour-level summary metrics
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Alert Hours", f"{n_alerts} / {n_rows}")
    m2.metric("Peak at Hour", str(peak_hour))
    if first_alert_h:
        m3, m4 = st.columns(2)
        m3.metric("First Alert Hour", str(first_alert_h))
        m4.metric("Alert Rate", f"{n_alerts/n_rows*100:.1f}%")

with summary_col:
    if patient_high:
        st.error("""**⚠️ Sepsis Alert — Elevated Risk Detected**

The patient's hourly ICU timeline contains observations where the predicted sepsis probability 
exceeds the clinical screening threshold (0.143). This indicates that the model identified 
physiological patterns consistent with elevated sepsis risk.

**This is a screening tool. Clinical verification is required.**""")
    elif st.session_state["s1_borderline"]:
        st.warning("""**⚡ Borderline Risk — Close Monitoring Recommended**

Peak probability is below the alert threshold (0.143) but above 8%. 
The patient may benefit from continued close observation.

Stage 2 post-sepsis risk assessment is available for further evaluation.""")
    else:
        st.success("""**✅ Low Risk — No Sepsis Alert**

The patient's hourly timeline did not produce any sepsis alerts at the current threshold. 
This result should be interpreted alongside clinical judgment and full patient context.""")

    # Clinical decision support
    st.markdown("#### 🩺 Clinical Review Prompts")
    st.caption("Structured prompts — SepsiArc does not prescribe treatment.")
    prompts = [
        "Review current vital signs against hourly trend in the timeline below.",
        "Assess SIRS criteria: fever/hypothermia, tachycardia, tachypnoe, leukocytosis.",
        "Review lactate, creatinine, WBC trajectory for organ dysfunction.",
        "Evaluate Shock Index (HR÷SBP) and MAP ≥65 mmHg.",
        "Consider qSOFA: altered mentation, respiratory rate ≥22, SBP ≤100.",
        "Escalate according to institutional sepsis protocol where appropriate.",
    ]
    for p in prompts:
        st.markdown(f"- {p}")

# ── Patient Summary Table ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Patient Summary")

_pid  = uploaded.name.replace(".psv","").replace(".csv","")
_age  = raw_df["Age"].dropna().iloc[0] if "Age" in raw_df.columns and raw_df["Age"].notna().any() else "N/A"
_gend = raw_df["Gender"].dropna().iloc[0] if "Gender" in raw_df.columns and raw_df["Gender"].notna().any() else "N/A"
_gend_lbl = ("Male" if float(_gend)==1 else "Female") if _gend != "N/A" else "N/A"
_hosp = raw_df["HospAdmTime"].dropna().iloc[0] if "HospAdmTime" in raw_df.columns and raw_df["HospAdmTime"].notna().any() else "N/A"
_obs_start = int(hours[0])  if len(hours) else 0
_obs_end   = int(hours[-1]) if len(hours) else 0

sum_data = {
    "Patient / File ID":       _pid,
    "Age":                     f"{float(_age):.0f} yrs" if _age != "N/A" else "N/A",
    "Gender":                  _gend_lbl,
    "Total Observations":      f"{n_rows} hours",
    "Observation Period":      f"Hour {_obs_start} – Hour {_obs_end}",
    "Peak Sepsis Probability": f"{peak_prob*100:.2f}%",
    "Decision Threshold":      f"{S1_THRESH:.3f} ({S1_THRESH*100:.1f}%)",
    "Alert Hours":             f"{n_alerts} / {n_rows}",
    "Peak Probability Hour":   f"Hour {peak_hour}",
    "First Alert Hour":        f"Hour {first_alert_h}" if first_alert_h else "None",
    "Final Prediction":        res_label,
}
sm1, sm2 = st.columns(2)
for i, (k, v) in enumerate(sum_data.items()):
    col = sm1 if i % 2 == 0 else sm2
    col.markdown(
        f'<div style="background:#F8FAFC;border-radius:8px;padding:0.55rem 0.85rem;'
        f'margin-bottom:0.35rem;border-left:3px solid {TEAL};">'
        f'<span style="font-size:0.7rem;color:{SLATE_400};text-transform:uppercase;letter-spacing:1px;">{k}</span>'
        f'<div style="font-size:0.9rem;font-weight:700;color:{NAVY};">{v}</div></div>',
        unsafe_allow_html=True,
    )

# ── Physiological Trend Summary ───────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Physiological Trend Summary")
st.caption("Values computed from available observations in the uploaded data. Trend: direction over the last 6 observations.")

_vitals_display = [
    ("HR",          "Heart Rate (bpm)"),
    ("O2Sat",       "O2 Saturation (%)"),
    ("Temp",        "Temperature (°C)"),
    ("SBP",         "Systolic BP (mmHg)"),
    ("MAP",         "Mean Arterial Pressure"),
    ("Resp",        "Respiratory Rate (br/min)"),
    ("Lactate",     "Lactate (mmol/L)"),
    ("Creatinine",  "Creatinine (mg/dL)"),
    ("WBC",         "WBC (K/uL)"),
]

def _trend_arrow(series):
    s = series.dropna()
    if len(s) < 2:
        return "Stable"
    delta = s.iloc[-1] - s.iloc[-min(6, len(s))]
    if delta > s.std() * 0.3:
        return "Increasing"
    elif delta < -s.std() * 0.3:
        return "Decreasing"
    return "Stable"

_trend_rows = []
for vname, vlabel in _vitals_display:
    if vname in raw_df.columns and raw_df[vname].notna().any():
        col_s = raw_df[vname].dropna()
        _trend_rows.append({
            "Parameter": vlabel,
            "Current / Recent Value": f"{col_s.iloc[-1]:.2f}",
            "Mean over Stay": f"{col_s.mean():.2f}",
            "Min": f"{col_s.min():.2f}",
            "Max": f"{col_s.max():.2f}",
            "Trend": _trend_arrow(col_s),
        })

if _trend_rows:
    st.dataframe(pd.DataFrame(_trend_rows), use_container_width=True, hide_index=True)
else:
    st.info("No standard vital sign columns found in the uploaded file for trend summary.")

# Store trend rows in session state so Stage 2 can include them in the combined PDF
st.session_state["s1_trend_rows"] = _trend_rows

# ══════════════════════════════════════════════════════════════════════════════
# PER-HOUR TIMELINE CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📈 Hourly Sepsis Probability Timeline")
st.caption(f"Sepsis Alert threshold at {S1_THRESH:.3f} (navy dashed) · Borderline threshold at 0.080 (amber dotted) · Red bars = sepsis alert hours.")

fig, ax = plt.subplots(figsize=(12, 3.6), facecolor="white")
bar_colors = [RED if l else "#BAE6FD" for l in labels]
ax.bar(hours, probs, color=bar_colors, alpha=0.85, width=0.7, edgecolor="white")
ax.axhline(S1_THRESH, color=NAVY, lw=1.4, linestyle="--", label=f"Sepsis Alert ({S1_THRESH:.3f})")
ax.axhline(0.080,    color=AMBER, lw=1.2, linestyle=":",  label="Borderline (0.080)")
ax.set_xlabel("ICU Hour (ICULOS)", fontsize=9, color="#475569")
ax.set_ylabel("Sepsis Probability", fontsize=9, color="#475569")
ax.set_title("Predicted Sepsis Probability per Hour", fontweight="bold", fontsize=10, color=NAVY)
ax.set_ylim(0, max(probs.max() * 1.15, S1_THRESH * 1.3, 0.2))
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.2)
ax.set_facecolor("#FAFCFF")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
_buf1 = io.BytesIO()
fig.savefig(_buf1, format="png", dpi=110, bbox_inches="tight", facecolor="white")
plt.close(fig)
_buf1.seek(0)
st.image(_buf1.getvalue(), width='stretch')

# True labels overlay (if available)
if "SepsisLabel" in raw_df.columns:
    true_labels = raw_df["SepsisLabel"].fillna(0).astype(int).values
    if true_labels.sum() > 0:
        st.caption("🔖 True SepsisLabel column detected in file — shown for reference only.")
        fig2, ax2 = plt.subplots(figsize=(12, 1.5), facecolor="white")
        ax2.bar(hours, true_labels, color=["#F87171" if v else "#D1FAE5" for v in true_labels],
                width=0.7, alpha=0.8, edgecolor="white")
        ax2.set_xlabel("ICU Hour", fontsize=8)
        ax2.set_ylabel("True Label", fontsize=8)
        ax2.set_title("True SepsisLabel (reference)", fontsize=9, color="#64748B")
        ax2.set_ylim(0, 1.4)
        ax2.set_facecolor("#FAFCFF")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        plt.tight_layout()
        _buf2 = io.BytesIO()
        fig2.savefig(_buf2, format="png", dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        _buf2.seek(0)
        st.image(_buf2.getvalue(), width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# HOUR-BY-HOUR TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📋 Hour-by-Hour Detail")

display_df = result_df.copy()
display_df["Sepsis Prob (%)"] = display_df["Sepsis Prob (%)"].map(lambda x: f"{x:.2f}%")
st.dataframe(display_df, use_container_width=True, hide_index=True)

# Download
csv_out = result_df.to_csv(index=False)
st.download_button(
    "⬇️ Download Results CSV",
    data=csv_out,
    file_name="sepsiarc_stage1_results.csv",
    mime="text/csv",
)

# ══════════════════════════════════════════════════════════════════════════════
# SHAP — Highest Risk Hour
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📊 Why This Prediction? (SHAP — Highest Risk Hour)")
st.caption(
    f"SHAP values for Hour {peak_hour} (peak probability {peak_prob*100:.1f}%). "
    "Red = increases sepsis risk · Green = decreases sepsis risk. "
    "SHAP describes model behaviour, not clinical causality."
)

with st.spinner("Computing SHAP explanation…"):
    try:
        import shap
        peak_row = model_X.iloc[[peak_hour_idx]]   # row for the highest-risk hour
        sv = None
        # Try new shap.Explainer first (avoids xgboost.core error on Cloud)
        try:
            _exp = shap.Explainer(s1_model)
            _sv  = _exp(peak_row)
            sv   = _sv.values
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            sv = sv[0]
        except Exception:
            pass
        # Fallback 1: classic TreeExplainer
        if sv is None:
            try:
                _exp2     = shap.TreeExplainer(s1_model)
                shap_vals = _exp2.shap_values(peak_row)
                sv        = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]
            except Exception:
                pass
        # Fallback 2: model built-in feature importances (always available)
        # s1_model is a Booster (not XGBClassifier) — use get_score(), NOT feature_importances_
        _shap_label = "SHAP Value"
        _shap_is_real = True
        if sv is None:
            _shap_is_real = False
            try:
                # get_score returns dict {feature_name: score}; align to S1_FEATURES order
                scores = s1_model.get_score(importance_type="gain")
                sv = np.array([scores.get(f, 0.0) for f in S1_FEATURES], dtype=float)
            except Exception:
                try:
                    # Legacy fallback: get_fscore() (same as get_score with weight type)
                    scores = s1_model.get_fscore()
                    sv = np.array([scores.get(f, 0.0) for f in S1_FEATURES], dtype=float)
                except Exception:
                    sv = np.zeros(len(S1_FEATURES), dtype=float)
            _shap_label = "Feature Importance (SHAP unavailable on this environment)"
            st.caption(
                "Note: SHAP could not be computed in this environment "
                "(XGBoost version incompatibility). Showing XGBoost gain-based "
                "feature importances instead — these rank features by overall predictive gain, "
                "not by direction of influence for this specific patient."
            )
        st.session_state["s1_shap_is_real"] = _shap_is_real
        st.session_state["s1_shap_label"]   = _shap_label
        shap_series = pd.Series(np.array(sv, dtype=float), index=S1_FEATURES)
        top10       = shap_series.abs().nlargest(10).index
        tv          = shap_series[top10]

        fig3, ax3 = plt.subplots(figsize=(8, 4), facecolor="white")
        if _shap_is_real:
            colors3 = [RED if v > 0 else GREEN for v in tv.values[::-1]]
            x_label3 = f"{_shap_label}  (positive = increases sepsis risk)"
            title3   = f"Top 10 Feature Contributions — Hour {peak_hour} (Prob {peak_prob*100:.1f}%)"
        else:
            colors3 = [TEAL for _ in tv]
            x_label3 = "XGBoost Gain Importance (Global Model Relative Weight)"
            title3   = "Global Model Feature Importance — SHAP Unavailable"

        ax3.barh(range(len(tv)), tv.values[::-1], color=colors3, alpha=0.85,
                 edgecolor="white", height=0.65)
        ax3.set_yticks(range(len(tv)))
        ax3.set_yticklabels(tv.index[::-1], fontsize=9)
        ax3.set_xlabel(x_label3, fontsize=9)
        ax3.set_title(title3, fontweight="bold", fontsize=10, color=NAVY)
        ax3.axvline(0, color="#334155", lw=0.8)
        ax3.grid(True, axis="x", alpha=0.25)
        ax3.set_facecolor("#FAFAFA")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        plt.tight_layout()
        _buf3 = io.BytesIO()
        fig3.savefig(_buf3, format="png", dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig3)
        _buf3.seek(0)
        st.image(_buf3.getvalue(), width='stretch')

        if _shap_is_real:
            sc1, sc2 = st.columns(2)
            sc1.markdown(f"<span style='color:{RED};font-size:0.8rem;'>🔴 Red — increases sepsis risk</span>",
                         unsafe_allow_html=True)
            sc2.markdown(f"<span style='color:{GREEN};font-size:0.8rem;'>🟢 Green — decreases sepsis risk</span>",
                         unsafe_allow_html=True)
        else:
            st.caption("ℹ️ Gain importance measures relative predictive contribution across trees. Values are non-negative and do not indicate direction of influence.")
    except Exception as e:
        st.info(f"SHAP explanation unavailable: {str(e)[:120]}")

# ══════════════════════════════════════════════════════════════════════════════
# LIME EXPLANATION  —  Stage 1 Patient-Specific
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🔬 Patient-Specific Explanation — LIME")
st.caption(
    "LIME approximates model behaviour around this specific patient's observation at the highest-risk hour. "
    "Results are approximate — they do not prove clinical causation."
)

with st.spinner("Computing LIME explanation for Stage 1..."):
    try:
        import lime.lime_tabular
        _lime_row   = model_X.iloc[[peak_hour_idx]]
        _col_means1 = model_X.mean().values
        _col_std1   = np.abs(_col_means1) * 0.15 + 1e-6
        _rng1       = np.random.default_rng(42)
        _bg1        = _col_means1 + _rng1.normal(0, 1, size=(200, len(S1_FEATURES))) * _col_std1
        _bg1[0]     = _lime_row.values[0]
        _bg1        = np.clip(_bg1, 0, None)

        _lime_exp_obj1 = lime.lime_tabular.LimeTabularExplainer(
            training_data=_bg1,
            feature_names=S1_FEATURES,
            class_names=["Low Risk", "Sepsis Alert"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
        # Use booster wrapper that exposes predict_proba for LIME
        def _s1_predict_proba(X):
            import xgboost as _xgb
            _d = _xgb.DMatrix(X, feature_names=S1_FEATURES)
            _p = s1_model.predict(_d)
            return np.column_stack([1 - _p, _p])

        _lime_exp1  = _lime_exp_obj1.explain_instance(
            data_row=_lime_row.values[0],
            predict_fn=_s1_predict_proba,
            num_features=10,
            num_samples=1000,
        )
        _lime_list1    = _lime_exp1.as_list()
        _lime_for1     = [(f, v) for f, v in _lime_list1 if v > 0]
        _lime_against1 = [(f, v) for f, v in _lime_list1 if v <= 0]
        _pred_cls1     = "Sepsis Alert" if patient_high else "Low Risk"

        lc1s, lc2s = st.columns(2)
        with lc1s:
            st.markdown(f"**Features supporting {_pred_cls1}**")
            for i, (f, v) in enumerate(sorted(_lime_for1, key=lambda x: -abs(x[1]))[:5], 1):
                st.markdown(f"{i}. {f}")
            if not _lime_for1:
                st.caption("None in top 10.")
        with lc2s:
            st.markdown(f"**Features opposing {_pred_cls1}**")
            for i, (f, v) in enumerate(sorted(_lime_against1, key=lambda x: abs(x[1]))[:5], 1):
                st.markdown(f"{i}. {f}")
            if not _lime_against1:
                st.caption("None in top 10.")

        _fn1 = [x[0] for x in _lime_list1]
        _fv1 = [x[1] for x in _lime_list1]
        _lc1 = [RED if v > 0 else GREEN for v in _fv1]
        _fig_l1, _ax_l1 = plt.subplots(figsize=(9, 4.5), facecolor="white")
        _yp1 = list(range(len(_fn1)))
        _ax_l1.barh(list(reversed(_yp1)), list(reversed(_fv1)),
                    color=list(reversed(_lc1)), alpha=0.85, height=0.65)
        _ax_l1.set_yticks(list(reversed(_yp1)))
        _ax_l1.set_yticklabels(list(reversed(_fn1)), fontsize=8)
        _ax_l1.set_xlabel("LIME contribution (positive → Sepsis Alert)", fontsize=9)
        _ax_l1.set_title(f"LIME Explanation — Hour {peak_hour} ({_pred_cls1})",
                         fontweight="bold", fontsize=10, color=NAVY)
        _ax_l1.axvline(0, color="#334155", lw=0.8)
        _ax_l1.grid(True, axis="x", alpha=0.25)
        _ax_l1.set_facecolor("#FAFAFA")
        _ax_l1.spines["top"].set_visible(False)
        _ax_l1.spines["right"].set_visible(False)
        plt.tight_layout()
        _buf_l1 = io.BytesIO()
        _fig_l1.savefig(_buf_l1, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(_fig_l1)
        _buf_l1.seek(0)
        st.image(_buf_l1.getvalue(), width='stretch')
        st.caption("LIME describes local model behaviour — not clinical causation.")
    except ImportError:
        st.info("⚠️ LIME not installed. Add `lime` to requirements.txt to enable this section.")
    except Exception as _le:
        st.info(f"LIME explanation unavailable: {str(_le)[:150]}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 ELIGIBILITY + NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🔄 Stage 2 Eligibility")

_s2_eligible = patient_high or st.session_state["s1_borderline"]
_s2_pre_hrs  = int((hours < first_alert_h).sum()) if first_alert_h else 0
_s2_post_hrs = int((hours >= first_alert_h).sum()) if first_alert_h else 0
st.session_state["s1_obs_start"] = _obs_start
st.session_state["s1_obs_end"]   = _obs_end
st.session_state["s1_pre_hrs"]   = _s2_pre_hrs
st.session_state["s1_post_hrs"]  = _s2_post_hrs

if _s2_eligible:
    st.markdown(f"""
<div style="background:#F0FDF4;border:1.5px solid #86EFAC;border-radius:12px;padding:1rem 1.4rem;margin-bottom:0.8rem;">
  <div style="font-size:0.9rem;font-weight:700;color:#166534;">✅ Stage 2 Post-Sepsis Risk Assessment — Available</div>
  <div style="font-size:0.78rem;color:#15803D;margin-top:6px;line-height:1.8;">
    Stage 1 Reference / First Alert Hour: <b>Hour {first_alert_h}</b><br>
    Pre-alert history available: <b>{_s2_pre_hrs} hour(s)</b><br>
    Post-alert observations available: <b>{_s2_post_hrs} hour(s)</b><br>
    Total observations passed to Stage 2: <b>{n_rows}</b>
  </div>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown(f"""
<div style="background:#FFF7ED;border:1.5px solid #FED7AA;border-radius:12px;padding:1rem 1.4rem;margin-bottom:0.8rem;">
  <div style="font-size:0.9rem;font-weight:700;color:#9A3412;">ℹ️ Stage 2 Not Required</div>
  <div style="font-size:0.78rem;color:#C2410C;margin-top:4px;">
    Peak probability ({peak_prob*100:.1f}%) is below both thresholds.<br>
    Stage 2 risk stratification is not applicable without a positive or borderline Stage 1 result.
  </div>
</div>
""", unsafe_allow_html=True)

# Stage 2 navigation buttons
if patient_high:
    st.markdown(
        '<style>.s2-link a{display:block;background:linear-gradient(135deg,#991B1B,#0D2137);'
        'color:#fff!important;text-align:center;padding:0.7rem 1.5rem;border-radius:9px;'
        'font-weight:700;font-size:0.9rem;text-decoration:none;margin-top:0.3rem;'
        'box-shadow:0 3px 10px rgba(153,27,27,0.3);}'
        '.s2-link a:hover{opacity:0.88;}</style>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="s2-link">', unsafe_allow_html=True)
    st.page_link("pages/2_Stage_2_Risk.py",
                 label="➡️  Continue to Stage 2 — Post-Sepsis Risk Assessment",
                 icon="🔬")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state["s1_borderline"]:
    st.markdown(
        '<style>.s2-link-b a{display:block;background:linear-gradient(135deg,#B45309,#0D2137);'
        'color:#fff!important;text-align:center;padding:0.7rem 1.5rem;border-radius:9px;'
        'font-weight:700;font-size:0.9rem;text-decoration:none;margin-top:0.3rem;'
        'box-shadow:0 3px 10px rgba(180,83,9,0.3);}'
        '.s2-link-b a:hover{opacity:0.88;}</style>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="s2-link-b">', unsafe_allow_html=True)
    st.page_link("pages/2_Stage_2_Risk.py",
                 label="➡️  Continue to Stage 2 — Borderline Pathway",
                 icon="⚡")
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD STAGE 1 REPORT (PDF)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📥 Download Stage 1 Report")

import datetime as _dt
import importlib as _ilib
try:
    import utils.pdf_report as _pdf_mod
    _ilib.reload(_pdf_mod)  # force reload so Cloud cache is bypassed
    _build_s1_pdf = _pdf_mod.build_stage1_pdf
    _hosp_str = (f"{float(_hosp):.1f} hr before ICU admission"
                 if _hosp != "N/A" else "N/A")
    _age_str  = (f"{float(_age):.0f} yrs" if _age != "N/A" else "N/A")

    # -- Build clean hourly_rows for all processed observations
    _hourly_rows_pdf = [
        {
            'Hour':     int(h),
            'Prob':     float(p * 100),
            'Category': "SEPSIS ALERT" if p >= S1_THRESH else ("BORDERLINE" if p >= 0.080 else "LOW RISK"),
            'Alert':    bool(p >= S1_THRESH),
        }
        for h, p in zip(hours, probs)
    ]

    # -- Retrieve chart PNG bytes (_buf1 already seeked when displayed)
    _chart_png = None
    try:
        _buf1.seek(0)
        _chart_png = _buf1.read()
    except Exception:
        _chart_png = None

    # -- Retrieve SHAP PNG bytes (_buf3 created inside SHAP block)
    _shap_png = None
    try:
        _buf3.seek(0)
        _shap_png = _buf3.read()
    except Exception:
        _shap_png = None

    # -- Retrieve LIME PNG bytes (_buf_l1 created inside LIME block)
    _lime_png1 = None
    try:
        _buf_l1.seek(0)
        _lime_png1 = _buf_l1.read()
    except Exception:
        _lime_png1 = None

    # -- Retrieve LIME text rules from session
    _lime_text1 = None
    try:
        if '_lime_list1' in locals() and _lime_list1:
            _lime_text1 = "\n".join([f"{f}: {v:+.4f}" for f, v in _lime_list1])
    except Exception:
        _lime_text1 = None

    _shap_lbl_to_pass = _shap_label if ('_shap_is_real' in locals() and not _shap_is_real) else None

    try:
        _pdf_bytes = _build_s1_pdf(
            patient_id      = _pid,
            age             = _age_str,
            gender          = _gend_lbl,
            n_rows          = n_rows,
            obs_start       = _obs_start,
            obs_end         = _obs_end,
            hosp_adm        = _hosp_str,
            peak_prob       = peak_prob,
            s1_thresh       = S1_THRESH,
            n_alerts        = n_alerts,
            peak_hour       = peak_hour,
            first_alert_h   = first_alert_h,
            res_label       = res_label,
            trend_rows      = _trend_rows,
            s2_eligible     = _s2_eligible,
            s2_pre_hrs      = _s2_pre_hrs,
            s2_post_hrs     = _s2_post_hrs,
            hourly_rows     = _hourly_rows_pdf,
            chart_png_bytes = _chart_png,
            shap_png_bytes  = _shap_png,
            lime_png_bytes  = _lime_png1,
            lime_text       = _lime_text1,
            shap_label      = _shap_lbl_to_pass,
        )
    except TypeError:
        # Fallback: old pdf_report signature
        _pdf_bytes = _build_s1_pdf(
            patient_id    = _pid,
            age           = _age_str,
            gender        = _gend_lbl,
            n_rows        = n_rows,
            obs_start     = _obs_start,
            obs_end       = _obs_end,
            hosp_adm      = _hosp_str,
            peak_prob     = peak_prob,
            s1_thresh     = S1_THRESH,
            n_alerts      = n_alerts,
            peak_hour     = peak_hour,
            first_alert_h = first_alert_h,
            res_label     = res_label,
            trend_rows    = _trend_rows,
            s2_eligible   = _s2_eligible,
            s2_pre_hrs    = _s2_pre_hrs,
            s2_post_hrs   = _s2_post_hrs,
        )
    _ts = _dt.datetime.now().strftime('%Y%m%d_%H%M')
    st.download_button(
        label="⬇️ Download Stage 1 Report (PDF)",
        data=_pdf_bytes,
        file_name=f"SepsiArc_Stage1_Report_{_pid}_{_ts}.pdf",
        mime="application/pdf",
    )
    st.caption("PDF report includes patient summary, prediction result, physiological trends, "
               "Stage 2 eligibility, clinical interpretation, and disclaimer.")
except Exception as _pdf_err:
    st.warning(f"PDF generation unavailable: {str(_pdf_err)[:100]}. "
               "Ensure `fpdf2` is listed in requirements.txt.")

render_footer()

