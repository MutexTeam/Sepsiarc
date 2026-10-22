"""
pages/1_Stage_1_Detection.py
SepsiArc — Stage 1: Sepsis Detection
Model: Regularized Cost-Sensitive XGBoost (Model A)
Threshold: 0.143 | Features: 135 | AUROC: 0.7587 | Recall: 70.97%
Input: PSV / CSV file upload (hourly patient timeline)
"""

import os, io, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        "Assess SIRS criteria: fever/hypothermia, tachycardia, tachypnoea, leukocytosis.",
        "Review lactate, creatinine, WBC trajectory for organ dysfunction.",
        "Evaluate Shock Index (HR÷SBP) and MAP ≥65 mmHg.",
        "Consider qSOFA: altered mentation, respiratory rate ≥22, SBP ≤100.",
        "Escalate according to institutional sepsis protocol where appropriate.",
    ]
    for p in prompts:
        st.markdown(f"- {p}")

# ══════════════════════════════════════════════════════════════════════════════
# PER-HOUR TIMELINE CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📈 Hourly Sepsis Probability Timeline")
st.caption(f"Threshold line at {S1_THRESH:.3f}. Red bars = sepsis alert hours.")

fig, ax = plt.subplots(figsize=(12, 3.6), facecolor="white")
bar_colors = [RED if l else "#BAE6FD" for l in labels]
ax.bar(hours, probs, color=bar_colors, alpha=0.85, width=0.7, edgecolor="white")
ax.axhline(S1_THRESH, color=NAVY, lw=1.4, linestyle="--", label=f"Threshold ({S1_THRESH:.3f})")
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
fig.savefig(_buf1, format="png", dpi=120, bbox_inches="tight", facecolor="white")
_buf1.seek(0)
st.image(_buf1, use_container_width=True)
plt.close(fig)

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
        fig2.savefig(_buf2, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        _buf2.seek(0)
        st.image(_buf2, use_container_width=True)
        plt.close(fig2)

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
        # Fallback: classic TreeExplainer
        if sv is None:
            _exp2     = shap.TreeExplainer(s1_model)
            shap_vals = _exp2.shap_values(peak_row)
            sv        = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]
        shap_series = pd.Series(np.array(sv, dtype=float), index=S1_FEATURES)
        top10       = shap_series.abs().nlargest(10).index
        tv          = shap_series[top10]

        fig3, ax3 = plt.subplots(figsize=(8, 4), facecolor="white")
        colors3   = [RED if v > 0 else GREEN for v in tv.values[::-1]]
        ax3.barh(range(len(tv)), tv.values[::-1], color=colors3, alpha=0.85,
                 edgecolor="white", height=0.65)
        ax3.set_yticks(range(len(tv)))
        ax3.set_yticklabels(tv.index[::-1], fontsize=9)
        ax3.set_xlabel("SHAP value  (positive → increases sepsis risk)", fontsize=9)
        ax3.set_title(
            f"Top 10 Feature Contributions — Hour {peak_hour} (Prob {peak_prob*100:.1f}%)",
            fontweight="bold", fontsize=10, color=NAVY,
        )
        ax3.axvline(0, color="#334155", lw=0.8)
        ax3.grid(True, axis="x", alpha=0.25)
        ax3.set_facecolor("#FAFAFA")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        plt.tight_layout()
        _buf3 = io.BytesIO()
        fig3.savefig(_buf3, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        _buf3.seek(0)
        st.image(_buf3, use_container_width=True)
        plt.close(fig3)

        sc1, sc2 = st.columns(2)
        sc1.markdown(f"<span style='color:{RED};font-size:0.8rem;'>🔴 Red — increases sepsis risk</span>",
                     unsafe_allow_html=True)
        sc2.markdown(f"<span style='color:{GREEN};font-size:0.8rem;'>🟢 Green — decreases sepsis risk</span>",
                     unsafe_allow_html=True)
    except Exception as e:
        st.info(f"SHAP explanation unavailable: {str(e)[:120]}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")

if patient_high:
    st.markdown(f"""
<div style="background:#FFF1F2;border:1.5px solid #FECDD3;border-radius:12px;
            padding:1rem 1.4rem;margin-bottom:0.8rem;">
  <div style="font-size:0.9rem;font-weight:700;color:#9F1239;">
    ⚠️ Sepsis Alert — Stage 2 Post-Sepsis Risk Assessment Available
  </div>
  <div style="font-size:0.78rem;color:#BE123C;margin-top:4px;">
    Peak probability ({peak_prob*100:.1f}%) exceeds threshold ({S1_THRESH:.3f}).
    Proceed to Stage 2 for post-sepsis risk stratification.
  </div>
</div>
""", unsafe_allow_html=True)
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
    st.markdown(f"""
<div style="background:#FFFBEB;border:1.5px solid #FDE68A;border-radius:12px;
            padding:1rem 1.4rem;margin-bottom:0.8rem;">
  <div style="font-size:0.9rem;font-weight:700;color:#92400E;">
    ⚡ Borderline Risk — Stage 2 Available with Caution
  </div>
  <div style="font-size:0.78rem;color:#B45309;margin-top:4px;">
    Peak probability ({peak_prob*100:.1f}%) is below alert threshold but above 8%.
    Stage 2 is available for further clinical evaluation. Apply additional clinical judgement.
  </div>
</div>
""", unsafe_allow_html=True)
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

else:
    st.markdown(f"""
<div style="background:#F0FDF4;border:1.5px solid #BBF7D0;border-radius:12px;
            padding:1rem 1.4rem;">
  <div style="font-size:0.9rem;font-weight:700;color:#166534;">✅ Low Risk — Stage 2 Not Required</div>
  <div style="font-size:0.78rem;color:#15803D;margin-top:4px;">
    Peak probability ({peak_prob*100:.1f}%) is below both the alert and borderline thresholds.
    Continue standard clinical monitoring.
  </div>
</div>
""", unsafe_allow_html=True)

render_footer()
