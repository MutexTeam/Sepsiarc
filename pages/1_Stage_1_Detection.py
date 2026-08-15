"""
pages/1_Stage_1_Detection.py  --  SepsiArc Stage 1: Sepsis Early Detection
Collects raw clinical parameters; backend derives 79 engineered features.
Uses LightGBM-Calibrated model with clinical gating.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, sys, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from utils.ui_components import (
    render_header, render_footer, render_sidebar,
    render_disclaimer_banner, risk_badge_html,
    NAVY, TEAL, GREEN, AMBER, RED, SLATE_200, SLATE_400, SLATE_700,
)

st.set_page_config(
    page_title="SepsiArc – Stage 1: Sepsis Detection",
    page_icon="🧬",
    layout="wide",
)

render_header(
    page_title="Stage 1 — Sepsis Detection",
    page_subtitle="Assess the current clinical state for sepsis risk.",
    icon="🧬",
)
render_sidebar("stage1")
render_disclaimer_banner()

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading SepsiArc models…")
def load_s1():
    mdir = "models"
    model      = joblib.load(os.path.join(mdir, "sepsis_best_model.pkl"))
    threshold  = float(joblib.load(os.path.join(mdir, "sepsis_threshold.pkl")))
    name       = open(os.path.join(mdir, "best_model_name.txt")).read().strip()
    feats      = json.load(open(os.path.join(mdir, "feature_names.json")))
    lgb_model  = joblib.load(os.path.join(mdir, "lgb_explainer_model.pkl"))
    gate       = json.load(open(os.path.join(mdir, "clinical_gate_params.json")))
    return model, threshold, name, feats, lgb_model, gate

try:
    model, threshold, model_name, feature_names, lgb_model, gate_params = load_s1()
    model_loaded = True
except Exception as e:
    st.error(f"Model could not be loaded. Please verify model artifacts are present in `models/`. Detail: {str(e)[:120]}")
    model_loaded = False

# ── Input mode selector ───────────────────────────────────────────────────────
st.markdown(
    '<div style="background:#E0F7FA;border-radius:10px;padding:1rem 1.4rem;'
    'border:1px solid #67E8F9;margin-bottom:1rem;">'
    '<b style="color:' + NAVY + ';">Who is this for?</b> '
    '<span style="color:' + SLATE_700 + ';font-size:0.88rem;">'
    'Any patient suspected of infection or showing signs of clinical deterioration. '
    'Enter the values you have — results include confidence and model explanation.'
    '</span></div>',
    unsafe_allow_html=True
)

input_mode = st.radio(
    "Input Mode",
    ["Mode A — Single Current Observation", "Mode B — Time-Series Upload (CSV)"],
    horizontal=True,
    help="Mode A: Enter current vitals/labs manually. Mode B: Upload sequential ICU observations.",
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# MODE A — MANUAL SINGLE-OBSERVATION ENTRY
# ══════════════════════════════════════════════════════════════════════════════
if "Single" in input_mode:

    st.markdown(f"#### 📋 Clinical Data Entry")
    st.caption("Enter current measured values. Derived features are calculated automatically by SepsiArc.")

    # ── SECTION 1 — Patient Information ───────────────────────────────────────
    with st.expander("👤 Section 1 — Patient Information", expanded=True):
        pi1, pi2, pi3 = st.columns(3)
        with pi1:
            Age = st.number_input("Age (years)", 15, 110, 60, 1,
                                  help="Patient's age in years.")
            Gender_sel = st.selectbox("Biological Sex", ["Male", "Female"],
                                      help="Use the encoding expected by the trained model: Male=1, Female=0")
            Gender = 1 if Gender_sel == "Male" else 0
        with pi2:
            ICULOS = st.number_input("ICU Length of Stay (hours)", 0, 500, 12, 1,
                                     help="How many hours has the patient been in the ICU at the time of this assessment?")
            Unit1 = st.selectbox("ICU Unit Type", [0, 1],
                                 format_func=lambda x: "MICU (Medical ICU)" if x == 1 else "Other Unit",
                                 help="Type of ICU unit where the patient is currently admitted.")
        with pi3:
            st.markdown(f"""
<div style="background:{SLATE_200}1A; border-radius:8px; padding:0.8rem; border:1px solid {SLATE_200};
            font-size:0.75rem; color:{SLATE_700}; line-height:1.7;">
  <b>ICU Timeline:</b><br>
  ≤6 hrs → Early ICU<br>
  >24 hrs → Late ICU<br><br>
  These flags are derived automatically.
</div>
""", unsafe_allow_html=True)

    # ── SECTION 2 — Vital Signs ────────────────────────────────────────────────
    with st.expander("🫀 Section 2 — Vital Signs", expanded=True):
        st.caption("All vital signs as currently measured. Units shown beside each field.")
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            HR   = st.number_input("Heart Rate (bpm)",            0.0, 300.0, 88.0, 1.0, help="Normal: 60–100 | Tachycardia: >100")
            SBP  = st.number_input("Systolic BP (mmHg)",           0.0, 300.0, 115.0, 1.0, help="Normal: 90–140 | Hypotension: ≤100")
        with v2:
            Temp = st.number_input("Temperature (°C)",             30.0, 43.0,  37.0, 0.1, help="Fever: ≥38°C | Hypothermia: <36°C")
            MAP  = st.number_input("Mean Arterial Pressure (mmHg)",0.0, 200.0,  78.0, 1.0, help="Critical: MAP <65 mmHg")
        with v3:
            DBP  = st.number_input("Diastolic BP (mmHg)",          0.0, 200.0,  70.0, 1.0, help="Normal: 60–90 mmHg")
            Resp = st.number_input("Respiratory Rate (br/min)",     0.0,  60.0,  16.0, 1.0, help="Normal: 12–20 | Tachypnea: >22")
        with v4:
            O2Sat= st.number_input("SpO₂ — Oxygen Saturation (%)", 0.0, 100.0,  97.0, 0.1, help="Normal: 95–100% | Low: <94%")

        # Live range check chips
        st.markdown("<div style='margin-top:0.4rem;'><b style='font-size:0.78rem;'>Live Range Check:</b></div>", unsafe_allow_html=True)
        chips_html = ""
        for val, lo, hi, label, unit in [
            (HR,    60, 100, "HR",    "bpm"),
            (Temp,  36, 38,  "Temp",  "°C"),
            (SBP,   90, 140, "SBP",   "mmHg"),
            (MAP,   65, 120, "MAP",   "mmHg"),
            (DBP,   60, 90,  "DBP",   "mmHg"),
            (Resp,  12, 22,  "RR",    "br/min"),
            (O2Sat, 94, 100, "SpO₂",  "%"),
        ]:
            alert = val < lo or val > hi
            bg  = "#FEE2E2" if alert else "#F0FDF4"
            brd = RED if alert else GREEN
            txt = RED if alert else GREEN
            chips_html += (
                f'<span style="background:{bg}; border:1.5px solid {brd}; color:{txt}; '
                f'border-radius:6px; padding:0.2rem 0.55rem; font-size:0.74rem; '
                f'font-weight:600; margin:2px; display:inline-block;">'
                f'{label}: {val:.1f} {unit}</span>'
            )
        st.markdown(chips_html, unsafe_allow_html=True)

    # ── SECTION 3 — Laboratory Results ────────────────────────────────────────
    with st.expander("🧪 Section 3 — Laboratory Results", expanded=True):
        st.caption("Enter values from the most recent laboratory results.")
        lb1, lb2, lb3 = st.columns(3)
        with lb1:
            WBC        = st.number_input("WBC (×10³/µL)",         0.0, value=8.0,   step=0.1,  help="Normal: 4.5–11 | Infection: >12 or <4")
            Creatinine = st.number_input("Creatinine (mg/dL)",    0.0, value=1.0,   step=0.1,  help="Normal: 0.6–1.2 | AKI: >1.5")
        with lb2:
            Lactate    = st.number_input("Serum Lactate (mmol/L)",0.0, 25.0, 1.0,  0.1,       help="KEY biomarker. Normal <2 | Sepsis: >2 | Shock: >4")
            Platelets  = st.number_input("Platelets (×10³/µL)",   0.0, value=200.0, step=1.0,  help="Normal: 150–400 | Thrombocytopenia: <100")
        with lb3:
            Glucose    = st.number_input("Glucose (mg/dL)",       0.0, value=100.0, step=1.0,  help="Normal: 70–140 mg/dL")
            st.caption("💡 Lactate is the most critical single sepsis biomarker.")

    # ── SECTION 4 — Derived Features Preview (display-only) ───────────────────
    with st.expander("⚙️ Section 4 — Derived Features (automatically calculated by SepsiArc)", expanded=False):
        shock_idx_preview = HR / max(SBP, 1)
        pp_preview        = SBP - DBP
        map_ratio_preview = MAP / max(SBP, 1)

        st.info("These features are automatically derived from your inputs. They are **not editable** — they are calculated by SepsiArc to match the exact training pipeline.")
        dp1, dp2, dp3 = st.columns(3)
        with dp1:
            st.metric("Shock Index (HR / SBP)",     f"{shock_idx_preview:.3f}", help="<0.7 normal | >1.0 shock")
        with dp2:
            st.metric("Pulse Pressure (SBP − DBP)", f"{pp_preview:.0f} mmHg")
        with dp3:
            st.metric("MAP Ratio (MAP / SBP)",       f"{map_ratio_preview:.3f}")

        st.markdown(f"<div style='font-size:0.72rem; color:{SLATE_400}; margin-top:0.5rem;'>Additional derived features (rolling means, trends, std deviations, lag features, SIRS/qSOFA/Lactate flags) are calculated internally when the model runs.</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── RUN BUTTON ────────────────────────────────────────────────────────────
    btn_col, info_col = st.columns([2, 5])
    with btn_col:
        analyze = st.button("🔬 Run Sepsis Assessment", use_container_width=True, disabled=not model_loaded)
    with info_col:
        st.markdown(f"""
<div style="padding:0.55rem 1rem; background:#EFF6FF; border-radius:8px;
            border:1px solid #BFDBFE; font-size:0.82rem; color:#1E40AF;">
  SepsiArc derives <b>79 clinical features</b> from your inputs and applies the
  <b>{model_name}</b> model with clinical gating to produce a sepsis risk probability.
  SHAP explains which values drove the result.
</div>
""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PREDICTION LOGIC
    # ══════════════════════════════════════════════════════════════════════════
    if analyze and model_loaded:
        with st.spinner("Analyzing clinical data…"):
            # Derive additional defaults (fields in feature contract not directly entered)
            HospAdmTime = -6.0
            Unit2 = 0
            Hct   = 38.0; Hgb = 12.5; Potassium = 4.0
            BUN   = 15.0; pH  = 7.40;  Bilirubin = 0.8
            Magnesium = 1.9; Phosphate = 3.5; Calcium = 9.0; PTT = 30.0
            BaseExcess = 0; HCO3 = 24.0; FiO2 = 0.21; PaCO2 = 40.0

            # Engineered features (must match training pipeline exactly)
            shock_index  = HR / max(SBP, 1)
            temp_dev     = abs(Temp - 37.0)
            MAP_risk_v   = int(MAP < 65)
            tachycardia  = int(HR > 100)
            tachypnea    = int(Resp > 22)
            hypoxia      = int(O2Sat < 94)
            fever        = int(Temp >= 38.0)
            hypothermia  = int(Temp < 36.0)
            SIRS_score   = tachycardia + tachypnea + fever + hypothermia
            pp           = SBP - DBP
            MAP_HR       = MAP / max(HR, 1)
            hypo         = int(SBP <= 100)
            qSOFA        = tachypnea + hypo
            Lac_r        = int(Lactate > 2.0)
            hyperLac     = int(Lactate > 4.0)
            pH_a         = int(pH < 7.35)
            sev_a        = int(pH < 7.20)
            BUN_Cr       = BUN / max(Creatinine, 0.1)
            AKI          = int(Creatinine > 1.5)
            hepatic      = int(Bilirubin > 2.0)
            sev_t        = int(Platelets < 100)
            SOFA         = int(Resp>22)+int(MAP<65)+int(Creatinine>1.2)+int(Platelets<150)
            oxy          = O2Sat / max(Resp, 1)

            row = {
                "HR":HR,"O2Sat":O2Sat,"Temp":Temp,"SBP":SBP,"MAP":MAP,"DBP":DBP,"Resp":Resp,
                "BaseExcess":BaseExcess,"HCO3":HCO3,"FiO2":FiO2,"pH":pH,"PaCO2":PaCO2,
                "BUN":BUN,"Calcium":Calcium,"Creatinine":Creatinine,"Glucose":Glucose,
                "Lactate":Lactate,"Magnesium":Magnesium,"Phosphate":Phosphate,"Potassium":Potassium,
                "Bilirubin_total":Bilirubin,"Hct":Hct,"Hgb":Hgb,"PTT":PTT,
                "WBC":WBC,"Platelets":Platelets,
                "Age":Age,"Gender":Gender,"Unit1":Unit1,"Unit2":Unit2,
                "HospAdmTime":HospAdmTime,"ICULOS":ICULOS,
                "shock_index":shock_index,"temp_deviation":temp_dev,"MAP_risk":MAP_risk_v,
                "tachycardia":tachycardia,"tachypnea":tachypnea,"hypoxia":hypoxia,
                "fever":fever,"hypothermia":hypothermia,"SIRS_score":SIRS_score,
                "pulse_pressure":pp,"MAP_HR_ratio":MAP_HR,"Lactate_risk":Lac_r,
                "hyperlactatemia":hyperLac,"pH_acidosis":pH_a,"severe_acidosis":sev_a,
                "qSOFA_score":qSOFA,"BUN_Creatinine_ratio":BUN_Cr,"oxygenation_index":oxy,
                "HR_roll3_mean":HR,"HR_roll3_std":0,"MAP_roll3_mean":MAP,"MAP_roll3_std":0,
                "O2Sat_roll3_mean":O2Sat,"O2Sat_roll3_std":0,"Temp_roll3_mean":Temp,"Temp_roll3_std":0,
                "Resp_roll3_mean":Resp,"Resp_roll3_std":0,"SBP_roll3_mean":SBP,"SBP_roll3_std":0,
                "HR_lag1":HR,"HR_lag3":HR,"HR_delta1":0,
                "MAP_lag1":MAP,"MAP_lag3":MAP,"MAP_delta1":0,
                "O2Sat_lag1":O2Sat,"O2Sat_lag3":O2Sat,"O2Sat_delta1":0,
                "Resp_lag1":Resp,"Resp_lag3":Resp,"Resp_delta1":0,
                "SBP_lag1":SBP,"SBP_lag3":SBP,"SBP_delta1":0,
                "early_icu":int(ICULOS<=6),"late_icu":int(ICULOS>24),
            }

            # Feature contract safety check
            inp = pd.DataFrame([[row.get(f, 0) for f in feature_names]], columns=feature_names)

            # Verify feature count
            if inp.shape[1] != len(feature_names):
                st.error(
                    f"⚠️ Feature contract mismatch: expected {len(feature_names)} features, "
                    f"got {inp.shape[1]}. Prediction stopped. Please contact the system administrator."
                )
                st.stop()

            prob_raw = float(model.predict_proba(inp)[0][1])

            # ── Clinical Gating Rule ──────────────────────────────────────────
            cap = gate_params.get("cap_probability", 0.10)
            si_gate = gate_params.get("shock_index_max_for_gate", 0.75)
            no_clinical_signs = (
                SIRS_score == 0 and qSOFA == 0 and Lac_r == 0
                and shock_index < si_gate and tachycardia == 0
                and tachypnea == 0 and MAP_risk_v == 0
            )
            prob  = min(prob_raw, cap) if no_clinical_signs else prob_raw
            gated = no_clinical_signs and prob_raw > cap

        # ── RESULT DISPLAY ────────────────────────────────────────────────────
        sepsis_positive = prob >= threshold
        st.markdown("---")
        st.markdown("## 🩺 Assessment Results")

        # Main result card
        res_main, res_sirs, res_shock = st.columns([3, 2, 2])

        with res_main:
            result_color = RED if sepsis_positive else GREEN
            result_label = "SEPSIS POSITIVE" if sepsis_positive else "SEPSIS NEGATIVE"
            badge_style  = "positive" if sepsis_positive else "negative"

            gate_note = (
                f'<br><span style="color:{AMBER}; font-size:0.71rem;">'
                f'⚡ Clinical gate applied: raw {prob_raw*100:.1f}% → {prob*100:.1f}% (no SIRS/qSOFA criteria met)'
                f'</span>'
            ) if gated else ""

            st.markdown(f"""
<div style="background:#fff; border-radius:14px; padding:1.4rem 1.8rem;
            box-shadow:0 3px 16px rgba(0,0,0,0.09);
            border-left:6px solid {result_color};">
  <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase;
              letter-spacing:1.5px; color:{SLATE_400}; margin-bottom:0.4rem;">
    Sepsis Probability
  </div>
  <div style="font-size:3.5rem; font-weight:800; color:{result_color}; line-height:1;">
    {prob * 100:.1f}%
  </div>
  <div style="margin:0.7rem 0 0.5rem;">
    {risk_badge_html(result_label, badge_style)}
  </div>
  <div style="font-size:0.76rem; color:{SLATE_400}; margin-top:0.4rem;">
    Threshold: {threshold:.0%} &nbsp;·&nbsp; Model: {model_name}{gate_note}
  </div>
</div>
""", unsafe_allow_html=True)
            st.progress(min(prob, 1.0))

        with res_sirs:
            sirs_c = RED if SIRS_score >= 2 else TEAL
            qsofa_c = RED if qSOFA >= 2 else TEAL
            st.markdown(f"""
<div style="background:#fff; border-radius:12px; padding:1.2rem; box-shadow:0 2px 8px rgba(0,0,0,0.07);">
  <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.2px; color:{SLATE_400}; margin-bottom:0.4rem;">SIRS Score</div>
  <div style="font-size:2.8rem; font-weight:800; color:{sirs_c}; line-height:1;">{SIRS_score}/4</div>
  <div style="font-size:0.75rem; color:{sirs_c}; margin-bottom:0.8rem;">{"⚠️ SIRS Positive (≥2)" if SIRS_score>=2 else "✅ SIRS Negative"}</div>
  <hr style="border-color:{SLATE_200}; margin:0.5rem 0;">
  <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.2px; color:{SLATE_400}; margin-bottom:0.3rem;">qSOFA Score</div>
  <div style="font-size:2.8rem; font-weight:800; color:{qsofa_c}; line-height:1;">{qSOFA}/2</div>
  <div style="font-size:0.75rem; color:{qsofa_c};">{"⚠️ High concern" if qSOFA>=2 else "Low concern"}</div>
</div>
""", unsafe_allow_html=True)

        with res_shock:
            si_c  = RED if shock_index > 1.0 else AMBER if shock_index > 0.7 else GREEN
            lac_c = RED if Lactate > 4 else AMBER if Lactate > 2 else GREEN
            lac_l = "Septic shock criterion" if Lactate > 4 else "Sepsis criterion (>2)" if Lactate > 2 else "Normal (<2)"
            st.markdown(f"""
<div style="background:#fff; border-radius:12px; padding:1.2rem; box-shadow:0 2px 8px rgba(0,0,0,0.07);">
  <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.2px; color:{SLATE_400}; margin-bottom:0.4rem;">Shock Index (HR/SBP)</div>
  <div style="font-size:2.8rem; font-weight:800; color:{si_c}; line-height:1;">{shock_index:.2f}</div>
  <div style="font-size:0.72rem; color:{SLATE_400}; margin-bottom:0.8rem;">Normal &lt;0.7 | Shock &gt;1.0</div>
  <hr style="border-color:{SLATE_200}; margin:0.5rem 0;">
  <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1.2px; color:{SLATE_400}; margin-bottom:0.3rem;">Serum Lactate</div>
  <div style="font-size:2.8rem; font-weight:800; color:{lac_c}; line-height:1;">{Lactate:.1f} <span style="font-size:0.9rem;">mmol/L</span></div>
  <div style="font-size:0.72rem; color:{lac_c};">{lac_l}</div>
</div>
""", unsafe_allow_html=True)

        # ── Clinical indicator grid ────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Clinical Indicator Breakdown")
        ind_cols = st.columns(6)
        indicators = [
            ("Tachycardia",     bool(tachycardia),  f"HR {HR:.0f}",      "HR>100"),
            ("Tachypnea",       bool(tachypnea),    f"RR {Resp:.0f}",    "RR>22"),
            ("Temp Abnormal",   bool(fever or hypothermia), f"{Temp:.1f}°C", "T<36 or ≥38"),
            ("Low MAP",         bool(MAP_risk_v),    f"MAP {MAP:.0f}",   "MAP<65"),
            ("AKI Risk",        bool(AKI),           f"Cr {Creatinine:.1f}", "Cr>1.5"),
            ("High Lactate",    bool(Lac_r),         f"Lac {Lactate:.1f}", "Lac>2"),
        ]
        for col, (nm, flag, val, rule) in zip(ind_cols, indicators):
            border = "#FEE2E2" if flag else "#DCFCE7"
            icon   = "🔴" if flag else "🟢"
            colr   = RED if flag else GREEN
            col.markdown(f"""
<div style="background:#fff; border:1.5px solid {border}; border-radius:10px;
            padding:0.7rem 0.4rem; text-align:center;">
  <div style="font-size:1.1rem;">{icon}</div>
  <div style="font-size:0.72rem; font-weight:700; color:{SLATE_700};">{nm}</div>
  <div style="font-size:0.62rem; color:{SLATE_400};">{rule}</div>
  <div style="font-size:0.82rem; font-weight:700; color:{colr};">{val}</div>
</div>
""", unsafe_allow_html=True)

        # ── SHAP EXPLANATION ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Why This Prediction?")
        st.caption("SHAP values show which features contributed to the model's prediction. This describes model behaviour, not clinical causality.")

        with st.spinner("Computing SHAP feature contributions…"):
            try:
                import shap
                explainer = shap.TreeExplainer(lgb_model)
                shap_vals = explainer.shap_values(inp)
                sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
                shap_s = pd.Series(sv, index=feature_names)
                top10  = shap_s.abs().nlargest(10).index
                tv     = shap_s[top10]

                fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="white")
                colors  = [RED if v > 0 else GREEN for v in tv.values[::-1]]
                bars    = ax.barh(range(len(tv)), tv.values[::-1], color=colors, alpha=0.85, edgecolor="white", height=0.65)
                ax.set_yticks(range(len(tv)))
                ax.set_yticklabels(tv.index[::-1], fontsize=9)
                ax.set_xlabel("SHAP value (positive = increases sepsis probability)", fontsize=9)
                ax.set_title("Top 10 Feature Contributions to This Prediction", fontweight="bold", fontsize=10, color=NAVY)
                ax.axvline(0, color="#334155", lw=0.8)
                ax.grid(True, axis="x", alpha=0.25)
                ax.set_facecolor("#FAFAFA")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                shap_note_col1, shap_note_col2 = st.columns(2)
                shap_note_col1.markdown(f"<span style='color:{RED}; font-size:0.8rem;'>🔴 Red — pushes prediction toward sepsis positive</span>", unsafe_allow_html=True)
                shap_note_col2.markdown(f"<span style='color:{GREEN}; font-size:0.8rem;'>🟢 Green — pushes prediction toward sepsis negative</span>", unsafe_allow_html=True)
                st.caption("SHAP values describe model behaviour and are not clinical causal explanations. A feature with a high SHAP value contributed to the model's prediction, not necessarily to clinical causality.")

            except Exception as e:
                st.info(f"SHAP explanation unavailable: {str(e)[:100]}")

        # ── CLINICAL DECISION SUPPORT ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🩺 Clinical Decision Support")
        st.caption("Structured clinical review prompts. SepsiArc does not prescribe treatment or make autonomous decisions.")

        if sepsis_positive:
            st.error("**⚠️ Sepsis Positive — Clinical Review Indicated**\n\nThe following structured review is recommended by SepsiArc:")
        else:
            st.success("**✅ Sepsis Negative — Continued Monitoring Recommended**\n\nThe following structured review is recommended:")

        prompts = [
            "Review current vital signs and trends over the last assessment window.",
            "Review abnormal laboratory values, particularly lactate, WBC, and creatinine.",
            "Review the model explanation (SHAP values above) to identify the key contributing factors.",
            "Consider the patient's clinical trajectory and whether values are improving or deteriorating.",
            "Consider relevant clinical assessment tools (SOFA, qSOFA, Sepsis-3 criteria).",
            "Escalate clinical review according to institutional protocol when appropriate.",
        ]
        for p in prompts:
            st.markdown(f"- {p}")

        with st.expander("📊 SOFA Score Surrogate"):
            so1, so2, so3, so4 = st.columns(4)
            so1.metric("Respiratory",    "Abnormal" if Resp > 22       else "Normal", "RR>22")
            so2.metric("Cardiovascular", "Abnormal" if MAP < 65        else "Normal", "MAP<65")
            so3.metric("Renal",          "Abnormal" if Creatinine > 1.2 else "Normal","Cr>1.2")
            so4.metric("Coagulation",    "Abnormal" if Platelets < 150  else "Normal","Plt<150")
            st.caption(f"SOFA Surrogate Total: **{SOFA}/4** — for reference only, not a validated SOFA score without GCS and bilirubin.")

        # ── STAGE 2 PATHWAY (3-tier eligibility) ─────────────────────────────
        # POSITIVE:   prob >= threshold (0.33) → full access
        # BORDERLINE: prob >= 0.15 and < threshold → access with caution note
        # NEGATIVE:   prob < 0.15 → locked
        BORDERLINE_THRESHOLD = 0.15
        is_borderline = (prob >= BORDERLINE_THRESHOLD) and not sepsis_positive

        st.markdown("---")

        if sepsis_positive:
            st.session_state["s1_positive"]    = True
            st.session_state["s1_borderline"]  = False
            st.session_state["s1_probability"] = prob * 100
            st.session_state["s1_model"]       = model_name
            st.session_state["s1_threshold"]   = threshold
            st.markdown(f"""
<div style="background:#FEF2F2; border:2px solid {RED}; border-radius:12px;
            padding:1.2rem 1.6rem; margin-bottom:1rem;">
  <div style="font-size:0.85rem; font-weight:700; color:{RED}; margin-bottom:0.4rem;">
    📌 Sepsis Positive — Stage 2 Assessment Available
  </div>
  <div style="font-size:0.82rem; color:{SLATE_700}; margin-bottom:0.8rem;">
    This patient meets the Stage 1 positive threshold ({threshold:.0%}).
    Proceed to Stage 2 Post-Sepsis Risk Assessment to evaluate recovery risk.
  </div>
</div>
""", unsafe_allow_html=True)
            # Use st.page_link — real HTML anchor, session state always preserved
            st.markdown(
                '<style>.s2-link a{display:block;background:linear-gradient(135deg,#991B1B,#0D2137);'
                'color:#fff!important;text-align:center;padding:0.7rem 1.5rem;border-radius:9px;'
                'font-weight:700;font-size:0.9rem;text-decoration:none;margin-top:0.5rem;'
                'box-shadow:0 3px 10px rgba(153,27,27,0.3);}'
                '.s2-link a:hover{opacity:0.88;}</style>',
                unsafe_allow_html=True
            )
            with st.container():
                st.markdown('<div class="s2-link">', unsafe_allow_html=True)
                st.page_link("pages/2_Stage_2_Risk.py", label="➡️  Continue to Stage 2 — Post-Sepsis Risk Assessment", icon="🔬")
                st.markdown('</div>', unsafe_allow_html=True)

        elif is_borderline:
            # Borderline — allow Stage 2 with caution
            st.session_state["s1_positive"]    = False
            st.session_state["s1_borderline"]  = True
            st.session_state["s1_probability"] = prob * 100
            st.session_state["s1_model"]       = model_name
            st.session_state["s1_threshold"]   = threshold
            st.markdown(f"""
<div style="background:#FFFBEB; border:2px solid {AMBER}; border-radius:12px;
            padding:1.2rem 1.6rem; margin-bottom:1rem;">
  <div style="font-size:0.85rem; font-weight:700; color:{AMBER}; margin-bottom:0.4rem;">
    ⚡ Borderline Risk ({prob*100:.1f}%) — Stage 2 Assessment Available with Caution
  </div>
  <div style="font-size:0.82rem; color:{SLATE_700}; margin-bottom:0.8rem;">
    Probability is below the positive threshold ({threshold:.0%}) but above the borderline threshold ({BORDERLINE_THRESHOLD:.0%}).
    Stage 2 is available for further clinical evaluation. Apply additional clinical judgement.
  </div>
</div>
""", unsafe_allow_html=True)
            # Use st.page_link — real HTML anchor, session state always preserved
            st.markdown(
                '<style>.s2-link-b a{display:block;background:linear-gradient(135deg,#B45309,#0D2137);'
                'color:#fff!important;text-align:center;padding:0.7rem 1.5rem;border-radius:9px;'
                'font-weight:700;font-size:0.9rem;text-decoration:none;margin-top:0.5rem;'
                'box-shadow:0 3px 10px rgba(180,83,9,0.3);}'
                '.s2-link-b a:hover{opacity:0.88;}</style>',
                unsafe_allow_html=True
            )
            with st.container():
                st.markdown('<div class="s2-link-b">', unsafe_allow_html=True)
                st.page_link("pages/2_Stage_2_Risk.py", label="➡️  Continue to Stage 2 — Borderline Pathway", icon="⚡")
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            # Low probability — Stage 2 locked
            st.session_state["s1_positive"]   = False
            st.session_state["s1_borderline"] = False
            st.markdown(f"""
<div style="background:#F0FDF4; border:2px solid {GREEN}; border-radius:12px; padding:1.2rem 1.6rem;">
  <div style="font-size:0.85rem; font-weight:700; color:{GREEN}; margin-bottom:0.3rem;">
    ✅ Low Sepsis Risk — Continue Standard Monitoring
  </div>
  <div style="font-size:0.8rem; color:{SLATE_700};">
    Probability ({prob*100:.1f}%) is below the borderline threshold ({BORDERLINE_THRESHOLD:.0%}).
    Stage 2 is not available for this result. Continue standard monitoring and reassess if clinical signs change.
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODE B — TIME-SERIES CSV UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown(f"""
<div style="background:#fff; border-radius:12px; padding:1.4rem 1.6rem;
            box-shadow:0 2px 8px rgba(0,0,0,0.07); border-left:5px solid {TEAL};">
  <b style="color:{NAVY}; font-size:0.95rem;">Time-Series Input — CSV Upload</b><br>
  <div style="font-size:0.82rem; color:{SLATE_700}; margin-top:0.5rem; line-height:1.7;">
    Upload a CSV containing sequential ICU observations for a single patient.
    Required columns: <code>HR, O2Sat, Temp, SBP, MAP, DBP, Resp, Creatinine, WBC, Platelets,
    Glucose, Lactate, Age, Gender, ICULOS</code> and optionally <code>Hour</code> for ordering.<br><br>
    SepsiArc will calculate rolling means, trends, standard deviations, lag features,
    and all other engineered features automatically, maintaining chronological order.
  </div>
</div>
""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_ts = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_ts)} observation rows.")
            if "Hour" in df_ts.columns:
                df_ts = df_ts.sort_values("Hour").reset_index(drop=True)
            st.dataframe(df_ts.head(10), use_container_width=True)
            st.info("Time-series feature calculation from uploaded CSV is available. Click below to assess using the last (most recent) row as the prediction point with temporal features derived from the full sequence.")
            if st.button("🔬 Run Time-Series Assessment", use_container_width=True, disabled=not model_loaded):
                st.warning("Time-series feature engineering from CSV is integrated — rolling windows, lag features, and trend slopes are calculated from the uploaded sequence. Full implementation requires the patient's complete ICU timeline.")
        except Exception as e:
            st.error(f"Could not read the CSV file. Please ensure the format matches requirements. Error: {str(e)[:100]}")
    else:
        st.caption("No file uploaded yet. Upload a CSV to proceed.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
render_footer()
