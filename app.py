"""
app.py  --  SepsiArc Home Dashboard
Sections: Navbar · Hero · Sepsis Stats · About Project ·
          Sepsis Overview · Epidemiology · Access Modules · Footer
"""
import os, sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp')  # ensure matplotlib cache is writable on Cloud
import matplotlib
matplotlib.use('Agg')  # must be before any other matplotlib/pyplot import
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.ui_components import (
    render_header, render_footer, render_sidebar,
    render_disclaimer_banner, stat_card,
    NAVY, TEAL, TEAL_LIGHT, GREEN, AMBER, RED,
    SLATE_100, SLATE_200, SLATE_400, SLATE_700, WHITE,
)

st.set_page_config(
    page_title="SepsiArc – Clinical Decision Support",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_header(active="home")
render_sidebar("home")
render_disclaimer_banner()

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="background:linear-gradient(135deg,' + NAVY + ' 0%,#0e3058 55%,' + TEAL + ' 100%);'
    'border-radius:16px;padding:2.8rem 2.5rem;margin-bottom:2rem;position:relative;overflow:hidden;">'
    '<div style="max-width:640px;position:relative;z-index:2;">'
    '<div style="display:inline-block;background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.25);'
    'border-radius:999px;padding:0.22rem 1rem;font-size:0.7rem;font-weight:600;color:#7DD3FC;'
    'letter-spacing:1.5px;text-transform:uppercase;margin-bottom:0.9rem;">'
    'Dual-Stage AI &nbsp;&middot;&nbsp; Explainable ML &nbsp;&middot;&nbsp; ICU Decision Support'
    '</div>'
    '<h1 style="font-size:2.6rem;font-weight:800;color:#fff;line-height:1.15;margin:0 0 0.4rem;">SepsiArc</h1>'
    '<p style="font-size:1rem;color:#BAE6FD;font-weight:500;margin:0 0 0.35rem;">From Sepsis Detection to Recovery Risk.</p>'
    '<p style="font-size:0.86rem;color:rgba(255,255,255,0.65);line-height:1.7;margin:0 0 1.6rem;">'
    'An explainable dual-stage machine learning framework for early sepsis detection and '
    'post-sepsis risk stratification — providing transparent, clinically meaningful insights '
    'for decision support and research.'
    '</p>'
    '</div>'
    '<div style="position:absolute;right:-60px;top:-60px;width:300px;height:300px;border-radius:50%;'
    'background:rgba(14,116,144,0.18);border:2px solid rgba(14,116,144,0.3);"></div>'
    '</div>',
    unsafe_allow_html=True
)

# Single Start Assessment CTA
cta1, cta2, _ = st.columns([1.2, 1.2, 4])
with cta1:
    if st.button("▶  Start Patient Assessment", use_container_width=True):
        st.switch_page("pages/1_Stage_1_Detection.py")
with cta2:
    if st.button("💬  Clinical Assistant", use_container_width=True):
        st.switch_page("pages/3_AI_Assistant.py")

st.markdown("<br>", unsafe_allow_html=True)

# ── SEPSIS STATS ──────────────────────────────────────────────────────────────
st.markdown("### ⚠️ Sepsis: A Global Health Emergency")
c1, c2, c3, c4 = st.columns(4)
stat_card(c1, "Annual Global Cases",         "48.9M",  color=TEAL)
stat_card(c2, "Deaths per Year",             "11M+",   color=RED,   sub="1 in 5 deaths globally")
stat_card(c3, "ICU Patients Affected",       "~30%",   color=AMBER)
stat_card(c4, "Critical Window",             "6 hrs",  color=GREEN, sub="Hour-1 Bundle target")

st.markdown("<br>", unsafe_allow_html=True)

# ── ABOUT THE PROJECT ─────────────────────────────────────────────────────────
st.markdown("### About SepsiArc")
ab1, ab2 = st.columns([3, 2])
with ab1:
    st.markdown(
        '<div style="background:#fff;border-radius:12px;padding:1.4rem 1.7rem;'
        'box-shadow:0 2px 10px rgba(14,116,144,0.08);border-left:5px solid ' + TEAL + ';margin-bottom:0.6rem;">'
        '<p style="color:' + SLATE_700 + ';line-height:1.85;font-size:0.89rem;margin:0;">'
        '<b style="color:' + NAVY + ';">SepsiArc</b> is a dual-stage machine learning framework developed to support '
        'early sepsis detection and post-sepsis risk assessment in ICU settings. '
        'The system combines predictive modelling with explainable AI (SHAP) to provide '
        'transparent and interpretable insights for clinical decision-support and research.<br><br>'
        'Built on the <b>PhysioNet/CinC Challenge 2019</b> dataset — comprising over 1.55 million '
        'ICU observations from 40,336 patients — SepsiArc delivers structured, evidence-driven outputs.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )
with ab2:
    for title, desc, col in [
        ("Dual-Stage ML",           "Separate models for sepsis detection and post-sepsis risk stratification.",  TEAL),
        ("Explainable AI (SHAP)",   "SHAP-based explanations for every prediction — full model transparency.",    GREEN),
        ("Clinical Decision Tools", "Structured review prompts — never autonomous treatment decisions.",           AMBER),
    ]:
        st.markdown(
            '<div style="background:#fff;border-radius:10px;padding:0.85rem 1.1rem;'
            'box-shadow:0 1px 6px rgba(0,0,0,0.06);border-left:4px solid ' + col + ';margin-bottom:0.6rem;">'
            '<div style="font-size:0.84rem;font-weight:700;color:' + NAVY + ';">' + title + '</div>'
            '<div style="font-size:0.75rem;color:' + SLATE_700 + ';margin-top:2px;">' + desc + '</div>'
            '</div>',
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── MODEL PERFORMANCE SUMMARY ─────────────────────────────────────────────────
st.markdown("### Model Performance Summary")
st.caption(
    "Experimental test results on the held-out patient split (PhysioNet/CinC Challenge 2019). "
    "These are not clinical validation results. Stage 1 and Stage 2 solve different tasks and their metrics are not directly comparable."
)

_perf_cols = st.columns(2)
with _perf_cols[0]:
    st.markdown(
        f'<div style="background:#fff;border-radius:14px;padding:1.3rem 1.6rem;'
        f'box-shadow:0 2px 12px rgba(14,116,144,0.08);border-top:5px solid {TEAL};">'
        f'<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:1.3px;color:{SLATE_400};margin-bottom:0.7rem;">Stage 1 — Hourly Sepsis Screening</div>'
        f'<div style="font-size:0.78rem;color:{SLATE_700};line-height:2.0;">'
        f'<b>Algorithm:</b> XGBoost (Regularized Cost-Sensitive, Model A)<br>'
        f'<b>Features:</b> 135 engineered clinical features<br>'
        f'<b>Decision Threshold:</b> 0.143 (Sepsis Alert) &nbsp;|&nbsp; 0.080 (Borderline)<br>'
        f'<b>Accuracy:</b> 68.05% &nbsp;·&nbsp; <b>Precision:</b> 11.20% &nbsp;·&nbsp; <b>Recall:</b> 70.97%<br>'
        f'<b>F1-score:</b> 19.35% &nbsp;·&nbsp; <b>ROC-AUC:</b> 0.7587 &nbsp;·&nbsp; <b>PR-AUC:</b> 0.1605'
        f'</div>'
        f'<div style="margin-top:0.7rem;font-size:0.69rem;color:{SLATE_400};font-style:italic;">'
        f'Low precision reflects severe class imbalance (~5.5% sepsis rows). Threshold is set low to maximise recall.'
        f'</div></div>',
        unsafe_allow_html=True,
    )

with _perf_cols[1]:
    st.markdown(
        f'<div style="background:#fff;border-radius:14px;padding:1.3rem 1.6rem;'
        f'box-shadow:0 2px 12px rgba(0,0,0,0.07);border-top:5px solid {RED};">'
        f'<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:1.3px;color:{SLATE_400};margin-bottom:0.7rem;">Stage 2 — Patient-Level Risk Stratification</div>'
        f'<div style="font-size:0.78rem;color:{SLATE_700};line-height:2.0;">'
        f'<b>Algorithm:</b> XGBoost (Patient-Level, max_depth=2)<br>'
        f'<b>Features:</b> 65 patient-level aggregate features<br>'
        f'<b>Decision Threshold:</b> 0.29 &nbsp;|&nbsp; Target: Risk_50pct physiological-risk proxy<br>'
        f'<b>Accuracy:</b> 71.48% &nbsp;·&nbsp; <b>Precision:</b> 62.35% &nbsp;·&nbsp; <b>Recall:</b> 85.48%<br>'
        f'<b>F1-score:</b> 72.11% &nbsp;·&nbsp; <b>ROC-AUC:</b> 0.8294 &nbsp;·&nbsp; <b>PR-AUC:</b> 0.8001'
        f'</div>'
        f'<div style="margin-top:0.7rem;font-size:0.69rem;color:{SLATE_400};font-style:italic;">'
        f'Stage 2 target is an engineered physiological-risk proxy — not a validated mortality or survivorship prediction.'
        f'</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── SEPSIS OVERVIEW ───────────────────────────────────────────────────────────
st.markdown("### Sepsis at a Glance")
st.caption("Key clinical criteria, warning signs, and biomarkers — evidence-based overview.")

ov1, ov2, ov3 = st.columns(3)
with ov1:
    st.markdown(
        '<div style="background:#fff;border-radius:12px;padding:1.15rem 1.35rem;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.06);height:100%;border-top:4px solid ' + RED + ';">'
        '<div style="font-weight:700;color:' + NAVY + ';margin-bottom:0.65rem;font-size:0.9rem;">⚠️ SIRS Criteria (≥2 of 4)</div>'
        '<ul style="font-size:0.79rem;color:' + SLATE_700 + ';padding-left:1rem;margin:0;line-height:1.9;">'
        '<li>Temperature &gt; 38&deg;C or &lt; 36&deg;C</li>'
        '<li>Heart Rate &gt; 90 bpm</li>'
        '<li>Respiratory Rate &gt; 20 br/min</li>'
        '<li>WBC &gt; 12,000 or &lt; 4,000 cells/&micro;L</li>'
        '</ul>'
        '<div style="margin-top:0.7rem;font-size:0.7rem;color:' + SLATE_400 + ';font-style:italic;">'
        'SIRS + suspected infection &rarr; Sepsis concern</div>'
        '</div>',
        unsafe_allow_html=True
    )
with ov2:
    st.markdown(
        '<div style="background:#fff;border-radius:12px;padding:1.15rem 1.35rem;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.06);height:100%;border-top:4px solid ' + AMBER + ';">'
        '<div style="font-weight:700;color:' + NAVY + ';margin-bottom:0.65rem;font-size:0.9rem;">📋 qSOFA Score (≥2 = concern)</div>'
        '<ul style="font-size:0.79rem;color:' + SLATE_700 + ';padding-left:1rem;margin:0;line-height:1.9;">'
        '<li>Respiratory Rate &ge; 22 br/min</li>'
        '<li>Altered Mental Status (GCS &lt; 15)</li>'
        '<li>Systolic BP &le; 100 mmHg</li>'
        '</ul>'
        '<div style="margin-top:0.7rem;font-size:0.7rem;color:' + SLATE_400 + ';font-style:italic;">'
        'Bedside tool &mdash; no labs required</div>'
        '</div>',
        unsafe_allow_html=True
    )
with ov3:
    st.markdown(
        '<div style="background:#fff;border-radius:12px;padding:1.15rem 1.35rem;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.06);height:100%;border-top:4px solid ' + TEAL + ';">'
        '<div style="font-weight:700;color:' + NAVY + ';margin-bottom:0.65rem;font-size:0.9rem;">🩸 Key Biomarkers</div>'
        '<ul style="font-size:0.79rem;color:' + SLATE_700 + ';padding-left:1rem;margin:0;line-height:1.9;">'
        '<li><b>Lactate &gt; 2 mmol/L</b> &mdash; hypoperfusion</li>'
        '<li><b>Lactate &gt; 4 mmol/L</b> &mdash; septic shock</li>'
        '<li><b>Creatinine &gt; 1.5</b> &mdash; AKI risk</li>'
        '<li><b>Platelets &lt; 100</b> &mdash; coagulopathy</li>'
        '<li><b>MAP &lt; 65 mmHg</b> &mdash; vasopressor needed</li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True
    )

# SepsiArc workflow note
st.markdown(
    '<div style="background:#E0F7FA;border:1px solid #67E8F9;border-radius:10px;'
    'padding:0.9rem 1.35rem;margin-top:0.75rem;font-size:0.81rem;color:' + SLATE_700 + ';line-height:1.7;">'
    '<b style="color:' + TEAL + ';">SepsiArc Stage 1</b> screens for sepsis risk using 135 engineered clinical features. '
    'A <b>Sepsis Alert (&ge;14.3%)</b> or <b>Borderline (8.0–14.3%)</b> result unlocks '
    '<b>Stage 2</b> for post-sepsis physiological-risk stratification with SHAP and LIME explanations. '
    'All outputs are for clinician review — not autonomous clinical decisions.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ── EPIDEMIOLOGY ──────────────────────────────────────────────────────────────
st.markdown("### Sepsis Epidemiology")
st.caption("Trend data and organ involvement — contextualising the global clinical burden.")

years     = list(range(2010, 2024))
incidence = [44.2,45.0,45.8,46.3,46.9,47.5,48.0,48.9,49.2,49.7,50.3,50.9,51.4,52.0]
deaths    = [12.1,11.9,11.6,11.4,11.2,11.1,11.0,11.0,10.9,10.8,11.3,11.0,10.8,10.5]
icu_mort  = [32,31,30,29,28,28,27,26,26,25,27,26,25,24]
organs    = ["Cardiovascular","Respiratory","Renal","Neurological","Hepatic","Hematologic"]
organ_pct = [70,65,40,30,20,15]

fig, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor="#F0F9FF")
fig.patch.set_facecolor("#F0F9FF")

ax = axes[0]; ax.set_facecolor("#FFFFFF")
ax.fill_between(years, incidence, alpha=0.12, color=TEAL)
ax.plot(years, incidence, "o-", color=TEAL, lw=2, ms=4, label="Cases (M)")
ax2 = ax.twinx()
ax2.plot(years, deaths, "s--", color=RED, lw=1.8, ms=4, label="Deaths (M)")
ax2.set_ylabel("Deaths (millions)", color=RED, fontsize=8); ax2.tick_params(colors=RED, labelsize=7)
ax.set_xlabel("Year", fontsize=8); ax.set_ylabel("Cases (millions)", color=TEAL, fontsize=8)
ax.tick_params(colors=TEAL, labelsize=7)
ax.set_title("Global Sepsis Incidence & Mortality", fontweight="bold", fontsize=9)
lines1, labs1 = ax.get_legend_handles_labels(); lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, loc="upper left", fontsize=7)
ax.grid(True, alpha=0.2)

ax = axes[1]; ax.set_facecolor("#FFFFFF")
ax.fill_between(years, icu_mort, alpha=0.1, color=AMBER)
ax.plot(years, icu_mort, "D-", color=AMBER, lw=2, ms=4)
ax.set_xlabel("Year", fontsize=8); ax.set_ylabel("ICU Mortality (%)", fontsize=8)
ax.set_title("ICU Sepsis Mortality Trend (%)", fontweight="bold", fontsize=9)
ax.tick_params(labelsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(20,38)

ax = axes[2]; ax.set_facecolor("#FFFFFF")
bars = ax.barh(organs, organ_pct, color=TEAL, alpha=0.78, edgecolor="white")
for bar, val in zip(bars, organ_pct):
    ax.text(val+0.5, bar.get_y()+bar.get_height()/2, f"{val}%",
            va="center", fontsize=7.5, fontweight="bold", color="#334155")
ax.set_xlabel("% of Sepsis Patients Affected", fontsize=8)
ax.set_title("Organ Systems Affected in Sepsis", fontweight="bold", fontsize=9)
ax.tick_params(labelsize=7.5); ax.grid(True, axis="x", alpha=0.2); ax.set_xlim(0,85)

plt.tight_layout(pad=2.0)
_buf_home = __import__('io').BytesIO()
fig.savefig(_buf_home, format='png', dpi=100, bbox_inches='tight', facecolor='white')
plt.close(fig)
_buf_home.seek(0)
st.image(_buf_home.getvalue(), width='stretch')

st.markdown("<br>", unsafe_allow_html=True)

# ── ACCESS CLINICAL MODULES ───────────────────────────────────────────────────
st.markdown("### Clinical Modules")
st.caption("Navigate to the clinical assessment and decision-support tools.")

s1_pos  = st.session_state.get("s1_positive",  False)
s1_bord = st.session_state.get("s1_borderline", False)
s2_ok   = s1_pos or s1_bord

m1, m2, m3 = st.columns(3)
modules = [
    (m1, "[S1]", "Stage 1 — Detection",
     "Upload hourly ICU observation timeline (PSV/CSV/TXT). Receive a per-hour sepsis probability score and SHAP/LIME explanation.",
     TEAL, "pages/1_Stage_1_Detection.py", "Open Stage 1"),
    (m2, "[S2]", "Stage 2 — Risk Assessment",
     ("Assess post-sepsis risk trajectory. Includes SHAP explanation and structured clinical review prompts."
      if s2_ok else "Run Stage 1 first. Available after a positive or borderline detection result."),
     RED if s2_ok else SLATE_400,
     "pages/2_Stage_2_Risk.py", "Open Stage 2"),
    (m3, "[AI]", "Clinical Assistant",
     "Evidence-based Q&A about sepsis, clinical scoring systems, SHAP, and how SepsiArc works.",
     GREEN, "pages/3_AI_Assistant.py", "Open Assistant"),
]
for col, icon, title, desc, color, page, btn in modules:
    with col:
        st.markdown(
            '<div style="background:#fff;border-radius:14px;padding:1.25rem 1.1rem;'
            'box-shadow:0 2px 12px rgba(14,116,144,0.08);border-top:4px solid ' + color + ';min-height:150px;">'
            '<div style="font-size:1.5rem;">' + icon + '</div>'
            '<div style="font-weight:700;color:' + NAVY + ';margin:0.35rem 0 0.25rem;font-size:0.88rem;">' + title + '</div>'
            '<div style="font-size:0.75rem;color:' + SLATE_700 + ';line-height:1.55;">' + desc + '</div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("<div style='margin-top:0.35rem;'></div>", unsafe_allow_html=True)
        if st.button(btn, key="mod_" + title, use_container_width=True):
            st.switch_page(page)

st.markdown("<br>", unsafe_allow_html=True)

# ── DISCLAIMER ────────────────────────────────────────────────────────────────
st.error(
    "**⚕ Research & Clinical Decision Support Only** — "
    "SepsiArc outputs are **not a medical diagnosis, prognosis, or treatment recommendation**. "
    "All outputs must be interpreted by a qualified healthcare professional. "
    "Stage 2 Post-Sepsis Risk is a project-defined proxy — not a validated survivorship prediction."
)

# ── FOOTER ────────────────────────────────────────────────────────────────────
render_footer()
