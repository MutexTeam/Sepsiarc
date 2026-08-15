"""
pages/5_About.py  --  SepsiArc About Page
About the project, methodology, dataset, and clinical disclaimer.
No team section — team has its own page if needed.
"""
import streamlit as st
import os, sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from utils.ui_components import (
    render_header, render_footer, render_sidebar,
    render_disclaimer_banner,
    NAVY, TEAL, GREEN, AMBER, RED, SLATE_200, SLATE_400, SLATE_700,
)

st.set_page_config(
    page_title="SepsiArc – About",
    page_icon="ℹ️",
    layout="wide",
)

render_header(
    page_title="About SepsiArc",
    page_subtitle="Project mission, dual-stage methodology, dataset, and clinical transparency statement.",
    icon="ℹ️",
    active="about",
)
render_sidebar("about")
render_disclaimer_banner()

# ── Mission ─────────────────────────────────────────────────────────────────────
st.markdown("### 🎯 Project Mission")
st.markdown(
    '<div style="background:#fff;border-radius:14px;padding:1.5rem 1.9rem;'
    'box-shadow:0 2px 10px rgba(14,116,144,0.08);border-left:5px solid ' + TEAL + ';margin-bottom:1.2rem;">'
    '<p style="color:' + SLATE_700 + ';font-size:0.9rem;line-height:1.85;margin:0;">'
    '<b style="color:' + NAVY + ';font-size:0.97rem;">SepsiArc</b> was developed to address the critical need for '
    '<b>early, explainable, and clinically integrated</b> machine learning tools in ICU settings. '
    'Sepsis remains one of the leading causes of preventable hospital deaths globally — affecting nearly '
    '49 million people each year. Early detection and timely intervention significantly reduce mortality, '
    'yet reliable, interpretable decision-support tools remain limited.<br><br>'
    'Our mission is to provide a <b>dual-stage framework</b> that not only detects sepsis risk early, '
    'but follows the patient journey through post-sepsis risk assessment — '
    'maintaining full transparency through explainable AI (SHAP) and a strict commitment to '
    'responsible clinical decision support.'
    '</p>'
    '</div>',
    unsafe_allow_html=True
)

# ── Methodology ────────────────────────────────────────────────────────────────
st.markdown("### 🔬 Methodology")
m1, m2 = st.columns(2)
with m1:
    st.markdown(
        '<div style="background:#fff;border-radius:12px;padding:1.3rem 1.5rem;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.06);border-top:4px solid ' + TEAL + ';margin-bottom:0.8rem;">'
        '<div style="font-size:0.9rem;font-weight:700;color:' + NAVY + ';margin-bottom:0.6rem;">🧬 Stage 1 — Sepsis Detection</div>'
        '<ul style="font-size:0.81rem;color:' + SLATE_700 + ';padding-left:1rem;margin:0;line-height:1.9;">'
        '<li><b>Model:</b> LightGBM-Calibrated (isotonic regression)</li>'
        '<li><b>Features:</b> 79 engineered clinical features</li>'
        '<li><b>Raw inputs:</b> 15 clinical parameters (vitals + labs + demographics)</li>'
        '<li><b>Positive threshold:</b> &ge; 0.33 (33%) &rarr; Stage 2 accessible</li>'
        '<li><b>Borderline threshold:</b> 0.15&ndash;0.33 &rarr; Stage 2 accessible with caution</li>'
        '<li><b>Clinical gate:</b> capped at 0.10 if no SIRS/qSOFA criteria met</li>'
        '<li><b>AUROC:</b> 0.8229 on PhysioNet test set</li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True
    )

with m2:
    st.markdown(
        '<div style="background:#fff;border-radius:12px;padding:1.3rem 1.5rem;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.06);border-top:4px solid ' + RED + ';margin-bottom:0.8rem;">'
        '<div style="font-size:0.9rem;font-weight:700;color:' + NAVY + ';margin-bottom:0.6rem;">🔬 Stage 2 — Post-Sepsis Risk</div>'
        '<ul style="font-size:0.81rem;color:' + SLATE_700 + ';padding-left:1rem;margin:0;line-height:1.9;">'
        '<li><b>Model:</b> LightGBM (scale_pos_weight for class imbalance)</li>'
        '<li><b>Features:</b> 65 patient-level clinical features</li>'
        '<li><b>Eligibility:</b> Stage 1 positive (&ge;33%) or borderline (15&ndash;33%)</li>'
        '<li><b>Risk threshold:</b> &ge; 0.58 = HIGH RISK</li>'
        '<li><b>No data leakage:</b> label-generating features excluded</li>'
        '<li><b>AUROC:</b> 0.9631 on PhysioNet test set</li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True
    )

# SHAP section
st.markdown(
    '<div style="background:#fff;border-radius:12px;padding:1.3rem 1.5rem;'
    'box-shadow:0 2px 8px rgba(0,0,0,0.06);border-top:4px solid #7C3AED;margin-bottom:1rem;">'
    '<div style="font-size:0.9rem;font-weight:700;color:' + NAVY + ';margin-bottom:0.6rem;">📊 Explainable AI — SHAP</div>'
    '<div style="font-size:0.81rem;color:' + SLATE_700 + ';line-height:1.8;">'
    'Both Stage 1 and Stage 2 use <b>SHAP (SHapley Additive exPlanations)</b> to provide '
    'feature-level explanations for every prediction. SHAP values indicate which clinical inputs '
    'contributed most to the model\'s decision — in which direction and by how much.<br><br>'
    '<b>Important:</b> SHAP values describe <i>model behaviour</i>, not clinical causality. '
    'A high SHAP value means a feature strongly influenced the model\'s prediction — '
    'not that it is the clinical cause of sepsis.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Dataset ─────────────────────────────────────────────────────────────────────
st.markdown("### 📂 Training Dataset")
ds1, ds2 = st.columns([3, 2])
with ds1:
    st.markdown(
        '<div style="background:#fff;border-radius:12px;padding:1.3rem 1.5rem;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.06);border-left:4px solid ' + TEAL + ';">'
        '<div style="font-size:0.9rem;font-weight:700;color:' + NAVY + ';margin-bottom:0.6rem;">'
        'PhysioNet/Computing in Cardiology Challenge 2019'
        '</div>'
        '<div style="font-size:0.81rem;color:' + SLATE_700 + ';line-height:1.8;">'
        '<b>Source:</b> PhysioNet (publicly available) &mdash; Reyna et al., 2019<br>'
        '<b>Patients:</b> 40,336 ICU patients across multiple hospital systems<br>'
        '<b>Observations:</b> ~1.55 million hourly ICU time-points<br>'
        '<b>Sepsis prevalence:</b> ~1.8% (severe class imbalance addressed via SMOTE + scale_pos_weight)<br>'
        '<b>Original features:</b> 40 clinical variables (vitals + labs + demographics)<br><br>'
        '<b>Citation:</b><br>'
        '<span style="font-style:italic;color:' + SLATE_400 + ';">'
        'Reyna MA, Josef C, Seyedi S et al. (2019). '
        '"Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019." '
        'Critical Care Medicine.'
        '</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
with ds2:
    for stat, val, color in [
        ("ICU Patients",       "40,336", TEAL),
        ("Hourly Observations","~1.55M", NAVY),
        ("Sepsis Prevalence",  "~1.8%",  AMBER),
        ("Stage 1 Features",   "79",     GREEN),
        ("Stage 2 Features",   "65",     RED),
    ]:
        st.markdown(
            '<div style="background:#fff;border-radius:10px;padding:0.65rem 1rem;margin-bottom:0.45rem;'
            'box-shadow:0 1px 6px rgba(0,0,0,0.06);border-left:4px solid ' + color + ';'
            'display:flex;justify-content:space-between;align-items:center;">'
            '<span style="font-size:0.79rem;color:' + SLATE_700 + ';font-weight:500;">' + stat + '</span>'
            '<span style="font-size:0.97rem;font-weight:800;color:' + color + ';">' + val + '</span>'
            '</div>',
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── What SepsiArc Is NOT ────────────────────────────────────────────────────────
st.markdown("### ⚕ Clinical & Research Disclaimer")
st.error("""
**SepsiArc is a clinical decision-support and research application — not an autonomous diagnostic system.**

- Model outputs are **not a medical diagnosis, prognosis, or treatment recommendation**.
- All outputs must be interpreted by an appropriately qualified healthcare professional in the context of the full clinical picture.
- **Stage 2 Post-Sepsis Risk** is a project-defined proxy severity classification. It is **not a validated long-term mortality or survivorship prediction**.
- The system does not replace clinical judgment, laboratory testing, imaging, or physician assessment.
- This application was developed using the PhysioNet/CinC Challenge 2019 dataset and has not undergone regulatory review, clinical trial validation, or institutional deployment review.
- **For research and educational use only.**
""")

# ── Sepsis information ───────────────────────────────────────────────────────────
st.markdown("### 🦠 What is Sepsis?")
si1, si2 = st.columns(2)
with si1:
    st.markdown(
        '<div style="background:#fff;border-radius:12px;padding:1.2rem 1.4rem;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.06);border-left:4px solid ' + TEAL + ';margin-bottom:0.8rem;">'
        '<div style="font-size:0.88rem;font-weight:700;color:' + NAVY + ';margin-bottom:0.5rem;">Definition</div>'
        '<div style="font-size:0.8rem;color:' + SLATE_700 + ';line-height:1.8;">'
        'Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. '
        'It is one of the most common and deadly conditions encountered in ICU settings worldwide. '
        'Without timely diagnosis and treatment, sepsis can rapidly progress to septic shock, '
        'multi-organ failure, and death.<br><br>'
        'The <b>Sepsis-3 definition</b> (Singer et al., 2016) defines sepsis as suspected or documented infection '
        'plus a SOFA score increase of &ge; 2 points.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
with si2:
    st.markdown(
        '<div style="background:#fff;border-radius:12px;padding:1.2rem 1.4rem;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.06);border-left:4px solid ' + RED + ';margin-bottom:0.8rem;">'
        '<div style="font-size:0.88rem;font-weight:700;color:' + NAVY + ';margin-bottom:0.5rem;">Why Early Detection Matters</div>'
        '<div style="font-size:0.8rem;color:' + SLATE_700 + ';line-height:1.8;">'
        'Every hour of delayed sepsis treatment is associated with a <b>7–10% increase in mortality</b>. '
        'The Surviving Sepsis Campaign <b>Hour-1 Bundle</b> recommends blood cultures, lactate measurement, '
        'and antibiotic administration within 1 hour of recognition.<br><br>'
        'Machine learning tools like SepsiArc can assist clinicians by <b>flagging at-risk patients earlier</b> '
        'using continuous ICU data streams.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── TEAM ──────────────────────────────────────────────────────────────────────
import base64

def _photo_b64(path_no_ext, search_dirs, exts=("jpg","jpeg","png")):
    """Return (data_uri, mime) for the first matching file, or (None, None)."""
    for d in search_dirs:
        for ext in exts:
            p = os.path.join(d, f"{path_no_ext}.{ext}")
            if os.path.exists(p):
                mime = "image/png" if ext == "png" else "image/jpeg"
                with open(p, "rb") as f:
                    return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}", mime
    return None, None

def _circle_photo_html(data_uri, fallback_icon, size=100, border_color="#0E7490",
                        border_width=3, shadow="0 4px 14px rgba(14,116,144,0.22)"):
    """Return HTML for a circular photo or a styled placeholder if no photo."""
    if data_uri:
        return (
            f'<div style="width:{size}px;height:{size}px;border-radius:50%;overflow:hidden;'
            f'margin:0 auto 0.8rem;border:{border_width}px solid {border_color};'
            f'box-shadow:{shadow};">'
            f'<img src="{data_uri}" style="width:100%;height:100%;object-fit:cover;display:block;"/>'
            f'</div>'
        )
    return (
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
        f'background:linear-gradient(135deg,#E0F7FA,#BAE6FD);'
        f'border:{border_width}px solid {border_color};'
        f'display:flex;align-items:center;justify-content:center;'
        f'margin:0 auto 0.8rem;font-size:{int(size*0.4)}px;'
        f'box-shadow:{shadow};">{fallback_icon}</div>'
    )

st.markdown("### 👥 Our Team")
st.caption("The researchers and developers behind the SepsiArc project.")

# Paths
_mentor_dir  = os.path.join(_root, "public", "images", "mentor")
_team_dir    = os.path.join(_root, "public", "images", "team")

# ── Mentor card (centered) ────────────────────────────────────────────────────
mentor_uri, _ = _photo_b64("mentor", [_mentor_dir])
mentor_circle  = _circle_photo_html(mentor_uri, "&#x1F468;&#x200D;&#x1F3EB;",
                                    size=120, border_width=4,
                                    shadow="0 6px 20px rgba(14,116,144,0.28)")

_, mentor_col, _ = st.columns([1.5, 2, 1.5])
with mentor_col:
    st.markdown(
        '<div style="background:#fff;border-radius:16px;padding:1.8rem 1.5rem;text-align:center;'
        'box-shadow:0 4px 20px rgba(14,116,144,0.12);border-top:5px solid ' + TEAL + ';margin-bottom:1.5rem;">'
        + mentor_circle +
        '<div style="font-size:1rem;font-weight:700;color:' + NAVY + ';">Dr. Dinesh L</div>'
        '<div style="font-size:0.8rem;color:' + TEAL + ';font-weight:600;margin-top:3px;">Project Mentor / Guide</div>'
        '<div style="font-size:0.73rem;color:' + SLATE_400 + ';margin-top:5px;line-height:1.6;">'
        'Department of Computer Science and Engineering &nbsp;&middot;&nbsp; Mangalore Institute of Technology and Engineering'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

# ── Team member cards (4 columns) ─────────────────────────────────────────────
members = [
    ("member1", "&#x1F469;&#x200D;&#x1F4BB;", "Affan Zama",
     "ML / Data Science",
     "Model development, feature engineering, SMOTE &amp; calibration"),
    ("member2", "&#x1F468;&#x200D;&#x1F4BB;", "Karthik Hegde",
     "Backend / Data Pipeline",
     "Model serving, API integration, preprocessing pipeline"),
    ("member3", "&#x1F469;&#x200D;&#x1F3A8;", "Manish Kumar",
     "Frontend / UX",
     "Dashboard design, Streamlit UI, design system"),
    ("member4", "&#x1F4DD;", "Jagadish",
     "Research / Documentation",
     "Literature review, clinical validation, reporting"),
]

tm_cols = st.columns(4)
for col, (photo_stem, fallback_icon, name, role, resp) in zip(tm_cols, members):
    uri, _ = _photo_b64(photo_stem, [_team_dir])
    circle = _circle_photo_html(uri, fallback_icon)
    with col:
        st.markdown(
            '<div style="background:#fff;border-radius:12px;padding:1.2rem 1rem;text-align:center;'
            'box-shadow:0 2px 10px rgba(14,116,144,0.09);border-top:4px solid ' + NAVY + ';">'
            + circle +
            '<div style="font-size:0.85rem;font-weight:700;color:' + NAVY + ';">' + name + '</div>'
            '<div style="font-size:0.75rem;color:' + TEAL + ';font-weight:600;margin-top:3px;">' + role + '</div>'
            '<div style="font-size:0.69rem;color:' + SLATE_400 + ';margin-top:6px;line-height:1.55;">' + resp + '</div>'
            '</div>',
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

render_footer()


# CSS for circular photos
st.markdown("""
<style>
.team-photo-wrap {
    width: 100px; height: 100px; border-radius: 50%;
    overflow: hidden; margin: 0 auto 0.75rem;
    border: 3px solid #0E7490;
    box-shadow: 0 4px 14px rgba(14,116,144,0.22);
}
.team-photo-wrap img { width:100%; height:100%; object-fit:cover; }
.team-photo-placeholder {
    width: 100px; height: 100px; border-radius: 50%;
    background: linear-gradient(135deg,#E0F7FA,#BAE6FD);
    border: 3px solid #0E7490;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 0.75rem;
    font-size: 2rem;
    box-shadow: 0 4px 14px rgba(14,116,144,0.22);
}
.mentor-photo-wrap {
    width: 120px; height: 120px; border-radius: 50%;
    overflow: hidden; margin: 0 auto 0.85rem;
    border: 4px solid #0E7490;
    box-shadow: 0 6px 20px rgba(14,116,144,0.28);
}
.mentor-photo-wrap img { width:100%; height:100%; object-fit:cover; }
.mentor-photo-placeholder {
    width: 120px; height: 120px; border-radius: 50%;
    background: linear-gradient(135deg,#0D2137,#0E7490);
    border: 4px solid #0E7490;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 0.85rem;
    font-size: 2.6rem;
    box-shadow: 0 6px 20px rgba(14,116,144,0.28);
}
</style>
""", unsafe_allow_html=True)

# ── Mentor card (centered) ────────────────────────────────────────────────────
_, mentor_col, _ = st.columns([1.5, 2, 1.5])
with mentor_col:
    mentor_photo = os.path.join(_root, "public", "team", "mentor.jpg")
    st.markdown(
        '<div style="background:#fff;border-radius:16px;padding:1.8rem 1.5rem;text-align:center;'
        'box-shadow:0 4px 20px rgba(14,116,144,0.12);border-top:5px solid ' + TEAL + ';margin-bottom:1.5rem;">',
        unsafe_allow_html=True
    )
    if os.path.exists(mentor_photo):
        # Show actual photo in circular frame
        st.markdown('<div class="mentor-photo-wrap">', unsafe_allow_html=True)
        st.image(mentor_photo, use_container_width=False, width=120)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Placeholder circle with icon — replace by adding mentor.jpg to public/team/
        st.markdown('<div class="mentor-photo-placeholder">&#x1F468;&#x200D;&#x1F3EB;</div>',
                    unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:1rem;font-weight:700;color:' + NAVY + ';">Dr. Dinesh L</div>'
        '<div style="font-size:0.8rem;color:' + TEAL + ';font-weight:600;margin-top:3px;">Project Mentor / Guide</div>'
        '<div style="font-size:0.73rem;color:' + SLATE_400 + ';margin-top:5px;line-height:1.6;">'
        'Department of Computer Science and Engineering &nbsp;&middot;&nbsp; Mangalore Institute of Technology and Engineering'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

# ── Team member cards (4 columns) ─────────────────────────────────────────────
# To add a photo: place member1.jpg, member2.jpg, member3.jpg, member4.jpg
# inside the  public/team/  folder. The circular photo appears automatically.
tm_cols = st.columns(4)
members = [
    ("member1.jpg", "&#x1F469;&#x200D;&#x1F4BB;", "Affan Zama",
     "ML / Data Science",
     "Model development, feature engineering, SMOTE & calibration"),
    ("member2.jpg", "&#x1F468;&#x200D;&#x1F4BB;", "Karthik Hegde",
     "Backend / Data Pipeline",
     "Model serving, API integration, preprocessing pipeline"),
    ("member3.jpg", "&#x1F469;&#x200D;&#x1F3A8;", "Manish Kumar",
     "Frontend / UX",
     "Dashboard design, Streamlit UI, design system"),
    ("member4.jpg", "&#x1F4DD;",                   "Jagadish",
     "Research / Documentation",
     "Literature review, clinical validation, reporting"),
]

for col, (photo_file, fallback_icon, name, role, resp) in zip(tm_cols, members):
    with col:
        photo_path = os.path.join(_root, "public", "team", photo_file)
        st.markdown(
            '<div style="background:#fff;border-radius:12px;padding:1.15rem 1rem;text-align:center;'
            'box-shadow:0 2px 10px rgba(14,116,144,0.09);border-top:4px solid ' + NAVY + ';">',
            unsafe_allow_html=True
        )
        if os.path.exists(photo_path):
            st.markdown('<div class="team-photo-wrap">', unsafe_allow_html=True)
            st.image(photo_path, use_container_width=False, width=100)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="team-photo-placeholder">' + fallback_icon + '</div>',
                unsafe_allow_html=True
            )
        st.markdown(
            '<div style="font-size:0.85rem;font-weight:700;color:' + NAVY + ';">' + name + '</div>'
            '<div style="font-size:0.75rem;color:' + TEAL + ';font-weight:600;margin-top:3px;">' + role + '</div>'
            '<div style="font-size:0.69rem;color:' + SLATE_400 + ';margin-top:5px;line-height:1.55;">' + resp + '</div>'
            '</div>',
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

render_footer()
