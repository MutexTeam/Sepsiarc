"""
utils/ui_components.py  --  SepsiArc global shared UI components
Light clinical theme · Navbar · Sidebar · Footer · Helpers
"""
import streamlit as st
import json, os, sys, joblib

# ── Design tokens ─────────────────────────────────────────────────────────────
NAVY       = "#0D2137"
TEAL       = "#0E7490"
TEAL_LIGHT = "#E0F7FA"
GREEN      = "#166534"
GREEN_BG   = "#DCFCE7"
AMBER      = "#B45309"
AMBER_BG   = "#FEF3C7"
RED        = "#991B1B"
RED_BG     = "#FEE2E2"
SLATE_50   = "#F8FAFC"
SLATE_100  = "#F0F9FF"   # primary page background — light clinical blue
SLATE_200  = "#E2E8F0"
SLATE_400  = "#94A3B8"
SLATE_700  = "#334155"
WHITE      = "#FFFFFF"


# ── CSS injection (global styles – light clinical theme) ──────────────────────
def inject_global_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Light blue-white clinical background */
.main { background: #F0F9FF !important; }
section.main > div:first-child { padding-top: 0 !important; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* Sticky navbar wrapper */
.sa-navbar-wrapper {
    position: sticky;
    top: 0;
    z-index: 9999;
    width: 100%;
}

/* Announcement bar */
.sa-announce {
    background: #0D2137;
    padding: 0.38rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.7rem;
    color: #7DD3FC;
    border-bottom: 1px solid #1e3a5f;
}
.sa-announce-center { flex: 1; text-align: center; font-weight: 500; }
.sa-announce-right  { font-size: 0.66rem; color: #94A3B8; white-space: nowrap; }

/* Main navbar */
.sa-navbar {
    background: #fff;
    border-bottom: 2px solid #BAE6FD;
    padding: 0 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 68px;
    box-shadow: 0 2px 10px rgba(14,116,144,0.08);
}
.sa-brand-chip {
    background: #0E7490;
    color: #fff;
    border-radius: 8px;
    padding: 0.28rem 0.65rem;
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 1.5px;
}
.sa-brand-name { font-size: 1.15rem; font-weight: 800; color: #0D2137; }
.sa-brand-sub  { font-size: 0.6rem; color: #64748B; font-style: italic; margin-top: 1px; }
.sa-nav-links  { display: flex; align-items: center; gap: 0.15rem; }
.sa-nav-link {
    color: #334155;
    font-size: 0.83rem;
    font-weight: 500;
    text-decoration: none;
    padding: 0.42rem 0.8rem;
    border-radius: 6px;
    transition: background 0.15s, color 0.15s;
    white-space: nowrap;
}
.sa-nav-link:hover { background: #E0F7FA; color: #0E7490; }
.sa-nav-link.active {
    color: #0E7490; font-weight: 700;
    background: #E0F7FA;
    border-bottom: 2px solid #0E7490;
}
.sa-nav-cta {
    background: linear-gradient(135deg, #0E7490 0%, #0D2137 100%);
    color: #fff !important;
    padding: 0.48rem 1.25rem;
    border-radius: 8px;
    font-size: 0.83rem;
    font-weight: 600;
    text-decoration: none;
    transition: opacity 0.15s;
    box-shadow: 0 2px 8px rgba(14,116,144,0.3);
}
.sa-nav-cta:hover { opacity: 0.88; }

/* Sidebar */
section[data-testid="stSidebar"] > div:first-child {
    background: #0D2137;
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] a:hover { color: #7DD3FC !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0E7490, #0D2137) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    padding: 0.55rem 1.3rem !important; font-weight: 600 !important;
    font-size: 0.86rem !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #E0F7FA; border-radius: 10px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; font-weight: 500 !important;
    padding: 0.42rem 0.9rem !important; font-size: 0.84rem !important;
}
.stTabs [aria-selected="true"] {
    background: #fff !important; color: #0E7490 !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #fff;
    border: 1px solid #BAE6FD;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}

/* Page sub-header strip */
.sa-page-header {
    background: #fff;
    border-bottom: 1px solid #BAE6FD;
    padding: 0.8rem 2rem;
    margin: 0 -1rem 1.2rem -1rem;
}

/* Risk badges */
.sa-badge { display: inline-block; border-radius: 999px; padding: 0.22rem 0.9rem; font-weight: 700; font-size: 0.85rem; }
.sa-badge-pos      { background: #FEE2E2; color: #991B1B; border: 2px solid #991B1B; }
.sa-badge-neg      { background: #DCFCE7; color: #166534; border: 2px solid #166534; }
.sa-badge-high     { background: #FEE2E2; color: #991B1B; border: 2px solid #991B1B; }
.sa-badge-low      { background: #DCFCE7; color: #166534; border: 2px solid #166534; }
.sa-badge-border   { background: #FEF3C7; color: #B45309; border: 2px solid #B45309; }
.sa-badge-borderline { background: #FEF3C7; color: #B45309; border: 2px solid #B45309; }

/* Footer */
.sa-footer {
    background: linear-gradient(180deg, #0D2137 0%, #061220 100%);
    color: #94A3B8;
    margin: 2.5rem -1rem -1rem -1rem;
    padding: 2.5rem 3rem 0;
    border-top: 3px solid #0E7490;
}
.sa-footer-grid {
    display: grid; grid-template-columns: 2fr 1fr 1fr 1fr;
    gap: 2rem; margin-bottom: 2rem;
}
.sa-footer-col h4 {
    color: #E2E8F0; font-size: 0.78rem; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.8rem;
}
.sa-footer-col a {
    display: block; color: #64748B; font-size: 0.78rem;
    text-decoration: none; margin-bottom: 0.45rem; transition: color 0.15s;
}
.sa-footer-col a:hover { color: #7DD3FC; }
.sa-footer-brand-name { font-size: 1.2rem; font-weight: 800; color: #E2E8F0; margin-bottom: 0.25rem; }
.sa-footer-brand-tag  { font-size: 0.75rem; color: #7DD3FC; font-style: italic; margin-bottom: 0.6rem; }
.sa-footer-brand-desc { font-size: 0.74rem; color: #64748B; line-height: 1.7; }
.sa-footer-disclaimer { border-top: 1px solid #1e3a5f; padding: 1rem 0; margin-top: 0.5rem; }
.sa-footer-bottom {
    border-top: 1px solid #1e3a5f; padding: 0.85rem 0;
    display: flex; justify-content: space-between;
    align-items: center; font-size: 0.7rem; color: #475569;
}

/* Inputs */
label { font-size: 0.81rem !important; font-weight: 500 !important; color: #334155 !important; }
</style>
""", unsafe_allow_html=True)


# ── Navbar ─────────────────────────────────────────────────────────────────────
def render_header(page_title: str = "", page_subtitle: str = "",
                  icon: str = "◎", active: str = "home"):
    """
    Renders announcement bar + sticky navbar.
    active: 'home' | 'stage1' | 'assistant' | 'models' | 'about'
    """
    inject_global_css()

    home_cls   = "sa-nav-link active" if active == "home"      else "sa-nav-link"
    stage1_cls = "sa-nav-link active" if active == "stage1"    else "sa-nav-link"
    asst_cls   = "sa-nav-link active" if active == "assistant" else "sa-nav-link"
    models_cls = "sa-nav-link active" if active == "models"    else "sa-nav-link"
    about_cls  = "sa-nav-link active" if active == "about"     else "sa-nav-link"

    nav_html = (
        '<div class="sa-navbar-wrapper">'
        # Announcement bar
        '<div class="sa-announce">'
        '<span style="font-size:0.63rem;color:#475569;">Research Prototype &middot; PhysioNet 2019</span>'
        '<span class="sa-announce-center">&#128300;&nbsp; Dual-Stage Explainable AI &mdash; Sepsis Detection &amp; Post-Sepsis Risk Assessment</span>'
        '<span class="sa-announce-right">Clinical Decision Support &middot; Research Use Only</span>'
        '</div>'
        # Navbar
        '<nav class="sa-navbar">'
        '<a href="/" style="display:flex;align-items:center;gap:0.5rem;text-decoration:none;">'
        '<div class="sa-brand-chip">ARC</div>'
        '<div>'
        '<div class="sa-brand-name">SepsiArc</div>'
        '<div class="sa-brand-sub">Explainable Sepsis Intelligence</div>'
        '</div>'
        '</a>'
        '<div class="sa-nav-links">'
        '<a href="/" class="' + home_cls + '">Home</a>'
        '<a href="/Stage_1_Detection" class="' + stage1_cls + '">Assessment</a>'
        '<a href="/AI_Assistant" class="' + asst_cls + '">Assistant</a>'
        '<a href="/Model_Information" class="' + models_cls + '">Research</a>'
        '<a href="/About" class="' + about_cls + '">About</a>'
        '</div>'
        '<a href="/Stage_1_Detection" class="sa-nav-cta">&#9654; Start Assessment</a>'
        '</nav>'
        '</div>'
    )
    st.markdown(nav_html, unsafe_allow_html=True)

    # Page sub-header (clinical pages only)
    if page_title:
        sub_html = (
            '<div class="sa-page-header">'
            '<div style="display:flex;align-items:center;gap:0.7rem;">'
            '<span style="font-size:1.35rem;">' + icon + '</span>'
            '<div>'
            '<div style="font-size:1.1rem;font-weight:700;color:' + NAVY + ';">' + page_title + '</div>'
            + ('<div style="font-size:0.76rem;color:' + SLATE_400 + ';margin-top:1px;">' + page_subtitle + '</div>' if page_subtitle else '') +
            '</div>'
            '</div>'
            '</div>'
        )
        st.markdown(sub_html, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar(current_page: str = "home"):
    with st.sidebar:
        st.markdown(
            '<div style="padding:1rem 0.5rem 0.5rem;text-align:center;">'
            '<div style="background:' + TEAL + ';border-radius:10px;padding:0.5rem;'
            'font-size:1.05rem;font-weight:800;color:#fff;letter-spacing:2px;">SepsiArc</div>'
            '<div style="font-size:0.63rem;color:#7DD3FC;font-style:italic;margin-top:4px;">'
            'Explainable sepsis intelligence</div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("<hr style='border-color:#1e3a5f;margin:0.5rem 0;'>", unsafe_allow_html=True)
        st.page_link("app.py",                       label="🏠  Home")
        st.page_link("pages/1_Stage_1_Detection.py", label="🧬  Stage 1 — Detection")

        s2_ok = st.session_state.get("s1_positive", False) or st.session_state.get("s1_borderline", False)
        s2_label = "🔬  Stage 2 — Risk Assessment" if s2_ok else "🔒  Stage 2 (run Stage 1 first)"
        st.page_link("pages/2_Stage_2_Risk.py", label=s2_label)
        st.page_link("pages/3_AI_Assistant.py",     label="💬  Clinical Assistant")
        st.page_link("pages/4_Model_Information.py",label="📊  Research & Models")
        st.page_link("pages/5_About.py",            label="ℹ️  About SepsiArc")

        st.markdown("<hr style='border-color:#1e3a5f;margin:0.5rem 0;'>", unsafe_allow_html=True)
        _render_sidebar_metrics()
        st.markdown("<hr style='border-color:#1e3a5f;margin:0.5rem 0;'>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.62rem;color:#64748B;line-height:1.6;padding:0.2rem 0;">'
            '<b style="color:#FCA5A5;">Research prototype only.</b><br>'
            'Not for autonomous clinical diagnosis.<br><br>'
            'PhysioNet &middot; Reyna et al., 2019'
            '</div>',
            unsafe_allow_html=True
        )


def _render_sidebar_metrics():
    try:
        mdir = "models"
        s1r  = json.load(open(os.path.join(mdir, "model_results.json")))
        s2r  = json.load(open(os.path.join(mdir, "stage2_results.json")))
        best = open(os.path.join(mdir, "best_model_name.txt")).read().strip()
        s1   = s1r.get(best, next(iter(s1r.values())))
        st.markdown(
            '<div style="font-size:0.63rem;font-weight:700;letter-spacing:1.5px;color:#64748B;padding:0.2rem 0;">MODEL PERFORMANCE</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div style="background:rgba(255,255,255,0.07);border-radius:8px;padding:0.55rem 0.75rem;margin-bottom:0.35rem;">'
            '<div style="font-size:0.7rem;font-weight:600;color:#E2E8F0;">Stage 1</div>'
            '<div style="font-size:0.64rem;color:#94A3B8;margin-top:2px;">' + best + '<br>AUROC ' + f'{s1["AUROC"]:.4f}' + '</div>'
            '</div>'
            '<div style="background:rgba(255,255,255,0.07);border-radius:8px;padding:0.55rem 0.75rem;">'
            '<div style="font-size:0.7rem;font-weight:600;color:#E2E8F0;">Stage 2</div>'
            '<div style="font-size:0.64rem;color:#94A3B8;margin-top:2px;">LightGBM<br>AUROC ' + f'{s2r["AUROC"]:.4f}' + '</div>'
            '</div>',
            unsafe_allow_html=True
        )
    except Exception:
        pass


# ── Disclaimer banner ──────────────────────────────────────────────────────────
def render_disclaimer_banner(stage2: bool = False):
    msg = (
        "**SepsiArc** is a clinical decision-support and research application. "
        "Model outputs are **not a diagnosis, prognosis, or treatment recommendation** "
        "and must be interpreted by an appropriately qualified healthcare professional."
    )
    if stage2:
        msg += " · **Post-sepsis risk** is a project-defined proxy classification."
    st.info(msg, icon="⚕")


# ── Footer ─────────────────────────────────────────────────────────────────────
def render_footer(extra_disclaimer: str = ""):
    extra_html = (
        "<br><span style='color:#FCA5A5;'>" + extra_disclaimer + "</span>"
        if extra_disclaimer else ""
    )
    html = (
        '<div class="sa-footer">'
        '<div class="sa-footer-grid">'
        # Brand
        '<div class="sa-footer-col">'
        '<div style="display:flex;align-items:center;gap:0.45rem;margin-bottom:0.6rem;">'
        '<div style="background:#0E7490;border-radius:7px;padding:0.25rem 0.6rem;font-size:0.8rem;font-weight:800;color:#fff;letter-spacing:1.5px;">ARC</div>'
        '<div class="sa-footer-brand-name">SepsiArc</div>'
        '</div>'
        '<div class="sa-footer-brand-tag">From Sepsis Detection to Recovery Risk.</div>'
        '<div class="sa-footer-brand-desc">An explainable dual-stage ML framework for sepsis detection and post-sepsis risk assessment. Built on PhysioNet/CinC Challenge 2019.</div>'
        '</div>'
        # Assessment
        '<div class="sa-footer-col">'
        '<h4>Assessment</h4>'
        '<a href="/Stage_1_Detection">Stage 1 &mdash; Detection</a>'
        '<a href="/Stage_2_Risk">Stage 2 &mdash; Risk</a>'
        '<a href="/AI_Assistant">Clinical Assistant</a>'
        '</div>'
        # Research
        '<div class="sa-footer-col">'
        '<h4>Research</h4>'
        '<a href="/Model_Information">Model Information</a>'
        '<a href="/Model_Information">Explainability (SHAP)</a>'
        '<a href="/About">About the Project</a>'
        '</div>'
        # Info
        '<div class="sa-footer-col">'
        '<h4>Information</h4>'
        '<a href="/About">About SepsiArc</a>'
        '<a href="/About">Clinical Disclaimer</a>'
        '<a href="/">Home</a>'
        '</div>'
        '</div>'
        # Disclaimer
        '<div class="sa-footer-disclaimer">'
        '<div style="font-size:0.71rem;color:#475569;line-height:1.7;text-align:center;max-width:780px;margin:0 auto;">'
        '<b style="color:#FCA5A5;">Clinical Disclaimer:</b> '
        'SepsiArc is a clinical decision-support and research application. '
        'Model outputs are <b style="color:#E2E8F0;">not a diagnosis, prognosis, or treatment recommendation</b> '
        'and must be interpreted by an appropriately qualified healthcare professional.'
        + extra_html +
        '</div>'
        '</div>'
        # Bottom
        '<div class="sa-footer-bottom">'
        '<span>&copy; 2026 SepsiArc &middot; All Rights Reserved</span>'
        '<span>Research &amp; Clinical Decision Support Use Only</span>'
        '<span>PhysioNet Dataset &middot; Reyna et al., 2019</span>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ── Risk badge ─────────────────────────────────────────────────────────────────
def risk_badge_html(label: str, style: str = "pos") -> str:
    cls_map = {
        "positive": "sa-badge-pos", "negative": "sa-badge-neg",
        "high": "sa-badge-high", "low": "sa-badge-low",
        "border": "sa-badge-border", "borderline": "sa-badge-borderline",
    }
    cls = cls_map.get(style, "sa-badge-border")
    return '<span class="sa-badge ' + cls + '">' + label + '</span>'


# ── Stat card ──────────────────────────────────────────────────────────────────
def stat_card(container, title: str, value: str, sub: str = "", color: str = "#0E7490"):
    container.markdown(
        '<div style="background:#fff;border-radius:12px;padding:1.05rem 1.2rem;'
        'box-shadow:0 2px 10px rgba(14,116,144,0.1);border-top:4px solid ' + color + ';text-align:center;">'
        '<div style="font-size:1.9rem;font-weight:800;color:' + color + ';line-height:1.1;">' + value + '</div>'
        '<div style="font-size:0.78rem;font-weight:600;color:#334155;margin-top:0.25rem;">' + title + '</div>'
        + ('<div style="font-size:0.68rem;color:#94A3B8;margin-top:0.1rem;">' + sub + '</div>' if sub else '') +
        '</div>',
        unsafe_allow_html=True
    )


# ── Section title ──────────────────────────────────────────────────────────────
def section_title(text: str, subtitle: str = ""):
    st.markdown("### " + text)
    if subtitle:
        st.caption(subtitle)
