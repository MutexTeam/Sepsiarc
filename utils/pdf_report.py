"""
utils/pdf_report.py
SepsiArc -- Professional PDF Report Generator
Uses fpdf2 (pure Python, no system dependencies, works on Streamlit Cloud).
"""
from __future__ import annotations
import datetime
from fpdf import FPDF

# Brand colours (R, G, B)
NAVY  = (13,  33,  55)
TEAL  = (14, 116, 144)
RED   = (185, 28,  28)
GREEN = (21, 128,  61)
AMBER = (180, 83,   9)
GREY  = (100, 116, 139)
LIGHT = (248, 250, 252)
WHITE = (255, 255, 255)
BLACK = (30,  30,  30)


def _safe(text: str) -> str:
    """Replace Unicode characters unsupported by Helvetica (Latin-1 only) with ASCII equivalents."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "\u2191": "^",          # ↑  Increasing
        "\u2193": "v",          # ↓  Decreasing
        "\u2192": "->",         # →  Stable
        "\u2014": "--",         # —  em dash
        "\u2013": "-",          # –  en dash
        "\u2019": "'",          # '  right single quote
        "\u2018": "'",          # '  left single quote
        "\u201c": '"',          # "  left double quote
        "\u201d": '"',          # "  right double quote
        "\u2022": "*",          # •  bullet
        "\u00b7": ".",          # ·  middle dot
        "\u00d7": "x",          # ×  multiplication sign
        "\u00b1": "+/-",        # ±  plus-minus
        "\u2265": ">=",         # ≥
        "\u2264": "<=",         # ≤
        "\u2260": "!=",         # ≠
        "\u00e9": "e",          # é
        "\u00e0": "a",          # à
        "\u00fc": "u",          # ü
        "\u00e4": "a",          # ä
        "\u00f6": "o",          # ö
        "\u00df": "ss",         # ß
        "\u2026": "...",        # …  ellipsis
    }
    for uni, asc in replacements.items():
        text = text.replace(uni, asc)
    # Final fallback: drop any remaining non-Latin-1 chars
    return text.encode("latin-1", errors="replace").decode("latin-1")


class SepsiArcPDF(FPDF):
    """Custom FPDF class with SepsiArc branding header and page-number footer."""

    def __init__(self, report_title: str, patient_id: str, generated: str):
        super().__init__("P", "mm", "A4")
        self.report_title = _safe(report_title)
        self.patient_id   = _safe(patient_id)
        self.generated    = _safe(generated)
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(18, 18, 18)

    # ── Auto-sanitize every cell/multi_cell write ────────────────────────
    def cell(self, w=0, h=0, text="", border=0, new_x="RIGHT", new_y="TOP",
             align="L", fill=False, link="", **kwargs):
        return super().cell(w=w, h=h, text=_safe(str(text)), border=border,
                            new_x=new_x, new_y=new_y, align=align,
                            fill=fill, link=link, **kwargs)

    def multi_cell(self, w, h=0, text="", border=0, align="J", fill=False,
                   split_only=False, link="", ln=0, max_line_height=None,
                   markdown=False, output=None, dry_run=False, **kwargs):
        return super().multi_cell(w=w, h=h, text=_safe(str(text)), border=border,
                                  align=align, fill=fill, link=link,
                                  markdown=markdown, **kwargs)

    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 14, style="F")
        self.set_fill_color(*TEAL)
        self.rect(8, 3, 12, 8, style="F")   # ARC badge (square corners, compatible with all fpdf2 versions)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*WHITE)
        self.set_xy(8, 4.5)
        self.cell(12, 5, "ARC", align="C")
        self.set_font("Helvetica", "B", 11)
        self.set_xy(22, 3)
        self.cell(80, 8, "SepsiArc", align="L")
        self.set_font("Helvetica", "", 7)
        self.set_xy(100, 5)
        self.cell(102, 4, "From Sepsis Detection to Recovery Risk.", align="R")
        self.ln(2)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(*TEAL)
        self.set_line_width(0.4)
        self.line(18, self.get_y(), 192, self.get_y())
        self.ln(1)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*GREY)
        page_txt = (
            f"SepsiArc Research Tool  |  NOT a clinical diagnosis  |"
            f"  Page {self.page_no()}"
        )
        self.cell(0, 5, page_txt, align="C")

    def title_block(self):
        self.ln(6)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*NAVY)
        self.cell(0, 10, self.report_title, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_fill_color(*LIGHT)
        self.set_draw_color(*TEAL)
        self.set_line_width(0.3)
        self.rect(18, self.get_y(), 174, 10, style="DF")
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*NAVY)
        self.set_xy(20, self.get_y() + 1.5)
        self.cell(86, 7, f"Patient / File ID :  {self.patient_id}", align="L")
        self.cell(86, 7, f"Report Generated :  {self.generated}", align="R")
        self.ln(13)

    def section_heading(self, title: str, color: tuple = NAVY):
        self.set_fill_color(*color)
        self.rect(18, self.get_y(), 3, 7, style="F")
        self.set_xy(23, self.get_y())
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(*color)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def key_value_row(self, label: str, value: str,
                      fill: bool = False, val_color: tuple = BLACK):
        y = self.get_y()
        if fill:
            self.set_fill_color(*LIGHT)
            self.rect(18, y, 174, 6.5, style="F")
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*GREY)
        self.set_xy(20, y + 0.5)
        self.cell(70, 5.5, _safe(label).upper(), align="L")
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*val_color)
        self.cell(102, 5.5, _safe(str(value)), align="L")
        self.ln(6.5)

    def result_badge(self, label: str, probability: str,
                     threshold: str, color: tuple = RED):
        y = self.get_y()
        self.set_draw_color(*color)
        self.set_fill_color(*WHITE)
        self.set_line_width(0.8)
        self.rect(18, y, 174, 28, style="DF")
        self.set_fill_color(*color)
        self.rect(18, y, 5, 28, style="F")
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*color)
        self.set_xy(28, y + 4)
        self.cell(120, 10, _safe(label), align="L")
        self.set_font("Helvetica", "B", 13)
        self.set_xy(28, y + 14)
        self.cell(80, 8, _safe(f"Probability: {probability}"), align="L")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*GREY)
        self.set_xy(28, y + 21)
        self.cell(80, 5, _safe(f"Decision Threshold: {threshold}"), align="L")
        self.ln(32)

    def table_header(self, cols: list, color: tuple = TEAL):
        self.set_fill_color(*color)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 8)
        for label, w in cols:
            self.cell(w, 7, _safe(label), border=0, fill=True, align="C")
        self.ln()

    def table_row(self, values: list, widths: list,
                  fill: bool = False, aligns: list = None):
        if fill:
            self.set_fill_color(*LIGHT)
        else:
            self.set_fill_color(*WHITE)
        self.set_text_color(*BLACK)
        self.set_font("Helvetica", "", 8)
        if aligns is None:
            aligns = ["L"] * len(values)
        for val, w, al in zip(values, widths, aligns):
            self.cell(w, 6.5, _safe(str(val)), border=0, fill=True, align=al)
        self.ln()

    def divider(self):
        self.set_draw_color(*TEAL)
        self.set_line_width(0.3)
        self.line(18, self.get_y(), 192, self.get_y())
        self.ln(4)

    def disclaimer_box(self):
        self.ln(2)
        y = self.get_y()
        self.set_fill_color(255, 251, 235)
        self.set_draw_color(202, 138, 4)
        self.set_line_width(0.4)
        self.rect(18, y, 174, 22, style="DF")
        self.set_xy(21, y + 2)
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(146, 64, 14)
        self.cell(0, 5, "DISCLAIMER", new_x="LMARGIN", new_y="NEXT")
        self.set_xy(21, y + 8)
        self.set_font("Helvetica", "", 7.8)
        msg = (
            "This report contains machine-learning-based predictions for research and "
            "decision-support purposes only. It is NOT a clinical diagnosis and should not "
            "be interpreted as a validated prediction of mortality, survival, or long-term "
            "patient outcome. Results should not replace professional medical judgment. "
            "SepsiArc was trained on the PhysioNet/CinC Challenge 2019 dataset."
        )
        self.multi_cell(168, 4.5, msg)
        self.ln(2)


# ── Public API ────────────────────────────────────────────────────────────────

def build_stage1_pdf(
    patient_id: str, age: str, gender: str,
    n_rows: int, obs_start: int, obs_end: int, hosp_adm: str,
    peak_prob: float, s1_thresh: float, n_alerts: int,
    peak_hour: int, first_alert_h,
    res_label: str, trend_rows: list,
    s2_eligible: bool, s2_pre_hrs: int, s2_post_hrs: int,
) -> bytes:
    generated  = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    prob_color = RED if peak_prob >= s1_thresh else (AMBER if peak_prob >= 0.08 else GREEN)

    pdf = SepsiArcPDF("Stage 1 - Sepsis Detection Report", patient_id, generated)
    pdf.add_page()
    pdf.title_block()

    # 1. Patient Info
    pdf.section_heading("1.  Patient Information", TEAL)
    rows1 = [
        ("Patient / File ID",      patient_id),
        ("Age",                    age),
        ("Gender",                 gender),
        ("Total ICU Observations", f"{n_rows} hours"),
        ("Observation Period",     f"Hour {obs_start} to Hour {obs_end}"),
        ("Hospital Admission",     hosp_adm),
    ]
    for i, (k, v) in enumerate(rows1):
        pdf.key_value_row(k, v, fill=(i % 2 == 0))
    pdf.ln(3)

    # 2. Stage 1 Result
    pdf.section_heading("2.  Stage 1 Prediction Result", NAVY)
    pdf.result_badge(
        label=res_label,
        probability=f"{peak_prob*100:.2f}%",
        threshold=f"{s1_thresh:.3f} ({s1_thresh*100:.1f}%)",
        color=prob_color,
    )
    rows2 = [
        ("Model",                   "Regularized Cost-Sensitive XGBoost (Model A)"),
        ("Features Used",           "135 engineered clinical features"),
        ("Decision Threshold",      f"{s1_thresh:.3f} ({s1_thresh*100:.1f}%)"),
        ("Peak Sepsis Probability", f"{peak_prob*100:.2f}%"),
        ("Peak Probability Hour",   f"Hour {peak_hour}"),
        ("First Alert Hour",        f"Hour {first_alert_h}" if first_alert_h else "None"),
        ("Alert Hours",             f"{n_alerts} / {n_rows}"),
        ("Final Prediction",        res_label),
    ]
    for i, (k, v) in enumerate(rows2):
        vc = prob_color if k == "Final Prediction" else BLACK
        pdf.key_value_row(k, v, fill=(i % 2 == 0), val_color=vc)
    pdf.ln(3)

    # 3. Physiological Trend
    if trend_rows:
        pdf.section_heading("3.  Physiological Trend Summary", TEAL)
        cols = [
            ("Parameter",   50), ("Recent Value", 30),
            ("Mean",        22), ("Min",           20),
            ("Max",         20), ("Trend",         32),
        ]
        widths = [c[1] for c in cols]
        pdf.table_header(cols, color=TEAL)
        for i, row in enumerate(trend_rows):
            pdf.table_row(
                [row["Parameter"], row["Current / Recent Value"],
                 row["Mean over Stay"], row["Min"], row["Max"], row["Trend"]],
                widths, fill=(i % 2 == 0),
            )
        pdf.ln(4)

    # 4. Stage 2 Eligibility
    pdf.section_heading("4.  Stage 2 Eligibility",
                        GREEN if s2_eligible else AMBER)
    elig = [
        ("Eligible for Stage 2",   "YES - Risk Stratification Available" if s2_eligible
                                    else "NO - Below alert and borderline thresholds"),
        ("Stage 1 Reference Hour", f"Hour {first_alert_h}" if first_alert_h else "N/A"),
        ("Pre-alert History",      f"{s2_pre_hrs} hour(s)"),
        ("Post-alert Observations",f"{s2_post_hrs} hour(s)"),
    ]
    for i, (k, v) in enumerate(elig):
        vc = GREEN if (k == "Eligible for Stage 2" and s2_eligible) else BLACK
        pdf.key_value_row(k, v, fill=(i % 2 == 0), val_color=vc)
    pdf.ln(3)

    # 5. Clinical Interpretation
    pdf.section_heading("5.  Clinical Interpretation", NAVY)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*BLACK)
    interp = (
        f"Stage 1 operates at the hourly ICU observation level. Each ICU hour receives "
        f"an independent sepsis probability. The patient-level decision is based on the "
        f"peak (maximum) probability across all {n_rows} observed hours. The model uses "
        f"a deliberately lowered threshold ({s1_thresh:.3f}) compared to the standard 0.5 "
        f"to maximise recall -- detecting more sepsis cases at the cost of a higher "
        f"false-positive rate. This is intentional for clinical screening purposes."
    )
    pdf.multi_cell(174, 5, interp)
    pdf.ln(3)
    pdf.disclaimer_box()
    return bytes(pdf.output())


def build_stage2_pdf(
    patient_id: str, age: str, gender: str,
    n_total_rows: int, s1_result_label: str, s1_prob,
    ref_hour: int, n_pre: int, n_early: int,
    prob_higher_risk: float, s2_thresh: float,
    risk_label: str, data_sufficient: bool, trend_data: list,
) -> bytes:
    generated  = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    risk_color = RED if risk_label == "HIGHER RISK" else GREEN

    pdf = SepsiArcPDF(
        "Stage 2 - Post-Sepsis Risk Stratification Report",
        patient_id, generated,
    )
    pdf.add_page()
    pdf.title_block()

    # 1. Stage 1 Context
    pdf.section_heading("1.  Stage 1 Context", TEAL)
    ctx = [
        ("Stage 1 Result",           s1_result_label),
        ("Stage 1 Peak Probability",  f"{s1_prob*100:.2f}%" if s1_prob else "N/A"),
        ("Stage 1 Reference Hour",    f"Hour {ref_hour}"),
    ]
    for i, (k, v) in enumerate(ctx):
        pdf.key_value_row(k, v, fill=(i % 2 == 0))
    pdf.ln(3)

    # 2. Patient Info
    pdf.section_heading("2.  Patient Information", TEAL)
    pt = [
        ("Patient / File ID",       patient_id),
        ("Age",                     age),
        ("Gender",                  gender),
        ("Total ICU Observations",  f"{n_total_rows} hours"),
        ("Pre-label History",       f"{n_pre} hour(s)"),
        ("Early Post-label Obs.",   f"{n_early} hour(s)"),
        ("Stage 2 Input Status",    "Sufficient" if data_sufficient else "Insufficient"),
    ]
    for i, (k, v) in enumerate(pt):
        pdf.key_value_row(k, v, fill=(i % 2 == 0))
    pdf.ln(3)

    # 3. Stage 2 Result
    pdf.section_heading("3.  Stage 2 Risk Stratification Result", NAVY)
    pdf.result_badge(
        label=risk_label,
        probability=f"{prob_higher_risk*100:.2f}%",
        threshold=f"{s2_thresh:.2f} ({s2_thresh*100:.2f}%)",
        color=risk_color,
    )
    pred = [
        ("Model",                   "XGBoost (patient-level, max_depth=2)"),
        ("Features Used",           "65 patient-level aggregate features"),
        ("Decision Threshold",      f"{s2_thresh:.2f} ({s2_thresh*100:.2f}%)"),
        ("Higher Risk Probability", f"{prob_higher_risk*100:.2f}%"),
        ("Final Classification",    risk_label),
    ]
    for i, (k, v) in enumerate(pred):
        vc = risk_color if k == "Final Classification" else BLACK
        pdf.key_value_row(k, v, fill=(i % 2 == 0), val_color=vc)
    pdf.ln(3)

    # 4. Physiological Trend
    if trend_data:
        pdf.section_heading("4.  Physiological Trend Summary", TEAL)
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 5,
                 "Pre = hours before Stage 1 reference  |  Post = hours from reference onward",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        cols = [
            ("Parameter", 52), ("Pre Mean", 28), ("Post Mean", 30),
            ("Change",    25), ("Trend",    39),
        ]
        widths = [c[1] for c in cols]
        pdf.table_header(cols, color=TEAL)
        for i, row in enumerate(trend_data):
            try:
                pm = float(row["Pre-label Mean"]) if row["Pre-label Mean"] not in ("-","--","") else None
                em = float(row["Early Post-label Mean"]) if row["Early Post-label Mean"] not in ("-","--","") else None
                change = f"{(em - pm):+.2f}" if (pm is not None and em is not None) else "-"
            except Exception:
                change = "-"
            pdf.table_row(
                [row["Parameter"], row["Pre-label Mean"],
                 row["Early Post-label Mean"], change, row["Trend"]],
                widths, fill=(i % 2 == 0),
            )
        pdf.ln(4)

    # 5. Interpretation
    pdf.section_heading("5.  Model-Based Interpretation", NAVY)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*BLACK)
    interp = (
        f"Stage 2 operates at the patient level. It uses aggregate features computed "
        f"from physiological data in two temporal windows relative to the Stage 1 "
        f"reference hour (Hour {ref_hour}): the pre-label historical window ({n_pre} hour(s)) "
        f"and the early post-label window ({n_early} hour(s)). The model classifies the "
        f"patient as {risk_label}. 'Higher Risk' refers to the model's risk_50pct proxy "
        f"label -- it is not a validated prediction of mortality, survival, or long-term "
        f"clinical outcome."
    )
    pdf.multi_cell(174, 5, interp)
    pdf.ln(3)
    pdf.disclaimer_box()
    return bytes(pdf.output())


def build_combined_pdf(
    # Stage 1 fields
    patient_id: str, age: str, gender: str,
    n_rows: int, obs_start: int, obs_end: int,
    s1_result_label: str, s1_prob: float, s1_thresh: float,
    n_alerts: int, peak_hour: int, first_alert_h,
    s1_trend_rows: list,
    # Stage 2 fields
    ref_hour: int, n_pre: int, n_early: int,
    prob_higher_risk: float, s2_thresh: float,
    risk_label: str, s2_trend_data: list,
) -> bytes:
    generated  = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    s1_color   = RED if s1_prob >= s1_thresh else (AMBER if s1_prob >= 0.08 else GREEN)
    risk_color = RED if risk_label == "HIGHER RISK" else GREEN

    pdf = SepsiArcPDF(
        "Complete SepsiArc Patient Analysis Report",
        patient_id, generated,
    )
    pdf.add_page()
    pdf.title_block()

    # 1. Patient Info
    pdf.section_heading("1.  Patient Information", TEAL)
    for i, (k, v) in enumerate([
        ("Patient / File ID",      patient_id),
        ("Age",                    age),
        ("Gender",                 gender),
        ("Total ICU Observations", f"{n_rows} hours"),
        ("Observation Period",     f"Hour {obs_start} to Hour {obs_end}"),
    ]):
        pdf.key_value_row(k, v, fill=(i % 2 == 0))
    pdf.ln(3)

    # 2. Stage 1
    pdf.section_heading("2.  Stage 1 -- Sepsis Detection", TEAL)
    pdf.result_badge(
        label=s1_result_label,
        probability=f"{s1_prob*100:.2f}%",
        threshold=f"{s1_thresh:.3f} ({s1_thresh*100:.1f}%)",
        color=s1_color,
    )
    for i, (k, v) in enumerate([
        ("Model",                  "Regularized Cost-Sensitive XGBoost (Model A)"),
        ("Features",               "135 engineered hourly features"),
        ("Decision Threshold",     f"{s1_thresh:.3f} ({s1_thresh*100:.1f}%)"),
        ("Peak Probability",       f"{s1_prob*100:.2f}%"),
        ("Stage 1 Reference Hour", f"Hour {first_alert_h}" if first_alert_h else "N/A"),
        ("Alert Hours",            f"{n_alerts} / {n_rows}"),
        ("Final Result",           s1_result_label),
    ]):
        vc = s1_color if k == "Final Result" else BLACK
        pdf.key_value_row(k, v, fill=(i % 2 == 0), val_color=vc)
    pdf.ln(3)

    if s1_trend_rows:
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*NAVY)
        pdf.cell(0, 6, "Physiological Trend Summary (Stage 1)",
                 new_x="LMARGIN", new_y="NEXT")
        cols = [
            ("Parameter", 50), ("Recent", 28), ("Mean", 22),
            ("Min", 20), ("Max", 20), ("Trend", 34),
        ]
        widths = [c[1] for c in cols]
        pdf.table_header(cols, color=TEAL)
        for i, row in enumerate(s1_trend_rows):
            pdf.table_row(
                [row["Parameter"], row["Current / Recent Value"],
                 row["Mean over Stay"], row["Min"], row["Max"], row["Trend"]],
                widths, fill=(i % 2 == 0),
            )
        pdf.ln(3)

    # 3. Transition
    pdf.divider()
    pdf.section_heading("3.  Stage 1 to Stage 2 Transition", AMBER)
    for i, (k, v) in enumerate([
        ("Stage 2 Eligible",        "YES"),
        ("Reference Point",         f"Stage 1 Reference Hour = Hour {ref_hour}"),
        ("Pre-label Window",        f"{n_pre} hour(s)"),
        ("Early Post-label Window", f"{n_early} hour(s)"),
    ]):
        pdf.key_value_row(k, v, fill=(i % 2 == 0))
    pdf.ln(3)

    # 4. Stage 2
    pdf.section_heading("4.  Stage 2 -- Post-Sepsis Risk Stratification", RED)
    pdf.result_badge(
        label=risk_label,
        probability=f"{prob_higher_risk*100:.2f}%",
        threshold=f"{s2_thresh:.2f} ({s2_thresh*100:.2f}%)",
        color=risk_color,
    )
    for i, (k, v) in enumerate([
        ("Model",                   "XGBoost (patient-level, max_depth=2)"),
        ("Features",                "65 patient-level aggregate features"),
        ("Decision Threshold",      f"{s2_thresh:.2f} ({s2_thresh*100:.2f}%)"),
        ("Higher Risk Probability", f"{prob_higher_risk*100:.2f}%"),
        ("Final Classification",    risk_label),
    ]):
        vc = risk_color if k == "Final Classification" else BLACK
        pdf.key_value_row(k, v, fill=(i % 2 == 0), val_color=vc)
    pdf.ln(3)

    if s2_trend_data:
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*NAVY)
        pdf.cell(0, 6, "Physiological Trend Summary (Stage 2)",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 5, "Pre = pre-label mean  |  Post = early post-label mean",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        cols = [
            ("Parameter", 52), ("Pre Mean", 28), ("Post Mean", 28),
            ("Change", 24), ("Trend", 42),
        ]
        widths = [c[1] for c in cols]
        pdf.table_header(cols, color=RED)
        for i, row in enumerate(s2_trend_data):
            try:
                pm = float(row["Pre-label Mean"]) if row["Pre-label Mean"] not in ("-","--","") else None
                em = float(row["Early Post-label Mean"]) if row["Early Post-label Mean"] not in ("-","--","") else None
                change = f"{(em - pm):+.2f}" if (pm is not None and em is not None) else "-"
            except Exception:
                change = "-"
            pdf.table_row(
                [row["Parameter"], row["Pre-label Mean"],
                 row["Early Post-label Mean"], change, row["Trend"]],
                widths, fill=(i % 2 == 0),
            )
        pdf.ln(3)

    # 5. Summary
    pdf.divider()
    pdf.section_heading("5.  Overall Model-Based Summary", NAVY)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*BLACK)
    summary = (
        f"Stage 1 identified a {s1_result_label.lower()} prediction at the reference "
        f"hour (Hour {ref_hour}). Stage 2 subsequently classified the patient as "
        f"{risk_label} based on the physiological trajectory available for risk "
        f"stratification. The complete dual-stage analysis is documented in this "
        f"report for research and clinical decision-support reference."
    )
    pdf.multi_cell(174, 5, summary)
    pdf.ln(3)
    pdf.disclaimer_box()
    return bytes(pdf.output())
