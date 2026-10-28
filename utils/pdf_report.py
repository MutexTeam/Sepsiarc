"""
utils/pdf_report.py
SepsiArc -- Professional PDF Report Generator (v2)
Uses fpdf2 (pure Python, no system dependencies, works on Streamlit Cloud).

Sections in Stage 1 PDF:
  1. Patient Information & Data Quality
  2. Stage 1 Prediction Result
  3. Hourly Prediction Table
  4. Probability Timeline Chart
  5. Physiological Trend Summary
  6. SHAP Explanation
  7. LIME Explanation
  8. Stage 2 Eligibility
  9. Clinical Interpretation & Disclaimer

Sections in Stage 2 PDF:
  1. Stage 1 Context & Report Header
  2. Patient / Input Information
  3. Eligibility & Reference Hour
  4. Stage 2 Risk Result
  5. Physiological Trend Summary
  6. SHAP Explanation
  7. LIME Explanation
  8. Model Interpretation & Limitations
  9. Disclaimer
"""
from __future__ import annotations
import io, datetime
from fpdf import FPDF

# ── Brand colours (R, G, B) ──────────────────────────────────────────────────
NAVY  = (13,  33,  55)
TEAL  = (14, 116, 144)
RED   = (185, 28,  28)
GREEN = (21, 128,  61)
AMBER = (180, 83,   9)
GREY  = (100, 116, 139)
LIGHT = (248, 250, 252)
WHITE = (255, 255, 255)
BLACK = (30,  30,  30)
AMBER_BG = (255, 251, 235)
AMBER_BD = (202, 138,   4)


def _safe(text: str) -> str:
    """Replace Unicode/emoji characters unsupported by Helvetica (Latin-1) with safe ASCII."""
    if not isinstance(text, str):
        text = str(text)
    # Strip variation selectors first (fe0f = emoji presentation)
    text = text.replace("\ufe0f", "")
    # Trend arrow emoji (composite sequences first)
    text = text.replace("\u2197", "Increasing")   # ↗
    text = text.replace("\u2198", "Decreasing")   # ↘
    text = text.replace("\u2194", "Stable")        # ↔
    text = text.replace("\u2191", "Increasing")   # ↑
    text = text.replace("\u2193", "Decreasing")   # ↓
    text = text.replace("\u2192", "Stable")        # →  (was "->", now "Stable" for trends)
    # Typography
    text = text.replace("\u2014", "--")            # — em dash
    text = text.replace("\u2013", "-")             # – en dash
    text = text.replace("\u2019", "'")             # ' right single quote
    text = text.replace("\u2018", "'")             # ' left single quote
    text = text.replace("\u201c", '"')             # " left double quote
    text = text.replace("\u201d", '"')             # " right double quote
    text = text.replace("\u2022", "*")             # • bullet
    text = text.replace("\u00b7", ".")             # · middle dot
    text = text.replace("\u00d7", "x")             # × multiplication
    text = text.replace("\u00b1", "+/-")           # ± plus-minus
    text = text.replace("\u2265", ">=")            # ≥
    text = text.replace("\u2264", "<=")            # ≤
    text = text.replace("\u2260", "!=")            # ≠
    text = text.replace("\u00b0", " deg")          # °  degree
    text = text.replace("\u03bc", "u")             # μ  micro
    # Accented Latin
    text = text.replace("\u00e9", "e")
    text = text.replace("\u00e0", "a")
    text = text.replace("\u00fc", "u")
    text = text.replace("\u00e4", "a")
    text = text.replace("\u00f6", "o")
    text = text.replace("\u00df", "ss")
    text = text.replace("\u2026", "...")           # …
    # Final fallback: drop any remaining non-Latin-1 chars
    return text.encode("latin-1", errors="replace").decode("latin-1")


class SepsiArcPDF(FPDF):
    """Custom FPDF subclass with SepsiArc branding, header, footer, and helpers."""

    def __init__(self, report_title: str, patient_id: str, generated: str):
        super().__init__("P", "mm", "A4")
        self.report_title = _safe(report_title)
        self.patient_id   = _safe(patient_id)
        self.generated    = _safe(generated)
        self.set_auto_page_break(auto=True, margin=22)
        self.set_margins(18, 18, 18)

    # ── Auto-sanitize every write ──────────────────────────────────────────
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

    # ── Branding header (auto-called on each page) ─────────────────────────
    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 14, style="F")
        self.set_fill_color(*TEAL)
        self.rect(8, 3, 12, 8, style="F")
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

    # ── Page footer ────────────────────────────────────────────────────────
    def footer(self):
        self.set_y(-14)
        self.set_draw_color(*TEAL)
        self.set_line_width(0.4)
        self.line(18, self.get_y(), 192, self.get_y())
        self.ln(1)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*GREY)
        self.cell(0, 5,
                  f"SepsiArc Research Tool  |  NOT a clinical diagnosis  |  Page {self.page_no()}",
                  align="C")

    # ── Report title block ─────────────────────────────────────────────────
    def title_block(self):
        self.ln(6)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*NAVY)
        self.cell(0, 9, self.report_title, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_fill_color(*LIGHT)
        self.set_draw_color(*TEAL)
        self.set_line_width(0.3)
        self.rect(18, self.get_y(), 174, 10, style="DF")
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*NAVY)
        self.set_xy(20, self.get_y() + 1.5)
        self.cell(86, 7, f"Patient / File ID :  {self.patient_id}", align="L")
        self.cell(86, 7, f"Report Generated  :  {self.generated}", align="R")
        self.ln(13)

    # ── Coloured section heading ───────────────────────────────────────────
    def section_heading(self, title: str, color: tuple = NAVY):
        # Ensure at least 18mm before adding heading (avoid orphan headings)
        if self.get_y() > (self.h - self.b_margin - 18):
            self.add_page()
        self.set_fill_color(*color)
        self.rect(18, self.get_y(), 3, 7, style="F")
        self.set_xy(23, self.get_y())
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(*color)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    # ── Key-value pair row ─────────────────────────────────────────────────
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

    # ── Result badge (coloured border box) ────────────────────────────────
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

    # ── Table header row ───────────────────────────────────────────────────
    def table_header(self, cols: list, color: tuple = TEAL):
        self.set_fill_color(*color)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 8)
        for label, w in cols:
            self.cell(w, 7, _safe(label), border=0, fill=True, align="C")
        self.ln()

    # ── Table data row ─────────────────────────────────────────────────────
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
            self.cell(w, 6.0, _safe(str(val)), border=0, fill=True, align=al)
        self.ln()

    # ── Horizontal divider ─────────────────────────────────────────────────
    def divider(self):
        self.set_draw_color(*TEAL)
        self.set_line_width(0.3)
        self.line(18, self.get_y(), 192, self.get_y())
        self.ln(4)

    # ── Embed PNG image from bytes ─────────────────────────────────────────
    def embed_png(self, png_bytes: bytes | None,
                  max_w: float = 174, caption: str = None):
        """Embed a PNG image from bytes. Safe if png_bytes is None or corrupt."""
        if not png_bytes:
            return
        try:
            img_io = io.BytesIO(png_bytes)
            # Check space — estimate image height at A4 width ~97mm for 16:4 ratio
            est_h = max_w * 0.28
            if self.get_y() + est_h > (self.h - self.b_margin - 5):
                self.add_page()
            self.image(img_io, x=18, w=max_w)
            if caption:
                self.set_font("Helvetica", "I", 7)
                self.set_text_color(*GREY)
                self.multi_cell(174, 4, caption, align="C")
            self.ln(3)
        except Exception as exc:
            self.set_font("Helvetica", "I", 7.5)
            self.set_text_color(*GREY)
            self.cell(0, 5, f"[Image could not be embedded: {str(exc)[:80]}]",
                      new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

    # ── Info / note box ────────────────────────────────────────────────────
    def note_box(self, text: str, color: tuple = TEAL):
        y = self.get_y()
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*color)
        self.set_fill_color(*(int(c * 0.12 + 243) for c in color))
        self.multi_cell(174, 5, text, fill=False)
        self.ln(2)

    # ── Disclaimer box (SINGLE, at report end) ─────────────────────────────
    def disclaimer_box(self):
        """Render disclaimer box. Ensures space; forces new page if needed."""
        self.ln(4)
        if self.get_y() > (self.h - self.b_margin - 40):
            self.add_page()
            self.ln(3)
        y = self.get_y()
        msg = (
            "RESEARCH AND CLINICAL DECISION-SUPPORT USE ONLY. "
            "This report contains machine-learning-based predictions. "
            "It is NOT a clinical diagnosis and must NOT be used as the sole basis for "
            "clinical decisions. Results should not replace professional medical judgment. "
            "Stage 1 provides model-based hourly sepsis risk scores. "
            "Stage 2 provides a patient-level physiological-risk stratification based on an "
            "engineered proxy label (Risk_50pct) -- it does NOT predict mortality, "
            "survival, or validated long-term clinical outcomes. "
            "SepsiArc was developed and evaluated using the PhysioNet/CinC Challenge 2019 dataset."
        )
        self.set_font("Helvetica", "", 7.8)
        lines = self.multi_cell(168, 4.5, msg, split_only=True)
        n_lines = len(lines) if lines else 5
        box_h = 11 + n_lines * 4.5 + 5
        self.set_fill_color(*AMBER_BG)
        self.set_draw_color(*AMBER_BD)
        self.set_line_width(0.4)
        self.rect(18, y, 174, box_h, style="DF")
        self.set_xy(21, y + 2.5)
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(146, 64, 14)
        self.cell(0, 5, "DISCLAIMER", new_x="LMARGIN", new_y="NEXT")
        self.set_xy(21, y + 9)
        self.set_font("Helvetica", "", 7.8)
        self.set_text_color(146, 64, 14)
        self.multi_cell(168, 4.5, msg)
        self.ln(3)


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def build_stage1_pdf(
    patient_id: str, age: str, gender: str,
    n_rows: int, obs_start: int, obs_end: int, hosp_adm: str,
    peak_prob: float, s1_thresh: float, n_alerts: int,
    peak_hour: int, first_alert_h,
    res_label: str, trend_rows: list,
    s2_eligible: bool, s2_pre_hrs: int, s2_post_hrs: int,
    # New optional parameters — all have safe defaults
    hourly_rows: list = None,
    chart_png_bytes: bytes = None,
    shap_png_bytes: bytes = None,
    lime_png_bytes: bytes = None,
    lime_text: str = None,
    borderline_thresh: float = 0.080,
) -> bytes:
    """Build Stage 1 Sepsis Detection PDF report."""
    generated  = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    prob_color = RED if peak_prob >= s1_thresh else (AMBER if peak_prob >= borderline_thresh else GREEN)

    pdf = SepsiArcPDF(
        "SepsiArc -- Stage 1: Sepsis Detection Report",
        patient_id, generated,
    )
    pdf.add_page()
    pdf.title_block()

    # ── Section 1: Patient Information ──────────────────────────────────────
    pdf.section_heading("1.  Patient Information", TEAL)
    rows1 = [
        ("Patient / File ID",      patient_id),
        ("Age",                    age),
        ("Gender",                 gender),
        ("Total ICU Observations", f"{n_rows} hourly rows"),
        ("Observation Period",     f"Hour {obs_start} to Hour {obs_end}"),
        ("Hospital Admission",     hosp_adm),
        ("Input Format",           "PSV / CSV (hourly ICU timeline)"),
        ("Preprocessing",          "Feature engineering: 135 clinical features including rolling stats, "
                                   "lags, missingness indicators, and recency flags applied."),
    ]
    for i, (k, v) in enumerate(rows1):
        pdf.key_value_row(k, v, fill=(i % 2 == 0))
    pdf.ln(3)

    # ── Section 2: Stage 1 Prediction Result ────────────────────────────────
    pdf.section_heading("2.  Stage 1 Prediction Result", NAVY)
    pdf.result_badge(
        label=res_label,
        probability=f"{peak_prob * 100:.2f}%",
        threshold=f"{s1_thresh:.3f} ({s1_thresh * 100:.1f}%)",
        color=prob_color,
    )
    rows2 = [
        ("Model",                    "Regularized Cost-Sensitive XGBoost (Model A)"),
        ("Features Used",            "135 engineered clinical features"),
        ("Decision Threshold",       f"{s1_thresh:.3f}  ({s1_thresh * 100:.1f}%)  -- Sepsis Alert"),
        ("Borderline Threshold",     f"{borderline_thresh:.3f}  ({borderline_thresh * 100:.1f}%)  -- Borderline"),
        ("Peak Sepsis Probability",  f"{peak_prob * 100:.2f}%"),
        ("Peak Probability Hour",    f"Hour {peak_hour}   (hour with maximum predicted probability)"),
        ("First Alert Hour",         f"Hour {first_alert_h}" if first_alert_h
                                     else "None  (no hour crossed the Sepsis Alert threshold)"),
        ("Alert Hours / Total",      f"{n_alerts} / {n_rows}  hours crossed the Sepsis Alert threshold"),
        ("Final Prediction",         res_label),
    ]
    for i, (k, v) in enumerate(rows2):
        vc = prob_color if k == "Final Prediction" else BLACK
        pdf.key_value_row(k, v, fill=(i % 2 == 0), val_color=vc)
    pdf.ln(3)

    # ── Section 3: Hourly Prediction Table ──────────────────────────────────
    pdf.section_heading("3.  Hourly Prediction Table", TEAL)
    if hourly_rows:
        MAX_ROWS = 60
        display_rows = hourly_rows[:MAX_ROWS]
        truncated = len(hourly_rows) > MAX_ROWS
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(*GREY)
        caption_txt = (
            f"Showing {len(display_rows)} of {len(hourly_rows)} processed hours. "
            f"Alert = hour crossed Sepsis Alert threshold ({s1_thresh:.3f})."
        )
        if truncated:
            caption_txt += f"  First {MAX_ROWS} hours shown. Download the CSV for the full table."
        pdf.cell(0, 5, caption_txt, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        cols = [
            ("Hour (ICULOS)", 40), ("Probability", 38),
            ("Category",      58), ("Alert",        38),
        ]
        widths = [c[1] for c in cols]
        pdf.table_header(cols, color=TEAL)
        for i, row in enumerate(display_rows):
            hr   = str(row.get("Hour",     row.get("hour",     "?")))
            prob = row.get("Prob",     row.get("prob",     row.get("Sepsis Prob (%)", 0.0)))
            cat  = str(row.get("Category", row.get("category", "")))
            alrt = row.get("Alert",    row.get("alert",    False))
            prob_str = f"{float(prob):.2f}%" if prob != "?" else "?"
            alrt_str = "YES" if alrt else "no"
            alrt_col = RED if alrt else GREY
            # Draw row with alert coloring
            y_r = pdf.get_y()
            if i % 2 == 0:
                pdf.set_fill_color(*LIGHT)
                pdf.rect(18, y_r, 174, 6.0, style="F")
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*BLACK)
            pdf.set_xy(18, y_r)
            pdf.cell(40, 6.0, hr,       fill=False, align="C")
            pdf.cell(38, 6.0, prob_str, fill=False, align="C")
            pdf.cell(58, 6.0, cat,      fill=False, align="L")
            pdf.set_text_color(*alrt_col)
            pdf.cell(38, 6.0, alrt_str, fill=False, align="C")
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 6, "Hourly prediction table not available in this report.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Section 4: Probability Timeline Chart ───────────────────────────────
    pdf.section_heading("4.  Hourly Probability Timeline Chart", TEAL)
    if chart_png_bytes:
        pdf.embed_png(
            chart_png_bytes, max_w=174,
            caption=(
                f"Figure: Predicted hourly sepsis probability. "
                f"Dashed line = Sepsis Alert threshold ({s1_thresh:.3f}). "
                f"Dotted line = Borderline threshold ({borderline_thresh:.3f}). "
                f"Red bars = hours that crossed the Sepsis Alert threshold."
            ),
        )
    else:
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 6, "Probability timeline chart not available in this report.",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # ── Section 5: Physiological Trend Summary ──────────────────────────────
    if trend_rows:
        pdf.section_heading("5.  Physiological Trend Summary", TEAL)
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 5, "Values computed from all available observations in the uploaded file.",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        cols = [
            ("Parameter",      50), ("Recent Value", 28),
            ("Mean",           22), ("Min",           20),
            ("Max",            20), ("Trend",         34),
        ]
        widths = [c[1] for c in cols]
        pdf.table_header(cols, color=TEAL)
        for i, row in enumerate(trend_rows):
            trend_val = _safe(str(row.get("Trend", "-")))
            # Ensure trend text is readable
            if "Increasing" in trend_val or "^" in trend_val:
                trend_val = "Increasing"
            elif "Decreasing" in trend_val or "v" in trend_val:
                trend_val = "Decreasing"
            elif "Stable" in trend_val or "->" in trend_val:
                trend_val = "Stable"
            pdf.table_row(
                [
                    row.get("Parameter", ""),
                    row.get("Current / Recent Value", "-"),
                    row.get("Mean over Stay", "-"),
                    row.get("Min", "-"),
                    row.get("Max", "-"),
                    trend_val,
                ],
                widths, fill=(i % 2 == 0),
                aligns=["L", "C", "C", "C", "C", "C"],
            )
        pdf.ln(4)
    else:
        pdf.section_heading("5.  Physiological Trend Summary", TEAL)
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 6, "No standard vital-sign columns found in the uploaded file.",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # ── Section 6: SHAP Explanation ─────────────────────────────────────────
    pdf.section_heading("6.  SHAP Explanation -- Highest-Risk Hour", NAVY)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(174, 5,
        f"SHAP (SHapley Additive exPlanations) describes which features contributed most "
        f"to the model's prediction at Hour {peak_hour} (peak probability "
        f"{peak_prob * 100:.1f}%). "
        "Positive values increase the predicted sepsis probability; negative values decrease it. "
        "SHAP describes model behaviour -- it does not prove clinical causation.")
    pdf.ln(2)
    if shap_png_bytes:
        pdf.embed_png(shap_png_bytes, max_w=160,
                      caption="Figure: Top-10 feature contributions at the highest-risk hour (SHAP values).")
    else:
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(*GREY)
        pdf.multi_cell(174, 5,
            "SHAP chart not available for this report. "
            "This may occur when the XGBoost version on the deployment environment does not support "
            "the SHAP library's tree explainer. Feature importances (gain-based) may have been shown "
            "in the web interface instead.")
    pdf.ln(3)

    # ── Section 7: LIME Explanation ─────────────────────────────────────────
    pdf.section_heading("7.  LIME Explanation -- Patient-Specific", NAVY)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(174, 5,
        "LIME (Local Interpretable Model-agnostic Explanations) approximates model behaviour "
        "around this specific patient's observation at the highest-risk hour. "
        "Results are approximate and vary across runs. They do not prove clinical causation.")
    pdf.ln(2)
    if lime_png_bytes:
        pdf.embed_png(lime_png_bytes, max_w=160,
                      caption="Figure: LIME feature contributions at the highest-risk hour.")
    elif lime_text:
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*BLACK)
        for line in lime_text.split("  "):
            if line.strip():
                pdf.cell(0, 5.5, line.strip(), new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(*GREY)
        pdf.multi_cell(174, 5,
            "LIME explanation not available for this report. "
            "LIME requires the `lime` package and a prediction function. "
            "If LIME was unavailable in the web interface, it will not appear here.")
    pdf.ln(3)

    # ── Section 8: Stage 2 Eligibility ──────────────────────────────────────
    elig_color = GREEN if s2_eligible else AMBER
    pdf.section_heading("8.  Stage 2 Eligibility", elig_color)
    elig_rows = [
        ("Eligible for Stage 2",
         "YES -- Risk Stratification Available" if s2_eligible
         else "NO -- Peak probability below both thresholds"),
        ("Stage 1 Reference Hour",
         f"Hour {first_alert_h}  (first hour crossing Sepsis Alert threshold)"
         if first_alert_h else "N/A"),
        ("Pre-alert History",
         f"{s2_pre_hrs} hour(s) of data before the reference hour"
         if s2_pre_hrs > 0
         else "0 hours  (reference hour is the first observation -- no prior history available)"),
        ("Post-alert Observations",
         f"{s2_post_hrs} hour(s) of data from the reference hour onward"),
        ("Eligibility Note",
         "Stage 2 eligibility indicates data availability only. "
         "It does not mean a Stage 2 prediction has been completed."),
    ]
    for i, (k, v) in enumerate(elig_rows):
        vc = GREEN if (k == "Eligible for Stage 2" and s2_eligible) else BLACK
        pdf.key_value_row(k, v, fill=(i % 2 == 0), val_color=vc)
    pdf.ln(3)

    # ── Section 9: Clinical Interpretation ──────────────────────────────────
    pdf.section_heading("9.  Clinical Interpretation", NAVY)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(174, 5,
        f"Stage 1 operates at the hourly ICU observation level. Each ICU hour receives "
        f"an independent sepsis probability from a regularized, cost-sensitive XGBoost model "
        f"trained on 135 engineered clinical features. "
        f"The patient-level decision is based on the peak (maximum) probability across all "
        f"{n_rows} observed hours. "
        f"The model uses a deliberately lowered Sepsis Alert threshold "
        f"({s1_thresh:.3f}, {s1_thresh * 100:.1f}%) compared to the standard 0.5 threshold. "
        f"This maximises recall -- the model intentionally identifies more potential sepsis cases "
        f"at the cost of a higher false-positive rate. "
        f"A Borderline threshold of {borderline_thresh:.3f} ({borderline_thresh * 100:.1f}%) "
        f"is also supported to flag cases that narrowly miss the alert threshold. "
        f"All predictions must be interpreted alongside full clinical assessment."
    )
    pdf.ln(3)
    pdf.disclaimer_box()
    return bytes(pdf.output())


def build_stage2_pdf(
    patient_id: str, age: str, gender: str,
    n_total_rows: int, s1_result_label: str, s1_prob,
    ref_hour: int, n_pre: int, n_early: int,
    prob_higher_risk: float, s2_thresh: float,
    risk_label: str, data_sufficient: bool, trend_data: list,
    # New optional parameters
    shap_png_bytes: bytes = None,
    lime_png_bytes: bytes = None,
    lime_text: str = None,
    obs_period: str = None,
) -> bytes:
    """Build Stage 2 Post-Sepsis Risk Stratification PDF report."""
    generated  = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    risk_color = RED if risk_label == "HIGHER RISK" else GREEN

    pdf = SepsiArcPDF(
        "SepsiArc -- Stage 2: Post-Sepsis Risk Stratification Report",
        patient_id, generated,
    )
    pdf.add_page()
    pdf.title_block()

    # ── Section 1: Stage 1 Context ──────────────────────────────────────────
    pdf.section_heading("1.  Stage 1 Context", TEAL)
    s1_color = RED if s1_result_label == "SEPSIS ALERT" else (AMBER if "BORDERLINE" in s1_result_label else GREY)
    ctx_rows = [
        ("Stage 1 Classification",  s1_result_label),
        ("Stage 1 Peak Probability", f"{s1_prob * 100:.2f}%" if s1_prob else "N/A"),
        ("Stage 1 Reference Hour",   f"Hour {ref_hour}  (first hour crossing Stage 1 alert threshold)"),
        ("Stage 2 Assessment Status","Completed -- risk stratification available"),
    ]
    for i, (k, v) in enumerate(ctx_rows):
        vc = s1_color if k == "Stage 1 Classification" else BLACK
        pdf.key_value_row(k, v, fill=(i % 2 == 0), val_color=vc)
    pdf.ln(3)

    # ── Section 2: Patient / Input Information ──────────────────────────────
    pdf.section_heading("2.  Patient / Input Information", TEAL)
    pt_rows = [
        ("Patient / File ID",       patient_id),
        ("Age",                     age),
        ("Gender",                  gender),
        ("Total ICU Observations",  f"{n_total_rows} hourly rows"),
        ("Observation Period",       obs_period if obs_period else f"Hour 1 to Hour {n_total_rows}"),
        ("Pre-label History",        f"{n_pre} hour(s) before Stage 1 reference hour"
                                     if n_pre > 0
                                     else "0 hours  (reference hour is the first observation -- "
                                          "no prior-window data available)"),
        ("Early Post-label Obs.",   f"{n_early} hour(s) from Stage 1 reference hour onward"),
        ("Stage 2 Input Status",     "Sufficient -- prediction completed" if data_sufficient
                                     else "Insufficient -- missing pre-label or post-label data"),
    ]
    for i, (k, v) in enumerate(pt_rows):
        pdf.key_value_row(k, v, fill=(i % 2 == 0))
    if not data_sufficient:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*AMBER)
        pdf.multi_cell(174, 4.5,
            "WARNING: Stage 2 input is flagged as insufficient. "
            "The prediction may be unreliable due to limited temporal coverage.")
        pdf.set_text_color(*BLACK)
    pdf.ln(3)

    # ── Section 3: Eligibility & Reference Hour ─────────────────────────────
    pdf.section_heading("3.  Eligibility and Reference-Hour Selection", GREEN)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(174, 5,
        f"The patient was assessed by Stage 1, which produced a '{s1_result_label}' result. "
        f"The Stage 1 reference hour (Hour {ref_hour}) is the first ICU hour at which the "
        f"Stage 1 model's predicted sepsis probability crossed the Sepsis Alert threshold (0.143). "
        f"Stage 2 uses this reference hour to define two temporal windows: "
        f"(a) pre-label historical window -- the {n_pre} hour(s) before Hour {ref_hour} -- "
        f"and (b) early post-label window -- the {n_early} hour(s) from Hour {ref_hour} onward. "
        f"Patient-level aggregate features are computed from these windows."
    )
    if n_pre == 0:
        pdf.ln(1)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*AMBER)
        pdf.multi_cell(174, 4.5,
            "Note: The pre-label window contains zero observations because the Stage 1 reference "
            "hour (Hour " + str(ref_hour) + ") coincides with or precedes the first uploaded "
            "observation. Pre-label means are therefore not available and have been imputed. "
            "This limits the reliability of Stage 2 features derived from the pre-label window.")
        pdf.set_text_color(*BLACK)
    pdf.ln(3)

    # ── Section 4: Stage 2 Risk Result ─────────────────────────────────────
    pdf.section_heading("4.  Stage 2 Risk Stratification Result", NAVY)
    pdf.result_badge(
        label=risk_label,
        probability=f"{prob_higher_risk * 100:.2f}%",
        threshold=f"{s2_thresh:.2f}  ({s2_thresh * 100:.2f}%)",
        color=risk_color,
    )
    pred_rows = [
        ("Model",                    "XGBoost (patient-level, max_depth=2)"),
        ("Features Used",            "65 patient-level aggregate features (pre + post windows)"),
        ("Target Variable",          "Risk_50pct  (physiological-risk proxy, NOT mortality)"),
        ("Decision Threshold",       f"{s2_thresh:.2f}  ({s2_thresh * 100:.2f}%)"),
        ("Higher Risk Probability",  f"{prob_higher_risk * 100:.2f}%"),
        ("Final Classification",     risk_label),
        ("Data Sufficiency",         "Sufficient" if data_sufficient else "Insufficient (see above)"),
    ]
    for i, (k, v) in enumerate(pred_rows):
        vc = risk_color if k == "Final Classification" else BLACK
        pdf.key_value_row(k, v, fill=(i % 2 == 0), val_color=vc)
    pdf.ln(3)

    # ── Section 5: Physiological Trend Summary ──────────────────────────────
    pdf.section_heading("5.  Physiological Trend Summary", TEAL)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 5,
             f"Pre = hours before Stage 1 reference (Hour {ref_hour})  |  "
             f"Post = hours from Stage 1 reference onward",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    if trend_data:
        cols = [
            ("Parameter",   52), ("Pre Mean",  28), ("Post Mean", 28),
            ("Change",      25), ("Trend",     41),
        ]
        widths = [c[1] for c in cols]
        pdf.table_header(cols, color=TEAL)
        for i, row in enumerate(trend_data):
            pm_raw = row.get("Pre-label Mean", "-")
            em_raw = row.get("Early Post-label Mean", "-")
            # Show "N/A" instead of "-" when pre-label window is empty
            pm_str = "N/A (no pre-label data)" if (pm_raw in ("-", "--", "", None) and n_pre == 0) \
                     else str(pm_raw) if pm_raw not in ("-", "--", "", None) else "-"
            try:
                pm_f = float(pm_raw) if pm_raw not in ("-", "--", "", None) else None
                em_f = float(em_raw) if em_raw not in ("-", "--", "", None) else None
                change = f"{(em_f - pm_f):+.2f}" if (pm_f is not None and em_f is not None) else "-"
            except Exception:
                change = "-"
            trend_val = _safe(str(row.get("Trend", "-")))
            if "Increasing" in trend_val or "^" in trend_val:
                trend_val = "Increasing"
            elif "Decreasing" in trend_val or "v" in trend_val:
                trend_val = "Decreasing"
            elif "Stable" in trend_val or "->" in trend_val:
                trend_val = "Stable"
            elif trend_val in ("-", "--"):
                trend_val = "N/A"
            pdf.table_row(
                [row.get("Parameter", ""), pm_str,
                 str(em_raw) if em_raw not in ("-", "--", "", None) else "-",
                 change, trend_val],
                widths, fill=(i % 2 == 0),
                aligns=["L", "C", "C", "C", "C"],
            )
        pdf.ln(4)
    else:
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 6, "No trend data available.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # ── Section 6: SHAP Explanation ─────────────────────────────────────────
    pdf.section_heading("6.  SHAP Explanation -- Stage 2 Prediction", NAVY)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(174, 5,
        "SHAP values show which patient-level aggregate features contributed most "
        "to the Stage 2 model's Higher Risk probability. "
        "Positive values increase the Higher Risk probability; negative values decrease it.")
    pdf.ln(2)
    if shap_png_bytes:
        pdf.embed_png(shap_png_bytes, max_w=160,
                      caption="Figure: Top feature contributions to Stage 2 risk classification (SHAP).")
    else:
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(*GREY)
        pdf.multi_cell(174, 5,
            "Stage 2 SHAP chart not available for this report. "
            "The SHAP explanation may have been shown in the web interface. "
            "Refer to the web application for the interactive SHAP chart.")
    pdf.ln(3)

    # ── Section 7: LIME Explanation ─────────────────────────────────────────
    pdf.section_heading("7.  LIME Explanation -- Stage 2 Prediction", NAVY)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(174, 5,
        "LIME approximates the Stage 2 model's behaviour around this specific patient's "
        "feature values. Results are approximate and do not prove clinical causation.")
    pdf.ln(2)
    if lime_png_bytes:
        pdf.embed_png(lime_png_bytes, max_w=160,
                      caption="Figure: LIME feature contributions to Stage 2 risk classification.")
    elif lime_text:
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*BLACK)
        for line in lime_text.split("  "):
            if line.strip():
                pdf.cell(0, 5.5, line.strip(), new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(*GREY)
        pdf.multi_cell(174, 5,
            "Stage 2 LIME explanation not available for this report. "
            "Refer to the web application for the interactive LIME chart.")
    pdf.ln(3)

    # ── Section 8: Model Interpretation & Limitations ───────────────────────
    pdf.section_heading("8.  Model Interpretation and Limitations", NAVY)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(174, 5,
        f"Stage 2 is a patient-level risk-stratification model. It does not replicate "
        f"Stage 1 -- it uses different features, a different model, and a different target. "
        f"The target variable (Risk_50pct) is an engineered physiological-risk proxy "
        f"constructed from observations available AFTER the Stage 1 reference hour. "
        f"This target was used for training and evaluation only; it is NOT a prediction of "
        f"mortality, survival probability, or validated long-term clinical outcome. "
        f"The model was trained and evaluated on the PhysioNet/CinC Challenge 2019 dataset. "
        f"Stage 2 uses 65 aggregate features derived from pre-label and post-label windows "
        f"relative to the Stage 1 reference hour. "
        f"When the pre-label window is empty (0 observations), the corresponding features are "
        f"imputed using the saved Stage 2 imputer -- this may reduce prediction reliability. "
        f"All Stage 2 predictions must be reviewed alongside the full clinical record."
    )
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
    """Build combined Stage 1 + Stage 2 PDF report (existing compact format)."""
    generated  = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    s1_color   = RED if s1_prob >= s1_thresh else (AMBER if s1_prob >= 0.08 else GREEN)
    risk_color = RED if risk_label == "HIGHER RISK" else GREEN

    pdf = SepsiArcPDF(
        "SepsiArc -- Complete Dual-Stage Patient Analysis Report",
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
        probability=f"{s1_prob * 100:.2f}%",
        threshold=f"{s1_thresh:.3f} ({s1_thresh * 100:.1f}%)",
        color=s1_color,
    )
    for i, (k, v) in enumerate([
        ("Model",                  "Regularized Cost-Sensitive XGBoost (Model A)"),
        ("Features",               "135 engineered hourly features"),
        ("Decision Threshold",     f"{s1_thresh:.3f} ({s1_thresh * 100:.1f}%)"),
        ("Peak Probability",       f"{s1_prob * 100:.2f}%"),
        ("Peak Probability Hour",  f"Hour {peak_hour}"),
        ("First Alert Hour",       f"Hour {first_alert_h}" if first_alert_h else "None"),
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
            trend_val = _safe(str(row.get("Trend", "-")))
            if "Increasing" in trend_val: trend_val = "Increasing"
            elif "Decreasing" in trend_val: trend_val = "Decreasing"
            elif "Stable" in trend_val: trend_val = "Stable"
            pdf.table_row(
                [row.get("Parameter", ""), row.get("Current / Recent Value", "-"),
                 row.get("Mean over Stay", "-"), row.get("Min", "-"),
                 row.get("Max", "-"), trend_val],
                widths, fill=(i % 2 == 0),
            )
        pdf.ln(3)

    # 3. Transition
    pdf.divider()
    pdf.section_heading("3.  Stage 1 to Stage 2 Transition", AMBER)
    for i, (k, v) in enumerate([
        ("Stage 2 Eligible",        "YES"),
        ("Reference Hour",          f"Hour {ref_hour}  (Stage 1 first alert hour)"),
        ("Pre-label Window",        f"{n_pre} hour(s)"
                                    if n_pre > 0
                                    else "0 hours  (reference is the first observation)"),
        ("Early Post-label Window", f"{n_early} hour(s)"),
    ]):
        pdf.key_value_row(k, v, fill=(i % 2 == 0))
    pdf.ln(3)

    # 4. Stage 2
    pdf.section_heading("4.  Stage 2 -- Post-Sepsis Risk Stratification", RED)
    pdf.result_badge(
        label=risk_label,
        probability=f"{prob_higher_risk * 100:.2f}%",
        threshold=f"{s2_thresh:.2f} ({s2_thresh * 100:.2f}%)",
        color=risk_color,
    )
    for i, (k, v) in enumerate([
        ("Model",                   "XGBoost (patient-level, max_depth=2)"),
        ("Features",                "65 patient-level aggregate features"),
        ("Target",                  "Risk_50pct  (physiological-risk proxy -- NOT mortality)"),
        ("Decision Threshold",      f"{s2_thresh:.2f} ({s2_thresh * 100:.2f}%)"),
        ("Higher Risk Probability", f"{prob_higher_risk * 100:.2f}%"),
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
            pm_raw = row.get("Pre-label Mean", "-")
            em_raw = row.get("Early Post-label Mean", "-")
            pm_str = "N/A" if (pm_raw in ("-", "--", "", None) and n_pre == 0) \
                     else str(pm_raw) if pm_raw not in ("-", "--", "", None) else "-"
            try:
                pm_f = float(pm_raw) if pm_raw not in ("-", "--", "", None) else None
                em_f = float(em_raw) if em_raw not in ("-", "--", "", None) else None
                change = f"{(em_f - pm_f):+.2f}" if (pm_f is not None and em_f is not None) else "-"
            except Exception:
                change = "-"
            trend_val = _safe(str(row.get("Trend", "-")))
            if "Increasing" in trend_val: trend_val = "Increasing"
            elif "Decreasing" in trend_val: trend_val = "Decreasing"
            elif "Stable" in trend_val: trend_val = "Stable"
            pdf.table_row(
                [row.get("Parameter", ""), pm_str,
                 str(em_raw) if em_raw not in ("-", "--", "", None) else "-",
                 change, trend_val],
                widths, fill=(i % 2 == 0),
            )
        pdf.ln(3)

    # 5. Summary
    pdf.divider()
    pdf.section_heading("5.  Overall Model-Based Summary", NAVY)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(174, 5,
        f"Stage 1 identified a {s1_result_label.lower()} result (Hour {ref_hour}). "
        f"Stage 2 classified this patient as {risk_label} based on the physiological "
        f"trajectory around the reference hour. "
        f"Both results are model-based research outputs and must not replace clinical assessment."
    )
    pdf.ln(3)
    pdf.disclaimer_box()
    return bytes(pdf.output())
