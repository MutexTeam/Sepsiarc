import streamlit as st
import os, sys

# Robust path insertion — works regardless of how Streamlit launches the page
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from utils.ui_components import (
    render_header, render_footer, render_sidebar,
    render_disclaimer_banner,
    NAVY, TEAL, GREEN, AMBER, RED, SLATE_200, SLATE_400, SLATE_700,
)

st.set_page_config(
    page_title="SepsiArc – Clinical Assistant",
    page_icon="💬",
    layout="wide",
)

render_header(
    page_title="SepsiArc Clinical Assistant",
    page_subtitle="Educational and explanatory information about sepsis and this AI system. Does not make predictions.",
    icon="💬",
)
render_sidebar("assistant")
render_disclaimer_banner()

# ── Knowledge Base ────────────────────────────────────────────────────────────
KB = [
    {
        "keys": ["hello", "hi", "hey", "good morning", "good afternoon", "who are you", "what can you do", "help"],
        "answer": """**Hello! I'm the SepsiArc Clinical Assistant.**

I'm an evidence-based educational Q&A system specialised in sepsis and critical care.

**I can help you understand:**
- 🔍 What is sepsis, septic shock, SIRS?
- ⚠️ Clinical scoring systems — qSOFA, SOFA, Lactate
- 💊 Sepsis-3 definitions and Hour-1 Bundle
- 🧠 Post-sepsis syndrome and recovery
- 🤖 How SepsiArc's Stage 1 and Stage 2 models work
- 📊 What SHAP explanations mean and how to interpret them
- 📈 What probabilities mean and why false positives/negatives occur

**Important:** I provide **educational information only**. I do not make Stage 1 or Stage 2 predictions — use the dedicated Stage 1 page for clinical assessment. Always consult a qualified healthcare professional for medical decisions.

*Click any Quick FAQ button below or type your question to begin.*"""
    },
    {
        "keys": ["what is sepsis", "define sepsis", "meaning of sepsis", "sepsis mean", "sepsis definition"],
        "answer": """**What is Sepsis?**

Sepsis is a life-threatening medical emergency caused by the body's extreme response to an infection. When the immune system goes into overdrive fighting an infection, it can damage its own tissues and organs.

**Key facts:**
- Triggered by bacterial, viral, fungal, or parasitic infections
- Affects ~49 million people annually worldwide
- Kills over 11 million people each year (1 in 5 deaths globally)
- Can progress to septic shock within hours
- **Time is critical** — every hour of delay increases mortality by ~7%

**Sepsis-3 Definition (2016):**
Life-threatening organ dysfunction caused by a dysregulated host response to infection (Sequential Organ Failure Assessment score ≥ 2)

**Practical difference from infection:**
All sepsis involves infection, but not all infection involves sepsis. Sepsis is specifically when the immune response to infection starts damaging the body's own organs."""
    },
    {
        "keys": ["septic shock", "what is septic shock", "shock", "vasopressor"],
        "answer": """**What is Septic Shock?**

Septic shock is the most severe form of sepsis — characterised by a dramatic drop in blood pressure that does not respond to fluid resuscitation alone.

**Definition (Sepsis-3, 2016):**
- Sepsis **PLUS**
- Need for vasopressors to maintain MAP ≥ 65 mmHg
- Serum Lactate > 2 mmol/L despite adequate fluid resuscitation

**Mortality:** 40–50% (vs 15–25% for sepsis without shock)

**First-line treatment:**
- Vasopressors: **Norepinephrine** (first-line)
- Add vasopressin if norepinephrine dose escalation needed
- Corticosteroids (hydrocortisone) if refractory shock
- Aggressive lactate-guided fluid resuscitation
- ICU admission is mandatory

*Note: SepsiArc does not prescribe vasopressor doses or treatment orders. Consult your institutional protocol.*"""
    },
    {
        "keys": ["sirs", "sirs criteria", "systemic inflammatory", "what is sirs"],
        "answer": """**What is SIRS? (Systemic Inflammatory Response Syndrome)**

SIRS indicates an extreme systemic inflammatory response — it can be triggered by infection, trauma, burns, or pancreatitis.

**SIRS is positive when ≥ 2 of these 4 criteria are met:**
1. **Temperature** > 38°C (fever) OR < 36°C (hypothermia)
2. **Heart Rate** > 90 bpm (tachycardia)
3. **Respiratory Rate** > 20 br/min OR PaCO₂ < 32 mmHg
4. **WBC** > 12,000/µL OR < 4,000/µL OR >10% bands

**Key interpretation:**
- SIRS alone ≠ sepsis (can occur in non-infectious conditions)
- SIRS + suspected/confirmed infection → raises concern for sepsis
- Newer Sepsis-3 (2016) uses SOFA instead of SIRS, but SIRS remains clinically useful
- SepsiArc computes SIRS score automatically as part of Stage 1 feature engineering"""
    },
    {
        "keys": ["qsofa", "quick sofa", "what is qsofa"],
        "answer": """**What is qSOFA? (Quick Sequential Organ Failure Assessment)**

qSOFA is a rapid bedside screening tool for suspected sepsis that requires no laboratory tests.

**Score 1 point for each:**
1. **Respiratory Rate** ≥ 22 breaths/min
2. **Altered mental status** (GCS < 15)
3. **Systolic BP** ≤ 100 mmHg

**Interpretation:**
- Score **≥ 2/3** → Likely organ dysfunction; investigate urgently for sepsis
- Score **0–1** → Lower likelihood (but sepsis is not excluded)

**Advantages:** Can be completed in 60 seconds at bedside; no lab tests required.

**SepsiArc:** qSOFA score is automatically computed from HR, Resp, and SBP inputs and used as an engineered feature in Stage 1."""
    },
    {
        "keys": ["sofa", "sofa score", "sequential organ failure", "organ failure score"],
        "answer": """**SOFA Score — Sequential Organ Failure Assessment**

SOFA quantifies severity of organ dysfunction across 6 systems. It is the basis of the Sepsis-3 definition.

| Organ System | Measure |
|---|---|
| Respiratory | PaO₂/FiO₂ ratio |
| Coagulation | Platelets |
| Liver | Bilirubin |
| Cardiovascular | MAP / vasopressor need |
| CNS | Glasgow Coma Scale (GCS) |
| Renal | Creatinine / urine output |

**Total range:** 0–24 points
- Score **≥ 2 increase** from baseline = organ dysfunction (Sepsis-3 criterion)
- Score ≥ 11 = approximately 50% hospital mortality
- Increasing SOFA = deterioration; decreasing SOFA = recovery

**SepsiArc** computes a SOFA surrogate from available vital signs and labs. A full SOFA score requires GCS and blood gas values."""
    },
    {
        "keys": ["lactate", "serum lactate", "lactate level", "what is lactate"],
        "answer": """**Understanding Serum Lactate in Sepsis**

Lactate is produced when cells switch to anaerobic metabolism due to inadequate oxygen delivery — a sign of tissue hypoperfusion.

| Lactate Level | Clinical Interpretation |
|---|---|
| < 2 mmol/L | Normal |
| 2–4 mmol/L | Elevated — suggests sepsis risk; increased mortality |
| > 4 mmol/L | Severely elevated — septic shock criterion; high mortality |

**Clinical use:**
- Lactate > 2 mmol/L is part of the Sepsis-3 shock definition
- Used to guide fluid resuscitation (target lactate clearance ≥ 10%)
- Repeat every 2 hours; improvement is a positive prognostic sign

**In SepsiArc:** Lactate is one of the 15 raw clinical inputs and is also used to derive `Lactate_risk` and `hyperlactatemia` flags as engineered features for Stage 1."""
    },
    {
        "keys": ["map", "mean arterial pressure", "why is map important", "map in sepsis"],
        "answer": """**Why is MAP Important in Sepsis?**

Mean Arterial Pressure (MAP) reflects average perfusion pressure to vital organs.

**Formula:** MAP ≈ DBP + (SBP − DBP) / 3

**Clinical significance:**
- MAP < 65 mmHg = end-organ hypoperfusion threshold
- Sustained MAP < 65 mmHg despite fluid resuscitation requires vasopressors
- MAP is a key component of septic shock definition

**SepsiArc uses MAP as:**
- A direct raw clinical input
- To derive `MAP_risk` (binary flag: MAP < 65)
- To derive `MAP_HR_ratio`, `MAP_roll3_mean`, lag features, and trend features"""
    },
    {
        "keys": ["shock index", "what is shock index", "what does shock index mean"],
        "answer": """**What is the Shock Index?**

Shock Index = Heart Rate ÷ Systolic Blood Pressure

**Interpretation:**
| Value | Interpretation |
|---|---|
| < 0.7 | Normal |
| 0.7–1.0 | Moderate concern |
| > 1.0 | Elevated — significant haemodynamic compromise |
| > 1.4 | Severe compromise — high-risk |

**Clinical use:**
- Simple bedside calculation requiring no laboratory tests
- Predicts haemodynamic instability in trauma, sepsis, and haemorrhage
- Higher values correlate with increased mortality

**SepsiArc:** Shock Index is automatically derived by the backend from HR and SBP inputs. It is not a user-entered field."""
    },
    {
        "keys": ["post-sepsis", "post sepsis syndrome", "long-term", "after sepsis", "recovery effects", "survivors"],
        "answer": """**Post-Sepsis Syndrome (PSS)**

40–60% of sepsis survivors develop Post-Sepsis Syndrome — a cluster of physical and psychological effects lasting months to years.

**Physical effects:**
- Extreme fatigue (in ~80% of survivors)
- Muscle weakness (ICU-acquired weakness)
- Breathlessness, reduced exercise tolerance
- Chronic pain or neuropathy
- Increased susceptibility to infections
- Chronic organ impairment (kidney, heart, lung)

**Cognitive effects:**
- Memory problems and difficulty concentrating
- PTSD — up to 30% of survivors
- Depression and anxiety
- Sleep disturbances

**Management framework:**
- Sepsis survivorship clinics (30/90/180-day follow-ups)
- Physiotherapy and occupational therapy
- Psychological support and counselling
- Regular organ function monitoring

*Note: SepsiArc's Stage 2 assesses post-sepsis risk — not a prediction of who will develop PSS.*"""
    },
    {
        "keys": ["shap", "explain", "explainability", "ai explain", "feature importance", "how does the ai", "interpretable", "what is shap"],
        "answer": """**What is SHAP and How Should I Interpret It?**

SHAP (SHapley Additive exPlanations) is a method from cooperative game theory used to explain individual AI predictions.

**How SHAP works:**
- Each input feature receives a SHAP value
- **Positive SHAP value** → this feature pushes the prediction toward the positive class (e.g., sepsis)
- **Negative SHAP value** → this feature pushes the prediction away from the positive class
- The sum of all SHAP values = the model's final log-odds contribution

**Example reading:**
> "Lactate = 4.2 mmol/L → SHAP +0.12" means: the elevated Lactate pushed the sepsis probability higher.

**Critical interpretation rules:**
1. SHAP values describe **model behaviour** — not clinical causality
2. A high SHAP value does NOT mean "this caused sepsis"
3. It means "given the model learned during training, this feature contributed to this prediction"
4. SHAP explanations are local — they apply to this specific patient's prediction

**Why SHAP matters in clinical AI:**
- Provides transparency in AI decisions
- Helps clinicians identify which values to prioritise in review
- Supports regulatory requirements for explainable AI in healthcare"""
    },
    {
        "keys": ["how does stage 1 work", "stage 1 model", "how stage 1", "detection stage", "stage 1 explain"],
        "answer": """**How Stage 1 (Sepsis Detection) Works**

Stage 1 uses a **LightGBM-Calibrated** model trained on the PhysioNet 2019 Sepsis Dataset.

**Raw inputs collected from you (15 fields):**
Vital signs: HR, O2Sat, Temp, SBP, MAP, DBP, Resp
Lab values: Creatinine, WBC, Platelets, Glucose, Lactate
Patient context: Age, Gender, ICULOS

**What the backend derives automatically (64 additional features):**
- Clinical flags: SIRS score, qSOFA, tachycardia, tachypnea, fever, hypoxia, MAP risk
- Derived ratios: Shock Index, Pulse Pressure, MAP Ratio, oxygenation index
- Rolling statistics: 3-point rolling mean, standard deviation for HR, MAP, O2Sat, Temp, Resp, SBP
- Lag features: 1-step and 3-step lags for key vitals
- Delta features: Change from previous observation
- Contextual flags: early_icu, late_icu

**Feature contract:** 79 total features verified before every prediction.

**Decision logic:**
1. Model outputs a probability (0–1)
2. Clinical gating rule: if NO SIRS/qSOFA criteria met, raw probability is capped
3. Final probability vs threshold (0.33) → SEPSIS POSITIVE or NEGATIVE

**Training set:** 1.55M ICU observations · 40,336 patients · PhysioNet 2019"""
    },
    {
        "keys": ["how does stage 2 work", "stage 2 model", "post sepsis model", "risk stratification", "stage 2 explain"],
        "answer": """**How Stage 2 (Post-Sepsis Risk) Works**

Stage 2 is a completely separate model from Stage 1 — different architecture, different features, different threshold.

**Key differences from Stage 1:**
- Stage 1: Detects IF sepsis is currently occurring
- Stage 2: Assesses the severity/risk of post-sepsis recovery

**Stage 2 is only accessible after a Stage 1 POSITIVE result.**

**Stage 2 inputs:** Current vitals (HR, O2Sat, Temp, SBP, DBP, Resp, MAP), laboratory values (BUN, WBC, Glucose, Calcium, Magnesium, Phosphate, Hgb, Hct, PTT), and patient context (Age, Gender, ICULOS, Unit).

**Feature contract:** 65 features, derived similarly to Stage 1 but without Lactate-specific features that were used to construct the Stage 2 label (no data leakage).

**Stage 2 label:** A project-defined proxy risk classification derived from clinical indicators. Not a validated long-term mortality prediction.

**Performance (test set):** AUROC 0.9631 · F1 0.7486 · Recall 0.7083"""
    },
    {
        "keys": ["probability", "what does probability mean", "interpret probability", "what does percent mean"],
        "answer": """**What Does the Probability Mean?**

The sepsis probability shown by SepsiArc is a **calibrated model output** — it represents the model's estimated likelihood that the patient falls into the positive class.

**How to interpret it:**
- A probability of 72% means: for patients with similar clinical features, the model classifies approximately 72% as sepsis-positive
- It is **not** a guarantee or a precise individual prediction
- The **threshold** (not a fixed 50%) determines the classification: if probability ≥ threshold → POSITIVE

**Important limitations:**
- The model was trained on population-level ICU data — individual patients may differ
- Class imbalance (only ~1.8% sepsis rate in training data) means the threshold is calibrated to optimise clinical utility, not statistical balance
- A 35% probability still warrants careful clinical assessment if the patient shows other signs

**The probability is a decision-support tool, not a diagnostic test result.**"""
    },
    {
        "keys": ["false positive", "false negative", "accuracy", "wrong prediction", "errors", "why wrong"],
        "answer": """**Why Can False Positives and False Negatives Occur?**

All predictive models, including SepsiArc's, will produce some incorrect predictions.

**False Positive** (predicted positive, actually negative):
- Patient flagged as sepsis risk but does not have sepsis
- Risk: unnecessary investigations, patient anxiety
- In imbalanced datasets, reducing false negatives (missing sepsis) is often prioritised over reducing false positives
- SepsiArc's clinical gating rule helps reduce false alarms

**False Negative** (predicted negative, actually positive):
- Patient not flagged, but sepsis is present
- Risk: delayed treatment
- This is more dangerous — hence Stage 1 threshold (0.33) is set to favour sensitivity (recall)

**Why errors occur:**
1. Missing data: many laboratory values have high missingness rates
2. Class imbalance: only 1.8% of ICU time-points are sepsis-positive
3. Temporal complexity: sepsis is a dynamic process; a single-point prediction is limited
4. Dataset limitations: model trained on PhysioNet; real-world performance may differ

**This is why SepsiArc's output must always be interpreted by a clinician — never used as a standalone decision.**"""
    },
    {
        "keys": ["treatment", "treat sepsis", "how is sepsis treated", "antibiotics", "therapy", "hour-1 bundle"],
        "answer": """**Sepsis Treatment — The Hour-1 Bundle (Surviving Sepsis Campaign)**

When sepsis is identified, these actions must happen within **1 hour**:

1. 🩸 **Blood Cultures × 2** — Before starting antibiotics
2. 💊 **Broad-spectrum IV Antibiotics** — Immediately; every hour of delay increases mortality
3. 💧 **IV Fluid Resuscitation** — 30 mL/kg crystalloid if hypoperfusion signs present
4. 🔬 **Measure Serum Lactate** — If > 4 mmol/L, aggressive resuscitation; repeat at 2h
5. 💉 **Vasopressors** — Norepinephrine if MAP < 65 mmHg despite adequate fluids

**Ongoing treatment elements:**
- ICU admission for severe sepsis
- Source control (drain abscess, remove infected catheter)
- Mechanical ventilation if respiratory failure
- Renal replacement therapy if severe AKI
- Glucose control (target 140–180 mg/dL)

*Note: SepsiArc does not provide medication doses or prescribe treatment. All clinical decisions must follow institutional protocol and physician assessment.*"""
    },
    {
        "keys": ["dataset", "physionet", "training data", "where does data come from", "reyna"],
        "answer": """**About the Training Dataset**

SepsiArc was trained on the **PhysioNet/Computing in Cardiology Challenge 2019** dataset.

**Dataset details:**
- **Source:** PhysioNet (Reyna et al., 2019) — publicly available
- **Patients:** 40,336 ICU patients across multiple hospital systems
- **Time-points:** ~1.55 million hourly ICU measurements
- **Sepsis rate:** 1.8% (severe class imbalance)
- **Features:** 40 clinical variables (vitals + labs + demographics)

**Stage 1 preprocessing:**
- Clinical gating applied for false alarm reduction
- 79 engineered features derived from 15 raw clinical inputs
- LightGBM-Calibrated model with isotonic regression calibration

**Stage 2:**
- Patient-level features aggregated from time-series
- 65 features with data-leakage prevention

**Citation:** Reyna MA, Josef C, Seyedi S et al. (2019). "Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019." Critical Care Medicine."""
    },
    {
        "keys": ["model performance", "accuracy", "auroc", "auprc", "f1", "metrics"],
        "answer": """**SepsiArc Model Performance (Project Test Set)**

These are evaluation metrics from the project test set — **not live clinical performance**.

**Stage 1 — Sepsis Detection (LightGBM-Calibrated):**
| Metric | Value |
|---|---|
| AUROC | 0.8229 |
| AUPRC | 0.1039 |
| F1 Score | 0.1507 |
| Recall | 0.5038 |
| Threshold | 0.33 |

**Stage 2 — Post-Sepsis Risk (LightGBM):**
| Metric | Value |
|---|---|
| AUROC | 0.9631 |
| F1 Score | 0.7486 |
| Precision | 0.7936 |
| Recall | 0.7083 |
| Threshold | 0.58 |

**Why Stage 1 AUROC is ~0.82 (not higher):**
The PhysioNet dataset has only ~1.8% positive rate. Traditional ML on tabular ICU data typically plateaus around AUROC 0.82–0.85 on this dataset. Higher performance typically requires deep learning on full time-series.

View the **Model Information** page for full details."""
    },
]


def get_response(question: str) -> str:
    """Match question to knowledge base and return response."""
    q = question.lower().strip()
    for item in KB:
        if any(k in q for k in item["keys"]):
            return item["answer"]
    # Broad fallback
    if any(w in q for w in ["sepsis", "infection", "icu", "hospital", "clinical", "medical", "patient", "bacteria"]):
        return """I understand you're asking about a clinical or sepsis-related topic.

I have detailed information on many topics — try asking more specifically:

- "What is sepsis?"
- "What are the symptoms of sepsis?"
- "How is sepsis treated?"
- "What is SIRS / qSOFA / SOFA?"
- "What is lactate?"
- "What is the shock index?"
- "What is SHAP?"
- "How does Stage 1 work?"
- "What does the probability mean?"
- "Why do false positives occur?"

Or click one of the **Quick FAQ buttons** above!"""

    return """I'm specialised in sepsis, critical care, and SepsiArc explanations.

I can help with:
- Sepsis definition, causes, and warning signs
- Clinical scoring systems (SIRS, qSOFA, SOFA, Lactate, MAP)
- Treatment protocols and Surviving Sepsis Campaign guidelines
- Post-sepsis syndrome and recovery
- How Stage 1 and Stage 2 models work
- SHAP explanations and AI interpretability
- Model performance and limitations

Please try rephrasing your question or use the **Quick FAQ buttons** below!"""


# ── QUICK FAQ BUTTONS ─────────────────────────────────────────────────────────
st.markdown("##### 💬 Quick Topics — Click to Ask:")

faqs = [
    ("What is Sepsis?",              "what is sepsis"),
    ("Septic Shock",                 "what is septic shock"),
    ("SIRS Criteria",                "what is sirs"),
    ("qSOFA Score",                  "what is qsofa"),
    ("Lactate Levels",               "what is lactate in sepsis"),
    ("Shock Index",                  "what is the shock index"),
    ("Post-Sepsis Syndrome",         "what is post-sepsis syndrome"),
    ("SHAP Explanations",            "what is shap"),
    ("How Stage 1 Works",            "how does stage 1 work"),
    ("How Stage 2 Works",            "how does stage 2 work"),
    ("What Probability Means",       "what does the probability mean"),
    ("False Positives/Negatives",    "why do false positives occur"),
]

# Initialise chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": get_response("hello")})

# FAQ buttons in rows
faq_row1 = st.columns(6)
faq_row2 = st.columns(6)
all_cols = list(faq_row1) + list(faq_row2)

for i, (label, q_text) in enumerate(faqs):
    if i < len(all_cols):
        if all_cols[i].button(label, key=f"faq_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": label})
            st.session_state.messages.append({"role": "assistant", "content": get_response(q_text)})
            st.rerun()

st.markdown("---")

# Sidebar clear button
with st.sidebar:
    st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── CHAT DISPLAY ─────────────────────────────────────────────────────────────
st.markdown("#### 💬 Chat")
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])

# ── CHAT INPUT ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about sepsis, SHAP, SOFA, how SepsiArc works…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚕ SepsiArc Clinical Assistant provides educational information based on published clinical guidelines. "
    "It does not make Stage 1 or Stage 2 predictions and must not be used as a substitute for clinical assessment. "
    "Sources: Surviving Sepsis Campaign (2021), Sepsis-3 Consensus Definitions (2016), PhysioNet (Reyna et al., 2019)."
)
render_footer()
