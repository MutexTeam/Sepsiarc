"""
pages/3_AI_Assistant.py  —  SepsiArc Medical AI Assistant
Conversational medical-information assistant powered by AIMLAPI
(OpenAI-compatible SDK, configured via Streamlit secrets).

Architecture:
  • get_ai_response()     — sends to AIMLAPI, returns text
  • SYSTEM_PROMPT         — professional identity + safety rules
  • SEPSIARC_CONTEXT      — verified project facts (update here only)
  • Quick Topic buttons   — route through the same AI function
  • Session state         — persists chat_messages across reruns
  • Fallback              — graceful error handling, no key leakage
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

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SepsiArc – Medical AI Assistant",
    page_icon="💬",
    layout="wide",
)

render_header(
    page_title="SepsiArc Medical AI Assistant",
    page_subtitle="Conversational educational assistant for sepsis, critical care, and SepsiArc. Powered by AI — for information only.",
    icon="💬",
)
render_sidebar("assistant")
render_disclaimer_banner()

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — professional identity + medical safety rules
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are the SepsiArc Medical AI Assistant. You provide clear, responsible, \
educational information about sepsis, general medical concepts, and the SepsiArc \
machine-learning project.

IDENTITY AND SCOPE:
- You are an educational assistant, not a licensed doctor or clinical decision-maker.
- Your primary role is to explain sepsis, critical care concepts, clinical scoring systems, \
and how the SepsiArc application works.
- You may answer ordinary general questions when appropriate.
- Politely redirect requests that require clinical judgment beyond education.

MEDICAL SAFETY RULES (strictly follow these):
1. Always state that your information is educational and not a substitute for professional \
medical assessment.
2. Never claim to confirm or rule out sepsis in a real patient.
3. Never prescribe medication, recommend specific dosages, or replace physician judgment.
4. If a user describes potentially life-threatening symptoms or a suspected medical emergency, \
advise them IMMEDIATELY to call emergency services (e.g. 112, 999, 911). Do not delay this \
with lengthy explanation.
5. Do not falsely reassure a user that they are safe because of a low model score.
6. Do not invent research papers, clinical guidelines, statistics, or medical facts.
7. Be transparent when information is uncertain or when you are reaching the limits of your \
knowledge.
8. Do not claim SepsiArc is clinically validated, approved for clinical use, or suitable for \
real-world diagnosis. It is a research and educational tool.
9. Explain that an ML probability or risk category is a model output, not the patient's \
true probability of disease.
10. Do not interpret SHAP feature contributions as proof of causation.
11. Do not fabricate patient data, prediction results, or model outputs.

COMMUNICATION STYLE:
- Adapt technical depth to the user's apparent background.
- Use clear language and explain medical jargon when you use it.
- Structure responses with headings and bullet points for clarity when appropriate.
- Keep responses concise but complete — do not pad unnecessarily.
- When relevant, mention that Stage 1 or Stage 2 assessment pages exist in SepsiArc.
"""

# ══════════════════════════════════════════════════════════════════════════════
# SEPSIARC PROJECT CONTEXT — verified facts from current source code
# Update this section when the project changes; keep separate from SYSTEM_PROMPT
# ══════════════════════════════════════════════════════════════════════════════
SEPSIARC_CONTEXT = """
VERIFIED SEPSIARC PROJECT FACTS (use these when asked about the project):

OVERVIEW:
SepsiArc is a dual-stage machine-learning clinical decision-support web application built \
with Streamlit (Python). It is deployed on Streamlit Community Cloud and is a research/\
educational tool — not a clinically validated or FDA/CE-approved product. It is trained on the \
PhysioNet/CinC Challenge 2019 ICU dataset.

STAGE 1 — Sepsis Detection:
- Algorithm: XGBoost (Regularized Cost-Sensitive Model A)
- Input method: PSV or CSV file upload (one patient's hourly ICU timeline)
- Each row = one ICU hour; columns matched by name (column order does not matter)
- Raw clinical columns used: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, BUN, Calcium, Creatinine, \
Glucose, Lactate, Hct, Hgb, WBC, Platelets, Age, Gender, HospAdmTime, ICULOS
- Feature engineering produces 135 total features via 10 steps:
  (1) Drop sparse columns, (2) Chronological sort by ICULOS, \
  (3) 18 missingness indicators, (4) 18 time-since-last-measurement features, \
  (5) Causal forward fill (max 4h vitals, max 3h lactate), \
  (6) 24 lag/change features (1h and 3h lags for 6 vitals), \
  (7) 48 rolling window statistics (3h and 6h × mean/min/max/std × 6 vitals), \
  (8) 5 clinical derived features (Shock_Index, Pulse_Pressure, MAP_Ratio, SIRS_partial, Low_MAP), \
  (9) Column median imputation, (10) Replace infinities
- Decision threshold: 0.143 (selected to achieve ≥70% recall — capturing more sepsis cases)
- Output per hour: sepsis probability 0–1
- Patient-level result: peak probability across all hours
  → ≥0.143: SEPSIS ALERT (Stage 2 accessible)
  → ≥0.08: BORDERLINE (Stage 2 accessible with caution)
  → <0.08: LOW RISK (Stage 2 locked)
- Test set performance: AUROC 0.7587, Recall 70.97%, Precision 11.2%, F1 19.35%
- SHAP explainability: TreeExplainer used on the peak-risk hour; top 10 features shown
- Model file: Stage1_Final_Model_A.json (XGBoost native JSON format)
- Training dataset: PhysioNet 2019, 11,932 patients after preprocessing (64/16/20 patient-level split)
- Sepsis rate in training rows: ~5.5%

STAGE 2 — Post-Sepsis Risk Assessment:
- Accessible only after Stage 1 ALERT or BORDERLINE result
- Algorithm: LightGBM
- Input: same PSV/CSV file as Stage 1 (re-upload)
- 65 engineered features including shock_index, SIRS_score, oxygenation_index, rolling 3h stats, \
lag/delta features, early_icu and late_icu timing flags
- Decision threshold: 0.58
- Output: HIGH RISK or LOW RISK per hour + overall patient assessment
- Test set performance: AUROC 0.9631, F1 0.7486, Precision 0.7936, Recall 0.7083
- Stage 2 label is a project-defined proxy severity classification — not a validated long-term \
mortality or survivorship prediction
- SHAP explanation provided for peak-risk hour

EXPLAINABILITY (SHAP):
- Both stages use SHAP TreeExplainer
- Positive SHAP value = feature pushes prediction toward sepsis/high-risk
- Negative SHAP value = feature reduces predicted risk
- SHAP values describe model behaviour — NOT clinical causality
- Top 10 features by absolute SHAP value are displayed

LIMITATIONS:
- Research and educational tool only; not clinically validated
- No real patient-identifying data should be entered
- Trained on PhysioNet 2019 (US ICU data); real-world generalisation not established
- Model performance metrics are from the project test set, not live clinical deployment
"""

# ══════════════════════════════════════════════════════════════════════════════
# API CLIENT SETUP
# ══════════════════════════════════════════════════════════════════════════════
# Free-model fallback list for OpenRouter (tried in order if primary model fails/empty)
_OPENROUTER_FREE_FALLBACKS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3.5-lightning:free",
    "poolside/laguna-xs-2.1:free",
    "apodex/apodex-1.1-mini:free",
    "liquid/lfm-2.5-2.6b:free",
]


def _load_secrets() -> tuple[str | None, str | None]:
    """
    Load API credentials from Streamlit secrets.
    Supports both AIMLAPI and OpenRouter keys (auto-detected by prefix).
    Returns (api_key, model_id) — both None if not configured.
    Never prints or exposes secret values.
    """
    try:
        api_key  = st.secrets.get("AIMLAPI_API_KEY",  None)
        model_id = st.secrets.get("AIMLAPI_MODEL",    None)
        # Treat empty strings as unconfigured
        api_key  = api_key.strip()  if api_key  and api_key.strip()  else None
        model_id = model_id.strip() if model_id and model_id.strip() else None
        return api_key, model_id
    except Exception:
        return None, None


def _get_base_url(api_key: str) -> str:
    """Auto-detect provider base URL from the key prefix."""
    if api_key.startswith("sk-or-"):
        return "https://openrouter.ai/api/v1"
    return "https://api.aimlapi.com/v1"


def _is_openrouter(api_key: str) -> bool:
    return api_key.startswith("sk-or-")


@st.cache_resource(show_spinner=False)
def _get_openai_client(api_key: str):
    """
    Instantiate an OpenAI-compatible client.
    Auto-detects provider (OpenRouter or AIMLAPI) from the key prefix.
    Cached so the client object is created once per session.
    """
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        base_url=_get_base_url(api_key),
        timeout=45.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# AI RESPONSE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def get_ai_response(question: str, conversation_history: list[dict]) -> str:
    """
    Send a question + recent conversation history to AIMLAPI and return the response.

    Parameters
    ----------
    question              : The current user question (already added to history by caller)
    conversation_history  : Full session message list [{role, content}, ...]

    Returns
    -------
    str — assistant reply, or a user-friendly error message
    """
    api_key, model_id = _load_secrets()

    # ── Guard: secrets not configured ─────────────────────────────────────
    if not api_key:
        return (
            "⚙️ **Configuration required:** The AI assistant API key is not set up yet.\n\n"
            "**For the administrator:** Add `AIMLAPI_API_KEY` and `AIMLAPI_MODEL` to "
            "`.streamlit/secrets.toml` for local development, or to the app's Secrets "
            "settings on Streamlit Community Cloud."
        )
    if not model_id:
        return (
            "⚙️ **Configuration required:** No model ID is configured.\n\n"
            "**For the administrator:** Set `AIMLAPI_MODEL` in `.streamlit/secrets.toml` "
            "(e.g. `AIMLAPI_MODEL = \"gpt-4o-mini\"`)."
        )

    # ── Build message payload ─────────────────────────────────────────────
    # System message carries identity + project context separately from chat
    system_content = SYSTEM_PROMPT.strip() + "\n\n" + SEPSIARC_CONTEXT.strip()

    # Limit history to the last 12 exchanges (24 messages) to control tokens
    # Always include the most recent user question (last message)
    history_window = conversation_history[-24:] if len(conversation_history) > 24 else conversation_history

    messages = [{"role": "system", "content": system_content}] + history_window

    # ── API call — with OpenRouter free-model fallback ──────────────────
    # Always include the current question explicitly in the messages sent to the model
    messages_with_q = messages.copy()
    # Avoid double-appending: only add if not already the last user message
    if not messages_with_q or messages_with_q[-1].get("content") != question:
        messages_with_q.append({"role": "user", "content": question})

    def _call_model(model: str) -> tuple[str | None, str | None]:
        """
        Try one model.
        Returns (reply_text, None) on success.
        Returns (None, None) on retryable / empty response — caller should try next model.
        Returns (None, error_msg) on fatal error — caller should stop and show error.
        """
        try:
            client = _get_openai_client(api_key)
            extra = {"extra_headers": {
                "HTTP-Referer": "https://mutexsepsiarc.streamlit.app",
                "X-Title": "SepsiArc Medical Assistant"
            }} if _is_openrouter(api_key) else {}

            response = client.chat.completions.create(
                model=model,
                messages=messages_with_q,
                max_tokens=800,
                temperature=0.4,
                top_p=0.9,
                **extra,
            )
            if (response.choices and
                    response.choices[0].message and
                    response.choices[0].message.content and
                    response.choices[0].message.content.strip()):
                return response.choices[0].message.content.strip(), None
            # Empty content — retryable
            return None, None

        except Exception as e:
            err = str(e)
            # Fatal errors — stop immediately
            if "401" in err or "authentication" in err.lower():
                return None, (
                    "🔐 **Authentication error:** The API key was rejected. "
                    "Please verify `AIMLAPI_API_KEY` is correct and enabled in your "
                    "provider dashboard (OpenRouter or AIMLAPI)."
                )
            if "403" in err and "only available" not in err.lower():
                return None, (
                    "🚫 **Access denied:** This model requires a paid plan. "
                    "Use a model ID ending in `:free` on OpenRouter."
                )
            # Retryable: 429, 404, 500, 503, empty — try next model
            return None, None

    # Try primary model, then fallbacks
    models_to_try = [model_id] + (
        [m for m in _OPENROUTER_FREE_FALLBACKS if m != model_id]
        if _is_openrouter(api_key) else []
    )

    last_error = None
    for attempt_model in models_to_try:
        try:
            result, fatal_err = _call_model(attempt_model)
        except Exception as e:
            last_error = str(e)
            continue

        if fatal_err:
            return fatal_err          # Auth or billing — stop immediately
        if result:
            return result             # ✅ Got a real answer
        # else: None — try next model

    # All models failed or returned empty
    return (
        "⚠️ The AI assistant is temporarily unavailable. "
        "All free-tier models are busy or returning empty responses right now. "
        "Please wait **30 seconds** and try again, or rephrase your question."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
    # Greeting from the assistant on first load
    welcome = (
        "**Hello! I'm the SepsiArc Medical AI Assistant.** 👋\n\n"
        "I can help you understand:\n"
        "- 🦠 What is sepsis, septic shock, SIRS, and related conditions\n"
        "- 📋 Clinical scoring systems — qSOFA, SOFA, Lactate, Shock Index\n"
        "- 💊 Treatment principles — the Hour-1 Bundle, vasopressors, antibiotics\n"
        "- 🧠 Post-Sepsis Syndrome and recovery\n"
        "- 🤖 How SepsiArc's Stage 1 and Stage 2 models work\n"
        "- 📊 What SHAP explanations mean and how to interpret them\n"
        "- 📈 What probabilities mean and the nature of false positives/negatives\n\n"
        "**⚕ Important:** I provide **educational information only** and am not a substitute "
        "for clinical judgment. For sepsis assessment of a real patient, use the "
        "**Stage 1 Detection** page.\n\n"
        "*Click a Quick Topic below or type your question to get started.*"
    )
    st.session_state.chat_messages.append({"role": "assistant", "content": welcome})

# Track the last FAQ question processed to prevent double-processing on reruns
if "last_faq_processed" not in st.session_state:
    st.session_state.last_faq_processed = None

# ══════════════════════════════════════════════════════════════════════════════
# INFO PANEL
# ══════════════════════════════════════════════════════════════════════════════
api_key_set, model_id_set = _load_secrets()

if not api_key_set or not model_id_set:
    st.warning(
        "⚙️ **AI assistant not yet configured.** "
        "Set `AIMLAPI_API_KEY` and `AIMLAPI_MODEL` in `.streamlit/secrets.toml` "
        "(local) or in the Streamlit Cloud app Secrets. "
        "The assistant will show configuration instructions in the chat until this is done.",
        icon="⚠️",
    )
else:
    provider = "OpenRouter" if (api_key_set and api_key_set.startswith("sk-or-")) else "AIMLAPI"
    st.success(
        f"✅ AI assistant connected — model: `{model_id_set}` via {provider}.",
        icon="🤖",
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# QUICK TOPIC BUTTONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("##### 💬 Quick Topics — click to ask:")

# Preserved topic labels and their prompt text (sent to AI, not hard-coded answers)
FAQS: list[tuple[str, str]] = [
    ("What is Sepsis?",           "What is sepsis? Please explain the definition, causes, and why it is dangerous."),
    ("Septic Shock",              "What is septic shock? How does it differ from sepsis and what is the treatment?"),
    ("SIRS Criteria",             "What is SIRS? Explain the four SIRS criteria and their clinical significance."),
    ("qSOFA Score",               "What is qSOFA? Explain the three criteria and how to interpret the score."),
    ("Lactate Levels",            "Explain serum lactate levels in sepsis — what the values mean and how they guide treatment."),
    ("Shock Index",               "What is the Shock Index? How is it calculated and what values indicate concern?"),
    ("Post-Sepsis Syndrome",      "What is Post-Sepsis Syndrome? What are the long-term effects on survivors?"),
    ("SHAP Explanations",         "What is SHAP in machine learning? How should I interpret the SHAP chart in SepsiArc?"),
    ("How Stage 1 Works",         "How does Stage 1 of SepsiArc work? Explain the model, features, threshold, and output."),
    ("How Stage 2 Works",         "How does Stage 2 of SepsiArc work? What model is used and what does the output mean?"),
    ("What Probability Means",    "What does the sepsis probability percentage in SepsiArc actually mean?"),
    ("False Positives/Negatives", "Why do false positives and false negatives occur in SepsiArc's predictions?"),
]

faq_row1 = st.columns(6)
faq_row2 = st.columns(6)
all_faq_cols = list(faq_row1) + list(faq_row2)

for i, (label, prompt_text) in enumerate(FAQS):
    if i < len(all_faq_cols):
        if all_faq_cols[i].button(label, key=f"faq_{i}", use_container_width=True):
            # Prevent processing the same FAQ twice on Streamlit reruns
            faq_key = f"faq_{i}_{label}"
            if st.session_state.last_faq_processed != faq_key:
                st.session_state.last_faq_processed = faq_key
                st.session_state.chat_messages.append({"role": "user", "content": label})
                with st.spinner("Thinking…"):
                    reply = get_ai_response(
                        label,
                        st.session_state.chat_messages[:-1],  # history before this user msg
                    )
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            st.rerun()

st.markdown("---")

# ── Sidebar: clear chat ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.last_faq_processed = None
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# CHAT DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("#### 💬 Chat")
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT
# ══════════════════════════════════════════════════════════════════════════════
if prompt := st.chat_input("Ask about sepsis, SHAP, SOFA, how SepsiArc works…"):
    if prompt.strip():
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt.strip()})
        # Show it immediately
        with st.chat_message("user"):
            st.markdown(prompt.strip())
        # Get AI response with spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = get_ai_response(
                    prompt.strip(),
                    st.session_state.chat_messages[:-1],  # history before current user msg
                )
            st.markdown(reply)
        # Persist response
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        # Reset FAQ dedup key so same FAQ can be asked again after a manual question
        st.session_state.last_faq_processed = None

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "⚕ SepsiArc Medical AI Assistant provides educational information only. "
    "Responses are AI-generated and may contain inaccuracies. "
    "This assistant does not make Stage 1 or Stage 2 predictions and must not be used as a "
    "substitute for clinical assessment by a qualified healthcare professional. "
    "Sources: Surviving Sepsis Campaign (2021), Sepsis-3 Consensus Definitions (2016), "
    "PhysioNet (Reyna et al., 2019). Powered by AIMLAPI."
)
render_footer()
