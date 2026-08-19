import os
import json
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from google import genai


# ============================================================
# KONFIGURÁCIA
# ============================================================

load_dotenv()


def get_gemini_api_key():
    """
    Lokálne číta z .env, na Streamlit Cloud zo Secrets.
    """
    key = os.getenv("GEMINI_API_KEY")

    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            key = None

    return key


GEMINI_API_KEY = get_gemini_api_key()

if not GEMINI_API_KEY:
    st.error("Chýba GEMINI_API_KEY. Lokálne ho dajte do .env, na Streamlit Cloud do Secrets.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

st.set_page_config(
    page_title="Neuro AI Asistent",
    layout="wide"
)

st.title("Neurologický AI Asistent – prototyp 0.6")


# ============================================================
# MODEL HANDLING
# ============================================================

PREFERRED_MODELS = [
    "models/gemini-flash-latest",
    "models/gemini-3.6-flash",
    "models/gemini-3.5-flash",
    "models/gemini-3.1-flash-lite",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-001",
    "models/gemini-flash-lite-latest",
    "models/gemini-pro-latest",
    "models/gemini-3-pro-preview",
]


@st.cache_data(show_spinner=False)
def get_available_models():
    available = []

    try:
        for m in client.models.list():
            name = getattr(m, "name", "")
            actions = getattr(m, "supported_actions", []) or []

            if "generateContent" in actions:
                available.append(name)

    except Exception as e:
        return [], str(e)

    return available, None


def choose_model(available_models):
    for model_name in PREFERRED_MODELS:
        if model_name in available_models:
            return model_name

    if available_models:
        return available_models[0]

    return None


def generate_with_fallback(contents, selected_model, available_models):
    models_to_try = []

    if selected_model and selected_model not in models_to_try:
        models_to_try.append(selected_model)

    for model_name in PREFERRED_MODELS:
        if model_name in available_models and model_name not in models_to_try:
            models_to_try.append(model_name)

    for model_name in available_models:
        if model_name not in models_to_try:
            models_to_try.append(model_name)

    errors = []

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents
            )
            return response, model_name, errors

        except Exception as e:
            errors.append(f"{model_name}: {e}")

    raise RuntimeError(
        "Nepodarilo sa použiť žiadny dostupný model.\n\n"
        + "\n\n".join(errors)
    )


available_models, model_error = get_available_models()

if model_error:
    st.error(f"Nepodarilo sa načítať dostupné modely: {model_error}")
    st.stop()

if not available_models:
    st.error("Pre tento API kľúč sa nenašiel žiadny model podporujúci generateContent.")
    st.stop()

default_model = choose_model(available_models)

if not default_model:
    st.error("Nepodarilo sa vybrať vhodný model.")
    st.stop()


# ============================================================
# PREDLOHY
# ============================================================

BUILT_IN_TEMPLATES = {
    "Ambulancia - vstupné": """
Anamnéza:
[stručne anamnestické údaje relevantné k vyšetreniu, vrátane TO, OA, LA, AA, RA podľa dostupnosti]

Objektívne neurologicky:
[objektívny neurologický nález, iba ak bol uvedený alebo nadiktovaný]

Doplňujúce vyšetrenia:
[laboratórne výsledky, CT/MR/RTG/EEG/EMG a iné výsledky iba ak boli uvedené]
""",
    "Ambulancia - kontrolné": """
Subjektívne od poslednej kontroly:
[aktuálny stav, vývoj ťažkostí, tolerancia liečby, nové príznaky]

Objektívne neurologicky:
[objektívny neurologický nález, iba ak bol uvedený alebo nadiktovaný]

Doplňujúce vyšetrenia:
[laboratórne výsledky, CT/MR/RTG/EEG/EMG a iné výsledky iba ak boli uvedené]
""",
    "CPO konzílium": """
Anamnéza aktuálnych ťažkostí:
[akútne ťažkosti, čas vzniku, priebeh, pozitívne a negované príznaky]

Objektívne neurologicky:
[objektívny neurologický nález, iba ak bol uvedený alebo nadiktovaný]

Doplňujúce vyšetrenia:
[laboratórne výsledky, CT/MR/RTG/EEG/EMG a iné výsledky iba ak boli uvedené]
""",
    "Príjem na hospitalizáciu": """
Príjmová anamnéza:
[dôvod prijatia a relevantná anamnéza]

Objektívne neurologicky:
[objektívny neurologický nález, iba ak bol uvedený alebo nadiktovaný]

Doplňujúce vyšetrenia:
[laboratórne výsledky, CT/MR/RTG/EEG/EMG a iné výsledky iba ak boli uvedené]
"""
}

if "custom_templates" not in st.session_state:
    st.session_state.custom_templates = {}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Nastavenie prípadu")

rezim = st.sidebar.radio(
    "Režim:",
    [
        "Rozhovor s pacientom",
        "Diktát lekára"
    ]
)

all_template_names = list(BUILT_IN_TEMPLATES.keys()) + list(st.session_state.custom_templates.keys())

selected_template_name = st.sidebar.selectbox(
    "Typ vyšetrenia / predloha:",
    all_template_names
)

if selected_template_name in BUILT_IN_TEMPLATES:
    selected_template_text = BUILT_IN_TEMPLATES[selected_template_name]
else:
    selected_template_text = st.session_state.custom_templates[selected_template_name]

model = st.sidebar.selectbox(
    "Model:",
    available_models,
    index=available_models.index(default_model)
)

st.sidebar.markdown("---")
st.sidebar.caption("Ak zvolený model zlyhá, aplikácia skúsi ďalšie dostupné modely.")


# ============================================================
# SPRÁVA PREDLÔH
# ============================================================

with st.expander("Vlastné predlohy", expanded=False):
    st.markdown(
        """
Tu si môžete pridať vlastnú predlohu.  
Predloha má byť štruktúra výstupu, napríklad:
