import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from google import genai

# ============================================================
# KONFIGURACIA
# ============================================================

load_dotenv()

def get_api_key():
    # 1) Streamlit Cloud secrets
    try:
        key = st.secrets.get("GEMINI_API_KEY", None)
        if key:
            return key
    except Exception:
        pass

    # 2) Local .env fallback
    return os.getenv("GEMINI_API_KEY")

GEMINI_API_KEY = get_api_key()

if not GEMINI_API_KEY:
    st.error("Chyba GEMINI_API_KEY. Lokalne pouzi .env, na Streamlit Cloud pouzi Settings -> Secrets.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

st.set_page_config(
    page_title="Neuro AI Asistent",
    layout="wide",
    page_icon="🧠"
)

st.title("Neurologicky AI Asistent - web prototyp")
st.caption("Pracovny prototyp pre anonymizovane testovanie. Nepouzivat na identifikovatelne realne pacientskke data bez pravneho/GDPR nastavenia.")

# ============================================================
# MODEL HANDLING
# ============================================================

PREFERRED_MODELS = [
    "models/gemini-flash-latest",
    "models/gemini-3.6-flash",
    "models/gemini-3.5-flash",
    "models/gemini-3.1-flash-lite",
    "models/gemini-3.1-flash-lite-preview",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-001",
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.0-flash-lite-001",
    "models/gemini-pro-latest",
    "models/gemini-3-pro-preview",
]

@st.cache_data(show_spinner=False, ttl=3600)
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

    raise RuntimeError("Nepodarilo sa pouzit ziadny dostupny model.\n\n" + "\n\n".join(errors))

available_models, model_error = get_available_models()

if model_error:
    st.error(f"Nepodarilo sa nacitat dostupne modely: {model_error}")
    st.stop()

if not available_models:
    st.error("Pre tento API kluc sa nenasiel ziadny model podporujuci generateContent.")
    st.stop()

default_model = choose_model(available_models)

if not default_model:
    st.error("Nepodarilo sa vybrat vhodny model.")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Nastavenie pripadu")

rezim = st.sidebar.radio(
    "Rezim nahravania:",
    ["Rozhovor s pacientom", "Diktat lekara"]
)

typ = st.sidebar.selectbox(
    "Typ vysetrenia:",
    ["Ambulancia - vstupne", "Ambulancia - kontrolne", "CPO konzilium", "Prijem na hospitalizaciu"]
)

model = st.sidebar.selectbox(
    "Model:",
    available_models,
    index=available_models.index(default_model)
)

st.sidebar.markdown("---")
st.sidebar.warning(
    "Webova verzia je vhodna na anonymne/modelove testy. Pri mobile nechaj obrazovku zapnutu pocas celeho nahravania."
)

with st.sidebar.expander("Dostupne modely"):
    for m in available_models:
        st.write(m)

# ============================================================
# SUHLAS
# ============================================================

st.subheader("1. Suhlas pacienta")

consent = st.radio(
    "Stav suhlasu:",
    ["Pacient podpisal / potvrdil suhlas", "Bez nahravania"]
)

if consent == "Bez nahravania":
    st.warning("Rezim bez nahravania: mozete manualne vlozit text alebo diktat nizsie.")

# ============================================================
# VSTUP
# ============================================================

st.subheader("2. Nahravka alebo manualny vstup")
st.info("Ak pouzivas mobil/tablet, pocas nahravania nezamykaj obrazovku a neprepínaj aplikacie.")

audio_data = None

if consent == "Pacient podpisal / potvrdil suhlas":
    audio_data = st.audio_input("Nahrajte vysetrenie / diktat")

    if audio_data:
        st.audio(audio_data)

manual_text = st.text_area(
    "Volitelne: manualne vlozeny transkript alebo poznamky lekara",
    height=220,
    placeholder="Sem mozete vlozit transkript, diktat alebo poznamky lekara."
)

# ============================================================
# PROMPT
# ============================================================

def build_prompt(document_type: str, mode: str, manual_context: str = "") -> str:
    return f"""
Si asistent neurologa v slovenskej ambulancii / nemocnici.

Kontext:
- Typ dokumentu: {document_type}
- Rezim: {mode}
- Ciel: pomoct lekarovi zdokumentovat vysetrenie v slovenskom neurologickom style UNB/Kramare.
- Vystup je len pracovny text na kontrolu lekarom.
- Lekar zostava plne zodpovedny za finalnu spravu.

Ak je k dispozicii audio, najprv ho prepis.
Ak je k dispozicii manualny text, pouzi ho ako zdroj informacii.
Ak je k dispozicii audio aj manualny text, pouzi manualny text ako doplnkovy kontext, nie ako nahradu audia.

Manualne doplneny kontext / poznamky:
{manual_context}

POSTUPUJ PRESNE V TYCHTO KROKOCH:

KROK 1 - TRANSKRIPT:
- Vytvor co najpresnejsi slovensky transkript.
- Ak ide iba o manualny text bez audia, napis, ze audio nebolo poskytnute a pracujes s manualnym textom.
- Ak ide o diktat lekara, oznac hovoriaceho ako Lekar / diktat.
- Ak ide o rozhovor, rozdel hovoriacich ako Speaker 1, Speaker 2, Speaker 3.
- Ak vies, dopln priblizne casove useky.
- Nesumarizuj v tomto kroku.

KROK 2 - INFERENCIA ROLI:
- Urci, kto je pravdepodobne Lekar, Pacient, Pribuzny alebo Ina osoba.
- Uved mieru istoty: vysoka / stredna / nizka.
- Kratko vysvetli preco.
- Ak ide iba o diktat lekara alebo manualne poznamky, napis to jasne.

KROK 3 - KLINICKE FAKTY:
Vypis iba fakty, ktore boli uvedene v audiu alebo manualnom texte:
- hlavny dovod vysetrenia,
- zaciatok a trvanie tazkosti,
- priebeh,
- sprievodne priznaky,
- negovane priznaky,
- osobna anamneza,
- liekova anamneza,
- alergie,
- rodinna anamneza,
- objektivny neurologicky nalez, iba ak bol uvedeny alebo nadiktovany,
- zaver, iba ak bol explicitne povedany alebo jasne vyplyva z diktatu lekara,
- odporucanie, iba ak bolo explicitne povedane.

KROK 4 - LEKARSKA SPRAVA:
Vygeneruj odbornu neurologicku spravu v slovencine.

Styl:
- UNB/Kramare,
- vecny,
- klinicky prirodzeny,
- pripraveny na copy-paste do MIS,
- bez vysvetlovania pacientovi,
- bez uvah, ktore neodzneli.

FORMATOVANIE:
- Kazdy nadpis musi byt na samostatnom riadku.
- Za kazdym nadpisom nasleduje text na novom riadku.
- Medzi sekciami nechaj jeden prazdny riadok.
- Nespajaj viacero hlaviciek do jedneho odseku.
- Nepouzivaj odrazky v samotnej lekarskej sprave, ak to nie je nevyhnutne.
- Vystup ma byt vhodny na priame skopirovanie do ambulantneho informacneho systemu.

Ak typ dokumentu je ambulantne vstupne, ambulantne kontrolne alebo CPO konzilium, pouzi presne tieto hlavicky:

Indikacia:
TO:
OA:
LA:
AA:
RA:
Objektivne neurologicky:
Zaver:
Odporucanie:

Ak typ dokumentu je Prijem na hospitalizaciu, pouzi presne tieto hlavicky:

Dovod prijatia:
TO:
OA:
LA:
AA:
RA:
Objektivne neurologicky:
Zaver:
Plan:

PRISNE PRAVIDLA:
- Nevymyslaj si fakty.
- Nevymyslaj si diagnozy.
- Nevymyslaj si lieky ani davkovanie.
- Nevymyslaj si normalny neurologicky nalez, ak nebol uvedeny.
- Nevymyslaj si odporucania, vysetrenia, MR, CT, laboratoria, liecbu ani kontroly, ak ich lekar nepovedal.
- Zachovaj negacie.
- Nezamienaj subjektivne tazkosti pacienta za objektivny nalez.
- Nepridavaj anatomicke alebo systemove casti vysetrenia, ktore neboli uvedene.
- Nepridavaj formulacie typu "treba zvazit MR/CT", ak to lekar nepovedal.
- Ak informacia chyba, napis "neudane" alebo sekciu ponechaj strucnu.
- Nadpis musi byt "Zaver:", nie "Zaver na doplnenie lekarom:".
- Ak odporucanie nebolo povedane, v sekcii Odporucanie napis presne: "Na doplnenie lekarom."
- Ak plan pri hospitalizacii nebol povedany, v sekcii Plan napis presne: "Na doplnenie lekarom."

KROK 5 - KRITICKA KONTROLA:
Na konci uved kratku sekciu:
"Kontrola kvality / neistoty:"

Uved iba:
- co bolo zle pocutelne alebo chybalo,
- kde je neista identifikacia hovoriaceho,
- ktore medicinske vyrazy si vyzaduju kontrolu,
- ktore casti spravy musi lekar overit.

V kritickej kontrole nenavrhuj ziadne nove diagnozy, vysetrenia ani liecbu, ak neboli uvedene v zdrojovom texte.
"""

# ============================================================
# SPRACOVANIE
# ============================================================

st.subheader("3. Spracovanie")

process_clicked = st.button("Spracovat vysetrenie")

if process_clicked:
    if not audio_data and not manual_text.strip():
        st.error("Nahrajte audio alebo vlozte manualny text.")
        st.stop()

    tmp_path = None
    uploaded_file = None

    with st.spinner("Spracovavam vstup a generujem vystup..."):
        try:
            prompt = build_prompt(
                document_type=typ,
                mode=rezim,
                manual_context=manual_text
            )

            contents = []

            if audio_data:
                # Streamlit audio_input vracia zvukovy subor.
                # Ulozime ho len docasne a po spracovani ho vymazeme.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_data.read())
                    tmp_path = tmp_file.name

                uploaded_file = client.files.upload(file=tmp_path)
                time.sleep(2)
                contents.append(uploaded_file)

            contents.append(prompt)

            response, used_model, fallback_errors = generate_with_fallback(
                contents=contents,
                selected_model=model,
                available_models=available_models
            )

            st.success(f"Spracovanie dokoncene. Pouzity model: {used_model}")

            if fallback_errors:
                with st.expander("Fallback pokusy / chyby modelov"):
                    for err in fallback_errors:
                        st.text(err)

            st.subheader("Vystup")
            st.markdown(response.text)

            st.download_button(
                label="Stiahnut vystup ako TXT",
                data=response.text,
                file_name="neurologicka_sprava.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Nastala chyba pri spracovani: {e}")

        finally:
            if uploaded_file:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

# ============================================================
# TECHNICKE INFO
# ============================================================

with st.expander("Technicke info"):
    st.write("Vybrany model:", model)
    st.write("Predvoleny model:", default_model)
    st.write("Pocet dostupnych modelov:", len(available_models))
