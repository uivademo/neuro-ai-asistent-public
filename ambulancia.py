import os
import json
import html
import tempfile
import time

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from google import genai


load_dotenv()


def get_gemini_api_key():
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

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1150px;
    }
    h1 {
        font-size: 2rem !important;
        font-weight: 650 !important;
    }
    h2 {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
    }
    h3 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    .stTextArea textarea {
        font-size: 15px !important;
        line-height: 1.45 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Neurologický AI Asistent")


# ============================================================
# MODELY
# ============================================================

PREFERRED_MODELS = [
    "models/gemini-flash-latest",
    "models/gemini-3.6-flash",
    "models/gemini-3.5-flash",
    "models/gemini-pro-latest",
    "models/gemini-3-pro-preview",
    "models/gemini-flash-lite-latest",
    "models/gemini-3.1-flash-lite",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-001",
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

DEFAULT_TEMPLATES = {
    "Ambulancia - vstupné": """Indikácia:

TO:

OA:

LA:

AA:

RA:

Objektívne neurologicky:

Doplňujúce vyšetrenia:

Záver:

Odporúčanie:
""",
    "Ambulancia - kontrolné": """Subjektívne od poslednej kontroly:

Aktuálne ťažkosti:

Liečba a tolerancia:

Objektívne neurologicky:

Doplňujúce vyšetrenia:

Záver:

Odporúčanie:
""",
    "CPO konzílium": """Indikácia:

TO:

OA:

LA:

AA:

Objektívne neurologicky:

Doplňujúce vyšetrenia:

Záver:

Odporúčanie:
""",
    "Príjem na hospitalizáciu": """Dôvod prijatia:

TO:

OA:

LA:

AA:

RA:

Objektívne neurologicky:

Doplňujúce vyšetrenia:

Záver:

Plán:
"""
}


if "templates" not in st.session_state:
    st.session_state.templates = DEFAULT_TEMPLATES.copy()

if "last_output" not in st.session_state:
    st.session_state.last_output = ""

if "last_main_output" not in st.session_state:
    st.session_state.last_main_output = ""

if "last_detail_output" not in st.session_state:
    st.session_state.last_detail_output = ""


def extract_between_markers(text, start_marker, end_marker=None):
    if not text:
        return ""

    if start_marker not in text:
        return ""

    part = text.split(start_marker, 1)[1]

    if end_marker and end_marker in part:
        part = part.split(end_marker, 1)[0]

    return part.strip()


def copy_button(text_to_copy, button_text="Kopírovať hlavný výstup"):
    safe_text = json.dumps(text_to_copy)

    components.html(
        f"""
        <button
            onclick='navigator.clipboard.writeText({safe_text}).then(() => {{
                const el = document.getElementById("copy-status");
                el.innerText = "Skopírované";
                setTimeout(() => el.innerText = "", 2000);
            }})'
            style="
                background-color: #0f6fff;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 0.55rem 0.9rem;
                font-size: 0.95rem;
                cursor: pointer;
            "
        >
            {html.escape(button_text)}
        </button>
        <span id="copy-status" style="margin-left: 12px; color: green; font-size: 0.95rem;"></span>
        """,
        height=55
    )


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

template_names = list(st.session_state.templates.keys())

selected_template_name = st.sidebar.selectbox(
    "Typ vyšetrenia / predloha:",
    template_names
)

model = st.sidebar.selectbox(
    "Model:",
    available_models,
    index=available_models.index(default_model)
)

st.sidebar.caption(
    "Defaultne sa preferuje gemini-flash-latest. Lite modely sú rýchlejšie, ale horšie klinicky upravujú text."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Predlohy")

template_action = st.sidebar.radio(
    "Akcia:",
    [
        "Pozrieť / upraviť predlohu",
        "Pridať novú predlohu",
        "Import / export predlôh"
    ]
)

if template_action == "Pozrieť / upraviť predlohu":
    st.sidebar.write(f"Aktuálna predloha: {selected_template_name}")

    edited_template = st.sidebar.text_area(
        "Text predlohy:",
        value=st.session_state.templates[selected_template_name],
        height=340
    )

    if st.sidebar.button("Uložiť úpravu predlohy"):
        st.session_state.templates[selected_template_name] = edited_template
        st.sidebar.success("Predloha uložená.")
        st.rerun()

    if selected_template_name not in DEFAULT_TEMPLATES:
        if st.sidebar.button("Vymazať túto vlastnú predlohu"):
            del st.session_state.templates[selected_template_name]
            st.sidebar.success("Predloha vymazaná.")
            st.rerun()

if template_action == "Pridať novú predlohu":
    new_template_name = st.sidebar.text_input("Názov novej predlohy")

    new_template_body = st.sidebar.text_area(
        "Text novej predlohy:",
        height=340,
        placeholder=(
            "Napríklad:\n\n"
            "Anamnéza:\n\n"
            "Objektívne neurologicky:\n\n"
            "Doplňujúce vyšetrenia:\n\n"
            "Záver:\n\n"
            "Odporúčanie:"
        )
    )

    if st.sidebar.button("Vytvoriť novú predlohu"):
        name = new_template_name.strip()
        body = new_template_body.strip()

        if not name:
            st.sidebar.error("Zadajte názov predlohy.")
        elif not body:
            st.sidebar.error("Zadajte text predlohy.")
        elif name in st.session_state.templates:
            st.sidebar.error("Predloha s týmto názvom už existuje.")
        else:
            st.session_state.templates[name] = body
            st.sidebar.success(f"Predloha vytvorená: {name}")
            st.rerun()

if template_action == "Import / export predlôh":
    templates_json = json.dumps(
        st.session_state.templates,
        ensure_ascii=False,
        indent=2
    )

    st.sidebar.download_button(
        label="Stiahnuť predlohy ako JSON",
        data=templates_json,
        file_name="predlohy_neuro_ai.json",
        mime="application/json"
    )

    uploaded_templates = st.sidebar.file_uploader(
        "Nahrať predlohy JSON",
        type=["json"]
    )

    if uploaded_templates is not None:
        try:
            loaded = json.load(uploaded_templates)

            if not isinstance(loaded, dict):
                st.sidebar.error("JSON musí byť objekt: názov predlohy → text predlohy.")
            else:
                cleaned = {}
                for key, value in loaded.items():
                    if isinstance(key, str) and isinstance(value, str):
                        cleaned[key] = value

                if not cleaned:
                    st.sidebar.error("V JSON sa nenašli použiteľné predlohy.")
                else:
                    st.session_state.templates.update(cleaned)
                    st.sidebar.success("Predlohy boli nahraté.")
                    st.rerun()

        except Exception as e:
            st.sidebar.error(f"Nepodarilo sa načítať JSON: {e}")


selected_template_text = st.session_state.templates[selected_template_name]


# ============================================================
# HLAVNÉ UI
# ============================================================

st.subheader("1. Súhlas pacienta")

consent = st.radio(
    "Stav súhlasu:",
    [
        "Pacient podpísal / potvrdil súhlas",
        "Bez nahrávania"
    ]
)

if consent == "Bez nahrávania":
    st.warning("Režim bez nahrávania: môžete manuálne vložiť text alebo diktát nižšie.")


st.subheader("2. Nahrávka alebo manuálny vstup")

audio_data = None

if consent == "Pacient podpísal / potvrdil súhlas":
    audio_data = st.audio_input("Nahrajte vyšetrenie / diktát")

    if audio_data:
        st.audio(audio_data)

manual_text = st.text_area(
    "Voliteľne: manuálne vložený transkript alebo poznámky lekára",
    height=180,
    placeholder="Sem môžete vložiť transkript, diktát alebo poznámky lekára."
)


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    template_name: str,
    template_text: str,
    mode: str,
    manual_context: str = ""
) -> str:
    return f"""
Si skúsený slovenský neurologický klinický asistent.

Tvoj cieľ:
- Premeň neupravený slovenský lekársky diktát alebo rozhovor lekár-pacient na kvalitnú neurologickú dokumentáciu.
- Výstup musí byť klinicky prirodzený, nie doslovný prepis.
- Výstup má byť vhodný na copy-paste do slovenského nemocničného / ambulantného informačného systému.
- Štýl: vecný, odborný, slovenský, UNB/Kramáre.
- Lekár zostáva plne zodpovedný za finálnu kontrolu.

Typ vyšetrenia / použitá predloha:
{template_name}

Predloha hlavného výstupu:
{template_text}

Režim:
{mode}

Manuálne doplnený kontext / poznámky:
{manual_context}

DÔLEŽITÉ PORADIE:
Najprv musí byť hlavný klinický výstup.
Až potom detailná analýza.

Použi presne tieto značky:
<<<HLAVNY_VYSTUP>>>
[sem daj hlavný výstup]

<<<DETAILNA_ANALYZA>>>
[sem daj transkript, klinické fakty a kontrolu kvality]

============================================================
HLAVNÝ VÝSTUP
============================================================

Hlavný výstup nie je transkript.
Je to upravená lekárska správa.

V hlavnom výstupe:
- použi štruktúru predlohy,
- oprav diktát do štandardnej lekárskej formulácie,
- odstráň výplňové slová a preklepy diktátu,
- odstráň opravy typu „sori“, „teda“, „vlastne“, „ako keby“, ak nie sú medicínsky dôležité,
- neopisuj neistoty z diktátu do hlavného textu, ale daj ich do detailnej analýzy,
- zachovaj význam,
- zachovaj negácie,
- nepíš doslovný neupravený prepis,
- nerozdeľuj text mechanicky, ale klinicky logicky.

FORMÁTOVANIE:
- každý nadpis predlohy musí byť na samostatnom riadku,
- text pod nadpisom musí byť na novom riadku,
- medzi sekciami nechaj jeden prázdny riadok,
- nespájaj viacero hlavičiek do jedného odseku,
- nepoužívaj odrážky v hlavnej správe, ak to nie je nutné.

KLINICKÁ NORMALIZÁCIA:
- „lasek“, „Laség“, „Lasegue“ normalizuj ako „Lasègue“,
- „emery“, „MRka“, „emrika“ normalizuj ako „MR“ alebo „MR LS chrbtice“, ak z kontextu vyplýva LS chrbtica,
- „sendomotorický“ oprav na „senzomotorický“,
- „senzibilne vypadnutý dermatom“ uprav ako „senzitívny deficit / hypestézia v dermatóme ...“,
- „M35“, „M 3 5“, „3 z 5“ interpretuj ako svalová sila 3/5, ak kontext sedí,
- „skudexu“, „skúdeksu“, „Skudexu“ oprav ako „Skudexa“, ak ide o liek proti bolesti; dávku nechaj iba ak bola jasne uvedená,
- ak je výraz neistý, v hlavnom texte použi najpravdepodobnejšiu klinickú formuláciu a neistotu uveď dole v detailnej analýze.

PRAVIDLÁ PRE ZÁVER A ODPORÚČANIE:
- Ak lekár explicitne povedal záver, uveď ho v sekcii Záver.
- Ak lekár explicitne povedal odporúčanie alebo plán, uveď ho v sekcii Odporúčanie alebo Plán.
- Ak záver nebol explicitne uvedený, napíš v sekcii Záver: „Na doplnenie lekárom.“
- Ak odporúčanie/plán nebol explicitne uvedený, napíš v sekcii Odporúčanie alebo Plán: „Na doplnenie lekárom.“
- Nevymýšľaj nové diagnózy, vyšetrenia, liečbu ani kontroly.

PRÍSNE ZÁKAZY:
- Nevymýšľaj si fakty.
- Nevymýšľaj si lieky ani dávkovanie.
- Nevymýšľaj si normálny neurologický nález, ak nebol uvedený.
- Nevymýšľaj MR, CT, laboratóriá, obstreky, operáciu ani hospitalizáciu, ak neboli uvedené.
- Nezamieňaj subjektívne ťažkosti pacienta za objektívny nález.
- Nepridávaj časti neurologického vyšetrenia, ktoré neboli uvedené.
- Ak informácia chýba, napíš „neudané“ alebo „Na doplnenie lekárom“, podľa sekcie.

DÔLEŽITÉ PRI NEJASNOM DIKTÁTE:
- Ak lekár povie dve časové možnosti, napr. „dva dni, respektíve päť dní“, v hlavnom texte použi bezpečnú formuláciu „v posledných dňoch“ a neistotu vysvetli dole.
- Ak lekár povie nepresný alebo chaotický diktát, urob z neho klinicky čistý text.
- Ak sa objaví nesprávny prepis ako „traumatický príjem“, ale z kontextu ide o lumbosakrálnu radikulopatiu bez traumy, nepíš traumatický príjem.
- Pri lumbosakrálnej radikulopatii dávaj pozor na:
  - trvanie ťažkostí,
  - propagáciu bolesti,
  - sfinktery,
  - perianogenitálnu / intímnu citlivosť,
  - Lasègue,
  - motorický deficit,
  - senzitivitu v dermatóme,
  - indikáciu prijatia alebo MR, iba ak bola povedaná.

============================================================
DETAILNÁ ANALÝZA
============================================================

Detailnú analýzu daj až za hlavný výstup.

KROK 1 — TRANSKRIPT:
- Vytvor čo najpresnejší slovenský transkript.
- Ak ide iba o manuálny text bez audia, napíš, že audio nebolo poskytnuté a pracuješ s manuálnym textom.
- Ak ide o diktát lekára, označ hovoriaceho ako Lekár / diktát.
- Ak ide o rozhovor, rozdeľ hovoriacich ako Speaker 1, Speaker 2, Speaker 3.
- Ak vieš, doplň približné časové úseky.
- Tu môže byť doslovnejší prepis, ale hlavný výstup musí byť klinicky upravený.

KROK 2 — INFERENCIA ROLÍ:
- Urči, kto je pravdepodobne Lekár, Pacient, Príbuzný alebo Iná osoba.
- Uveď mieru istoty: vysoká / stredná / nízka.
- Krátko vysvetli prečo.

KROK 3 — KLINICKÉ FAKTY:
Vypíš fakty, ktoré boli uvedené:
- hlavný dôvod vyšetrenia,
- začiatok a trvanie ťažkostí,
- priebeh,
- sprievodné príznaky,
- negované príznaky,
- osobná anamnéza,
- lieková anamnéza,
- alergie,
- rodinná anamnéza,
- objektívny neurologický nález,
- výsledky laboratórnych, zobrazovacích alebo iných doplňujúcich vyšetrení,
- záver, ak bol explicitne uvedený,
- odporúčanie alebo plán, ak boli explicitne uvedené.

KROK 5 — KONTROLA KVALITY / NEISTOTY:
- čo bolo zle počuteľné alebo chýbalo,
- kde je neistá identifikácia hovoriaceho,
- ktoré medicínske výrazy si vyžadujú kontrolu,
- ktoré časti správy musí lekár overiť.
- Nenavrhuj nové diagnózy, vyšetrenia ani liečbu, ak neboli uvedené v zdrojovom texte.
"""


# ============================================================
# SPRACOVANIE
# ============================================================

st.subheader("3. Spracovanie")

process_clicked = st.button("Spracovať vyšetrenie")

if process_clicked:
    if not audio_data and not manual_text.strip():
        st.error("Nahrajte audio alebo vložte manuálny text.")
        st.stop()

    tmp_path = None
    uploaded_file = None

    with st.spinner("Spracovávam vstup a generujem výstup..."):
        try:
            prompt = build_prompt(
                template_name=selected_template_name,
                template_text=selected_template_text,
                mode=rezim,
                manual_context=manual_text
            )

            contents = []

            if audio_data:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_data.read())
                    tmp_path = tmp_file.name

                uploaded_file = client.files.upload(file=tmp_path)

                time.sleep(1)

                contents.append(uploaded_file)

            contents.append(prompt)

            response, used_model, fallback_errors = generate_with_fallback(
                contents=contents,
                selected_model=model,
                available_models=available_models
            )

            output_text = response.text or ""

            main_output = extract_between_markers(
                output_text,
                "<<<HLAVNY_VYSTUP>>>",
                "<<<DETAILNA_ANALYZA>>>"
            )

            detail_output = extract_between_markers(
                output_text,
                "<<<DETAILNA_ANALYZA>>>",
                None
            )

            if not main_output:
                main_output = output_text

            st.session_state.last_output = output_text
            st.session_state.last_main_output = main_output
            st.session_state.last_detail_output = detail_output

            st.success(f"Spracovanie dokončené. Použitý model: {used_model}")

            if fallback_errors:
                with st.expander("Fallback pokusy / chyby modelov"):
                    for err in fallback_errors:
                        st.text(err)

        except Exception as e:
            st.error(f"Nastala chyba pri spracovaní: {e}")

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


if st.session_state.last_main_output:
    st.subheader("4. Výstup na kopírovanie do MIS")

    st.text_area(
        "Hlavný výstup",
        value=st.session_state.last_main_output,
        height=360
    )

    copy_button(st.session_state.last_main_output)

    st.download_button(
        label="Stiahnuť hlavný výstup ako TXT",
        data=st.session_state.last_main_output,
        file_name="hlavny_vystup.txt",
        mime="text/plain"
    )

if st.session_state.last_detail_output:
    with st.expander("Detailná analýza, transkript a kontrola kvality", expanded=False):
        st.markdown(st.session_state.last_detail_output)

if st.session_state.last_output:
    with st.expander("Celý surový výstup modelu", expanded=False):
        st.markdown(st.session_state.last_output)

with st.expander("Technické info"):
    st.write("Vybraný model:", model)
    st.write("Predvolený model:", default_model)
    st.write("Počet dostupných modelov:", len(available_models))
