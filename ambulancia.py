import os
import json
import tempfile
import time

import streamlit as st
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

st.title("Neurologický AI Asistent – prototyp 0.6")


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


with st.expander("Vlastné predlohy", expanded=False):
    st.write("Tu si môžete pridať vlastnú predlohu výstupu.")

    new_template_name = st.text_input("Názov novej predlohy")

    new_template_body = st.text_area(
        "Text predlohy",
        height=220,
        placeholder="Napríklad:\n\nAnamnéza:\n...\n\nObjektívne neurologicky:\n...\n\nDoplňujúce vyšetrenia:\n..."
    )

    if st.button("Pridať predlohu"):
        if not new_template_name.strip():
            st.error("Zadajte názov predlohy.")
        elif not new_template_body.strip():
            st.error("Zadajte text predlohy.")
        else:
            st.session_state.custom_templates[new_template_name.strip()] = new_template_body.strip()
            st.success(f"Predloha pridaná: {new_template_name.strip()}")
            st.rerun()

    if st.session_state.custom_templates:
        st.write("Aktuálne vlastné predlohy:")

        for name in st.session_state.custom_templates:
            st.write(f"- {name}")

        templates_json = json.dumps(
            st.session_state.custom_templates,
            ensure_ascii=False,
            indent=2
        )

        st.download_button(
            label="Stiahnuť moje predlohy ako JSON",
            data=templates_json,
            file_name="moje_predlohy.json",
            mime="application/json"
        )

    uploaded_templates = st.file_uploader(
        "Nahrať uložené predlohy JSON",
        type=["json"]
    )

    if uploaded_templates is not None:
        try:
            loaded = json.load(uploaded_templates)

            if not isinstance(loaded, dict):
                st.error("JSON musí obsahovať objekt: názov predlohy → text predlohy.")
            else:
                st.session_state.custom_templates.update(loaded)
                st.success("Predlohy boli nahraté.")
                st.rerun()

        except Exception as e:
            st.error(f"Nepodarilo sa načítať JSON: {e}")


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


def build_prompt(
    template_name: str,
    template_text: str,
    mode: str,
    manual_context: str = ""
) -> str:
    return f"""
Si asistent neurológa v slovenskej ambulancii / nemocnici.

Cieľ:
- Pomôcť lekárovi zdokumentovať vyšetrenie.
- Výstup má byť praktický, stručný a pripravený na copy-paste do MIS.
- Lekár zostáva plne zodpovedný za finálnu kontrolu.

Typ vyšetrenia / predloha:
{template_name}

Použitá predloha hlavného výstupu:
{template_text}

Režim:
{mode}

Manuálne doplnený kontext / poznámky:
{manual_context}

Najprv uveď HLAVNÝ VÝSTUP NA KOPÍROVANIE DO MIS.
Až potom uveď krátku kontrolnú časť.

============================================================
HLAVNÝ VÝSTUP NA KOPÍROVANIE DO MIS
============================================================

V tejto hlavnej časti použi predlohu vyššie.

Do hlavného výstupu zahrň iba:
- anamnézu,
- subjektívne ťažkosti,
- objektívny / neurologický nález,
- výsledky laboratórnych, zobrazovacích alebo iných doplňujúcich vyšetrení, ak boli uvedené.

V hlavnom výstupe NEUVÁDZAJ:
- diagnózu,
- diferenciálnu diagnostiku,
- odporúčania,
- plán,
- liečbu,
- návrhy ďalších vyšetrení,
pokiaľ nie sú priamo súčasťou vloženej predlohy ako samostatné sekcie. Ak sú v predlohe, ale neboli explicitne uvedené, napíš „Na doplnenie lekárom.“

FORMÁTOVANIE HLAVNÉHO VÝSTUPU:
- Každý nadpis predlohy musí byť na samostatnom riadku.
- Text pod nadpisom musí byť na novom riadku.
- Medzi sekciami nechaj jeden prázdny riadok.
- Nespájaj viacero hlavičiek do jedného odseku.
- Nepíš zbytočné vysvetľujúce formulácie.
- Zachovaj slovenský neurologický štýl UNB/Kramáre.

PRÍSNE PRAVIDLÁ:
- Nevymýšľaj si fakty.
- Nevymýšľaj si diagnózy.
- Nevymýšľaj si lieky ani dávkovanie.
- Nevymýšľaj si normálny neurologický nález, ak nebol uvedený.
- Nevymýšľaj si odporúčania, MR, CT, laboratóriá, liečbu ani kontroly, ak ich lekár nepovedal.
- Zachovaj negácie.
- Nezamieňaj subjektívne ťažkosti za objektívny nález.
- Nepridávaj časti vyšetrenia, ktoré neboli uvedené.
- Ak informácia chýba, napíš „neudané“ alebo sekciu ponechaj stručnú.

============================================================
KONTROLNÁ ČASŤ PRE LEKÁRA
============================================================

Túto časť daj až za hlavný výstup a urob ju krátku.

Uveď maximálne tieto podsekcie:

1. Krátky technický súhrn:
- zdroj: audio / manuálny text / audio + manuálny text,
- pravdepodobné role hovoriacich: lekár / pacient / príbuzný, ak sa dali určiť.

2. Diagnóza / záver:
- iba ak bol explicitne uvedený lekárom alebo jasne vyplýval z diktátu lekára,
- ak nebol uvedený, napíš: „Na doplnenie lekárom.“

3. Odporúčanie / plán:
- iba ak bolo explicitne uvedené lekárom,
- ak nebolo uvedené, napíš: „Na doplnenie lekárom.“

4. Neistoty:
- uveď iba 1–4 krátke body,
- čo bolo zle počuteľné,
- čo chýba,
- čo musí lekár skontrolovať.

V kontrolnej časti nenavrhuj nové diagnózy, vyšetrenia ani liečbu, ak neboli uvedené v zdrojovom texte.
"""


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

            st.success(f"Spracovanie dokončené. Použitý model: {used_model}")

            if fallback_errors:
                with st.expander("Fallback pokusy / chyby modelov"):
                    for err in fallback_errors:
                        st.text(err)

            output_text = response.text

            st.subheader("Výstup")
            st.markdown(output_text)

            st.download_button(
                label="Stiahnuť výstup ako TXT",
                data=output_text,
                file_name="neurologicka_sprava.txt",
                mime="text/plain"
            )

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


with st.expander("Technické info"):
    st.write("Vybraný model:", model)
    st.write("Predvolený model:", default_model)
    st.write("Počet dostupných modelov:", len(available_models))
