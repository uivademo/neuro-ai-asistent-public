import os
import json
import html
import re
import tempfile
import time

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from google import genai


# ---------- setup ----------
load_dotenv()

st.set_page_config(page_title="Neurologický AI Asistent", layout="wide")
st.title("Neurologický AI Asistent")

st.markdown("""
<style>
.block-container {padding-top: 1.7rem; padding-bottom: 2rem; max-width: 1180px;}
.stTextArea textarea {
    font-size: 15px !important;
    line-height: 1.45 !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace !important;
}
</style>
""", unsafe_allow_html=True)


def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            key = None
    return key


GEMINI_API_KEY = get_api_key()
if not GEMINI_API_KEY:
    st.error("Chýba GEMINI_API_KEY. Lokálne ho dajte do .env, na Streamlit Cloud do Secrets.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)


# ---------- models ----------
MODEL_PRESETS = {
    "Rýchle": [
        "models/gemini-flash-lite-latest",
        "models/gemini-3.1-flash-lite",
        "models/gemini-2.0-flash-lite",
        "models/gemini-flash-latest",
    ],
    "Vyvážené": [
        "models/gemini-flash-latest",
        "models/gemini-3.6-flash",
        "models/gemini-3.5-flash",
        "models/gemini-2.0-flash",
    ],
    "Presné": [
        "models/gemini-pro-latest",
        "models/gemini-3-pro-preview",
        "models/gemini-3.6-flash",
        "models/gemini-3.5-flash",
        "models/gemini-flash-latest",
    ],
}
ALL_MODELS = []
for group in MODEL_PRESETS.values():
    for m in group:
        if m not in ALL_MODELS:
            ALL_MODELS.append(m)


@st.cache_data(show_spinner=False)
def get_available_models():
    try:
        out = []
        for m in client.models.list():
            name = getattr(m, "name", "")
            actions = getattr(m, "supported_actions", []) or []
            if "generateContent" in actions:
                out.append(name)
        return out, None
    except Exception as e:
        return [], str(e)


def choose_model(preset, available):
    for m in MODEL_PRESETS.get(preset, []):
        if m in available:
            return m
    for m in ALL_MODELS:
        if m in available:
            return m
    return available[0] if available else None


def call_gemini(contents, chosen_model, preset, available):
    to_try = []
    if chosen_model:
        to_try.append(chosen_model)
    for m in MODEL_PRESETS.get(preset, []) + ALL_MODELS + available:
        if m in available and m not in to_try:
            to_try.append(m)

    errors = []
    for m in to_try:
        try:
            return client.models.generate_content(model=m, contents=contents), m, errors
        except Exception as e:
            errors.append(f"{m}: {e}")

    raise RuntimeError("Nepodarilo sa použiť žiadny dostupný model.\n\n" + "\n\n".join(errors))


available_models, model_error = get_available_models()
if model_error:
    st.error(f"Nepodarilo sa načítať modely: {model_error}")
    st.stop()
if not available_models:
    st.error("Pre tento API kľúč sa nenašiel žiadny model podporujúci generateContent.")
    st.stop()


# ---------- templates ----------
DOCUMENT_TYPES = [
    "Ambulancia - vstupné",
    "Ambulancia - kontrolné",
    "CPO konzílium",
    "Príjem na hospitalizáciu",
]

DEFAULT_TEMPLATES = {
    "UNB - ambulancia vstupné": {
        "document_type": "Ambulancia - vstupné",
        "body": """Indikácia:

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
    },
    "UNB - ambulancia kontrolné": {
        "document_type": "Ambulancia - kontrolné",
        "body": """Subjektívne od poslednej kontroly:

Aktuálne ťažkosti:

Liečba a tolerancia:

Objektívne neurologicky:

Doplňujúce vyšetrenia:

Záver:

Odporúčanie:
""",
    },
    "UNB - CPO konzílium": {
        "document_type": "CPO konzílium",
        "body": """Indikácia:

TO:

OA:

LA:

AA:

Objektívne neurologicky:

Doplňujúce vyšetrenia:

Záver:

Odporúčanie:
""",
    },
    "UNB - príjem na hospitalizáciu": {
        "document_type": "Príjem na hospitalizáciu",
        "body": """Dôvod prijatia:

TO:

OA:

LA:

AA:

RA:

Objektívne neurologicky:

Doplňujúce vyšetrenia:

Záver:

Plán:
""",
    },
    "Príjem - radikulopatia": {
        "document_type": "Príjem na hospitalizáciu",
        "body": """Dôvod prijatia:

TO:

OA:

LA:

AA:

RA:

Objektívne neurologicky:

Doplňujúce vyšetrenia:

Záver:

Plán:
""",
    },
}


def normalize_templates(raw):
    if not isinstance(raw, dict):
        return DEFAULT_TEMPLATES.copy()

    cleaned = {}
    for name, val in raw.items():
        if isinstance(val, dict):
            doc_type = val.get("document_type", DOCUMENT_TYPES[0])
            body = val.get("body", "")
        else:
            doc_type = DOCUMENT_TYPES[0]
            body = val

        if isinstance(name, str) and isinstance(body, str):
            cleaned[name] = {
                "document_type": doc_type if doc_type in DOCUMENT_TYPES else DOCUMENT_TYPES[0],
                "body": body,
            }

    return cleaned or DEFAULT_TEMPLATES.copy()


if "templates" not in st.session_state:
    st.session_state.templates = DEFAULT_TEMPLATES.copy()
if "case" not in st.session_state:
    st.session_state.case = None
if "editable_main" not in st.session_state:
    st.session_state.editable_main = ""


# ---------- small UI helpers ----------
def copy_button(text, label="Kopírovať"):
    safe_text = json.dumps(text or "")
    uid = "copy_" + str(abs(hash((text or "") + label)))
    components.html(f"""
    <button onclick='navigator.clipboard.writeText({safe_text}).then(() => {{
        const el = document.getElementById("{uid}");
        el.innerText = "Skopírované";
        setTimeout(() => el.innerText = "", 1800);
    }})'
    style="background:#0f6fff;color:white;border:none;border-radius:6px;padding:0.52rem 0.85rem;font-size:0.95rem;cursor:pointer;">
        {html.escape(label)}
    </button>
    <span id="{uid}" style="margin-left:12px;color:green;font-size:0.95rem;"></span>
    """, height=52)


def wake_lock_widget():
    components.html("""
    <button id="wakeBtn" style="background:#f2f2f2;border:1px solid #ccc;border-radius:6px;padding:0.45rem 0.75rem;cursor:pointer;">
        Aktivovať nezhasínanie obrazovky
    </button>
    <span id="wakeStatus" style="margin-left:10px;color:#666;font-size:14px;"></span>
    <script>
    let wakeLock = null;
    const statusEl = document.getElementById("wakeStatus");
    async function requestWakeLock() {
        try {
            if (!("wakeLock" in navigator)) {
                statusEl.textContent = "Wake Lock nie je v tomto prehliadači podporovaný.";
                return;
            }
            wakeLock = await navigator.wakeLock.request("screen");
            statusEl.textContent = "Obrazovka by mala zostať zapnutá.";
            wakeLock.addEventListener("release", () => { statusEl.textContent = "Wake Lock bol uvoľnený."; });
        } catch (err) {
            statusEl.textContent = "Nepodarilo sa aktivovať Wake Lock.";
        }
    }
    document.getElementById("wakeBtn").addEventListener("click", requestWakeLock);
    document.addEventListener("visibilitychange", async () => {
        if (wakeLock !== null && document.visibilityState === "visible") await requestWakeLock();
    });
    </script>
    """, height=58)


def parse_json_response(text):
    raw = text or ""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {
            "main_output": raw,
            "doctor_warnings": ["Model nevrátil validný JSON. Zobrazený je surový text."],
            "transcript": "",
            "role_inference": "",
            "clinical_facts": "",
            "quality_check": "",
            "raw_output": raw,
            "parse_ok": False,
        }

    try:
        data = json.loads(cleaned[start:end + 1])
    except Exception:
        return {
            "main_output": raw,
            "doctor_warnings": ["Model nevrátil validný JSON. Zobrazený je surový text."],
            "transcript": "",
            "role_inference": "",
            "clinical_facts": "",
            "quality_check": "",
            "raw_output": raw,
            "parse_ok": False,
        }

    def txt(v):
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        return json.dumps(v, ensure_ascii=False, indent=2)

    warnings = data.get("doctor_warnings", [])
    if isinstance(warnings, str):
        warnings = [warnings]
    if not isinstance(warnings, list):
        warnings = []

    return {
        "main_output": txt(data.get("main_output", "")),
        "doctor_warnings": [str(x) for x in warnings if str(x).strip()],
        "transcript": txt(data.get("transcript", "")),
        "role_inference": txt(data.get("role_inference", "")),
        "clinical_facts": txt(data.get("clinical_facts", "")),
        "quality_check": txt(data.get("quality_check", "")),
        "raw_output": raw,
        "parse_ok": True,
    }


def clear_case():
    st.session_state.case = None
    st.session_state.editable_main = ""


# ---------- sidebar ----------
st.sidebar.header("Nastavenie")

document_type = st.sidebar.selectbox(
    "Typ dokumentu:",
    DOCUMENT_TYPES,
    index=DOCUMENT_TYPES.index("Príjem na hospitalizáciu"),
)

matching = [
    name for name, item in st.session_state.templates.items()
    if item.get("document_type") == document_type
]
if not matching:
    matching = list(st.session_state.templates.keys())

template_name = st.sidebar.selectbox("Predloha:", matching)
template_body = st.session_state.templates[template_name]["body"]

input_mode = st.sidebar.radio("Režim vstupu:", ["Diktát lekára", "Rozhovor s pacientom"])
quality = st.sidebar.radio("Režim kvality:", ["Rýchle", "Vyvážené", "Presné"], index=1)
include_details = st.sidebar.checkbox(
    "Generovať detailnú analýzu / transkript",
    value=False,
    help="Vypnuté = rýchlejší praktický výstup. Zapnuté = transkript, role, klinické fakty a kontrola kvality.",
)

with st.sidebar.expander("Pokročilé nastavenia", expanded=False):
    default_model = choose_model(quality, available_models)
    manual_model = st.checkbox("Vybrať model manuálne", value=False)
    if manual_model:
        model = st.selectbox(
            "Model:",
            available_models,
            index=available_models.index(default_model) if default_model in available_models else 0,
        )
    else:
        model = default_model
        st.write("Použitý model:", model)

st.sidebar.markdown("---")
st.sidebar.subheader("Predlohy")

template_action = st.sidebar.radio(
    "Akcia:",
    ["Pozrieť / upraviť", "Duplikovať", "Pridať novú", "Import / export"],
)

if template_action == "Pozrieť / upraviť":
    st.sidebar.write(f"Aktuálna: {template_name}")
    edited_doc_type = st.sidebar.selectbox(
        "Typ dokumentu predlohy:",
        DOCUMENT_TYPES,
        index=DOCUMENT_TYPES.index(st.session_state.templates[template_name].get("document_type", document_type)),
    )
    edited_body = st.sidebar.text_area("Text predlohy:", value=template_body, height=310)

    if st.sidebar.button("Uložiť predlohu"):
        st.session_state.templates[template_name] = {"document_type": edited_doc_type, "body": edited_body}
        st.sidebar.success("Predloha uložená.")
        st.rerun()

    if template_name in DEFAULT_TEMPLATES:
        if st.sidebar.button("Resetovať vstavanú predlohu"):
            st.session_state.templates[template_name] = DEFAULT_TEMPLATES[template_name].copy()
            st.rerun()
    else:
        if st.sidebar.button("Vymazať vlastnú predlohu"):
            del st.session_state.templates[template_name]
            st.rerun()

elif template_action == "Duplikovať":
    st.sidebar.write(f"Zdroj: {template_name}")
    duplicate_name = st.sidebar.text_input("Názov kópie:", value=f"{template_name} - kópia")
    if st.sidebar.button("Vytvoriť kópiu"):
        name = duplicate_name.strip()
        if not name:
            st.sidebar.error("Zadajte názov.")
        elif name in st.session_state.templates:
            st.sidebar.error("Predloha s týmto názvom už existuje.")
        else:
            st.session_state.templates[name] = {
                "document_type": st.session_state.templates[template_name].get("document_type", document_type),
                "body": template_body,
            }
            st.rerun()

elif template_action == "Pridať novú":
    new_name = st.sidebar.text_input("Názov novej predlohy")
    new_doc_type = st.sidebar.selectbox("Typ dokumentu:", DOCUMENT_TYPES, index=DOCUMENT_TYPES.index(document_type))
    new_body = st.sidebar.text_area(
        "Text novej predlohy:",
        height=310,
        placeholder="Napríklad:\n\nDôvod prijatia:\n\nTO:\n\nObjektívne neurologicky:\n\nZáver:\n\nPlán:",
    )
    if st.sidebar.button("Vytvoriť predlohu"):
        name, body = new_name.strip(), new_body.strip()
        if not name:
            st.sidebar.error("Zadajte názov predlohy.")
        elif not body:
            st.sidebar.error("Zadajte text predlohy.")
        elif name in st.session_state.templates:
            st.sidebar.error("Predloha s týmto názvom už existuje.")
        else:
            st.session_state.templates[name] = {"document_type": new_doc_type, "body": body}
            st.rerun()

elif template_action == "Import / export":
    templates_json = json.dumps(st.session_state.templates, ensure_ascii=False, indent=2)
    st.sidebar.download_button(
        "Stiahnuť predlohy ako JSON",
        data=templates_json,
        file_name="predlohy_neuro_ai.json",
        mime="application/json",
    )
    uploaded_templates = st.sidebar.file_uploader("Nahrať predlohy JSON", type=["json"])
    if uploaded_templates is not None:
        try:
            loaded = json.load(uploaded_templates)
            st.session_state.templates.update(normalize_templates(loaded))
            st.sidebar.success("Predlohy boli nahraté.")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Nepodarilo sa načítať JSON: {e}")


# ---------- main UI ----------
st.subheader("1. Súhlas pacienta")
consent = st.radio(
    "Stav súhlasu:",
    ["Pacient podpísal / potvrdil súhlas", "Bez nahrávania"],
    horizontal=True,
)

if consent == "Bez nahrávania":
    st.warning("Režim bez nahrávania: môžete manuálne vložiť text alebo diktát nižšie.")

st.subheader("2. Nahrávka alebo manuálny vstup")

audio_data = None
if consent == "Pacient podpísal / potvrdil súhlas":
    with st.expander("Mobil / tablet: nezhasínanie obrazovky", expanded=False):
        st.write("Počas nahrávania neprepínajte aplikácie a nezamykajte obrazovku. Tlačidlo sa pokúsi udržať displej zapnutý, ak to prehliadač podporuje.")
        wake_lock_widget()

    audio_data = st.audio_input("Nahrajte vyšetrenie / diktát")
    if audio_data:
        st.audio(audio_data)

manual_text = st.text_area(
    "Voliteľne: manuálne vložený transkript alebo poznámky lekára",
    height=160,
    placeholder="Sem môžete vložiť transkript, diktát alebo doplňujúce poznámky.",
)


def build_prompt(document_type, template_name, template_body, input_mode, quality, include_details, manual_text):
    detail_part = (
        "Generuj aj plnú detailnú analýzu: transcript, role_inference, clinical_facts, quality_check."
        if include_details
        else "Negeneruj plný transkript ani dlhú analýzu. transcript, role_inference a clinical_facts nechaj prázdne alebo veľmi stručné. quality_check nech je krátky."
    )

    return f"""
Si skúsený slovenský neurologický klinický asistent.

Úloha:
Premeň slovenský lekársky diktát alebo rozhovor lekár-pacient na kvalitnú neurologickú dokumentáciu.
Výstup musí byť klinicky upravený, nie doslovný prepis.
Lekár zostáva plne zodpovedný za finálnu kontrolu.

Typ dokumentu:
{document_type}

Predloha:
{template_name}

Text predlohy:
{template_body}

Režim vstupu:
{input_mode}

Režim kvality:
{quality}

Manuálne doplnený kontext / poznámky:
{manual_text}

Vráť iba validný JSON objekt. Nepíš markdown, nepíš ```json, nepíš vysvetlenie mimo JSON.

JSON schéma:
{{
  "main_output": "hlavná lekárska správa pripravená na copy-paste do MIS",
  "doctor_warnings": ["krátke upozornenia pre lekára na overenie"],
  "transcript": "transkript, iba ak bol vyžiadaný",
  "role_inference": "role hovoriacich, iba ak relevantné",
  "clinical_facts": "štruktúrované klinické fakty, iba ak boli vyžiadané",
  "quality_check": "krátka kontrola kvality a neistôt"
}}

HLAVNÝ VÝSTUP:
- main_output je najdôležitejší.
- Musí byť použiteľný priamo v slovenskej neurologickej dokumentácii.
- Použi štruktúru predlohy.
- Každý nadpis predlohy daj na samostatný riadok.
- Text pod nadpisom daj na nový riadok.
- Medzi sekciami nechaj jeden prázdny riadok.
- Výstup má byť vecný, odborný, slovenský, v štýle UNB/Kramáre.
- Nepoužívaj odrážky v hlavnej správe, ak to nie je nevyhnutné.
- Dôvod prijatia / indikácia má byť stručná, ideálne 1 veta.
- TO má obsahovať klinický príbeh, nie doslovný prepis.
- Hlavný výstup nemá obsahovať meta-komentáre o kvalite nahrávky.

KLINICKÁ NORMALIZÁCIA:
- Odstráň výplňové slová a sebaopravy lekára: "sori", "teda", "vlastne", "ako keby", ak nie sú medicínsky dôležité.
- Ak lekár povie dve časové možnosti, napr. "dva dni, respektíve päť dní", v hlavnom texte použi opatrnú formuláciu "v posledných dňoch" a neistotu daj do doctor_warnings.
- Ak sa objaví zjavný ASR omyl ako "traumatický príjem", ale z kontextu ide o netraumatickú radikulopatiu, nepíš "traumatický".
- "lasek", "Laség", "Lasegue" normalizuj ako "Lasègue".
- "emery", "MRka", "emrika" normalizuj ako "MR"; pri LS ťažkostiach ako "MR LS chrbtice".
- "sendomotorický" oprav na "senzomotorický".
- "M35", "M 3 5", "3 z 5" interpretuj ako svalová sila 3/5, ak kontext sedí.
- "skudexu", "skúdeksu", "Skudexu" oprav ako "Skudexa", ak ide o analgetikum.
- Pri neistej alebo neštandardnej dávke lieku nepíš definitívne dávkovanie ako istý fakt; použi "dávku overiť" a pridaj upozornenie do doctor_warnings.

PRAVIDLÁ PRE ZÁVER A PLÁN:
- Ak lekár explicitne povedal záver, uveď ho v sekcii Záver.
- Ak lekár explicitne povedal odporúčanie/plán, uveď ho v sekcii Odporúčanie alebo Plán.
- Ak záver nebol explicitne povedaný, napíš "Na doplnenie lekárom."
- Ak plán/odporúčanie nebol explicitne povedaný, napíš "Na doplnenie lekárom."
- Nevymýšľaj nové diagnózy, liečbu, MR, CT, laboratóriá, kontroly ani hospitalizáciu.

BEZPEČNOSTNÉ PRAVIDLÁ:
- Nevymýšľaj si fakty.
- Nevymýšľaj si lieky ani dávkovanie.
- Nevymýšľaj si normálny neurologický nález, ak nebol uvedený.
- Nepíš "neurologický nález v norme", ak to nebolo explicitne povedané.
- Ak lekár povie "štandardný neurologický nález okrem...", môžeš uviesť len "ostatný neurologický nález podľa diktátu bez nápadnej patológie", ale neinventuj detaily.
- Zachovaj negácie.
- Nezamieňaj subjektívne ťažkosti za objektívny nález.
- Nepridávaj časti neurologického vyšetrenia, ktoré neboli uvedené.
- Výsledky CT/MR/lab/EEG/EMG uveď len vtedy, ak boli uvedené.

DOCTOR_WARNINGS:
- Uveď krátke praktické upozornenia hneď pre lekára.
- Maximálne 5 položiek.
- Typicky: nejasná dávka lieku, nejasný časový údaj, chýbajúce OA/AA/RA, neistý fonetický výraz, potrebná kontrola laterality.
- Nenavrhuj nové diagnózy ani liečbu.

DETAILY:
{detail_part}
"""


# ---------- processing ----------
st.subheader("3. Spracovanie")
col_a, col_b = st.columns([1, 1])
process_clicked = col_a.button("Spracovať vyšetrenie", type="primary")
clear_clicked = col_b.button("Vymazať prípad")

if clear_clicked:
    clear_case()
    st.rerun()

if process_clicked:
    if not audio_data and not manual_text.strip():
        st.error("Nahrajte audio alebo vložte manuálny text.")
        st.stop()

    tmp_path, uploaded_file = None, None

    with st.spinner("Spracovávam vstup a generujem výstup..."):
        try:
            prompt = build_prompt(
                document_type=document_type,
                template_name=template_name,
                template_body=template_body,
                input_mode=input_mode,
                quality=quality,
                include_details=include_details,
                manual_text=manual_text,
            )

            contents = []

            if audio_data:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_data.read())
                    tmp_path = tmp.name

                uploaded_file = client.files.upload(file=tmp_path)
                time.sleep(1)
                contents.append(uploaded_file)

            contents.append(prompt)

            response, used_model, fallback_errors = call_gemini(
                contents=contents,
                chosen_model=model,
                preset=quality,
                available=available_models,
            )

            parsed = parse_json_response(response.text or "")
            parsed.update({
                "used_model": used_model,
                "fallback_errors": fallback_errors,
                "document_type": document_type,
                "template_name": template_name,
                "quality": quality,
                "include_details": include_details,
            })

            st.session_state.case = parsed
            st.session_state.editable_main = parsed.get("main_output", "")

            st.success(f"Spracovanie dokončené. Použitý model: {used_model}")

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


# ---------- output ----------
case = st.session_state.case

if case:
    st.subheader("4. Výstup do MIS")

    st.text_area(
        "Hlavný výstup (môžete upraviť pred kopírovaním)",
        key="editable_main",
        height=410,
    )

    col_copy, col_download = st.columns([1, 1])
    with col_copy:
        copy_button(st.session_state.editable_main, "Kopírovať celý výstup")
    with col_download:
        st.download_button(
            label="Stiahnuť hlavný výstup ako TXT",
            data=st.session_state.editable_main,
            file_name="hlavny_vystup.txt",
            mime="text/plain",
        )

    st.subheader("Na kontrolu lekárom")
    warnings = case.get("doctor_warnings", [])
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("Bez zásadných upozornení. Finálnu správu musí lekár aj tak skontrolovať.")

    if include_details:
        with st.expander("Detailná analýza", expanded=False):
            st.markdown("### Transkript")
            st.write(case.get("transcript") or "Negenerované / prázdne.")

            st.markdown("### Inferencia rolí")
            st.write(case.get("role_inference") or "Negenerované / prázdne.")

            st.markdown("### Klinické fakty")
            st.write(case.get("clinical_facts") or "Negenerované / prázdne.")

            st.markdown("### Kontrola kvality")
            st.write(case.get("quality_check") or "Negenerované / prázdne.")
    else:
        with st.expander("Krátka kontrola kvality", expanded=False):
            st.write(case.get("quality_check") or "Detailná analýza nebola generovaná.")

    with st.expander("Technické info", expanded=False):
        st.write("Typ dokumentu:", case.get("document_type"))
        st.write("Predloha:", case.get("template_name"))
        st.write("Režim kvality:", case.get("quality"))
        st.write("Použitý model:", case.get("used_model"))
        st.write("JSON parse OK:", case.get("parse_ok"))

        fallback_errors = case.get("fallback_errors", [])
        if fallback_errors:
            st.write("Fallback chyby:")
            for err in fallback_errors:
                st.text(err)

        with st.expander("Surový výstup modelu"):
            st.text(case.get("raw_output", ""))
else:
    st.info("Nahrajte audio alebo vložte text, potom kliknite na „Spracovať vyšetrenie“.")
