# Neuro AI Asistent - Streamlit deploy

## Lokalny start

1. Vytvor `.env`:

```env
GEMINI_API_KEY=tvoj_kluc
```

2. Nainstaluj baliky:

```bash
pip install -r requirements.txt
```

3. Spusti:

```bash
streamlit run ambulancia.py
```

## Streamlit Cloud

1. Nahraj `ambulancia.py`, `requirements.txt`, `.gitignore`, `README.md` do GitHub repozitara.
2. V Streamlit Community Cloud vytvor appku z GitHub repozitara.
3. Main file path: `ambulancia.py`.
4. Do Secrets pridaj:

```toml
GEMINI_API_KEY = "tvoj_kluc"
```

## Bezpecnost

Nepouzivaj realne identifikovatelne pacientske data v demo/cloude bez pravneho a bezpecnostneho nastavenia.
