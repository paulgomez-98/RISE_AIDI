# RISE Scoring & Feedback (Streamlit Demo)

A minimal Streamlit app that:
- uploads a CSV of applications,
- drops exact & near-duplicate rows by a chosen text column,
- uses an existing `score` column **or** computes a naive demo score,
- shows Top‑K,
- generates very simple feedback text,
- and lets you download results as CSV.

## Local run

```bash
# (Recommended) create & activate a virtualenv first
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this folder to a **public GitHub repo**.
2. Go to https://share.streamlit.io/ (Streamlit Community Cloud).
3. **New app** → choose your repo/branch → main file `app.py` → Deploy.

## Notes
- Replace the `simple_autoscore()` and `gen_feedback()` with your actual model.
- Ensure any extra packages are added to `requirements.txt`.
