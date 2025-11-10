import streamlit as st
import pandas as pd
import numpy as np
import re
from openai import OpenAI

# ==== Load LLaMA via Groq =====
from streamlit.runtime.secrets import secrets
client = OpenAI(
    api_key=secrets["GROQ_API_KEY"],
    base_url=secrets["GROQ_BASE_URL"],
)

MODEL = secrets["LLAMA_MODEL"]
TEMP = float(secrets.get("LLAMA_TEMPERATURE", 0.2))
MAXTOK = int(secrets.get("LLAMA_MAX_TOKENS", 256))


# ==== UI Setup ====
st.set_page_config(layout="wide", page_title="RISE Smart Scoring")

# Logo + Title
logo_col, title_col = st.columns([1, 5])
with logo_col:
    st.image("static/georgian_logo.png", width=110)
with title_col:
    st.markdown("<h2>RISE — Smart Scoring & Feedback</h2>", unsafe_allow_html=True)

st.write("Upload your project submissions and get ranked results based on clarity, originality, rigor, impact, and entrepreneurship.")


# ==== Upload ====
file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

if file is None:
    st.stop()

# Load
if file.name.endswith(".csv"):
    df = pd.read_csv(file)
else:
    df = pd.read_excel(file)

# Detect Columns
title_col = next((c for c in df.columns if "title" in c.lower()), df.columns[0])
abs_col = next((c for c in df.columns if "abstract" in c.lower() or "description" in c.lower()), df.columns[1])

work = df[[title_col, abs_col]].copy()
work.columns = ["title", "abstract"]

# ==== Remove duplicates by normalized title =====
norm = work["title"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True)
dupes = work[norm.duplicated(keep=False)]
unique = work.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

if len(dupes) > 0:
    st.subheader("Duplicate Project Titles Found")
    dup_titles = sorted(dupes["title"].unique())
    chosen = st.selectbox("Select a repeated title to view:", dup_titles)
    st.table(dupes[dupes["title"] == chosen])

work = unique.copy()

# ==== Scoring with LLaMA ====
def score_with_llama(text):
    prompt = f"""
    Evaluate the following research abstract across 5 criteria:
    - Originality
    - Clarity
    - Rigor (quality of method reasoning)
    - Impact (real-world benefit)
    - Entrepreneurship (problem-solving / practicality)

    Return values ONLY in JSON:
    {{
      "originality": 1-5,
      "clarity": 1-5,
      "rigor": 1-5,
      "impact": 1-5,
      "entrepreneurship": 1-5,
      "feedback": "One-paragraph helpful improvement advice."
    }}

    Abstract:
    {text}
    """

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMP,
        max_tokens=MAXTOK,
    )

    raw = resp.choices[0].message.content

    # Extract JSON safely
    import json
    raw = re.sub(r"```json|```", "", raw).strip()
    data = json.loads(raw)
    return data


st.subheader("Processing...")
results = []
for _, row in work.iterrows():
    data = score_with_llama(row["abstract"])
    results.append({
        "title": row["title"],
        **data
    })

scored = pd.DataFrame(results)
scored["overall"] = scored[["originality","clarity","rigor","impact","entrepreneurship"]].mean(axis=1).round(2)

# ==== Display Top 10 ====
st.subheader("Top 10 Projects")
top10 = scored.sort_values("overall", ascending=False).head(10).reset_index(drop=True)
st.dataframe(top10, use_container_width=True)

# ==== Download buttons aligned right ====
btn_col = st.columns([5,1,1])
with btn_col[1]:
    st.download_button("Download Top 10 CSV", top10.to_csv(index=False), "top10.csv")
with btn_col[2]:
    st.download_button("Download Full Results CSV", scored.to_csv(index=False), "all_scored.csv")

