import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from openai import OpenAI
from streamlit.runtime.secrets import secrets

# ========== LLaMA via Groq ==========
client = OpenAI(
    api_key=secrets["GROQ_API_KEY"],
    base_url=secrets["GROQ_BASE_URL"],
)

MODEL = secrets["LLAMA_MODEL"]
TEMP = float(secrets.get("LLAMA_TEMPERATURE", 0.2))
MAXTOK = int(secrets.get("LLAMA_MAX_TOKENS", 256))

# ========== UI Settings ==========
st.set_page_config(page_title="RISE Smart Scoring", layout="wide")

header_left, header_right = st.columns([5,1])
with header_left:
    st.markdown("<h2>RISE — Smart Scoring & Feedback</h2>", unsafe_allow_html=True)
with header_right:
    st.image("static/georgian_logo.png", width=110)

st.write("Upload your RISE project submissions to automatically evaluate and rank them.")

# ========== File Upload ==========
file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if file is None:
    st.stop()

# Load file
try:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
except Exception:
    st.error("Invalid file format. Please upload a proper CSV/XLSX file.")
    st.stop()

# Auto-detect columns
title_col = next((c for c in df.columns if "title" in c.lower()), df.columns[0])
abs_col = next((c for c in df.columns if "abstract" in c.lower() or "description" in c.lower()), df.columns[1])

work = df[[title_col, abs_col]].copy()
work.columns = ["title", "abstract"]

# ========== Duplicate Detection ==========
norm_title = work["title"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True)

duplicates = work[norm_title.duplicated(keep=False)]

if len(duplicates) > 0:
    st.subheader("⚠️ Duplicate Project Titles Detected")
    dup_list = sorted(duplicates["title"].unique())
    choice = st.selectbox("Select a duplicate title to inspect:", dup_list)
    st.table(duplicates[duplicates["title"] == choice])

# Keep first instance of each
work = work.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

# ========== LLaMA Scoring ==========
def score_with_llama(title, abstract):
    prompt = f"""
    Evaluate the following student project abstract for the RISE competition.
    Score each criterion from 1 (weak) to 5 (excellent).

    Criteria:
    - Originality: Novelty of ideas
    - Clarity: Writing structure & understanding
    - Rigor: Strength of method / reasoning
    - Impact: Benefit / usefulness of outcome
    - Entrepreneurship: Practical problem-solving & initiative shown

    Return ONLY JSON (no explanations outside JSON).

    {{
      "title": "{title}",
      "originality": X,
      "clarity": X,
      "rigor": X,
      "impact": X,
      "entrepreneurship": X,
      "feedback": "One helpful paragraph of improvement-focused feedback."
    }}

    Abstract:
    {abstract}
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMP,
        max_tokens=MAXTOK,
    )

    text = response.choices[0].message.content
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)


st.subheader("🔄 Scoring Projects... Please wait.")
results = []

progress = st.progress(0)
for i, row in work.iterrows():
    scored = score_with_llama(row["title"], row["abstract"])
    results.append(scored)
    progress.progress((i+1)/len(work))

scored_df = pd.DataFrame(results)
scored_df["overall"] = scored_df[["originality","clarity","rigor","impact","entrepreneurship"]].mean(axis=1).round(2)

# ========== Show Top 10 ==========
st.subheader("🏆 Top 10 Ranked Projects")
top10 = scored_df.sort_values("overall", ascending=False).head(10).reset_index(drop=True)
st.dataframe(top10, use_container_width=True)

# ========== Download Buttons ==========
dl_col1, dl_col2, space = st.columns([1,1,5])
with dl_col1:
    st.download_button(
        "⬇️ Download Top 10 CSV",
        top10.to_csv(index=False),
        "top10_projects.csv",
        mime="text/csv"
    )

with dl_col2:
    st.download_button(
        "⬇️ Download Full Results CSV",
        scored_df.to_csv(index=False),
        "all_scored_projects.csv",
        mime="text/csv"
    )
