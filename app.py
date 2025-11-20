# app.py — RISE Smart Scoring (Clean Single File Version)
# - UI and AI logic separated into functions (but same file)
# - NO calibration section
# - Failed API calls displayed clearly
# - Feedback for ALL applicants
# - Uses EXACT required headers

import os, re, json, requests
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ============================================================
#                    STREAMLIT PAGE SETTINGS
# ============================================================
st.set_page_config(page_title="RISE Smart Scoring", layout="wide")

st.markdown("<h2>RISE — Smart Scoring & Feedback</h2>", unsafe_allow_html=True)
st.write("Upload CSV/XLSX → Score with LLaMA → View Top-K → View All → Download")

# ============================================================
#                    CONSTANTS
# ============================================================
TARGET_TITLE = "What is the title of your research/capstone project?"
TARGET_ABS   = "Please provide a description or abstract of your research."

DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_BASE  = "https://api.groq.com/openai/v1"
TEMPERATURE   = 0.1
MAX_TOKENS    = 256

# ============================================================
#                    NORMALIZATION FUNCTION
# ============================================================
def norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip().lower()
    s = s.replace("’","'")
    return s

# ============================================================
#                    AI SCORING FUNCTION
# ============================================================
def llama_score(title: str, abstract: str, api_key: str):
    """
    Returns dict:
    {
      "success": bool,
      "title": ...,
      "originality": float,
      "clarity": float,
      "rigor": float,
      "impact": float,
      "entrepreneurship": float,
      "feedback": str
    }
    """

    prompt = f"""
You are scoring a student research project.

Return ONLY a compact JSON object:
{{
  "originality": number 1-5,
  "clarity": number 1-5,
  "rigor": number 1-5,
  "impact": number 1-5,
  "entrepreneurship": number 1-5,
  "feedback": "one constructive feedback sentence"
}}

Title: {title}
Abstract: {abstract}
"""

    try:
        url = DEFAULT_BASE + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "top_p": 0.9
        }

        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()

        content = r.json()["choices"][0]["message"]["content"]
        content = re.sub(r"```json|```", "", content).strip()

        data = json.loads(content)

        return {
            "success": True,
            "title": title,
            "originality": float(data.get("originality", 3)),
            "clarity": float(data.get("clarity", 3)),
            "rigor": float(data.get("rigor", 3)),
            "impact": float(data.get("impact", 3)),
            "entrepreneurship": float(data.get("entrepreneurship", 3)),
            "feedback": data.get("feedback","").strip()
        }

    except Exception:
        return {
            "success": False,
            "title": title,
            "originality": 0,
            "clarity": 0,
            "rigor": 0,
            "impact": 0,
            "entrepreneurship": 0,
            "feedback": "API ERROR — scoring failed."
        }

# ============================================================
#                    FILE UPLOAD
# ============================================================
file = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx"])
if not file:
    st.stop()

try:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
except Exception as e:
    st.error(f"File error: {e}")
    st.stop()

if df.empty:
    st.error("Uploaded file is empty.")
    st.stop()

# ============================================================
#                    COLUMN VALIDATION
# ============================================================
normmap = {norm(c): c for c in df.columns}
missing = []

if norm(TARGET_TITLE) not in normmap:  missing.append(TARGET_TITLE)
if norm(TARGET_ABS) not in normmap:    missing.append(TARGET_ABS)

if missing:
    st.error("Missing required column(s):\n" + "\n".join([f"• {m}" for m in missing]))
    st.stop()

title_col = normmap[norm(TARGET_TITLE)]
abs_col   = normmap[norm(TARGET_ABS)]

work = df[[title_col, abs_col]].copy()
work.columns = ["title","abstract"]

# Clean data
work["title"] = work["title"].astype(str).str.strip()
work["abstract"] = work["abstract"].astype(str).str.strip()
work = work[work["title"].str.len() >= 3]

# Remove yes/no style junk titles
work = work[~work["title"].str.lower().isin({"yes","no","true","false"})]

# Remove duplicates
work = work.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

# ============================================================
#                    API KEY
# ============================================================
api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Enter GROQ API Key", type="password")
if not api_key:
    st.stop()

# ============================================================
#                    SCORING LOOP
# ============================================================
st.subheader("Scoring With LLaMA…")
results = []
prog = st.progress(0.0)

for i, row in work.iterrows():
    result = llama_score(row["title"], row["abstract"], api_key)
    results.append(result)
    prog.progress((i+1)/len(work))

scored = pd.DataFrame(results)

# ============================================================
#                    DISPLAY FAILED ENTRIES
# ============================================================
failed = scored[scored["success"] == False]
if not failed.empty:
    st.warning(f"{len(failed)} entries FAILED to score.")
    st.dataframe(failed[["title","feedback"]])

# ============================================================
#                    OVERALL SCORE (equal weights)
# ============================================================
weights = np.ones(5) / 5
crit = scored[["originality","clarity","rigor","impact","entrepreneurship"]].to_numpy(float)
scored["overall"] = (crit @ weights).round(2)

# ============================================================
#                    TOP-K DISPLAY
# ============================================================
st.subheader("Top Rankings")

TOP_K = st.slider("Top-K", min_value=5, max_value=50, value=10, step=5)

top = scored.sort_values("overall", ascending=False).reset_index(drop=True)

st.write(f"**Top {TOP_K} Projects:**")
st.dataframe(top.head(TOP_K)[["title","overall","originality","clarity","rigor","impact","entrepreneurship"]])

selected = st.selectbox("View detailed score:", top.head(TOP_K)["title"].tolist())

sel = top[top["title"] == selected].iloc[0]
st.write(
    f"**Title:** {sel['title']}\n\n"
    f"**Overall:** {sel['overall']:.2f}\n"
    f"- Originality: {sel['originality']}\n"
    f"- Clarity: {sel['clarity']}\n"
    f"- Rigor: {sel['rigor']}\n"
    f"- Impact: {sel['impact']}\n"
    f"- Entrepreneurship: {sel['entrepreneurship']}\n"
)

if str(sel["feedback"]).strip():
    st.caption(f"Feedback: {sel['feedback']}")

# ============================================================
#                    ALL RESULTS
# ============================================================
st.subheader("All Results (With Feedback)")
display_cols = ["title","originality","clarity","rigor","impact","entrepreneurship","overall","feedback"]
st.dataframe(top[display_cols])

# ============================================================
#                    DOWNLOADS
# ============================================================
st.download_button(
    "⬇️ Download Top-K (CSV)",
    top.head(TOP_K)[display_cols].to_csv(index=False),
    file_name=f"rise_top_{TOP_K}.csv",
    mime="text/csv"
)

st.download_button(
    "⬇️ Download All (CSV)",
    top[display_cols].to_csv(index=False),
    file_name="rise_all_scored.csv",
    mime="text/csv"
)
