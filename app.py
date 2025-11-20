# app.py — RISE Smart Scoring (Final, Clean, White Background + Logo)
# - White background that keeps text readable
# - Original Georgian logo loading preserved
# - No calibration
# - AI scoring separated into a function
# - All applicants get feedback
# - Failed API calls shown clearly
# - Uses EXACT two column names

import os, re, json, requests
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ============================================================
#                       WHITE BACKGROUND FIX
# ============================================================
# This CSS forces ONLY background white but keeps text black
white_bg_css = """
<style>
    .stApp {
        background-color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
    }
    .block-container {
        background-color: #FFFFFF !important;
    }

    /* Keep text readable */
    body, p, span, label, div, input, textarea {
        color: #000000 !important;
    }

    /* Input widgets */
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    .stDataFrame, .stTable {
        background-color: #FFFFFF !important;
    }
</style>
"""
st.markdown(white_bg_css, unsafe_allow_html=True)

# ============================================================
#                       PAGE CONFIG
# ============================================================
st.set_page_config(page_title="RISE Smart Scoring", layout="wide")

left, right = st.columns([5,1])
with left:
    st.markdown("<h2>RISE — Smart Scoring & Feedback</h2>", unsafe_allow_html=True)
    st.write("Upload → Score using LLaMA → View Top-K → View All → Download")

# ============================================================
#                       GEORGIAN LOGO LOADING
# ============================================================
with right:
    logo_url = st.secrets.get("LOGO_URL", os.getenv("LOGO_URL", ""))
    shown = False
    if logo_url:
        try:
            st.image(logo_url, use_container_width=True); shown = True
        except:
            pass

    if not shown:
        for p in [
            Path("static/georgian_logo.png"),
            Path("static/georgian_logo.jpg"),
            Path("georgian_logo.png"),
            Path("georgian_logo.jpg")
        ]:
            if p.exists():
                st.image(str(p), use_container_width=True); shown = True; break

    if not shown:
        st.caption("")

# ============================================================
#                       CONSTANTS
# ============================================================
TARGET_TITLE = "What is the title of your research/capstone project?"
TARGET_ABS   = "Please provide a description or abstract of your research."

MODEL_NAME   = "llama-3.1-8b-instant"
GROQ_URL     = "https://api.groq.com/openai/v1"
TEMPERATURE  = 0.1
MAX_TOKENS   = 256

def norm(s):
    s = re.sub(r"\s+", " ", str(s)).strip().lower()
    return s.replace("’", "'")

# ============================================================
#                       LLaMA SCORING FUNCTION
# ============================================================
def llama_score(title, abstract, api_key):
    prompt = f"""
You are scoring a student research project.

Return ONLY this JSON format:
{{
  "originality": 1-5,
  "clarity": 1-5,
  "rigor": 1-5,
  "impact": 1-5,
  "entrepreneurship": 1-5,
  "feedback": "one constructive feedback sentence"
}}

Title: {title}
Abstract: {abstract}
"""

    try:
        url = GROQ_URL + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": MODEL_NAME,
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
            "feedback": data.get("feedback", "").strip()
        }

    except:
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
#                       FILE UPLOAD
# ============================================================
file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
if not file:
    st.stop()

df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

if df.empty:
    st.error("The uploaded file is empty.")
    st.stop()

# ============================================================
#                       COLUMN VALIDATION
# ============================================================
normmap = {norm(c): c for c in df.columns}
missing = []

if norm(TARGET_TITLE) not in normmap:
    missing.append(TARGET_TITLE)

if norm(TARGET_ABS) not in normmap:
    missing.append(TARGET_ABS)

if missing:
    st.error("Missing required columns:\n" + "\n".join([f"• {m}" for m in missing]))
    st.stop()

title_col = normmap[norm(TARGET_TITLE)]
abs_col   = normmap[norm(TARGET_ABS)]

work = df[[title_col, abs_col]].copy()
work.columns = ["title", "abstract"]

# clean
work["title"] = work["title"].astype(str).strip()
work["abstract"] = work["abstract"].astype(str).strip()
work = work[work["title"].str.len() >= 3]
work = work.drop_duplicates(subset=["title"], keep="first").reset_index()

# ============================================================
#                       API KEY
# ============================================================
api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Enter GROQ API Key", type="password")
if not api_key:
    st.stop()

# ============================================================
#                       SCORING LOOP
# ============================================================
st.subheader("Scoring With LLaMA…")

results = []
prog = st.progress(0.0)

for i, row in work.iterrows():
    results.append(llama_score(row["title"], row["abstract"], api_key))
    prog.progress((i+1)/len(work))

scored = pd.DataFrame(results)

# ============================================================
#                       SHOW FAILED API CALLS
# ============================================================
failed = scored[scored["success"] == False]
if not failed.empty:
    st.warning(f"{len(failed)} entries FAILED during scoring.")
    st.dataframe(failed[["title", "feedback"]])

# ============================================================
#                       OVERALL SCORE
# ============================================================
weights = np.ones(5) / 5
crit = scored[["originality","clarity","rigor","impact","entrepreneurship"]].to_numpy(float)
scored["overall"] = (crit @ weights).round(2)

# ============================================================
#                       TOP-K
# ============================================================
st.subheader("Top Rankings")

TOP_K = st.slider("Top-K", min_value=5, max_value=50, value=10, step=5)

top = scored.sort_values("overall", ascending=False).reset_index(drop=True)

st.dataframe(
    top.head(TOP_K)[["title","overall","originality","clarity","rigor","impact","entrepreneurship"]],
    use_container_width=True
)

pick = st.selectbox("View details:", top.head(TOP_K)["title"].tolist())

sel = top[top["title"] == pick].iloc[0]

st.write(f"""
**Title:** {sel['title']}

**Overall:** {sel['overall']}

- Originality: {sel['originality']}
- Clarity: {sel['clarity']}
- Rigor: {sel['rigor']}
- Impact: {sel['impact']}
- Entrepreneurship: {sel['entrepreneurship']}
""")

if sel["feedback"]:
    st.caption("Feedback: " + sel["feedback"])

# ============================================================
#                       ALL RESULTS
# ============================================================
st.subheader("All Results (Full Table)")

display_cols = [
    "title","originality","clarity","rigor",
    "impact","entrepreneurship","overall","feedback"
]

st.dataframe(top[display_cols], use_container_width=True)

# ============================================================
#                       DOWNLOADS
# ============================================================
st.download_button(
    "⬇️ Download Top-K (CSV)",
    top.head(TOP_K)[display_cols].to_csv(index=False),
    file_name=f"rise_top_{TOP_K}.csv",
    mime="text/csv"
)

st.download_button(
    "⬇️ Download All Results (CSV)",
    top[display_cols].to_csv(index=False),
    file_name="rise_all_scored.csv",
    mime="text/csv"
)
