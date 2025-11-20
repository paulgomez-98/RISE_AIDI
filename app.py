# app.py — RISE Smart Scoring (Final, Clean, White Theme + Logo)
# - Full light UI theme override (header, sidebar, uploader, all)
# - Georgian logo preserved exactly as original
# - Feedback for ALL applicants
# - Failed API calls shown
# - No calibration
# - Single-file deployment for Streamlit Cloud

import os, re, json, requests
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ============================================================
#                     FULL LIGHT MODE OVERRIDE
# ============================================================
light_theme_css = """
<style>

    /* Make Streamlit header white */
    header[data-testid="stHeader"] {
        background-color: white !important;
    }

    header[data-testid="stHeader"]::before {
        background: white !important;
    }

    /* Main app background */
    .stApp {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
    }

    /* Global text color */
    body, p, span, label, div, h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }

    /* File uploader box */
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #005EA5 !important; /* Georgian blue */
        color: #000000 !important;
        border-radius: 10px;
        padding: 20px;
    }

    /* File uploader icon */
    div[data-testid="stFileUploaderDropzone"] svg {
        fill: #005EA5 !important;
    }

    /* Inputs & textboxes */
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 6px;
    }

    /* Select dropdowns */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    /* Tables / DataFrames */
    .stDataFrame, .stTable {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    /* Buttons */
    button[kind="primary"] {
        background-color: #005EA5 !important; /* Georgian Blue */
        color: white !important;
        border-radius: 6px !important;
    }

    button[kind="secondary"] {
        background-color: white !important;
        color: #005EA5 !important;
        border: 1px solid #005EA5 !important;
        border-radius: 6px !important;
    }

</style>
"""
st.markdown(light_theme_css, unsafe_allow_html=True)

# ============================================================
#                     PAGE HEADER + LOGO
# ============================================================
st.set_page_config(page_title="RISE Smart Scoring", layout="wide")

left, right = st.columns([5,1])
with left:
    st.markdown("<h2>RISE — Smart Scoring & Feedback</h2>", unsafe_allow_html=True)
    st.write("Upload → Score using LLaMA → View Top-K → View All → Download")

with right:
    logo_url = st.secrets.get("LOGO_URL", os.getenv("LOGO_URL", ""))
    shown = False

    if logo_url:
        try:
            st.image(logo_url, use_container_width=True); shown=True
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
                st.image(str(p), use_container_width=True); shown=True; break

    if not shown:
        st.caption("")

# ============================================================
#                     CONSTANTS
# ============================================================
TARGET_TITLE = "What is the title of your research/capstone project?"
TARGET_ABS   = "Please provide a description or abstract of your research."

MODEL_NAME  = "llama-3.1-8b-instant"
GROQ_URL    = "https://api.groq.com/openai/v1"
TEMPERATURE = 0.1
MAX_TOKENS  = 256

def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower().replace("’","'")

# ============================================================
#                     LLaMA SCORING FUNCTION
# ============================================================
def llama_score(title, abstract, api_key):
    prompt = f"""
You are scoring a student research project.

Return ONLY JSON:
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
        res = requests.post(
            GROQ_URL + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
            },
            timeout=60
        )

        res.raise_for_status()
        content = re.sub(r"```json|```", "", res.json()["choices"][0]["message"]["content"]).strip()
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
#                     FILE UPLOAD
# ============================================================
file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
if not file:
    st.stop()

df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

if df.empty:
    st.error("The uploaded file is empty.")
    st.stop()

# ============================================================
#                     COLUMN VALIDATION
# ============================================================
normmap = {norm(c): c for c in df.columns}
missing = []

if norm(TARGET_TITLE) not in normmap: missing.append(TARGET_TITLE)
if norm(TARGET_ABS) not in normmap: missing.append(TARGET_ABS)

if missing:
    st.error("Missing required columns:\n" + "\n".join([f"• {m}" for m in missing]))
    st.stop()

title_col = normmap[norm(TARGET_TITLE)]
abs_col   = normmap[norm(TARGET_ABS)]

work = df[[title_col, abs_col]].copy()
work.columns = ["title","abstract"]

work["title"] = work["title"].astype(str).str.strip()
work["abstract"] = work["abstract"].astype(str).str.strip()
work = work[work["title"].str.len() >= 3]
work = work.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

# ============================================================
#                     API KEY
# ============================================================
api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Enter GROQ API Key", type="password")
if not api_key:
    st.stop()

# ============================================================
#                     SCORING LOOP
# ============================================================
st.subheader("Scoring with LLaMA…")

results = []
prog = st.progress(0.0)

for i, row in work.iterrows():
    results.append(llama_score(row["title"], row["abstract"], api_key))
    prog.progress((i+1)/len(work))

scored = pd.DataFrame(results)

# ============================================================
#                     FAILED API RESULTS
# ============================================================
failed = scored[scored["success"] == False]
if not failed.empty:
    st.warning(f"{len(failed)} entries FAILED during scoring.")
    st.dataframe(failed[["title","feedback"]])

# ============================================================
#                     OVERALL SCORE (equal weights)
# ============================================================
weights = np.ones(5)/5
crit = scored[["originality","clarity","rigor","impact","entrepreneurship"]].to_numpy(float)
scored["overall"] = (crit @ weights).round(2)

# ============================================================
#                     TOP-K SECTION
# ============================================================
st.subheader("Top Rankings")

TOP_K = st.slider("Select Top-K", 5, 50, 10, 5)
top = scored.sort_values("overall", ascending=False).reset_index(drop=True)

st.dataframe(top.head(TOP_K)[["title","overall","originality","clarity","rigor","impact","entrepreneurship"]])

pick = st.selectbox("View detailed result:", top.head(TOP_K)["title"].tolist())
sel = top[top["title"] == pick].iloc[0]

st.write(f"""
### {sel['title']}

**Overall Score:** {sel['overall']}

- Originality: {sel['originality']}
- Clarity: {sel['clarity']}
- Rigor: {sel['rigor']}
- Impact: {sel['impact']}
- Entrepreneurship: {sel['entrepreneurship']}
""")

st.caption("Feedback: " + sel["feedback"])

# ============================================================
#                     ALL RESULTS
# ============================================================
st.subheader("All Scored Results")

display_cols = [
    "title","originality","clarity","rigor",
    "impact","entrepreneurship","overall","feedback"
]

st.dataframe(top[display_cols], use_container_width=True)

# ============================================================
#                     DOWNLOAD BUTTONS
# ============================================================
st.download_button(
    "⬇️ Download Top-K (CSV)",
    top.head(TOP_K)[display_cols].to_csv(index=False),
    file_name=f"rise_top_{TOP_K}.csv",
    mime="text/csv"
)

st.download_button(
    "⬇️ Download ALL Results (CSV)",
    top[display_cols].to_csv(index=False),
    file_name="rise_all_scored.csv",
    mime="text/csv"
)
