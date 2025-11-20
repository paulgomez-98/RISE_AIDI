# app.py — RISE Smart Scoring (ULTRA FAST MODE for Groq Free Tier)
# - Default Streamlit theme
# - Georgian logo
# - EDA + duplicate detection
# - No batch mode (Groq free tier doesn't allow it)
# - Minimal delay (max speed)
# - Only 1 retry attempt
# - Reduced token usage

import os, re, json, requests, time, random
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


# ============================================================
#                    PAGE CONFIG
# ============================================================
st.set_page_config(page_title="RISE Smart Scoring", layout="wide")


# ============================================================
#                    HEADER + LOGO
# ============================================================
left, right = st.columns([5,1])
with left:
    st.markdown("<h2>RISE — Smart Scoring & Feedback</h2>", unsafe_allow_html=True)
    st.write("Upload → EDA → Score using LLaMA → View Top-K → Download")

with right:
    logo_url = st.secrets.get("LOGO_URL", os.getenv("LOGO_URL", ""))
    shown = False

    if logo_url:
        try:
            st.image(logo_url, use_container_width=True)
            shown = True
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
                st.image(str(p), use_container_width=True)
                shown = True
                break

    if not shown:
        st.caption("")


# ============================================================
#                    CONSTANTS
# ============================================================
TARGET_TITLE = "What is the title of your research/capstone project?"
TARGET_ABS   = "Please provide a description or abstract of your research."

MODEL_NAME  = "llama-3.1-8b-instant"
GROQ_URL    = "https://api.groq.com/openai/v1"
TEMPERATURE = 0.1
MAX_TOKENS  = 120     # reduced for speed


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower().replace("’","'")


# ============================================================
#                    FAST LLaMA CALL
# ============================================================
def llama_score_fast(title, abstract, api_key):
    prompt = f"""
Score this project from 1–5 on:
originality, clarity, rigor, impact, entrepreneurship.

Return ONLY JSON:
{{
  "originality": number,
  "clarity": number,
  "rigor": number,
  "impact": number,
  "entrepreneurship": number,
  "feedback": "one short positive sentence"
}}

Title: {title}
Abstract: {abstract}
"""

    url = GROQ_URL + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    # only 1 retry (FAST MODE)
    for attempt in range(2):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)

            # Rate limit → retry once
            if r.status_code == 429:
                time.sleep(0.2)
                continue

            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            cleaned = re.sub(r"```json|```", "", raw).strip()

            data = json.loads(cleaned)

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
            time.sleep(0.1)

    # if failed
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
file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
if not file:
    st.stop()

df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
if df.empty:
    st.error("Uploaded file is empty.")
    st.stop()


# ============================================================
#                    COLUMN VALIDATION
# ============================================================
normmap = {norm(c): c for c in df.columns}
missing = []

if norm(TARGET_TITLE) not in normmap: missing.append(TARGET_TITLE)
if norm(TARGET_ABS) not in normmap:   missing.append(TARGET_ABS)

if missing:
    st.error("Missing required columns:\n" + "\n".join(f"• {m}" for m in missing))
    st.stop()

title_col = normmap[norm(TARGET_TITLE)]
abs_col   = normmap[norm(TARGET_ABS)]

work = df[[title_col, abs_col]].copy()
work.columns = ["title", "abstract"]

work["title"] = work["title"].astype(str).str.strip()
work["abstract"] = work["abstract"].astype(str).str.strip()

# truncate abstract for safety
work["abstract"] = work["abstract"].apply(lambda x: x[:1500])

# remove junk/short titles
work = work[work["title"].str.len() >= 3]


# ============================================================
#                    EDA — DUPLICATE CHECK
# ============================================================
st.subheader("📊 EDA — File Summary")

total_rows = len(work)
unique_titles = work["title"].nunique()
duplicate_count = total_rows - unique_titles

st.write(f"**Total rows:** {total_rows}")
st.write(f"**Unique titles:** {unique_titles}")
st.write(f"**Duplicate title entries:** {duplicate_count}")

norm_titles = work["title"].str.lower().str.replace(r"\s+"," ", regex=True)
dupes = work[norm_titles.duplicated(keep=False)]

if len(dupes) > 0:
    st.warning(f"⚠️ {len(dupes)} duplicates found. Showing duplicate rows:")
    st.dataframe(dupes, use_container_width=True)
else:
    st.success("No duplicate titles found.")

# keep only unique for scoring
work = work.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)


# ============================================================
#                    API KEY
# ============================================================
api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Enter GROQ API key", type="password")
if not api_key:
    st.stop()


# ============================================================
#                    SCORING LOOP (FAST MODE)
# ============================================================
st.subheader("⚡ Scoring with LLaMA (Fast Mode)…")

results = []
prog = st.progress(0.0)

for i, row in work.iterrows():

    results.append(llama_score_fast(row["title"], row["abstract"], api_key))

    # **FAST MODE delay**
    time.sleep(0.05)

    prog.progress((i+1)/len(work))

scored = pd.DataFrame(results)


# ============================================================
#                    API FAILURES
# ============================================================
failed = scored[scored["success"] == False]
if not failed.empty:
    st.warning(f"{len(failed)} entries FAILED during scoring.")
    st.dataframe(failed[["title","feedback"]])


# ============================================================
#                    OVERALL SCORE
# ============================================================
weights = np.ones(5) / 5
crit = scored[["originality","clarity","rigor","impact","entrepreneurship"]].to_numpy(float)
scored["overall"] = (crit @ weights).round(2)


# ============================================================
#                    TOP K
# ============================================================
st.subheader("🏆 Top Rankings")

TOP_K = st.slider("Select Top-K", 5, 50, 10, 5)
top = scored.sort_values("overall", ascending=False).reset_index(drop=True)

st.dataframe(
    top.head(TOP_K)[["title","overall","originality","clarity","rigor","impact","entrepreneurship"]],
    use_container_width=True
)

pick = st.selectbox("View details:", top.head(TOP_K)["title"].tolist())
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
#                    ALL RESULTS
# ============================================================
st.subheader("📄 All Scored Results")
display_cols = [
    "title","originality","clarity","rigor",
    "impact","entrepreneurship","overall","feedback"
]

st.dataframe(top[display_cols], use_container_width=True)


# ============================================================
#                    DOWNLOAD
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
