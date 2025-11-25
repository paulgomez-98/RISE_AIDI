# app.py — RISE Smart Scoring (Default Theme + Stable API + Logo)
# - Default Streamlit theme (no overrides)
# - Georgian logo preserved
# - Retry-safe API calls (fixes failing after few requests)
# - Gentle throttling to avoid Groq rate limits
# - Feedback for ALL applicants
# - Failed API results shown
# - No calibration
# - Single-file deployment

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
#                    CONSTANTS
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
#        SAFE + RETRYING LLaMA CALL (FIXES API FAILURES)
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

    url = GROQ_URL + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    # Retry loop — exponential backoff
    for attempt in range(5):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)

            # Rate limit handling
            if r.status_code == 429:
                time.sleep(2 ** attempt + random.random())
                continue

            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            cleaned = re.sub(r"```json|```", "", raw).strip()

            try:
                data = json.loads(cleaned)
            except:
                # Model returned malformed JSON
                raise ValueError("Bad JSON")

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
            time.sleep(2 ** attempt + random.random())

    # After all retries fail
    return {
        "success": False,
        "title": title,
        "originality": 0,
        "clarity": 0,
        "rigor": 0,
        "impact": 0,
        "entrepreneurship": 0,
        "feedback": "API ERROR — scoring failed after retries."
    }

# ============================================================
#                    FILE UPLOAD
# ============================================================
file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
if not file:
    st.stop()

df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
if df.empty:
    st.error("The uploaded file is empty.")
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
work.columns = ["title","abstract"]

work["title"] = work["title"].astype(str).str.strip()
work["abstract"] = work["abstract"].astype(str).str.strip()

# very long abstracts cause token overload → truncate safely
work["abstract"] = work["abstract"].apply(lambda x: x[:1500])

# Count duplicates BEFORE dropping
duplicate_count = work.duplicated(subset=["title"]).sum()
unique_count = len(work) - duplicate_count

# Remove duplicates
work = work.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)


# ============================================================
#                    API KEY INPUT
# ============================================================
api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Enter GROQ API Key", type="password")
if not api_key:
    st.stop()

# ============================================================
#                    SCORING LOOP  (WITH THROTTLING)
# ============================================================
st.subheader("Scoring with LLaMA…")

results = []
prog = st.progress(0.0)

for i, row in work.iterrows():
    results.append(llama_score(row["title"], row["abstract"], api_key))

    # Throttle slightly to avoid rate limits
    time.sleep(0.3 + random.random()/5)

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
st.subheader("Top Rankings")

TOP_K = st.slider("Top-K", 5, 50, 10, 5)
top = scored.sort_values("overall", ascending=False).reset_index(drop=True)

st.dataframe(
    top.head(TOP_K)[["title","overall","originality","clarity","rigor","impact","entrepreneurship"]],
    use_container_width=True
)

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
#                    SUMMARY OF DUPLICATES
# ============================================================
st.subheader("Data Summary")

col1, col2 = st.columns(2)
col1.metric("Duplicate Titles Removed", duplicate_count)
col2.metric("Unique Titles Scored", unique_count)

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

