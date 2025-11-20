# ============================================
# RISE — Smart Scoring & Feedback (FAST Batch Version)
# Batch size = 10 (≈ 20–35 sec for 200 rows)
# Stable JSON parsing + retries
# EDA + logo + clean UI
# ============================================

import os, re, json, requests, time, random
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# --------------------------------------------
# PAGE CONFIG
# --------------------------------------------
st.set_page_config(page_title="RISE Smart Scoring", layout="wide")

# --------------------------------------------
# HEADER + LOGO
# --------------------------------------------
left, right = st.columns([5,1])
with left:
    st.markdown("<h2>RISE — Smart Scoring & Feedback</h2>", unsafe_allow_html=True)
    st.write("Upload → EDA → Batch Score using LLaMA → Top-K → Download")

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

# --------------------------------------------
# CONSTANTS
# --------------------------------------------
TARGET_TITLE = "What is the title of your research/capstone project?"
TARGET_ABS   = "Please provide a description or abstract of your research."

MODEL_NAME  = "llama-3.1-8b-instant"
GROQ_URL    = "https://api.groq.com/openai/v1"
BATCH_SIZE  = 10
MAX_TOKENS  = 512

def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower().replace("’","'")

# --------------------------------------------
# BATCH LLaMA CALL
# --------------------------------------------
def score_batch(batch, api_key):
    """
    batch = list of dicts:
    [{'title': ..., 'abstract': ...}, ...]
    """

    # Build JSON array for model to score
    items_json = json.dumps(batch, ensure_ascii=False)

    prompt = f"""
You are evaluating several student research projects.

Return ONLY valid JSON list.
For EACH item, return an object with this exact structure:

[
  {{
    "title": "...",
    "originality": 1-5,
    "clarity": 1-5,
    "rigor": 1-5,
    "impact": 1-5,
    "entrepreneurship": 1-5,
    "feedback": "Start with a positive sentence. Add a short suggestion only if needed."
  }}
]

Now evaluate these items:
{items_json}
"""

    url = GROQ_URL + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": MAX_TOKENS
    }

    for attempt in range(4):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)

            if r.status_code == 429:
                time.sleep(1.5 + random.random())
                continue

            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            cleaned = re.sub(r"```json|```", "", raw).strip()

            try:
                data = json.loads(cleaned)
                return data
            except:
                # JSON malformed → retry
                time.sleep(1 + random.random())

        except:
            time.sleep(1 + random.random())

    # If everything fails → return fallback empty results
    fallback = []
    for item in batch:
        fallback.append({
            "title": item["title"],
            "originality": 0,
            "clarity": 0,
            "rigor": 0,
            "impact": 0,
            "entrepreneurship": 0,
            "feedback": "API ERROR — batch scoring failed after retries."
        })
    return fallback

# --------------------------------------------
# FILE UPLOAD
# --------------------------------------------
file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
if not file:
    st.stop()

df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

if df.empty:
    st.error("The uploaded file is empty.")
    st.stop()

# --------------------------------------------
# COLUMN VALIDATION
# --------------------------------------------
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

# Clean
work["title"] = work["title"].astype(str).str.strip()
work["abstract"] = work["abstract"].astype(str).str.strip()
work["abstract"] = work["abstract"].apply(lambda x: x[:1500])

# --------------------------------------------
# EDA — Duplicates
# --------------------------------------------
st.subheader("📊 EDA — File Summary")

total_rows = len(work)
unique_titles = work["title"].nunique()
duplicate_entries = total_rows - unique_titles

st.markdown(f"""
**Total rows:** {total_rows}  
**Unique titles:** {unique_titles}  
**Duplicate title entries:** {duplicate_entries}  
""")

dupe_mask = work["title"].str.lower().duplicated(keep=False)
dupes = work[dupe_mask]

if len(dupes) > 0:
    st.warning(f"{duplicate_entries} duplicates found. Showing duplicate rows below:")
    st.dataframe(dupes, use_container_width=True)
else:
    st.success("No duplicate titles found.")

# Remove duplicates before scoring
work = work.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

# --------------------------------------------
# API KEY
# --------------------------------------------
api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Enter GROQ API Key", type="password")
if not api_key:
    st.stop()

# --------------------------------------------
# BATCH SCORING
# --------------------------------------------
st.subheader("⚡ Scoring with LLaMA (Batch Mode)…")

results = []
prog = st.progress(0.0)

total_batches = (len(work) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(total_batches):
    batch_df = work.iloc[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

    batch_items = batch_df.to_dict("records")
    scored_batch = score_batch(batch_items, api_key)

    results.extend(scored_batch)
    prog.progress((batch_idx + 1) / total_batches)

scored = pd.DataFrame(results)

# --------------------------------------------
# FAILURE REPORT
# --------------------------------------------
failed = scored[(scored["originality"] == 0) & (scored["clarity"] == 0)]
if not failed.empty:
    st.warning(f"{len(failed)} entries FAILED during scoring.")
    st.dataframe(failed[["title","feedback"]])

# --------------------------------------------
# OVERALL SCORE
# --------------------------------------------
weights = np.ones(5) / 5
crit = scored[["originality","clarity","rigor","impact","entrepreneurship"]].to_numpy(float)
scored["overall"] = (crit @ weights).round(2)

# --------------------------------------------
# TOP-K RESULTS
# --------------------------------------------
st.subheader("🏆 Top Rankings")

TOP_K = st.slider("Select Top-K", 5, 50, 10, 5)
top = scored.sort_values("overall", ascending=False).reset_index(drop=True)

st.dataframe(
    top.head(TOP_K)[["title","overall","originality","clarity","rigor","impact","entrepreneurship"]],
    use_container_width=True
)

pick = st.selectbox("View details of:", top.head(TOP_K)["title"].tolist())
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

# --------------------------------------------
# ALL RESULTS
# --------------------------------------------
st.subheader("📄 All Scored Results")

display_cols = [
    "title","originality","clarity","rigor",
    "impact","entrepreneurship","overall","feedback"
]

st.dataframe(top[display_cols], use_container_width=True)

# --------------------------------------------
# DOWNLOAD BUTTONS
# --------------------------------------------
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
