# ============================================================
# app.py — RISE Smart Scoring (Batched + Parallel + Fast + Stable)
# ============================================================
# - Default Streamlit theme
# - Georgian logo preserved
# - Batch scoring (5 per request)
# - Parallel execution (10 threads)
# - Retry-safe API calls
# - Basic EDA (duplicates only)
# - Detailed paragraph feedback
# - No calibration
# - Single-file deployment
# ============================================================

import os, re, json, requests, time, random
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="RISE Smart Scoring", layout="wide")

# ============================================================
# HEADER + LOGO
# ============================================================
left, right = st.columns([5,1])
with left:
    st.markdown("<h2>RISE — Smart Scoring & Feedback</h2>", unsafe_allow_html=True)
    st.write("Upload → Check Duplicates → Score (Fast Mode) → Results → Download")

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
# CONSTANTS
# ============================================================
TARGET_TITLE = "What is the title of your research/capstone project?"
TARGET_ABS   = "Please provide a description or abstract of your research."

MODEL_NAME  = "llama-3.1-8b-instant"
GROQ_URL    = "https://api.groq.com/openai/v1"
TEMPERATURE = 0.1
MAX_TOKENS  = 512   # Increased for batch feedback
BATCH_SIZE  = 5
MAX_THREADS = 10

def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower().replace("’","'")

# ============================================================
# BATCHED PROMPT BUILDER
# ============================================================
def build_batch_prompt(batch):
    items = []
    for entry in batch:
        items.append({
            "title": entry["title"],
            "abstract": entry["abstract"]
        })

    # JSON list with title included — as requested
    return f"""
You are evaluating a batch of student research projects. 
For EACH project, return ONE detailed JSON object inside a JSON LIST.

Rules for feedback:
- Begin with a positive, encouraging sentence
- Provide 2–3 sentences of guidance
- Suggestions ONLY if needed
- Keep tone supportive and helpful
- DO NOT add extra text outside the JSON list

Return ONLY a JSON list like this:
[
  {{
    "title": "...",
    "originality": 1-5,
    "clarity": 1-5,
    "rigor": 1-5,
    "impact": 1-5,
    "entrepreneurship": 1-5,
    "feedback": "3–5 sentence constructive paragraph"
  }},
  ...
]

Projects to score:
{json.dumps(items, indent=2)}
"""

# ============================================================
# SAFE API CALL (BATCHED)
# ============================================================
def llama_score_batch(batch, api_key):

    prompt = build_batch_prompt(batch)

    url = GROQ_URL + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    for attempt in range(5):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)

            if r.status_code == 429:
                time.sleep(2 ** attempt + random.random())
                continue

            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            cleaned = re.sub(r"```json|```", "", raw).strip()

            data_list = json.loads(cleaned)

            results = []
            for obj in data_list:
                results.append({
                    "success": True,
                    "title": obj.get("title", ""),
                    "originality": float(obj.get("originality", 3)),
                    "clarity": float(obj.get("clarity", 3)),
                    "rigor": float(obj.get("rigor", 3)),
                    "impact": float(obj.get("impact", 3)),
                    "entrepreneurship": float(obj.get("entrepreneurship", 3)),
                    "feedback": obj.get("feedback", "").strip()
                })

            return results

        except Exception:
            time.sleep(2 ** attempt + random.random())

    # On failure, return fallback entries for each in batch
    return [{
        "success": False,
        "title": x["title"],
        "originality": 0,
        "clarity": 0,
        "rigor": 0,
        "impact": 0,
        "entrepreneurship": 0,
        "feedback": "API ERROR — batch scoring failed after retries."
    } for x in batch]

# ============================================================
# FILE UPLOAD
# ============================================================
file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
if not file:
    st.stop()

df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
if df.empty:
    st.error("The uploaded file is empty.")
    st.stop()

# ============================================================
# COLUMN VALIDATION
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

# Clean
work["title"] = work["title"].astype(str).str.strip()
work["abstract"] = work["abstract"].astype(str).str.strip()
work["abstract"] = work["abstract"].apply(lambda x: x[:1500])  # Safety

# ============================================================
# BASIC EDA
# ============================================================
st.subheader("📊 File Summary (Before Scoring)")

total_rows = len(work)
unique_titles = work["title"].nunique()

st.markdown(f"**Total rows:** {total_rows}  \n**Unique titles:** {unique_titles}")

norm_titles = work["title"].str.lower().str.replace(r"\s+"," ", regex=True)
dupes = work[norm_titles.duplicated(keep=False)]

if len(dupes) > 0:
    st.warning(f"⚠️ Duplicate project titles detected ({len(dupes)} entries):")
    st.dataframe(dupes, use_container_width=True)
else:
    st.success("No duplicate project titles found.")

# Remove duplicates for scoring
work = work.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

# ============================================================
# API KEY
# ============================================================
api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Enter GROQ API Key", type="password")
if not api_key:
    st.stop()

# ============================================================
# BATCHING + PARALLEL SCORING
# ============================================================
st.subheader("🔄 Scoring with LLaMA (Fast Mode)…")

batches = [
    work.iloc[i:i+BATCH_SIZE].to_dict(orient="records")
    for i in range(0, len(work), BATCH_SIZE)
]

results = []
prog = st.progress(0.0)

with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = {executor.submit(llama_score_batch, batch, api_key): batch for batch in batches}

    completed = 0
    for future in as_completed(futures):
        batch_results = future.result()
        results.extend(batch_results)
        completed += 1
        prog.progress(completed / len(batches))

scored = pd.DataFrame(results)

# ============================================================
# API FAILURES
# ============================================================
failed = scored[scored["success"] == False]
if not failed.empty:
    st.warning(f"{len(failed)} entries FAILED during scoring.")
    st.dataframe(failed[["title","feedback"]])

# ============================================================
# OVERALL SCORE
# ============================================================
weights = np.ones(5) / 5
crit = scored[["originality","clarity","rigor","impact","entrepreneurship"]].to_numpy(float)
scored["overall"] = (crit @ weights).round(2)

# ============================================================
# TOP-K
# ============================================================
st.subheader("🏆 Top Rankings")

TOP_K = st.slider("Select Top-K", 5, 50, 10, 5)
top = scored.sort_values("overall", ascending=False).reset_index(drop=True)

st.dataframe(top.head(TOP_K)[[
    "title","overall","originality","clarity","rigor","impact","entrepreneurship"
]])

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

# ============================================================
# ALL RESULTS
# ============================================================
st.subheader("📄 All Scored Results")

display_cols = [
    "title","originality","clarity","rigor",
    "impact","entrepreneurship","overall","feedback"
]

st.dataframe(top[display_cols], use_container_width=True)

# ============================================================
# DOWNLOAD BUTTONS
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
