# ============================================================
#     RISE SMART SCORING — OPTIMIZED + CLEAR UI/LOGIC SPLIT
#     SAFE FOR GROQ FREE TIER
# ============================================================

import os, re, json, requests, time, random, heapq
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="RISE Smart Scoring", layout="wide")

# ============================================================
#                    1. CORE CONSTANTS & HELPERS
# ============================================================

TARGET_TITLE = "What is the title of your research/capstone project?"
TARGET_ABS   = "Please provide a description or abstract of your research."

MODEL_NAME = "llama-3.1-8b-instant"
GROQ_URL   = "https://api.groq.com/openai/v1"
TEMP       = 0.1
MAXTOK     = 256

def norm(s):
    """Normalize titles for duplicate removal."""
    s = str(s).strip().lower().replace("’","'")
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return re.sub(r"\s+", " ", s)

# ============================================================
#        2. MODEL SCORING LOGIC (NON-UI, PURE BACKEND)
# ============================================================

def llama_score(title, abstract, api_key):
    """Safe + fast Groq request optimized for free tier."""
    prompt = f"""
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

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role":"user","content":prompt}],
        "temperature": TEMP,
        "max_tokens": MAXTOK,
    }

    # Groq FREE-tier safe delay
    time.sleep(0.5 + random.random() / 3)

    for attempt in range(3):
        try:
            r = requests.post(
                GROQ_URL + "/chat/completions",
                headers=headers,
                json=payload,
                timeout=40
            )
            if r.status_code == 429:
                time.sleep(2 + attempt)
                continue

            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            cleaned = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(cleaned)

            return {
                "title": title,
                "success": True,
                "scores": [
                    float(data.get("originality", 3)),
                    float(data.get("clarity", 3)),
                    float(data.get("rigor", 3)),
                    float(data.get("impact", 3)),
                    float(data.get("entrepreneurship", 3)),
                ],
                "feedback": data.get("feedback","").strip()
            }

        except:
            time.sleep(1)

    return {
        "title": title,
        "success": False,
        "scores": [0,0,0,0,0],
        "feedback": "API ERROR"
    }


# ============================================================
#         3. UI SECTION — HEADER & LOGO
# ============================================================

left, right = st.columns([5,1])
with left:
    st.markdown("<h2>RISE — Smart Scoring & Feedback (Optimized)</h2>", unsafe_allow_html=True)
    st.write("Fast scoring → Free-tier safe → Accurate Top-K selection")

with right:
    logo_url = st.secrets.get("LOGO_URL", "")
    if logo_url:
        st.image(logo_url, use_container_width=True)

st.divider()

# ============================================================
#        4. UI — FILE UPLOAD + VALIDATION
# ============================================================

file = st.file_uploader("Upload CSV/Excel file", type=["csv","xlsx"])
if not file:
    st.stop()

df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
if df.empty:
    st.error("Uploaded file is empty.")
    st.stop()

# Validate required columns
normmap = {norm(c): c for c in df.columns}
missing = []

for col in [TARGET_TITLE, TARGET_ABS]:
    if norm(col) not in normmap:
        missing.append(col)

if missing:
    st.error("Missing required columns:\n" + "\n".join(missing))
    st.stop()

title_col = normmap[norm(TARGET_TITLE)]
abs_col   = normmap[norm(TARGET_ABS)]

work = df[[title_col, abs_col]].copy()
work.columns = ["title", "abstract"]
work["abstract"] = work["abstract"].astype(str).str.strip().str[:1500]

# ============================================================
#        5. LOGIC — DUPLICATE HANDLING (NON-UI PART)
# ============================================================

work["norm"] = work["title"].apply(norm)
duplicates = work[work.duplicated("norm", keep="first")]
unique_df = work.drop_duplicates("norm", keep="first")[["title","abstract"]].reset_index(drop=True)

# ============================================================
#        6. UI — DUPLICATE DISPLAY
# ============================================================

st.subheader("Duplicate Titles Detected")

if duplicates.empty:
    st.success("No duplicates found.")
else:
    st.warning(f"{duplicates.shape[0]} duplicate entries removed.")
    st.dataframe(duplicates[["title", "abstract"]])

st.divider()

# ============================================================
#        7. UI — API KEY INPUT
# ============================================================

api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Enter Groq API Key", type="password")
if not api_key:
    st.stop()

# ============================================================
#        8. LOGIC — SCORING LOOP (BACKEND)
# ============================================================

st.subheader("Scoring (Optimized for Groq Free Tier)")
progress = st.progress(0.0)
results = []

for i, row in unique_df.iterrows():
    results.append(llama_score(row["title"], row["abstract"], api_key))
    progress.progress((i+1)/len(unique_df))

# Transform results
sc = pd.DataFrame(results)
sc[["originality","clarity","rigor","impact","entrepreneurship"]] = pd.DataFrame(sc["scores"].tolist())
sc["overall"] = sc[["originality","clarity","rigor","impact","entrepreneurship"]].mean(axis=1)

# ============================================================
#        9. LOGIC — FAST TOP-10 USING MIN-HEAP
# ============================================================

TOP_K = 10
heap = [(-row.overall, row.title) for _, row in sc.iterrows()]
heapq.heapify(heap)

top_titles = []
for _ in range(min(TOP_K, len(heap))):
    score, title = heapq.heappop(heap)
    top_titles.append(title)

top_df = sc[sc["title"].isin(top_titles)].sort_values("overall", ascending=False)

# ============================================================
#        10. UI — DISPLAY RESULTS & DOWNLOAD
# ============================================================

st.subheader("Top 10 Projects")
st.dataframe(top_df[["title","overall","originality","clarity","rigor","impact","entrepreneurship"]])

st.download_button(
    "⬇️ Download Top-10 (CSV)",
    top_df.to_csv(index=False),
    file_name="rise_top10.csv"
)

st.download_button(
    "⬇️ Download All Results (CSV)",
    sc.to_csv(index=False),
    file_name="rise_all_results.csv"
)
