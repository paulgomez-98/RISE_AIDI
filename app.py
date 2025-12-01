# ============================================================
#     RISE SMART SCORING
# ============================================================

import os, re, json, requests, time, random, heapq
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="RISE Smart Scoring", layout="wide")

# ============================================================
#               1. CONSTANTS & BACKEND HELPERS
# ============================================================

TARGET_TITLE = "What is the title of your research/capstone project?"
TARGET_ABS   = "Please provide a description or abstract of your research."

MODEL_NAME = "llama-3.1-8b-instant"
GROQ_URL   = "https://api.groq.com/openai/v1"
TEMP       = 0.1
MAXTOK     = 256

def norm(s):
    """Normalize for duplicate removal."""
    s = str(s).strip().lower().replace("’","'")
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return re.sub(r"\s+", " ", s)


# ============================================================
#             2. BACKEND LOGIC — LLaMA SCORING
# ============================================================

def llama_score(title, abstract, api_key):
    """Free-tier-safe Groq scoring function."""
    
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
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMP,
        "max_tokens": MAXTOK,
    }

    # Delay for Groq Free Tier
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
                "feedback": data.get("feedback", "").strip()
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
#              3. UI — HEADER + LOGO DISPLAY
# ============================================================

left, right = st.columns([5,1])
with left:
    st.markdown("<h2>RISE — Smart Scoring & Feedback</h2>", unsafe_allow_html=True)

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
            Path("georgian_logo.jpg"),
        ]:
            if p.exists():
                st.image(str(p), use_container_width=True)
                break

st.divider()


# ============================================================
#              4. UI — FILE UPLOAD + VALIDATION
# ============================================================

file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if not file:
    st.stop()

df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
if df.empty:
    st.error("The uploaded file is empty.")
    st.stop()

# Validate columns
normmap = {norm(c): c for c in df.columns}
missing = []

for col in [TARGET_TITLE, TARGET_ABS]:
    if norm(col) not in normmap:
        missing.append(col)

if missing:
    st.error("Missing required columns:\n" + "\n".join(f"• {m}" for m in missing))
    st.stop()

title_col = normmap[norm(TARGET_TITLE)]
abs_col   = normmap[norm(TARGET_ABS)]

work = df[[title_col, abs_col]].copy()
work.columns = ["title", "abstract"]
work["abstract"] = work["abstract"].astype(str).str.strip().str[:1500]


# ============================================================
#         5. LOGIC — DUPLICATE REMOVAL
# ============================================================

work["norm"] = work["title"].apply(norm)
duplicates_df = work[work.duplicated("norm", keep="first")]
unique_df = work.drop_duplicates("norm", keep="first")[["title", "abstract"]].reset_index(drop=True)

duplicate_count = duplicates_df.shape[0]
unique_count = unique_df.shape[0]


# ============================================================
#        6. UI — SIMPLE DATASET SUMMARY (NOT FULL TABLE)
# ============================================================

st.subheader("Dataset Summary")

col1, col2 = st.columns(2)
col1.metric("Unique Applications", unique_count)
col2.metric("Duplicate Applications", duplicate_count)

st.divider()


# ============================================================
#              7. UI — API KEY INPUT
# ============================================================

api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Enter Groq API Key", type="password")
if not api_key:
    st.stop()


# ============================================================
#              8. UI — SCORING WITH LLaMA
# ============================================================

st.subheader("Scoring with LLaMA Model")
progress = st.progress(0.0)

results = []
for i, row in unique_df.iterrows():
    results.append(llama_score(row["title"], row["abstract"], api_key))
    progress.progress((i+1)/len(unique_df))


# Convert results
sc = pd.DataFrame(results)
sc[["originality","clarity","rigor","impact","entrepreneurship"]] = pd.DataFrame(sc["scores"].tolist())
sc["overall"] = sc[["originality","clarity","rigor","impact","entrepreneurship"]].mean(axis=1)


# ============================================================
#     9. UI — NOW SHOW THE TOP-K SLIDER (BELOW SCORING)
# ============================================================

TOP_K = st.slider("Select Top-K Projects", 5, 50, 10, 5)


# ============================================================
#        10. LOGIC — FAST TOP-K USING MIN-HEAP
# ============================================================

heap = [(-row.overall, row.title) for _, row in sc.iterrows()]
heapq.heapify(heap)

top_titles = []
for _ in range(min(TOP_K, len(heap))):
    score, title = heapq.heappop(heap)
    top_titles.append(title)

top_df = sc[sc["title"].isin(top_titles)].sort_values("overall", ascending=False)


# ============================================================
#      11. UI — DISPLAY RESULTS + DOWNLOADS (TOP-K + ALL)
# ============================================================

st.subheader(f"Top {TOP_K} Ranked Projects")
st.dataframe(top_df[[
    "title","overall","originality","clarity","rigor","impact","entrepreneurship"
]])

st.download_button(
    "⬇️ Download Top-K Results (CSV)",
    top_df.to_csv(index=False),
    file_name=f"rise_top_{TOP_K}.csv"
)

st.download_button(
    "⬇️ Download ALL Results (CSV)",
    sc.to_csv(index=False),
    file_name="rise_all_results.csv"
)

# ============================================================
#     12. UI — DOWNLOAD DUPLICATE APPLICATIONS (AT END)
# ============================================================

st.subheader("Download Duplicate Applications")

st.download_button(
    "⬇️ Download Duplicate Applications (CSV)",
    duplicates_df[["title", "abstract"]].to_csv(index=False),
    file_name="rise_duplicate_entries.csv"
)
