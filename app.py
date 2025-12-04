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
#       1. CONSTANTS & HELPERS
# ============================================================

TARGET_TITLE = "What is the title of your research/capstone project?"
TARGET_ABS   = "Please provide a description or abstract of your research."

MODEL_NAME = "llama-3.1-8b-instant"
GROQ_URL   = "https://api.groq.com/openai/v1"
TEMP       = 0.1
MAXTOK     = 256

def norm(s):
    s = str(s).strip().lower().replace("’","'")
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return re.sub(r"\s+", " ", s)


# ============================================================
#       2. LLaMA SCORING FUNCTION
# ============================================================

def llama_score(title, abstract, api_key):

    prompt = f"""
Return ONLY JSON:
{{
"originality": 1-5,
"clarity": 1-5,
"rigor": 1-5,
"impact": 1-5,
"entrepreneurship": 1-5,
"feedback": "one constructive feedback sentence",
"ranking_reason": "one sentence explaining why this project deserves its ranking"
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

    time.sleep(0.5 + random.random()/3)

    for attempt in range(3):
        try:
            r = requests.post(GROQ_URL + "/chat/completions",
                              headers=headers,
                              json=payload,
                              timeout=40)

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
                "feedback": data.get("feedback", ""),
                "ranking_reason": data.get("ranking_reason", "")
            }

        except:
            time.sleep(1)

    return {
        "title": title,
        "success": False,
        "scores": [0,0,0,0,0],
        "feedback": "API ERROR",
        "ranking_reason": "No ranking explanation available (API failure)."
    }


# ============================================================
#   3. HEADER + LOGO
# ============================================================

left, right = st.columns([5,1])
with left:
    st.markdown("<h2>RISE — Smart Scoring & Feedback</h2>", unsafe_allow_html=True)

with right:
    logo_path = "static/georgian_logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
        except:
            pass

st.divider()


# ============================================================
#   4. FILE UPLOAD & VALIDATION
# ============================================================

file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if not file:
    st.stop()

df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)

if df.empty:
    st.error("Uploaded file is empty.")
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

# Extract correct columns
title_col = normmap[norm(TARGET_TITLE)]
abs_col   = normmap[norm(TARGET_ABS)]

work = df[[title_col, abs_col]].copy()
work.columns = ["title", "abstract"]
work["abstract"] = work["abstract"].astype(str).str.strip().str[:1500]


# ============================================================
#   5. DUPLICATE REMOVAL
# ============================================================

work["norm"] = work["title"].apply(norm)
duplicates_df = work[work.duplicated("norm", keep="first")]
unique_df = work.drop_duplicates("norm", keep="first")[["title", "abstract"]].reset_index(drop=True)

st.subheader("Dataset Summary")
col1, col2 = st.columns(2)
col1.metric("Unique Applications", unique_df.shape[0])
col2.metric("Duplicate Applications", duplicates_df.shape[0])

st.divider()


# ============================================================
#   6. API KEY
# ============================================================

api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Enter Groq API Key", type="password")
if not api_key:
    st.stop()


# ============================================================
#   7. SCORING
# ============================================================

@st.cache_data(show_spinner=False)
def score_all(unique_df, api_key):
    results = []
    progress = st.progress(0.0)

    for i, row in unique_df.iterrows():
        results.append(llama_score(row["title"], row["abstract"], api_key))
        progress.progress((i+1)/len(unique_df))

    return pd.DataFrame(results)

sc = score_all(unique_df, api_key)

st.divider()


# ============================================================
#   8. RUBRIC CALCULATIONS
# ============================================================

sc[["originality","clarity","rigor","impact","entrepreneurship"]] = pd.DataFrame(sc["scores"].tolist())
sc["overall"] = sc[["originality","clarity","rigor","impact","entrepreneurship"]].mean(axis=1)

TOP_K = st.slider("Select Top-K Projects", 5, 50, 10, 5)

heap = [(-row.overall, row.title) for _, row in sc.iterrows()]
heapq.heapify(heap)
top_titles = [heapq.heappop(heap)[1] for _ in range(min(TOP_K, len(heap)))]
top_df = sc[sc["title"].isin(top_titles)].sort_values("overall", ascending=False)


# ============================================================
#   9. RUBRIC DEFINITIONS
# ============================================================

st.markdown("""
### Rubric Criteria Explained

**Originality** – Uniqueness and creativity of the idea.  
**Clarity** – Abstract is structured, clear and easy to understand.  
**Rigor** – Strength of methodology and research depth.  
**Impact** – Potential real-world usefulness of the idea.  
**Entrepreneurship** – Potential for innovation or commercialization.
""")


# ============================================================
#   10. TABLE
# ============================================================

st.subheader(f"Top {TOP_K} Ranked Projects")

st.dataframe(
    top_df[[
        "title","overall","originality","clarity","rigor","impact","entrepreneurship"
    ]],
    use_container_width=True,
    hide_index=True
)

selected_title = st.selectbox(
    "Select a project to view full details:",
    options=top_df["title"].tolist()
)

this = top_df[top_df["title"] == selected_title].iloc[0]


# ============================================================
#   11. DETAILS AREA — ALWAYS UPDATES NOW
# ============================================================

st.markdown("---")
st.subheader("Project Details")

st.markdown(f"""
### **{this['title']}**
**Overall Score:** {round(this['overall'], 2)}

**Rubric Breakdown**
- Originality: {this['originality']}
- Clarity: {this['clarity']}
- Rigor: {this['rigor']}
- Impact: {this['impact']}
- Entrepreneurship: {this['entrepreneurship']}

---

### 📝 Why This Project Ranked Highly  
**{this['ranking_reason']}**

---

### 💬 Constructive Feedback  
{this['feedback']}
""")

st.divider()


# ============================================================
#   12. DOWNLOAD BUTTONS
# ============================================================

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

st.subheader("Download Duplicate Applications")

st.download_button(
    "⬇️ Download Duplicate Applications (CSV)",
    duplicates_df[["title","abstract"]].to_csv(index=False),
    file_name="rise_duplicate_entries.csv"
)


