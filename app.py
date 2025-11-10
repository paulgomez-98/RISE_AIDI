# app.py — RISE Smart Scoring (Groq LLaMA via requests, no openai SDK)

import re
import json
import requests
import pandas as pd
import streamlit as st
from pathlib import Path

# ================= Page config =================
st.set_page_config(page_title="RISE Smart Scoring", layout="wide")

# ================= Header ======================
left, right = st.columns([5, 1])
with left:
    st.markdown("<h2>RISE — Smart Scoring & Feedback</h2>", unsafe_allow_html=True)
    st.write("Upload CSV/XLSX → score with LLaMA → see Top-10 & export results.")
with right:
    # Robust logo load (static path or optional URL in secrets)
    logo_url = st.secrets.get("LOGO_URL", "")
    candidates = [
        Path("static/georgian_logo.png"),
        Path("static/georgian_logo.jpg"),
        Path("georgian_logo.png"),
        Path("georgian_logo.jpg"),
    ]
    try:
        if logo_url:
            st.image(logo_url, use_container_width=True)
        else:
            shown = False
            for p in candidates:
                if p.exists():
                    st.image(str(p), use_container_width=True)
                    shown = True
                    break
            if not shown:
                st.caption("")  # keep layout clean
    except Exception:
        st.caption("")

# ================= Upload ======================
file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])
if not file:
    st.stop()

# ================= Load file ===================
try:
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

if df.empty:
    st.warning("The uploaded file appears to be empty.")
    st.stop()

# ================= Column detection (auto) ===================
def pick(colnames, needles):
    for c in colnames:
        lc = str(c).lower()
        if any(n in lc for n in needles):
            return c
    return None

title_col = pick(df.columns, ["title", "capstone"])
abs_col   = pick(df.columns, ["abstract", "description"])

if title_col is None or abs_col is None:
    st.error(
        "Your file must contain columns for **title** and **abstract/description**.\n\n"
        "Examples:\n- 'What is the title of your research/capstone project?'\n"
        "- 'Please provide a description or abstract of your research.'"
    )
    st.stop()

work = df[[title_col, abs_col]].copy()
work.columns = ["title", "abstract"]

# ================= Duplicate detection ======================
norm_title = work["title"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True)
dupes = work[norm_title.duplicated(keep=False)]
if len(dupes) > 0:
    st.subheader("Repeated Projects")
    dup_titles = sorted(dupes["title"].unique())
    chosen = st.selectbox("Select a repeated title to view:", dup_titles)
    if chosen:
        st.dataframe(dupes[dupes["title"] == chosen][["title", "abstract"]], use_container_width=True)

# keep first instance of each title
work = work.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

# ================= Secrets (Groq) ===========================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
GROQ_BASE_URL = st.secrets.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
LLAMA_MODEL = st.secrets.get("LLAMA_MODEL", "llama-3.1-8b-instant")
LLAMA_TEMPERATURE = float(st.secrets.get("LLAMA_TEMPERATURE", 0.2))
LLAMA_MAX_TOKENS = int(st.secrets.get("LLAMA_MAX_TOKENS", 256))

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing in Secrets.")
    st.stop()

# ================= LLaMA call (OpenAI-compatible /chat/completions) =========
def llama_score(title: str, abstract: str) -> dict:
    prompt = f"""
You are a strict judge for a student research competition (RISE).
Score the project 1 (very weak) to 5 (excellent) on five criteria.

Return ONLY compact JSON with these keys:
title, originality, clarity, rigor, impact, entrepreneurship, feedback

Where:
- originality: novelty of ideas
- clarity: writing quality & structure
- rigor: soundness of method/plan
- impact: potential real-world benefit
- entrepreneurship: practical problem-solving initiative

Title: {title}
Abstract: {abstract}
"""

    url = GROQ_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": LLAMA_TEMPERATURE,
        "max_tokens": LLAMA_MAX_TOKENS,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    content = re.sub(r"```json|```", "", content).strip()

    try:
        data = json.loads(content)
        # Ensure numeric fields are floats 1..5
        for k in ["originality", "clarity", "rigor", "impact", "entrepreneurship"]:
            data[k] = float(data.get(k, 3))
        data["title"] = str(data.get("title", title))[:300]
        data["feedback"] = str(data.get("feedback", "")).strip()
        return data
    except Exception:
        # If the model didn't return proper JSON, provide a safe fallback
        return {
            "title": title,
            "originality": 3.0,
            "clarity": 3.0,
            "rigor": 3.0,
            "impact": 3.0,
            "entrepreneurship": 3.0,
            "feedback": "",
        }

# ================= Scoring loop ============================
st.subheader("Scoring with LLaMA (Groq)…")
results = []
prog = st.progress(0.0)
total = len(work)

for i, row in work.iterrows():
    try:
        scored = llama_score(row["title"], row["abstract"])
    except Exception as e:
        # network / rate limit fallback
        scored = {
            "title": row["title"],
            "originality": 3.0,
            "clarity": 3.0,
            "rigor": 3.0,
            "impact": 3.0,
            "entrepreneurship": 3.0,
            "feedback": "",
        }
    results.append(scored)
    prog.progress((i + 1) / total)

scored_df = pd.DataFrame(results)
scored_df["overall"] = scored_df[["originality", "clarity", "rigor", "impact", "entrepreneurship"]].mean(axis=1).round(2)

# ================= Legends ================================
with st.expander("Scoring Legends (1–5)", expanded=True):
    st.write(
        "- **1** Very weak • **2** Weak • **3** Adequate • **4** Strong • **5** Excellent\n"
        "- **Originality** — Novelty of ideas\n"
        "- **Clarity** — Writing quality & structure\n"
        "- **Rigor** — Soundness of method or plan\n"
        "- **Impact** — Potential real-world benefit\n"
        "- **Entrepreneurship** — Practical problem-solving & initiative"
    )

# ================= Top-10 first ============================
st.subheader("Top Ranks")
TOP_K = 10
top = scored_df.sort_values("overall", ascending=False).reset_index(drop=True)

st.markdown(f"**Top-{TOP_K} recommended**")
st.dataframe(
    top.head(TOP_K)[["title", "overall", "originality", "clarity", "rigor", "impact", "entrepreneurship"]],
    use_container_width=True,
)

if len(top) > 0:
    pick = st.selectbox("View scores for a Top-10 project:", top.head(TOP_K)["title"].tolist())
    sel = top[top["title"] == pick].iloc[0]
    st.write(
        f"**Overall:** {sel['overall']}  |  "
        f"Originality {sel['originality']:.1f} • Clarity {sel['clarity']:.1f} • "
        f"Rigor {sel['rigor']:.1f} • Impact {sel['impact']:.1f} • "
        f"Entrepreneurship {sel['entrepreneurship']:.1f}"
    )
    if str(sel.get("feedback", "")).strip():
        st.caption(f"Feedback: {sel['feedback']}")

# ================= All results + downloads on right =========
st.subheader("All Results")
st.dataframe(
    top[["title", "originality", "clarity", "rigor", "impact", "entrepreneurship", "overall", "feedback"]],
    use_container_width=True,
)

spacer, dl = st.columns([0.6, 0.4])
with dl:
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download Top-10 (CSV)",
            top.head(TOP_K)[["title", "originality", "clarity", "rigor", "impact", "entrepreneurship", "overall", "feedback"]]
            .to_csv(index=False, encoding="utf-8"),
            file_name="rise_top10.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "⬇️ Download ALL (CSV)",
            top[["title", "originality", "clarity", "rigor", "impact", "entrepreneurship", "overall", "feedback"]]
            .to_csv(index=False, encoding="utf-8"),
            file_name="rise_all_scored.csv",
            mime="text/csv",
        )
