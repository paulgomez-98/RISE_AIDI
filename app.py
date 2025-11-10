# app.py — RISE Smart Scoring (Groq LLaMA via requests)
# Uses EXACT two columns you specified; robust header matching; no sidebar.

import os, re, json, requests
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------- Page & Header ----------------
st.set_page_config(page_title="RISE Smart Scoring", layout="wide")
left, right = st.columns([5, 1])
with left:
    st.markdown("<h2>RISE — Smart Scoring & Feedback</h2>", unsafe_allow_html=True)
    st.write("Upload CSV/XLSX → score with LLaMA → see Top-10 & export results.")
with right:
    # Logo from static or optional URL secret/env
    logo_url = st.secrets.get("LOGO_URL", os.getenv("LOGO_URL", ""))
    shown = False
    if logo_url:
        try:
            st.image(logo_url, use_container_width=True); shown = True
        except Exception:
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
                shown = True
                break
    if not shown:
        st.caption("")

# ---------------- Upload ----------------
file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])
if not file:
    st.stop()

# ---------------- Load file ----------------
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

# ---------------- EXACT column selection ----------------
# The two headers we MUST use:
TARGET_TITLE = "What is the title of your research/capstone project?"
TARGET_ABS   = "Please provide a description or abstract of your research."

def norm(s: str) -> str:
    """lowercase, collapse spaces, trim punctuation-like whitespace"""
    s = re.sub(r"\s+", " ", str(s)).strip().lower()
    s = s.replace("’", "'")
    return s

want = {norm(TARGET_TITLE): "title", norm(TARGET_ABS): "abstract"}

# Build a map of normalized header -> original header in file
norm2orig = {norm(c): c for c in df.columns}

missing = [orig for key, orig in [(TARGET_TITLE, TARGET_TITLE), (TARGET_ABS, TARGET_ABS)]
           if norm(orig) not in norm2orig]

if missing:
    # Try soft match (sometimes extra spaces/case changes); show helpful info
    st.error(
        "Required column(s) not found:\n\n"
        + "\n".join([f"• {m}" for m in missing])
        + "\n\nHeaders found in your file:\n"
        + "\n".join([f"- {c}" for c in df.columns])
        + "\n\nPlease ensure your Excel/CSV uses the exact headers above (minor spacing/case is okay)."
    )
    st.stop()

title_col = norm2orig[norm(TARGET_TITLE)]
abs_col   = norm2orig[norm(TARGET_ABS)]

work = df[[title_col, abs_col]].copy()
work.columns = ["title", "abstract"]

# Clean obvious junk
work["title"] = work["title"].astype(str).str.strip()
work["abstract"] = work["abstract"].astype(str).str.strip()
work = work[work["title"].str.len() >= 3]
work = work[~work["title"].str.lower().isin({"yes", "no", "true", "false"})]
work = work.reset_index(drop=True)

# ---------------- Duplicates viewer ----------------
norm_title = work["title"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True)
dupes = work[norm_title.duplicated(keep=False)]
if len(dupes) > 0:
    st.subheader("Repeated Projects")
    dup_titles = sorted(dupes["title"].unique())
    chosen = st.selectbox("Select a repeated title to view:", dup_titles)
    st.dataframe(dupes[dupes["title"] == chosen][["title", "abstract"]], use_container_width=True)

# keep first instance of each title
work = work.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

# ---------------- Secrets / runtime config ----------------
def get(name: str, default=None):
    if name in st.secrets:
        return st.secrets[name]
    v = os.getenv(name, None)
    return v if v is not None else default

def ensure_api_key():
    key = get("GROQ_API_KEY")
    if key:
        return key
    st.warning("GROQ_API_KEY not found. Paste it below to continue (kept only for this session).")
    k = st.text_input("Enter GROQ API Key", type="password")
    if k:
        st.session_state["GROQ_API_KEY"] = k
        return k
    st.stop()

GROQ_API_KEY = st.session_state.get("GROQ_API_KEY") or ensure_api_key()
GROQ_BASE_URL = get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
LLAMA_MODEL = get("LLAMA_MODEL", "llama-3.1-8b-instant")
LLAMA_TEMPERATURE = float(get("LLAMA_TEMPERATURE", 0.2))
LLAMA_MAX_TOKENS = int(get("LLAMA_MAX_TOKENS", 256))

# ---------------- LLaMA scoring via requests ----------------
def llama_score(title: str, abstract: str) -> dict:
    prompt = f"""
You are a strict judge for a student research competition (RISE).
Score the project from 1 (very weak) to 5 (excellent) on five criteria.

Return ONLY compact JSON with keys:
title, originality, clarity, rigor, impact, entrepreneurship, feedback

Definitions:
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
        # coerce numbers
        for k in ["originality", "clarity", "rigor", "impact", "entrepreneurship"]:
            data[k] = float(data.get(k, 3))
        data["title"] = str(data.get("title", title))[:300]
        data["feedback"] = str(data.get("feedback", "")).strip()
        return data
    except Exception:
        # safe fallback if model returns malformed JSON
        return {
            "title": title,
            "originality": 3.0,
            "clarity": 3.0,
            "rigor": 3.0,
            "impact": 3.0,
            "entrepreneurship": 3.0,
            "feedback": "",
        }

# ---------------- Scoring loop ----------------
st.subheader("Scoring with LLaMA (Groq)…")
results = []
prog = st.progress(0.0)
n = len(work)

for i, row in work.iterrows():
    try:
        results.append(llama_score(row["title"], row["abstract"]))
    except Exception:
        results.append({
            "title": row["title"],
            "originality": 3.0, "clarity": 3.0, "rigor": 3.0,
            "impact": 3.0, "entrepreneurship": 3.0, "feedback": ""
        })
    prog.progress((i + 1) / n)

scored = pd.DataFrame(results)
scored["overall"] = scored[["originality","clarity","rigor","impact","entrepreneurship"]].mean(axis=1).round(2)

# ---------------- Legends + Top-10 ----------------
with st.expander("Scoring Legends (1–5)", expanded=True):
    st.write(
        "- **1** Very weak • **2** Weak • **3** Adequate • **4** Strong • **5** Excellent\n"
        "- **Originality** — Novelty of ideas\n"
        "- **Clarity** — Writing quality & structure\n"
        "- **Rigor** — Soundness of method or plan\n"
        "- **Impact** — Potential real-world benefit\n"
        "- **Entrepreneurship** — Practical problem-solving & initiative"
    )

st.subheader("Top Ranks")
TOP_K = 10
top = scored.sort_values("overall", ascending=False).reset_index(drop=True)

st.markdown(f"**Top-{TOP_K} recommended**")
st.dataframe(
    top.head(TOP_K)[["title","overall","originality","clarity","rigor","impact","entrepreneurship"]],
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

# ---------------- All results + right-side downloads --------
st.subheader("All Results")
display_cols = ["title","originality","clarity","rigor","impact","entrepreneurship","overall","feedback"]
st.dataframe(top[display_cols], use_container_width=True)

spacer, dl = st.columns([0.6, 0.4])
with dl:
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download Top-10 (CSV)",
            top.head(TOP_K)[display_cols].to_csv(index=False, encoding="utf-8"),
            file_name="rise_top10.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "⬇️ Download ALL (CSV)",
            top[display_cols].to_csv(index=False, encoding="utf-8"),
            file_name="rise_all_scored.csv",
            mime="text/csv",
        )
