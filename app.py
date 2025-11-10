# app.py — RISE Smart Scoring (Groq LLaMA via requests)
# - Uses the EXACT two headers you specified
# - Never changes the CSV title
# - Top-K slider (5..50)
# - Optional "Judge Calibration": learn criterion weights from pasted winners

import os, re, json, requests
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------- Page & Header ----------------
st.set_page_config(page_title="RISE Smart Scoring", layout="wide")
left, right = st.columns([5, 1])
with left:
    st.markdown("<h2>RISE — Smart Scoring & Feedback</h2>", unsafe_allow_html=True)
    st.write("Upload CSV/XLSX → score with LLaMA → (optionally) calibrate to judge picks → see Top-K & export.")
with right:
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
                st.image(str(p), use_container_width=True); shown = True; break
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
TARGET_TITLE = "What is the title of your research/capstone project?"
TARGET_ABS   = "Please provide a description or abstract of your research."

def norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip().lower()
    s = s.replace("’", "'")
    return s

norm2orig = {norm(c): c for c in df.columns}
missing = []
if norm(TARGET_TITLE) not in norm2orig:
    missing.append(TARGET_TITLE)
if norm(TARGET_ABS) not in norm2orig:
    missing.append(TARGET_ABS)

if missing:
    st.error(
        "Required column(s) not found:\n\n" + "\n".join([f"• {m}" for m in missing]) +
        "\n\nHeaders in your file:\n" + "\n".join([f"- {c}" for c in df.columns])
    )
    st.stop()

title_col = norm2orig[norm(TARGET_TITLE)]
abs_col   = norm2orig[norm(TARGET_ABS)]

work = df[[title_col, abs_col]].copy()
work.columns = ["title", "abstract"]

# Clean & dedupe titles
work["title"] = work["title"].astype(str).str.strip()
work["abstract"] = work["abstract"].astype(str).str.strip()
work = work[work["title"].str.len() >= 3]
work = work[~work["title"].str.lower().isin({"yes", "no", "true", "false"})]
# duplicate viewer
norm_title = work["title"].str.lower().str.replace(r"\s+", " ", regex=True)
dupes = work[norm_title.duplicated(keep=False)]
if len(dupes) > 0:
    st.subheader("Repeated Projects")
    choice = st.selectbox("Select a repeated title to view:", sorted(dupes["title"].unique()))
    st.dataframe(dupes[dupes["title"] == choice][["title", "abstract"]], use_container_width=True)
# keep first
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
LLAMA_TEMPERATURE = float(get("LLAMA_TEMPERATURE", 0.1))  # lower for stability
LLAMA_MAX_TOKENS = int(get("LLAMA_MAX_TOKENS", 256))

# ---------------- LLaMA scoring via requests ----------------
def llama_score(title: str, abstract: str) -> dict:
    prompt = f"""
You are scoring a student research project.

Return ONLY a compact JSON object like this (no extra text):
{{
  "originality": 1-5 number,
  "clarity": 1-5 number,
  "rigor": 1-5 number,
  "impact": 1-5 number,
  "entrepreneurship": 1-5 number,
  "feedback": "one short feedback sentence"
}}

DO NOT return a 'title' field.

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
        "top_p": 0.9
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    content = re.sub(r"```json|```", "", content).strip()

    try:
        data = json.loads(content)
        for k in ["originality", "clarity", "rigor", "impact", "entrepreneurship"]:
            data[k] = float(data.get(k, 3))
        data["title"] = title  # keep CSV title
        data["feedback"] = str(data.get("feedback", "")).strip()
        return data
    except Exception:
        return {
            "title": title,
            "originality": 3.0, "clarity": 3.0, "rigor": 3.0,
            "impact": 3.0, "entrepreneurship": 3.0, "feedback": ""
        }

# ---------------- Scoring loop ----------------
st.subheader("Scoring with LLaMA (Groq)…")
results, n = [], len(work)
prog = st.progress(0.0)
for i, row in work.iterrows():
    try:
        results.append(llama_score(row["title"], row["abstract"]))
    except Exception:
        results.append({
            "title": row["title"],
            "originality": 3.0, "clarity": 3.0, "rigor": 3.0,
            "impact": 3.0, "entrepreneurship": 3.0, "feedback": ""
        })
    prog.progress((i + 1) / max(n, 1))

scored = pd.DataFrame(results)

# ---------------- Optional Judge Calibration ----------------
with st.expander("Optional: Calibrate to judge-picked winners"):
    st.caption("Paste the winners' *exact* titles (one per line). We'll learn criterion weights to match judge taste.")
    gold_text = st.text_area("Judge winners (one title per line):", height=140,
        placeholder="A Comparative Study of Modern Philosophical Thought\nAnalyzing the Vulnerabilities in IoT Device Security\n…")
    use_calibration = st.checkbox("Apply calibration to re-rank", value=False)

# Build normalized map for title matching
def norm_title_str(s): return re.sub(r"\s+", " ", str(s)).strip().lower()
scored["_norm_title"] = scored["title"].map(norm_title_str)

# default weights (equal)
weights = np.array([1, 1, 1, 1, 1], dtype=float)
weights = weights / weights.sum()

if use_calibration and gold_text.strip():
    winners = [norm_title_str(x) for x in gold_text.strip().splitlines() if x.strip()]
    if len(winners) >= 3:
        scored["is_winner"] = scored["_norm_title"].isin(winners).astype(int)
        X = scored[["originality","clarity","rigor","impact","entrepreneurship"]].to_numpy(dtype=float)
        # normalize columns to [0,1] for stability
        Xmin, Xmax = X.min(axis=0), X.max(axis=0)
        rng = np.where((Xmax - Xmin) == 0, 1.0, (Xmax - Xmin))
        Xn = (X - Xmin) / rng
        y = scored["is_winner"].to_numpy(dtype=float)
        # least-squares weights
        try:
            w, *_ = np.linalg.lstsq(Xn, y, rcond=None)
            w = np.clip(w, 0, None)
            if w.sum() > 0:
                weights = w / w.sum()
        except Exception:
            pass
        st.caption(f"Learned weights → Orig:{weights[0]:.2f}  Clar:{weights[1]:.2f}  Rigor:{weights[2]:.2f}  Impact:{weights[3]:.2f}  Entr:{weights[4]:.2f}")
    else:
        st.info("Enter at least 3 winner titles to calibrate.")

# overall score (either equal or calibrated weights)
crit = scored[["originality","clarity","rigor","impact","entrepreneurship"]].to_numpy(dtype=float)
scored["overall"] = (crit @ weights).round(2) * 5 / (weights.sum() if weights.sum() else 1)

# ---------------- Legends ----------------
with st.expander("Scoring Legends (1–5)", expanded=True):
    st.write(
        "- **1** Very weak • **2** Weak • **3** Adequate • **4** Strong • **5** Excellent\n"
        "- **Originality** — Novelty of ideas\n"
        "- **Clarity** — Writing quality & structure\n"
        "- **Rigor** — Soundness of method or plan\n"
        "- **Impact** — Potential real-world benefit\n"
        "- **Entrepreneurship** — Practical problem-solving & initiative"
    )

# ---------------- Top-K slider + display ----------------
st.subheader("Top Ranks")
TOP_K = st.slider("How many top ranks to display?", min_value=5, max_value=50, value=10, step=5)

top = scored.sort_values("overall", ascending=False).reset_index(drop=True)
st.markdown(f"**Top-{TOP_K} recommended**")
st.dataframe(
    top.head(TOP_K)[["title","overall","originality","clarity","rigor","impact","entrepreneurship"]],
    use_container_width=True,
)

if len(top) > 0:
    pick = st.selectbox("View scores for a Top-K project:", top.head(TOP_K)["title"].tolist())
    sel = top[top["title"] == pick].iloc[0]
    st.write(
        f"**Title:** {sel['title']}\n\n"
        f"**Overall:** {sel['overall']:.2f}  |  "
        f"Originality {sel['originality']:.1f} • Clarity {sel['clarity']:.1f} • "
        f"Rigor {sel['rigor']:.1f} • Impact {sel['impact']:.1f} • "
        f"Entrepreneurship {sel['entrepreneurship']:.1f}"
    )
    if str(sel.get("feedback", "")).strip():
        st.caption(f"Feedback: {sel['feedback']}")

# ---------------- All results + downloads (right) --------
st.subheader("All Results")
display_cols = ["title","originality","clarity","rigor","impact","entrepreneurship","overall","feedback"]
st.dataframe(top[display_cols], use_container_width=True)

spacer, dl = st.columns([0.6, 0.4])
with dl:
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download Top-K (CSV)",
            top.head(TOP_K)[display_cols].to_csv(index=False, encoding="utf-8"),
            file_name=f"rise_top_{TOP_K}.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "⬇️ Download ALL (CSV)",
            top[display_cols].to_csv(index=False, encoding="utf-8"),
            file_name="rise_all_scored.csv",
            mime="text/csv",
        )
