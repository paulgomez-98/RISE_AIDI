# app.py — RISE Smart Scoring & Feedback (Groq-hosted LLaMA, simplified UI)

import os, re, json, textwrap, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ========================= Page / constants =========================
st.set_page_config(page_title="RISE Smart Scoring", page_icon="", layout="wide")

REQ_TITLE = "What is the title of your research/capstone project?"
REQ_ABS   = "Please provide a description or abstract of your research."

RUBRIC = {
    "originality":      "Does the project introduce unique or innovative ideas and stand out from common solutions?",
    "clarity":          "Is the abstract well-organized, concise, and easy to follow with clear objectives and methods?",
    "rigor":            "Are the methods realistic, well-scoped, and aligned to the goals?",
    "impact":           "Does the project address a meaningful problem with potential real-world benefit?",
    "entrepreneurship": "Does it show problem-solving and practical planning to overcome challenges?",
}

SYSTEM_PROMPT = """You are a strict academic judge. Score student abstracts 1-5 on the criteria given.
Return ONLY compact JSON with numeric fields from 1 to 5, no extra text.
The JSON keys must be: originality, clarity, rigor, impact, entrepreneurship, and rationale.
'rationale' is a short (<= 40 words) reason for the scores."""

USER_TMPL = """Title: {title}
Abstract: {abstract}

Score this project 1–5 on:
- originality
- clarity
- rigor
- impact
- entrepreneurship

Return JSON like:
{{"originality": 3, "clarity": 4, "rigor": 3, "impact": 4, "entrepreneurship": 3, "rationale":"..."}}
"""

CUSTOM_CSS = """
<style>
.block-container {max-width: 1200px !important;}
h1, h2, h3 {letter-spacing: .2px;}
.header {display:flex; align-items:center; justify-content:space-between; padding: 10px 0 12px 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 10px;}
.header .title {font-size: 28px; font-weight: 700; margin: 0;}
.header .subtitle {margin: 0; color: #6b7280;}
.badge {display:inline-block; padding:4px 8px; border-radius:20px; background:#eef2ff; color:#3730a3; font-size:12px; margin-left:8px;}
.dataframe td, .dataframe th {padding: 6px 10px;}
.logo-box {background:#ffffff; padding:8px 10px; border:1px solid #e5e7eb; border-radius:8px;}
.small-note {font-size:12px; color:#6b7280}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ========================= Header w/ logo ===========================
col_left, col_right = st.columns([0.72, 0.28])
with col_left:
    st.markdown(
        "<div class='header'>"
        "<div><div class='title'>RISE — Smart Scoring & Feedback</div>"
        "<p class='subtitle'>Upload CSV/XLSX and get ranked projects with rubric-based scores and feedback.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
with col_right:
    st.markdown("<div class='logo-box'>", unsafe_allow_html=True)
    # Place your logo at: ./static/georgian_logo.png
    st.image("georgian_logo.png", caption=None, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ========================= Upload ==============================
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if not uploaded:
    st.stop()

def load_df(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(file, engine="python", sep=None, encoding="latin1", on_bad_lines="skip")
        except Exception:
            file.seek(0); return pd.read_csv(file)
    xls = pd.ExcelFile(file, engine="openpyxl")
    return pd.read_excel(xls, sheet_name=xls.sheet_names[0], engine="openpyxl")

df = load_df(uploaded)

# Enforce required columns only
missing = [c for c in [REQ_TITLE, REQ_ABS] if c not in df.columns]
if missing:
    st.error(
        f"Your file must contain these exact headers:\n• {REQ_TITLE}\n• {REQ_ABS}\n\n"
        f"Missing: {', '.join(missing)}"
    )
    st.stop()

work = df[[REQ_TITLE, REQ_ABS]].copy()
work.rename(columns={REQ_TITLE: "title", REQ_ABS: "abstract"}, inplace=True)

# ========================= Duplicates (simple dropdown) ==========
def normalize_title(s: pd.Series) -> pd.Series:
    return (s.fillna("").astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True))

work["_t_norm"] = normalize_title(work["title"])
dup_counts = work["_t_norm"].value_counts()
dup_titles = dup_counts[dup_counts > 1].index.tolist()

st.subheader("Repeated Projects")
if not dup_titles:
    st.caption("No repeated project titles found (case/whitespace-insensitive).")
else:
    first_display = work.groupby("_t_norm")["title"].first().reindex(dup_titles).tolist()
    choice = st.selectbox("Select a repeated title to view", options=first_display)
    if choice:
        picked_norm = normalize_title(pd.Series([choice])).iloc[0]
        repeated = work[work["_t_norm"] == picked_norm][["title", "abstract"]].reset_index(drop=True)
        st.dataframe(repeated, use_container_width=True)
work.drop(columns=["_t_norm"], inplace=True)

# ========================= Optional text clean (spaCy if present) ============
try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm")
    except Exception:
        _NLP = None
except Exception:
    _NLP = None

def spacy_clean(text: str) -> str:
    if _NLP is not None:
        doc = _NLP(str(text).lower())
        kept = []
        for t in doc:
            if t.is_stop or t.is_punct or t.is_space: continue
            lemma = t.lemma_.strip()
            if re.search(r"[a-z0-9]", lemma): kept.append(lemma)
        return " ".join(kept)
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))

work["clean_text"] = (work["title"].astype(str) + ". " + work["abstract"].astype(str)).apply(spacy_clean)

# ========================= Groq (OpenAI-compatible) ==========================
# Read from Streamlit secrets (preferred) or env vars (fallback)
PROVIDER = st.secrets.get("PROVIDER", os.getenv("PROVIDER", "")).lower()
GROQ_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
GROQ_BASE = st.secrets.get("GROQ_BASE_URL", os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
LLAMA_MODEL = st.secrets.get("LLAMA_MODEL", os.getenv("LLAMA_MODEL", "llama-3.1-8b-instant"))
LLAMA_TEMP = float(st.secrets.get("LLAMA_TEMPERATURE", os.getenv("LLAMA_TEMPERATURE", 0.2)))
LLAMA_MAXTOK = int(st.secrets.get("LLAMA_MAX_TOKENS", os.getenv("LLAMA_MAX_TOKENS", 256)))

def _post_openai_compatible(base_url, api_key, model, system_msg, user_msg, temperature=0.2, max_tokens=256):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def remote_llama_row(title: str, abstract: str) -> dict:
    if PROVIDER != "groq" or not GROQ_KEY:
        raise RuntimeError("Groq not configured. Set PROVIDER='groq' and GROQ_API_KEY in secrets.")
    user_msg = USER_TMPL.format(title=title, abstract=abstract)
    content = _post_openai_compatible(GROQ_BASE, GROQ_KEY, LLAMA_MODEL, SYSTEM_PROMPT, user_msg, LLAMA_TEMP, LLAMA_MAXTOK)
    try:
        m = re.search(r"\{.*\}", content, flags=re.S)
        js = json.loads(m.group(0)) if m else json.loads(content)
        return {
            "originality": float(js.get("originality", 3)),
            "clarity": float(js.get("clarity", 3)),
            "rigor": float(js.get("rigor", 3)),
            "impact": float(js.get("impact", 3)),
            "entrepreneurship": float(js.get("entrepreneurship", 3)),
            "rationale": str(js.get("rationale", "")).strip()[:200],
        }
    except Exception:
        # If the model returns non-JSON, give mid scores for this row
        return {"originality": 3, "clarity": 3, "rigor": 3, "impact": 3, "entrepreneurship": 3, "rationale": ""}

def simple_autoscore(texts: pd.Series) -> pd.Series:
    texts = texts.fillna("").astype(str)
    L = texts.str.len()
    U = texts.apply(lambda s: len(set(w.lower().strip(".,;:!?") for w in s.split() if w)))
    L = (L - L.min()) / (L.max() - L.min() + 1e-9)
    U = (U - U.min()) / (U.max() - U.min() + 1e-9)
    return (0.6 * U + 0.4 * L) * 4 + 1

# ========================= Scoring ===========================================
with st.status("Scoring with **LLaMA (Groq)** …", expanded=False):
    rows = []
    total = len(work)
    prog = st.progress(0.0)
    for i, r in work.iterrows():
        try:
            rows.append(remote_llama_row(r["title"], r["abstract"]))
        except Exception:
            # Final fallback (per-row) is a tiny heuristic—no SBERT (as requested)
            s = simple_autoscore(pd.Series([r["clean_text"]]))[0]
            rows.append({"originality": s, "clarity": s, "rigor": s, "impact": s, "entrepreneurship": s, "rationale": ""})
        prog.progress((i + 1) / total)
    prog.empty()

scored_df = pd.DataFrame(rows)
work = pd.concat([work.reset_index(drop=True), scored_df.reset_index(drop=True)], axis=1)
work["overall"] = work[["originality","clarity","rigor","impact","entrepreneurship"]].mean(axis=1).round(2)

def feedback_for_row(row):
    def msg(score, hi, mid, low, very_low):
        if score >= 5: return hi
        if score >= 4: return mid
        if score >= 3: return "OK, but could go deeper."
        if score >= 2: return low
        return very_low
    parts = []
    parts.append(msg(row["originality"], "Very creative and stands out.", "Good uniqueness; add a bit more novelty.",
                     "Feels common—explain what’s new.", "Not original—clarify the new angle."))
    parts.append(msg(row["clarity"], "Very clear and easy to follow.", "Mostly clear; tighten a few lines.",
                     "Hard to follow—reorganize your points.", "Confusing—use simple, direct sentences."))
    parts.append(msg(row["rigor"], "Method is realistic and well explained.", "Good plan; add a little more detail.",
                     "Plan is vague—explain steps & tools.", "No clear method—outline the process."))
    parts.append(msg(row["impact"], "Strong real-world value.", "Relevant; clarify who benefits.",
                     "Benefit is unclear—explain the value.", "No purpose shown—explain why it matters."))
    parts.append(msg(row["entrepreneurship"], "Shows solid problem-solving.", "Good attempt; add practical details.",
                     "Needs a clearer solution strategy.", "No strategy—explain how you’ll execute."))
    return " ".join(parts)

work["feedback"] = [feedback_for_row(r) for _, r in work.iterrows()]

# ========================= Legends & Results ================================
with st.expander("Scoring Legends (1–5)", expanded=True):
    st.write(
        "- **1** = Very weak  •  **2** = Weak  •  **3** = Adequate  •  **4** = Strong  •  **5** = Excellent\n"
        "- **Originality** — " + RUBRIC["originality"] + "\n"
        "- **Clarity** — " + RUBRIC["clarity"] + "\n"
        "- **Rigor** — " + RUBRIC["rigor"] + "\n"
        "- **Impact** — " + RUBRIC["impact"] + "\n"
        "- **Entrepreneurship** — " + RUBRIC["entrepreneurship"]
    )

st.subheader("Top Ranks")
TOP_K = 10
top = (work[["title","abstract","originality","clarity","rigor","impact","entrepreneurship","overall","rationale","feedback"]]
       .sort_values("overall", ascending=False)
       .reset_index(drop=True))

st.markdown(f"**Top-{TOP_K} recommended** <span class='badge'>sorted by overall</span>", unsafe_allow_html=True)
st.dataframe(top.head(TOP_K)[["title","overall","originality","clarity","rigor","impact","entrepreneurship"]], use_container_width=True)

if len(top) > 0:
    titles_top10 = top.head(TOP_K)["title"].tolist()
    choice_top = st.selectbox("View scores for a Top-10 project", options=titles_top10)
    if choice_top:
        sel = top[top["title"] == choice_top].iloc[0]
        st.write(
            f"**Overall:** {sel['overall']}  |  "
            f"Originality {sel['originality']:.1f} • Clarity {sel['clarity']:.1f} • "
            f"Rigor {sel['rigor']:.1f} • Impact {sel['impact']:.1f} • "
            f"Entrepreneurship {sel['entrepreneurship']:.1f}"
        )
        if sel.get("rationale", ""):
            st.caption(f"Rationale: {sel['rationale']}")

st.subheader("All Results")
st.dataframe(
    top[["title","abstract","originality","clarity","rigor","impact","entrepreneurship","overall","feedback"]],
    use_container_width=True
)

# ========================= Right-side downloads ==============================
c_spacer, c_dl = st.columns([0.60, 0.40])
with c_dl:
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "⬇️ Download Top-10 (CSV)",
            data=top.head(TOP_K)[["title","abstract","originality","clarity","rigor","impact","entrepreneurship","overall","feedback"]]
                 .to_csv(index=False, encoding="utf-8"),
            file_name="rise_top10.csv",
            mime="text/csv",
        )
    with col_b:
        st.download_button(
            "⬇️ Download Feedback for ALL (CSV)",
            data=top[["title","abstract","originality","clarity","rigor","impact","entrepreneurship","overall","feedback"]]
                 .to_csv(index=False, encoding="utf-8"),
            file_name="rise_feedback_all.csv",
            mime="text/csv",
        )
