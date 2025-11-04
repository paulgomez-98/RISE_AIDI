import io, re, textwrap, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RISE Scoring & Feedback", page_icon="🎯", layout="wide")
st.title("RISE Scoring & Feedback")

st.write("Upload CSV or Excel (.xlsx). We’ll auto-detect Title/Abstract, remove duplicates, score, "
         "show **Top-10**, and generate **feedback for ALL applicants** as a CSV.")

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

sheet_name = None
if uploaded and uploaded.name.lower().endswith(".xlsx"):
    try:
        xls = pd.ExcelFile(uploaded, engine="openpyxl")
        sheet_name = st.selectbox("Choose Excel sheet", xls.sheet_names, index=0)
    except Exception as e:
        st.error(f"Could not read Excel: {e}")
        st.stop()

# ---------- helpers ----------
@st.cache_data(show_spinner=False)
def load_df(file, sheet):
    if file.name.lower().endswith(".csv"):
        try:
            return pd.read_csv(file, engine="python", sep=None, encoding="latin1", on_bad_lines="skip")
        except Exception:
            file.seek(0)
            return pd.read_csv(file)
    xls = pd.ExcelFile(file, engine="openpyxl")
    return pd.read_excel(xls, sheet_name=sheet or xls.sheet_names[0], engine="openpyxl")

def normalize_series(s: pd.Series) -> pd.Series:
    return (s.fillna("").astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True))

def guess_col(df: pd.DataFrame, needles):
    needles = [n.lower() for n in needles]
    for c in df.columns:
        lc = str(c).lower()
        if any(n in lc for n in needles):
            return c
    return df.columns[0]

# rubric & anchors (your text)
RUBRIC = {
    "originality":      "Does the project introduce unique ideas, creative approaches, or innovative concepts? Is the project fresh and distinct from common solutions?",
    "clarity":          "Is the abstract well-organized and easy to follow? Are objectives, methods, and rationale clearly explained? Does the writing flow logically?",
    "rigor":            "Does the abstract demonstrate a well-developed and appropriate research approach? Are methods feasible and aligned with the project’s goals?",
    "impact":           "Does the project address a meaningful issue or opportunity? Could it result in measurable benefits to industry, community, or society?",
    "entrepreneurship": "Does the project demonstrate creative approaches to addressing challenges? Are critical thinking and entrepreneurial skills applied effectively?",
}

# try to import heavy libs; fall back if not available
def try_get_models():
    try:
        from sentence_transformers import SentenceTransformer, util as sbert_util
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu"), sbert_util
    except Exception:
        return None, None

def spacy_cleaner():
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        def _clean(text: str) -> str:
            doc = nlp(str(text).lower())
            kept = []
            for t in doc:
                if t.is_stop or t.is_punct or t.is_space:
                    continue
                lemma = t.lemma_.strip()
                if re.search(r"[a-z0-9]", lemma):
                    kept.append(lemma)
            return " ".join(kept)
        return _clean
    except Exception:
        # safe fallback
        return lambda t: " ".join(re.findall(r"[a-z0-9]+", str(t).lower()))

def scale_to_1_5(col):
    q1, q99 = np.percentile(col, [1, 99])
    col = np.clip(col, q1, q99)
    return 1 + 4 * (col - col.min()) / (col.max() - col.min() + 1e-9)

def simple_autoscore(texts: pd.Series) -> pd.Series:
    texts = texts.fillna("").astype(str)
    L = texts.str.len()
    U = texts.apply(lambda s: len(set(w.lower().strip(".,;:!?") for w in s.split() if w)))
    L = (L - L.min()) / (L.max() - L.min() + 1e-9)
    U = (U - U.min()) / (U.max() - U.min() + 1e-9)
    return (0.6 * U + 0.4 * L) * 4 + 1  # map to 1..5 so tables look consistent

def build_feedback_row(row):
    def msg(score, hi, mid, low, very_low):
        if score >= 5: return hi
        if score >= 4: return mid
        if score >= 3: return "The idea is okay, but needs more depth."
        if score >= 2: return low
        return very_low
    parts = []
    parts.append(msg(row["originality"],
        "The idea is genuinely creative and stands out.",
        "The idea is good and has some uniqueness.",
        "The idea feels common. Show what makes it different.",
        "The idea is not original. You need a clearer new angle.",
    ))
    parts.append(msg(row["clarity"],
        "The abstract is clear, well-written, and easy to understand.",
        "Mostly clear; a few sentences could be smoother.",
        "It’s difficult to follow. Reorganize your explanation.",
        "The abstract is confusing. Rewrite with simple, direct sentences.",
    ))
    parts.append(msg(row["rigor"],
        "The methodology is well explained and realistic.",
        "Clear but could use a bit more detail.",
        "The plan is vague. Explain steps and tools you will use.",
        "No clear method. Describe exactly how the work will be done.",
    ))
    parts.append(msg(row["impact"],
        "The project has strong real-world value.",
        "Relevant; clarify how it helps others.",
        "The benefit is unclear. Explain who gains from this.",
        "No real-world purpose is shown. Explain why this project matters.",
    ))
    parts.append(msg(row["entrepreneurship"],
        "Shows strong problem-solving and practical thinking.",
        "Good attempt at problem-solving; add more detail.",
        "Needs a clearer plan for solving challenges.",
        "No strategy is shown for solving problems. Explain your approach.",
    ))
    return textwrap.fill(" ".join(parts), 120)

if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

# ---------- load & normalize ----------
df = load_df(uploaded, sheet_name)

# auto-detect your long headers
title_col    = guess_col(df, ["title of your research", "capstone", "title"])
abstract_col = guess_col(df, ["description", "abstract"])

st.info(f"Normalized columns: {{'{title_col}': 'title', '{abstract_col}': 'abstract'}}")

st.subheader("1) Original Data")
st.write(f"({df.shape[0]}, {df.shape[1]})")
st.dataframe(df.head(20), use_container_width=True)

# de-dup exactly like your notebook: title+abstract norm combo
work = df.copy()
work["_t_norm"] = normalize_series(work[title_col])
work["_a_norm"] = normalize_series(work[abstract_col])
before = len(work)
work = work.drop_duplicates(subset=["_t_norm", "_a_norm"], keep="first").reset_index(drop=True)
work = work.drop(columns=["_t_norm", "_a_norm"])
st.success(f"Removed {before - len(work)} duplicate copies. Remaining: {len(work)}")

# ---------- scoring ----------
st.subheader("2) Scoring")
cleaner = spacy_cleaner()
work["clean_text"] = (work[title_col].astype(str) + ". " + work[abstract_col].astype(str)).apply(cleaner)

model, sbert_util = try_get_models()
if model is not None:
    st.write("Using **SBERT semantic rubric** (1–5).")
    abs_emb = model.encode(work["clean_text"].tolist(), convert_to_tensor=True, normalize_embeddings=True)
    prompt_emb = {k: model.encode(v, convert_to_tensor=True, normalize_embeddings=True) for k, v in RUBRIC.items()}
    raw = {}
    for crit in RUBRIC.keys():
        sim = sbert_util.cos_sim(abs_emb, prompt_emb[crit]).cpu().numpy().ravel()
        raw[crit] = sim
    raw_df = pd.DataFrame(raw)
    scaled = raw_df.apply(scale_to_1_5).clip(1, 5)
    for c in ["originality","clarity","rigor","impact","entrepreneurship"]:
        work[c] = scaled[c]
else:
    st.warning("SBERT not available on this environment; falling back to simple auto-score (scaled to 1–5).")
    s = simple_autoscore(work["clean_text"])
    for c in ["originality","clarity","rigor","impact","entrepreneurship"]:
        work[c] = s

work["overall"] = work[["originality","clarity","rigor","impact","entrepreneurship"]].mean(axis=1).round(2)

# ---------- outputs you asked for ----------
st.subheader("3) Top-10 with scores")
top10 = (work
         [[title_col, "originality","clarity","rigor","impact","entrepreneurship","overall"]]
         .sort_values("overall", ascending=False)
         .head(10)
         .reset_index(drop=True))
st.dataframe(top10, use_container_width=True)

# feedback for ALL rows
st.subheader("4) Feedback for ALL applicants (download)")
all_feedback = work.copy()
all_feedback["feedback"] = [build_feedback_row(r) for _, r in all_feedback.iterrows()]

# tidy columns
display_cols = [title_col, "originality","clarity","rigor","impact","entrepreneurship","overall","feedback"]
top10_out = (all_feedback[display_cols]
             .sort_values("overall", ascending=False)
             .head(10)
             .reset_index(drop=True)
             .rename(columns={title_col: "title"}))

all_out = (all_feedback[display_cols]
           .rename(columns={title_col: "title"})
           .reset_index(drop=True))

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Download Top-10 (CSV)",
        data=top10_out.to_csv(index=False, encoding="utf-8"),
        file_name="rise_top10_recommended.csv",
        mime="text/csv",
    )
with c2:
    st.download_button(
        "⬇️ Download Feedback for ALL (CSV)",
        data=all_out.to_csv(index=False, encoding="utf-8"),
        file_name="rise_feedback_all_applicants.csv",
        mime="text/csv",
    )

st.caption("Tip: If SBERT is missing on Streamlit Cloud, add `sentence-transformers`, `torch`, and `spacy` to requirements.txt.")
