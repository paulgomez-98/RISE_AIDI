# app.py — RISE Smart Scoring & Feedback
# Fast, robust, and styled Streamlit app with graceful fallbacks.

import re, textwrap, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

# =============== Optional imports (guarded) ===============
try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
    HAVE_SBERT = True
except Exception:
    HAVE_SBERT = False

try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm")
    except Exception:
        _NLP = None
    HAVE_SPACY = True
except Exception:
    _NLP = None
    HAVE_SPACY = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVR
    from sklearn.metrics import mean_absolute_error, r2_score
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# =============== Page config & styles ===============
st.set_page_config(page_title="RISE Smart Scoring", page_icon="🎯", layout="wide")

CUSTOM_CSS = """
<style>
/* page width & fonts */
.block-container {max-width: 1200px !important;}
h1, h2, h3 {letter-spacing: 0.2px;}
/* dark hero band */
.hero {
  background: linear-gradient(135deg,#0f172a,#111827);
  border-radius: 20px;
  padding: 26px 28px;
  color: #fff;
  margin-bottom: 22px;
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 6px 18px rgba(0,0,0,.25);
}
.hero h1 {margin:0 0 6px 0;font-size: 28px;}
.hero p {opacity: .9; margin:0;}
/* compact table */
thead tr th {white-space: nowrap;}
.dataframe td, .dataframe th {padding: 6px 10px;}
/* subtle cards */
.stAlert {border-radius: 12px;}
/* pill badges */
.badge {
  display:inline-block; padding:4px 8px; border-radius:20px;
  background:#eef2ff; color:#3730a3; font-size:12px; margin-left:8px;
}
.small-note { font-size: 12px; color:#6b7280; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    "<div class='hero'><h1>RISE — Smart Scoring & Feedback</h1>"
    "<p>Upload CSV/XLSX, remove duplicates, score (1–5), view Top-K, and export feedback for all applicants.</p></div>",
    unsafe_allow_html=True,
)

# =============== Sidebar controls ===============
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    top_k = st.slider("Top-K to display", 5, 50, 10, step=5)
    drop_exact_dupes = st.checkbox("Drop exact row duplicates", True)
    drop_text_dupes = st.checkbox("Drop duplicate Titles (normalized)", True)

    st.divider()
    scoring_mode = st.radio(
        "Scoring method",
        options=["Semantic (SBERT)", "Fast auto-score"],
        index=0 if HAVE_SBERT else 1,
        help="SBERT gives better semantic scores; auto-score is a fast fallback.",
    )

    show_baseline = st.checkbox(
        "Show TF-IDF + LinearSVR baseline (sklearn)",
        value=False and HAVE_SKLEARN,
        help="For explainability; skipped if scikit-learn is unavailable."
    )

# =============== Helpers ===============
@st.cache_data(show_spinner=False)
def load_df(file, sheet):
    name = file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(file, engine="python", sep=None, encoding="latin1", on_bad_lines="skip")
        except Exception:
            file.seek(0)
            return pd.read_csv(file)
    xls = pd.ExcelFile(file, engine="openpyxl")
    return pd.read_excel(xls, sheet_name=sheet or xls.sheet_names[0], engine="openpyxl")

def normalize_series(s: pd.Series) -> pd.Series:
    return (s.fillna("").astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True))

def guess(df, needles):
    needles = [n.lower() for n in needles]
    for c in df.columns:
        lc = str(c).lower()
        if any(n in lc for n in needles):
            return c
    return df.columns[0]

def is_yes_no_like(df, col):
    vals = pd.Series(df[col]).astype(str).str.strip().str.lower().unique()[:8]
    return set(vals).issubset({"yes","no","y","n","true","false","nan",""})

def spacy_clean(text: str) -> str:
    if _NLP is not None:
        doc = _NLP(str(text).lower())
        kept = []
        for t in doc:
            if t.is_stop or t.is_punct or t.is_space:
                continue
            lemma = t.lemma_.strip()
            if re.search(r"[a-z0-9]", lemma):
                kept.append(lemma)
        return " ".join(kept)
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))

def scale_to_1_5(x):
    q1, q99 = np.percentile(x, [1, 99])
    x = np.clip(x, q1, q99)
    return 1 + 4 * (x - x.min()) / (x.max() - x.min() + 1e-9)

def simple_autoscore(texts: pd.Series) -> pd.Series:
    texts = texts.fillna("").astype(str)
    L = texts.str.len()
    U = texts.apply(lambda s: len(set(w.lower().strip(".,;:!?") for w in s.split() if w)))
    L = (L - L.min()) / (L.max() - L.min() + 1e-9)
    U = (U - U.min()) / (U.max() - U.min() + 1e-9)
    return (0.6 * U + 0.4 * L) * 4 + 1

def feedback_for_row(row):
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
    return " ".join(parts)

# =============== Main workflow ===============
if not uploaded:
    st.info("Upload a CSV or Excel file to begin.")
    st.stop()

sheet_name = None
if uploaded.name.lower().endswith(".xlsx"):
    try:
        xls = pd.ExcelFile(uploaded, engine="openpyxl")
        sheet_name = st.selectbox("Sheet", xls.sheet_names, index=0)
    except Exception as e:
        st.error(f"Could not open Excel: {e}")
        st.stop()

df = load_df(uploaded, sheet_name)

st.subheader("1) Map columns")
all_cols = list(df.columns)
g_title = guess(df, ["title of your research", "capstone", "title"])
g_abs   = guess(df, ["description", "abstract"])

c1, c2 = st.columns(2)
with c1:
    title_col = st.selectbox("Project Title column", options=all_cols,
                             index=all_cols.index(g_title) if g_title in all_cols else 0)
with c2:
    abstract_col = st.selectbox("Abstract / Description column", options=all_cols,
                                index=all_cols.index(g_abs) if g_abs in all_cols else 0)

warns = []
if is_yes_no_like(df, title_col):
    warns.append(f"‘{title_col}’ looks like a Yes/No field — not a title.")
if df[title_col].astype(str).str.len().median() < 8:
    warns.append(f"‘{title_col}’ values look very short for titles.")
if warns:
    st.warning(" ".join(warns))

st.write("**Title preview:**", df[title_col].astype(str).head(5).tolist())
st.write("**Abstract preview:**", df[abstract_col].astype(str).head(3).tolist())

st.subheader("2) Clean & de-duplicate")
work = df.copy()
if drop_exact_dupes:
    before = len(work)
    work = work.drop_duplicates().reset_index(drop=True)
    st.success(f"Removed {before - len(work)} exact duplicate rows.")

if drop_text_dupes:
    work["_t_norm"] = normalize_series(work[title_col])
    before = len(work)
    work = work.drop_duplicates(subset=["_t_norm"], keep="first").reset_index(drop=True)
    st.success(f"Removed {before - len(work)} duplicate titles.")
    work = work.drop(columns=["_t_norm"])

st.caption(f"Remaining rows: **{len(work)}**")

# Build clean_text
work["clean_text"] = (work[title_col].astype(str) + ". " + work[abstract_col].astype(str)).apply(spacy_clean)

# =============== Scoring ===============
st.subheader("3) Scoring (1–5)")

RUBRIC = {
    "originality":      "Does the project introduce unique ideas, creative approaches, or innovative concepts? Is the project fresh and distinct from common solutions?",
    "clarity":          "Is the abstract well-organized and easy to follow? Are objectives, methods, and rationale clearly explained? Does the writing flow logically?",
    "rigor":            "Does the abstract demonstrate a well-developed and appropriate research approach? Are methods feasible and aligned with the project’s goals?",
    "impact":           "Does the project address a meaningful issue or opportunity? Could it result in measurable benefits to industry, community, or society?",
    "entrepreneurship": "Does the project demonstrate creative approaches to addressing challenges? Are critical thinking and entrepreneurial skills applied effectively?",
}

if scoring_mode == "Semantic (SBERT)" and HAVE_SBERT:
    st.write("Using **SBERT semantic rubric**.")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
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
    if scoring_mode == "Semantic (SBERT)" and not HAVE_SBERT:
        st.warning("`sentence-transformers` not installed; falling back to fast auto-score.")
    s = simple_autoscore(work["clean_text"])
    for c in ["originality","clarity","rigor","impact","entrepreneurship"]:
        work[c] = s

work["overall"] = work[["originality","clarity","rigor","impact","entrepreneurship"]].mean(axis=1).round(2)

# =============== Optional baseline (explainable) ===============
if show_baseline and HAVE_SKLEARN:
    st.subheader("4) Explainable baseline (TF-IDF → LinearSVR)")
    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=3, sublinear_tf=True, lowercase=True)
    X = tfidf.fit_transform(work["clean_text"])
    idx = np.arange(X.shape[0])
    X_train, X_test, idx_train, idx_test = train_test_split(X, idx, test_size=0.15, random_state=42)

    rows = []
    for crit in ["originality","clarity","rigor","impact","entrepreneurship"]:
        y = work[crit].values
        y_train, y_test = y[idx_train], y[idx_test]
        reg = LinearSVR(C=1.0, epsilon=0.0, loss="squared_epsilon_insensitive", max_iter=5000, random_state=42)
        reg.fit(X_train, y_train)
        y_pred = np.clip(reg.predict(X_test), 1.0, 5.0)
        rows.append([crit, round(mean_absolute_error(y_test, y_pred), 3), round(r2_score(y_test, y_pred), 3)])
    metrics_df = pd.DataFrame(rows, columns=["criterion","MAE","R2"])
    st.dataframe(metrics_df, use_container_width=True)
elif show_baseline and not HAVE_SKLEARN:
    st.info("scikit-learn not installed — baseline skipped.")

# =============== Feedback & Downloads ===============
st.subheader("5) Results & Exports")
all_feedback = work.copy()
all_feedback["feedback"] = [feedback_for_row(r) for _, r in all_feedback.iterrows()]

display_cols = [title_col, "originality","clarity","rigor","impact","entrepreneurship","overall","feedback"]
top = (all_feedback[display_cols]
       .sort_values("overall", ascending=False)
       .head(top_k)
       .reset_index(drop=True)
       .rename(columns={title_col: "title"}))

st.markdown(f"**Top-{top_k} recommended** <span class='badge'>sorted by overall</span>", unsafe_allow_html=True)
st.dataframe(top, use_container_width=True)

all_out = (all_feedback[display_cols]
           .rename(columns={title_col: "title"})
           .reset_index(drop=True))

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Download Top-K (CSV)",
        data=top.to_csv(index=False, encoding="utf-8"),
        file_name="rise_topk_recommended.csv",
        mime="text/csv",
    )
with c2:
    st.download_button(
        "⬇️ Download Feedback for ALL (CSV)",
        data=all_out.to_csv(index=False, encoding="utf-8"),
        file_name="rise_feedback_all_applicants.csv",
        mime="text/csv",
    )

st.markdown("<span class='small-note'>Pro tip: keep your dataset headers consistent (e.g., 'title', 'abstract').</span>", unsafe_allow_html=True)
