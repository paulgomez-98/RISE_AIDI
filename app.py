# app.py — RISE Smart Scoring (matches your notebook outputs)

import io, re, textwrap, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

# ML / NLP
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Optional visual quick checks
import altair as alt

# ===== Streamlit page =====
st.set_page_config(page_title="RISE Smart Scoring", page_icon="🎯", layout="wide")
st.title("RISE — Smart Scoring & Feedback")

st.caption(
    "Upload CSV or Excel (.xlsx) → clean duplicates → SBERT rubric scoring (1–5) → "
    "train explainable TF-IDF + LinearSVR baseline → per-row feedback → Top-10 + CSV exports."
)

uploaded = st.file_uploader("Upload dataset (CSV or XLSX)", type=["csv", "xlsx"])
sheet_name = None
if uploaded and uploaded.name.lower().endswith(".xlsx"):
    try:
        xls = pd.ExcelFile(uploaded, engine="openpyxl")
        sheet_name = st.selectbox("Choose Excel sheet", xls.sheet_names, index=0)
    except Exception as e:
        st.error(f"Could not open Excel: {e}")
        st.stop()

# -------- Helpers --------
@st.cache_data(show_spinner=False)
def load_df(_uploaded, _sheet):
    name = _uploaded.name.lower()
    if name.endswith(".csv"):
        # be lenient about encodings
        try:
            return pd.read_csv(_uploaded, engine="python", sep=None, encoding="latin1", on_bad_lines="skip")
        except Exception:
            _uploaded.seek(0)
            return pd.read_csv(_uploaded)
    # Excel
    xls = pd.ExcelFile(_uploaded, engine="openpyxl")
    return pd.read_excel(xls, sheet_name=_sheet or xls.sheet_names[0], engine="openpyxl")

def normalize_for_compare(s: pd.Series) -> pd.Series:
    return (
        s.fillna("").astype(str)
         .str.strip().str.lower()
         .str.replace(r"\s+", " ", regex=True)
    )

# Your rubric & anchors (from screenshots)
RUBRIC = {
    "originality":      "Does the project introduce unique ideas, creative approaches, or innovative concepts? Is the project fresh and distinct from common solutions?",
    "clarity":          "Is the abstract well-organized and easy to follow? Are objectives, methods, and rationale clearly explained? Does the writing flow logically?",
    "rigor":            "Does the abstract demonstrate a well-developed and appropriate research approach? Are methods feasible and aligned with the project’s goals?",
    "impact":           "Does the project address a meaningful issue or opportunity? Could it result in measurable benefits to industry, community, or society?",
    "entrepreneurship": "Does the project demonstrate creative approaches to addressing challenges? Are critical thinking and entrepreneurial skills applied effectively?",
}

ANCHORS = {
    "originality":      "highly original, inventive, novel concept, new methods, breakthrough innovation",
    "clarity":          "exceptionally clear, concise, well structured, organized, easy to follow",
    "rigor":            "robust methodology, detailed methods, rigorous, feasible, strong alignment to project objectives",
    "impact":           "significant real-world outcomes, high value, societal benefit, measurable industry impact",
    "entrepreneurship": "highly innovative and entrepreneurial approach; outstanding application of creative solutions",
}

# Load + column mapping
if not uploaded:
    st.info("Upload your dataset to begin.")
    st.stop()

df = load_df(uploaded, sheet_name)
st.subheader("1) Original Data")
st.write(f"**{df.shape[0]} rows × {df.shape[1]} cols**")
st.dataframe(df.head(20), use_container_width=True)

# Column mapper (your exact long headers)
all_cols = list(df.columns)
def _guess(col_substring):
    for c in all_cols:
        if col_substring in c.lower():
            return c
    return None

default_title = _guess("title of your research") or _guess("title")
default_abs   = _guess("description") or _guess("abstract")

c1, c2 = st.columns(2)
with c1:
    TITLE_COL = st.selectbox("Project Title column", options=all_cols,
                             index=all_cols.index(default_title) if default_title in all_cols else 0)
with c2:
    ABSTRACT_COL = st.selectbox("Abstract / Description column", options=all_cols,
                                index=all_cols.index(default_abs) if default_abs in all_cols else 0)

# --- Dedupe: keep one unique project by normalized title+abstract (matches your cell) ---
st.subheader("2) Clean duplicates (keep one per unique project)")
work = df.copy()
work["_t_norm"] = normalize_for_compare(work[TITLE_COL])
work["_a_norm"] = normalize_for_compare(work[ABSTRACT_COL])

before = len(work)
work = work.drop_duplicates(subset=["_t_norm", "_a_norm"], keep="first").reset_index(drop=True)
after = len(work)
removed = before - after

work = work.drop(columns=["_t_norm", "_a_norm"])
st.success(f"Unique project count: **{after}**  (removed **{removed}** duplicate copies)")
st.dataframe(work[[TITLE_COL, ABSTRACT_COL]].head(10), use_container_width=True)

# ---- spaCy preprocessing (tokenize + lemmatize) ----
@st.cache_resource(show_spinner=True)
def get_spacy():
    # try to load; if not available, fallback to a tiny regex tokenizer
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        return nlp
    except Exception:
        return None

nlp = get_spacy()

def spacy_clean(text: str) -> str:
    if nlp is None:
        # safe fallback: lowercase, keep alphanumerics
        toks = re.findall(r"[a-z0-9]+", str(text).lower())
        return " ".join(toks)
    doc = nlp(str(text).lower())
    kept = []
    for t in doc:
        if t.is_stop or t.is_punct or t.is_space:
            continue
        lemma = t.lemma_.strip()
        if re.search(r"[a-z0-9]", lemma):
            kept.append(lemma)
    return " ".join(kept)

st.subheader("3) Build clean text (title + abstract → lemmas)")
work["clean_text"] = (work[TITLE_COL].astype(str) + ". " + work[ABSTRACT_COL].astype(str)).apply(spacy_clean)
st.dataframe(work[["clean_text"]].head(3), use_container_width=True)

# ---- SBERT semantic scoring mapped to 1–5 ----
@st.cache_resource(show_spinner=True)
def get_sbert():
    from sentence_transformers import SentenceTransformer, util  # noqa
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return model

from sentence_transformers import util as sbert_util  # only util is needed here
model = get_sbert()

st.subheader("4) SBERT rubric scoring → scale to 1–5")

# embeddings
abs_emb = model.encode(work["clean_text"].tolist(), convert_to_tensor=True, normalize_embeddings=True)
prompt_emb = {k: model.encode(v, convert_to_tensor=True, normalize_embeddings=True) for k, v in RUBRIC.items()}

# cosine similarities per criterion
raw_scores = {}
for crit in RUBRIC.keys():
    sim = sbert_util.cos_sim(abs_emb, prompt_emb[crit]).cpu().numpy().ravel()
    raw_scores[crit] = sim

raw_df = pd.DataFrame(raw_scores)

# scale each criterion to 1..5 using robust percentiles (matches your idea)
def scale_to_1_5(col):
    q1, q99 = np.percentile(col, [1, 99])
    col = np.clip(col, q1, q99)  # trim tails
    return 1 + 4 * (col - col.min()) / (col.max() - col.min() + 1e-9)

scaled_df = raw_df.apply(scale_to_1_5).clip(lower=1, upper=5)
work[["originality","clarity","rigor","impact","entrepreneurship"]] = scaled_df[["originality","clarity","rigor","impact","entrepreneurship"]]
work["overall"] = scaled_df.mean(axis=1)

st.dataframe(work[[TITLE_COL, "originality","clarity","rigor","impact","entrepreneurship","overall"]].head(10), use_container_width=True)

# ---- Simple explainable baseline: TF-IDF + LinearSVR using pseudo-labels ----
st.subheader("5) Explainable baseline (TF-IDF → LinearSVR)")

tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=3, sublinear_tf=True, lowercase=True)
X = tfidf.fit_transform(work["clean_text"])

idx = np.arange(X.shape[0])
X_train, X_test, idx_train, idx_test = train_test_split(X, idx, test_size=0.15, random_state=42)

metrics_rows = []
for crit in ["originality","clarity","rigor","impact","entrepreneurship"]:
    y = work[crit].values
    y_train, y_test = y[idx_train], y[idx_test]
    reg = LinearSVR(C=1.0, epsilon=0.0, loss="squared_epsilon_insensitive", max_iter=5000, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = np.clip(reg.predict(X_test), 1.0, 5.0)
    metrics_rows.append([crit, mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)])
metrics_df = pd.DataFrame(metrics_rows, columns=["criterion","MAE","R2"])
st.dataframe(metrics_df, use_container_width=True)

# ---- Feedback per row (exactly your conditional style) ----
def feedback_for_row(row):
    s = row[["originality","clarity","rigor","impact","entrepreneurship","overall"]]
    comments = []
    # ORIGINIALITY
    if s["originality"] >= 5: comments.append("The idea is genuinely creative and stands out.")
    elif s["originality"] >= 4: comments.append("The idea is good and has some uniqueness.")
    elif s["originality"] >= 3: comments.append("The idea is okay, but not very distinct.")
    elif s["originality"] >= 2: comments.append("The idea feels common. Show what makes it different.")
    else: comments.append("The idea is not original. You need a clearer new angle.")

    # CLARITY
    if s["clarity"] >= 5: comments.append("The abstract is clear, well-written, and easy to understand.")
    elif s["clarity"] >= 4: comments.append("Mostly clear; a few sentences could be smoother.")
    elif s["clarity"] >= 3: comments.append("Some parts are understandable, but the flow needs work.")
    elif s["clarity"] >= 2: comments.append("It’s difficult to follow. Reorganize your explanation.")
    else: comments.append("The abstract is confusing. Rewrite with simple, direct sentences.")

    # RIGOR
    if s["rigor"] >= 5: comments.append("The methodology is well explained and realistic.")
    elif s["rigor"] >= 4: comments.append("Clear but could use a bit more detail.")
    elif s["rigor"] >= 3: comments.append("The method is mentioned but not clearly explained.")
    elif s["rigor"] >= 2: comments.append("The plan is vague. Explain steps and tools you will use.")
    else: comments.append("No clear method. Describe exactly how the work will be done.")

    # IMPACT
    if s["impact"] >= 5: comments.append("The project has strong real-world value.")
    elif s["impact"] >= 4: comments.append("Relevant; clarify how it helps others.")
    elif s["impact"] >= 3: comments.append("Possible impact but not well explained.")
    elif s["impact"] >= 2: comments.append("The benefit is unclear. Explain who gains from this.")
    else: comments.append("No real-world purpose is shown. Explain why this project matters.")

    # ENTREPRENEURSHIP
    if s["entrepreneurship"] >= 5: comments.append("Shows strong problem-solving and practical thinking.")
    elif s["entrepreneurship"] >= 4: comments.append("Good attempt at problem-solving; add more detail.")
    elif s["entrepreneurship"] >= 3: comments.append("A strategy is shown, but the plan is basic.")
    elif s["entrepreneurship"] >= 2: comments.append("Needs a clearer plan for solving challenges.")
    else: comments.append("No strategy is shown for solving problems. Explain your approach.")

    return textwrap.fill(" ".join(comments), 120)

# Build feedback frame ordered by overall
feedback_df = work.copy()
feedback_df = feedback_df.assign(
    feedback = [feedback_for_row(r) for _, r in feedback_df.iterrows()]
).sort_values("overall", ascending=False).reset_index(drop=True)

# ---- Top 10 table like your output ----
st.subheader("6) Top 10 Recommended Applicants (by overall)")
top10 = feedback_df[[TITLE_COL,"originality","clarity","rigor","impact","entrepreneurship","overall","feedback"]].head(10).copy()
st.dataframe(top10, use_container_width=True)

# ---- Downloads identical to your names ----
TOP10_PATH = "rise_top10_recommended.csv"
ALL_PATH  = "rise_feedback_all_applicants.csv"

c1, c2 = st.columns(2)
with c1:
    st.download_button("⬇️ Download Top-10 CSV", data=top10.to_csv(index=False, encoding="utf-8"), file_name=TOP10_PATH, mime="text/csv")
with c2:
    # full dataset feedback (ensure required cols exist)
    required_cols = [TITLE_COL, "originality","clarity","rigor","impact","entrepreneurship","overall","feedback"]
    safe = feedback_df[required_cols].rename(columns={TITLE_COL:"title"})
    st.download_button("⬇️ Download Feedback for ALL (CSV)", data=safe.to_csv(index=False, encoding="utf-8"),
                       file_name=ALL_PATH, mime="text/csv")

# ---- Optional quick EDA lengths (like your histograms) ----
with st.expander("Optional: quick EDA (length histograms)"):
    lens = pd.DataFrame({
        "title_len":   work[TITLE_COL].astype(str).str.len(),
        "abs_len":     work[ABSTRACT_COL].astype(str).str.len(),
    })
    chart1 = alt.Chart(lens).mark_bar().encode(alt.X("title_len:Q", bin=alt.Bin(maxbins=40)), y="count()").properties(height=160)
    chart2 = alt.Chart(lens).mark_bar().encode(alt.X("abs_len:Q",   bin=alt.Bin(maxbins=40)), y="count()").properties(height=160)
    st.altair_chart(chart1, use_container_width=True)
    st.altair_chart(chart2, use_container_width=True)
