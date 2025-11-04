import io
import time
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RISE Scoring & Feedback", page_icon="✅", layout="wide")
st.title("RISE Scoring & Feedback (Demo)")

st.write(
    "Upload a **CSV** or **Excel (.xlsx)** of applications. We'll clean duplicates, compute a basic score "
    "(or use your existing 'score' column), sort top-10, and generate simple feedback text."
)

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# Options
colA, colB, colC = st.columns(3)
with colA:
    text_col = st.text_input("Text column for duplicate check", value="title")
with colB:
    score_col = st.text_input("Score column (leave blank to auto-score)", value="")
with colC:
    top_k = st.number_input("How many top applications?", 10, 100, 10, step=5)

def simple_autoscore(df: pd.DataFrame, text_column: str) -> pd.Series:
    """
    Very naive heuristic score when a 'score' column is not provided.
    Replace with your real model later.
    """
    texts = df[text_column].fillna("").astype(str)
    lengths = texts.str.len()
    uniq = texts.apply(lambda s: len(set(w.lower().strip(".,;:!?") for w in s.split() if w)))
    # Normalise 0..1 (safe for constant vectors)
    L = (lengths - lengths.min()) / (lengths.max() - lengths.min() + 1e-9)
    U = (uniq - uniq.min()) / (uniq.max() - uniq.min() + 1e-9)
    return 0.6 * U + 0.4 * L

def load_dataframe(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith(".xlsx"):
        # If multi-sheet Excel, let user pick
        xls = pd.ExcelFile(file, engine="openpyxl")
        sheet = st.selectbox("Choose Excel sheet", xls.sheet_names, index=0)
        return pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
    else:
        st.error("Unsupported file type. Please upload .csv or .xlsx")
        st.stop()

if uploaded:
    df = load_dataframe(uploaded)

    st.subheader("1) Original Data")
    st.write(df.shape)
    st.dataframe(df.head(20))

    # 1) Clean duplicates (exact row duplicates)
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    after_exact = df.shape[0]

    # 2) Drop near-duplicate texts within the selected column
    if text_col not in df.columns:
        st.error(f"Column '{text_col}' not found. Available: {list(df.columns)}")
        st.stop()

    text_norm = df[text_col].fillna("").astype(str).str.strip().str.casefold()
    dedupe_mask = ~text_norm.duplicated(keep="first")
    df = df[dedupe_mask].reset_index(drop=True)
    after_text = df.shape[0]

    st.success(
        f"Removed {before - after_exact} exact duplicate rows and "
        f"{after_exact - after_text} text duplicates based on '{text_col}'. "
        f"Remaining: {after_text}"
    )

    # 3) Score
    if score_col and score_col in df.columns:
        st.info(f"Using existing score column: '{score_col}'")
        scores = df[score_col].astype(float)
    else:
        st.warning("No score column supplied; using a simple auto-score (demo only).")
        scores = simple_autoscore(df, text_col)
        df["score"] = scores
        score_col = "score"

    # 4) Sort and select top-k
    top = df.sort_values(score_col, ascending=False).head(top_k).reset_index(drop=True)

    st.subheader("2) Top Selections")
    st.dataframe(top)

    # 5) Feedback generation (simple template; replace with your real logic)
    def gen_feedback(txt: str, sc: float) -> str:
        if not isinstance(txt, str) or not txt.strip():
            return "No submission text provided. Please add more details about your project scope and impact."
        parts = []
        parts.append("Strength: clear theme and direction." if len(txt) > 60 else "Strength: concise and to the point.")
        parts.append("Consider adding measurable outcomes (e.g., % improvement, timeline, user impact).")
        parts.append("Clarify methods, data sources, and expected deliverables.")
        return " ".join(parts) + f" [demo-score={sc:.2f}]"

    st.subheader("3) Feedback")
    fb = top[[text_col, score_col]].copy()
    fb["feedback"] = [gen_feedback(t, s) for t, s in zip(fb[text_col], fb[score_col])]
    st.dataframe(fb[[text_col, score_col, "feedback"]])

    # 6) Downloads
    st.download_button("⬇️ Download Top-K as CSV", data=top.to_csv(index=False).encode("utf-8"), file_name="top_k.csv")
    st.download_button("⬇️ Download Feedback as CSV", data=fb.to_csv(index=False).encode("utf-8"), file_name="feedback.csv")

    st.caption("This is a demo scaffold. Replace the auto-score and feedback template with your actual model/logic.")
else:
    st.info("Upload a CSV or Excel file to begin.")
