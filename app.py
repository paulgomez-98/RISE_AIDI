import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RISE Scoring & Feedback", page_icon="✅", layout="wide")
st.title("RISE Scoring & Feedback")

st.write("Upload a CSV or Excel (.xlsx). The app will auto-detect Title/Abstract columns, "
         "remove duplicates, score, show Top-K, and generate simple feedback.")

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

def simple_autoscore(df: pd.DataFrame, text_column: str) -> pd.Series:
    texts = df[text_column].fillna("").astype(str)
    lengths = texts.str.len()
    uniq = texts.apply(lambda s: len(set(w.lower().strip(".,;:!?") for w in s.split() if w)))
    L = (lengths - lengths.min()) / (lengths.max() - lengths.min() + 1e-9)
    U = (uniq - uniq.min()) / (uniq.max() - uniq.min() + 1e-9)
    return 0.6 * U + 0.4 * L

def load_dataframe(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    # Excel with sheet picker
    xls = pd.ExcelFile(file, engine="openpyxl")
    sheet = st.selectbox("Choose Excel sheet", xls.sheet_names, index=0)
    return pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")

def normalize_columns(df: pd.DataFrame):
    cols = list(df.columns)
    low = [str(c).strip().lower() for c in cols]
    title_idx, abstract_idx = None, None
    for i, c in enumerate(low):
        if title_idx is None and ("title" in c and ("research" in c or "capstone" in c)):
            title_idx = i
        if abstract_idx is None and ("description" in c or "abstract" in c):
            abstract_idx = i
    rename_map = {}
    if title_idx is not None:   rename_map[cols[title_idx]] = "title"
    if abstract_idx is not None: rename_map[cols[abstract_idx]] = "abstract"
    if rename_map: df = df.rename(columns=rename_map)
    return df, rename_map

if uploaded:
    df = load_dataframe(uploaded)
    df, rename_map = normalize_columns(df)
    if rename_map:
        st.info(f"Normalized columns: {rename_map}")
    else:
        st.warning("Couldn't auto-detect 'title'/'abstract'. Pick columns below.")

    st.subheader("1) Original Data")
    st.write(df.shape)
    st.dataframe(df.head(20))

    st.markdown("---")
    st.subheader("Select columns to use")
    all_cols = list(df.columns)
    default_title = "title" if "title" in df.columns else all_cols[0]
    default_feedback = "abstract" if "abstract" in df.columns else default_title

    c1, c2, c3 = st.columns(3)
    with c1:
        text_col = st.selectbox("Text column for duplicate check", all_cols,
                                index=all_cols.index(default_title))
    with c2:
        score_col = st.selectbox("Score column (optional; blank = auto-score)",
                                 [""] + all_cols, index=0)
    with c3:
        top_k = st.number_input("How many top applications?", 10, 100, 10, step=5)

    # 1) Exact duplicate rows
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after_exact = len(df)

    # 2) Near-duplicate by chosen text column
    text_norm = df[text_col].fillna("").astype(str).str.strip().str.casefold()
    df = df[~text_norm.duplicated(keep="first")].reset_index(drop=True)
    after_text = len(df)

    st.success(
        f"Removed {before - after_exact} exact dup rows and "
        f"{after_exact - after_text} text dups based on '{text_col}'. Remaining: {after_text}"
    )

    # 3) Score
    if score_col and score_col in df.columns:
        st.info(f"Using existing score column: '{score_col}'")
        scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
    else:
        st.warning("No score column supplied; using a simple auto-score (demo only).")
        scores = simple_autoscore(df, text_col)
        df["score"] = scores
        score_col = "score"

    # 4) Top-K
    top = df.sort_values(score_col, ascending=False).head(top_k).reset_index(drop=True)
    st.subheader("2) Top Selections")
    st.dataframe(top)

    # 5) Feedback
    feedback_text_col = default_feedback if default_feedback in df.columns else text_col

    def gen_feedback(txt: str, sc: float) -> str:
        if not isinstance(txt, str) or not txt.strip():
            return "No submission text provided. Please add more details about scope and impact."
        parts = []
        parts.append("Strength: clear theme and direction." if len(txt) > 60 else "Strength: concise and to the point.")
        parts.append("Consider adding measurable outcomes (e.g., % improvement, timeline, user impact).")
        parts.append("Clarify methods, data sources, and expected deliverables.")
        return " ".join(parts) + f" [demo-score={sc:.2f}]"

    st.subheader("3) Feedback")
    fb = top[[feedback_text_col, score_col]].copy()
    fb.rename(columns={feedback_text_col: "text"}, inplace=True)
    fb["feedback"] = [gen_feedback(t, s) for t, s in zip(fb["text"], fb[score_col])]
    st.dataframe(fb[["text", score_col, "feedback"]])

    # 6) Downloads
    st.download_button("⬇️ Download Top-K CSV", top.to_csv(index=False).encode("utf-8"), "top_k.csv")
    st.download_button("⬇️ Download Feedback CSV", fb.to_csv(index=False).encode("utf-8"), "feedback.csv")

    st.caption("Demo scaffold. Replace auto-score/feedback with your real logic as needed.")
else:
    st.info("Upload a CSV or Excel file to begin. The app will auto-detect Title/Abstract columns if present.")
