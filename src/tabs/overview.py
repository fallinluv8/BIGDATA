# src/tabs/overview.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from .utils import load_all, is_ocr_error

def show_page(data_dir: str):
    st.header("1) Tổng quan & Metrics")

    with st.spinner("Đang load dữ liệu CSV..."):
        df_queries, df_results, df_ocr, df_metrics = load_all(data_dir)

    # ===== Basic stats =====
    n_queries = df_queries["query_id"].nunique()
    n_docs_ocr = df_ocr["doc_id"].nunique()
    n_results = len(df_results)

    # OCR quality
    ocr_texts = df_ocr["text_ocr"].fillna("").astype(str)
    n_empty = (ocr_texts.str.strip() == "").sum()
    n_error = ocr_texts.apply(is_ocr_error).sum()

    # Metrics
    avg_p10 = None
    if "precision_at_10" in df_metrics.columns:
        p10 = pd.to_numeric(df_metrics["precision_at_10"], errors="coerce")
        avg_p10 = p10.dropna().mean()

    # ===== KPI =====
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Queries", f"{n_queries:,}")
    c2.metric("Docs (OCR)", f"{n_docs_ocr:,}")
    c3.metric("Retrieval rows", f"{n_results:,}")
    c4.metric("Avg Precision@10", f"{avg_p10:.4f}" if avg_p10 is not None else "N/A")

    st.markdown("---")

    # ===== Charts =====
    left, right = st.columns(2)

    with left:
        st.subheader("Precision@10 distribution")
        if avg_p10 is None:
            st.info("Không có dữ liệu Precision@10.")
        else:
            fig = plt.figure()
            plt.hist(p10.dropna(), bins=20)
            plt.xlabel("Precision@10")
            plt.ylabel("Count")
            st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("OCR quality overview")
        total = len(df_ocr)
        fig = plt.figure()
        plt.bar(["OK", "Empty", "Error"], [total - n_empty - n_error, n_empty, n_error])
        plt.ylabel("Rows")
        st.pyplot(fig, clear_figure=True)

    st.markdown("---")

    st.subheader("Metrics preview")
    st.dataframe(df_metrics.head(50), use_container_width=True)
