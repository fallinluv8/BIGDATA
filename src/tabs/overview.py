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
    n_queries = df_queries["query_id"].nunique() if "query_id" in df_queries.columns else len(df_queries)
    n_docs_ocr = df_ocr["doc_id"].nunique() if "doc_id" in df_ocr.columns else len(df_ocr)
    n_results = len(df_results)

    # OCR quality quick scan
    if "text_ocr" in df_ocr.columns:
        ocr_texts = df_ocr["text_ocr"].fillna("").astype(str)
        n_empty = (ocr_texts.str.strip() == "").sum()
        n_error = ocr_texts.apply(is_ocr_error).sum()
    else:
        n_empty, n_error = 0, 0

    # metrics overview
    avg_p10 = None
    if "precision_at_10" in df_metrics.columns and len(df_metrics) > 0:
        avg_p10 = pd.to_numeric(df_metrics["precision_at_10"], errors="coerce").dropna().mean()

    # ===== KPI cards =====
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Queries", f"{n_queries:,}")
    c2.metric("Docs (OCR available)", f"{n_docs_ocr:,}")
    c3.metric("Retrieval rows", f"{n_results:,}")
    c4.metric("Avg Precision@10", f"{avg_p10:.4f}" if avg_p10 is not None else "N/A")

    st.markdown("---")

    # ===== Charts =====
    left, right = st.columns([1, 1])

    with left:
        st.subheader(" Precision@10 distribution")
        if "precision_at_10" not in df_metrics.columns or len(df_metrics) == 0:
            st.info("Không có dữ liệu metrics hoặc thiếu cột precision_at_10.")
        else:
            vals = pd.to_numeric(df_metrics["precision_at_10"], errors="coerce").dropna()
            fig = plt.figure()
            plt.hist(vals, bins=20)
            plt.xlabel("Precision@10")
            plt.ylabel("Count")
            st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader(" OCR quality quick check")
        total = len(df_ocr)
        if total == 0:
            st.info("web_data_ocr.csv rỗng.")
        else:
            ok_count = total - n_empty - n_error
            fig = plt.figure()
            plt.bar(["OK", "Empty", "Error"], [ok_count, n_empty, n_error])
            plt.ylabel("Rows")
            st.pyplot(fig, clear_figure=True)

            

    st.markdown("---")

    # ===== Tables =====
    st.subheader(" Metrics table (preview)")
    st.dataframe(df_metrics.head(50), use_container_width=True)

   
