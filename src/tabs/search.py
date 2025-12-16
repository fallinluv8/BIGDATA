# src/tabs/search.py
import streamlit as st
import pandas as pd

from .utils import load_all, is_ocr_error, summarize_text

def show_page(data_dir: str):
    st.header("3) Demo Tìm kiếm (Retrieval)")

    with st.spinner("Đang load dữ liệu..."):
        df_queries, df_results, df_ocr, df_metrics = load_all(data_dir)

    required = {"query_id", "doc_id"}
    if not required.issubset(set(df_results.columns)):
        st.error("web_data_results.csv phải có cột query_id và doc_id")
        st.stop()

    if not {"query_id", "query_text"}.issubset(set(df_queries.columns)):
        st.error("web_data_queries.csv phải có cột query_id và query_text")
        st.stop()

    # ===== Controls =====
    st.subheader(" Chọn query để tìm kiếm")
    df_queries = df_queries.copy()
    df_queries["label"] = df_queries["query_id"].astype(str) + " | " + df_queries["query_text"].astype(str)

    # search/filter queries
    q_filter = st.text_input("Lọc query theo keyword (optional):", "")
    df_q_show = df_queries
    if q_filter.strip():
        df_q_show = df_q_show[df_q_show["query_text"].astype(str).str.contains(q_filter.strip(), case=False, na=False)]

    if df_q_show.empty:
        st.warning("Không có query nào khớp.")
        return

    selected = st.selectbox("Query:", df_q_show["label"].tolist())
    query_id = selected.split(" | ")[0].strip()
    query_text = " | ".join(selected.split(" | ")[1:]).strip()

    topk = st.slider("Top-K hiển thị:", min_value=3, max_value=50, value=10, step=1)

    # ===== Results for query =====
    df_r = df_results[df_results["query_id"].astype(str) == str(query_id)].copy()
    if df_r.empty:
        st.warning("Query này không có kết quả retrieval.")
        return

    # sort
    if "rank" in df_r.columns:
        df_r = df_r.sort_values("rank")
    elif "similarity" in df_r.columns:
        df_r = df_r.sort_values("similarity", ascending=False)

    df_r = df_r.head(topk)

    # Merge metric if available
    metric_val = None
    if "query_id" in df_metrics.columns and "precision_at_10" in df_metrics.columns:
        rowm = df_metrics[df_metrics["query_id"].astype(str) == str(query_id)].head(1)
        if not rowm.empty:
            metric_val = rowm.iloc[0]["precision_at_10"]

    # ===== Layout =====
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("###  Query")
        st.write(f"**ID:** `{query_id}`")
        st.write(f"**Text:** {query_text}")

        if metric_val is not None:
            st.metric("Precision@10 (nếu có)", f"{float(metric_val):.4f}")

        st.markdown("###  Top results")
        st.dataframe(df_r, use_container_width=True, height=350)

        # Download results filtered for this query
        csv_bytes = df_r.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Top-K CSV (this query)",
            data=csv_bytes,
            file_name=f"topk_{query_id}.csv",
            mime="text/csv"
        )

    with right:
        st.markdown("###  Xem OCR của document")
        doc_ids = df_r["doc_id"].astype(str).tolist()
        doc_pick = st.selectbox("Chọn doc_id:", doc_ids)

        # get OCR
        if "doc_id" not in df_ocr.columns or "text_ocr" not in df_ocr.columns:
            st.error("web_data_ocr.csv thiếu doc_id/text_ocr")
            st.stop()

        row = df_ocr[df_ocr["doc_id"].astype(str) == str(doc_pick)].head(1)
        if row.empty:
            st.warning("Không tìm thấy OCR cho doc này.")
            return

        text = str(row.iloc[0]["text_ocr"])

        if is_ocr_error(text):
            st.error("OCR lỗi model/protobuf:")
            st.code(text)
        elif text.strip() == "":
            st.warning("OCR rỗng.")
        else:
            st.success("OCR OK")
            st.text_area("OCR text", value=text, height=330)

        st.markdown("####  Preview nhanh")
        st.write(summarize_text(text))
