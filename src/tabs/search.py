# src/tabs/search.py
import streamlit as st

from .utils import load_all, is_ocr_error, summarize_text

def show_page(data_dir: str):
    st.header("3) Demo Tìm kiếm (Retrieval)")

    with st.spinner("Đang load dữ liệu..."):
        df_queries, df_results, df_ocr, df_metrics = load_all(data_dir)

    df_queries["label"] = (
        df_queries["query_id"].astype(str) +
        " | " +
        df_queries["query_text"].astype(str)
    )

    selected = st.selectbox("Chọn Query:", df_queries["label"].tolist())
    query_id = selected.split(" | ")[0]
    query_text = selected.split(" | ", 1)[1]

    topk = st.slider("Top-K:", 3, 50, 10)

    df_r = (
        df_results[df_results["query_id"].astype(str) == query_id]
        .sort_values("rank")
        .head(topk)
    )

    metric_val = None
    rowm = df_metrics[df_metrics["query_id"].astype(str) == query_id]
    if not rowm.empty:
        metric_val = rowm.iloc[0]["precision_at_10"]

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Query")
        st.write(query_text)

        if metric_val is not None:
            st.metric("Precision@10", f"{metric_val:.4f}")

        st.markdown("### Top results")
        st.dataframe(df_r, use_container_width=True)

    with right:
        doc_pick = st.selectbox("Xem OCR doc:", df_r["doc_id"].astype(str).tolist())
        row = df_ocr[df_ocr["doc_id"].astype(str) == doc_pick].iloc[0]

        text = str(row["text_ocr"])
        if text.strip():
            st.text_area("OCR text", text, height=330)
            st.markdown("#### Tóm tắt")
            st.write(summarize_text(text))
        else:
            st.warning("OCR rỗng")
