# src/tabs/ocr.py
import streamlit as st

from .utils import load_all, is_ocr_error, summarize_text

def show_page(data_dir: str):
    st.header("2) Kiểm tra OCR")

    with st.spinner("Đang load dữ liệu..."):
        df_queries, df_results, df_ocr, _ = load_all(data_dir)

    mode = st.radio(
        "Chọn chế độ xem:",
        ["Xem theo doc_id", "Xem theo query (top results)"],
        horizontal=True
    )

    st.markdown("---")

    # ===============================
    # MODE 1: DOC_ID
    # ===============================
    if mode == "Xem theo doc_id":
        keyword = st.text_input("Lọc doc_id (optional):", "").strip()

        df_view = df_ocr.copy()
        if keyword:
            df_view = df_view[df_view["doc_id"].astype(str).str.contains(keyword, case=False)]

        st.caption(f"{len(df_view):,} documents")
        doc_ids = df_view["doc_id"].astype(str).head(5000).tolist()

        if not doc_ids:
            st.warning("Không có doc_id phù hợp.")
            return

        doc_id = st.selectbox("Chọn doc_id:", doc_ids)
        row = df_ocr[df_ocr["doc_id"].astype(str) == doc_id].iloc[0]

        text = str(row["text_ocr"])

        if is_ocr_error(text):
            st.error("OCR lỗi")
            st.code(text)
        elif text.strip() == "":
            st.warning("OCR rỗng")
        else:
            st.success("OCR OK")
            st.text_area("OCR text", text, height=350)

    # ===============================
    # MODE 2: QUERY → DOC
    # ===============================
    else:
        df_queries["label"] = (
            df_queries["query_id"].astype(str) +
            " | " +
            df_queries["query_text"].astype(str)
        )

        selected = st.selectbox("Chọn Query:", df_queries["label"].tolist())
        query_id = selected.split(" | ")[0]

        df_r = df_results[df_results["query_id"].astype(str) == query_id].sort_values("rank")
        st.dataframe(df_r, use_container_width=True)

        doc_pick = st.selectbox("Chọn doc_id:", df_r["doc_id"].astype(str).tolist())
        row = df_ocr[df_ocr["doc_id"].astype(str) == doc_pick].iloc[0]

        text = str(row["text_ocr"])
        st.text_area("OCR text", text, height=350)

        st.markdown("### Tóm tắt nhanh")
        st.write(summarize_text(text))
