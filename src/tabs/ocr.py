# src/tabs/ocr.py
import streamlit as st
import pandas as pd

from .utils import load_all, is_ocr_error, summarize_text

def show_page(data_dir: str):
    st.header("2) Kiểm tra OCR")

    with st.spinner("Đang load dữ liệu..."):
        df_queries, df_results, df_ocr, _ = load_all(data_dir)

    if "doc_id" not in df_ocr.columns or "text_ocr" not in df_ocr.columns:
        st.error("web_data_ocr.csv phải có cột doc_id và text_ocr")
        st.stop()

    mode = st.radio("Chọn chế độ xem:", ["Xem theo doc_id", "Xem theo query (top results)"], horizontal=True)
    st.markdown("---")

    if mode == "Xem theo doc_id":
        st.subheader(" Tra cứu theo doc_id")

        # search box
        keyword = st.text_input("Nhập một phần doc_id để lọc (optional):", "")
        df_view = df_ocr.copy()
        if keyword.strip():
            df_view = df_view[df_view["doc_id"].astype(str).str.contains(keyword.strip(), case=False, na=False)]

        st.caption(f"Đang có {len(df_view):,} rows phù hợp.")
        options = df_view["doc_id"].astype(str).head(5000).tolist()  # tránh dropdown quá nặng
        if not options:
            st.warning("Không có doc_id nào khớp.")
            return

        doc_id = st.selectbox("Chọn doc_id:", options)

        row = df_ocr[df_ocr["doc_id"].astype(str) == str(doc_id)].head(1)
        if row.empty:
            st.warning("Không tìm thấy doc_id trong OCR.")
            return

        text = str(row.iloc[0]["text_ocr"]) if "text_ocr" in row.columns else ""
        if is_ocr_error(text):
            st.error("OCR bị lỗi model load / protobuf. Nội dung lỗi:")
            st.code(text)
        elif text.strip() == "":
            st.warning("OCR rỗng (ảnh lỗi / không có text / model trả về rỗng).")
        else:
            st.success("OCR OK")
            st.text_area("OCR text", value=text, height=350)

        st.markdown("---")
        st.subheader(" Preview row")
        st.dataframe(row, use_container_width=True)

    else:
        st.subheader(" Xem OCR theo Query (Top-K results)")
        if not {"query_id", "doc_id"}.issubset(set(df_results.columns)):
            st.error("web_data_results.csv phải có cột query_id và doc_id")
            st.stop()

        # choose query
        if not {"query_id", "query_text"}.issubset(set(df_queries.columns)):
            st.error("web_data_queries.csv phải có cột query_id và query_text")
            st.stop()

        df_queries_small = df_queries.copy()
        df_queries_small["label"] = df_queries_small["query_id"].astype(str) + " | " + df_queries_small["query_text"].astype(str)

        selected = st.selectbox("Chọn Query:", df_queries_small["label"].tolist())
        query_id = selected.split(" | ")[0].strip()

        # get top results for that query
        df_r = df_results[df_results["query_id"].astype(str) == str(query_id)].copy()
        if "rank" in df_r.columns:
            df_r = df_r.sort_values("rank")
        else:
            df_r = df_r.head(10)

        st.caption(f"Top results: {len(df_r)}")
        st.dataframe(df_r, use_container_width=True)

        # pick doc from results
        doc_pick = st.selectbox("Chọn doc_id để xem OCR:", df_r["doc_id"].astype(str).tolist())
        row = df_ocr[df_ocr["doc_id"].astype(str) == str(doc_pick)].head(1)

        if row.empty:
            st.warning("Không có OCR cho doc này.")
            return

        text = str(row.iloc[0]["text_ocr"])
        if is_ocr_error(text):
            st.error("OCR lỗi:")
            st.code(text)
        elif text.strip() == "":
            st.warning("OCR rỗng.")
        else:
            st.text_area("OCR text", value=text, height=350)

        st.markdown("### ✂️ Tóm tắt nhanh")
        st.write(summarize_text(text))
